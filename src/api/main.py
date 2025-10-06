from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from pathlib import Path
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import pandas as pd
import json
from typing import List

from src.db.session import get_db
from src.db import crud, schemas, models
from src.db.init_db import init_db
from src.pipeline import run_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="DeepSea-AI Pipeline API", version="2.0.0", lifespan=lifespan)

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)

@app.post("/jobs", response_model=schemas.Job)
def create_job(job: schemas.JobCreate, db: Session = Depends(get_db)):
    """Create a new pipeline job."""
    job_id = uuid.uuid4().hex[:8]
    raw_dir = f"data/jobs/{job_id}/raw"
    interim_dir = f"data/jobs/{job_id}/interim"
    processed_dir = f"data/jobs/{job_id}/processed"
    
    # Create job directories
    for dir_path in [raw_dir, interim_dir, processed_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    db_job = crud.create_job(db, job, raw_dir, interim_dir, processed_dir)
    return db_job

@app.post("/jobs/{job_id}/files")
def upload_files(job_id: int, files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Upload FASTQ files to a job."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != models.JobStatus.PENDING:
        raise HTTPException(status_code=400, detail="Can only upload to pending jobs")
    
    raw_dir = Path(job.raw_dir)
    uploaded_files = []
    
    for file in files:
        if not file.filename.lower().endswith(('.fastq', '.fastq.gz', '.fq', '.fq.gz')):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")
        
        file_path = raw_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(str(file_path))
    
    return {"uploaded_files": uploaded_files, "count": len(uploaded_files)}

def _run_pipeline_background(job_id: int):
    """Background task to run pipeline and update job status."""
    from src.db.session import SessionLocal
    db = SessionLocal()
    try:
        job = crud.get_job(db, job_id)
        if not job:
            return
        
        crud.update_job_status(db, job_id, models.JobStatus.RUNNING)

        progress = {"history": []}

        def record_progress(step: str, status: str, message: str | None = None):
            entry = {
                "step": step,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            if message:
                entry["message"] = message
            progress["step"] = step
            progress["status"] = status
            history = progress.get("history", [])
            progress["history"] = history + [entry]
            crud.update_job_meta(db, job_id, {"progress": progress})

        record_progress("fastp", "running", "Starting quality control")

        # Run pipeline
        metadata = run_pipeline(
            raw_dir=job.raw_dir,
            interim_dir=job.interim_dir,
            processed_dir=job.processed_dir
        )

        progress["step"] = "complete"
        progress["status"] = "finished"
        progress["history"] = progress.get("history", []) + [{
            "step": "complete",
            "status": "finished",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Pipeline completed"
        }]
        metadata["progress"] = progress

        crud.update_job_status(db, job_id, models.JobStatus.COMPLETED, meta=metadata)
        
    except Exception as e:
        record_progress("error", "failed", str(e)) if 'record_progress' in locals() else None
        crud.update_job_status(db, job_id, models.JobStatus.FAILED, error=str(e))
    finally:
        db.close()

@app.post("/jobs/{job_id}/run")
def run_job(job_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Trigger pipeline execution for a job."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != models.JobStatus.PENDING:
        raise HTTPException(status_code=400, detail="Job is not in pending state")
    
    # Check if files exist
    raw_files = list(Path(job.raw_dir).glob("*.fastq*"))
    if not raw_files:
        raise HTTPException(status_code=400, detail="No FASTQ files uploaded")
    
    # Submit background task
    background_tasks.add_task(_run_pipeline_background, job_id)
    
    return {"message": "Pipeline started", "job_id": job_id}

@app.get("/jobs", response_model=List[schemas.Job])
def list_jobs(db: Session = Depends(get_db)):
    """List all jobs."""
    return crud.list_jobs(db)

@app.get("/jobs/{job_id}", response_model=schemas.Job)
def get_job(job_id: int, db: Session = Depends(get_db)):
    """Get job details."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/jobs/{job_id}/metadata")
def get_job_metadata(job_id: int, db: Session = Depends(get_db)):
    """Get detailed job metadata including pipeline outputs."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job.status,
        "metadata": job.meta,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at
    }

# Discovery Engine Endpoints

@app.get("/jobs/{job_id}/discovery/status", response_model=schemas.DiscoveryStatus)
def get_discovery_status(job_id: int, db: Session = Depends(get_db)):
    """Get discovery engine status for a job."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if discovery results exist
    processed_dir = Path(job.processed_dir)
    discovery_file = processed_dir / "discovery_engine_results.csv"
    
    # Extract discovery info from job metadata
    meta = job.meta or {}
    stage2_enabled = meta.get("stage2_enabled", False)
    discovery_completed = False
    error_message = None
    
    # Check progress history for discovery completion
    progress = meta.get("progress", {})
    history = progress.get("history", [])
    
    for entry in history:
        if entry.get("step") == "discovery":
            if entry.get("status") == "finished":
                discovery_completed = True
            elif entry.get("status") == "failed":
                error_message = entry.get("message", "Discovery engine failed")
    
    return schemas.DiscoveryStatus(
        job_id=job_id,
        has_discovery_results=discovery_file.exists(),
        discovery_completed=discovery_completed,
        stage2_enabled=stage2_enabled,
        results_file=str(discovery_file) if discovery_file.exists() else None,
        error_message=error_message
    )

@app.get("/jobs/{job_id}/discovery/results", response_model=schemas.DiscoveryResults)
def get_discovery_results(job_id: int, db: Session = Depends(get_db)):
    """Get discovery engine results for a job."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if discovery results exist
    processed_dir = Path(job.processed_dir)
    discovery_file = processed_dir / "discovery_engine_results.csv"
    
    if not discovery_file.exists():
        raise HTTPException(
            status_code=404, 
            detail="Discovery results not found. Make sure Stage 2 was enabled and completed successfully."
        )
    
    try:
        # Load discovery results
        df = pd.read_csv(discovery_file)
        
        # Validate required columns
        required_cols = ['sequence', 'count', 'cluster_id']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=500,
                detail=f"Discovery results file missing required columns: {required_cols}"
            )
        
        # Create sequence list
        sequences = [
            schemas.DiscoverySequence(
                sequence=row['sequence'],
                count=int(row['count']),
                cluster_id=int(row['cluster_id'])
            )
            for _, row in df.iterrows()
        ]
        
        # Calculate cluster summaries
        cluster_stats = df[df['cluster_id'] != -1].groupby('cluster_id').agg({
            'sequence': 'count',
            'count': ['sum', 'idxmax']
        }).round().astype(int)
        
        clusters = []
        if not cluster_stats.empty:
            for cluster_id in cluster_stats.index:
                sequence_count = cluster_stats.loc[cluster_id, ('sequence', 'count')]
                total_reads = cluster_stats.loc[cluster_id, ('count', 'sum')]
                max_idx = cluster_stats.loc[cluster_id, ('count', 'idxmax')]
                representative_seq = df.loc[max_idx, 'sequence']
                
                clusters.append(schemas.ClusterSummary(
                    cluster_id=int(cluster_id),
                    sequence_count=int(sequence_count),
                    total_reads=int(total_reads),
                    representative_sequence=representative_seq
                ))
        
        # Calculate summary statistics
        total_sequences = len(df)
        clusters_found = len(df[df['cluster_id'] != -1]['cluster_id'].unique())
        noise_sequences = len(df[df['cluster_id'] == -1])
        
        return schemas.DiscoveryResults(
            job_id=job_id,
            total_sequences=total_sequences,
            clusters_found=clusters_found,
            noise_sequences=noise_sequences,
            clusters=clusters,
            sequences=sequences
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing discovery results: {str(e)}"
        )

@app.get("/jobs/{job_id}/discovery/download")
def download_discovery_results(job_id: int, db: Session = Depends(get_db)):
    """Download discovery results CSV file."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    processed_dir = Path(job.processed_dir)
    discovery_file = processed_dir / "discovery_engine_results.csv"
    
    if not discovery_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Discovery results not found"
        )
    
    return FileResponse(
        path=discovery_file,
        filename=f"discovery_results_job_{job_id}.csv",
        media_type="text/csv"
    )

@app.get("/jobs/{job_id}/discovery/clusters/{cluster_id}")
def get_cluster_details(job_id: int, cluster_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific cluster."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    processed_dir = Path(job.processed_dir)
    discovery_file = processed_dir / "discovery_engine_results.csv"
    
    if not discovery_file.exists():
        raise HTTPException(status_code=404, detail="Discovery results not found")
    
    try:
        df = pd.read_csv(discovery_file)
        cluster_data = df[df['cluster_id'] == cluster_id]
        
        if cluster_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Cluster {cluster_id} not found"
            )
        
        sequences = [
            {
                "sequence": row['sequence'],
                "count": int(row['count']),
                "length": len(row['sequence'])
            }
            for _, row in cluster_data.iterrows()
        ]
        
        # Sort by count (descending)
        sequences.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            "cluster_id": cluster_id,
            "sequence_count": len(sequences),
            "total_reads": int(cluster_data['count'].sum()),
            "sequences": sequences,
            "representative_sequence": sequences[0]['sequence'] if sequences else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing cluster data: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "deepsea-ai-pipeline-api",
        "version": "2.0.0",
        "features": ["stage1_pipeline", "stage2_discovery", "cluster_analysis"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)