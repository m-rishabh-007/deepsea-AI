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
from typing import List

from src.db.session import get_db
from src.db import crud, schemas, models
from src.db.init_db import init_db
from src.pipeline import run_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="DeepSea-AI Stage 1 API", version="1.0.0", lifespan=lifespan)

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

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "deepsea-stage1-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)