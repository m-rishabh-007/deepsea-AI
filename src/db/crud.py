from sqlalchemy.orm import Session
from . import models
from .schemas import JobCreate
from datetime import datetime

def create_job(db: Session, job_in: JobCreate, raw_dir: str, interim_dir: str, processed_dir: str):
    job = models.PipelineJob(
        name=job_in.name,
        description=job_in.description,
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        kmer_k=job_in.kmer_k,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_job_status(db: Session, job_id: int, status: models.JobStatus, meta=None, error: str | None = None):
    job = db.query(models.PipelineJob).filter(models.PipelineJob.id == job_id).first()
    if not job:
        return None
    job.status = status
    if meta is not None:
        job.meta = meta
    if error:
        job.error = error
    job.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: int):
    return db.query(models.PipelineJob).filter(models.PipelineJob.id == job_id).first()


def list_jobs(db: Session, limit: int = 50):
    return db.query(models.PipelineJob).order_by(models.PipelineJob.id.desc()).limit(limit).all()
