from sqlalchemy import Column, Integer, String, DateTime, JSON, Enum, Text
from sqlalchemy.sql import func
from enum import Enum as PyEnum
from .base import Base

class JobStatus(str, PyEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"

class PipelineJob(Base):
    __tablename__ = "pipeline_jobs"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, index=True)
    raw_dir = Column(String(255), nullable=False)
    interim_dir = Column(String(255), nullable=False)
    processed_dir = Column(String(255), nullable=False)
    meta = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
