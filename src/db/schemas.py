from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"

class JobCreate(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    kmer_k: int = 6

class Job(BaseModel):
    id: int
    name: str
    description: Optional[str]
    status: JobStatus
    kmer_k: int
    meta: Optional[Any]
    error: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True
