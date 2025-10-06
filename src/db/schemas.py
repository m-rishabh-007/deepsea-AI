from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, List
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

class Job(BaseModel):
    id: int
    name: str
    description: Optional[str]
    status: JobStatus
    meta: Optional[Any]
    error: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


# Discovery Engine Schemas
class DiscoverySequence(BaseModel):
    """Schema for a single sequence in discovery results."""
    sequence: str = Field(..., description="DNA sequence string")
    count: int = Field(..., ge=0, description="Read count for this sequence")
    cluster_id: int = Field(..., description="Cluster ID (-1 for noise)")

class ClusterSummary(BaseModel):
    """Schema for cluster summary statistics."""
    cluster_id: int = Field(..., description="Cluster ID")
    sequence_count: int = Field(..., ge=0, description="Number of sequences in cluster")
    total_reads: int = Field(..., ge=0, description="Total read count across all sequences")
    representative_sequence: Optional[str] = Field(None, description="Most abundant sequence in cluster")

class DiscoveryResults(BaseModel):
    """Schema for complete discovery engine results."""
    job_id: int = Field(..., description="Associated pipeline job ID")
    total_sequences: int = Field(..., ge=0, description="Total number of ASV sequences processed")
    clusters_found: int = Field(..., ge=0, description="Number of clusters discovered")
    noise_sequences: int = Field(..., ge=0, description="Number of sequences classified as noise")
    clusters: List[ClusterSummary] = Field(default_factory=list, description="Summary of each cluster")
    sequences: List[DiscoverySequence] = Field(default_factory=list, description="All sequences with cluster assignments")

class DiscoveryStatus(BaseModel):
    """Schema for discovery engine processing status."""
    job_id: int
    has_discovery_results: bool = Field(..., description="Whether discovery results are available")
    discovery_completed: bool = Field(..., description="Whether discovery stage completed successfully")
    stage2_enabled: bool = Field(..., description="Whether Stage 2 was enabled for this job")
    results_file: Optional[str] = Field(None, description="Path to discovery results CSV file")
    error_message: Optional[str] = Field(None, description="Error message if discovery failed")
