import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os
from pathlib import Path

from src.api.main import app
from src.db.base import Base
from src.db.session import get_db

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture
def client():
    # Create test database
    Base.metadata.create_all(bind=engine)
    
    # Create test client
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def sample_fastq_content():
    """Sample FASTQ content for testing."""
    return """@SEQ_ID_1
GATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGA
+
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
@SEQ_ID_2
CATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGA
+
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
"""

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "deepsea-stage1-api"

def test_create_job(client):
    """Test job creation."""
    job_data = {
        "name": "Test Job",
        "description": "Test job for API testing",
        "kmer_k": 6
    }
    response = client.post("/jobs", json=job_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == job_data["name"]
    assert data["description"] == job_data["description"]
    assert data["kmer_k"] == job_data["kmer_k"]
    assert data["status"] == "PENDING"
    assert "id" in data
    
    return data["id"]

def test_list_jobs(client):
    """Test listing jobs."""
    # Create a job first
    test_create_job(client)
    
    response = client.get("/jobs")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) >= 1
    assert data[0]["name"] == "Test Job"

def test_get_job(client):
    """Test getting specific job."""
    job_id = test_create_job(client)
    
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == job_id
    assert data["name"] == "Test Job"

def test_get_nonexistent_job(client):
    """Test getting non-existent job."""
    response = client.get("/jobs/99999")
    assert response.status_code == 404

def test_upload_files(client, sample_fastq_content):
    """Test file upload."""
    job_id = test_create_job(client)
    
    # Create temporary FASTQ file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fastq', delete=False) as f:
        f.write(sample_fastq_content)
        temp_file_path = f.name
    
    try:
        with open(temp_file_path, 'rb') as f:
            files = [("files", ("test.fastq", f, "text/plain"))]
            response = client.post(f"/jobs/{job_id}/files", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["uploaded_files"]) == 1
    finally:
        os.unlink(temp_file_path)

def test_upload_invalid_file_type(client):
    """Test uploading invalid file type."""
    job_id = test_create_job(client)
    
    # Create temporary non-FASTQ file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is not a FASTQ file")
        temp_file_path = f.name
    
    try:
        with open(temp_file_path, 'rb') as f:
            files = [("files", ("test.txt", f, "text/plain"))]
            response = client.post(f"/jobs/{job_id}/files", files=files)
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    finally:
        os.unlink(temp_file_path)

def test_run_job_without_files(client):
    """Test running job without uploaded files."""
    job_id = test_create_job(client)
    
    response = client.post(f"/jobs/{job_id}/run")
    assert response.status_code == 400
    assert "No FASTQ files uploaded" in response.json()["detail"]

def test_run_job_invalid_status(client, sample_fastq_content):
    """Test running job with invalid status."""
    job_id = test_create_job(client)
    
    # Upload files first
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fastq', delete=False) as f:
        f.write(sample_fastq_content)
        temp_file_path = f.name
    
    try:
        with open(temp_file_path, 'rb') as f:
            files = [("files", ("test.fastq", f, "text/plain"))]
            client.post(f"/jobs/{job_id}/files", files=files)
        
        # Try to run job twice (should fail on second attempt)
        response1 = client.post(f"/jobs/{job_id}/run")
        assert response1.status_code == 200
        
        # Second attempt should fail
        response2 = client.post(f"/jobs/{job_id}/run")
        assert response2.status_code == 400
        assert "not in pending state" in response2.json()["detail"]
    finally:
        os.unlink(temp_file_path)

def test_get_job_metadata(client):
    """Test getting job metadata."""
    job_id = test_create_job(client)
    
    response = client.get(f"/jobs/{job_id}/metadata")
    assert response.status_code == 200
    
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "PENDING"
    assert "created_at" in data
    assert "updated_at" in data

def test_download_vectors_job_not_completed(client):
    """Test downloading vectors from incomplete job."""
    job_id = test_create_job(client)
    
    response = client.get(f"/jobs/{job_id}/vectors")
    assert response.status_code == 400
    assert "Job not completed" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main([__file__])