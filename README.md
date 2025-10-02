# DeepSea-AI: Stage 1 - Preprocessing & Vectorization

## Overview

DeepSea-AI Stage 1 is a production-ready bioinformatics pipeline for preprocessing deep-sea eDNA sequencing data and converting it to numerical vectors for downstream AI analysis. The system automatically processes FASTQ files through quality control, ASV inference, and k-mer vectorization.

## Architecture

### Components

1. **Preprocessing Pipeline** (`src/preprocessing/`)
   - `fastp_wrapper.py`: Quality control using fastp
   - `dada2_wrapper.py`: ASV inference using DADA2 R package
   - `kmer_vectorizer.py`: K-mer profiling and vectorization
   - `read_detection.py`: Automatic paired/single-end detection

2. **Web Interface** (`src/web/`)
   - `app.py`: Streamlit interface with dual mode support
   - Direct mode: Local pipeline execution
   - API mode: Backend integration for production

3. **REST API Backend** (`src/api/`)
   - `main.py`: FastAPI service with job management
   - Endpoints: Job creation, file upload, pipeline execution, status monitoring

4. **Database Layer** (`src/db/`)
   - PostgreSQL integration with SQLAlchemy ORM
   - Job tracking, status management, and metadata storage

5. **Configuration** (`config/`)
   - `pipeline.yaml`: Pipeline parameters and tool configurations

## Features

- **Automatic Format Detection**: Supports both paired-end and single-end FASTQ files
- **Production-Ready**: Docker containerization, database persistence, job isolation
- **Scalable Architecture**: Background job processing with status tracking
- **Flexible Interface**: Both direct and API-based processing modes
- **Comprehensive Testing**: Unit tests for API and database functionality

## Quick Start

### Option 1: Docker Compose (Recommended for Production)

```bash
# 1. Clone and setup
git clone <repository>
cd SIH-Project

# 2. Configure environment
cp .env.example .env
# Edit .env as needed

# 3. Build and run all services
cd docker
docker-compose up --build

# 4. Access interfaces
# Streamlit UI: http://localhost:8501
# FastAPI docs: http://localhost:8000/docs
# PostgreSQL: localhost:5432
```

### Option 2: Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install system dependencies
# Ubuntu/Debian:
sudo apt-get install r-base fastp

# 3. Install R packages
R -e "install.packages(c('dada2', 'jsonlite'), repos='https://cloud.r-project.org/')"

# 4. Setup database (optional - for API mode)
# Install PostgreSQL locally or use Docker:
docker run -d --name postgres \
  -e POSTGRES_DB=deepsea_ai \
  -e POSTGRES_USER=deepsea \
  -e POSTGRES_PASSWORD=deepsea123 \
  -p 5432:5432 postgres:14

# 5. Run Streamlit interface
streamlit run src/web/app.py

# 6. Or run FastAPI backend
uvicorn src.api.main:app --reload
```

## Usage

### Web Interface

1. **Upload FASTQ Files**: Drag and drop or browse for `.fastq`, `.fq`, or `.gz` files
2. **Configure Parameters**: Set k-mer length (default: 6)
3. **Process Data**: Click "Run Pipeline" or "Create Job" → "Start Processing"
4. **Download Results**: Get k-mer vectors as CSV file

### API Usage

```python
import requests

# 1. Create job
job = requests.post("http://localhost:8000/jobs", json={
    "name": "My Analysis",
    "description": "Deep-sea sample analysis",
    "kmer_k": 6
}).json()

# 2. Upload files
files = [("files", open("sample.fastq", "rb"))]
requests.post(f"http://localhost:8000/jobs/{job['id']}/files", files=files)

# 3. Start processing
requests.post(f"http://localhost:8000/jobs/{job['id']}/run")

# 4. Check status
status = requests.get(f"http://localhost:8000/jobs/{job['id']}").json()

# 5. Download results (when completed)
vectors = requests.get(f"http://localhost:8000/jobs/{job['id']}/vectors")
```

## Pipeline Details

### Stage 1 Workflow

1. **Input**: Raw FASTQ files (paired-end or single-end)
2. **Quality Control**: fastp removes low-quality reads and adapters
3. **ASV Inference**: DADA2 resolves Amplicon Sequence Variants
4. **Vectorization**: K-mer profiling converts sequences to numerical vectors
5. **Output**: CSV file with k-mer frequency vectors

### Supported File Formats

- **FASTQ**: `.fastq`, `.fq`
- **Compressed**: `.fastq.gz`, `.fq.gz`
- **Naming Conventions**: 
  - Paired-end: `*_R1.fastq` & `*_R2.fastq`, `*_1.fastq` & `*_2.fastq`
  - Single-end: Any `.fastq` file not matching paired patterns

## Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# API Settings
API_MODE=false  # Set to 'true' for production
BACKEND_URL=http://localhost:8000

# Pipeline
KMER_DEFAULT_K=6
MAX_CONCURRENT_JOBS=2

# Processing Limits
MAX_FILE_SIZE_MB=500
PIPELINE_TIMEOUT=3600
```

### Pipeline Configuration (config/pipeline.yaml)

```yaml
fastp:
  qualified_quality_phred: 15
  length_required: 15
  
dada2:
  truncLen: [240, 160]  # For paired-end
  maxEE: [2, 2]
  
kmer:
  k: 6  # Can be overridden per job
```

## Directory Layout

```
SIH-Project/
├── src/
│   ├── preprocessing/     # Core pipeline modules
│   ├── web/              # Streamlit interface
│   ├── api/              # FastAPI backend
│   ├── db/               # Database layer
│   └── utils/            # Utilities
├── scripts/              # R scripts and helpers
├── config/               # Configuration files
├── docker/               # Container definitions
├── tests/                # Test suite
└── data/                 # Data directory (created at runtime)
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src/
```

### Adding New Features

1. **Pipeline Extensions**: Add new modules in `src/preprocessing/`
2. **API Endpoints**: Extend `src/api/main.py`
3. **Database Models**: Update `src/db/models.py` and create migrations
4. **UI Components**: Modify `src/web/app.py`

## Production Deployment

### Docker Deployment

The provided `docker-compose.yml` includes:
- **PostgreSQL**: Database service with persistent storage
- **Backend**: FastAPI service with job processing
- **Frontend**: Streamlit interface with API integration

### Scaling Considerations

- **Job Isolation**: Each job runs in separate directories
- **Background Processing**: Non-blocking pipeline execution
- **Database Persistence**: Job history and metadata storage
- **Resource Limits**: Configurable timeouts and file size limits

## Troubleshooting

### Common Issues

1. **R/DADA2 Installation**: Ensure R and DADA2 package are properly installed
2. **fastp Not Found**: Install fastp binary or use Docker version
3. **Database Connection**: Verify PostgreSQL is running and credentials are correct
4. **Memory Issues**: Large FASTQ files may require increased container memory

### Log Files

- **Streamlit**: Console output
- **FastAPI**: Structured JSON logs
- **Pipeline**: Per-job log files in `data/jobs/{job_id}/logs/`

## Next Steps

This Stage 1 module prepares data for:
- **Stage 2**: Nucleotide Transformer classification
- **Stage 3**: UMAP/HDBSCAN discovery engine
- **Integration**: Full DeepSea-AI hybrid system

## License

None .
