# DeepSea-AI: Complete Pipeline (Stage 1 + Stage 2)

## Overview

DeepSea-AI is a comprehensive bioinformatics pipeline for processing deep-sea eDNA sequencing data and discovering novel taxonomic groups. The system combines traditional ASV analysis (**Stage 1**) with cutting-edge AI-powered discovery (**Stage 2**) to identify potentially new marine species and taxonomic groups.

### Pipeline Stages

- **Stage 1**: Preprocessing & ASV Analysis (Quality Control → DADA2 → ASV generation)
- **Stage 2**: AI-Powered Discovery Engine (DNABERT-S embeddings → HDBSCAN clustering)

## Architecture

### Components

1. **Preprocessing Pipeline** (`src/preprocessing/`)
   - `fastp_wrapper.py`: Quality control using fastp
   - `dada2_wrapper.py`: ASV inference using DADA2 R package
   - `read_detection.py`: Automatic paired/single-end detection

2. **Discovery Engine** (`src/discovery/`)
   - `discovery_engine.py`: AI-powered novel taxa discovery using DNABERT-S and HDBSCAN
   - ASV sequence → DNABERT-S embeddings → clustering → novel taxonomic groups

3. **Web Interface** (`src/web/`)
   - `app.py`: Streamlit interface with dual mode support
   - Direct mode: Local pipeline execution
   - API mode: Backend integration for production
   - Discovery results visualization and cluster analysis

4. **REST API Backend** (`src/api/`)
   - `main.py`: FastAPI service with job management
   - Stage 1 endpoints: Job creation, file upload, pipeline execution, status monitoring
   - Stage 2 endpoints: Discovery status, results access, cluster details

5. **Database Layer** (`src/db/`)
   - PostgreSQL integration with SQLAlchemy ORM
   - Job tracking, status management, and metadata storage

6. **Configuration** (`config/`)
   - `pipeline.yaml`: Pipeline parameters and tool configurations

## Features

- **Two-Stage Analysis**: Complete workflow from raw sequences to novel taxa discovery
- **AI-Powered Discovery**: DNABERT-S foundation model for genomic embeddings
- **Unsupervised Clustering**: HDBSCAN for identifying novel taxonomic groups
- **Automatic Format Detection**: Supports both paired-end and single-end FASTQ files
- **Production-Ready**: Docker containerization, database persistence, job isolation
- **Scalable Architecture**: Background job processing with status tracking
- **Flexible Interface**: Both direct and API-based processing modes
- **Comprehensive Testing**: Unit tests for all pipeline components
- **Interactive Visualization**: Cluster analysis and discovery results display

## Quick Start

### Option 1: Docker Compose (Recommended for Production)

```bash
# 1. Clone and setup
git clone <repository>
cd SIH-Project

# 2. Configure environment
cp .env.example .env
# Edit .env as needed

# 3. Build and run all services (backend now installs R + DADA2 automatically)
docker compose -f docker/docker-compose.yml up --build

# 4. Access interfaces
# Streamlit UI: http://localhost:8501
# FastAPI docs: http://localhost:8000/docs
# PostgreSQL: localhost:5432
```

### Option 2: Local Development

```bash
# 1. Install Python dependencies (includes PyTorch, Transformers, HDBSCAN)
pip install -r requirements.txt

# 2. Install system dependencies (Ubuntu/Debian example)
sudo apt-get update && sudo apt-get install -y \
  r-base r-base-dev fastp \
  libcurl4-openssl-dev libssl-dev libxml2-dev \
  libgit2-dev libbz2-dev liblzma-dev libpcre2-dev

# 3. Install R packages (matches Docker image)
R -q -e "if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager', repos='https://cloud.r-project.org/')" \
     -e "if (!requireNamespace('dada2', quietly = TRUE)) BiocManager::install('dada2', ask = FALSE, update = FALSE)" \
     -e "if (!requireNamespace('jsonlite', quietly = TRUE)) install.packages('jsonlite', repos='https://cloud.r-project.org/')"

# 4. Setup database (optional - for API mode)
docker run -d --name postgres \
  -e POSTGRES_DB=deepsea_ai \
  -e POSTGRES_USER=deepsea \
  -e POSTGRES_PASSWORD=deepsea123 \
  -p 5432:5432 postgres:14

# 5. Run FastAPI backend
uvicorn src.api.main:app --reload

# 6. Run Streamlit interface
streamlit run src/web/app.py
```

## Usage

### Web Interface

1. **Upload FASTQ Files**: Drag and drop or browse for `.fastq`, `.fq`, or `.gz` files
2. **Configure Pipeline**: Enable/disable Stage 2 discovery engine
3. **Process Data**: Click "Run Pipeline" or "Create Job" → "Start Processing"
4. **View Results**: 
   - Stage 1: ASV table, quality metrics, processing statistics
   - Stage 2: Discovery clusters, novel taxa groups, cluster details
5. **Download Results**: ASV table and discovery results as CSV files

### API Usage

```python
import requests

# 1. Create job
job = requests.post("http://localhost:8000/jobs", json={
    "name": "My Analysis",
    "description": "Deep-sea sample analysis"
}).json()

# 2. Upload files
files = [("files", open("sample.fastq", "rb"))]
requests.post(f"http://localhost:8000/jobs/{job['id']}/files", files=files)

# 3. Start processing
requests.post(f"http://localhost:8000/jobs/{job['id']}/run")

# 4. Check status
status = requests.get(f"http://localhost:8000/jobs/{job['id']}").json()

# 5. Get discovery results (Stage 2 - if enabled)
discovery_status = requests.get(f"http://localhost:8000/jobs/{job['id']}/discovery/status").json()
if discovery_status['has_discovery_results']:
    discovery_data = requests.get(f"http://localhost:8000/jobs/{job['id']}/discovery/results").json()
    print(f"Clusters found: {discovery_data['clusters_found']}")
    
    # Get cluster details
    cluster_details = requests.get(f"http://localhost:8000/jobs/{job['id']}/discovery/clusters/0").json()
    
    # Download discovery results
    csv_data = requests.get(f"http://localhost:8000/jobs/{job['id']}/discovery/download").content
```

## Pipeline Details

### Complete Workflow

1. **Input**: Raw FASTQ files (paired-end or single-end)
2. **Stage 1**: Quality Control → ASV Inference
   - **Quality Control**: fastp removes low-quality reads and adapters
   - **ASV Inference**: DADA2 resolves Amplicon Sequence Variants
3. **Stage 2**: AI-Powered Discovery (optional)
   - **Embedding Generation**: DNABERT-S creates 768-dimensional sequence embeddings
   - **Clustering**: HDBSCAN identifies novel taxonomic groups
   - **Analysis**: Cluster statistics and representative sequences
4. **Output**: ASV table + Discovery results with cluster assignments

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

# Discovery Engine (Stage 2)
discovery:
  enabled: true                    # Enable/disable Stage 2
  min_cluster_size: 5             # Minimum sequences per cluster
  batch_size: 32                  # Embedding generation batch size
  model_name: "zhihan1996/DNABERT-S"  # Pre-trained model
```

## Directory Layout

```
SIH-Project/
├── src/
│   ├── preprocessing/     # Stage 1 pipeline modules
│   ├── discovery/         # Stage 2 discovery engine
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
# Run all unit tests
pytest tests/

# Run specific test modules
pytest tests/test_api.py -v                    # API endpoints
pytest tests/test_discovery_engine.py -v       # Discovery engine
pytest tests/test_dada2_wrapper.py -v         # DADA2 integration

# Run pipeline integration smoke test (requires Docker)
pytest tests/test_integration.py -m integration

# Run with coverage
pytest tests/ --cov=src/
```

### Integration Smoke Test Details

The integration test spins up PostgreSQL and the FastAPI backend via Docker Compose, uploads a real FASTQ sample (`fastq_dataset/real_sample.fastq`), runs the full pipeline, and asserts that fastp, DADA2, and k-mer outputs are produced. To skip these slower integration tests, add `-m "not integration"` to the pytest command.

### Adding New Features

1. **Pipeline Extensions**: Add new modules in `src/preprocessing/` or `src/discovery/`
2. **API Endpoints**: Extend `src/api/main.py` with new discovery or analysis endpoints
3. **Database Models**: Update `src/db/models.py` and create migrations
4. **UI Components**: Modify `src/web/app.py` for new visualization features
5. **Discovery Algorithms**: Extend `src/discovery/discovery_engine.py` with new AI models

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

1. **PyTorch/CUDA Setup**: Ensure compatible PyTorch version for your system (CPU/GPU)
2. **DNABERT-S Model Download**: First run downloads ~400MB model from HuggingFace
3. **Memory Requirements**: Discovery engine needs sufficient RAM for embeddings (recommend 8GB+)
4. **R/DADA2 Installation**: Ensure R and DADA2 package are properly installed
5. **fastp Not Found**: Install fastp binary or use Docker version
6. **Database Connection**: Verify PostgreSQL is running and credentials are correct

### Log Files

- **Streamlit**: Console output
- **FastAPI**: Structured JSON logs
- **Pipeline**: Per-job log files in `data/jobs/{job_id}/logs/`

## Next Steps

This complete pipeline provides:
- **Production-Ready ASV Analysis**: High-quality sequence variant calling
- **AI-Powered Discovery**: Novel taxonomic group identification
- **Scalable Architecture**: Ready for high-throughput marine biodiversity studies
- **Research Integration**: Compatible with downstream phylogenetic and ecological analyses

## License

None .
