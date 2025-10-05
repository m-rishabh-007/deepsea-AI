# Use pre-built Bioconductor image to avoid compiling heavy R packages from source
FROM bioconductor/bioconductor_docker:RELEASE_3_18 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install Python runtime and supporting tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential \
    wget curl git ca-certificates fastp file \
    && rm -rf /var/lib/apt/lists/*

# Ensure python command is available and upgrade pip stack
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Verify DADA2 is present; install only if missing (should be bundled with base image)
RUN R -q -e "if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager'); if (!requireNamespace('dada2', quietly = TRUE)) { BiocManager::install('dada2', ask = FALSE, update = FALSE) }"

# Set workdir
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application sources
COPY src ./src
COPY scripts ./scripts
COPY config ./config

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "src/web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
