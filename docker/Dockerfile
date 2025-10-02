# Base image with Python and R (use rocker project for R + add Python) OR start from python and add R
FROM ubuntu:22.04 as base

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential \
    wget curl git ca-certificates \
    r-base r-base-dev \
    libz-dev libssl-dev libxml2-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install fastp
RUN wget -O /usr/local/bin/fastp http://opengene.org/fastp/fastp && chmod +x /usr/local/bin/fastp

# Install DADA2 in R
RUN R -q -e "install.packages('BiocManager'); BiocManager::install(c('dada2'), ask=FALSE, update=FALSE)"

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source
COPY src ./src
COPY scripts ./scripts
COPY config ./config

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "src/web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
