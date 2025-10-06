"""
DeepSea-AI Discovery Engine Module

This module implements Stage 2 of the DeepSea-AI pipeline, focusing on unsupervised
discovery of novel microbial taxa using deep learning and clustering techniques.

The discovery engine processes Amplicon Sequence Variants (ASVs) from Stage 1 
and uses the DNABERT-S pre-trained genomic foundation model to generate 
biologically meaningful embeddings, which are then clustered using HDBSCAN
to identify potential novel taxonomic groups.

Key Components:
- discovery_engine.py: Core implementation of the discovery pipeline
- DNABERT-S integration for sequence embeddings
- HDBSCAN clustering for taxonomic discovery
"""

from .discovery_engine import (
    load_asv_data,
    generate_embeddings,
    perform_clustering,
    run_discovery_pipeline,
)

__all__ = [
    "load_asv_data",
    "generate_embeddings", 
    "perform_clustering",
    "run_discovery_pipeline",
]