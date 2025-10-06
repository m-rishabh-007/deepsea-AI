"""
DeepSea-AI Discovery Engine

This module implements the core functionality for Stage 2 of the DeepSea-AI pipeline.
Complete pipeline flow: fastp → dada2 → asv → embedding transformer + hdbscan → if failed → k-mer vectorization → umap + hdbscan

It processes Amplicon Sequence Variants (ASVs) to discover novel taxonomic groups using:

1. PRIMARY: DNABERT-S pre-trained genomic foundation model (768D embeddings) + HDBSCAN
2. FALLBACK: K-mer frequency vectorization (256D) → UMAP reduction (50D) + HDBSCAN

Author: DeepSea-AI Team
Date: October 6, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Optional
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def load_asv_data(filepath: str) -> pd.DataFrame:
    """
    Load ASV data from CSV file.
    
    Args:
        filepath: Path to the ASV CSV file with 'sequence' and 'count' columns
        
    Returns:
        pandas.DataFrame: DataFrame containing sequence and count data
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        logger.info(f"Loading ASV data from {filepath}")
        
        # Check if file exists
        if not Path(filepath).exists():
            raise FileNotFoundError(f"ASV file not found: {filepath}")
            
        # Load CSV file
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_columns = ['sequence', 'count']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        logger.info(f"Successfully loaded {len(df)} ASV sequences")
        return df
        
    except Exception as e:
        logger.error(f"Error loading ASV data: {str(e)}")
        raise


def load_dnabert_model(device: Optional[torch.device] = None):
    """
    Load the DNABERT-S model and tokenizer.
    
    Args:
        device: PyTorch device to load model on (auto-detected if None)
        
    Returns:
        tuple: (model, tokenizer, device) or (None, None, None) if loading fails
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Auto-detect device if not provided
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        logger.info(f"Attempting to load DNABERT-S model on device: {device}")
        
        # Load tokenizer and model with trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-S", 
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-S", 
            trust_remote_code=True
        )
        
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        
        logger.info("DNABERT-S model loaded successfully")
        return model, tokenizer, device
        
    except ImportError as e:
        logger.warning(f"Required library not installed: {str(e)}")
        logger.warning("Will fallback to k-mer embeddings")
        return None, None, None
    except Exception as e:
        logger.warning(f"Error loading DNABERT-S model: {str(e)}")
        logger.warning("Will fallback to k-mer embeddings")
        return None, None, None


def generate_dnabert_embeddings(
    sequences: List[str], 
    model,
    tokenizer, 
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate DNABERT-S embeddings for DNA sequences.
    
    Args:
        sequences: List of DNA sequence strings
        model: Pre-loaded DNABERT-S model
        tokenizer: Pre-loaded DNABERT-S tokenizer
        device: PyTorch device (CPU or CUDA)
        batch_size: Number of sequences to process in each batch
        
    Returns:
        numpy.ndarray: Array of 768-dimensional embeddings
    """
    logger.info(f"Generating DNABERT-S embeddings for {len(sequences)} sequences using batch size {batch_size}")
    
    embeddings_list = []
    successful_count = 0
    failed_count = 0
    
    # Process sequences in batches
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
        
        batch_embeddings = []
        
        for sequence in batch_sequences:
            try:
                # Clean sequence - ensure only valid DNA characters
                clean_sequence = ''.join(c.upper() for c in sequence if c.upper() in 'ATCGN')
                
                if len(clean_sequence) < 6:  # Skip very short sequences
                    logger.warning(f"Sequence too short after cleaning (length {len(clean_sequence)})")
                    batch_embeddings.append(np.zeros(768))
                    failed_count += 1
                    continue
                
                # Tokenize the DNA sequence
                inputs = tokenizer(
                    clean_sequence, 
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )["input_ids"].to(device)
                
                with torch.no_grad():
                    # Get hidden states from the model
                    outputs = model(inputs)
                    
                    # Get the last hidden state (sequence representation)
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state  # Shape: [1, sequence_length, 768]
                    else:
                        hidden_states = outputs[0]  # Fallback for different model architectures
                    
                    # Apply mean pooling across sequence length dimension
                    embedding = torch.mean(hidden_states, dim=1).squeeze()  # Shape: [768]
                    
                    # Convert to numpy and add to batch
                    batch_embeddings.append(embedding.cpu().numpy())
                    successful_count += 1
                    
            except AssertionError as e:
                # Specific handling for CUDA assertion errors in DNABERT-S
                if "cuda" in str(e).lower():
                    logger.error("DNABERT-S requires CUDA but only CPU is available")
                    raise Exception("DNABERT-S CPU incompatibility - will fallback to k-mer")
                else:
                    raise e
            except Exception as e:
                error_msg = str(e) if str(e) else "Unknown error"
                logger.warning(f"Error processing sequence with DNABERT-S (length {len(sequence)}): {error_msg}")
                # Add zero embedding for failed sequences
                batch_embeddings.append(np.zeros(768))
                failed_count += 1
        
        embeddings_list.extend(batch_embeddings)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list)
    logger.info(f"Successfully generated DNABERT-S embeddings with shape: {embeddings_array.shape}")
    logger.info(f"Processing summary: {successful_count} successful, {failed_count} failed")
    
    return embeddings_array


def generate_kmer_embeddings(
    sequences: List[str], 
    batch_size: int = 32,
    use_umap: bool = True,
    umap_n_components: int = 50
) -> np.ndarray:
    """
    Generate k-mer frequency embeddings for DNA sequences with optional UMAP dimensionality reduction.
    CPU-friendly fallback when DNABERT-S is not available.
    
    Args:
        sequences: List of DNA sequence strings
        batch_size: Number of sequences to process in each batch (unused for k-mer approach)
        use_umap: Whether to apply UMAP dimensionality reduction
        umap_n_components: Number of UMAP components (dimensions) for reduction
        
    Returns:
        numpy.ndarray: Array of k-mer frequency embeddings (256D raw or reduced via UMAP)
    """
    logger.info(f"Generating k-mer frequency embeddings for {len(sequences)} sequences")
    
    # Use 4-mers (k=4) for a good balance of specificity and generality
    k = 4
    
    # Generate all possible k-mers
    bases = ['A', 'T', 'C', 'G']
    from itertools import product
    all_kmers = [''.join(kmer) for kmer in product(bases, repeat=k)]
    kmer_to_index = {kmer: i for i, kmer in enumerate(all_kmers)}
    
    embeddings_list = []
    successful_count = 0
    failed_count = 0
    
    logger.info(f"Computing {len(all_kmers)}-dimensional k-mer vectors...")
    
    for sequence in sequences:
        try:
            # Clean sequence - ensure only valid DNA characters
            clean_sequence = ''.join(c.upper() for c in sequence if c.upper() in 'ATCG')
            
            if len(clean_sequence) < k:
                logger.warning(f"Sequence too short after cleaning (length {len(clean_sequence)})")
                # Add zero vector for failed sequences
                embeddings_list.append(np.zeros(len(all_kmers)))
                failed_count += 1
                continue
            
            # Count k-mer frequencies
            kmer_counts = np.zeros(len(all_kmers))
            total_kmers = len(clean_sequence) - k + 1
            
            for i in range(total_kmers):
                kmer = clean_sequence[i:i+k]
                if kmer in kmer_to_index:
                    kmer_counts[kmer_to_index[kmer]] += 1
            
            # Normalize to frequencies
            if total_kmers > 0:
                kmer_frequencies = kmer_counts / total_kmers
            else:
                kmer_frequencies = kmer_counts
            
            embeddings_list.append(kmer_frequencies)
            successful_count += 1
            
        except Exception as e:
            error_msg = str(e) if str(e) else "Unknown error"
            logger.warning(f"Error processing sequence (length {len(sequence)}): {error_msg}")
            # Add zero embedding for failed sequences
            embeddings_list.append(np.zeros(len(all_kmers)))
            failed_count += 1
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list)
    logger.info(f"Generated raw k-mer embeddings with shape: {embeddings_array.shape}")
    logger.info(f"K-mer processing summary: {successful_count} successful, {failed_count} failed")
    
    # Apply UMAP dimensionality reduction if requested
    if use_umap and successful_count > 10:  # Need sufficient data for UMAP
        try:
            import umap
            logger.info(f"Applying UMAP dimensionality reduction: {embeddings_array.shape[1]}D → {umap_n_components}D")
            
            # Configure UMAP with parameters optimized for genomic data
            umap_reducer = umap.UMAP(
                n_components=umap_n_components,
                n_neighbors=min(15, max(2, successful_count // 3)),  # Adaptive neighbors
                min_dist=0.1,  # Preserve local structure
                metric='cosine',  # Good for frequency vectors
                random_state=42,
                verbose=False
            )
            
            # Fit and transform the embeddings
            reduced_embeddings = umap_reducer.fit_transform(embeddings_array)
            
            logger.info(f"UMAP reduction completed: {embeddings_array.shape} → {reduced_embeddings.shape}")
            logger.info(f"UMAP explained variance preservation: ~{umap_reducer.embedding_.var().sum():.2f}")
            
            return reduced_embeddings
            
        except ImportError:
            logger.warning("UMAP library not installed. Install with: pip install umap-learn")
            logger.info("Proceeding with raw k-mer embeddings...")
        except Exception as e:
            logger.warning(f"UMAP reduction failed: {str(e)}")
            logger.info("Proceeding with raw k-mer embeddings...")
    
    elif use_umap and successful_count <= 10:
        logger.info(f"Too few sequences ({successful_count}) for UMAP reduction. Using raw k-mer embeddings.")
    
    return embeddings_array


def generate_embeddings(
    sequences: List[str], 
    batch_size: int = 32,
    use_dnabert: bool = True
) -> Tuple[np.ndarray, str]:
    """
    Generate embeddings for DNA sequences following the pipeline:
    1. DNABERT-S transformer embeddings + HDBSCAN (preferred)
    2. If failed → K-mer vectorization + UMAP + HDBSCAN (fallback)
    
    Args:
        sequences: List of DNA sequence strings
        batch_size: Number of sequences to process in each batch
        use_dnabert: Whether to attempt DNABERT-S (True) or force k-mer (False)
        
    Returns:
        tuple: (embeddings_array, embedding_method) where method is "dnabert" or "kmer_umap"
    """
    if use_dnabert:
        # Primary: Try DNABERT-S transformer embeddings
        model, tokenizer, device = load_dnabert_model()
        
        if model is not None:
            try:
                embeddings = generate_dnabert_embeddings(sequences, model, tokenizer, device, batch_size)
                logger.info("✅ Using DNABERT-S transformer embeddings (768D)")
                return embeddings, "dnabert"
            except Exception as e:
                logger.warning(f"DNABERT-S transformer failed: {str(e)}")
                logger.info("→ Falling back to K-mer + UMAP pipeline")
    
    # Fallback: K-mer vectorization + UMAP + HDBSCAN
    logger.info("🔄 Executing fallback: K-mer vectorization → UMAP → HDBSCAN")
    embeddings = generate_kmer_embeddings(sequences, batch_size, use_umap=True, umap_n_components=50)
    logger.info("✅ Using K-mer + UMAP embeddings")
    return embeddings, "kmer_umap"


def perform_clustering(
    embeddings: np.ndarray, 
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None
) -> np.ndarray:
    """
    Perform HDBSCAN clustering on sequence embeddings.
    
    Args:
        embeddings: Array of sequence embeddings
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples parameter for HDBSCAN
        
    Returns:
        numpy.ndarray: Cluster labels for each sequence (-1 indicates noise)
    """
    try:
        import hdbscan
        logger.info(f"Performing HDBSCAN clustering on {embeddings.shape[0]} embeddings")
        
        # Check if we have mostly zero embeddings (failed processing)
        non_zero_embeddings = np.any(embeddings, axis=1).sum()
        logger.info(f"Non-zero embeddings: {non_zero_embeddings} out of {len(embeddings)}")
        
        if non_zero_embeddings < min_cluster_size:
            logger.warning(f"Too few valid embeddings ({non_zero_embeddings}) for clustering. Adjusting parameters.")
            min_cluster_size = max(2, non_zero_embeddings // 2)
            min_samples = 1
        
        # Initialize HDBSCAN clusterer with more lenient parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples or 1,
            cluster_selection_epsilon=0.1,  # Allow more clusters
            prediction_data=True
        )
        
        # Fit and predict clusters
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Log clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Clustering completed: {n_clusters} clusters found, {n_noise} noise points")
        
        # If no clusters found, try with even more lenient parameters
        if n_clusters == 0 and non_zero_embeddings >= 2:
            logger.info("No clusters found, trying with minimum parameters...")
            clusterer_min = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=1,
                cluster_selection_epsilon=0.5,
                prediction_data=True
            )
            cluster_labels = clusterer_min.fit_predict(embeddings)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            logger.info(f"Minimum clustering completed: {n_clusters} clusters found, {n_noise} noise points")
        
        return cluster_labels
        
    except ImportError:
        logger.error("HDBSCAN library not installed. Please install with: pip install hdbscan")
        raise
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        raise


def run_discovery_pipeline(
    asv_filepath: str,
    output_filepath: str = "discovery_engine_results.csv",
    min_cluster_size: int = 5,
    batch_size: int = 32,
    use_dnabert: bool = True
) -> pd.DataFrame:
    """
    Run the complete discovery engine pipeline following exact specification:
    fastp → dada2 → asv → embedding transformer + hdbscan → if failed → k-mer vectorization → umap + hdbscan
    
    Args:
        asv_filepath: Path to input ASV CSV file (output from dada2)
        output_filepath: Path for output CSV file
        min_cluster_size: Minimum cluster size for HDBSCAN
        batch_size: Batch size for embedding generation
        use_dnabert: Whether to attempt DNABERT-S transformer (True) or force k-mer (False)
        
    Returns:
        pandas.DataFrame: Results with sequence, count, and cluster_id columns
    """
    logger.info("🚀 Starting DeepSea-AI Discovery Engine Pipeline")
    logger.info("Pipeline: fastp → dada2 → asv → embedding transformer + hdbscan → if failed → k-mer vectorization → umap + hdbscan")
    
    try:
        # Step 1: Load ASV data (already processed by fastp → dada2)
        logger.info("📊 Step 1: Loading ASV data (from dada2 output)")
        asv_df = load_asv_data(asv_filepath)
        sequences = asv_df['sequence'].tolist()
        
        # Step 2: Generate embeddings with fallback chain
        logger.info("🧬 Step 2: Generating sequence embeddings")
        embeddings, embedding_method = generate_embeddings(sequences, batch_size, use_dnabert)
        
        # Step 3: Perform clustering (HDBSCAN for both methods)
        logger.info("🔍 Step 3: Performing HDBSCAN clustering")
        cluster_labels = perform_clustering(embeddings, min_cluster_size)
        
        # Step 4: Create results DataFrame
        logger.info("📝 Step 4: Creating results dataset")
        results_df = asv_df.copy()
        results_df['cluster_id'] = cluster_labels
        results_df['embedding_method'] = embedding_method  # Track which method was used
        
        # Step 5: Save results
        results_df.to_csv(output_filepath, index=False)
        logger.info(f"💾 Results saved to: {output_filepath}")
        
        # Log comprehensive summary statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        clustering_success_rate = ((len(sequences) - n_noise) / len(sequences)) * 100
        
        logger.info("✅ Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info("📈 DISCOVERY ENGINE RESULTS SUMMARY:")
        logger.info(f"  • Total ASV sequences processed: {len(sequences)}")
        logger.info(f"  • Embedding method used: {embedding_method.upper()}")
        logger.info(f"  • Embedding dimensions: {embeddings.shape[1]}D")
        logger.info(f"  • Novel taxonomic clusters discovered: {n_clusters}")
        logger.info(f"  • Sequences successfully clustered: {len(sequences) - n_noise}")
        logger.info(f"  • Noise/singleton sequences: {n_noise}")
        logger.info(f"  • Clustering success rate: {clustering_success_rate:.1f}%")
        logger.info("=" * 60)
        
        return results_df
        
    except Exception as e:
        logger.error(f"❌ Discovery pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python discovery_engine.py <asv_file.csv> [output_file.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "discovery_engine_results.csv"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    
    # Run pipeline
    results = run_discovery_pipeline(input_file, output_file)
    print(f"Discovery engine completed. Results saved to: {output_file}")