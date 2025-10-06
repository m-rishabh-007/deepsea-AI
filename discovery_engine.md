
# Implementing the DeepSea-AI Discovery Engine

**Role:** You are an expert Python developer specializing in building bioinformatics and machine learning pipelines. Your task is to write a complete, production-quality Python script for the "Discovery Engine" of the DeepSea-AI project.

**Primary Objective:** Create a Python script that implements the hybrid discovery pipeline: **fastp → dada2 → asv → embedding transformer + hdbscan → if failed → k-mer vectorization → umap + hdbscan**. The system attempts DNABERT-S transformer embeddings first, then automatically falls back to an enhanced k-mer + UMAP approach for compatibility across different hardware configurations.

**Detailed Project Context:**

The goal of the Discovery Engine is the unsupervised discovery of novel microbial taxa from environmental DNA (eDNA). The pipeline starts with a list of unique DNA sequences called Amplicon Sequence Variants (ASVs). This implementation uses a sophisticated hybrid approach:

1. **PRIMARY PATH**: DNABERT-S pre-trained genomic foundation model (768D embeddings) + HDBSCAN clustering
2. **FALLBACK PATH**: K-mer frequency vectorization (256D) → UMAP dimensionality reduction (50D) + HDBSCAN clustering

**Input Data Specification:**

  * **Filename:** The script should expect a CSV file (typically `asvs.csv` from DADA2 output).
  * **Format:** The CSV file has two columns: `sequence` and `count`.
  * **Example:**
    ```csv
    sequence,count
    GTATGAATCCCGCCTGAAGGGAA...,10
    GATATACGCAGCGAATTGAGCGG...,10
    GGGCTACACACGTGCTACAATGG...,9
    ```
  * **Processing Note:** The script uses the `sequence` column for embedding generation and clustering. The `count` column is preserved in the final output along with the assigned cluster IDs and embedding method used.

-----

### Core Implementation Steps:

Please implement the following steps in a clean, modular Python script following the exact pipeline specification.

**1. Setup and Dependencies:**
The script requires the following core libraries:

  * `pandas` for data handling
  * `numpy` for numerical operations
  * `torch` for deep learning operations
  * `transformers` for DNABERT-S model loading
  * `hdbscan` for clustering algorithm
  * `umap-learn` for dimensionality reduction (fallback path)
  * `itertools` for k-mer generation

**2. Hybrid Embedding Generation Pipeline:**

The system implements an intelligent fallback mechanism:

**PRIMARY: DNABERT-S Transformer Embeddings**
  * Load the pre-trained `zhihan1996/DNABERT-S` model from Hugging Face
  * Process DNA sequences through the transformer to generate 768-dimensional embeddings
  * Apply mean pooling across sequence length to create fixed-size representations
  * If successful, proceed directly to HDBSCAN clustering

**FALLBACK: Enhanced K-mer + UMAP Pipeline**  
  * Generate 4-mer frequency vectors (256 dimensions) for all DNA sequences
  * Apply UMAP dimensionality reduction to compress to 50 dimensions
  * This preserves sequence relationships while making clustering more effective
  * Proceed to HDBSCAN clustering on reduced embeddings

**3. Pipeline Flow Implementation:**

```python
def generate_embeddings(sequences, batch_size=32, use_dnabert=True):
    """
    Generate embeddings following the hybrid pipeline:
    1. DNABERT-S transformer embeddings (preferred)
    2. If failed → K-mer vectorization + UMAP (fallback)
    """
    if use_dnabert:
        # Try DNABERT-S first
        model, tokenizer, device = load_dnabert_model()
        if model is not None:
            try:
                return generate_dnabert_embeddings(sequences, model, tokenizer, device, batch_size), "dnabert"
            except Exception as e:
                logger.warning("DNABERT-S failed, falling back to k-mer + UMAP")
    
    # Fallback: K-mer + UMAP
    embeddings = generate_kmer_embeddings(sequences, batch_size, use_umap=True, umap_n_components=50)
    return embeddings, "kmer_umap"
```

**4. DNABERT-S Implementation Details:**

```python
def generate_dnabert_embeddings(sequences, model, tokenizer, device, batch_size=32):
    """Generate 768D embeddings using DNABERT-S transformer."""
    embeddings_list = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        
        for sequence in batch_sequences:
            # Clean sequence
            clean_sequence = ''.join(c.upper() for c in sequence if c.upper() in 'ATCGN')
            
            # Tokenize
            inputs = tokenizer(clean_sequence, return_tensors='pt', 
                             padding=True, truncation=True, max_length=512)["input_ids"].to(device)
            
            with torch.no_grad():
                # Get hidden states
                outputs = model(inputs)
                hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                
                # Apply mean pooling
                embedding = torch.mean(hidden_states, dim=1).squeeze()
                embeddings_list.append(embedding.cpu().numpy())
    
    return np.array(embeddings_list)
```

**5. Enhanced K-mer + UMAP Fallback:**

```python
def generate_kmer_embeddings(sequences, batch_size=32, use_umap=True, umap_n_components=50):
    """Generate k-mer frequency embeddings with optional UMAP reduction."""
    # Generate 4-mer frequency vectors (256D)
    k = 4
    bases = ['A', 'T', 'C', 'G']
    all_kmers = [''.join(kmer) for kmer in product(bases, repeat=k)]
    
    # Process sequences to k-mer frequencies
    embeddings_array = compute_kmer_frequencies(sequences, all_kmers)
    
    # Apply UMAP dimensionality reduction
    if use_umap and len(sequences) > 10:
        umap_reducer = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=min(15, max(2, len(sequences) // 3)),
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        embeddings_array = umap_reducer.fit_transform(embeddings_array)
    
    return embeddings_array
```

**6. Clustering with HDBSCAN:**

Both embedding approaches use the same HDBSCAN clustering:

```python
def perform_clustering(embeddings, min_cluster_size=5, min_samples=None):
    """Perform HDBSCAN clustering on embeddings."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or 1,
        cluster_selection_epsilon=0.1,
        prediction_data=True
    )
    return clusterer.fit_predict(embeddings)
```

**7. Complete Pipeline Integration:**

```python
def run_discovery_pipeline(asv_filepath, output_filepath="discovery_engine_results.csv", 
                          min_cluster_size=5, batch_size=32, use_dnabert=True):
    """
    Run the complete hybrid discovery pipeline.
    
    Pipeline: fastp → dada2 → asv → embedding transformer + hdbscan → 
              if failed → k-mer vectorization → umap + hdbscan
    """
    # Step 1: Load ASV data (from DADA2 output)
    asv_df = load_asv_data(asv_filepath)
    sequences = asv_df['sequence'].tolist()
    
    # Step 2: Generate embeddings with hybrid approach
    embeddings, embedding_method = generate_embeddings(sequences, batch_size, use_dnabert)
    
    # Step 3: Perform HDBSCAN clustering
    cluster_labels = perform_clustering(embeddings, min_cluster_size)
    
    # Step 4: Create results with method tracking
    results_df = asv_df.copy()
    results_df['cluster_id'] = cluster_labels
    results_df['embedding_method'] = embedding_method  # Track which method was used
    
    # Step 5: Save results
    results_df.to_csv(output_filepath, index=False)
    
    return results_df
```

**8. Method Tracking and Transparency:**

The system tracks which embedding method was used for each run:
- `"dnabert"`: DNABERT-S transformer embeddings (768D) were used successfully
- `"kmer_umap"`: K-mer + UMAP fallback was used (due to DNABERT-S failure or being disabled)

This information is included in the output CSV and logged for analysis and debugging.

**Expected Output Format:**

The final CSV file contains four columns:
```csv
sequence,count,cluster_id,embedding_method
GTATGAATCCCGCCTGAAGGGAA...,10,0,dnabert
GATATACGCAGCGAATTGAGCGG...,10,0,dnabert
GGGCTACACACGTGCTACAATGG...,9,1,dnabert
ATCGATCGATCGATCGATCGATCG...,5,-1,kmer_umap
```

Where:
- `cluster_id`: Cluster assigned by HDBSCAN (-1 indicates noise/singleton)
- `embedding_method`: Which embedding approach was used for this analysis

**Hardware Compatibility:**

- **GPU Available**: DNABERT-S will be attempted and likely succeed, providing highest quality embeddings
- **CPU Only**: DNABERT-S will typically fail due to CUDA requirements, automatically falling back to the enhanced k-mer + UMAP approach
- **Limited Resources**: K-mer approach can be forced by setting `use_dnabert=False`

This hybrid design ensures the pipeline works across different deployment environments while maximizing the quality of results when advanced hardware is available.



here is the SOP for the Discovery Engine of DeepSea-AI:

-----

# **Standard Operating Procedure (SOP)**

| **Document Title:** | Implementation of the DeepSea-AI Discovery Engine |
| :--- | :--- |
| **Document ID:** | DS-AI-DE-SOP-V1.0 |
| **Effective Date:** | October 6, 2025 |
| **Author:** | Lead AI Writer |
| **Approved By:** | Project Lead |

-----

### **1.0 Purpose**

This Standard Operating Procedure (SOP) provides a detailed, step-by-step methodology for the development of the Python-based **Discovery Engine** for the DeepSea-AI project. The objective is to ensure the correct implementation of an unsupervised machine learning pipeline that processes Amplicon Sequence Variants (ASVs), generates embeddings using the **DNABERT-S** model, and performs clustering with **HDBSCAN** to identify potential novel taxa.

### **2.0 Scope**

This SOP applies to all development and engineering personnel tasked with creating, verifying, and maintaining the Discovery Engine module. It covers the entire workflow from data ingestion to the generation of the final clustered output file. This procedure supersedes the previous UMAP-based methodology.

### **3.0 Responsibilities**

  * **Lead Developer / Bioinformatician:** Responsible for executing the procedures outlined in this document, writing the Python script, and ensuring all technical specifications are met.
  * **Project Lead:** Responsible for reviewing the final implementation against this SOP and approving the module for integration into the main DeepSea-AI application.

### **4.0 Referenced Documents & Models**

  * **Input Data Specification:** `asv_table.csv` (format: `sequence`, `count`) - the name can differ
  * **Primary Model:** DNABERT-S (Hugging Face Identifier: `zhihan1996/DNABERT-S`) [1, 2]
  * **Clustering Algorithm:** HDBSCAN
  * **Core Libraries:** PyTorch, Hugging Face Transformers, pandas, HDBSCAN, NumPy

### **5.0 Procedure**

The implementation shall be divided into six distinct phases, executed sequentially.

#### **5.1 Phase 1: Environment and Project Setup**

1.1. **Establish Dependencies:** Ensure the project's Python environment includes the following core libraries:
\*   `pandas`
\*   `torch`
\*   `transformers`
\*   `hdbscan`
\*   `numpy`

1.2. **Script Structure:** Create a modular Python script with distinct functions for each major operational phase (e.g., data loading, embedding generation, clustering). Adhere to standard coding practices, including type hinting and descriptive comments.

#### **5.2 Phase 2: Data Ingestion**

2.1. **Load Input Data:** Implement a function to read the specified input file (`asv_table.csv`) into a pandas DataFrame.

2.2. **Validate Data:** Verify that the DataFrame contains the required columns: `sequence` and `count`.

2.3. **Extract Sequences:** Isolate the `sequence` column into a list of strings. This list will serve as the primary input for the embedding generation phase.

#### **5.3 Phase 3: Model Loading and Configuration**

3.1. **Device Configuration:** Programmatically detect and select the appropriate compute device. Prioritize CUDA-enabled GPUs if available; otherwise, default to CPU.
` python import torch device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  `

3.2. **Load Pre-trained Model:** Load the DNABERT-S tokenizer and model from the Hugging Face Hub using the identifier `zhihan1996/DNABERT-S`.
\`\`\`python
from transformers import AutoTokenizer, AutoModel

````
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
```
````

3.3. **Prepare Model for Inference:** Move the loaded model to the selected compute device and set it to evaluation mode to disable dropout and other training-specific layers.
` python model.to(device) model.eval()  `

#### **5.4 Phase 4: Embedding Generation**

4.1. **Develop Batch Processing Function:** Create a function that accepts the list of DNA sequences and processes them in batches to manage memory usage efficiently. A configurable `batch_size` parameter (e.g., default of 32) should be included.

4.2. **Tokenize and Embed:** Within the batch processing loop, for each batch of sequences:
1\.  Tokenize the DNA strings using the loaded tokenizer.
2\.  Move the tokenized tensors to the selected compute device.
3\.  Pass the tensors through the model within a `torch.no_grad()` context to disable gradient calculations.

4.3. **Apply Mean Pooling:** For each sequence's output hidden states, apply a mean pooling operation across the sequence length dimension (`dim=1`) to produce a single, fixed-size 768-dimensional embedding vector.

4.4. **Aggregate Embeddings:** Collect the embedding vectors from all batches and consolidate them into a single NumPy array. This array will be the input for the clustering phase.

#### **5.5 Phase 5: Unsupervised Clustering**

5.1. **Initialize Clusterer:** Instantiate the HDBSCAN clusterer from the `hdbscan` library.
\*   The `min_cluster_size` parameter should be configurable (e.g., default to 5).
\*   Set `prediction_data=True` to enable future predictions on new data points if required.
` python import hdbscan clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)  `

5.2. **Fit and Predict Clusters:** Fit the HDBSCAN model to the NumPy array of embeddings.
` python cluster_labels = clusterer.fit_predict(embeddings_array)  `
*Note: The resulting `cluster_labels` array will contain the cluster ID for each sequence. Noise points will be assigned a label of -1.*

#### **5.6 Phase 6: Output Generation**

6.1. **Integrate Results:** Add the `cluster_labels` array as a new column, named `cluster_id`, to the original pandas DataFrame from Phase 2.

6.2. **Validate Output DataFrame:** Confirm the final DataFrame contains the three required columns: `sequence`, `count`, and `cluster_id`.

6.3. **Save Results:** Export the final DataFrame to a CSV file named `discovery_engine_results.csv`, ensuring the index is not written to the file.

### **6.0 Approval**

| **Role** | **Name** | **Signature** | **Date** |
| :--- | :--- | :--- | :--- |
| **Author** | | | |
| **Approved By** | | | |

-----