"""
Test suite for Discovery Engine module

This module tests the Stage 2 functionality of DeepSea-AI, including:
- ASV data loading and validation
- Hybrid embedding generation (DNABERT-S primary, k-mer+UMAP fallback)
- HDBSCAN clustering functionality
- Complete discovery pipeline integration
- Pipeline flow: fastp → dada2 → asv → embedding transformer + hdbscan → if failed → k-mer vectorization → umap + hdbscan

Author: DeepSea-AI Team
Date: October 6, 2025
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.discovery.discovery_engine import (
    load_asv_data,
    generate_embeddings,
    generate_kmer_embeddings,
    perform_clustering,
    load_dnabert_model,
    run_discovery_pipeline
)


class TestLoadAsvData:
    """Test ASV data loading functionality."""
    
    def test_load_valid_asv_data(self):
        """Test loading valid ASV CSV file."""
        # Create test data
        test_data = {
            'sequence': ['ATCGATCGATCG', 'GCTAGCTAGCTA', 'TTAAGGCCTTAA'],
            'count': [10, 5, 3]
        }
        df = pd.DataFrame(test_data)
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Test loading
            result_df = load_asv_data(temp_path)
            
            # Assertions
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 3
            assert list(result_df.columns) == ['sequence', 'count']
            assert result_df['sequence'].iloc[0] == 'ATCGATCGATCG'
            assert result_df['count'].iloc[0] == 10
            
        finally:
            Path(temp_path).unlink()  # Clean up
    
    def test_load_missing_file(self):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_asv_data('nonexistent_file.csv')
    
    def test_load_missing_columns(self):
        """Test loading CSV with missing required columns raises ValueError."""
        # Create CSV with wrong columns
        test_data = {'wrong_col': [1, 2, 3], 'another_col': [4, 5, 6]}
        df = pd.DataFrame(test_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_asv_data(temp_path)
        finally:
            Path(temp_path).unlink()


class TestGenerateEmbeddings:
    """Test hybrid embedding generation functionality."""
    
    @patch('src.discovery.discovery_engine.load_dnabert_model')
    @patch('src.discovery.discovery_engine.generate_dnabert_embeddings')
    def test_generate_embeddings_dnabert_success(self, mock_dnabert_embeddings, mock_load_model):
        """Test successful DNABERT-S embedding generation."""
        # Mock successful model loading
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_device = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer, mock_device)
        
        # Mock successful DNABERT embeddings
        test_embeddings = np.random.rand(2, 768)
        mock_dnabert_embeddings.return_value = test_embeddings
        
        sequences = ['ATCGATCGATCG', 'GCTAGCTAGCTA']
        
        # Call function
        embeddings, method = generate_embeddings(sequences, use_dnabert=True)
        
        # Assertions
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)
        assert method == "dnabert"
        mock_load_model.assert_called_once()
        mock_dnabert_embeddings.assert_called_once()
    
    @patch('src.discovery.discovery_engine.load_dnabert_model')
    @patch('src.discovery.discovery_engine.generate_dnabert_embeddings')
    @patch('src.discovery.discovery_engine.generate_kmer_embeddings')
    def test_generate_embeddings_fallback_to_kmer(self, mock_kmer_embeddings, 
                                                 mock_dnabert_embeddings, mock_load_model):
        """Test fallback to k-mer embeddings when DNABERT-S fails."""
        # Mock successful model loading but failed embedding generation
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_device = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer, mock_device)
        
        # Mock DNABERT failure
        mock_dnabert_embeddings.side_effect = Exception("DNABERT-S CPU incompatibility")
        
        # Mock successful k-mer embeddings
        test_kmer_embeddings = np.random.rand(2, 50)  # UMAP reduced
        mock_kmer_embeddings.return_value = test_kmer_embeddings
        
        sequences = ['ATCGATCGATCG', 'GCTAGCTAGCTA']
        
        # Call function
        embeddings, method = generate_embeddings(sequences, use_dnabert=True)
        
        # Assertions
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 50)  # K-mer + UMAP dimensions
        assert method == "kmer_umap"
        mock_load_model.assert_called_once()
        mock_dnabert_embeddings.assert_called_once()
        mock_kmer_embeddings.assert_called_once_with(sequences, 32, use_umap=True, umap_n_components=50)
    
    @patch('src.discovery.discovery_engine.generate_kmer_embeddings')
    def test_generate_embeddings_force_kmer(self, mock_kmer_embeddings):
        """Test forcing k-mer embeddings (skip DNABERT-S)."""
        # Mock k-mer embeddings
        test_kmer_embeddings = np.random.rand(2, 256)  # Raw k-mer
        mock_kmer_embeddings.return_value = test_kmer_embeddings
        
        sequences = ['ATCGATCGATCG', 'GCTAGCTAGCTA']
        
        # Call function with use_dnabert=False
        embeddings, method = generate_embeddings(sequences, use_dnabert=False)
        
        # Assertions
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 256)
        assert method == "kmer_umap"
        mock_kmer_embeddings.assert_called_once_with(sequences, 32, use_umap=True, umap_n_components=50)


class TestGenerateKmerEmbeddings:
    """Test k-mer embedding generation with UMAP."""
    
    @patch('umap.UMAP')
    def test_generate_kmer_embeddings_with_umap(self, mock_umap_class):
        """Test k-mer embedding generation with UMAP reduction."""
        # Mock UMAP reducer
        mock_reducer = Mock()
        mock_reduced_embeddings = np.random.rand(3, 20)  # Reduced dimensions
        mock_reducer.fit_transform.return_value = mock_reduced_embeddings
        mock_reducer.embedding_ = Mock()
        mock_reducer.embedding_.var.return_value.sum.return_value = 45.67
        mock_umap_class.return_value = mock_reducer
        
        sequences = ['ATCGATCGATCGATCG', 'GCTAGCTAGCTAGCTA', 'TTAAGGCCTTAAGGCC']
        
        # Call function
        embeddings = generate_kmer_embeddings(sequences, use_umap=True, umap_n_components=20)
        
        # Assertions
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 20)  # UMAP reduced
        mock_umap_class.assert_called_once()
        mock_reducer.fit_transform.assert_called_once()
    
    def test_generate_kmer_embeddings_without_umap(self):
        """Test k-mer embedding generation without UMAP (raw k-mer)."""
        sequences = ['ATCGATCGATCGATCG', 'GCTAGCTAGCTAGCTA']
        
        # Call function
        embeddings = generate_kmer_embeddings(sequences, use_umap=False)
        
        # Assertions
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 256)  # Raw k-mer dimensions (4^4)
        
        # Check that frequencies are normalized
        for i in range(len(embeddings)):
            # Each row should be a valid frequency distribution
            assert embeddings[i].sum() <= 1.0  # Frequencies sum to ≤ 1.0
            assert np.all(embeddings[i] >= 0)   # All frequencies ≥ 0
    
    @patch('umap.UMAP')
    def test_generate_kmer_embeddings_umap_fallback(self, mock_umap_class):
        """Test UMAP fallback when too few sequences."""
        # Don't need to mock UMAP failure since function checks sequence count
        sequences = ['ATCGATCGATCGATCG']  # Only 1 sequence, below threshold
        
        # Call function (should fallback to raw k-mer due to insufficient data)
        embeddings = generate_kmer_embeddings(sequences, use_umap=True, umap_n_components=20)
        
        # Assertions - should return raw k-mer, not UMAP reduced
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 256)  # Raw k-mer, not UMAP reduced
        mock_umap_class.assert_not_called()  # UMAP should not be called


class TestPerformClustering:
    """Test HDBSCAN clustering functionality."""
    
    @patch('hdbscan.HDBSCAN')
    def test_perform_clustering_basic(self, mock_hdbscan_class):
        """Test basic clustering functionality."""
        # Mock HDBSCAN clusterer
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, -1])
        mock_hdbscan_class.return_value = mock_clusterer
        
        # Test data
        embeddings = np.random.rand(5, 768)
        
        # Call function
        cluster_labels = perform_clustering(embeddings, min_cluster_size=2)
        
        # Assertions
        assert isinstance(cluster_labels, np.ndarray)
        assert len(cluster_labels) == 5
        mock_hdbscan_class.assert_called_once_with(
            min_cluster_size=2,
            min_samples=None,
            prediction_data=True
        )
    
    @patch('builtins.__import__')
    def test_perform_clustering_import_error(self, mock_import):
        """Test clustering handles missing HDBSCAN library."""
        def import_side_effect(name, *args, **kwargs):
            if name == 'hdbscan':
                raise ImportError("HDBSCAN not found")
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        embeddings = np.random.rand(3, 768)
        
        with pytest.raises(ImportError):
            perform_clustering(embeddings)


class TestLoadDnabertModel:
    """Test DNABERT-S model loading functionality."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.device')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_load_dnabert_model_success(self, mock_auto_model, mock_auto_tokenizer, 
                                       mock_torch_device, mock_cuda_available):
        """Test successful model loading."""
        # Mock device detection
        mock_cuda_available.return_value = True
        mock_torch_device.return_value = "cuda:0"
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_auto_model.return_value = mock_model
        mock_auto_tokenizer.return_value = mock_tokenizer
        
        # Call function
        model, tokenizer, device = load_dnabert_model()
        
        # Assertions
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_auto_tokenizer.assert_called_once_with(
            "zhihan1996/DNABERT-S", trust_remote_code=True
        )
        mock_auto_model.assert_called_once_with(
            "zhihan1996/DNABERT-S", trust_remote_code=True
        )
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()
    
    @patch('transformers.AutoTokenizer.from_pretrained', side_effect=ImportError("transformers not found"))
    def test_load_dnabert_model_import_error(self, mock_auto_tokenizer):
        """Test model loading handles missing transformers library."""
        with pytest.raises(ImportError):
            load_dnabert_model()


class TestRunDiscoveryPipeline:
    """Test complete discovery pipeline functionality."""
    
    @patch('src.discovery.discovery_engine.load_asv_data')
    @patch('src.discovery.discovery_engine.generate_embeddings')
    @patch('src.discovery.discovery_engine.perform_clustering')
    def test_run_discovery_pipeline_success(self, mock_clustering, mock_embeddings, mock_load_asv):
        """Test successful complete pipeline run with embedding method tracking."""
        # Mock ASV data
        test_asv_data = pd.DataFrame({
            'sequence': ['ATCGATCGATCG', 'GCTAGCTAGCTA', 'TTAAGGCCTTAA'],
            'count': [10, 5, 3]
        })
        mock_load_asv.return_value = test_asv_data
        
        # Mock embeddings with method tracking
        mock_embeddings.return_value = (np.random.rand(3, 768), "dnabert")
        
        # Mock clustering
        mock_clustering.return_value = np.array([0, 0, 1])
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_asv_data.to_csv(f.name, index=False)
            input_path = f.name
        
        # Create temporary output path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            # Call function
            result_df = run_discovery_pipeline(
                asv_filepath=input_path,
                output_filepath=output_path,
                min_cluster_size=2,
                batch_size=16
            )
            
            # Assertions
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 3
            assert 'cluster_id' in result_df.columns
            assert 'sequence' in result_df.columns
            assert 'count' in result_df.columns
            assert 'embedding_method' in result_df.columns  # New column
            
            # Check embedding method tracking
            assert result_df['embedding_method'].iloc[0] == "dnabert"
            
            # Check that output file was created
            assert Path(output_path).exists()
            
            # Read output file and verify embedding method is saved
            saved_df = pd.read_csv(output_path)
            assert 'embedding_method' in saved_df.columns
            assert saved_df['embedding_method'].iloc[0] == "dnabert"
            
            # Verify function calls
            mock_load_asv.assert_called_once_with(input_path)
            mock_embeddings.assert_called_once()
            mock_clustering.assert_called_once()
            
        finally:
            # Clean up
            Path(input_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()
    
    @patch('src.discovery.discovery_engine.load_asv_data')
    @patch('src.discovery.discovery_engine.generate_embeddings')
    @patch('src.discovery.discovery_engine.perform_clustering')
    def test_run_discovery_pipeline_fallback_method(self, mock_clustering, mock_embeddings, mock_load_asv):
        """Test pipeline with k-mer fallback method tracking."""
        # Mock ASV data
        test_asv_data = pd.DataFrame({
            'sequence': ['ATCGATCGATCG', 'GCTAGCTAGCTA'],
            'count': [10, 5]
        })
        mock_load_asv.return_value = test_asv_data
        
        # Mock embeddings with k-mer fallback
        mock_embeddings.return_value = (np.random.rand(2, 50), "kmer_umap")
        
        # Mock clustering
        mock_clustering.return_value = np.array([0, 1])
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_asv_data.to_csv(f.name, index=False)
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            # Call function
            result_df = run_discovery_pipeline(
                asv_filepath=input_path,
                output_filepath=output_path,
                use_dnabert=True  # Should attempt DNABERT but fall back
            )
            
            # Assertions
            assert isinstance(result_df, pd.DataFrame)
            assert result_df['embedding_method'].iloc[0] == "kmer_umap"
            
        finally:
            # Clean up
            Path(input_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()
    
    @patch('src.discovery.discovery_engine.load_asv_data')
    def test_run_discovery_pipeline_file_not_found(self, mock_load_asv):
        """Test pipeline handles file not found error."""
        mock_load_asv.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            run_discovery_pipeline("nonexistent.csv", "output.csv")


@pytest.mark.integration
class TestDiscoveryEngineIntegration:
    """Integration tests for Discovery Engine with actual dependencies."""
    
    def test_small_dataset_end_to_end(self):
        """Test complete pipeline with small synthetic dataset."""
        # Create small test dataset
        test_sequences = [
            'ATCGATCGATCGATCGATCG',  # 20bp sequences
            'GCTAGCTAGCTAGCTAGCTA',
            'TTAAGGCCTTAAGGCCTTAA',
            'CGTACGTACGTACGTACGTA'
        ]
        
        test_data = pd.DataFrame({
            'sequence': test_sequences,
            'count': [10, 8, 6, 4]
        })
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            # Test data loading (should work without mocking)
            loaded_data = load_asv_data(input_path)
            assert len(loaded_data) == 4
            assert list(loaded_data.columns) == ['sequence', 'count']
            
            # Test clustering with random embeddings (HDBSCAN should work)
            random_embeddings = np.random.rand(4, 768)
            cluster_labels = perform_clustering(random_embeddings, min_cluster_size=2)
            assert isinstance(cluster_labels, np.ndarray)
            assert len(cluster_labels) == 4
            
        finally:
            # Clean up
            Path(input_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])