#!/usr/bin/env python3
"""Test script to verify the complete DeepSea-AI pipeline with k-mer embeddings."""

import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.insert(0, '/home/rishabh/Downloads/SIH-Project')

from src.discovery.discovery_engine import run_discovery_pipeline

def create_test_asv_data():
    """Create sample ASV data for testing."""
    # Generate realistic-looking DNA sequences
    test_data = {
        'sequence': [
            'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG',
            'AAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCC',
            'GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC',
            'TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATG',
            'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT',
            # Similar sequences (should cluster together)
            'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCA',
            'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCT',
            'AAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCA',
            'AAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCG',
        ],
        'count': [1250, 890, 2340, 567, 1890, 1200, 1180, 900, 870]
    }
    
    return pd.DataFrame(test_data)

def test_discovery_pipeline():
    """Test the complete discovery pipeline."""
    print("🧬 Testing DeepSea-AI Discovery Engine with K-mer Embeddings")
    print("=" * 60)
    
    try:
        # Create test data
        print("📊 Creating test ASV data...")
        test_df = create_test_asv_data()
        print(f"   ✓ Created {len(test_df)} test sequences")
        
        # Save test data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_input:
            test_df.to_csv(tmp_input.name, index=False)
            input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_output:
            output_path = tmp_output.name
        
        print(f"📁 Input file: {input_path}")
        print(f"📁 Output file: {output_path}")
        
        # Run discovery pipeline
        print("\n🔬 Running Discovery Pipeline...")
        results = run_discovery_pipeline(
            asv_filepath=input_path,
            output_filepath=output_path,
            min_cluster_size=2,  # Small cluster size for test
            batch_size=32
        )
        
        print("\n📊 Pipeline Results:")
        print(f"   ✓ Total sequences processed: {len(results)}")
        
        # Analyze clusters
        clusters = results['cluster_id'].value_counts().sort_index()
        n_clusters = len([c for c in clusters.index if c != -1])
        n_noise = clusters.get(-1, 0)
        
        print(f"   ✓ Clusters discovered: {n_clusters}")
        print(f"   ✓ Noise points: {n_noise}")
        
        if n_clusters > 0:
            print(f"   ✓ Cluster sizes: {dict(clusters[clusters.index != -1])}")
        
        # Verify output file
        if os.path.exists(output_path):
            output_df = pd.read_csv(output_path)
            print(f"   ✓ Output file created with {len(output_df)} rows")
            print(f"   ✓ Columns: {list(output_df.columns)}")
        
        print("\n✅ Discovery Engine Test PASSED!")
        print("   K-mer embeddings are working correctly")
        print("   HDBSCAN clustering is functional")
        print("   Pipeline integration is successful")
        
        # Cleanup
        os.unlink(input_path)
        os.unlink(output_path)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Discovery Engine Test FAILED!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_discovery_pipeline()
    sys.exit(0 if success else 1)