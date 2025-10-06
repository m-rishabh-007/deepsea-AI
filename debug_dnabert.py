#!/usr/bin/env python3
"""Debug script for DNABERT-S tokenizer issues."""

import traceback
import sys
import torch
from transformers import AutoTokenizer, AutoModel

def test_dnabert_tokenizer():
    """Test DNABERT-S tokenizer with various sequences."""
    print("Testing DNABERT-S tokenizer...")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-S', trust_remote_code=True)
        print("✓ Tokenizer loaded successfully")
        
        # Test sequences of different lengths
        test_sequences = [
            "ATCG",  # Very short
            "ATCGATCGATCGATCG",  # Medium
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",  # Long (150bp)
        ]
        
        for i, seq in enumerate(test_sequences):
            print(f"\nTesting sequence {i+1} (length {len(seq)}):")
            print(f"Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
            
            try:
                # Test tokenization
                result = tokenizer(seq, return_tensors='pt', padding=True, truncation=True, max_length=512)
                print(f"✓ Tokenization successful - input_ids shape: {result['input_ids'].shape}")
                
                # Test with model if available
                try:
                    print("Testing with model...")
                    model = AutoModel.from_pretrained('zhihan1996/DNABERT-S', trust_remote_code=True)
                    
                    with torch.no_grad():
                        outputs = model(result['input_ids'])
                        if hasattr(outputs, 'last_hidden_state'):
                            hidden_states = outputs.last_hidden_state
                        else:
                            hidden_states = outputs[0]
                        
                        embedding = torch.mean(hidden_states, dim=1).squeeze()
                        print(f"✓ Model processing successful - embedding shape: {embedding.shape}")
                        
                except Exception as model_e:
                    print(f"✗ Model processing failed: {model_e}")
                    traceback.print_exc()
                    
            except Exception as token_e:
                print(f"✗ Tokenization failed: {token_e}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_dnabert_tokenizer()