from src.preprocessing.kmer_vectorizer import compute_kmer_vector

def test_kmer_vector_simple():
    seq = "ACGTACGT"
    k = 2
    vec = compute_kmer_vector(seq, k)
    # All 16 possible 2-mers should exist
    assert len(vec) == 4 ** k
    # Frequencies sum to ~1
    assert abs(sum(vec.values()) - 1.0) < 1e-6
    # Check specific k-mer count
    assert vec['AC'] > 0
