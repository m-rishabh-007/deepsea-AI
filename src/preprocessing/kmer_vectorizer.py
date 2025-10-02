from collections import Counter
from itertools import product
from pathlib import Path
from typing import List, Dict
import pandas as pd
import math

ALPHABET = ["A", "C", "G", "T"]


def _all_kmers(k: int) -> List[str]:
    return ["".join(p) for p in product(ALPHABET, repeat=k)]


def compute_kmer_vector(sequence: str, k: int) -> Dict[str, float]:
    seq = sequence.upper()
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1) if set(seq[i:i+k]) <= set(ALPHABET)]
    counts = Counter(kmers)
    total = sum(counts.values()) or 1
    return {kmer: counts.get(kmer, 0) / total for kmer in _all_kmers(k)}


def vectorize_asv_table(asv_csv: str, k: int, normalize: bool = True) -> pd.DataFrame:
    df = pd.read_csv(asv_csv)
    if 'sequence' not in df.columns:
        raise ValueError('ASV table must have a sequence column')
    vectors = []
    kmers = _all_kmers(k)
    for _, row in df.iterrows():
        vec = compute_kmer_vector(row['sequence'], k)
        if not normalize:
            # Multiply back by length-k+1 to get raw counts approximated
            length = len(row['sequence']) - k + 1
            for m in kmers:
                vec[m] = round(vec[m] * max(length, 1))
        vec['sequence'] = row['sequence']
        vec['count'] = row.get('count', math.nan)
        vectors.append(vec)
    return pd.DataFrame(vectors)


def save_vectors(df: pd.DataFrame, output_csv: str):
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
