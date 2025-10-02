from pathlib import Path
from typing import List, Dict, Tuple
import re

PAIR_PATTERNS = [
    (re.compile(r"(.+?)(?:_R)?1(\.fastq(?:\.gz)?)$", re.IGNORECASE), "R1", "R2"),
    (re.compile(r"(.+?)\.1(\.fastq(?:\.gz)?)$", re.IGNORECASE), ".1", ".2"),
]


def detect_fastq_layout(raw_dir: str | Path) -> Dict[str, List[Tuple[str, str]]]:
    """Detect paired and single-end FASTQ files.

    Returns dict with keys:
      pairs: list of (r1, r2)
      singles: list of single-end files
    """
    raw_dir = Path(raw_dir)
    fastqs = [p for p in raw_dir.glob("*.fastq*") if p.is_file()]
    used = set()
    pairs: List[Tuple[str, str]] = []
    singles: List[str] = []

    for f in fastqs:
        if f in used:
            continue
        fname = f.name
        matched = False
        for pattern, tag1, tag2 in PAIR_PATTERNS:
            m = pattern.match(fname)
            if m:
                base, ext = m.groups()
                # Construct candidate R2 file by replacing tag1 with tag2 in matching logic
                if tag1 == "R1":
                    r2_candidates = [raw_dir / f"{base}R2{ext}", raw_dir / f"{base}_R2{ext}"]
                else:
                    # .1 style
                    r2_candidates = [raw_dir / f"{base}.2{ext}"]
                for r2 in r2_candidates:
                    if r2.exists():
                        pairs.append((str(f), str(r2)))
                        used.add(f)
                        used.add(r2)
                        matched = True
                        break
            if matched:
                break
    # Remaining files are singles
    for f in fastqs:
        if f not in used:
            singles.append(str(f))
    return {"pairs": pairs, "singles": singles}


def representative_inputs(layout: Dict[str, List[Tuple[str, str]]]) -> List[str]:
    """Return a representative list of input files for initial QC (fastp wrapper currently accepts 1 or 2)."""
    if layout["pairs"]:
        # Use first pair for now (full multi-sample support can expand later)
        return list(layout["pairs"][0])
    if layout["singles"]:
        return [layout["singles"][0]]
    return []
