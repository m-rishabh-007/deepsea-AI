import subprocess
from pathlib import Path
from loguru import logger
from typing import Dict, Literal


def run_dada2(clean_dir: str, output_dir: str, output_prefix: str = "dada2", mode: Literal['single','paired'] = 'single',
              max_ee: int = 2, trunc_q: int = 2, pool_method: str = "pseudo") -> Dict[str, str]:
    """Invoke the DADA2 R script and return output paths.

    Parameters
    ----------
    clean_dir : str
        Directory containing *_clean.fastq.gz files from fastp.
    output_dir : str
        Directory to place ASV outputs.
    output_prefix : str
        Prefix for output files.
    max_ee : int
        Maximum expected errors (currently passed for future extension; not used directly in script filtering).
    trunc_q : int
        Truncation quality score.
    pool_method : str
        DADA2 pooling strategy.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    script_path = Path("scripts") / "run_dada2.R"
    cmd = [
        "Rscript", str(script_path), clean_dir, output_prefix, output_dir, mode, str(max_ee), str(trunc_q), pool_method
    ]
    logger.info(f"Running DADA2: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    asv_csv = Path(output_dir) / f"{output_prefix}_asv_table.csv"
    summary_json = Path(output_dir) / f"{output_prefix}_summary.json"
    if not asv_csv.exists():
        raise FileNotFoundError("Expected ASV table not found: " + str(asv_csv))
    return {"asv_table": str(asv_csv), "summary": str(summary_json)}
