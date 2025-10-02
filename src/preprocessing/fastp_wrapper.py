import subprocess
from pathlib import Path
from loguru import logger
from typing import List, Optional


def run_fastp(input_fastq: List[str], output_dir: str, threads: int = 4, qualified_quality_phred: int = 15,
              length_required: int = 50, detect_adapter_for_pe: bool = True, json_report: str = "fastp_report.json",
              html_report: str = "fastp_report.html") -> dict:
    """Run fastp on one or two FASTQ files.

    Parameters
    ----------
    input_fastq : list
        One (single-end) or two (paired-end) FASTQ file paths.
    output_dir : str
        Directory where cleaned FASTQ and reports are written.
    threads : int
        Number of threads.
    ... (other parameters)

    Returns
    -------
    dict : Paths to outputs.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if len(input_fastq) not in (1, 2):
        raise ValueError("input_fastq must contain 1 (SE) or 2 (PE) files")

    base_names = [Path(f).stem for f in input_fastq]
    if len(input_fastq) == 1:
        out_clean = Path(output_dir) / f"{base_names[0]}_clean.fastq.gz"
        cmd = [
            "fastp",
            "-i", input_fastq[0],
            "-o", str(out_clean),
            "-w", str(threads),
            "-j", str(Path(output_dir) / json_report),
            "-h", str(Path(output_dir) / html_report),
            "-q", str(qualified_quality_phred),
            "-l", str(length_required)
        ]
    else:
        out_r1 = Path(output_dir) / f"{base_names[0]}_clean.fastq.gz"
        out_r2 = Path(output_dir) / f"{base_names[1]}_clean.fastq.gz"
        cmd = [
            "fastp",
            "-i", input_fastq[0],
            "-I", input_fastq[1],
            "-o", str(out_r1),
            "-O", str(out_r2),
            "-w", str(threads),
            "-j", str(Path(output_dir) / json_report),
            "-h", str(Path(output_dir) / html_report),
            "-q", str(qualified_quality_phred),
            "-l", str(length_required)
        ]
        if detect_adapter_for_pe:
            cmd.append("--detect_adapter_for_pe")

    logger.info(f"Running fastp: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    return {
        "clean_reads": [str(p) for p in Path(output_dir).glob("*_clean.fastq.gz")],
        "json_report": str(Path(output_dir) / json_report),
        "html_report": str(Path(output_dir) / html_report)
    }
