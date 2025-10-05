import yaml
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import List, Optional, Dict
from loguru import logger

from src.utils.logging_setup import setup_logging
from src.preprocessing.fastp_wrapper import run_fastp
from src.preprocessing.dada2_wrapper import run_dada2
from src.preprocessing.read_detection import detect_fastq_layout, representative_inputs
def load_config(path: str = 'config/pipeline.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def run_pipeline(
    raw_dir: str | Path,
    interim_dir: str | Path,
    processed_dir: str | Path,
    config_path: str = 'config/pipeline.yaml'
) -> dict:
    """Run the Stage 1 pipeline for a specific job.

    Parameters
    ----------
    raw_dir / interim_dir / processed_dir : job-specific directories.
    """
    cfg = load_config(config_path)
    logger_obj = setup_logging(cfg.get('paths', {}).get('logs_dir', 'logs'), cfg.get('logging', {}).get('level', 'INFO'))
    logger_obj.info(f'Starting Stage 1 pipeline job with raw_dir={raw_dir}')

    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    processed_dir = Path(processed_dir)
    ensure_dir(interim_dir)
    ensure_dir(processed_dir)

    layout = detect_fastq_layout(raw_dir)
    representative = representative_inputs(layout)
    if not representative:
        raise FileNotFoundError(f'No FASTQ files found in {raw_dir}')
    mode = 'paired' if len(representative) == 2 else 'single'

    progress_meta = {"history": []}

    def append_progress(step: str, status: str, message: str | None = None):
        entry = {
            "step": step,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if message:
            entry["message"] = message
        progress_meta["history"].append(entry)
        progress_meta["step"] = step
        progress_meta["status"] = status

    append_progress("fastp", "running", "Preparing to run fastp")

    fastp_cfg = cfg['fastp']
    fastp_outputs = None
    if fastp_cfg.get('enabled', True):
        # For now pick first two if paired, will upgrade with detection later.
        inputs = representative
        append_progress("fastp", "running", "Executing fastp")
        fastp_outputs = run_fastp(
            input_fastq=inputs,
            output_dir=str(interim_dir),
            threads=fastp_cfg.get('threads', 4),
            qualified_quality_phred=fastp_cfg.get('qualified_quality_phred', 15),
            length_required=fastp_cfg.get('length_required', 50),
            detect_adapter_for_pe=fastp_cfg.get('detect_adapter_for_pe', True),
            json_report=fastp_cfg.get('json_report', 'fastp_report.json'),
            html_report=fastp_cfg.get('html_report', 'fastp_report.html')
        )
        append_progress("fastp", "completed", "fastp finished")
    else:
        logger.info('Skipping fastp step per config.')
        append_progress("fastp", "skipped", "fastp disabled in configuration")

    append_progress("dada2", "running", "Preparing to run DADA2")
    dada2_cfg = cfg['dada2']
    dada2_outputs = None
    if dada2_cfg.get('enabled', True):
        append_progress("dada2", "running", "Executing DADA2")
        dada2_outputs = run_dada2(
            clean_dir=str(interim_dir),
            output_dir=str(processed_dir),
            output_prefix=dada2_cfg.get('output_prefix', 'dada2'),
            mode=mode,
            max_ee=dada2_cfg.get('max_ee', 2),
            trunc_q=dada2_cfg.get('trunc_q', 2),
            pool_method=dada2_cfg.get('pool_method', 'pseudo')
        )
        append_progress("dada2", "completed", "DADA2 finished")
    else:
        logger.info('Skipping DADA2 step per config.')
        append_progress("dada2", "skipped", "DADA2 disabled in configuration")

    meta = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
        'raw_dir': str(raw_dir),
        'interim_dir': str(interim_dir),
        'processed_dir': str(processed_dir),
        'fastp': fastp_outputs,
        'dada2': dada2_outputs | {"mode": mode} if dada2_outputs else None,
        'progress': progress_meta
    }
    meta_path = processed_dir / 'stage1_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f'Metadata written to {meta_path}')
    return meta


if __name__ == '__main__':
    # Fallback to config default paths for manual invocation
    cfg = load_config()
    run_pipeline(
        raw_dir=cfg['paths']['raw_fastq_dir'],
        interim_dir=cfg['paths']['interim_dir'],
        processed_dir=cfg['paths']['processed_dir']
    )
