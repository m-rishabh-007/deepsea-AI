import yaml
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import List, Optional, Dict
import numpy as np

from src.utils.logging_setup import setup_logging
from src.preprocessing.fastp_wrapper import run_fastp
from src.preprocessing.dada2_wrapper import run_dada2
from src.preprocessing.read_detection import detect_fastq_layout, representative_inputs
from src.discovery.discovery_engine import run_discovery_pipeline

logger = setup_logging()
from src.discovery.discovery_engine import run_discovery_pipeline
def load_config(config_path: Path = None):
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'pipeline.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def convert_numpy_types(obj):
    """Convert numpy types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def run_pipeline(
    raw_dir: str | Path,
    interim_dir: str | Path,
    processed_dir: str | Path,
    config_path: str = 'config/pipeline.yaml',
    enable_stage2: bool = True
) -> dict:
    """Run the complete DeepSea-AI pipeline (Stage 1 + Stage 2).

    Parameters
    ----------
    raw_dir / interim_dir / processed_dir : job-specific directories.
    config_path : Path to pipeline configuration file.
    enable_stage2 : Whether to run Stage 2 (Discovery Engine).
    """
    cfg = load_config(config_path)
    logger_obj = setup_logging(cfg.get('paths', {}).get('logs_dir', 'logs'), cfg.get('logging', {}).get('level', 'INFO'))
    logger_obj.info(f'Starting DeepSea-AI pipeline with raw_dir={raw_dir}, stage2_enabled={enable_stage2}')

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

    # Stage 2: Discovery Engine
    discovery_outputs = None
    if enable_stage2 and dada2_outputs and cfg.get('discovery', {}).get('enabled', True):
        append_progress("discovery", "running", "Preparing to run Discovery Engine")
        
        # Check if ASV table exists
        asv_table_path = processed_dir / dada2_outputs.get('asv_table', 'dada2_asv_table.csv')
        
        if asv_table_path.exists():
            try:
                logger.info(f"Running Discovery Engine on ASV table: {asv_table_path}")
                append_progress("discovery", "running", "Executing Discovery Engine (DNABERT-S + HDBSCAN)")
                
                discovery_cfg = cfg.get('discovery', {})
                discovery_output_path = processed_dir / 'discovery_engine_results.csv'
                
                discovery_results = run_discovery_pipeline(
                    asv_filepath=str(asv_table_path),
                    output_filepath=str(discovery_output_path),
                    min_cluster_size=discovery_cfg.get('min_cluster_size', 5),
                    batch_size=discovery_cfg.get('batch_size', 32)
                )
                
                discovery_outputs = {
                    'results_file': str(discovery_output_path),
                    'clusters_found': int(len(set(discovery_results['cluster_id'])) - (1 if -1 in discovery_results['cluster_id'].values else 0)),
                    'noise_points': int((discovery_results['cluster_id'] == -1).sum()),
                    'total_sequences': int(len(discovery_results))
                }
                
                append_progress("discovery", "completed", f"Discovery Engine completed: {discovery_outputs['clusters_found']} clusters found")
                logger.info(f"Discovery Engine completed successfully: {discovery_outputs}")
                
            except Exception as e:
                error_msg = f"Discovery Engine failed: {str(e)}"
                logger.error(error_msg)
                append_progress("discovery", "failed", error_msg)
                discovery_outputs = {"error": str(e)}
        else:
            error_msg = f"ASV table not found: {asv_table_path}"
            logger.warning(error_msg)
            append_progress("discovery", "skipped", error_msg)
    elif not enable_stage2:
        logger.info('Skipping Discovery Engine: Stage 2 disabled')
        append_progress("discovery", "skipped", "Stage 2 disabled")
    elif not dada2_outputs:
        logger.info('Skipping Discovery Engine: No DADA2 output available')
        append_progress("discovery", "skipped", "No DADA2 output available")
    else:
        logger.info('Skipping Discovery Engine per config.')
        append_progress("discovery", "skipped", "Discovery Engine disabled in configuration")

    meta = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'raw_dir': str(raw_dir),
        'interim_dir': str(interim_dir),
        'processed_dir': str(processed_dir),
        'stage1': {
            'fastp': fastp_outputs,
            'dada2': dada2_outputs | {"mode": mode} if dada2_outputs else None,
        },
        'stage2': {
            'discovery': discovery_outputs
        } if enable_stage2 else None,
        'progress': progress_meta
    }
    
    # Update metadata filename to reflect both stages
    meta_filename = 'pipeline_metadata.json' if enable_stage2 else 'stage1_metadata.json'
    meta_path = processed_dir / meta_filename
    with open(meta_path, 'w') as f:
        # Convert numpy types before JSON serialization
        json_safe_meta = convert_numpy_types(meta)
        json.dump(json_safe_meta, f, indent=2)
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
