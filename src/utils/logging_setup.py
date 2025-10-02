from loguru import logger
from pathlib import Path
import sys

def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(Path(log_dir) / "pipeline.log", level=level, rotation="5 MB", retention=10)
    return logger
