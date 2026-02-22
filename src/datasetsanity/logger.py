import os
import logging
import sys
from typing import Optional

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout),
    ],
)

_logger = logging.getLogger("Package_Name")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the configured package logger (or a child logger)."""
    return _logger if name is None else _logger.getChild(name)