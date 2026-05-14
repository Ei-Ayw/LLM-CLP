"""
Unified logging configuration.

Sets up structured logging to both console and file.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Union

from .io import ensure_dir


def setup_logger(
    name: str = "llm_clp",
    output_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """Configure a logger with console and optional file output.

    Args:
        name: Logger name.
        output_dir: Directory for log file. If None, only console output.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        format_str: Custom format string. Defaults to:
            "[%(asctime)s] [%(levelname)s] %(message)s"

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    if format_str is None:
        format_str = "[%(asctime)s] [%(levelname)s] %(message)s"
    formatter = logging.Formatter(format_str, datefmt="%H:%M:%S")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if output_dir is not None:
        ensure_dir(output_dir)
        log_path = Path(output_dir) / "run.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger