"""
Logging configuration for CodeMix Toolkit.
This module provides a centralized logging setup that can be used throughout the toolkit.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import config


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: Name of the logger
        log_file: Optional path to log file. If None, logs will only go to console
        level: Optional logging level. If None, uses LOG_LEVEL from config
        format_string: Optional format string for log messages

    Returns:
        logging.Logger: Configured logger instance
    """
    # Get logger
    logger = logging.getLogger(name)

    # Set level from config if not specified
    if level is None:
        level = config.LOG_LEVEL

    logger.setLevel(getattr(logging, level.upper()))

    # Create formatters
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file is not None:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the toolkit's default configuration.

    Args:
        name: Name of the logger (typically __name__ of the calling module)

    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logger(name)


# Create default toolkit logger
toolkit_logger = get_logger("codemixtoolkit")
