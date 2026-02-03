"""
Logging configuration for Shorts Bot.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColorFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Get or create a logger with consistent formatting.

    Args:
        name: Logger name (usually module name)
        level: Logging level
        log_file: Optional file path to write logs
        use_colors: Whether to use colored output in terminal

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors and sys.stdout.isatty():
        console_format = ColorFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )

    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def setup_global_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None
) -> Path:
    """
    Setup global logging configuration.

    Args:
        level: Global logging level
        log_dir: Directory for log files

    Returns:
        Path to the log file
    """
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"shorts_bot_{timestamp}.log"
    else:
        log_file = None

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColorFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    root_logger.addHandler(console_handler)

    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(file_handler)

    return log_file


class ProgressLogger:
    """Helper class for logging progress in long-running operations."""

    def __init__(self, logger: logging.Logger, total: int, description: str = "Processing"):
        self.logger = logger
        self.total = total
        self.current = 0
        self.description = description

    def update(self, n: int = 1, message: str = ""):
        """Update progress by n steps."""
        self.current += n
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        msg = f"{self.description}: {self.current}/{self.total} ({percent:.1f}%)"
        if message:
            msg += f" - {message}"
        self.logger.info(msg)

    def complete(self, message: str = ""):
        """Mark operation as complete."""
        msg = f"{self.description}: Complete"
        if message:
            msg += f" - {message}"
        self.logger.info(msg)
