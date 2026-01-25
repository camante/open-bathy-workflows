#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
logging_config.py - Centralized Logging Configuration for SDB Pipeline

This module provides a single source of truth for logging configuration
across all pipeline modules. Import and call setup_logging() from your
entrypoint BEFORE importing other modules.

Usage in entrypoints (bathy_main.py, sdb_main.py):
--------------------------------------------------
    # At the VERY TOP of the file, before other imports:
    from logging_config import setup_logging
    setup_logging()
    
    # Now import other modules...
    import atl
    import fusion
    # etc.

Why this matters:
-----------------
Python's logging.basicConfig() only configures the root logger ONCE.
If multiple modules call basicConfig() at import time, the first one wins.
This leads to inconsistent log formats depending on import order.

Solution:
---------
1. Call setup_logging() once in entrypoint before other imports
2. All other modules should use get_logger() instead of basicConfig()
3. Module-level logging is configured to propagate to root by default
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


# Global state to prevent double-initialization
_LOGGING_INITIALIZED: bool = False


# Standard format matching original sdb_main.py
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = None  # Use logging default


def setup_logging(
    level: int = logging.INFO,
    log_format: str = DEFAULT_FORMAT,
    date_format: Optional[str] = DEFAULT_DATE_FORMAT,
    log_file: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> None:
    """
    Configure the root logger for the entire pipeline.
    
    Call this ONCE at the top of your entrypoint script, BEFORE importing
    other pipeline modules.
    
    Args:
        level: Logging level (default: logging.INFO)
        log_format: Log message format string
        date_format: Date format string (None for logging default)
        log_file: Optional path to write logs to file
        force: If True, reconfigure even if already initialized
    
    Example:
        # In bathy_main.py, at the very top:
        from logging_config import setup_logging
        setup_logging(level=logging.DEBUG, log_file="output/run.log")
        
        # Now import other modules
        import atl
        import fusion
    """
    global _LOGGING_INITIALIZED
    
    if _LOGGING_INITIALIZED and not force:
        return
    
    # Get the root logger
    root = logging.getLogger()
    
    # Remove any existing handlers to ensure clean state
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Set level on root logger
    root.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Add console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(str(log_path), mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    
    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.
    
    This is the preferred way to get a logger in pipeline modules.
    It returns a named logger that inherits from the root logger
    configured by setup_logging().
    
    Args:
        name: Logger name (typically __name__ or module name)
    
    Returns:
        logging.Logger instance
    
    Example:
        # In atl.py:
        from logging_config import get_logger
        log = get_logger("sdb.atl")
        
        # Use normally:
        log.info("Processing ATL data...")
    """
    return logging.getLogger(name)


def add_file_handler(
    log_file: Union[str, Path],
    level: int = logging.INFO,
    log_format: Optional[str] = None,
) -> logging.FileHandler:
    """
    Add a file handler to the root logger.
    
    Useful for adding per-run log files after initial setup.
    
    Args:
        log_file: Path to log file
        level: Logging level for this handler
        log_format: Optional custom format (uses default if None)
    
    Returns:
        The created FileHandler (can be removed later if needed)
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    handler = logging.FileHandler(str(log_path), mode='a')
    handler.setLevel(level)
    
    fmt = log_format or DEFAULT_FORMAT
    handler.setFormatter(logging.Formatter(fmt))
    
    logging.getLogger().addHandler(handler)
    return handler


def is_initialized() -> bool:
    """Check if logging has been initialized."""
    return _LOGGING_INITIALIZED


# Convenience aliases for common levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
