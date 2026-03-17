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


import logging
import sys
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger("logging_config")


class _TeeStream:
    """Mirror writes to the original stream and a log file."""

    def __init__(self, primary, mirror_fp):
        self._primary = primary
        self._mirror_fp = mirror_fp

    def write(self, data):
        self._primary.write(data)
        self._mirror_fp.write(data)
        return len(data)

    def flush(self):
        self._primary.flush()
        self._mirror_fp.flush()

    def close(self):
        try:
            self.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._primary.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._primary.fileno()

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", None)



class _RunContextFilter(logging.Filter):
    """Inject run_id/step into LogRecord (safe defaults if not configured)."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Defaults
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        if not hasattr(record, "step"):
            record.step = "-"
        try:
            # Import lazily to avoid import-order issues.
            from flight_recorder import current_run_id, current_step

            record.run_id = current_run_id()
            record.step = current_step()
        except Exception:
            log.debug("ignored", exc_info=True)  # flight_recorder not available
        return True


class _FlightRecorderHandler(logging.Handler):
    """Logging handler that mirrors records into the FlightRecorder JSONL."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from flight_recorder import FlightRecorder
            from datetime import datetime, timezone

            rec = FlightRecorder.global_instance()
            if rec is None:
                return
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds")
            rec._write(
                {
                    "ts": ts,
                    "run_id": getattr(record, "run_id", "-"),
                    "step": getattr(record, "step", "-"),
                    "kind": "log",
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                    "pathname": record.pathname,
                    "lineno": record.lineno,
                    "func": record.funcName,
                    "process": record.process,
                    "thread": record.thread,
                }
            )
        except Exception:
            # Never break the pipeline for recorder issues.
            return


# Global state to prevent double-initialization
_LOGGING_INITIALIZED: bool = False
_SCREEN_LOG_STATE = None  # dict with original streams, tee streams, and file handle


# Standard format matching original sdb_main.py
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# A more detailed format suitable for file logs.
DEFAULT_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s [run=%(run_id)s step=%(step)s]: %(message)s"
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
    
    # Inject run context into all records
    ctx_filter = _RunContextFilter()
    root.addFilter(ctx_filter)

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Add console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ctx_filter)
    root.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_path), mode='a')
        file_handler.setLevel(level)
        # Use a richer format for file logs unless caller provided a custom one.
        if log_format == DEFAULT_FORMAT:
            file_handler.setFormatter(logging.Formatter(DEFAULT_FILE_FORMAT, datefmt=date_format))
        else:
            file_handler.setFormatter(formatter)
        file_handler.addFilter(ctx_filter)
        root.addHandler(file_handler)

    # Flight recorder handler (enabled when FlightRecorder.start_global() is called)
    fr_handler = _FlightRecorderHandler()
    fr_handler.setLevel(logging.DEBUG)  # record everything; filtering happens elsewhere
    fr_handler.addFilter(ctx_filter)
    root.addHandler(fr_handler)
    
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
    
    fmt = log_format or DEFAULT_FILE_FORMAT
    handler.setFormatter(logging.Formatter(fmt))

    # Keep run context on added handlers as well
    handler.addFilter(_RunContextFilter())
    
    logging.getLogger().addHandler(handler)
    return handler


def start_flight_recorder(out_dir: Union[str, Path], run_id: str) -> Optional[Path]:
    """Start a per-run flight recorder JSONL file in <out_dir>/run_logs/."""
    try:
        from flight_recorder import FlightRecorder

        out_dir = Path(out_dir)
        log_dir = out_dir / "run_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fr_path = log_dir / f"flight_recorder_{run_id}.jsonl"
        FlightRecorder.start_global(fr_path, run_id=run_id)
        # Best-effort: write summaries at process exit (works even on crashes)
        try:
            import atexit
            from run_summary import write_run_summary_files

            def _write_summaries():
                try:
                    write_run_summary_files(out_dir, run_id=run_id, stats=None, fr_path=fr_path)
                except Exception:
                    return

            atexit.register(_write_summaries)
        except Exception:
            log.debug("atexit summary registration failed", exc_info=True)
        return fr_path
    except Exception:
        return None


def is_initialized() -> bool:
    """Check if logging has been initialized."""
    return _LOGGING_INITIALIZED


# Convenience aliases for common levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def restore_screen_log() -> None:
    """Restore original stdout/stderr and close the mirror file if installed."""
    global _SCREEN_LOG_STATE
    state = _SCREEN_LOG_STATE
    if not state:
        return

    sys.stdout = state["stdout_original"]
    sys.stderr = state["stderr_original"]

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if stream is state["stdout_tee"]:
                handler.setStream(state["stdout_original"])
            elif stream is state["stderr_tee"]:
                handler.setStream(state["stderr_original"])

    try:
        state["mirror_fp"].flush()
        state["mirror_fp"].close()
    except Exception:
        pass

    _SCREEN_LOG_STATE = None


def install_screen_log(log_file: Union[str, Path], mode: str = "a") -> Path:
    """Mirror everything written to stdout/stderr into a plain-text log file.

    This captures console logging, progress bars, warnings, and incidental prints so
    the saved file matches what the user saw on screen as closely as possible.
    Reinstalling the screen log is idempotent: any prior mirror is restored first.
    """
    global _SCREEN_LOG_STATE

    # Avoid stacking tee streams across reruns or nested entrypoints.
    restore_screen_log()

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mirror_fp = open(log_path, mode, encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = _TeeStream(original_stdout, mirror_fp)
    stderr_tee = _TeeStream(original_stderr, mirror_fp)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if stream is original_stdout:
                handler.setStream(stdout_tee)
            elif stream is original_stderr:
                handler.setStream(stderr_tee)

    _SCREEN_LOG_STATE = {
        "stdout_original": original_stdout,
        "stderr_original": original_stderr,
        "stdout_tee": stdout_tee,
        "stderr_tee": stderr_tee,
        "mirror_fp": mirror_fp,
        "path": log_path,
    }

    try:
        import atexit
        atexit.register(restore_screen_log)
    except Exception:
        pass

    return log_path
