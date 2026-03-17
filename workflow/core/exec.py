"""Subprocess execution helpers.

Phase-1 refactor: extract subprocess plumbing from bathy_main.py.

Design goals:
  - list-argv execution only (no shell=True)
  - optional streaming to logger
  - optional stdout/stderr log files
  - return (rc, stdout_tail, stderr_tail) to preserve current contract
"""

from __future__ import annotations

import os
import shlex
import subprocess
import logging
import re
from collections import deque
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from .paths import ensure_dir


def run_command(
    cmd: Any,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    prefix: str = "",
    stream_stdout: bool = True,
    stream_stderr: bool = True,
    stdout_log_path: Optional[Path] = None,
    stderr_log_path: Optional[Path] = None,
    max_lines: int = 8000,
    tail_chars: int = 16000,
) -> Tuple[int, str, str]:
    """Run a command with optional streaming and log capture.

    Args mirror the original bathy_main.run_command for drop-in compatibility.
    """
    log = logging.getLogger(__name__)
    env = env or os.environ.copy()

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    cmd = [str(c) for c in cmd]

    stdout_lines = deque(maxlen=max_lines)
    stderr_lines = deque(maxlen=max_lines)
    stdout_fh = None
    stderr_fh = None
    try:
        if stdout_log_path is not None:
            ensure_dir(Path(stdout_log_path).parent)
            stdout_fh = open(stdout_log_path, "w", buffering=1, encoding="utf-8")
        if stderr_log_path is not None:
            ensure_dir(Path(stderr_log_path).parent)
            stderr_fh = open(stderr_log_path, "w", buffering=1, encoding="utf-8")
    except OSError:
        stdout_fh = None
        stderr_fh = None

    cmd_str = " ".join(str(c) for c in cmd)
    log.debug("Executing: %s", cmd_str)

    # Flight recorder hooks (best effort)
    _fr = None
    try:
        from flight_recorder import FlightRecorder

        _fr = FlightRecorder.global_instance()
    except ImportError:
        _fr = None

    if _fr is not None:
        _fr.record_event(
            "subprocess_start",
            cmd=cmd,
            cmd_str=cmd_str,
            cwd=str(cwd) if cwd else None,
        )

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )


    def _log_stderr_line(line: str, pfx: str):
        """Classify stderr lines to reduce log noise (progress/info on stderr)."""
        s = line.rstrip()
        # Many subprocesses write structured logs to stderr (including level tags like [INFO]).
        # Use a regex so we don't depend on exact spacing.
        try:
            if re.search(r"\[\s*ERROR\s*\]", s):
                log.error("%s%s", pfx, s) if pfx else log.error("%s", s)
            elif re.search(r"\[\s*WARNING\s*\]", s):
                log.warning("%s%s", pfx, s) if pfx else log.warning("%s", s)
            elif re.search(r"\[\s*INFO\s*\]", s):
                log.info("%s%s", pfx, s) if pfx else log.info("%s", s)
            else:
                log.warning("%s%s", pfx, s) if pfx else log.warning("%s", s)
        except re.error:
            log.warning("%s%s", pfx, s) if pfx else log.warning("%s", s)

    def _pump(stream, sink, log_fn, pfx: str, enabled: bool, fh=None):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                sink.append(line)
                if fh is not None:
                    try:
                        fh.write(line)
                    except OSError:
                        # Best effort; never fail the pipeline due to optional logging.
                        log.debug("Optional log write failed", exc_info=True)
                if enabled:
                    if log_fn is log.warning and "[stderr]" in (pfx or ""):
                        _log_stderr_line(line, pfx)
                    else:
                        log_fn(f"{pfx}{line.rstrip()}" if pfx else line.rstrip())
        finally:
            try:
                stream.close()
            except OSError:
                log.debug("ignored", exc_info=True)

    threads: List[threading.Thread] = []
    try:
        if proc.stdout is not None:
            threads.append(
                threading.Thread(
                    target=_pump,
                    args=(proc.stdout, stdout_lines, log.info, prefix, stream_stdout, stdout_fh),
                    daemon=True,
                )
            )
        if proc.stderr is not None:
            err_pfx = f"{prefix}[stderr] " if prefix else "[stderr] "
            threads.append(
                threading.Thread(
                    target=_pump,
                    args=(proc.stderr, stderr_lines, log.warning, err_pfx, stream_stderr, stderr_fh),
                    daemon=True,
                )
            )

        for t in threads:
            t.start()

        rc = proc.wait()

        for t in threads:
            t.join(timeout=30.0)  # wait up to 30 s for I/O threads to drain

    finally:
        try:
            if stdout_fh:
                stdout_fh.close()
            if stderr_fh:
                stderr_fh.close()
        except OSError:
            log.debug("ignored", exc_info=True)

    # stdout_lines/stderr_lines include newline characters; match original behavior
    stdout_tail = "".join(stdout_lines)[-tail_chars:]
    stderr_tail = "".join(stderr_lines)[-tail_chars:]

    if _fr is not None:
        _fr.record_event(
            "subprocess_end",
            cmd=cmd,
            cmd_str=cmd_str,
            returncode=rc,
        )

    return rc, stdout_tail, stderr_tail


def run_command_stdout_to_file(
    cmd: Any,
    out_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    prefix: str = "",
    stderr_log_path: Optional[Path] = None,
    tail_chars: int = 16000,
) -> Tuple[int, str]:
    """Run a command and write stdout to out_path (atomic temp file).

    Returns (returncode, stderr_tail).

    This is used for tools like `dlim` that produce data on stdout.
    """
    log = logging.getLogger(__name__)
    env = env or os.environ.copy()

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    cmd = [str(c) for c in cmd]

    ensure_dir(out_path.parent)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            log.debug("ignored", exc_info=True)

    stderr_fh = None
    if stderr_log_path is not None:
        try:
            ensure_dir(Path(stderr_log_path).parent)
            stderr_fh = open(stderr_log_path, "w", buffering=1, encoding="utf-8")
        except OSError:
            stderr_fh = None

    cmd_str = " ".join(str(c) for c in cmd)
    log.debug("Executing: %s", cmd_str)

    proc = None
    rc = -1
    stderr_tail = ""
    try:
        with open(tmp_path, "w", encoding="utf-8") as f_out:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=f_out,
                stderr=subprocess.PIPE,
                text=True,
            )
            stderr = proc.stderr.read() if proc.stderr else ""
            stderr_tail = (stderr or "")[-tail_chars:]
            if stderr_fh and stderr:
                stderr_fh.write(stderr)
            rc = proc.wait()
    finally:
        try:
            if stderr_fh:
                stderr_fh.close()
        except OSError:
            log.debug("ignored", exc_info=True)

    # Caller decides whether tmp_path is valid enough to promote.
    return rc, stderr_tail
