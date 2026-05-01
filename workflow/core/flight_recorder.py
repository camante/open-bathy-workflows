#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flight_recorder.py - Structured "flight recorder" logging for every run.

Goal
----
Provide a durable, machine-readable run record that captures:
  - every Python log record (level/name/message + location)
  - explicit pipeline events (step start/stop, artifacts, metrics)
  - uncaught exceptions

The recorder writes JSON Lines (one JSON object per line) so it can be streamed,
tailed, and post-processed safely even if the process crashes.

Design constraints
------------------
* No third-party deps.
* Safe to enable by default (best effort; never crash the pipeline because
  the recorder couldn't write).
* Context propagation via contextvars (run_id + current step).
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_cv_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")
_cv_step: contextvars.ContextVar[str] = contextvars.ContextVar("step", default="-")


log = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class FlightRecorderConfig:
    path: Path
    run_id: str
    flush_every: int = 1


class FlightRecorder:
    """Singleton-style recorder."""

    _global: Optional["FlightRecorder"] = None

    def __init__(self, cfg: FlightRecorderConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._fh = None  # type: ignore
        self._events_written = 0
        self._dropped = 0

    @classmethod
    def global_instance(cls) -> Optional["FlightRecorder"]:
        return cls._global

    @classmethod
    def start_global(cls, path: Path, run_id: str) -> "FlightRecorder":
        """Start (or replace) the global recorder."""
        rec = FlightRecorder(FlightRecorderConfig(path=path, run_id=run_id))
        rec.start()
        cls._global = rec
        return rec

    @classmethod
    def stop_global(cls) -> None:
        rec = cls._global
        if rec is not None:
            rec.stop()
        cls._global = None

    def start(self) -> None:
        try:
            self.cfg.path.parent.mkdir(parents=True, exist_ok=True)
            # line-buffered text; each write is a full JSON line
            self._fh = open(self.cfg.path, "a", encoding="utf-8", buffering=1)
            _cv_run_id.set(self.cfg.run_id)
            self.record_event("run_start", pid=os.getpid(), argv=sys.argv)
        except OSError:
            # Best effort: do not raise.
            self._fh = None
        # Ensure we attempt to stop/flush
        atexit.register(self.stop)

        # Capture uncaught exceptions into the recorder (best effort)
        try:
            prev_hook = sys.excepthook

            def _hook(exc_type, exc, tb):
                try:
                    self.record_event(
                        "uncaught_exception",
                        exc_type=getattr(exc_type, "__name__", str(exc_type)),
                        message=str(exc),
                        traceback="".join(traceback.format_exception(exc_type, exc, tb)),
                    )
                except (OSError, TypeError, ValueError):
                    log.debug("ignored", exc_info=True)  # don't let recorder failure mask original exception
                return prev_hook(exc_type, exc, tb)

            sys.excepthook = _hook  # type: ignore
        except (AttributeError, RuntimeError):
            log.debug("excepthook installation failed", exc_info=True)

    def stop(self) -> None:
        # Best effort stop
        try:
            self.record_event(
                "run_end",
                events_written=self._events_written,
                dropped=self._dropped,
            )
        except (OSError, TypeError, ValueError):
            log.debug("ignored", exc_info=True)  # don't let stop-record failure prevent file close
        try:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
        except OSError:
            log.debug("ignored", exc_info=True)  # close error
        self._fh = None

    def record_event(self, event: str, **fields: Any) -> None:
        obj: Dict[str, Any] = {
            "ts": _utc_now_iso(),
            "run_id": _cv_run_id.get(),
            "step": _cv_step.get(),
            "kind": "event",
            "event": event,
        }
        obj.update(fields)
        self._write(obj)

    def record_exception(self, where: str, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.record_event(
            "exception",
            where=where,
            exc_type=type(exc).__name__,
            message=str(exc),
            traceback=tb,
        )

    def _write(self, obj: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        try:
            line = json.dumps(obj, ensure_ascii=False, sort_keys=True)
            with self._lock:
                self._fh.write(line + "\n")
                self._events_written += 1
                if self.cfg.flush_every and (self._events_written % self.cfg.flush_every == 0):
                    self._fh.flush()
        except (OSError, TypeError, ValueError):
            self._dropped += 1


@contextlib.contextmanager
def step(name: str, **fields: Any):
    """Context manager to tag records with a pipeline step."""
    rec = FlightRecorder.global_instance()
    token = _cv_step.set(name)
    if rec is not None:
        rec.record_event("step_start", step=name, **fields)
    t0 = time.time()
    try:
        yield
        ok = True
    except Exception as e:
        log.debug("step: suppressed exception", exc_info=True)
        ok = False
        if rec is not None:
            rec.record_exception(where=f"step:{name}", exc=e)
        raise
    finally:
        dt = time.time() - t0
        if rec is not None:
            rec.record_event("step_end", step=name, ok=ok, elapsed_s=dt)
        _cv_step.reset(token)


def current_run_id() -> str:
    return _cv_run_id.get()


def current_step() -> str:
    return _cv_step.get()


def current_flight_path() -> str:
    # Return the current flight recorder path (or empty string).
    rec = FlightRecorder.global_instance()
    try:
        return str(rec.cfg.path) if rec is not None else ''
    except AttributeError:
        return ''


# Convenience API ------------------------------------------------------------

def emit_event(event_type: str, **payload: Any) -> None:
    """Record a structured event to the global flight recorder.

    Best-effort only: never raises.
    """
    rec = FlightRecorder.global_instance()
    if rec is None:
        return
    try:
        rec.record_event(event_type, **payload)
    except (OSError, TypeError, ValueError):
        return


def emit_artifact_written(path: Any, *, kind: str = "artifact", role: str = "") -> None:
    """Record that an artifact was written.

    Parameters
    ----------
    path: Any
        Path-like or string.
    kind: str
        Artifact kind (e.g., raster, json, log, plot).
    role: str
        Optional semantic role (e.g., final_depth, river_mask, report).
    """
    try:
        p = str(path)
    except (TypeError, ValueError):
        p = ""
    if not p:
        return
    emit_event(
        "artifact_written",
        artifact_path=p,
        artifact_kind=str(kind),
        artifact_role=str(role),
    )
