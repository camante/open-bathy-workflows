"""errors_scientific.py — Scientific fallback classification.

Every exception handler in a scientific code path should declare its intent
using one of three fallback classes:

    SAFE        — Missing optional output (e.g., plot, debug log).
                  Pipeline continues, no scientific impact.

    DEGRADED    — Scientifically weaker output (e.g., fell back from
                  spatial validation to random split, used default Kd,
                  skipped a correction step). Output is still usable
                  but must be labeled in provenance.

    INVALID     — Cannot produce reliable output (e.g., training collapsed,
                  required datum transform failed, authoritative data
                  missing where required). Pipeline should stop or
                  suppress the output.

Usage::

    from errors_scientific import FallbackClass, record_fallback, FallbackRegistry

    try:
        run_datum_transform(...)
    except (FileNotFoundError, RuntimeError) as e:
        record_fallback(
            registry, "datum_transform", FallbackClass.DEGRADED,
            stage="post_processing", error=e,
            detail="VDatum grid not found; output remains in MSL",
        )

After a run, ``registry.summary()`` produces a machine-readable dict
suitable for embedding in ``run_report.json``.

This module is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class FallbackClass(enum.Enum):
    """Classification of a scientific fallback."""
    SAFE = "safe"
    DEGRADED = "degraded"
    INVALID = "invalid"


@dataclass
class FallbackEvent:
    """A single recorded fallback event."""
    name: str
    fallback_class: FallbackClass
    stage: str = ""
    detail: str = ""
    error_type: str = ""
    error_msg: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class": self.fallback_class.value,
            "stage": self.stage,
            "detail": self.detail,
            "error_type": self.error_type,
            "error_msg": self.error_msg,
            "timestamp": self.timestamp,
        }


class FallbackRegistry:
    """Accumulates fallback events during a pipeline run.

    Provides a summary suitable for ``run_report.json`` and determines
    the overall run validity.
    """

    def __init__(self) -> None:
        self.events: List[FallbackEvent] = []

    def record(self, name: str, cls: FallbackClass, *,
               stage: str = "", detail: str = "",
               error: Optional[Exception] = None) -> None:
        """Record a fallback event."""
        evt = FallbackEvent(
            name=name,
            fallback_class=cls,
            stage=stage,
            detail=detail,
            error_type=type(error).__name__ if error else "",
            error_msg=str(error)[:500] if error else "",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.events.append(evt)

        # Log at appropriate level
        if cls == FallbackClass.INVALID:
            log.error("[FALLBACK:INVALID] %s — %s: %s", stage, name, detail)
        elif cls == FallbackClass.DEGRADED:
            log.warning("[FALLBACK:DEGRADED] %s — %s: %s", stage, name, detail)
        else:
            log.info("[FALLBACK:SAFE] %s — %s: %s", stage, name, detail)

    @property
    def has_invalid(self) -> bool:
        return any(e.fallback_class == FallbackClass.INVALID for e in self.events)

    @property
    def has_degraded(self) -> bool:
        return any(e.fallback_class == FallbackClass.DEGRADED for e in self.events)

    @property
    def run_validity(self) -> str:
        """Overall run validity: 'valid', 'degraded', or 'invalid'."""
        if self.has_invalid:
            return "invalid"
        if self.has_degraded:
            return "degraded"
        return "valid"

    def summary(self) -> Dict[str, Any]:
        """Machine-readable summary for run_report.json."""
        return {
            "run_validity": self.run_validity,
            "n_events": len(self.events),
            "n_safe": sum(1 for e in self.events if e.fallback_class == FallbackClass.SAFE),
            "n_degraded": sum(1 for e in self.events if e.fallback_class == FallbackClass.DEGRADED),
            "n_invalid": sum(1 for e in self.events if e.fallback_class == FallbackClass.INVALID),
            "events": [e.to_dict() for e in self.events],
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def record_fallback(
    registry: Optional[FallbackRegistry],
    name: str,
    cls: FallbackClass,
    **kwargs,
) -> None:
    """Record a fallback if a registry is available, otherwise just log."""
    if registry is not None:
        registry.record(name, cls, **kwargs)
    else:
        # No registry — still log
        level = {
            FallbackClass.SAFE: logging.INFO,
            FallbackClass.DEGRADED: logging.WARNING,
            FallbackClass.INVALID: logging.ERROR,
        }.get(cls, logging.WARNING)
        log.log(level, "[FALLBACK:%s] %s — %s: %s",
                cls.value.upper(), kwargs.get("stage", ""),
                name, kwargs.get("detail", ""))
