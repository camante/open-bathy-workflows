"""Reporting wrappers for river guidance products.

The active canonical workflow registers its products through ``river_runner`` and
``active_pipeline``. These functions remain for older final-route callers and do
not mutate science rasters.
"""

from __future__ import annotations
from typing import Any


def apply_guidance_controls_with_reporting(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = kwargs.get("report")
    if isinstance(report, dict):
        report.setdefault("river_guidance", {})["controls"] = "not_applied_by_legacy_wrapper"
    return {"status": "not_applied_by_legacy_wrapper"}


def write_guidance_artifacts_with_reporting(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = kwargs.get("report")
    if isinstance(report, dict):
        report.setdefault("river_guidance", {})["artifacts"] = "registered_by_active_pipeline"
    return {"status": "registered_by_active_pipeline"}

__all__ = ["apply_guidance_controls_with_reporting", "write_guidance_artifacts_with_reporting"]
