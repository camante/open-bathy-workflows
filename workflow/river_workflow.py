"""Public active river workflow façade.

Bundle A makes the canonical parent -> AOI export workflow the named active
river workflow while preserving the existing implementation underneath.  The
implementation still uses some historical module names internally; later
bundles can move those internals stage-by-stage without changing this public
entry point.
"""

from __future__ import annotations

from typing import Any

from river_runner import run_river_workflow_direct


ACTIVE_RIVER_WORKFLOW_NAME = "canonical_parent_aoi_export"


def run_river_workflow(cfg: Any, report: dict[str, Any], **kwargs: Any) -> Any:
    """Run the single active river workflow.

    This is the stable public name for the active path.  It delegates to the
    existing runner during Bundle A so behavior remains unchanged while logs,
    imports, and future refactors can converge on ``river_workflow`` naming.
    """
    return run_river_workflow_direct(cfg, report, **kwargs)


__all__ = ["ACTIVE_RIVER_WORKFLOW_NAME", "run_river_workflow"]
