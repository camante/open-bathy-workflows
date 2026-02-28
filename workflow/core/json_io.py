"""JSON read/write helpers.

Lifted from bathy_main.py (Phase-2 refactor). No behavior changes intended.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from core.paths import ensure_dir


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON to disk (and emit a flight-recorder breadcrumb when available)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

    # Flight recorder breadcrumb (best effort)
    try:
        from flight_recorder import emit_artifact_written
        emit_artifact_written(path, kind="json", role="report_or_metadata")
    except Exception as e:
        logging.getLogger(__name__).debug("Optional flight-recorder emit failed: %s", e)
