"""Run initialization helpers split out of bathy_main."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def initialize_run_args(args: Any, *, logger=None) -> str:
    run_id = getattr(args, "run_id", None)
    if not run_id:
        run_id = "bathy_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            setattr(args, "run_id", run_id)
        except Exception:
            pass
    if logger is not None:
        logger.info("[RUN] run_id=%s", run_id)
    return str(run_id)


def resolve_authoritative_base_args(args: Any) -> dict[str, Any]:
    return {
        "authoritative_base": getattr(args, "authoritative_base", None),
        "export_authoritative_base": getattr(args, "export_authoritative_base", None),
        "authoritative_base_tile_index_url": getattr(args, "authoritative_base_tile_index_url", None),
        "authoritative_base_spatial_meta_url": getattr(args, "authoritative_base_spatial_meta_url", None),
    }


__all__ = ["initialize_run_args", "resolve_authoritative_base_args"]
