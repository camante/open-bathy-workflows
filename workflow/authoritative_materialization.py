"""Coordinator for authoritative-base materialization and AOI-keyed cache reuse."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_authoritative_base(cfg: Any, args: argparse.Namespace, *, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    """Resolve or auto-build authoritative_base for the processing AOI.

    Behavior:
      - an explicit existing path wins
      - an explicit missing path is preserved when auto materialization is disabled
      - otherwise, auto-materialize from NOAA CUDEM tile index + spatial metadata
        into <cache_root>/authoritative_base/<cache_key>/ with reuse on repeat AOIs
    """
    log = logger or logging.getLogger(__name__)
    auth_path = getattr(cfg, "authoritative_base", None)
    auto = bool(getattr(cfg, "authoritative_base_auto", False))

    if auth_path is not None:
        auth_path = Path(auth_path)
        if auth_path.exists():
            cfg.authoritative_base = auth_path
            setattr(args, "_authoritative_base_auto_report", {
                "mode": "explicit_path",
                "authoritative_base": str(auth_path),
                "cache_hit": None,
            })
            return auth_path
        if not auto:
            log.warning("[AUTHORITATIVE] Provided authoritative_base does not exist: %s", auth_path)
            cfg.authoritative_base = auth_path
            return auth_path
        log.info("[AUTHORITATIVE] Explicit authoritative_base path not found; falling back to auto-materialization for AOI.")

    if not auto:
        return auth_path

    try:
        from cudem_authoritative import materialize_authoritative_base_for_aoi
    except ImportError as exc:
        log.error("[AUTHORITATIVE] Failed to import cudem_authoritative auto-builder: %s", exc, exc_info=True)
        raise

    build_info = materialize_authoritative_base_for_aoi(
        aoi=str(getattr(cfg, "aoi", "") or ""),
        cache_root=Path(cfg.cache_root),
        tile_index_url=str(getattr(cfg, "authoritative_base_tile_index_url", "") or ""),
        spatial_meta_url=str(getattr(cfg, "authoritative_base_spatial_meta_url", "") or ""),
        missing_meta_policy=str(getattr(cfg, "authoritative_base_missing_meta_policy", "skip") or "skip"),
        tile_url_field=getattr(cfg, "authoritative_base_tile_url_field", None),
        force_rebuild=bool(getattr(cfg, "authoritative_base_force_rebuild", False)),
        logger=log,
    )
    cfg.authoritative_base = Path(build_info["authoritative_base"]).resolve()
    setattr(args, "_authoritative_base_auto_report", build_info)
    log.info(
        "[AUTHORITATIVE] %s authoritative_base: %s",
        "Reused cached" if bool(build_info.get("cache_hit")) else "Materialized",
        cfg.authoritative_base,
    )
    return cfg.authoritative_base
