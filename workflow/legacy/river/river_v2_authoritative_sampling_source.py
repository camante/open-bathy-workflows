from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from core.json_io import write_json
from legacy.river.river_v2_context import RiverV2Context


def _rewrite_local_artifact_from_source(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        rel = os.path.relpath(source, destination.parent)
        destination.symlink_to(rel)
    except Exception:
        shutil.copy2(source, destination)


def materialize_authoritative_sampling_source_for_solve_aoi(
    ctx: RiverV2Context,
    *,
    solve_aoi: str,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Materialize the authoritative raster source used for centerline bed sampling.

    This is intentionally separate from the authoritative lock base. When the solve
    domain expands downstream, the sampling source must expand too so that the
    support proof and the bed-sampling raster refer to the same spatial footprint.
    """
    log = logger or logging.getLogger(__name__)
    cfg = ctx.cfg
    out_path = ctx.paths.authoritative_sampling_source_raster
    summary_path = ctx.paths.authoritative_sampling_source_summary
    summary: Dict[str, Any] = {
        "requested_solve_aoi": str(solve_aoi),
        "cfg_aoi": str(getattr(cfg, "aoi", "") or ""),
        "status": "unresolved",
        "mode": None,
        "source_raster": None,
    }

    explicit_auth = Path(ctx.authoritative_base_path) if ctx.authoritative_base_path is not None else None
    auto_enabled = bool(getattr(cfg, "authoritative_base_auto", False))

    if auto_enabled:
        from cudem_authoritative import materialize_authoritative_base_for_aoi

        info = materialize_authoritative_base_for_aoi(
            aoi=str(solve_aoi),
            cache_root=Path(cfg.cache_root),
            tile_index_url=str(getattr(cfg, "authoritative_base_tile_index_url", "") or ""),
            spatial_meta_url=str(getattr(cfg, "authoritative_base_spatial_meta_url", "") or ""),
            missing_meta_policy=str(getattr(cfg, "authoritative_base_missing_meta_policy", "skip") or "skip"),
            tile_url_field=getattr(cfg, "authoritative_base_tile_url_field", None),
            force_rebuild=bool(getattr(cfg, "authoritative_base_force_rebuild", False)),
            logger=log,
        )
        source = Path(info["authoritative_base"]).resolve()
        _rewrite_local_artifact_from_source(source=source, destination=out_path)
        log.info("[RIVER][V2][AUTHORITATIVE] Materialized expanded sampling source for solve AOI=%s: %s", solve_aoi, source)
        support_coverage = info.get("authoritative_support_coverage")
        ctx.solve_aoi_authoritative_base_path = source
        ctx.solve_aoi_authoritative_support_coverage_path = Path(support_coverage) if support_coverage else None
        summary.update(
            {
                "status": "success",
                "mode": "auto_cudem_expanded_solve_aoi",
                "source_raster": str(source),
                "local_artifact": str(out_path),
                "cache_hit": bool(info.get("cache_hit")),
                "cache_dir": info.get("cache_dir"),
                "support_coverage": support_coverage,
            }
        )
        write_json(summary_path, summary)
        ctx.authoritative_sampling_source_raster_path = out_path if out_path.exists() else source
        return ctx.authoritative_sampling_source_raster_path

    if explicit_auth is not None and explicit_auth.exists():
        _rewrite_local_artifact_from_source(source=explicit_auth, destination=out_path)
        log.info("[RIVER][V2][AUTHORITATIVE] Reusing explicit authoritative base as sampling source for solve AOI=%s: %s", solve_aoi, explicit_auth)
        ctx.solve_aoi_authoritative_base_path = explicit_auth
        ctx.solve_aoi_authoritative_support_coverage_path = None
        summary.update(
            {
                "status": "success",
                "mode": "explicit_authoritative_base_reused",
                "source_raster": str(explicit_auth),
                "local_artifact": str(out_path),
                "warning": "authoritative_base_auto_disabled_sampling_source_not_expanded",
            }
        )
        write_json(summary_path, summary)
        ctx.authoritative_sampling_source_raster_path = out_path if out_path.exists() else explicit_auth
        return ctx.authoritative_sampling_source_raster_path

    summary.update(
        {
            "status": "missing",
            "mode": "no_sampling_source_available",
            "warning": "authoritative_sampling_source_unavailable",
        }
    )
    write_json(summary_path, summary)
    return None


__all__ = ["materialize_authoritative_sampling_source_for_solve_aoi"]
