from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import rasterio


def run_structured_river_stage(
    *,
    cfg,
    report: Dict[str, Any],
    river_dir: Path,
    work_dir: Path,
    cache_dir: Path,
    logger: logging.Logger,
    build_domain_masks_fn: Callable[..., tuple[Optional[Path], Optional[Path], Optional[Path]]],
    estuary_clip_fn: Callable[..., Any],
    build_structured_helper_rasters_fn: Callable[..., Dict[str, Any]],
    write_guidance_artifacts_fn: Callable[..., None],
    build_constraint_summary_fn: Callable[..., None],
    apply_depth_metadata_fn: Callable[..., None],
    apply_elevation_metadata_fn: Callable[..., None],
) -> Path:
    river_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    channel_mask_tif, open_water_mask_tif, mainstem_mask_tif = build_domain_masks_fn(work_dir, strict=bool(getattr(cfg, "strict", False)))

    precomputed_estuary_clip = work_dir / "estuary_clip_mask.tif"
    precomputed_estuary_transition = work_dir / "estuary_transition_mask.tif"
    estuary_receipt = report.setdefault("river", {}).setdefault("execution_receipts", {})
    if precomputed_estuary_clip.exists():
        estuary_receipt["structured_estuary_clip_source"] = "precomputed"
        estuary_receipt["structured_estuary_clip_mask"] = str(precomputed_estuary_clip)
        if precomputed_estuary_transition.exists():
            report.setdefault("river", {}).setdefault("outputs", {})["estuary_transition"] = str(precomputed_estuary_transition)
        report.setdefault("river", {}).setdefault("outputs", {})["estuary_clip_mask"] = str(precomputed_estuary_clip)
        logger.info("[RIVER][STRUCTURED] Reusing precomputed estuary clip artifacts: %s", precomputed_estuary_clip)
    else:
        ocean_mask_path = None
        try:
            ocean_raw = getattr(cfg, "waffles_ocean_mask", None)
            if ocean_raw:
                ocean_candidate = Path(ocean_raw)
                if ocean_candidate.exists():
                    ocean_mask_path = ocean_candidate
        except Exception:
            ocean_mask_path = None
        estuary_receipt["structured_estuary_clip_source"] = "recomputed"
        estuary_receipt["structured_estuary_clip_mask"] = str(precomputed_estuary_clip)
        estuary_clip_fn(channel_mask_tif, cfg, ocean_mask_path=ocean_mask_path, report=report)

    with rasterio.open(channel_mask_tif) as channel_ds:
        channel_pixels_after_estuary = int((channel_ds.read(1) > 0).sum())
    estuary_receipt["structured_channel_pixels_after_estuary"] = channel_pixels_after_estuary
    if channel_pixels_after_estuary <= 0:
        raise RuntimeError("structured_river_stage_zero_channel_pixels_after_estuary_clip")

    bed_tif = cache_dir / "river_structured_helper_bed.tif"
    structured_helper_depth_tif = cache_dir / "river_structured_helper_depth.tif"
    helper_receipt = build_structured_helper_rasters_fn(
        cfg=cfg,
        river_dem=Path(cfg.river_dem),
        channel_mask_tif=Path(channel_mask_tif),
        bed_tif=bed_tif,
        depth_tif=structured_helper_depth_tif,
        logger=logger,
    )

    report.setdefault("river", {}).setdefault("structured_stage", {}).update({
        "status": "success",
        "helper_receipt": helper_receipt,
        "channel_mask_tif": str(channel_mask_tif),
        "mainstem_mask_tif": str(mainstem_mask_tif) if mainstem_mask_tif is not None else None,
    })
    report.setdefault("river", {}).setdefault("execution_receipts", {}).update({
        "river_method_executed": "structured",
        "structured_mode_active": True,
        "structured_stage_module": "river_structured_stage.run_structured_river_stage",
        "legacy_xs_inputs_expected": False,
        "legacy_xs_inputs_permitted": False,
    })
    report.setdefault("river", {}).setdefault("outputs", {}).update({
        "structured_helper_bed": str(bed_tif),
        "structured_helper_depth": str(structured_helper_depth_tif),
        "channel_mask": str(channel_mask_tif),
        "mainstem_mask": str(mainstem_mask_tif) if mainstem_mask_tif is not None else None,
    })

    out_depth = river_dir / "river_depth_terrain_patch.tif"
    out_bed = river_dir / "river_bottom_navd88_patch.tif"
    publish_mode = {}
    for src, dst in ((structured_helper_depth_tif, out_depth), (bed_tif, out_bed)):
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            publish_mode[str(dst.name)] = 'symlink'
        except OSError:
            logger.debug("river_structured_stage: symlink failed; copying %s -> %s", src, dst, exc_info=True)
            shutil.copy2(src, dst)
            publish_mode[str(dst.name)] = 'copy'

    report.setdefault("river", {}).setdefault("structured_stage", {}).setdefault("publish_mode", {}).update(publish_mode)

    try:
        apply_depth_metadata_fn(out_depth, depth_reference="terrain_surface")
    except Exception:
        logger.debug("river_structured_stage: apply_depth_metadata failed", exc_info=True)
    try:
        apply_elevation_metadata_fn(out_bed, vertical_datum="NAVD88")
    except Exception:
        logger.debug("river_structured_stage: apply_elevation_metadata failed", exc_info=True)

    report.setdefault("river", {}).setdefault("outputs", {}).update({
        "depth_terrain": str(out_depth),
        "bottom_elevation": str(out_bed),
        "bottom_elevation_internal_helper": str(out_bed),
        "depth_terrain_internal_helper": str(out_depth),
    })
    report.setdefault("river", {}).setdefault("guidance", {}).update({
        "structured_stage_bypassed_hybrid": True,
        "structured_stage_module": "river_structured_stage.run_structured_river_stage",
        "allow_absolute_bed_fallback": bool(getattr(cfg, "river_allow_absolute_bed_fallback", False)),
    })

    write_guidance_artifacts_fn(
        writer=cfg._write_river_guidance_artifacts,
        cfg=cfg,
        out_bed=Path(out_bed),
        out_depth=Path(out_depth),
        channel_mask_tif=(Path(channel_mask_tif) if channel_mask_tif is not None else None),
        river_dir=river_dir,
        report=report,
        logger=logger,
    )
    build_constraint_summary_fn(cfg, report)
    report["river"]["status"] = "success"
    report["river"]["status_family"] = "success"
    report["river"]["execution_mode"] = "structured_direct"

    primary_depth = report.get("river", {}).get("outputs", {}).get("depth_terrain")
    if primary_depth and Path(primary_depth).exists():
        logger.info("[RIVER][STRUCTURED] Success: %s", primary_depth)
        return Path(primary_depth)
    logger.info("[RIVER][STRUCTURED] Success (helper depth retained): %s", out_depth)
    return out_depth
