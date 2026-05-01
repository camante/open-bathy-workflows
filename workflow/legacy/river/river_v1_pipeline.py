from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from legacy.river.river_v1_support import build_v1_support_products, build_v1_bank_guidance_products
from legacy.river.river_v1_backbone import build_v1_backbone_products
from legacy.river.river_v1_surface import build_v1_surface_products


def run_river_v1_stage(
    *,
    cfg,
    report: Dict[str, Any],
    river_dir: Path,
    work_dir: Path,
    network_gpkg: Path,
    build_domain_masks_fn: Callable[[Path], tuple[Path, Optional[Path], Optional[Path]]],
    load_support_points_fn: Callable[[Any], Any],
    logger,
) -> Path:
    river_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    channel_mask_tif, open_water_mask_tif, mainstem_mask_tif = build_domain_masks_fn(work_dir, strict=bool(getattr(cfg, "strict", False)))
    if channel_mask_tif is None or not Path(channel_mask_tif).exists():
        raise RuntimeError("river_v1_missing_channel_mask")

    support_summary = build_v1_support_products(
        cfg=cfg,
        network_gpkg=Path(network_gpkg),
        river_dem=Path(cfg.river_dem),
        channel_mask_tif=Path(channel_mask_tif),
        river_dir=Path(river_dir),
        load_support_points_fn=load_support_points_fn,
        logger=logger,
    )
    backbone_summary = build_v1_backbone_products(
        cfg=cfg,
        network_gpkg=Path(network_gpkg),
        river_dem=Path(cfg.river_dem),
        channel_mask_tif=Path(channel_mask_tif),
        river_dir=Path(river_dir),
        support_points_path=Path(support_summary["support_points_path"]),
        logger=logger,
    )
    bank_summary = build_v1_bank_guidance_products(
        cfg=cfg,
        river_dem=Path(cfg.river_dem),
        channel_mask_tif=Path(channel_mask_tif),
        centerline_points_path=Path(backbone_summary["centerline_points_path"]),
        river_dir=Path(river_dir),
        logger=logger,
    )
    surface_summary = build_v1_surface_products(
        cfg=cfg,
        river_dem=Path(cfg.river_dem),
        channel_mask_tif=Path(channel_mask_tif),
        river_dir=Path(river_dir),
        support_points_path=Path(support_summary["support_points_path"]),
        centerline_points_path=Path(backbone_summary["centerline_points_path"]),
        logger=logger,
    )

    river_report = report.setdefault("river", {})
    river_report["status"] = "success"
    river_report["status_family"] = "success"
    river_report["execution_mode"] = "v1_minimal_surface"
    river_report.setdefault("steps", {})["v1_support"] = {
        "status": "success",
        "channel_mask_tif": str(channel_mask_tif),
        "open_water_mask_tif": str(open_water_mask_tif) if open_water_mask_tif is not None else None,
        "mainstem_mask_tif": str(mainstem_mask_tif) if mainstem_mask_tif is not None else None,
    }
    river_report.setdefault("steps", {})["v1_backbone"] = {"status": "success"}
    river_report.setdefault("steps", {})["v1_surface"] = {
        "status": "success",
        "river_primary_surface_path": str(surface_summary["river_primary_surface_path"]),
    }
    river_report.setdefault("outputs", {}).update({
        "channel_mask": str(channel_mask_tif),
        "mainstem_mask": str(mainstem_mask_tif) if mainstem_mask_tif is not None else None,
        "river_support_points": str(support_summary["support_points_path"]),
        "river_support_summary": str(support_summary["support_summary_path"]),
        "bank_influence": str(bank_summary["bank_influence_path"]),
        "bank_wse_edge_guidance": str(bank_summary["bank_elevation_path"]),
        "bank_elevation_xs": str(bank_summary["bank_elevation_path"]),
        "river_bank_guidance_summary": str(bank_summary["bank_summary_path"]),
        "river_centerline_points": str(backbone_summary["centerline_points_path"]),
        "river_backbone_summary": str(backbone_summary["backbone_summary_path"]),
        "river_primary_surface": str(surface_summary["river_primary_surface_path"]),
        "primary_river_guidance_surface": str(surface_summary["river_primary_surface_path"]),
        "river_surface_summary": str(surface_summary["river_surface_summary_path"]),
        "river_primary_surface_contract": str(surface_summary["river_primary_surface_contract_path"]),
    })
    river_report.setdefault("execution_receipts", {}).update({
        "river_method_executed": "v1",
        "v1_support_backbone_only": False,
        "v1_surface_stage_present": True,
        "v1_conditioning_stage_present": False,
        "bank_guidance_source": bank_summary.get("bank_guidance_source"),
        "bank_finite_pixels": int(bank_summary.get("finite_bank_pixels", 0) or 0),
        "bank_active_influence_pixels": int(bank_summary.get("active_bank_influence_pixels", 0) or 0),
    })
    logger.info(
        "[RIVER][V1] Minimal surface artifacts complete: support_points=%s centerline_points=%s river_primary_surface=%s bank_wse_edge_guidance=%s bank_influence=%s",
        support_summary["support_points_path"],
        backbone_summary["centerline_points_path"],
        surface_summary["river_primary_surface_path"],
        bank_summary["bank_elevation_path"],
        bank_summary["bank_influence_path"],
    )
    return Path(surface_summary["river_primary_surface_path"])
