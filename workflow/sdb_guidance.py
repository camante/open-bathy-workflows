from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from final_dem_policy import default_final_dem_policy
from simple_river_stage_contract import (
    ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
    ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
)

import numpy as np

from memory_diag import memory_checkpoint

log = logging.getLogger(__name__)


def guidance_artifact_paths(depth_raster: str | Path) -> Dict[str, Path]:
    depth = Path(depth_raster)
    stem = depth.stem
    parent = depth.parent
    return {
        "depth_raster": depth,
        "confidence_raster": parent / f"{stem}_confidence.tif",
        "provenance_raster": parent / f"{stem}_provenance.tif",
        "uncertainty_raster": parent / f"{stem}_uncertainty.tif",
        "guidance_weight_raster": parent / f"{stem}_guidance_weight.tif",
        "trusted_interior_raster": parent / f"{stem}_trusted_interior.tif",
        "admissibility_raster": parent / f"{stem}_admissibility.tif",
        "regime_class_raster": parent / f"{stem}_regime_class.tif",
        "guide_points": parent / f"{stem}_guide_points.gpkg",
        "lower_bound_raster": parent / f"{stem}_lower_bound.tif",
        "upper_bound_raster": parent / f"{stem}_upper_bound.tif",
        "guidance_manifest": parent / f"{stem}_guidance_manifest.json",
    }


def write_sdb_guidance_bounds(depth_path: str | Path, uncertainty_path: str | Path, *, logger: Optional[logging.Logger] = None) -> Dict[str, Optional[str]]:
    import rasterio

    depth_path = Path(depth_path)
    uncertainty_path = Path(uncertainty_path)
    paths = guidance_artifact_paths(depth_path)
    lower_path = paths["lower_bound_raster"]
    upper_path = paths["upper_bound_raster"]
    active_log = logger or log

    if (not depth_path.exists()) or (not uncertainty_path.exists()):
        return {"lower_bound_raster": None, "upper_bound_raster": None}

    with rasterio.open(depth_path) as ds_depth, rasterio.open(uncertainty_path) as ds_unc:
        depth = ds_depth.read(1).astype("float32")
        unc = ds_unc.read(1).astype("float32")
        nodata = ds_depth.nodata if ds_depth.nodata is not None else -9999.0
        valid = np.isfinite(depth) & np.isfinite(unc) & (depth != nodata)
        prof = ds_depth.profile.copy()
        lower = np.full(depth.shape, nodata, dtype=np.float32)
        upper = np.full(depth.shape, nodata, dtype=np.float32)
        lower[valid] = depth[valid] - unc[valid]
        upper[valid] = depth[valid] + unc[valid]
        prof.update(dtype="float32", count=1, compress="deflate", nodata=nodata)
        with rasterio.open(lower_path, "w", **prof) as dst:
            dst.write(lower, 1)
        with rasterio.open(upper_path, "w", **prof) as dst:
            dst.write(upper, 1)
    active_log.info("SDB guidance bounds written: %s ; %s", lower_path, upper_path)
    return {"lower_bound_raster": str(lower_path), "upper_bound_raster": str(upper_path)}




def _pick_sdb_guide_value_column(columns: list[str]) -> Optional[str]:
    preferred = [
        "depth_m",
        "bottom_elevation",
        "bed_elev",
        "elevation_m",
        "elevation",
        "depth",
        "z",
        "value",
    ]
    lowered = {str(c).lower(): c for c in columns}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    return None


def rasterize_sdb_guide_points_to_template(guide_points_path: str | Path, template_raster: str | Path, *, logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    import rasterio
    from rasterio.transform import rowcol
    try:
        import geopandas as gpd
    except Exception:
        log.debug("rasterize_sdb_guide_points_to_template: suppressed exception", exc_info=True)
        return None

    gp = Path(guide_points_path)
    tmpl = Path(template_raster)
    active_log = logger or log
    if (not gp.exists()) or (not tmpl.exists()):
        return None
    mem_start = memory_checkpoint("sdb_guide_rasterize_start", guide_points=str(gp), template=str(tmpl))
    active_log.info("[MEMORY][SDB] %s", mem_start)

    try:
        gdf = gpd.read_file(gp)
    except Exception:
        log.debug("rasterize_sdb_guide_points_to_template: suppressed exception", exc_info=True)
        return None
    if gdf is None or gdf.empty or 'geometry' not in gdf.columns:
        return None
    value_col = _pick_sdb_guide_value_column(list(gdf.columns))
    if value_col is None:
        return None
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        return None

    with rasterio.open(tmpl) as ds:
        if gdf.crs is not None and ds.crs is not None and str(gdf.crs) != str(ds.crs):
            try:
                gdf = gdf.to_crs(ds.crs)
            except Exception:
                log.debug("rasterize_sdb_guide_points_to_template: suppressed exception", exc_info=True)
                return None
        arr = np.full((ds.height, ds.width), np.nan, dtype=np.float32)
        sums = np.zeros((ds.height, ds.width), dtype=np.float64)
        counts = np.zeros((ds.height, ds.width), dtype=np.uint32)
        vals = np.asarray(gdf[value_col], dtype=float)
        for geom, val in zip(gdf.geometry, vals):
            if geom is None or not np.isfinite(val):
                continue
            try:
                r, c = rowcol(ds.transform, geom.x, geom.y)
            except Exception:
                log.debug("rasterize_sdb_guide_points_to_template: suppressed exception", exc_info=True)
                continue
            if 0 <= int(r) < ds.height and 0 <= int(c) < ds.width:
                sums[int(r), int(c)] += float(val)
                counts[int(r), int(c)] += 1
        valid = counts > 0
        if not np.any(valid):
            return None
        arr[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    populated = int(np.sum(np.isfinite(arr)))
    mem_end = memory_checkpoint(
        "sdb_guide_rasterize_end",
        guide_points=str(gp),
        template=str(tmpl),
        populated_cells=populated,
        point_rows=int(len(gdf)),
    )
    active_log.info('Rasterized SDB guide points onto template grid: %s -> %s populated cells', gp, populated)
    active_log.info('[MEMORY][SDB] %s', mem_end)
    return arr



def write_sdb_authoritative_locked_guidance(
    depth_path: str | Path,
    *,
    support_mask_path: str | Path | None,
    support_values_path: str | Path | None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    import rasterio
    from rasterio.warp import reproject, Resampling

    depth_path = Path(depth_path)
    support_mask_path = Path(support_mask_path) if support_mask_path else None
    support_values_path = Path(support_values_path) if support_values_path else None
    active_log = logger or log
    paths = guidance_artifact_paths(depth_path)
    locked_path = depth_path.with_name(depth_path.stem + "_authoritative_locked" + depth_path.suffix)
    diff_path = depth_path.with_name(depth_path.stem + "_lock_diff_before_overwrite" + depth_path.suffix)
    contract_path = depth_path.with_name(depth_path.stem + "_lock_contract.json")

    result: Dict[str, Any] = {
        "locked_guidance_raster": None,
        "lock_diff_before_overwrite_raster": None,
        "lock_contract": None,
        "authoritative_support_active": False,
        "authoritative_supported_cell_count": 0,
        "unlocked_predicted_cell_count": 0,
        "authoritative_supported_cells_changed_before_lock_count": 0,
        "authoritative_supported_cells_after_lock_mismatch_count": 0,
    }
    if (not depth_path.exists()) or support_mask_path is None or support_values_path is None or (not support_mask_path.exists()) or (not support_values_path.exists()):
        contract_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        result['lock_contract'] = str(contract_path)
        return result

    with rasterio.open(depth_path) as ds_depth:
        depth = ds_depth.read(1).astype(np.float32)
        prof = ds_depth.profile.copy()
        depth_nodata = ds_depth.nodata if ds_depth.nodata is not None else -9999.0
        depth_valid = np.isfinite(depth) & (depth != depth_nodata)
        target_shape = depth.shape
        target_transform = ds_depth.transform
        target_crs = ds_depth.crs

        def _read_aligned(src_path: Path, *, is_mask: bool) -> np.ndarray:
            with rasterio.open(src_path) as ds_src:
                src_arr = ds_src.read(1)
                if ds_src.shape == target_shape and ds_src.transform == target_transform and str(ds_src.crs) == str(target_crs):
                    arr = src_arr.astype(np.float32 if not is_mask else np.uint8, copy=False)
                else:
                    dst = np.zeros(target_shape, dtype=np.float32 if not is_mask else np.uint8)
                    reproject(
                        source=src_arr,
                        destination=dst,
                        src_transform=ds_src.transform,
                        src_crs=ds_src.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        src_nodata=ds_src.nodata,
                        dst_nodata=0 if is_mask else depth_nodata,
                        resampling=Resampling.nearest,
                    )
                    arr = dst
                return arr

        support_mask = _read_aligned(support_mask_path, is_mask=True)
        support_values = _read_aligned(support_values_path, is_mask=False)
        support_valid = (support_mask > 0) & np.isfinite(support_values) & (support_values != depth_nodata)

        locked = depth.copy()
        diff = np.full(target_shape, depth_nodata, dtype=np.float32)
        changed = support_valid & depth_valid & (np.abs(depth - support_values) > 1.0e-6)
        diff[support_valid] = np.where(depth_valid[support_valid], depth[support_valid] - support_values[support_valid], depth_nodata).astype(np.float32)
        locked[support_valid] = support_values[support_valid].astype(np.float32)
        mismatch_after = support_valid & np.isfinite(locked) & (locked != depth_nodata) & (np.abs(locked - support_values) > 1.0e-6)

        prof.update(dtype='float32', count=1, compress='deflate', nodata=depth_nodata)
        with rasterio.open(locked_path, 'w', **prof) as dst:
            dst.write(locked.astype(np.float32), 1)
        with rasterio.open(diff_path, 'w', **prof) as dst:
            dst.write(diff.astype(np.float32), 1)

    result.update({
        "locked_guidance_raster": str(locked_path),
        "lock_diff_before_overwrite_raster": str(diff_path),
        "authoritative_support_active": bool(np.any(support_valid)),
        "authoritative_supported_cell_count": int(np.sum(support_valid)),
        "unlocked_predicted_cell_count": int(np.sum(np.isfinite(locked) & (locked != depth_nodata) & (~support_valid))),
        "authoritative_supported_cells_changed_before_lock_count": int(np.sum(changed)),
        "authoritative_supported_cells_after_lock_mismatch_count": int(np.sum(mismatch_after)),
        "support_mask_path": str(support_mask_path),
        "support_values_path": str(support_values_path),
        "raw_prediction_raster": str(depth_path),
    })
    contract_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    result['lock_contract'] = str(contract_path)
    active_log.info(
        "SDB authoritative lock written: %s (supported=%d changed_before_lock=%d)",
        locked_path,
        int(result['authoritative_supported_cell_count']),
        int(result['authoritative_supported_cells_changed_before_lock_count']),
    )
    return result

def build_sdb_guidance_manifest(*, out_root: str | Path, depth_raster: str | Path, args: Any) -> Dict[str, Any]:
    out_root = Path(out_root)
    depth_raster = Path(depth_raster)
    paths = guidance_artifact_paths(depth_raster)
    locked_path = depth_raster.with_name(depth_raster.stem + "_authoritative_locked" + depth_raster.suffix)
    diff_path = depth_raster.with_name(depth_raster.stem + "_lock_diff_before_overwrite" + depth_raster.suffix)
    contract_path = depth_raster.with_name(depth_raster.stem + "_lock_contract.json")

    def _rel_or_abs(p: Path) -> Optional[str]:
        if not p.exists():
            return None
        try:
            return str(p.relative_to(out_root))
        except ValueError:
            return str(p)

    artifacts: Dict[str, Optional[str]] = {}
    for key, p in paths.items():
        if key == "guidance_manifest":
            continue
        rel = _rel_or_abs(p)
        if rel:
            artifacts[key] = rel

    locked_rel = _rel_or_abs(locked_path)
    diff_rel = _rel_or_abs(diff_path)
    contract_rel = _rel_or_abs(contract_path)
    if locked_rel:
        artifacts["depth_raster"] = locked_rel
        artifacts["sdb_guidance_active"] = locked_rel
        artifacts["raw_prediction_raster"] = _rel_or_abs(depth_raster)
        artifacts["sdb_locked_guidance_raster"] = locked_rel
    else:
        artifacts["depth_raster"] = _rel_or_abs(depth_raster)
        artifacts["sdb_guidance_active"] = _rel_or_abs(depth_raster)
    if diff_rel:
        artifacts["lock_diff_before_overwrite_raster"] = diff_rel
    if contract_rel:
        artifacts["sdb_lock_contract"] = contract_rel

    artifacts["guidance_mode"] = "authoritative_locked_guidance" if locked_rel else "guidance_first"
    artifacts["depth_raster_role"] = "active_guidance" if locked_rel else "diagnostic_only"
    auth_base = getattr(args, "authoritative_base", None)
    if auth_base:
        artifacts["authoritative_base"] = str(auth_base)
    for attr, key in [
        ("sdb_authoritative_support_mask", "authoritative_support_mask"),
        ("sdb_authoritative_support_values", "authoritative_support_values"),
        ("sdb_authoritative_support_points", "authoritative_support_points"),
        ("sdb_authoritative_support_contract", "authoritative_support_contract"),
    ]:
        val = getattr(args, attr, None)
        if val:
            artifacts[key] = str(val)
    if hasattr(args, "_authoritative_base_auto_report"):
        artifacts["authoritative_base_auto"] = getattr(args, "_authoritative_base_auto_report")

    policy = default_final_dem_policy()
    manifest = {
        "schema_version": 2,
        "artifact_family": "sdb_guidance",
        "guidance_only": True,
        "current_route_mode": ROUTE_MODE_LEGACY_STRUCTURED_TRANSITION,
        "target_route_mode": ROUTE_MODE_SIMPLE_RIVER_PLAN_V1,
        "artifacts": artifacts,
        "final_dem_policy": {
            "final_dem_filename": policy.final_dem_filename,
            "internal_final_dem_filename": policy.internal_final_dem_filename,
            "write_final_dem_once": bool(policy.write_final_dem_once),
            "verify_only_postwrite": bool(policy.verify_only_postwrite),
        },
        "artifact_roles": {
            "depth_raster": "active_guidance" if locked_rel else "diagnostic_only",
            "sdb_guidance_active": "active_guidance",
            "raw_prediction_raster": "diagnostic_only",
            "sdb_locked_guidance_raster": "active_guidance",
            "confidence_raster": "confidence",
            "provenance_raster": "provenance",
            "guidance_weight_raster": "soft_guidance_weight",
            "trusted_interior_raster": "trusted_export_region",
            "admissibility_raster": "soft_guidance_domain",
            "regime_class_raster": "shared_regime_contract",
            "guide_points": "sparse_guidance_points",
            "lower_bound_raster": "plausible_lower_bound",
            "upper_bound_raster": "plausible_upper_bound",
            "lock_diff_before_overwrite_raster": "diagnostic_only",
            "sdb_lock_contract": "contract",
            "authoritative_support_mask": "authoritative_support",
            "authoritative_support_values": "authoritative_support",
            "authoritative_support_points": "authoritative_support",
            "authoritative_support_contract": "contract",
        },
        "final_route_contract": {
            "route_role": "subordinate_guidance_artifacts_only",
            "allowed_structural_artifacts": [
                "admissibility_raster",
                "confidence_raster",
                "sdb_guidance_active",
                "guide_points",
                "guidance_weight_raster",
                "lower_bound_raster",
                "provenance_raster",
                "regime_class_raster",
                "trusted_interior_raster",
                "upper_bound_raster",
            ],
            "diagnostic_only_artifacts": ["raw_prediction_raster", "lock_diff_before_overwrite_raster"],
            "forbidden_structural_inputs": [
                "legacy_fused_candidate_raster",
                "dense_river_depth_raster_as_peer_surface",
                "dense_sdb_depth_raster_as_peer_surface",
                "weighted_overlap_blended_bathymetry_as_structural_input",
            ],
        },
        "notes": {
            "depth_raster": "Legacy canonical SDB guidance path retained for compatibility; points to the active guidance product.",
            "sdb_guidance_active": "Canonical workflow SDB guidance product used downstream. Authoritative-supported cells are hard-locked when available; unsupported cells remain predictive.",
            "raw_prediction_raster": "Raw dense SDB prediction retained for diagnostics and QA only.",
            "guide_points": "Spatially thinned pseudo-soundings for confidence-weighted interpolation guidance.",
            "bounds": "Lower/upper bounds are uncertainty-derived plausible guidance envelopes, not hard truth.",
            "final_route_contract": "Only the listed subordinate guidance artifacts may structurally enter the final DEM route; raw dense SDB depth remains diagnostic-only.",
        },
    }
    return manifest


def write_sdb_guidance_manifest(*, out_root: str | Path, depth_raster: str | Path, args: Any, logger: Optional[logging.Logger] = None) -> Path:
    manifest = build_sdb_guidance_manifest(out_root=out_root, depth_raster=depth_raster, args=args)
    manifest_path = guidance_artifact_paths(depth_raster)["guidance_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (logger or log).info("Wrote SDB guidance manifest: %s", manifest_path)
    return manifest_path
