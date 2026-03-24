from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.warp import reproject, Resampling

from final_route_inputs_stage import find_sdb_depth_raster, find_sdb_guidance_artifact, resolve_existing_output_path
from final_route_receipts import write_json_receipt
import logging
log = logging.getLogger(__name__)



_REQUIRED_RIVER_STRUCTURAL_KEYS = (
    "guide_points",
    "admissibility",
    "corridor_mask",
    "bank_influence",
    "bank_elevation_xs",
    "bank_continuity_weight",
    "bank_graph_confidence",
    "bank_confluence_damping",
    "bank_estuary_side_decay",
    "centerline_elevation",
    "centerline_influence",
    "centerline_stationing",
)

_OPTIONAL_RIVER_STRUCTURAL_KEYS = (
    "xs_support_elevation",
    "xs_support_weight",
)

_REQUIRED_SDB_STRUCTURAL_KEYS = (
    "guide_points",
    "admissibility_raster",
    "guidance_weight_raster",
)


def _river_guidance_requested(river_outputs: dict) -> bool:
    if not isinstance(river_outputs, dict):
        return False
    for key in ("guidance_manifest", "guide_points", "admissibility", "corridor_mask", "centerline_stationing"):
        if river_outputs.get(key):
            return True
    return False


def _sdb_guidance_requested(*, sdb_depth_path: Optional[Path], sdb_adm_path: Optional[Path], sdb_gw_path: Optional[Path], sdb_guide_points_path: Optional[Path], sdb_manifest_path: Optional[Path]) -> bool:
    return any(p is not None for p in (sdb_depth_path, sdb_adm_path, sdb_gw_path, sdb_guide_points_path, sdb_manifest_path))


def _require_sdb_structural_artifacts(*, sdb_depth_path: Optional[Path], sdb_adm_path: Optional[Path], sdb_gw_path: Optional[Path], sdb_guide_points_path: Optional[Path], sdb_manifest_path: Optional[Path], arrays: dict) -> dict:
    present_optional: dict[str, bool] = {}
    if not _sdb_guidance_requested(
        sdb_depth_path=sdb_depth_path,
        sdb_adm_path=sdb_adm_path,
        sdb_gw_path=sdb_gw_path,
        sdb_guide_points_path=sdb_guide_points_path,
        sdb_manifest_path=sdb_manifest_path,
    ):
        return {"requested": False, "missing_paths": [], "semantic_errors": []}

    missing_paths: list[str] = []
    if sdb_guide_points_path is None:
        missing_paths.append("guide_points")
    if sdb_adm_path is None:
        missing_paths.append("admissibility_raster")
    if sdb_gw_path is None:
        missing_paths.append("guidance_weight_raster")

    semantic_errors: list[str] = []
    adm = arrays.get("sdb_adm")
    if adm is not None:
        adm_mask = np.asarray(adm) > 0
        if np.any(adm_mask):
            sdb_candidate = arrays.get("sdb_candidate")
            if sdb_candidate is None or not np.any(np.isfinite(np.asarray(sdb_candidate)[adm_mask])):
                semantic_errors.append("sdb_candidate_has_no_finite_values_in_admissible_domain")
            sdb_gw = arrays.get("sdb_gw")
            if sdb_gw is None or not np.any(np.isfinite(np.asarray(sdb_gw)[adm_mask])):
                semantic_errors.append("sdb_guidance_weight_has_no_finite_values_in_admissible_domain")
            else:
                gw_vals = np.asarray(sdb_gw)[adm_mask]
                finite = gw_vals[np.isfinite(gw_vals)]
                if finite.size and (np.nanmin(finite) < -1e-6 or np.nanmax(finite) > 1.0 + 1e-6):
                    semantic_errors.append("sdb_guidance_weight_out_of_0_1_range")
            sdb_conf = arrays.get("sdb_confidence")
            if sdb_conf is not None:
                conf_vals = np.asarray(sdb_conf)[adm_mask]
                finite = conf_vals[np.isfinite(conf_vals)]
                if finite.size and (np.nanmin(finite) < -1e-6 or np.nanmax(finite) > 1.0 + 1e-6):
                    semantic_errors.append("sdb_confidence_out_of_0_1_range")
            lower = arrays.get("sdb_lower_bound")
            upper = arrays.get("sdb_upper_bound")
            if lower is not None and upper is not None:
                lv = np.asarray(lower)[adm_mask]
                uv = np.asarray(upper)[adm_mask]
                both = np.isfinite(lv) & np.isfinite(uv)
                if np.any(both) and np.any(lv[both] > uv[both]):
                    semantic_errors.append("sdb_bounds_inverted_in_admissible_domain")

    if missing_paths or semantic_errors:
        details=[]
        if missing_paths:
            details.append("missing=" + ",".join(missing_paths))
        if semantic_errors:
            details.append("semantic=" + ",".join(semantic_errors))
        raise RuntimeError("missing_required_sdb_structural_guidance_artifacts_for_final_route: " + "; ".join(details))

    return {"requested": True, "missing_paths": [], "semantic_errors": [], "present_optional": present_optional}


def _require_river_structural_artifacts(*, river_outputs: dict, outputs_base_dir: Path, arrays: dict, river_guide_points_path: Optional[Path]) -> dict:
    if not _river_guidance_requested(river_outputs):
        return {"requested": False, "missing_paths": [], "semantic_errors": []}

    missing_paths: list[str] = []
    present_optional: dict[str, bool] = {}
    for key in _REQUIRED_RIVER_STRUCTURAL_KEYS:
        if key == "guide_points":
            if river_guide_points_path is None:
                missing_paths.append(key)
            continue
        if resolve_existing_output_path(river_outputs, key, base_dir=outputs_base_dir) is None:
            missing_paths.append(key)
    for key in _OPTIONAL_RIVER_STRUCTURAL_KEYS:
        present_optional[key] = resolve_existing_output_path(river_outputs, key, base_dir=outputs_base_dir) is not None

    def _has_finite_in_corridor(arr: object, mask: object) -> bool:
        if arr is None or mask is None:
            return False
        arr_np = np.asarray(arr)
        mask_np = np.asarray(mask, dtype=bool)
        if arr_np.shape != mask_np.shape or not np.any(mask_np):
            return False
        return bool(np.any(np.isfinite(arr_np[mask_np])))

    semantic_errors: list[str] = []
    river_corridor = arrays.get("river_corridor")
    river_adm = arrays.get("river_adm")
    corridor_mask = None
    if river_corridor is not None:
        corridor_mask = np.asarray(river_corridor) > 0
    elif river_adm is not None:
        corridor_mask = np.asarray(river_adm) > 0
    if corridor_mask is not None and np.any(corridor_mask):
        stationing = arrays.get("river_centerline_stationing")
        if stationing is None or not np.any(np.isfinite(np.asarray(stationing)[corridor_mask])):
            semantic_errors.append("river_centerline_stationing_has_no_finite_values_in_corridor")
        centerline_elev = arrays.get("river_centerline_elevation")
        if centerline_elev is None or not np.any(np.isfinite(np.asarray(centerline_elev)[corridor_mask])):
            semantic_errors.append("river_centerline_elevation_has_no_finite_values_in_corridor")
        xs_elev = arrays.get("river_xs_support_elevation")
        xs_weight = arrays.get("river_xs_support_weight")
        xs_has_support = _has_finite_in_corridor(xs_elev, corridor_mask) or _has_finite_in_corridor(xs_weight, corridor_mask)
        if not xs_has_support:
            present_optional["xs_support_elevation"] = False
            present_optional["xs_support_weight"] = False
        bank_infl = arrays.get("river_bank_influence")
        if bank_infl is None or not np.any(np.isfinite(np.asarray(bank_infl)[corridor_mask])):
            semantic_errors.append("river_bank_influence_has_no_finite_values_in_corridor")

    if missing_paths or semantic_errors:
        details=[]
        if missing_paths:
            details.append("missing=" + ",".join(missing_paths))
        if semantic_errors:
            details.append("semantic=" + ",".join(semantic_errors))
        raise RuntimeError("missing_required_river_structural_guidance_artifacts_for_final_route: " + "; ".join(details))

    return {"requested": True, "missing_paths": [], "semantic_errors": [], "present_optional": present_optional}


@dataclass
class GuidanceAssembly:
    profile: dict
    nodata: float
    auth: np.ndarray
    legacy_candidate: Optional[np.ndarray]
    sdb_depth_path: Optional[Path]
    sdb_guide_points_path: Optional[Path]
    river_guide_points_path: Optional[Path]
    arrays: dict
    source_candidate: dict
    candidate_prov: np.ndarray
    pixel_size_m: float
    support_params: dict
    support_note_route: str
    receipt_path: Optional[Path]
    structural_artifacts: dict
    diagnostic_artifacts: dict


def grid_pixel_size_m(transform, crs, ref_lat_deg: Optional[float] = None) -> float:
    try:
        dx = abs(float(getattr(transform, "a", 0.0) or 0.0))
        dy = abs(float(getattr(transform, "e", 0.0) or 0.0))
    except Exception:
        log.debug("grid_pixel_size_m: suppressed exception", exc_info=True)
        dx = dy = 0.0
    px = max((dx + dy) / 2.0, 0.0)
    try:
        is_geographic = bool(getattr(crs, "is_geographic", False))
    except Exception:
        log.debug("grid_pixel_size_m: suppressed exception", exc_info=True)
        is_geographic = False
    if not is_geographic:
        return max(px, 1.0)
    lat = float(ref_lat_deg if ref_lat_deg is not None else 0.0)
    lat_rad = np.deg2rad(lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2.0 * lat_rad) + 1.175 * np.cos(4.0 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3.0 * lat_rad)
    dx_m = dx * max(abs(m_per_deg_lon), 1.0)
    dy_m = dy * max(abs(m_per_deg_lat), 1.0)
    return max((dx_m + dy_m) / 2.0, 1.0)


def assemble_guidance_inputs(*, cfg, paths, candidate_path: Optional[Path], provenance_path: Optional[Path], report: dict) -> GuidanceAssembly:
    del provenance_path
    river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
    outputs_base_dir = Path(getattr(cfg, "out_dir", Path.cwd()))
    river_adm_path = resolve_existing_output_path(river_outputs, "admissibility", base_dir=outputs_base_dir)
    sdb_dir = outputs_base_dir / "sdb"
    sdb_depth_path = find_sdb_depth_raster(sdb_dir)
    sdb_adm_path = find_sdb_guidance_artifact(sdb_dir, "admissibility_raster", "_admissibility")
    sdb_gw_path = find_sdb_guidance_artifact(sdb_dir, "guidance_weight_raster", "_guidance_weight")
    sdb_ti_path = find_sdb_guidance_artifact(sdb_dir, "trusted_interior_raster", "_trusted_interior")
    sdb_guide_points_path = find_sdb_guidance_artifact(sdb_dir, "guide_points", "_guide_points")
    sdb_manifest_path = sdb_dir / "sdb_guidance_manifest.json"
    if not sdb_manifest_path.exists():
        sdb_manifest_path = None
    river_depth_path = resolve_existing_output_path(river_outputs, "depth_terrain", base_dir=outputs_base_dir)

    with rasterio.open(paths.template_path) as cand_ds:
        profile = cand_ds.profile.copy()
        nodata = cand_ds.nodata
        if nodata is None:
            nodata = -9999.0

        def _align(src_path: Optional[Path], *, dtype: str = "float32", nodata_value: float | int = -9999.0, resampling=Resampling.nearest):
            if src_path is None or (not Path(src_path).exists()):
                return None
            arr = np.full((cand_ds.height, cand_ds.width), nodata_value, dtype=dtype)
            with rasterio.open(src_path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=cand_ds.transform,
                    dst_crs=cand_ds.crs,
                    resampling=resampling,
                    src_nodata=src.nodata,
                    dst_nodata=nodata_value,
                )
            return arr

        auth = _align(paths.auth_src, dtype="float32", nodata_value=np.float32(nodata), resampling=Resampling.nearest)
        if auth is None:
            raise RuntimeError("failed to align authoritative base to final-route template")
        auth = auth.astype("float32")
        auth[np.isclose(auth, np.float32(nodata))] = np.nan

        legacy_candidate = _align(candidate_path, dtype="float32", nodata_value=np.float32(nodata), resampling=Resampling.bilinear)
        if legacy_candidate is not None:
            legacy_candidate = legacy_candidate.astype("float32")
            legacy_candidate[np.isclose(legacy_candidate, np.float32(nodata))] = np.nan

        arrays = {
            "sdb_candidate": _align(sdb_depth_path, dtype="float32", nodata_value=np.float32(nodata), resampling=Resampling.bilinear),
            "sdb_adm": _align(sdb_adm_path, dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_adm": _align(river_adm_path, dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "sdb_gw": _align(sdb_gw_path, dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "sdb_ti": _align(sdb_ti_path, dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "sdb_confidence": _align(find_sdb_guidance_artifact(sdb_dir, "confidence_raster", "_confidence"), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "sdb_lower_bound": _align(find_sdb_guidance_artifact(sdb_dir, "lower_bound_raster", "_lower_bound"), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "sdb_upper_bound": _align(find_sdb_guidance_artifact(sdb_dir, "upper_bound_raster", "_upper_bound"), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_gw": _align(resolve_existing_output_path(river_outputs, "guidance_weight", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_ti": _align(resolve_existing_output_path(river_outputs, "trusted_interior", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_estuary_transition": _align(resolve_existing_output_path(river_outputs, "estuary_transition", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_support": _align(resolve_existing_output_path(river_outputs, "authoritative_support", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_support_depth": _align(resolve_existing_output_path(river_outputs, "authoritative_support_depth", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_corridor": _align(resolve_existing_output_path(river_outputs, "corridor_mask", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_bank_influence": _align(resolve_existing_output_path(river_outputs, "bank_influence", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_bank_elevation_xs": _align(resolve_existing_output_path(river_outputs, "bank_elevation_xs", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_bank_pair_weight": _align(resolve_existing_output_path(river_outputs, "bank_pair_weight", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_bank_continuity_weight": _align(resolve_existing_output_path(river_outputs, "bank_continuity_weight", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_bank_graph_confidence": _align(resolve_existing_output_path(river_outputs, "bank_graph_confidence", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_bank_confluence_damping": _align(resolve_existing_output_path(river_outputs, "bank_confluence_damping", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_bank_estuary_side_decay": _align(resolve_existing_output_path(river_outputs, "bank_estuary_side_decay", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_centerline_elevation": _align(resolve_existing_output_path(river_outputs, "centerline_elevation", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_centerline_influence": _align(resolve_existing_output_path(river_outputs, "centerline_influence", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_centerline_stationing": _align(resolve_existing_output_path(river_outputs, "centerline_stationing", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_longitudinal_profile_elevation": _align(resolve_existing_output_path(river_outputs, "longitudinal_profile_elevation", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_longitudinal_profile_uncertainty": _align(resolve_existing_output_path(river_outputs, "longitudinal_profile_uncertainty", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_longitudinal_profile_influence": _align(resolve_existing_output_path(river_outputs, "longitudinal_profile_influence", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_xs_support_elevation": _align(resolve_existing_output_path(river_outputs, "xs_support_elevation", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_xs_support_weight": _align(resolve_existing_output_path(river_outputs, "xs_support_weight", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
        }
        if arrays["sdb_candidate"] is not None:
            arrays["sdb_candidate"] = arrays["sdb_candidate"].astype("float32")
            arrays["sdb_candidate"][np.isclose(arrays["sdb_candidate"], np.float32(nodata))] = np.nan

        sdb_guide_points_path_active = sdb_guide_points_path if sdb_guide_points_path and Path(sdb_guide_points_path).exists() else None
        river_guide_points_path_active = resolve_existing_output_path(river_outputs, "guide_points", base_dir=outputs_base_dir)

        source_candidate = {
            "candidate": np.full(auth.shape, np.nan, dtype="float32"),
            "provenance": np.zeros(auth.shape, dtype="uint8"),
            "stats": {
                "candidate_pixels": 0,
                "sdb_pixels": 0,
                "river_pixels": 0,
                "blended_pixels": 0,
                "legacy_fallback_pixels": 0,
                "legacy_gap_only_backstop_pixels": 0,
                "legacy_blocked_in_river_corridor_pixels": 0,
                "unused_pixels": int(auth.size),
            },
            "backstop_policy": {
                "legacy_candidate_enabled": False,
                "legacy_candidate_role": "disabled",
                "active_final_route_mode": "direct_guidance_artifacts_only",
                "sdb_guidance_source": "native_sparse_guide_points" if sdb_guide_points_path_active is not None else "no_sparse_guide_points_available",
                "river_guidance_source": "native_structured_sparse_guide_points" if river_guide_points_path_active is not None else "no_structured_sparse_guide_points_available",
            },
            "provenance_codes": {"0": "not_used_in_active_final_route"},
        }
        candidate_prov = source_candidate["provenance"].astype("uint8")
        pixel_size_m = grid_pixel_size_m(cand_ds.transform, cand_ds.crs, ref_lat_deg=(cfg.tile_bbox[1] + cfg.tile_bbox[3]) / 2.0 if getattr(cfg, "tile_bbox", None) else None)

    support_params = {
        "pixel_size_m": pixel_size_m,
        "support_decay_m": float(getattr(cfg, "authoritative_support_decay_m", 300.0) or 300.0),
        "support_density_radius_m": float(getattr(cfg, "authoritative_support_density_radius_m", 250.0) or 250.0),
        "coastal_sdb_support_transition_m": float(getattr(cfg, "coastal_sdb_support_transition_m", 600.0) or 600.0),
        "river_anchor_density_radius_m": float(getattr(cfg, "river_anchor_density_radius_m", 200.0) or 200.0),
        "river_scaffold_transition_m": float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0),
    }
    sdb_requirements = _require_sdb_structural_artifacts(
        sdb_depth_path=sdb_depth_path,
        sdb_adm_path=sdb_adm_path,
        sdb_gw_path=sdb_gw_path,
        sdb_guide_points_path=sdb_guide_points_path_active,
        sdb_manifest_path=sdb_manifest_path,
        arrays=arrays,
    )
    river_requirements = _require_river_structural_artifacts(
        river_outputs=river_outputs,
        outputs_base_dir=outputs_base_dir,
        arrays=arrays,
        river_guide_points_path=river_guide_points_path_active,
    )

    structural_artifacts = {
        "sdb_guide_points": str(sdb_guide_points_path_active) if sdb_guide_points_path_active is not None else None,
        "river_guide_points": str(river_guide_points_path_active) if river_guide_points_path_active is not None else None,
        "sdb_guidance_weight": bool(arrays["sdb_gw"] is not None),
        "sdb_admissibility": bool(arrays["sdb_adm"] is not None),
        "sdb_trusted_interior": bool(arrays["sdb_ti"] is not None),
        "sdb_confidence": bool(arrays["sdb_confidence"] is not None),
        "sdb_lower_bound": bool(arrays["sdb_lower_bound"] is not None),
        "sdb_upper_bound": bool(arrays["sdb_upper_bound"] is not None),
        "river_guidance_weight": bool(arrays["river_gw"] is not None),
        "river_admissibility": bool(arrays["river_adm"] is not None),
        "river_trusted_interior": bool(arrays["river_ti"] is not None),
        "river_corridor_mask": bool(arrays["river_corridor"] is not None),
        "river_bank_influence": bool(arrays["river_bank_influence"] is not None),
        "river_bank_elevation_xs": bool(arrays["river_bank_elevation_xs"] is not None),
        "river_centerline_elevation": bool(arrays["river_centerline_elevation"] is not None),
        "river_longitudinal_profile_elevation": bool(arrays["river_longitudinal_profile_elevation"] is not None),
        "river_centerline_influence": bool(arrays["river_centerline_influence"] is not None),
        "river_centerline_stationing": bool(arrays["river_centerline_stationing"] is not None),
        "river_xs_support_elevation": bool(river_requirements.get("present_optional", {}).get("xs_support_elevation", False)),
        "river_xs_support_weight": bool(river_requirements.get("present_optional", {}).get("xs_support_weight", False)),
    }
    diagnostic_artifacts = {
        "legacy_candidate": str(candidate_path) if candidate_path else None,
        "dense_sdb_depth_raster": str(sdb_depth_path) if sdb_depth_path else None,
        "dense_river_depth_raster": str(river_depth_path) if river_depth_path else None,
    }
    write_json_receipt(paths.guidance_receipt_path, {
        "stage": "guidance_assembly",
        "route_mode": "staged_final_route_single_source_of_truth",
        "structural_artifacts": structural_artifacts,
        "diagnostic_only_artifacts": diagnostic_artifacts,
        "support_params": support_params,
        "pixel_size_m": pixel_size_m,
        "sdb_requirements": sdb_requirements,
        "river_requirements": river_requirements,
    })
    return GuidanceAssembly(
        profile=profile,
        nodata=float(nodata),
        auth=auth,
        legacy_candidate=legacy_candidate,
        sdb_depth_path=sdb_depth_path,
        sdb_guide_points_path=sdb_guide_points_path_active,
        river_guide_points_path=river_guide_points_path_active,
        arrays=arrays,
        source_candidate=source_candidate,
        candidate_prov=candidate_prov,
        pixel_size_m=pixel_size_m,
        support_params=support_params,
        support_note_route="direct_guidance_artifacts_only",
        receipt_path=paths.guidance_receipt_path,
        structural_artifacts=structural_artifacts,
        diagnostic_artifacts=diagnostic_artifacts,
    )
