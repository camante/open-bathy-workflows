from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.warp import reproject, Resampling

from pipeline.final_route.final_route_inputs_stage import find_sdb_depth_raster, find_sdb_guidance_artifact, resolve_existing_output_path, resolve_baseline_cudem_interpolation
from pipeline.final_route.final_route_receipts import write_json_receipt
from core.nodata_utils import prepare_array_for_reproject, sanitize_array, sanitize_for_output

log = logging.getLogger(__name__)


def _same_grid(src_ds, template_ds) -> bool:
    return (
        src_ds.width == template_ds.width
        and src_ds.height == template_ds.height
        and src_ds.crs == template_ds.crs
        and src_ds.transform == template_ds.transform
    )


def _read_aligned_exact(src_ds, *, dtype: str, nodata_value: float | int):
    src_band = src_ds.read(1, masked=False)
    src_prepared, src_prepared_nodata = prepare_array_for_reproject(
        src_band,
        src_ds.nodata,
        dtype='float32',
        default_nodata=-9999.0,
    )
    arr = src_prepared.astype(dtype, copy=False)
    if np.issubdtype(np.dtype(dtype), np.floating):
        arr = sanitize_for_output(arr, nodata=src_prepared_nodata, dtype=dtype)
    return arr, src_prepared_nodata



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
    "channel_surface",
    "channel_surface_confidence",
    "channel_surface_source_class",
    "channel_surface_support_count",
    "channel_surface_authoritative_lock_scope",
    "channel_surface_authoritative_lock_applied",
    "channel_surface_prediction_support_confidence",
    "channel_surface_measured_anchor_fraction",
    "channel_surface_structure_only_fraction",
    "channel_surface_low_support_caution",
    "channel_surface_prediction_admissibility",
)

_REQUIRED_SDB_STRUCTURAL_KEYS = (
    "admissibility_raster",
    "guidance_weight_raster",
)

# guide_points is structurally expected but degrades gracefully when absent
# (e.g., authoritative-dense AOIs where SDB has no confident predictions).
_EXPECTED_SDB_STRUCTURAL_KEYS = (
    "guide_points",
)


def _river_guidance_requested(river_outputs: dict) -> bool:
    if not isinstance(river_outputs, dict):
        return False
    for key in ("guidance_manifest", "guide_points", "admissibility", "corridor_mask", "centerline_stationing"):
        if river_outputs.get(key):
            return True
    return False


def _sdb_guidance_requested(*, sdb_depth_path: Optional[Path], sdb_adm_path: Optional[Path], sdb_gw_path: Optional[Path], sdb_guide_points_path: Optional[Path], sdb_manifest_path: Optional[Path]) -> bool:
    # sdb_depth_path is diagnostic-only in the guidance-first architecture.
    # Guidance is "requested" only when at least one actual guidance artifact
    # (admissibility, weight, guide_points, or manifest) is present.  A bare
    # depth raster left over from a failed or degraded SDB run should not
    # trigger the structural-artifact requirement check.
    return any(p is not None for p in (sdb_adm_path, sdb_gw_path, sdb_guide_points_path, sdb_manifest_path))


def _require_sdb_structural_artifacts(*, sdb_depth_path: Optional[Path], sdb_adm_path: Optional[Path], sdb_gw_path: Optional[Path], sdb_guide_points_path: Optional[Path], sdb_manifest_path: Optional[Path], arrays: dict) -> dict:
    if not _sdb_guidance_requested(
        sdb_depth_path=sdb_depth_path,
        sdb_adm_path=sdb_adm_path,
        sdb_gw_path=sdb_gw_path,
        sdb_guide_points_path=sdb_guide_points_path,
        sdb_manifest_path=sdb_manifest_path,
    ):
        return {"requested": False, "missing_paths": [], "semantic_errors": []}

    missing_paths: list[str] = []
    warned_paths: list[str] = []
    present_optional: dict[str, bool] = {}

    # guide_points is structurally expected but not fatal when absent.  In
    # authoritative-dense AOIs the SDB may produce an empty (or missing)
    # guide_points file — the admissibility and weight rasters are still
    # useful for the terrain interpolator.
    if sdb_guide_points_path is None:
        warned_paths.append("guide_points")
        log.warning("SDB guide_points file not found; SDB guidance will proceed "
                     "with admissibility and weight rasters only.")
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

    # Fatal only for missing core artifacts (admissibility/weight) or
    # semantic errors.  Missing guide_points is a degraded but valid state.
    if missing_paths or semantic_errors:
        details=[]
        if missing_paths:
            details.append("missing=" + ",".join(missing_paths))
        if semantic_errors:
            details.append("semantic=" + ",".join(semantic_errors))
        raise RuntimeError("missing_required_sdb_structural_guidance_artifacts_for_final_route: " + "; ".join(details))

    return {"requested": True, "missing_paths": warned_paths, "semantic_errors": [], "present_optional": present_optional}


def _require_river_structural_artifacts(*, river_outputs: dict, outputs_base_dir: Path, arrays: dict, river_guide_points_path: Optional[Path]) -> dict:
    if not _river_guidance_requested(river_outputs):
        return {"requested": False, "missing_paths": [], "semantic_errors": []}

    river_v2_direct_guidance = bool(
        isinstance(river_outputs, dict)
        and (
            river_outputs.get("river_v2_summary")
            or river_outputs.get("river_v2_primary_surface_trusted_export")
            or river_outputs.get("primary_river_guidance_surface")
        )
    )

    required_keys = _REQUIRED_RIVER_STRUCTURAL_KEYS
    if river_v2_direct_guidance:
        required_keys = (
            "admissibility",
            "guidance_weight",
            "trusted_interior",
        )

    missing_paths: list[str] = []
    present_optional: dict[str, bool] = {}
    for key in required_keys:
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
        if river_v2_direct_guidance:
            primary_surface = arrays.get("primary_river_guidance_surface")
            if primary_surface is None or not np.any(np.isfinite(np.asarray(primary_surface)[corridor_mask])):
                semantic_errors.append("river_primary_guidance_surface_has_no_finite_values_in_corridor")
            river_gw = arrays.get("river_gw")
            if river_gw is None or not np.any(np.isfinite(np.asarray(river_gw)[corridor_mask])):
                semantic_errors.append("river_guidance_weight_has_no_finite_values_in_corridor")
        else:
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

    return {"requested": True, "missing_paths": [], "semantic_errors": [], "present_optional": present_optional, "river_v2_direct_guidance": river_v2_direct_guidance}


@dataclass
class GuidanceAssembly:
    profile: dict
    nodata: float
    auth: np.ndarray
    legacy_candidate: Optional[np.ndarray]
    baseline_cudem_path: Optional[Path]
    baseline_background: Optional[np.ndarray]
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


_GUIDANCE_LOCK_ZERO_KEYS = (
    "sdb_adm",
    "sdb_gw",
    "sdb_ti",
    "sdb_confidence",
    "river_adm",
    "river_gw",
    "river_ti",
    "river_bank_pair_weight",
    "river_bank_continuity_weight",
    "river_bank_graph_confidence",
    "river_bank_confluence_damping",
    "river_bank_estuary_side_decay",
    "river_centerline_influence",
    "river_longitudinal_profile_influence",
    "river_longitudinal_profile_local_authoritative_reconciliation",
    "river_longitudinal_profile_local_authoritative_reconciliation_influence",
    "river_xs_support_weight",
)

_GUIDANCE_LOCK_NAN_KEYS = (
    "sdb_candidate",
    "sdb_lower_bound",
    "sdb_upper_bound",
    "river_support_depth",
    "primary_river_guidance_surface",
    "river_centerline_elevation",
    "river_centerline_stationing",
    "river_channel_surface",
    "river_longitudinal_profile_elevation",
    "river_longitudinal_profile_local_authoritative_reconciliation",
    "river_longitudinal_profile_uncertainty",
    "river_xs_support_elevation",
)


def _maybe_harmonize_authoritative_semantics(*, auth: np.ndarray, legacy_candidate: Optional[np.ndarray], baseline_background: Optional[np.ndarray], river_support_depth: Optional[np.ndarray], primary_river_guidance_surface: Optional[np.ndarray], logger=None) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], dict]:
    """Audit vertical/sign semantics without mutating authoritative arrays.

    The authoritative base is the hard-control elevation reference for the final route.
    It must never be auto-negated here based on heuristic overlap tests with auxiliary
    river products. Earlier versions attempted to flip authoritative/base arrays when a
    sign mismatch seemed likely, but that makes the final DEM vulnerable to silent
    global sign/reference reversals. Keep the arrays unchanged and only report the
    diagnostic comparison.
    """
    log = logger or logging.getLogger(__name__)
    receipt = {
        "applied": False,
        "reason": "authoritative_semantics_preserved",
        "median_abs_diff_before_m": None,
        "median_abs_diff_after_m": None,
        "overlap_pixels": 0,
        "reference_source": None,
        "flipped_arrays": [],
    }

    def _choose_reference():
        refs = []
        if primary_river_guidance_surface is not None:
            arr = np.asarray(primary_river_guidance_surface, dtype=np.float32)
            refs.append(("primary_river_guidance_surface", arr))
        if river_support_depth is not None:
            arr = np.asarray(river_support_depth, dtype=np.float32)
            refs.append(("river_support_depth", arr))
        for name, ref in refs:
            m = np.isfinite(ref)
            if np.count_nonzero(m) > 100:
                return name, ref
        return None, None

    ref_name, ref = _choose_reference()
    if ref is None:
        receipt["reason"] = "no_reference_surface"
        return auth, legacy_candidate, baseline_background, receipt

    auth_f = np.asarray(auth, dtype=np.float32)
    m = np.isfinite(auth_f) & np.isfinite(ref)
    overlap = int(np.count_nonzero(m))
    receipt["overlap_pixels"] = overlap
    receipt["reference_source"] = ref_name
    if overlap < 100:
        receipt["reason"] = "insufficient_overlap"
        return auth, legacy_candidate, baseline_background, receipt

    before = np.abs(auth_f[m] - ref[m])
    after = np.abs((-auth_f[m]) - ref[m])
    med_before = float(np.nanmedian(before)) if before.size else None
    med_after = float(np.nanmedian(after)) if after.size else None
    receipt["median_abs_diff_before_m"] = med_before
    receipt["median_abs_diff_after_m"] = med_after
    if np.isfinite(med_before) and np.isfinite(med_after) and med_after + 0.5 < med_before:
        receipt["reason"] = "detected_possible_sign_mismatch_preserved_authoritative_reference"
        log.warning(
            "[FINAL_ROUTE][SEMANTICS] Detected possible sign mismatch against %s (median abs diff %.3f -> %.3f m if negated), but preserved authoritative/base semantics.",
            ref_name,
            med_before,
            med_after,
        )
    else:
        receipt["reason"] = "already_aligned"
    return auth, legacy_candidate, baseline_background, receipt


def _exclude_guidance_from_authoritative_locked_cells(arrays: dict, locked: np.ndarray) -> dict:
    """Mask guidance over authoritative cells without duplicating every large array.

    The guidance arrays are assembled solely for the active final-route stage, so
    mutating them in place is safe here and avoids a second full-size copy of the
    guidance stack just before terrain interpolation.
    """
    locked = np.asarray(locked, dtype=bool)
    sanitized = arrays
    receipt: dict[str, dict[str, object]] = {}

    masked_pixels = int(np.count_nonzero(locked))
    for key in _GUIDANCE_LOCK_ZERO_KEYS:
        arr = sanitized.get(key)
        if arr is None:
            continue
        out = np.asarray(arr)
        changed = int(np.count_nonzero(out[locked] != 0)) if out.size else 0
        out[locked] = 0
        sanitized[key] = out
        receipt[key] = {"policy": "zero_on_locked", "masked_pixels": masked_pixels, "changed_pixels": changed}

    for key in _GUIDANCE_LOCK_NAN_KEYS:
        arr = sanitized.get(key)
        if arr is None:
            continue
        out = np.asarray(arr)
        if not np.issubdtype(out.dtype, np.floating):
            out = out.astype(np.float32, copy=False)
        finite_locked = locked & np.isfinite(np.asarray(out, dtype=np.float32))
        changed = int(np.count_nonzero(finite_locked))
        out[locked] = np.nan
        sanitized[key] = out
        receipt[key] = {"policy": "nan_on_locked", "masked_pixels": masked_pixels, "changed_pixels": changed}

    for key in ("river_bank_influence", "river_bank_elevation_xs"):
        arr = sanitized.get(key)
        if arr is None:
            continue
        out = np.asarray(arr)
        finite_or_active = int(np.count_nonzero(np.isfinite(out))) if np.issubdtype(out.dtype, np.floating) else int(np.count_nonzero(out))
        receipt[key] = {
            "policy": "preserve_for_upstream_bank_guidance",
            "masked_pixels": masked_pixels,
            "finite_or_active_pixels": finite_or_active,
        }

    return {"arrays": sanitized, "receipt": receipt}


def _disable_empty_direct_primary_handoff(arrays: dict) -> dict:
    """Disable direct-primary routing when lock exclusion removes every usable pixel.

    River v2 writes several river artifacts, but only an *active* direct-primary
    surface should participate in final conditioning. After authoritative-lock
    exclusion, a surface can become semantically empty even if the source raster on
    disk was non-empty. Clearing it here keeps the route decision tied to the actual
    arrays entering terrain conditioning, which is simpler and easier to debug.
    """
    receipt = {
        "present_after_authoritative_exclusion": False,
        "finite_pixels_after_authoritative_exclusion": 0,
        "disabled": False,
        "reason": "no_primary_surface",
    }
    primary = arrays.get("primary_river_guidance_surface")
    if primary is None:
        return receipt
    primary_arr = np.asarray(primary, dtype=np.float32)
    finite_after = int(np.count_nonzero(np.isfinite(primary_arr)))
    receipt.update({
        "present_after_authoritative_exclusion": True,
        "finite_pixels_after_authoritative_exclusion": finite_after,
        "reason": "usable_primary_surface_retained" if finite_after > 0 else "no_unlocked_primary_surface_after_authoritative_exclusion",
    })
    if finite_after <= 0:
        arrays["primary_river_guidance_surface"] = None
        receipt["disabled"] = True
    return receipt


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

    canonical_template_path = Path(paths.template_path).resolve()
    authoritative_template_path = Path(paths.auth_src).resolve()
    if canonical_template_path != authoritative_template_path:
        raise RuntimeError(
            f"final-route canonical template drift: template_path={canonical_template_path} authoritative_base={authoritative_template_path}"
        )

    with rasterio.open(paths.template_path) as cand_ds:
        profile = cand_ds.profile.copy()
        nodata = cand_ds.nodata
        if nodata is None:
            nodata = -9999.0

        alignment_notes: dict[str, dict[str, object]] = {
            "canonical_final_grid": {
                "template_path": str(canonical_template_path),
                "authoritative_base": str(authoritative_template_path),
                "matches_authoritative_base": True,
            }
        }

        def _align(src_path: Optional[Path], *, dtype: str = "float32", nodata_value: float | int = -9999.0, resampling=Resampling.nearest, role: Optional[str] = None):
            if src_path is None or (not Path(src_path).exists()):
                if role:
                    alignment_notes[role] = {"path": str(src_path) if src_path is not None else None, "present": False}
                return None
            src_path = Path(src_path)
            arr = np.full((cand_ds.height, cand_ds.width), nodata_value, dtype=dtype)
            with rasterio.open(src_path) as src:
                if _same_grid(src, cand_ds):
                    arr, exact_nodata = _read_aligned_exact(src, dtype=dtype, nodata_value=nodata_value)
                    if role:
                        alignment_notes[role] = {
                            "path": str(src_path),
                            "present": True,
                            "mode": "direct_read_exact_grid",
                            "resampling": "none",
                            "source_nodata": None if exact_nodata is None else float(exact_nodata),
                        }
                    return arr
                src_band = src.read(1, masked=False)
                src_prepared, src_prepared_nodata = prepare_array_for_reproject(
                    src_band,
                    src.nodata,
                    dtype='float32',
                    default_nodata=-9999.0,
                )
                reproject(
                    source=src_prepared,
                    destination=arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=cand_ds.transform,
                    dst_crs=cand_ds.crs,
                    resampling=resampling,
                    src_nodata=src_prepared_nodata,
                    dst_nodata=nodata_value,
                )
                if role:
                    alignment_notes[role] = {
                        "path": str(src_path),
                        "present": True,
                        "mode": "reproject",
                        "resampling": str(getattr(resampling, "name", resampling)).lower(),
                        "source_nodata": None if src_prepared_nodata is None else float(src_prepared_nodata),
                    }
            if np.issubdtype(np.dtype(dtype), np.floating):
                arr = sanitize_for_output(arr, nodata=nodata_value, dtype=dtype)
            return arr

        auth = _align(paths.auth_src, dtype="float32", nodata_value=np.float32(nodata), resampling=Resampling.nearest, role="authoritative_base")
        if auth is None:
            raise RuntimeError("failed to align authoritative base to final-route template")
        auth = auth.astype("float32")
        auth[np.isclose(auth, np.float32(nodata))] = np.nan

        legacy_candidate = _align(candidate_path, dtype="float32", nodata_value=np.float32(nodata), resampling=Resampling.bilinear, role="legacy_candidate")
        if legacy_candidate is not None:
            legacy_candidate = legacy_candidate.astype("float32")
            legacy_candidate[np.isclose(legacy_candidate, np.float32(nodata))] = np.nan

        baseline_cudem_path = getattr(paths, "baseline_cudem_src", None) or resolve_baseline_cudem_interpolation(cfg=cfg, report=report, auth_src=paths.auth_src)
        baseline_background = _align(baseline_cudem_path, dtype="float32", nodata_value=np.float32(nodata), resampling=Resampling.nearest, role="baseline_cudem_interpolation") if baseline_cudem_path is not None else None
        if baseline_background is not None:
            baseline_background = baseline_background.astype("float32")
            baseline_background[np.isclose(baseline_background, np.float32(nodata))] = np.nan

        primary_river_guidance_surface_path = resolve_existing_output_path(river_outputs, "primary_river_guidance_surface", base_dir=outputs_base_dir)
        if primary_river_guidance_surface_path is None:
            primary_river_guidance_surface_path = resolve_existing_output_path(river_outputs, "river_primary_surface", base_dir=outputs_base_dir)
        
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
            "river_support": _align(resolve_existing_output_path(river_outputs, "authoritative_support", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.nearest),
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
            "primary_river_guidance_surface": _align(primary_river_guidance_surface_path, dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_channel_surface": _align(resolve_existing_output_path(river_outputs, "channel_surface", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_channel_surface_confidence": _align(resolve_existing_output_path(river_outputs, "channel_surface_confidence", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_channel_surface_source_class": _align(resolve_existing_output_path(river_outputs, "channel_surface_source_class", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_channel_surface_support_count": _align(resolve_existing_output_path(river_outputs, "channel_surface_support_count", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_channel_surface_authoritative_lock_scope": _align(resolve_existing_output_path(river_outputs, "channel_surface_authoritative_lock_scope", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_channel_surface_authoritative_lock_applied": _align(resolve_existing_output_path(river_outputs, "channel_surface_authoritative_lock_applied", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_channel_surface_prediction_support_confidence": _align(resolve_existing_output_path(river_outputs, "channel_surface_prediction_support_confidence", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_channel_surface_measured_anchor_fraction": _align(resolve_existing_output_path(river_outputs, "channel_surface_measured_anchor_fraction", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_channel_surface_structure_only_fraction": _align(resolve_existing_output_path(river_outputs, "channel_surface_structure_only_fraction", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_channel_surface_low_support_caution": _align(resolve_existing_output_path(river_outputs, "channel_surface_low_support_caution", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_channel_surface_prediction_admissibility": _align(resolve_existing_output_path(river_outputs, "channel_surface_prediction_admissibility", base_dir=outputs_base_dir), dtype="uint8", nodata_value=0, resampling=Resampling.nearest),
            "river_longitudinal_profile_elevation": _align(resolve_existing_output_path(river_outputs, "longitudinal_profile_elevation", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_longitudinal_profile_uncertainty": _align(resolve_existing_output_path(river_outputs, "longitudinal_profile_uncertainty", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_longitudinal_profile_influence": _align(resolve_existing_output_path(river_outputs, "longitudinal_profile_influence", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_longitudinal_profile_local_authoritative_reconciliation": _align(resolve_existing_output_path(river_outputs, "longitudinal_profile_local_authoritative_reconciliation", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_longitudinal_profile_local_authoritative_reconciliation_influence": _align(resolve_existing_output_path(river_outputs, "longitudinal_profile_local_authoritative_reconciliation_influence", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_xs_support_elevation": _align(resolve_existing_output_path(river_outputs, "xs_support_elevation", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
            "river_xs_support_weight": _align(resolve_existing_output_path(river_outputs, "xs_support_weight", base_dir=outputs_base_dir), dtype="float32", nodata_value=np.nan, resampling=Resampling.bilinear),
        }
        if arrays["sdb_candidate"] is not None:
            arrays["sdb_candidate"] = arrays["sdb_candidate"].astype("float32")
            arrays["sdb_candidate"][np.isclose(arrays["sdb_candidate"], np.float32(nodata))] = np.nan

        sdb_guide_points_path_active = sdb_guide_points_path if sdb_guide_points_path and Path(sdb_guide_points_path).exists() else None
        river_guide_points_path_active = resolve_existing_output_path(river_outputs, "guide_points", base_dir=outputs_base_dir)

        auth, legacy_candidate, baseline_background, semantics_harmonization = _maybe_harmonize_authoritative_semantics(
            auth=auth,
            legacy_candidate=legacy_candidate,
            baseline_background=baseline_background,
            river_support_depth=arrays.get("river_support_depth"),
            primary_river_guidance_surface=arrays.get("primary_river_guidance_surface"),
            logger=log,
        )

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
                "background_surface_role": "baseline_cudem_interpolation_outside_river_domain" if baseline_background is not None else "none_available",
                "active_final_route_mode": "direct_guidance_artifacts_only_with_cudem_background" if baseline_background is not None else "direct_guidance_artifacts_only",
                "sdb_guidance_source": "native_sparse_guide_points" if sdb_guide_points_path_active is not None else "no_sparse_guide_points_available",
                "river_guidance_source": "native_structured_sparse_guide_points" if river_guide_points_path_active is not None else "no_structured_sparse_guide_points_available",
            },
            "provenance_codes": {"0": "not_used_in_active_final_route"},
        }
        candidate_prov = source_candidate["provenance"].astype("uint8")
        pixel_size_m = grid_pixel_size_m(cand_ds.transform, cand_ds.crs, ref_lat_deg=(cfg.tile_bbox[1] + cfg.tile_bbox[3]) / 2.0 if getattr(cfg, "tile_bbox", None) else None)

    authoritative_locked = np.isfinite(auth)
    river_corridor = arrays.get("river_corridor")
    river_support = arrays.get("river_support")
    authoritative_lock_semantics = {
        "base_policy": "finite_authoritative_base_cells_locked",
        "river_corridor_policy": "authoritative_base_only",
    }
    if river_corridor is not None and river_support is not None:
        river_corridor_mask = np.asarray(river_corridor) > 0
        river_support_mask = np.isfinite(np.asarray(river_support, dtype=np.float32))
        if river_corridor_mask.shape == authoritative_locked.shape and river_support_mask.shape == authoritative_locked.shape:
            authoritative_locked = authoritative_locked.copy()
            authoritative_locked[river_corridor_mask] = river_support_mask[river_corridor_mask]
            authoritative_lock_semantics.update({
                "river_corridor_policy": "explicit_river_authoritative_support_overrides_finite_authoritative_base",
                "river_corridor_pixels": int(np.count_nonzero(river_corridor_mask)),
                "river_support_locked_pixels_in_corridor": int(np.count_nonzero(river_support_mask & river_corridor_mask)),
            })

    # Validate structural guidance semantics *before* authoritative-lock masking.
    # In authoritative-dense AOIs the final-route stage intentionally blanks
    # guidance over locked cells, but those masked arrays should not be used to
    # decide whether the upstream river/SDB guidance artifacts were constructed
    # correctly. Otherwise a truthful, finite centerline/bank/station raster can
    # become semantically empty only because every corridor pixel is locked.
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

    locked_guidance_exclusion = _exclude_guidance_from_authoritative_locked_cells(arrays, authoritative_locked)
    arrays = locked_guidance_exclusion["arrays"]
    direct_primary_handoff = _disable_empty_direct_primary_handoff(arrays)

    workflow_name = str(
        getattr(cfg, "river_workflow", None)
        or getattr(cfg, "workflow_name", None)
        or ""
    ).strip().lower()
    require_explicit_bank_guidance = workflow_name in {"river_workflow", "shared_solve"}

    support_params = {
        "pixel_size_m": pixel_size_m,
        "support_decay_m": float(getattr(cfg, "authoritative_support_decay_m", 300.0) or 300.0),
        "support_density_radius_m": float(getattr(cfg, "authoritative_support_density_radius_m", 250.0) or 250.0),
        "coastal_sdb_support_transition_m": float(getattr(cfg, "coastal_sdb_support_transition_m", 600.0) or 600.0),
        "river_anchor_density_radius_m": float(getattr(cfg, "river_anchor_density_radius_m", 200.0) or 200.0),
        "river_scaffold_transition_m": float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0),
        "require_explicit_bank_guidance": require_explicit_bank_guidance,
    }

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
        "river_bank_influence_active_pixels_after_handoff": int(np.count_nonzero(np.clip(np.nan_to_num(arrays["river_bank_influence"], nan=0.0), 0.0, 1.0) > 0.05)) if arrays["river_bank_influence"] is not None else 0,
        "river_bank_elevation_finite_pixels_after_handoff": int(np.count_nonzero(np.isfinite(arrays["river_bank_elevation_xs"]))) if arrays["river_bank_elevation_xs"] is not None else 0,
        "river_centerline_elevation": bool(arrays["river_centerline_elevation"] is not None),
        "river_channel_surface": bool(river_requirements.get("present_optional", {}).get("channel_surface", False)),
        "river_channel_surface_confidence": bool(river_requirements.get("present_optional", {}).get("channel_surface_confidence", False)),
        "river_channel_surface_source_class": bool(river_requirements.get("present_optional", {}).get("channel_surface_source_class", False)),
        "river_channel_surface_support_count": bool(river_requirements.get("present_optional", {}).get("channel_surface_support_count", False)),
        "river_longitudinal_profile_elevation": bool(arrays["river_longitudinal_profile_elevation"] is not None),
        "river_longitudinal_profile_local_authoritative_reconciliation": bool(arrays["river_longitudinal_profile_local_authoritative_reconciliation"] is not None),
        "river_longitudinal_profile_local_authoritative_reconciliation_influence": bool(arrays["river_longitudinal_profile_local_authoritative_reconciliation_influence"] is not None),
        "river_centerline_influence": bool(arrays["river_centerline_influence"] is not None),
        "river_centerline_stationing": bool(arrays["river_centerline_stationing"] is not None),
        "river_xs_support_elevation": bool(river_requirements.get("present_optional", {}).get("xs_support_elevation", False)),
        "river_xs_support_weight": bool(river_requirements.get("present_optional", {}).get("xs_support_weight", False)),
    }
    diagnostic_artifacts = {
        "legacy_candidate": str(candidate_path) if candidate_path else None,
        "baseline_cudem_interpolation": str(baseline_cudem_path) if baseline_cudem_path else None,
        "dense_sdb_depth_raster": str(sdb_depth_path) if sdb_depth_path else None,
        "dense_river_depth_raster": str(river_depth_path) if river_depth_path else None,
    }
    write_json_receipt(paths.guidance_receipt_path, {
        "stage": "guidance_assembly",
        "route_mode": "staged_final_route_single_source_of_truth",
        "structural_artifacts": {**structural_artifacts, "baseline_cudem_interpolation": str(baseline_cudem_path) if baseline_cudem_path else None},
        "diagnostic_only_artifacts": diagnostic_artifacts,
        "support_params": support_params,
        "pixel_size_m": pixel_size_m,
        "sdb_requirements": sdb_requirements,
        "river_requirements": river_requirements,
        "structural_alignment": alignment_notes,
        "authoritative_locked_guidance_exclusion": {
            "locked_pixels": int(np.count_nonzero(authoritative_locked)),
            "lock_semantics": authoritative_lock_semantics,
            "arrays": locked_guidance_exclusion["receipt"],
        },
        "river_direct_primary_handoff": direct_primary_handoff,
    })
    return GuidanceAssembly(
        profile=profile,
        nodata=float(nodata),
        auth=auth,
        legacy_candidate=legacy_candidate,
        baseline_cudem_path=baseline_cudem_path,
        baseline_background=baseline_background,
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
