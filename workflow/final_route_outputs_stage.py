from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import rasterio

from nodata_utils import fill_output_nodata, sanitize_for_output, array_valid_mask, collect_array_nodata_audit
from raster_contract import validate_gdal_output, validate_raster_mask_values

from precedence_audit import summarize_precedence_audit, write_precedence_audit
from conditioning_audit import write_conditioning_audit
from final_guidance_uncertainty_contract import write_guidance_uncertainty_contract
from final_output_layer_contract import write_final_output_layer_contract
from final_vertical_semantics_contract import write_final_vertical_semantics_contract
from final_dem_policy import build_final_dem_policy_dict
from final_route_contract import validate_final_route_contract
from final_route_receipts import write_json_receipt
from legacy_cleanup_stage import write_legacy_cleanup_receipt

from support_classes import SUPPORT_CLASS_CODE_TO_NAME, REGIME_CLASS_CODE_TO_NAME, SupportClass
from river_primary_surface_contract import RIVER_PRIMARY_SURFACE_SOURCE_NAMES, write_river_primary_surface_contract



def _append_dem_enhanced_touch(debug_path: Path, *, action: str, path: Path, source: Path | None = None, note: str | None = None) -> None:
    payload = {
        "action": str(action),
        "path": str(path),
        "exists": bool(path.exists()),
        "source": str(source) if source is not None else None,
        "source_exists": bool(source.exists()) if source is not None else None,
        "note": note,
    }
    with debug_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _optional_fraction_array(arr: np.ndarray | None, *, shape: tuple[int, ...], default: float) -> np.ndarray:
    if arr is None:
        return np.full(shape, default, dtype=np.float32)
    return np.clip(np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=default), 0.0, 1.0).astype(np.float32)


def _write_u8_with_validation(path: Path, arr: np.ndarray, prof: dict, *, tags: dict | None = None, allowed_values: set[int] | None = None):
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.asarray(arr, dtype=np.uint8), 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items() if v is not None})
    validate_gdal_output(
        path,
        operation="final_route_write",
        expected_crs=prof.get("crs"),
        expected_nodata=float(prof.get("nodata")) if prof.get("nodata") is not None else None,
        expected_dtype="uint8",
    )
    if allowed_values is not None:
        validate_raster_mask_values(
            path,
            operation="final_route_write",
            allowed_values={int(v) for v in allowed_values},
            nodata=int(prof.get("nodata")) if prof.get("nodata") is not None else None,
        )


def _normalized_write_profile(profile: dict, *, dtype: str, nodata: float | int, count: int = 1) -> dict:
    out = profile.copy()
    out.update(dtype=dtype, nodata=nodata, count=count, compress="deflate")

    # Template-derived profiles can carry block settings that are only valid for tiled
    # GTiffs and only when block sizes are multiples of 16. Normalize them here so final
    # route writes do not inherit invalid creation options from an upstream source raster.
    bx = out.get("blockxsize", out.get("BLOCKXSIZE"))
    by = out.get("blockysize", out.get("BLOCKYSIZE"))

    def _valid_block(v):
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return False
        return iv > 0 and (iv % 16 == 0)

    valid_blocks = _valid_block(bx) and _valid_block(by)
    if valid_blocks:
        out["blockxsize"] = int(bx)
        out["blockysize"] = int(by)
        out.pop("BLOCKXSIZE", None)
        out.pop("BLOCKYSIZE", None)
        out["tiled"] = True
    else:
        out.pop("blockxsize", None)
        out.pop("blockysize", None)
        out.pop("BLOCKXSIZE", None)
        out.pop("BLOCKYSIZE", None)
        out.pop("tiled", None)
        out.pop("TILED", None)
    return out


def _normalize_authoritative_lock_state(*, locked: np.ndarray, auth: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    sanitized_locked = np.asarray(locked, dtype=bool) & np.isfinite(np.asarray(auth, dtype=np.float32))
    support_out = np.asarray(support, dtype=np.uint8).copy()
    stale_locked = support_out == np.uint8(int(SupportClass.AUTHORITATIVE_LOCKED))
    stale_outside = stale_locked & (~sanitized_locked)
    if np.any(stale_outside):
        support_out[stale_outside] = np.uint8(int(SupportClass.ANCHORED_INTERPOLATION))
    support_out[sanitized_locked] = np.uint8(int(SupportClass.AUTHORITATIVE_LOCKED))
    return sanitized_locked, support_out, {
        "locked_input_pixels": int(np.count_nonzero(np.asarray(locked, dtype=bool))),
        "locked_finite_pixels": int(np.count_nonzero(sanitized_locked)),
        "stale_locked_support_pixels_cleared": int(np.count_nonzero(stale_outside)),
    }


def _validate_written_authoritative_lock_contract(*, final_path: Path, aligned_auth_path: Path, support_path: Path, tolerance: float = 1.0e-6) -> dict:
    from final_dem_contract_validator import validate_written_final_dem_contract

    payload = validate_written_final_dem_contract(
        final_depth=final_path,
        aligned_authoritative_base=aligned_auth_path,
        support_class=support_path,
        atol=float(tolerance),
    )
    hard_lock = payload.get("authoritative_hard_lock", {}) if isinstance(payload.get("authoritative_hard_lock", {}), dict) else {}
    payload["tolerance_m"] = float(tolerance)
    payload["ok"] = bool(payload.get("validated") and (not payload.get("skipped")) and hard_lock.get("ok") is True)
    return payload




def _maybe_harmonize_conditioned_elevation(*, conditioned: np.ndarray, auth: np.ndarray, river_primary_surface: np.ndarray | None, support: np.ndarray | None, logger=None) -> tuple[np.ndarray, dict]:
    """Audit final DEM semantics without mutating the conditioned surface.

    The terrain stage already enforces authoritative locks before this write stage.
    Do not negate the conditioned DEM here based on heuristic overlap tests. That kind
    of write-time sign flip can silently turn a correct NAVD88 elevation surface into
    the wrong product even when the primary river surface and authoritative base are
    already aligned.
    """
    import logging

    log = logger or logging.getLogger(__name__)
    receipt = {
        "applied": False,
        "reason": "conditioned_elevation_preserved",
        "reference_source": None,
        "overlap_pixels": 0,
        "median_abs_diff_before_m": None,
        "median_abs_diff_after_m": None,
    }

    cond = np.asarray(conditioned, dtype=np.float32)
    auth_ref = np.asarray(auth, dtype=np.float32)
    m = np.isfinite(cond) & np.isfinite(auth_ref)

    if support is not None:
        support_arr = np.asarray(support, dtype=np.uint8)
        try:
            locked_mask = support_arr == np.uint8(int(SupportClass.AUTHORITATIVE_LOCKED))
            if np.count_nonzero(m & locked_mask) >= 100:
                m = m & locked_mask
                receipt["reference_source"] = "authoritative_locked_cells"
        except Exception:
            pass

    if receipt["reference_source"] is None and river_primary_surface is not None:
        rps = np.asarray(river_primary_surface, dtype=np.float32)
        mr = m & np.isfinite(rps)
        if np.count_nonzero(mr) >= 100:
            m = mr
            auth_ref = rps
            receipt["reference_source"] = "river_primary_surface"

    if receipt["reference_source"] is None:
        receipt["reference_source"] = "aligned_authoritative_base"

    overlap = int(np.count_nonzero(m))
    receipt["overlap_pixels"] = overlap
    if overlap < 100:
        receipt["reason"] = "insufficient_overlap"
        return cond, receipt

    before = np.abs(cond[m] - auth_ref[m])
    after = np.abs((-cond[m]) - auth_ref[m])
    med_before = float(np.nanmedian(before)) if before.size else None
    med_after = float(np.nanmedian(after)) if after.size else None
    receipt["median_abs_diff_before_m"] = med_before
    receipt["median_abs_diff_after_m"] = med_after

    if np.isfinite(med_before) and np.isfinite(med_after) and med_after + 0.5 < med_before:
        receipt["reason"] = "detected_possible_sign_mismatch_preserved_conditioned_surface"
        log.warning(
            "[FINAL_ROUTE][SEMANTICS] Detected possible conditioned/output sign mismatch (median abs diff %.3f -> %.3f m if negated), but preserved conditioned elevation semantics.",
            med_before,
            med_after,
        )
    else:
        receipt["reason"] = "already_aligned"
    return cond, receipt



def _stage_overlap_stats(a: np.ndarray, b: np.ndarray) -> dict:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    m = np.isfinite(aa) & np.isfinite(bb)
    n = int(np.count_nonzero(m))
    if n == 0:
        return {"overlap_pixels": 0, "mean_diff_m": None, "median_diff_m": None, "min_diff_m": None, "max_diff_m": None, "rmse_m": None}
    d = (aa[m] - bb[m]).astype(np.float64)
    return {
        "overlap_pixels": n,
        "mean_diff_m": float(np.mean(d)),
        "median_diff_m": float(np.median(d)),
        "min_diff_m": float(np.min(d)),
        "max_diff_m": float(np.max(d)),
        "rmse_m": float(np.sqrt(np.mean(np.square(d)))),
    }


def _resolve_support_points_path(report: dict, *, base_dir: Path) -> Path | None:
    river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
    candidates = [river_outputs.get("river_support_points"), river_outputs.get("support_points")]
    for cand in candidates:
        if not cand:
            continue
        try:
            p = Path(str(cand))
        except (TypeError, ValueError, OSError):
            continue
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return None


def _support_role_column(gdf) -> str | None:
    for name in ("support_role", "role", "point_role", "sample_role", "__role__", "support_class", "kind"):
        if name in gdf.columns:
            return name
    return None


def _support_semantics_column(gdf) -> str | None:
    for name in ("value_semantics", "vertical_semantics"):
        if name in gdf.columns:
            return name
    return None


def _sample_raster_at_points(raster_path: Path, gdf):
    with rasterio.open(raster_path) as ds:
        sgdf = gdf
        try:
            if getattr(sgdf, "crs", None) is not None and ds.crs is not None and sgdf.crs != ds.crs:
                sgdf = sgdf.to_crs(ds.crs)
        except Exception:
            pass
        coords = [(geom.x, geom.y) for geom in sgdf.geometry]
        samples = [v[0] if len(v) else np.nan for v in ds.sample(coords)]
        rows = []
        cols = []
        for geom in sgdf.geometry:
            try:
                r, c = ds.index(geom.x, geom.y)
            except Exception:
                r, c = -1, -1
            rows.append(int(r))
            cols.append(int(c))
    return np.asarray(samples, dtype=np.float32), np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def _write_stage_divergence_audit(*, debug_dir: Path, report: dict, base_dir: Path, stage_paths: dict[str, Path], stage_semantics: dict[str, dict], logger) -> dict:
    payload = {
        "stage_rasters": {k: str(v) for k, v in stage_paths.items()},
        "stage_pair_overlap_stats": {},
        "support_point_stage_divergence_summary_csv": None,
        "support_point_stage_divergence_cell_trace_csv": None,
        "support_point_stage_divergence_receipt_json": None,
        "stage_semantics_receipt_json": None,
        "support_points_path": None,
        "status": "skipped",
        "reason": None,
    }
    stage_names = list(stage_paths)
    for i in range(len(stage_names) - 1):
        a = stage_names[i]
        b = stage_names[i + 1]
        with rasterio.open(stage_paths[a]) as dsa, rasterio.open(stage_paths[b]) as dsb:
            aa = dsa.read(1).astype(np.float32)
            bb = dsb.read(1).astype(np.float32)
        payload["stage_pair_overlap_stats"][f"{a}__vs__{b}"] = _stage_overlap_stats(aa, bb)

    semantics_path = debug_dir / "stage_semantics_receipt.json"
    write_json_receipt(semantics_path, {"stage": "final_route_stage_semantics", "stages": stage_semantics})
    payload["stage_semantics_receipt_json"] = str(semantics_path)

    support_points_path = _resolve_support_points_path(report, base_dir=base_dir)
    if support_points_path is None:
        payload["status"] = "skipped"
        payload["reason"] = "support_points_not_found"
        receipt_path = debug_dir / "stage_divergence_receipt.json"
        write_json_receipt(receipt_path, payload)
        payload["support_point_stage_divergence_receipt_json"] = str(receipt_path)
        return payload
    payload["support_points_path"] = str(support_points_path)

    try:
        import geopandas as gpd
    except Exception as exc:
        payload["status"] = "skipped"
        payload["reason"] = f"geopandas_unavailable:{type(exc).__name__}"
        receipt_path = debug_dir / "stage_divergence_receipt.json"
        write_json_receipt(receipt_path, payload)
        payload["support_point_stage_divergence_receipt_json"] = str(receipt_path)
        return payload

    try:
        gdf = gpd.read_file(support_points_path)
    except Exception as exc:
        payload["status"] = "skipped"
        payload["reason"] = f"support_points_read_failed:{type(exc).__name__}"
        receipt_path = debug_dir / "stage_divergence_receipt.json"
        write_json_receipt(receipt_path, payload)
        payload["support_point_stage_divergence_receipt_json"] = str(receipt_path)
        return payload

    if gdf.empty or "geometry" not in gdf.columns:
        payload["status"] = "skipped"
        payload["reason"] = "support_points_empty"
        receipt_path = debug_dir / "stage_divergence_receipt.json"
        write_json_receipt(receipt_path, payload)
        payload["support_point_stage_divergence_receipt_json"] = str(receipt_path)
        return payload

    if "source" in gdf.columns:
        gdf = gdf[gdf["source"].astype(str) == "authoritative_base"].copy()
    semantics_col = _support_semantics_column(gdf)
    if semantics_col is not None:
        gdf = gdf[gdf[semantics_col].astype(str) == "absolute_elevation"].copy()
    value_col = None
    for cand in ("elevation", "z", "value", "bed_elevation_m", "support_z_m", "authoritative_value_m"):
        if cand in gdf.columns:
            value_col = cand
            break
    if value_col is None:
        payload["status"] = "skipped"
        payload["reason"] = "authoritative_value_column_missing"
        receipt_path = debug_dir / "stage_divergence_receipt.json"
        write_json_receipt(receipt_path, payload)
        payload["support_point_stage_divergence_receipt_json"] = str(receipt_path)
        return payload

    gdf = gdf[np.isfinite(gdf[value_col].to_numpy(dtype=float))].copy()
    if gdf.empty:
        payload["status"] = "skipped"
        payload["reason"] = "no_finite_authoritative_points"
        receipt_path = debug_dir / "stage_divergence_receipt.json"
        write_json_receipt(receipt_path, payload)
        payload["support_point_stage_divergence_receipt_json"] = str(receipt_path)
        return payload

    role_col = _support_role_column(gdf)
    if role_col is None:
        gdf["__role__"] = "unknown"
        role_col = "__role__"

    base_vals = gdf[value_col].to_numpy(dtype=np.float32)
    stage_samples = {}
    rows = cols = None
    for name, rp in stage_paths.items():
        vals, rr, cc = _sample_raster_at_points(rp, gdf)
        stage_samples[name] = vals
        if rows is None:
            rows, cols = rr, cc

    summary_rows = []
    trace_rows = []
    import pandas as pd
    sdf = pd.DataFrame({
        "role": gdf[role_col].astype(str).to_numpy(),
        "authoritative_value_m": base_vals.astype(np.float64),
        "row": rows,
        "col": cols,
    })
    for name, vals in stage_samples.items():
        sdf[name] = vals.astype(np.float64)
        sdf[f"{name}_diff_m"] = sdf[name] - sdf["authoritative_value_m"]

    roles = sorted(set(sdf["role"].astype(str)))
    for name in stage_names:
        for role in ["ALL"] + roles:
            sub = sdf if role == "ALL" else sdf[sdf["role"] == role]
            diff = sub[f"{name}_diff_m"].to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]
            if diff.size == 0:
                continue
            summary_rows.append({
                "stage": name,
                "role": role,
                "count": int(diff.size),
                "mean_diff_m": float(np.mean(diff)),
                "median_diff_m": float(np.median(diff)),
                "rmse_m": float(np.sqrt(np.mean(np.square(diff)))),
                "min_diff_m": float(np.min(diff)),
                "max_diff_m": float(np.max(diff)),
            })

    final_name = stage_names[-1]
    sort_idx = np.argsort(-np.abs(sdf[f"{final_name}_diff_m"].to_numpy(dtype=float)))
    for idx in sort_idx[:20]:
        row = {
            "point_index": int(idx),
            "role": str(sdf.iloc[idx]["role"]),
            "row": int(sdf.iloc[idx]["row"]),
            "col": int(sdf.iloc[idx]["col"]),
            "authoritative_value_m": float(sdf.iloc[idx]["authoritative_value_m"]),
        }
        geom = gdf.geometry.iloc[idx]
        row["x"] = float(geom.x)
        row["y"] = float(geom.y)
        for name in stage_names:
            row[name] = float(sdf.iloc[idx][name]) if np.isfinite(sdf.iloc[idx][name]) else None
            dv = sdf.iloc[idx][f"{name}_diff_m"]
            row[f"{name}_diff_m"] = float(dv) if np.isfinite(dv) else None
        trace_rows.append(row)

    summary_csv = debug_dir / "stage_divergence_summary.csv"
    trace_csv = debug_dir / "stage_divergence_cell_trace.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["stage", "role", "count", "mean_diff_m", "median_diff_m", "rmse_m", "min_diff_m", "max_diff_m"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    with trace_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(trace_rows[0].keys()) if trace_rows else ["point_index", "role", "row", "col", "x", "y", "authoritative_value_m"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in trace_rows:
            writer.writerow(row)

    receipt_path = debug_dir / "stage_divergence_receipt.json"
    write_json_receipt(receipt_path, {
        "stage": "final_route_stage_divergence_audit",
        "support_points_path": str(support_points_path),
        "support_point_count": int(len(sdf)),
        "support_role_column": role_col,
        "support_value_column": value_col,
        "stage_rasters": {k: str(v) for k, v in stage_paths.items()},
        "stage_pair_overlap_stats": payload["stage_pair_overlap_stats"],
        "summary_csv": str(summary_csv),
        "cell_trace_csv": str(trace_csv),
    })
    payload.update({
        "support_point_stage_divergence_summary_csv": str(summary_csv),
        "support_point_stage_divergence_cell_trace_csv": str(trace_csv),
        "support_point_stage_divergence_receipt_json": str(receipt_path),
        "status": "success",
        "reason": None,
    })
    return payload

def _repair_conditioned_invalid_cells(conditioned: np.ndarray, auth: np.ndarray, guidance_surface: np.ndarray) -> tuple[np.ndarray, int, str]:
    repaired = conditioned.astype(np.float32, copy=True)
    note_parts = []
    invalid = ~array_valid_mask(repaired, min_allowed=-1000.0, max_allowed=1000.0)
    if not np.any(invalid):
        return repaired, 0, "already_valid"

    auth_valid = array_valid_mask(auth, min_allowed=-1000.0, max_allowed=1000.0)
    take_auth = invalid & auth_valid
    if np.any(take_auth):
        repaired[take_auth] = auth[take_auth]
        note_parts.append(f"auth={int(np.count_nonzero(take_auth))}")
        invalid = ~array_valid_mask(repaired, min_allowed=-1000.0, max_allowed=1000.0)

    guidance_valid = array_valid_mask(guidance_surface, min_allowed=-1000.0, max_allowed=1000.0)
    take_guidance = invalid & guidance_valid
    if np.any(take_guidance):
        repaired[take_guidance] = guidance_surface[take_guidance]
        note_parts.append(f"guidance={int(np.count_nonzero(take_guidance))}")
        invalid = ~array_valid_mask(repaired, min_allowed=-1000.0, max_allowed=1000.0)

    if np.any(invalid):
        try:
            from scipy.ndimage import distance_transform_edt
            valid = array_valid_mask(repaired, min_allowed=-1000.0, max_allowed=1000.0)
            if np.any(valid):
                _, nearest = distance_transform_edt(~valid, return_indices=True)
                repaired[invalid] = repaired[nearest[0][invalid], nearest[1][invalid]]
                note_parts.append(f"nearest={int(np.count_nonzero(invalid))}")
                invalid = ~array_valid_mask(repaired, min_allowed=-1000.0, max_allowed=1000.0)
        except Exception:
            pass
    return repaired, int(np.count_nonzero(invalid)), ";".join(note_parts) or "no_repair"


def _validate_runtime_authoritative_output_state(*, conditioned: np.ndarray, auth: np.ndarray, support: np.ndarray, provenance: np.ndarray, guidance_influence: np.ndarray, locked: np.ndarray) -> tuple[bool, dict]:
    from provenance_schema import ProvenanceClass

    locked_valid = np.asarray(locked, dtype=bool) & np.isfinite(np.asarray(auth, dtype=np.float32))
    if not np.any(locked_valid):
        return True, {
            "locked_finite_pixels": 0,
            "conditioned_mismatch_pixels": 0,
            "support_mismatch_pixels": 0,
            "provenance_mismatch_pixels": 0,
            "guidance_influence_mismatch_pixels": 0,
            "ok": True,
            "policy": "final_route_validates_runtime_authoritative_state_without_semantic_repair",
        }

    auth_f32 = np.asarray(auth, dtype=np.float32)
    conditioned_f32 = np.asarray(conditioned, dtype=np.float32)
    support_u8 = np.asarray(support, dtype=np.uint8)
    provenance_u8 = np.asarray(provenance, dtype=np.uint8)
    guidance_f32 = np.asarray(guidance_influence, dtype=np.float32)
    conditioned_match = np.isfinite(conditioned_f32) & np.isclose(conditioned_f32, auth_f32, atol=1.0e-6, rtol=0.0)
    conditioned_mismatch = locked_valid & (~conditioned_match)
    support_mismatch = locked_valid & (support_u8 != np.uint8(int(SupportClass.AUTHORITATIVE_LOCKED)))
    provenance_mismatch = locked_valid & (provenance_u8 != np.uint8(int(ProvenanceClass.AUTHORITATIVE_LOCKED)))
    guidance_mismatch = locked_valid & (~np.isclose(guidance_f32, 0.0, atol=1.0e-6, rtol=0.0))
    receipt = {
        "locked_finite_pixels": int(np.count_nonzero(locked_valid)),
        "conditioned_mismatch_pixels": int(np.count_nonzero(conditioned_mismatch)),
        "support_mismatch_pixels": int(np.count_nonzero(support_mismatch)),
        "provenance_mismatch_pixels": int(np.count_nonzero(provenance_mismatch)),
        "guidance_influence_mismatch_pixels": int(np.count_nonzero(guidance_mismatch)),
        "ok": not (np.any(conditioned_mismatch) or np.any(support_mismatch) or np.any(provenance_mismatch) or np.any(guidance_mismatch)),
        "policy": "final_route_validates_runtime_authoritative_state_without_semantic_repair",
    }
    return bool(receipt["ok"]), receipt


def write_final_route_outputs(*, cfg, paths, guidance, terrain, candidate_path: Path | None, provenance_path: Path | None, report: dict) -> tuple[Path, Path, Path, Path, Path, Path]:
    result = terrain.result
    auth = guidance.auth
    profile = guidance.profile
    nodata = guidance.nodata
    source_candidate = terrain.source_candidate
    candidate_prov = terrain.candidate_prov

    locked = result["locked"]
    gap = result["gap"]
    eligible = result["eligible"]
    support = result["support"]
    regime = result["regime"]
    support_distance_m = result["support_distance_m"]
    support_density = result["support_density"]
    anchor_uncertainty = result["anchor_uncertainty"]
    guidance_uncertainty = result["guidance_uncertainty"]
    conditioned_uncertainty = result["conditioned_uncertainty"]
    guidance_influence = result["guidance_influence"]
    coastal_sdb_confidence = result["coastal_sdb_confidence"]
    river_anchor_distance_m = result["river_anchor_distance_m"]
    river_anchor_density = result["river_anchor_density"]
    river_scaffold_confidence = result["river_scaffold_confidence"]
    river_bank_distance_m = result["river_bank_distance_m"]
    river_bank_influence_runtime = result["river_bank_influence"]
    river_bank_elevation = result["river_bank_elevation"]
    river_primary_surface = np.asarray(result.get("river_primary_surface", np.full(auth.shape, np.nan, dtype=np.float32)), dtype=np.float32)
    river_primary_surface_confidence = np.clip(np.nan_to_num(np.asarray(result.get("river_primary_surface_confidence", np.zeros(auth.shape, dtype=np.float32)), dtype=np.float32), nan=0.0), 0.0, 1.0).astype(np.float32)
    river_primary_surface_source_class = np.asarray(result.get("river_primary_surface_source_class", np.zeros(auth.shape, dtype=np.uint8)), dtype=np.uint8)
    river_primary_surface_support_count = np.asarray(result.get("river_primary_surface_support_count", np.zeros(auth.shape, dtype=np.uint8)), dtype=np.uint8)
    river_primary_surface_domain = np.asarray(result.get("river_primary_surface_domain", np.zeros(auth.shape, dtype=np.uint8)), dtype=np.uint8)
    river_primary_surface_channel_core_preserve = np.asarray(result.get("river_primary_surface_channel_core_preserve", np.zeros(auth.shape, dtype=np.uint8)), dtype=np.uint8)
    river_channel_core_preservation_zone = np.asarray(result.get("river_channel_core_preservation_zone", np.zeros(auth.shape, dtype=np.uint8)), dtype=np.uint8)
    river_channel_core_prepost_delta = np.asarray(result.get("river_channel_core_prepost_delta", np.full(auth.shape, np.nan, dtype=np.float32)), dtype=np.float32)
    river_channel_core_bank_pull_risk = np.clip(np.nan_to_num(np.asarray(result.get("river_channel_core_bank_pull_risk", np.zeros(auth.shape, dtype=np.float32)), dtype=np.float32), nan=0.0), 0.0, 1.0).astype(np.float32)
    river_channel_core_preservation_receipt = result.get("river_channel_core_preservation_receipt") if isinstance(result.get("river_channel_core_preservation_receipt"), dict) else None
    river_primary_surface_contract = result.get("river_primary_surface_contract") if isinstance(result.get("river_primary_surface_contract"), dict) else None
    river_primary_guidance_summary = result.get("river_primary_guidance_summary") if isinstance(result.get("river_primary_guidance_summary"), dict) else None
    river_bank_continuity_weight_runtime = result.get("river_bank_continuity_weight")
    river_bank_graph_confidence_runtime = result.get("river_bank_graph_confidence")
    river_bank_confluence_damping_runtime = result.get("river_bank_confluence_damping")
    river_bank_estuary_side_decay_runtime = result.get("river_bank_estuary_side_decay")
    conditioned = result["conditioned"]
    conditioned_before_final_route = np.asarray(conditioned, dtype=np.float32).copy()
    prov_out = result["provenance"]
    support_note = result["support_note"]
    authoritative_first_contract = result.get("authoritative_first_contract") if isinstance(result.get("authoritative_first_contract"), dict) else None

    prof_f32 = _normalized_write_profile(profile, dtype="float32", nodata=float(nodata), count=1)
    prof_u8 = _normalized_write_profile(profile, dtype="uint8", nodata=0, count=1)

    vertical_datum = str(getattr(cfg, "output_vdatum", None) or getattr(cfg, "target_vcrs", None) or getattr(cfg, "working_vcrs", None) or "NAVD88")
    vertical_datum_epsg = str(getattr(cfg, "output_vdatum_epsg", None) or getattr(cfg, "target_vdatum_epsg", None) or getattr(cfg, "working_vdatum_epsg", None) or "5703")
    log = getattr(cfg, "logger", None) or logging.getLogger(__name__)

    def _write(path: Path, arr: np.ndarray, prof: dict, *, tags: dict | None = None):
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(arr, 1)
            if tags:
                dst.update_tags(**{k: str(v) for k, v in tags.items() if v is not None})

    def _write_float(path: Path, arr: np.ndarray, prof: dict, *, tags: dict | None = None, min_allowed: float | None = None, max_allowed: float | None = None):
        out = fill_output_nodata(arr, nodata=float(prof.get("nodata", nodata)), dtype=np.float32, min_allowed=min_allowed, max_allowed=max_allowed)
        _write(path, out.astype("float32"), prof, tags=tags)
        validate_gdal_output(path, operation="final_route_write", expected_crs=prof.get("crs"), expected_nodata=float(prof.get("nodata")), expected_dtype="float32", min_allowed=min_allowed, max_allowed=max_allowed)
        return out

    debug_dir = paths.combined_dir / "debug_final_route"
    debug_dir.mkdir(parents=True, exist_ok=True)

    auth = sanitize_for_output(auth, dtype=np.float32)
    conditioned = sanitize_for_output(conditioned, dtype=np.float32, min_allowed=-1000.0, max_allowed=1000.0)
    locked, support, lock_state_receipt = _normalize_authoritative_lock_state(
        locked=locked,
        auth=auth,
        support=support,
    )
    guidance_surface = result.get("guidance_surface")
    if guidance_surface is None:
        guidance_surface = np.full(auth.shape, np.nan, dtype=np.float32)
    else:
        guidance_surface = sanitize_for_output(guidance_surface, dtype=np.float32)

    conditioned, unresolved_conditioned, conditioned_repair_note = _repair_conditioned_invalid_cells(conditioned, auth, guidance_surface)
    if unresolved_conditioned:
        raise RuntimeError(f"authoritative conditioning produced invalid final DEM cells after sanitization: {unresolved_conditioned}")

    if authoritative_first_contract is None:
        raise RuntimeError("terrain stage did not provide authoritative-first contract payload to final route stage")
    if not authoritative_first_contract.get("ok", False):
        raise RuntimeError(f"terrain stage authoritative-first contract failed before final route stage: {authoritative_first_contract}")
    runtime_authoritative_ok, runtime_authoritative_validation = _validate_runtime_authoritative_output_state(
        conditioned=conditioned,
        auth=auth,
        support=support,
        provenance=prov_out,
        guidance_influence=guidance_influence,
        locked=locked,
    )
    if not runtime_authoritative_ok:
        raise RuntimeError(
            "final-route inputs violate authoritative-lock semantics before any write-time validation: "
            f"conditioned_mismatch_pixels={int(runtime_authoritative_validation.get('conditioned_mismatch_pixels', 0) or 0)} "
            f"support_mismatch_pixels={int(runtime_authoritative_validation.get('support_mismatch_pixels', 0) or 0)} "
            f"provenance_mismatch_pixels={int(runtime_authoritative_validation.get('provenance_mismatch_pixels', 0) or 0)} "
            f"guidance_influence_mismatch_pixels={int(runtime_authoritative_validation.get('guidance_influence_mismatch_pixels', 0) or 0)}"
        )

    conditioned, conditioned_semantics_harmonization = _maybe_harmonize_conditioned_elevation(
        conditioned=conditioned,
        auth=auth,
        river_primary_surface=river_primary_surface,
        support=support,
        logger=log,
    )

    auth_out = fill_output_nodata(auth, nodata=float(nodata), dtype=np.float32)
    cond_after_harmonize = np.asarray(conditioned, dtype=np.float32).copy()
    cond_out = fill_output_nodata(conditioned, nodata=float(nodata), dtype=np.float32)

    _write_float(paths.aligned_auth_path, auth_out, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "aligned_authoritative_base"})
    _write_u8_with_validation(paths.gap_mask_path, gap.astype("uint8"), prof_u8, allowed_values={0, 1})
    _write_u8_with_validation(paths.eligible_mask_path, eligible.astype("uint8"), prof_u8, allowed_values={0, 1})
    _write_u8_with_validation(paths.support_class_path, support.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "support_class"}, allowed_values=set(SUPPORT_CLASS_CODE_TO_NAME))
    _write_u8_with_validation(paths.regime_class_path, regime.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "regime_class"}, allowed_values=set(REGIME_CLASS_CODE_TO_NAME))
    _write_float(paths.source_candidate_path, guidance_surface, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "guidance_surface_diagnostic"})
    _write_u8_with_validation(paths.source_candidate_prov_path, candidate_prov.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "guidance_surface_provenance"})
    _write_float(paths.support_distance_path, support_distance_m, prof_f32, tags={"VALUE_TYPE": "distance", "UNITS": "meters", "ROLE": "support_distance"})
    _write_float(paths.support_density_path, np.clip(support_density, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "support_density"})
    _write_float(paths.anchor_uncertainty_path, anchor_uncertainty, prof_f32, tags={"VALUE_TYPE": "uncertainty", "UNITS": "meters", "ROLE": "anchor_uncertainty"})
    _write_float(paths.guidance_uncertainty_path, guidance_uncertainty, prof_f32, tags={"VALUE_TYPE": "uncertainty", "UNITS": "meters", "ROLE": "guidance_uncertainty"})
    _write_float(paths.conditioned_uncertainty_path, conditioned_uncertainty, prof_f32, tags={"VALUE_TYPE": "uncertainty", "UNITS": "meters", "ROLE": "conditioned_uncertainty"})
    _write_float(paths.guidance_influence_path, np.clip(guidance_influence, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "guidance_influence"})
    _write_float(paths.coastal_sdb_confidence_path, np.clip(coastal_sdb_confidence, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "coastal_sdb_confidence"})
    _write_float(paths.river_anchor_distance_path, river_anchor_distance_m, prof_f32, tags={"VALUE_TYPE": "distance", "UNITS": "meters", "ROLE": "river_anchor_distance"})
    _write_float(paths.river_anchor_density_path, np.clip(river_anchor_density, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_anchor_density"})
    _write_float(paths.river_scaffold_confidence_path, np.clip(river_scaffold_confidence, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_scaffold_confidence"})
    _write_float(paths.river_bank_distance_path, river_bank_distance_m, prof_f32, tags={"VALUE_TYPE": "distance", "UNITS": "meters", "ROLE": "river_bank_distance"})
    _write_float(paths.river_bank_influence_runtime_path, np.clip(river_bank_influence_runtime, 0.0, 1.0).astype("float32"), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_influence"})
    _write_float(paths.river_bank_elevation_path, river_bank_elevation, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "river_bank_elevation_guidance"})
    if river_primary_surface_contract is None:
        raise RuntimeError("river primary surface contract payload missing from terrain stage result")
    write_river_primary_surface_contract(paths.river_primary_surface_contract_path, river_primary_surface_contract)
    if river_primary_guidance_summary is None:
        metrics = river_primary_surface_contract.get("metrics", {}) if isinstance(river_primary_surface_contract.get("metrics", {}), dict) else {}
        source_counts = metrics.get("source_class_counts", {}) if isinstance(metrics.get("source_class_counts", {}), dict) else {}
        dominant_source = max(source_counts.items(), key=lambda kv: (int(kv[1] or 0), str(kv[0])))[0] if source_counts else "none"
        river_primary_guidance_summary = {
            "active_product_name": "river_primary_surface",
            "active_product_role": "primary",
            "diagnostic_guidance_surface_name": "guidance_surface",
            "diagnostic_guidance_surface_role": "diagnostic_only",
            "primary_builder_mode": ("channel_surface_scaffold" if int(source_counts.get("channel_surface_scaffold", 0) or 0) > 0 else ("backbone_fallback" if float(metrics.get("backbone_fraction", 0.0) or 0.0) > 0.0 else "no_primary_surface")),
            "dominant_primary_source_class": str(dominant_source),
            "primary_surface_contract_ok": bool(river_primary_surface_contract.get("ok", False)),
            "primary_surface_domain_pixels": int(metrics.get("domain_pixels", 0) or 0),
            "primary_surface_finite_pixels": int(metrics.get("finite_pixels", 0) or 0),
            "primary_surface_coverage_fraction": float(metrics.get("coverage_fraction", 0.0) or 0.0),
            "primary_source_class_counts": {str(k): int(v) for k, v in source_counts.items()},
            "backbone_fraction": float(metrics.get("backbone_fraction", 0.0) or 0.0),
            "continuity_safeguard_used": False,
            "continuity_safeguard_labels": [],
            "degraded_mode_active": False,
            "degraded_mode_reasons": [],
            "support_note_tokens": [],
        }
    write_json_receipt(paths.river_primary_guidance_summary_path, river_primary_guidance_summary)
    if not river_primary_surface_contract.get("ok", False):
        raise RuntimeError(f"river primary surface contract failed before final DEM stage: {river_primary_surface_contract}")
    _write_float(paths.river_primary_surface_path, river_primary_surface, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "river_primary_surface"})
    _write_float(paths.river_primary_surface_confidence_path, river_primary_surface_confidence, {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_primary_surface_confidence"})
    _write_u8_with_validation(paths.river_primary_surface_source_class_path, river_primary_surface_source_class.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "river_primary_surface_source_class"}, allowed_values=set(RIVER_PRIMARY_SURFACE_SOURCE_NAMES))
    _write_u8_with_validation(paths.river_primary_surface_support_count_path, river_primary_surface_support_count.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "count", "UNITS": "contributors", "ROLE": "river_primary_surface_support_count"})
    _write_u8_with_validation(paths.river_primary_surface_domain_path, river_primary_surface_domain.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "mask", "UNITS": "binary", "ROLE": "river_primary_surface_domain"}, allowed_values={0,1})
    _write_u8_with_validation(paths.river_primary_surface_channel_core_preserve_path, river_primary_surface_channel_core_preserve.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "mask", "UNITS": "binary", "ROLE": "river_primary_surface_channel_core_preserve"}, allowed_values={0,1})
    _write_u8_with_validation(paths.river_channel_core_preservation_zone_path, river_channel_core_preservation_zone.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "mask", "UNITS": "binary", "ROLE": "river_channel_core_preservation_zone"}, allowed_values={0,1})
    _write_float(paths.river_channel_core_prepost_delta_path, river_channel_core_prepost_delta, prof_f32, tags={"VALUE_TYPE": "difference", "UNITS": "meters", "ROLE": "river_channel_core_prepost_delta"}, min_allowed=-1000.0, max_allowed=1000.0)
    _write_float(paths.river_channel_core_bank_pull_risk_path, river_channel_core_bank_pull_risk, {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_channel_core_bank_pull_risk"}, min_allowed=0.0, max_allowed=1.0)
    river_bank_continuity_runtime_path = paths.combined_dir / "river_bank_continuity_weight.tif"
    river_bank_graph_confidence_runtime_path = paths.combined_dir / "river_bank_graph_confidence.tif"
    river_bank_confluence_damping_runtime_path = paths.combined_dir / "river_bank_confluence_damping.tif"
    river_bank_estuary_side_decay_runtime_path = paths.combined_dir / "river_bank_estuary_side_decay.tif"
    _write_float(river_bank_continuity_runtime_path, _optional_fraction_array(river_bank_continuity_weight_runtime, shape=auth.shape, default=0.0), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_continuity_weight"})
    _write_float(river_bank_graph_confidence_runtime_path, _optional_fraction_array(river_bank_graph_confidence_runtime, shape=auth.shape, default=0.0), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_graph_confidence"})
    _write_float(river_bank_confluence_damping_runtime_path, _optional_fraction_array(river_bank_confluence_damping_runtime, shape=auth.shape, default=1.0), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_confluence_damping"})
    _write_float(river_bank_estuary_side_decay_runtime_path, _optional_fraction_array(river_bank_estuary_side_decay_runtime, shape=auth.shape, default=1.0), {**prof_f32, "nodata": -9999.0}, tags={"VALUE_TYPE": "fraction", "UNITS": "unitless", "ROLE": "river_bank_estuary_side_decay"})
    conditioned_internal_path = paths.combined_dir / "conditioned_final_dem_internal.tif"
    _write_float(conditioned_internal_path, conditioned, prof_f32, min_allowed=-1000.0, max_allowed=1000.0, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "conditioned_final_dem_internal"})
    shutil.copy2(conditioned_internal_path, paths.conditioned_path)
    validate_gdal_output(paths.conditioned_path, operation="final_route_write_copy", expected_crs=prof_f32.get("crs"), expected_nodata=float(prof_f32.get("nodata")), expected_dtype="float32", min_allowed=-1000.0, max_allowed=1000.0)
    with rasterio.open(paths.conditioned_path, "r+") as dst:
        dst.update_tags(VALUE_TYPE="elevation", UNITS="meters", SIGN_CONVENTION="relative_to_datum", VERTICAL_SEMANTICS="absolute_elevation", VERTICAL_DATUM=str(vertical_datum), VERTICAL_DATUM_EPSG=str(vertical_datum_epsg), ROLE="conditioned_final_depth")
    _append_dem_enhanced_touch(paths.combined_dir / "dem_enhanced_touch_log.jsonl", action="final_route_write", path=paths.conditioned_path, source=conditioned_internal_path, note="copied conditioned_final_dem_internal.tif to DEM_enhanced.tif as sole deliverable write")
    _write_u8_with_validation(paths.conditioned_prov_path, prov_out.astype("uint8"), prof_u8, tags={"VALUE_TYPE": "classification", "UNITS": "code", "ROLE": "conditioned_provenance"})
    stage_lock_validation = _validate_written_authoritative_lock_contract(
        final_path=paths.conditioned_path,
        aligned_auth_path=paths.aligned_auth_path,
        support_path=paths.support_class_path,
    )
    stage_lock_validation_path = paths.combined_dir / "final_route_authoritative_lock_validation.json"
    write_json_receipt(stage_lock_validation_path, {
        "stage": "final_route_outputs",
        "lock_state_normalization": lock_state_receipt,
        "runtime_authoritative_output_validation": runtime_authoritative_validation,
        "authoritative_first_contract": authoritative_first_contract,
        "conditioned_semantics_harmonization": conditioned_semantics_harmonization,
        "written_artifact_validation": stage_lock_validation,
        "artifacts": {
            "conditioned_depth": str(paths.conditioned_path),
        "conditioned_final_dem_internal": str(conditioned_internal_path),
            "aligned_authoritative_base": str(paths.aligned_auth_path),
            "support_class": str(paths.support_class_path),
        },
    })
    stage_paths = {
        "01_authoritative_aligned": debug_dir / "01_authoritative_aligned.tif",
        "02_river_primary_surface_handoff": debug_dir / "02_river_primary_surface_handoff.tif",
        "03_conditioned_before_final_route": debug_dir / "03_conditioned_before_final_route.tif",
        "04_conditioned_after_harmonize_conditioned_elevation": debug_dir / "04_conditioned_after_harmonize_conditioned_elevation.tif",
        "05_final_route_input": debug_dir / "05_final_route_input.tif",
        "06_DEM_enhanced_written": debug_dir / "06_DEM_enhanced_written.tif",
    }
    _write_float(stage_paths["01_authoritative_aligned"], auth_out, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "stage_debug_authoritative_aligned"})
    _write_float(stage_paths["02_river_primary_surface_handoff"], river_primary_surface, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "stage_debug_river_primary_surface_handoff"})
    _write_float(stage_paths["03_conditioned_before_final_route"], conditioned_before_final_route, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "stage_debug_conditioned_before_final_route"}, min_allowed=-1000.0, max_allowed=1000.0)
    _write_float(stage_paths["04_conditioned_after_harmonize_conditioned_elevation"], cond_after_harmonize, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "stage_debug_conditioned_after_harmonize"}, min_allowed=-1000.0, max_allowed=1000.0)
    _write_float(stage_paths["05_final_route_input"], cond_out, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "stage_debug_final_route_input"}, min_allowed=-1000.0, max_allowed=1000.0)
    _write_float(stage_paths["06_DEM_enhanced_written"], conditioned, prof_f32, tags={"VALUE_TYPE": "elevation", "UNITS": "meters", "SIGN_CONVENTION": "relative_to_datum", "VERTICAL_SEMANTICS": "absolute_elevation", "VERTICAL_DATUM": vertical_datum, "VERTICAL_DATUM_EPSG": vertical_datum_epsg, "ROLE": "stage_debug_dem_enhanced_written"}, min_allowed=-1000.0, max_allowed=1000.0)

    stage_semantics = {
        "01_authoritative_aligned": {"path": str(stage_paths["01_authoritative_aligned"]), "role": "aligned_authoritative_base", "vertical_semantics": "absolute_elevation", "vertical_datum": vertical_datum, "vertical_datum_epsg": vertical_datum_epsg, "sign_convention": "relative_to_datum", "crs": str(prof_f32.get("crs"))},
        "02_river_primary_surface_handoff": {"path": str(stage_paths["02_river_primary_surface_handoff"]), "role": "river_primary_surface_handoff", "vertical_semantics": "absolute_elevation", "vertical_datum": vertical_datum, "vertical_datum_epsg": vertical_datum_epsg, "sign_convention": "relative_to_datum", "crs": str(prof_f32.get("crs"))},
        "03_conditioned_before_final_route": {"path": str(stage_paths["03_conditioned_before_final_route"]), "role": "conditioned_before_final_route", "vertical_semantics": "absolute_elevation", "vertical_datum": vertical_datum, "vertical_datum_epsg": vertical_datum_epsg, "sign_convention": "relative_to_datum", "crs": str(prof_f32.get("crs"))},
        "04_conditioned_after_harmonize_conditioned_elevation": {"path": str(stage_paths["04_conditioned_after_harmonize_conditioned_elevation"]), "role": "conditioned_after_harmonize_conditioned_elevation", "vertical_semantics": "absolute_elevation", "vertical_datum": vertical_datum, "vertical_datum_epsg": vertical_datum_epsg, "sign_convention": "relative_to_datum", "crs": str(prof_f32.get("crs")), "harmonization_receipt": conditioned_semantics_harmonization},
        "05_final_route_input": {"path": str(stage_paths["05_final_route_input"]), "role": "final_route_input", "vertical_semantics": "absolute_elevation", "vertical_datum": vertical_datum, "vertical_datum_epsg": vertical_datum_epsg, "sign_convention": "relative_to_datum", "crs": str(prof_f32.get("crs"))},
        "06_DEM_enhanced_written": {"path": str(stage_paths["06_DEM_enhanced_written"]), "role": "conditioned_final_depth", "vertical_semantics": "absolute_elevation", "vertical_datum": vertical_datum, "vertical_datum_epsg": vertical_datum_epsg, "sign_convention": "relative_to_datum", "crs": str(prof_f32.get("crs"))},
    }
    stage_divergence_audit = _write_stage_divergence_audit(
        debug_dir=debug_dir,
        report=report,
        base_dir=Path(getattr(cfg, "out_dir", paths.combined_dir)),
        stage_paths=stage_paths,
        stage_semantics=stage_semantics,
        logger=log,
    )
    stage_debug_manifest_path = paths.combined_dir / "debug_final_route_manifest.json"
    required_debug_outputs = {
        "debug_dir": str(debug_dir),
        **{name: str(path) for name, path in stage_paths.items()},
        "stage_semantics_receipt": str(debug_dir / "stage_semantics_receipt.json"),
        "stage_divergence_receipt": str(debug_dir / "stage_divergence_receipt.json"),
    }
    missing_debug_outputs = [name for name, path in required_debug_outputs.items() if name != "debug_dir" and not Path(path).exists()]
    write_json_receipt(stage_debug_manifest_path, {
        "stage": "final_route_debug_bundle",
        "required_outputs": required_debug_outputs,
        "missing_required_outputs": missing_debug_outputs,
        "stage_divergence_audit": stage_divergence_audit,
    })
    log.info("[FINAL_ROUTE][DEBUG] Debug final-route bundle: %s", debug_dir)
    log.info("[FINAL_ROUTE][DEBUG] Debug final-route manifest: %s", stage_debug_manifest_path)
    if missing_debug_outputs:
        raise RuntimeError(
            "final-route debug bundle missing required outputs: " + ", ".join(missing_debug_outputs)
        )

    write_json_receipt(stage_lock_validation_path, {
        "stage": "final_route_outputs",
        "lock_state_normalization": lock_state_receipt,
        "runtime_authoritative_output_validation": runtime_authoritative_validation,
        "authoritative_first_contract": authoritative_first_contract,
        "conditioned_semantics_harmonization": conditioned_semantics_harmonization,
        "written_artifact_validation": stage_lock_validation,
        "stage_divergence_audit": stage_divergence_audit,
        "artifacts": {
            "conditioned_depth": str(paths.conditioned_path),
            "aligned_authoritative_base": str(paths.aligned_auth_path),
            "support_class": str(paths.support_class_path),
        },
    })

    if not stage_lock_validation.get("ok", False):
        hard_lock = stage_lock_validation.get("authoritative_hard_lock", {}) if isinstance(stage_lock_validation.get("authoritative_hard_lock", {}), dict) else {}
        raise RuntimeError(
            "final-route write altered authoritative-locked cells before downstream validation: "
            f"mismatch_pixels={int(hard_lock.get('mismatch_pixels', 0) or 0)} "
            f"max_abs_diff_m={float(hard_lock.get('max_abs_diff_m', 0.0) or 0.0):.12g}"
        )
    if river_channel_core_preservation_receipt is None:
        river_channel_core_preservation_receipt = {}
    write_json_receipt(paths.river_channel_core_preservation_receipt_path, {
        "stage": "final_route_outputs",
        "artifacts": {
            "river_primary_surface": str(paths.river_primary_surface_path),
            "river_primary_guidance_summary": str(paths.river_primary_guidance_summary_path),
            "conditioned_depth": str(paths.conditioned_path),
            "river_primary_surface_channel_core_preserve": str(paths.river_primary_surface_channel_core_preserve_path),
            "river_channel_core_preservation_zone": str(paths.river_channel_core_preservation_zone_path),
            "river_channel_core_prepost_delta": str(paths.river_channel_core_prepost_delta_path),
            "river_channel_core_bank_pull_risk": str(paths.river_channel_core_bank_pull_risk_path),
            "river_anchor_density": str(paths.river_anchor_density_path),
            "river_bank_influence": str(paths.river_bank_influence_runtime_path),
        },
        "metrics": river_channel_core_preservation_receipt,
    })

    support_note = f"{support_note}; authoritative_first=locked_changed:{authoritative_first_contract.get('locked_changed_count', 0)},background_changed:{authoritative_first_contract.get('background_changed_count', 0)},low_conf_outside:{authoritative_first_contract.get('low_confidence_fill_outside_guidance_count', 0)}; final_route_authoritative_validation=runtime_ok:{int(bool(runtime_authoritative_validation.get('ok', False)))},conditioned_mismatch:{int(runtime_authoritative_validation.get('conditioned_mismatch_pixels', 0) or 0)},support_mismatch:{int(runtime_authoritative_validation.get('support_mismatch_pixels', 0) or 0)},provenance_mismatch:{int(runtime_authoritative_validation.get('provenance_mismatch_pixels', 0) or 0)},guidance_mismatch:{int(runtime_authoritative_validation.get('guidance_influence_mismatch_pixels', 0) or 0)}; channel_core_zone:{int(river_channel_core_preservation_receipt.get('channel_core_zone_pixels', 0) or 0)},channel_core_abs_p95:{float(river_channel_core_preservation_receipt.get('delta_abs_p95_m', 0.0) or 0.0):.4f},bank_pull_risk_p95:{float(river_channel_core_preservation_receipt.get('bank_pull_risk_p95', 0.0) or 0.0):.4f}"

    audit = summarize_precedence_audit(auth=auth, conditioned=conditioned, support=support, provenance=prov_out, guidance_influence=guidance_influence)
    write_precedence_audit(paths.precedence_audit_path, audit, extra={
        "source_authoritative_base": str(paths.auth_src),
        "candidate_input": str(candidate_path) if candidate_path else None,
        "source_aware_candidate": str(paths.source_candidate_path),
        "conditioned_output": str(paths.conditioned_path),
    })
    write_conditioning_audit(
        paths.conditioning_audit_path,
        auth=auth,
        conditioned=conditioned,
        support=support,
        provenance=prov_out,
        guidance_influence=guidance_influence,
        conditioned_uncertainty=conditioned_uncertainty,
        extra={
            "source_authoritative_base": str(paths.auth_src),
            "conditioned_output": str(paths.conditioned_path),
            "conditioned_uncertainty": str(paths.conditioned_uncertainty_path),
            "guidance_influence_raster": str(paths.guidance_influence_path),
        },
    )

    nodata_audit = {
        "aligned_authoritative_base": collect_array_nodata_audit(auth, nodata=float(nodata)),
        "guidance_surface": collect_array_nodata_audit(guidance_surface, nodata=float(nodata)),
        "conditioned": collect_array_nodata_audit(conditioned, nodata=float(nodata)),
    }
    contract_outputs = {
        "support_class": str(paths.support_class_path),
        "regime_class": str(paths.regime_class_path),
        "support_distance": str(paths.support_distance_path),
        "support_density": str(paths.support_density_path),
        "anchor_uncertainty": str(paths.anchor_uncertainty_path),
        "guidance_uncertainty": str(paths.guidance_uncertainty_path),
        "conditioned_uncertainty": str(paths.conditioned_uncertainty_path),
        "guidance_influence": str(paths.guidance_influence_path),
        "conditioned_depth": str(paths.conditioned_path),
        "conditioned_final_dem_internal": str(conditioned_internal_path),
        "conditioned_provenance": str(paths.conditioned_prov_path),
        "coastal_sdb_confidence": str(paths.coastal_sdb_confidence_path),
        "river_anchor_distance": str(paths.river_anchor_distance_path),
        "river_anchor_density": str(paths.river_anchor_density_path),
        "river_scaffold_confidence": str(paths.river_scaffold_confidence_path),
        "river_bank_distance": str(paths.river_bank_distance_path),
        "river_bank_influence": str(paths.river_bank_influence_runtime_path),
        "river_bank_elevation": str(paths.river_bank_elevation_path),
        "river_primary_surface": str(paths.river_primary_surface_path),
        "river_primary_surface_confidence": str(paths.river_primary_surface_confidence_path),
        "river_primary_surface_source_class": str(paths.river_primary_surface_source_class_path),
        "river_primary_surface_support_count": str(paths.river_primary_surface_support_count_path),
        "river_primary_surface_domain": str(paths.river_primary_surface_domain_path),
        "river_primary_surface_channel_core_preserve": str(paths.river_primary_surface_channel_core_preserve_path),
        "river_channel_core_preservation_zone": str(paths.river_channel_core_preservation_zone_path),
        "river_channel_core_prepost_delta": str(paths.river_channel_core_prepost_delta_path),
        "river_channel_core_bank_pull_risk": str(paths.river_channel_core_bank_pull_risk_path),
        "river_channel_core_preservation_receipt": str(paths.river_channel_core_preservation_receipt_path),
        "river_primary_surface_contract": str(paths.river_primary_surface_contract_path),
        "river_primary_guidance_summary": str(paths.river_primary_guidance_summary_path),
    }
    write_guidance_uncertainty_contract(
        paths.guidance_uncertainty_contract_path,
        outputs=contract_outputs,
        support_note=support_note,
    )
    write_final_output_layer_contract(
        paths.final_output_layer_contract_path,
        outputs=contract_outputs,
    )
    write_final_vertical_semantics_contract(
        paths.final_vertical_semantics_contract_path,
        outputs={**contract_outputs,
            "source_candidate": str(paths.source_candidate_path),
            "river_bank_continuity_weight": str(river_bank_continuity_runtime_path),
            "river_bank_graph_confidence": str(river_bank_graph_confidence_runtime_path),
            "river_bank_confluence_damping": str(river_bank_confluence_damping_runtime_path),
            "river_bank_estuary_side_decay": str(river_bank_estuary_side_decay_runtime_path),
        },
    )

    runtime_artifact_roles = {
        "aligned_authoritative_base": "primary",
        "conditioned_depth": "primary",
        "conditioned_uncertainty": "diagnostic_only",
        "conditioned_provenance": "diagnostic_only",
        "precedence_audit": "diagnostic_only",
        "conditioning_audit": "diagnostic_only",
        "guidance_uncertainty_contract": "diagnostic_only",
        "river_primary_surface_contract": "diagnostic_only",
        "river_primary_guidance_summary": "diagnostic_only",
        "river_channel_core_preservation_receipt": "diagnostic_only",
        "final_output_layer_contract": "diagnostic_only",
        "final_vertical_semantics_contract": "diagnostic_only",
        "guidance_surface": "diagnostic_only",
        "guidance_surface_provenance": "diagnostic_only",
        "river_primary_surface": "primary",
        "river_primary_surface_confidence": "diagnostic_only",
        "river_primary_surface_source_class": "diagnostic_only",
        "river_primary_surface_support_count": "diagnostic_only",
        "river_primary_surface_domain": "diagnostic_only",
        "river_primary_surface_channel_core_preserve": "diagnostic_only",
        "river_channel_core_preservation_zone": "diagnostic_only",
        "river_channel_core_prepost_delta": "diagnostic_only",
        "river_channel_core_bank_pull_risk": "diagnostic_only",
    }

    write_json_receipt(paths.outputs_receipt_path, {
        "stage": "final_route_outputs",
        "route_mode": "staged_final_route_single_source_of_truth",
        "written_outputs": {
            "aligned_authoritative_base": str(paths.aligned_auth_path),
            "conditioned_depth": str(paths.conditioned_path),
            "river_primary_surface": str(paths.river_primary_surface_path),
            "conditioned_uncertainty": str(paths.conditioned_uncertainty_path),
            "conditioned_provenance": str(paths.conditioned_prov_path),
            "precedence_audit": str(paths.precedence_audit_path),
            "conditioning_audit": str(paths.conditioning_audit_path),
            "guidance_uncertainty_contract": str(paths.guidance_uncertainty_contract_path),
            "river_primary_surface_contract": str(paths.river_primary_surface_contract_path),
            "river_primary_guidance_summary": str(paths.river_primary_guidance_summary_path),
            "river_channel_core_preservation_receipt": str(paths.river_channel_core_preservation_receipt_path),
            "final_output_layer_contract": str(paths.final_output_layer_contract_path),
            "final_vertical_semantics_contract": str(paths.final_vertical_semantics_contract_path),
            "stage_divergence_debug_dir": str(debug_dir),
            "stage_divergence_receipt": str(debug_dir / "stage_divergence_receipt.json"),
            "stage_semantics_receipt": str(debug_dir / "stage_semantics_receipt.json"),
        },
        "diagnostic_only_outputs": {
            "guidance_surface": str(paths.source_candidate_path),
            "guidance_surface_provenance": str(paths.source_candidate_prov_path),
            "river_primary_surface_confidence": str(paths.river_primary_surface_confidence_path),
            "river_primary_surface_source_class": str(paths.river_primary_surface_source_class_path),
            "river_primary_surface_support_count": str(paths.river_primary_surface_support_count_path),
            "river_primary_surface_domain": str(paths.river_primary_surface_domain_path),
            "river_primary_surface_channel_core_preserve": str(paths.river_primary_surface_channel_core_preserve_path),
            "river_channel_core_preservation_zone": str(paths.river_channel_core_preservation_zone_path),
            "river_channel_core_prepost_delta": str(paths.river_channel_core_prepost_delta_path),
            "river_channel_core_bank_pull_risk": str(paths.river_channel_core_bank_pull_risk_path),
            "river_channel_core_preservation_receipt": str(paths.river_channel_core_preservation_receipt_path),
            "stage_divergence_summary_csv": stage_divergence_audit.get("support_point_stage_divergence_summary_csv"),
            "stage_divergence_cell_trace_csv": stage_divergence_audit.get("support_point_stage_divergence_cell_trace_csv"),
        },
        "primary_runtime_outputs": [
            "aligned_authoritative_base",
            "conditioned_depth",
            "river_primary_surface",
        ],
        "runtime_drilldown_outputs": [
            "conditioned_uncertainty",
            "conditioned_provenance",
            "precedence_audit",
            "conditioning_audit",
            "guidance_uncertainty_contract",
            "river_primary_guidance_summary",
            "river_primary_surface_contract",
            "river_channel_core_preservation_receipt",
            "final_output_layer_contract",
            "final_vertical_semantics_contract",
        ],
        "artifact_roles": runtime_artifact_roles,
        "invariants": {
            "continuous_output": True,
            "authoritative_hard_lock": True,
            "diagnostic_dense_surfaces_only": True,
            "single_output_nodata_policy": float(nodata),
        },
        "conditioned_repair_note": conditioned_repair_note,
        "authoritative_first_contract": authoritative_first_contract,
        "nodata_audit": nodata_audit,
    })

    legacy_cleanup_path = paths.combined_dir / "legacy_cleanup_receipt.json"
    legacy_cleanup = write_legacy_cleanup_receipt(report=report, receipt_path=legacy_cleanup_path)

    final_route_receipt = {
        "route_mode": "staged_final_route_single_source_of_truth",
        "single_authoritative_route_active": True,
        "legacy_parallel_route_retired": True,
        "stage_receipts": {
            "inputs": str(paths.inputs_receipt_path),
            "guidance": str(getattr(guidance, "receipt_path", paths.guidance_receipt_path)),
            "terrain": str(getattr(terrain, "receipt_path", paths.terrain_receipt_path)),
            "outputs": str(paths.outputs_receipt_path),
        },
        "structural_inputs": {
            "authoritative_base": str(paths.auth_src),
            "baseline_cudem_interpolation": str(guidance.baseline_cudem_path) if getattr(guidance, "baseline_cudem_path", None) is not None else None,
            "sdb_guide_points": str(guidance.sdb_guide_points_path) if guidance.sdb_guide_points_path is not None else None,
            "river_guide_points": str(guidance.river_guide_points_path) if guidance.river_guide_points_path is not None else None,
        },
        "diagnostic_only_inputs": {
            "legacy_candidate": str(candidate_path) if candidate_path else None,
            "dense_sdb_depth_raster": str(guidance.sdb_depth_path) if guidance.sdb_depth_path else None,
        },
        "written_outputs": {
            "final_depth": str(paths.conditioned_path),
            "final_uncertainty": str(paths.conditioned_uncertainty_path),
            "final_provenance": str(paths.conditioned_prov_path),
            "precedence_audit": str(paths.precedence_audit_path),
            "conditioning_audit": str(paths.conditioning_audit_path),
            "guidance_uncertainty_contract": str(paths.guidance_uncertainty_contract_path),
            "river_primary_surface_contract": str(paths.river_primary_surface_contract_path),
            "river_primary_guidance_summary": str(paths.river_primary_guidance_summary_path),
            "river_channel_core_preservation_receipt": str(paths.river_channel_core_preservation_receipt_path),
            "final_output_layer_contract": str(paths.final_output_layer_contract_path),
            "final_vertical_semantics_contract": str(paths.final_vertical_semantics_contract_path),
            "stage_divergence_debug_dir": str(debug_dir),
            "stage_divergence_receipt": str(debug_dir / "stage_divergence_receipt.json"),
            "stage_semantics_receipt": str(debug_dir / "stage_semantics_receipt.json"),
        },
        "artifacts": {
            "diagnostic_guidance_surface": str(paths.source_candidate_path),
            "diagnostic_guidance_surface_provenance": str(paths.source_candidate_prov_path),
            "river_primary_surface": str(paths.river_primary_surface_path),
            "river_primary_surface_confidence": str(paths.river_primary_surface_confidence_path),
            "river_primary_surface_source_class": str(paths.river_primary_surface_source_class_path),
            "river_primary_surface_support_count": str(paths.river_primary_surface_support_count_path),
            "river_primary_surface_domain": str(paths.river_primary_surface_domain_path),
            "river_primary_surface_channel_core_preserve": str(paths.river_primary_surface_channel_core_preserve_path),
            "river_channel_core_preservation_zone": str(paths.river_channel_core_preservation_zone_path),
            "river_channel_core_prepost_delta": str(paths.river_channel_core_prepost_delta_path),
            "river_channel_core_bank_pull_risk": str(paths.river_channel_core_bank_pull_risk_path),
            "river_channel_core_preservation_receipt": str(paths.river_channel_core_preservation_receipt_path),
            "stage_divergence_summary_csv": stage_divergence_audit.get("support_point_stage_divergence_summary_csv"),
            "stage_divergence_cell_trace_csv": stage_divergence_audit.get("support_point_stage_divergence_cell_trace_csv"),
        },
        "primary_runtime_receipts": {
            "river_active_product_story": "river_active_runtime_summary.json",
            "river_runtime_drilldowns": [
                "river_primary_guidance_summary.json",
                "river_primary_surface_contract.json",
                "final_route_outputs_receipt.json",
                "final_route_receipt.json",
            ],
            "active_runtime_products": [
                "aligned_authoritative_base",
                "conditioned_depth",
                "river_primary_surface",
            ],
        },
        "artifact_roles": runtime_artifact_roles,
        "manifest_contract": validate_final_route_contract(report),
        "authoritative_first_contract": authoritative_first_contract,
        "legacy_cleanup_receipt": str(legacy_cleanup_path),
        "legacy_cleanup": legacy_cleanup,
    }
    write_json_receipt(paths.final_route_receipt_path, final_route_receipt)

    report.setdefault("authoritative_base", {})["status"] = "applied"
    report["authoritative_base"]["inputs"] = {"source": str(paths.auth_src)}
    baseline_cudem_interpolation = None
    auth_auto = report.get("authoritative_base_auto", {}) if isinstance(report.get("authoritative_base_auto"), dict) else {}
    cand_baseline = auth_auto.get("baseline_cudem_interpolation")
    if cand_baseline:
        try:
            baseline_path = Path(str(cand_baseline))
        except (TypeError, ValueError, OSError):
            baseline_path = None
        if baseline_path and baseline_path.exists():
            baseline_cudem_interpolation = str(baseline_path)
    if baseline_cudem_interpolation is None:
        try:
            sibling = Path(str(paths.auth_src)).with_name("cudem_baseline_interpolation.tif")
        except (TypeError, ValueError, OSError):
            sibling = None
        if sibling and sibling.exists():
            baseline_cudem_interpolation = str(sibling)

    report["authoritative_base"]["outputs"] = {
        "aligned_authoritative_base": str(paths.aligned_auth_path),
        "baseline_cudem_interpolation": baseline_cudem_interpolation,
        "gap_mask": str(paths.gap_mask_path),
        "eligible_fill_mask": str(paths.eligible_mask_path),
        "support_class": str(paths.support_class_path),
        "regime_class": str(paths.regime_class_path),
        "source_aware_candidate": str(paths.source_candidate_path),
        "source_aware_candidate_provenance": str(paths.source_candidate_prov_path),
        "support_distance": str(paths.support_distance_path),
        "support_density": str(paths.support_density_path),
        "anchor_uncertainty": str(paths.anchor_uncertainty_path),
        "guidance_uncertainty": str(paths.guidance_uncertainty_path),
        "conditioned_uncertainty": str(paths.conditioned_uncertainty_path),
        "guidance_influence": str(paths.guidance_influence_path),
        "coastal_sdb_confidence": str(paths.coastal_sdb_confidence_path),
        "river_anchor_distance": str(paths.river_anchor_distance_path),
        "river_anchor_density": str(paths.river_anchor_density_path),
        "river_scaffold_confidence": str(paths.river_scaffold_confidence_path),
        "river_bank_distance": str(paths.river_bank_distance_path),
        "river_bank_influence": str(paths.river_bank_influence_runtime_path),
        "river_bank_elevation": str(paths.river_bank_elevation_path),
        "river_primary_surface": str(paths.river_primary_surface_path),
        "river_primary_surface_confidence": str(paths.river_primary_surface_confidence_path),
        "river_primary_surface_source_class": str(paths.river_primary_surface_source_class_path),
        "river_primary_surface_support_count": str(paths.river_primary_surface_support_count_path),
        "river_primary_surface_domain": str(paths.river_primary_surface_domain_path),
        "river_primary_surface_channel_core_preserve": str(paths.river_primary_surface_channel_core_preserve_path),
        "river_channel_core_preservation_zone": str(paths.river_channel_core_preservation_zone_path),
        "river_channel_core_prepost_delta": str(paths.river_channel_core_prepost_delta_path),
        "river_channel_core_bank_pull_risk": str(paths.river_channel_core_bank_pull_risk_path),
        "river_channel_core_preservation_receipt": str(paths.river_channel_core_preservation_receipt_path),
        "river_primary_surface_contract": str(paths.river_primary_surface_contract_path),
        "river_primary_guidance_summary": str(paths.river_primary_guidance_summary_path),
        "river_bank_continuity_weight": str(river_bank_continuity_runtime_path),
        "river_bank_graph_confidence": str(river_bank_graph_confidence_runtime_path),
        "river_bank_confluence_damping": str(river_bank_confluence_damping_runtime_path),
        "river_bank_estuary_side_decay": str(river_bank_estuary_side_decay_runtime_path),
        "conditioned_depth": str(paths.conditioned_path),
        "conditioned_provenance": str(paths.conditioned_prov_path),
        "source_provenance_input": str(provenance_path) if provenance_path else None,
        "precedence_audit": str(paths.precedence_audit_path),
        "conditioning_audit": str(paths.conditioning_audit_path),
        "guidance_uncertainty_contract": str(paths.guidance_uncertainty_contract_path),
        "final_output_layer_contract": str(paths.final_output_layer_contract_path),
        "final_vertical_semantics_contract": str(paths.final_vertical_semantics_contract_path),
        "final_route_receipt": str(paths.final_route_receipt_path),
        "stage_divergence_debug_dir": str(debug_dir),
        "stage_divergence_debug_manifest": str(stage_debug_manifest_path),
        "stage_divergence_receipt": str(debug_dir / "stage_divergence_receipt.json"),
        "stage_semantics_receipt": str(debug_dir / "stage_semantics_receipt.json"),
        "stage_divergence_summary_csv": stage_divergence_audit.get("support_point_stage_divergence_summary_csv"),
        "stage_divergence_cell_trace_csv": stage_divergence_audit.get("support_point_stage_divergence_cell_trace_csv"),
        "stage_debug_01_authoritative_aligned": str(stage_paths["01_authoritative_aligned"]),
        "stage_debug_02_river_primary_surface_handoff": str(stage_paths["02_river_primary_surface_handoff"]),
        "stage_debug_03_conditioned_before_final_route": str(stage_paths["03_conditioned_before_final_route"]),
        "stage_debug_04_conditioned_after_harmonize_conditioned_elevation": str(stage_paths["04_conditioned_after_harmonize_conditioned_elevation"]),
        "stage_debug_05_final_route_input": str(stage_paths["05_final_route_input"]),
        "stage_debug_06_dem_enhanced_written": str(stage_paths["06_DEM_enhanced_written"]),
        "legacy_cleanup_receipt": str(legacy_cleanup_path),
        "final_route_authoritative_lock_validation": str(stage_lock_validation_path),
    }
    report["authoritative_base"]["candidate_generation"] = {
        "mode": "staged_final_route_single_source_of_truth",
        "template_path": str(paths.template_path),
        "sdb_depth": str(guidance.sdb_depth_path) if guidance.sdb_depth_path else None,
        "river_depth": None,
        "primary_river_guidance_surface": str(paths.river_primary_surface_path),
        "baseline_cudem_interpolation": str(guidance.baseline_cudem_path) if getattr(guidance, "baseline_cudem_path", None) is not None else None,
        "legacy_candidate": None,
        "diagnostic_only_artifacts": {
            "legacy_candidate": str(candidate_path) if candidate_path else None,
            "guidance_surface": str(paths.source_candidate_path),
        },
        "stats": source_candidate.get("stats", {}),
        "backstop_policy": source_candidate.get("backstop_policy", {}),
        "provenance_codes": source_candidate.get("provenance_codes", {}),
    }
    report["final_dem_route"] = final_route_receipt
    report["authoritative_base"]["policy"] = build_final_dem_policy_dict(
        cfg,
        support_note=f"{support_note}; direct_guidance_route=authoritative_base_plus_guidance_artifacts",
        guidance_masks={
            "sdb_admissibility": None,
            "sdb_guidance_weight": None,
            "sdb_trusted_interior": None,
            "river_admissibility": None,
            "river_guidance_weight": None,
            "river_trusted_interior": None,
            "river_authoritative_support": None,
            "river_authoritative_support_depth": None,
        },
    )
    try:
        report["authoritative_base"]["precedence_audit"] = json.loads(paths.precedence_audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return paths.conditioned_path, paths.conditioned_prov_path, paths.aligned_auth_path, paths.gap_mask_path, paths.eligible_mask_path, paths.support_class_path
