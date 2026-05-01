"""Reporting helpers for authoritative-base-centered final DEM products.

Extracted from bathy_main.py to reduce orchestration-file size and isolate output logic.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.nodata_utils import sanitize_array
from core.json_io import write_json
from core.paths import ensure_dir
from output_products import build_final_output_contract
from canonical_river_scaffold import nested_aoi_relationship

from validation.validation_invariance_framework import run_validation_invariance_framework
from validation.scientific_validation_stage import write_scientific_validation_summary


def _grid_pixel_size_m(transform, crs, ref_lat_deg: Optional[float] = None) -> float:
    """Return a representative pixel size in meters for summary calculations."""
    dx = abs(float(getattr(transform, "a", 0.0) or 0.0))
    dy = abs(float(getattr(transform, "e", 0.0) or 0.0))
    px = max((dx + dy) / 2.0, 0.0)
    is_geographic = bool(getattr(crs, "is_geographic", False))
    if not is_geographic:
        return max(px, 1.0)
    lat = float(ref_lat_deg if ref_lat_deg is not None else 0.0)
    lat_rad = np.deg2rad(lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2.0 * lat_rad) + 1.175 * np.cos(4.0 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3.0 * lat_rad)
    dx_m = dx * max(abs(m_per_deg_lon), 1.0)
    dy_m = dy * max(abs(m_per_deg_lat), 1.0)
    return max((dx_m + dy_m) / 2.0, 1.0)


def _link_or_copy_file(src: Path, dst: Path, *, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    log = logger or logging.getLogger(__name__)
    try:
        src = Path(src).resolve()
        if not src.exists():
            return None
        ensure_dir(dst.parent)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        rel = os.path.relpath(str(src), str(dst.parent.resolve()))
        try:
            dst.symlink_to(rel)
        except OSError:
            shutil.copy2(src, dst)
        return dst
    except (OSError, shutil.Error, RuntimeError, ValueError):
        log.debug("[COMPARE] Failed linking/copying %s -> %s", src, dst, exc_info=True)
        return None


def resolve_baseline_cudem_interpolation(cfg: Any, report: Dict[str, Any], *, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    """Delegate baseline-CUDEM resolution to the canonical final-route helper."""
    from pipeline.final_route.final_route_inputs_stage import resolve_baseline_cudem_interpolation as _canonical

    try:
        return _canonical(cfg=cfg, report=report, auth_src=None)
    except Exception:
        (logger or logging.getLogger(__name__)).debug(
            "[COMPARE] Failed resolving baseline CUDEM interpolation via canonical helper",
            exc_info=True,
        )
        return None


def write_authoritative_cache_receipt(cfg: Any, report: Dict[str, Any]) -> Optional[Path]:
    info = report.get("authoritative_base_auto")
    if not isinstance(info, dict) or not info:
        return None
    receipt = {
        "mode": info.get("mode"),
        "cache_key": info.get("cache_key"),
        "cache_hit": info.get("cache_hit"),
        "cache_dir": info.get("cache_dir"),
        "authoritative_base": info.get("authoritative_base") or (str(getattr(cfg, "authoritative_base", "") or "") or None),
        "baseline_cudem_interpolation": info.get("baseline_cudem_interpolation"),
        "shared_root": info.get("shared_root"),
        "shared_tile_cache": info.get("shared_tile_cache"),
        "shared_cache_reuse": info.get("shared_cache_reuse"),
        "downstream_child_passthrough": info.get("downstream_child_passthrough", {}),
    }
    out_path = Path(cfg.out_dir) / "authoritative_base_cache_receipt.json"
    write_json(out_path, receipt)
    report.setdefault("authoritative_base_auto", {})["cache_receipt"] = receipt
    report.setdefault("outputs", {})["authoritative_base_cache_receipt"] = str(out_path)
    return out_path


def _safe_class_stats(values: np.ndarray, cls: np.ndarray, class_codes: Dict[str, str], *, pixel_area_m2: float) -> Dict[str, Any]:
    cls = np.asarray(cls)
    values = np.asarray(values, dtype=np.float32) if values is not None else None
    domain = np.isfinite(values) if values is not None else (cls > 0)
    total_count = int(np.count_nonzero(domain)) if domain is not None else int(cls.size)
    out: Dict[str, Any] = {"total_count": total_count, "pixel_area_m2": float(pixel_area_m2)}
    by_class: Dict[str, Any] = {}
    for code_str, label in (class_codes or {}).items():
        try:
            code = int(code_str)
        except (TypeError, ValueError):
            continue
        mask = cls == code
        count = int(np.count_nonzero(mask))
        if count <= 0:
            continue
        payload: Dict[str, Any] = {
            "label": str(label),
            "count": count,
            "area_m2": float(count * pixel_area_m2),
            "fraction_of_domain": float(count / max(total_count, 1)),
        }
        if values is not None:
            v = values[mask & np.isfinite(values)]
            if v.size > 0:
                payload.update({
                    "mean": float(np.nanmean(v)),
                    "mean_abs": float(np.nanmean(np.abs(v))),
                    "rmse": float(np.sqrt(np.nanmean(np.square(v)))),
                    "min": float(np.nanmin(v)),
                    "max": float(np.nanmax(v)),
                })
        by_class[str(code)] = payload
    out["by_class"] = by_class
    return out


def write_comparison_summary(cfg: Any, report: Dict[str, Any], packaged: Dict[str, Optional[str]], out_dir: Path, *, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    log = logger or logging.getLogger(__name__)
    try:
        import rasterio
    except ImportError:
        log.debug("[COMPARE] rasterio unavailable; skipping comparison summary", exc_info=True)
        return None

    try:
        final_path = packaged.get("final_depth_native")
        baseline_aligned_path = packaged.get("baseline_cudem_interpolation_aligned_to_final")
        diff_path = packaged.get("conditioned_minus_baseline_cudem")
        if not final_path or not baseline_aligned_path or not diff_path:
            return None

        support_path = packaged.get("support_class")
        prov_path = packaged.get("final_provenance_native")
        if not support_path and not prov_path:
            return None

        with rasterio.open(final_path) as final_ds:
            final_arr = sanitize_array(final_ds.read(1), final_ds.nodata, dtype="float32")
            final_nodata = np.float32(final_ds.nodata if final_ds.nodata is not None else -9999.0)
            ref_lat = (cfg.tile_bbox[1] + cfg.tile_bbox[3]) / 2.0 if getattr(cfg, "tile_bbox", None) else None
            pixel_size_m = _grid_pixel_size_m(final_ds.transform, final_ds.crs, ref_lat_deg=ref_lat)
            pixel_area_m2 = float(max(pixel_size_m, 1.0) ** 2)

            with rasterio.open(diff_path) as diff_ds:
                diff_arr = sanitize_array(diff_ds.read(1), diff_ds.nodata, dtype="float32")
                diff_nodata = np.float32(diff_ds.nodata if diff_ds.nodata is not None else -9999.0)

            has_diff = bool(np.any(np.isfinite(diff_arr)))
            summary: Dict[str, Any] = {
                "aoi": str(cfg.aoi),
                "pixel_area_m2": pixel_area_m2,
                "conditioned_minus_baseline_overall": {
                    "count": int(np.count_nonzero(np.isfinite(diff_arr))),
                    "mean": float(np.nanmean(diff_arr)) if has_diff else None,
                    "mean_abs": float(np.nanmean(np.abs(diff_arr))) if has_diff else None,
                    "rmse": float(np.sqrt(np.nanmean(np.square(diff_arr)))) if has_diff else None,
                    "min": float(np.nanmin(diff_arr)) if has_diff else None,
                    "max": float(np.nanmax(diff_arr)) if has_diff else None,
                },
            }

            policy = report.get("authoritative_base", {}).get("policy", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
            support_codes = policy.get("support_class_codes", {}) if isinstance(policy.get("support_class_codes", {}), dict) else {}
            prov_codes = policy.get("provenance_class_codes", {}) if isinstance(policy.get("provenance_class_codes", {}), dict) else {}
            regime_codes = policy.get("regime_class_codes", {}) if isinstance(policy.get("regime_class_codes", {}), dict) else {}

            if support_path:
                with rasterio.open(support_path) as ds:
                    support_arr = ds.read(1)
                summary["support_class_summary"] = _safe_class_stats(diff_arr, support_arr, support_codes, pixel_area_m2=pixel_area_m2)
            if prov_path:
                with rasterio.open(prov_path) as ds:
                    prov_arr = ds.read(1)
                summary["provenance_class_summary"] = _safe_class_stats(diff_arr, prov_arr, prov_codes, pixel_area_m2=pixel_area_m2)
            regime_summary = {}
            final_regime_path = packaged.get("regime_class")
            if final_regime_path:
                with rasterio.open(final_regime_path) as ds:
                    regime_arr = ds.read(1)
                regime_summary["final_regime_class"] = _safe_class_stats(diff_arr, regime_arr, regime_codes, pixel_area_m2=pixel_area_m2)
            for regime_key in ("sdb_regime_class", "river_regime_class"):
                regime_path = packaged.get(regime_key)
                if not regime_path:
                    continue
                with rasterio.open(regime_path) as ds:
                    regime_arr = ds.read(1)
                regime_summary[regime_key] = _safe_class_stats(diff_arr, regime_arr, regime_codes, pixel_area_m2=pixel_area_m2)
            if regime_summary:
                summary["regime_class_summary"] = regime_summary

        out_path = Path(out_dir) / "comparison_summary.json"
        write_json(out_path, summary)
        report.setdefault("outputs", {})["comparison_summary"] = str(out_path)
        return out_path
    except (OSError, ValueError, RuntimeError):
        log.debug("[COMPARE] Failed writing comparison summary", exc_info=True)
        return None


def write_final_dem_selection_receipt(cfg: Any, report: Dict[str, Any], *, final_native: Optional[Path], final_for_user: Optional[Path | str], final_provenance: Optional[Path | str]) -> Optional[Path]:
    out_path = Path(cfg.out_dir) / "final_dem_selection_receipt.json"
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)
    rationale = []
    preference_order = []
    if contract.get("selected_final_user"):
        preference_order.append("final_depth_user")
        rationale.append("Selected user-space delivery raster because it is the requested deliverable representation of the chosen native DEM.")
    if contract.get("selected_final_native"):
        preference_order.append("final_depth_native")
    route = contract.get("final_generation_route")
    if route == "support_aware_terrain_interpolator_plus_gapfill":
        rationale.append("Preferred gapfill output because it preserves authoritative locks while refining only eligible gaps after support-aware terrain interpolation.")
    elif route == "support_aware_terrain_interpolator":
        rationale.append("Preferred terrain_interpolator output because it enforces hard control on authoritative cells and applies support-aware guidance-conditioned interpolation.")
    elif route == "legacy_fusion_only":
        rationale.append("Using fusion output because no later support-aware conditioning stage produced a more preferred existing DEM.")
    elif contract.get("selected_final_native"):
        rationale.append("Selected the explicit final native DEM stage produced by the workflow.")
    if contract.get("runtime_engine", {}).get("authoritative_conditioning_applied"):
        rationale.append("Authoritative-base-centered policy remains in force: finite authoritative cells are hard-locked in the selected final DEM.")
    if contract.get("selected_final_provenance"):
        rationale.append("A matched provenance raster exists for the selected final DEM.")
    payload = dict(contract)
    payload.update({
        "selected_final_depth": contract.get("selected_final_depth"),
        "selected_final_stage": contract.get("selected_final_stage"),
        "selected_final_provenance": contract.get("selected_final_provenance"),
        "preference_order": preference_order,
        "selection_rationale": rationale,
    })
    write_json(out_path, payload)
    report.setdefault("outputs", {})["final_dem_selection_receipt"] = str(out_path)
    report.setdefault("final_dem_runtime", {}).update({
        "final_generation_route": contract.get("final_generation_route"),
        "runtime_engine": contract.get("runtime_engine"),
    })
    return out_path


def write_comparison_package(cfg: Any, report: Dict[str, Any], *, final_native: Optional[Path], final_for_user: Optional[Path | str], final_provenance: Optional[Path | str], logger: Optional[logging.Logger] = None) -> Optional[Path]:
    log = logger or logging.getLogger(__name__)
    baseline_path = resolve_baseline_cudem_interpolation(cfg, report, logger=log)
    native_path = Path(final_native) if final_native else None
    if native_path is None or not native_path.exists():
        return None
    out_dir = ensure_dir(Path(cfg.out_dir) / "comparison_package")
    manifest_path = out_dir / "comparison_package_manifest.json"

    ab_out = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    packaged: Dict[str, Optional[str]] = {}

    def _pkg(src: Optional[Path | str], name: str) -> Optional[str]:
        if not src:
            return None
        linked = _link_or_copy_file(Path(str(src)), out_dir / name, logger=log)
        return str(linked) if linked else None

    packaged["authoritative_base"] = _pkg(ab_out.get("aligned_authoritative_base") or getattr(cfg, "authoritative_base", None), "authoritative_base.tif")
    packaged["final_depth_native"] = _pkg(native_path, "conditioned_final_depth_native.tif")
    packaged["final_depth_user"] = _pkg(final_for_user, "conditioned_final_depth_user.tif") if final_for_user else None
    packaged["final_provenance_native"] = _pkg(final_provenance, "conditioned_final_provenance_native.tif") if final_provenance else None
    for key in (
        "support_class",
        "support_distance",
        "support_density",
        "guidance_influence",
        "coastal_sdb_confidence",
        "river_anchor_distance",
        "river_anchor_density",
        "river_scaffold_confidence",
    ):
        cand = ab_out.get(key)
        if cand:
            packaged[key] = _pkg(cand, f"{key}.tif")

    if baseline_path and baseline_path.exists():
        packaged["baseline_cudem_interpolation_native"] = _pkg(baseline_path, "baseline_cudem_interpolation_native.tif")
        try:
            import rasterio
            from rasterio.warp import reproject, Resampling
        except ImportError:
            log.debug("[COMPARE] rasterio unavailable; skipping aligned baseline outputs", exc_info=True)
        else:
            try:
                baseline_aligned_path = out_dir / "baseline_cudem_interpolation_aligned_to_final.tif"
                diff_path = out_dir / "conditioned_minus_baseline_cudem.tif"
                with rasterio.open(native_path) as ref_ds, rasterio.open(baseline_path) as base_ds:
                    out = np.full((ref_ds.height, ref_ds.width), np.float32(np.nan), dtype=np.float32)
                    reproject(
                        source=rasterio.band(base_ds, 1),
                        destination=out,
                        src_transform=base_ds.transform,
                        src_crs=base_ds.crs,
                        dst_transform=ref_ds.transform,
                        dst_crs=ref_ds.crs,
                        src_nodata=base_ds.nodata,
                        dst_nodata=np.float32(np.nan),
                        resampling=Resampling.bilinear,
                    )
                    prof = ref_ds.profile.copy()
                    prof.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
                    out_write = out.copy()
                    out_write[~np.isfinite(out_write)] = np.float32(-9999.0)
                    with rasterio.open(baseline_aligned_path, "w", **prof) as dst:
                        dst.write(out_write.astype("float32"), 1)

                    final_arr = sanitize_array(ref_ds.read(1), ref_ds.nodata, dtype="float32")
                    diff = final_arr - out
                    diff[~np.isfinite(final_arr) | ~np.isfinite(out)] = np.nan
                    diff_write = diff.copy()
                    diff_write[~np.isfinite(diff_write)] = np.float32(-9999.0)
                    with rasterio.open(diff_path, "w", **prof) as dst:
                        dst.write(diff_write.astype("float32"), 1)
                packaged["baseline_cudem_interpolation_aligned_to_final"] = str(baseline_aligned_path)
                packaged["conditioned_minus_baseline_cudem"] = str(diff_path)
            except (OSError, ValueError, RuntimeError):
                log.debug("[COMPARE] Failed writing baseline-vs-conditioned comparison rasters", exc_info=True)

    payload = {
        "authoritative_base": packaged.get("authoritative_base"),
        "baseline_cudem_interpolation_native": packaged.get("baseline_cudem_interpolation_native"),
        "baseline_cudem_interpolation_aligned_to_final": packaged.get("baseline_cudem_interpolation_aligned_to_final"),
        "conditioned_final_depth_native": packaged.get("final_depth_native"),
        "conditioned_final_depth_user": packaged.get("final_depth_user"),
        "conditioned_final_provenance_native": packaged.get("final_provenance_native"),
        "conditioned_minus_baseline_cudem": packaged.get("conditioned_minus_baseline_cudem"),
        "support_artifacts": {k: packaged.get(k) for k in (
            "support_class",
            "support_distance",
            "support_density",
            "guidance_influence",
            "coastal_sdb_confidence",
            "river_anchor_distance",
            "river_anchor_density",
            "river_scaffold_confidence",
        )},
    }
    summary_path = write_comparison_summary(cfg, report, packaged, out_dir, logger=log)
    if summary_path:
        payload["comparison_summary"] = str(summary_path)
    write_json(manifest_path, payload)
    report.setdefault("outputs", {})["comparison_package_manifest"] = str(manifest_path)
    report.setdefault("outputs", {})["comparison_package_dir"] = str(out_dir)
    return manifest_path


def write_explicit_final_outputs_manifest(cfg: Any, report: Dict[str, Any], *, final_native: Optional[Path], final_for_user: Optional[Path | str], final_provenance: Optional[Path | str]) -> Optional[Path]:
    out_path = Path(cfg.out_dir) / "final_outputs.json"
    baseline_path = resolve_baseline_cudem_interpolation(cfg, report)
    ab_out = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)
    payload = dict(contract)
    payload.update({
        "authoritative_base": str(getattr(cfg, "authoritative_base", "") or "") or None,
        "baseline_cudem_interpolation": str(baseline_path) if baseline_path else None,
        "conditioned_authoritative_base": ab_out.get("aligned_authoritative_base"),
        "support_class": ab_out.get("support_class"),
        "support_distance": ab_out.get("support_distance"),
        "support_density": ab_out.get("support_density"),
        "guidance_influence": ab_out.get("guidance_influence"),
        "coastal_sdb_confidence": ab_out.get("coastal_sdb_confidence"),
        "river_anchor_distance": ab_out.get("river_anchor_distance"),
        "river_anchor_density": ab_out.get("river_anchor_density"),
        "river_scaffold_confidence": ab_out.get("river_scaffold_confidence"),
        "river_trusted_interior": report.get("outputs", {}).get("river_trusted_interior") if isinstance(report.get("outputs", {}), dict) else None,
        "river_scaffold_domains": report.get("outputs", {}).get("river_scaffold_domains") if isinstance(report.get("outputs", {}), dict) else None,
        "river_trusted_interior_summary": report.get("outputs", {}).get("river_trusted_interior_summary") if isinstance(report.get("outputs", {}), dict) else None,
        "river_channel_surface_graph_mode": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("channel_surface_graph_mode"),
        "river_channel_surface_support_class": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("channel_surface_support_class"),
        "river_channel_surface_uncertainty": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("channel_surface_uncertainty"),
        "river_channel_surface_hard_lock": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("channel_surface_hard_lock"),
        "river_channel_surface_junction_constrained": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("channel_surface_junction_constrained"),
        "river_channel_surface_unsupported_span": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("channel_surface_unsupported_span"),
        "river_channel_surface_unsupported_regime": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("channel_surface_unsupported_regime"),
        "river_channel_surface_residual_to_candidate": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("channel_surface_residual_to_candidate"),
        "river_graph_backbone_diagnostics": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("graph_backbone_diagnostics"),
        "river_graph_physical_plausibility_contract": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("graph_physical_plausibility_contract"),
        "river_support_uncertainty_contract": (report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}).get("support_uncertainty_contract"),
        "final_depth_native": contract.get("selected_final_native"),
        "final_depth_user": contract.get("selected_final_user"),
        "final_provenance_native": contract.get("selected_final_provenance"),
        "selected_final_depth": contract.get("selected_final_depth"),
        "selected_final_invariant": contract.get("selected_final_invariant"),
        "selected_final_invariant_lock_validation": contract.get("selected_final_invariant_lock_validation"),
        "selected_final_invariant_lock_validation_target": contract.get("selected_final_invariant_lock_validation_target"),
        "selected_final_provenance": contract.get("selected_final_provenance"),
    })
    write_json(out_path, payload)
    report.setdefault("outputs", {})["final_outputs_manifest"] = str(out_path)
    report.setdefault("outputs", {})["final_depth_native"] = payload["final_depth_native"]
    report.setdefault("outputs", {})["final_depth_user"] = payload["final_depth_user"]
    report.setdefault("outputs", {})["final_provenance_native"] = payload["final_provenance_native"]
    report.setdefault("outputs", {})["selected_final_depth"] = payload["selected_final_depth"]
    report.setdefault("outputs", {})["selected_final_invariant"] = payload["selected_final_invariant"]
    report.setdefault("outputs", {})["selected_final_invariant_lock_validation"] = payload["selected_final_invariant_lock_validation"]
    report.setdefault("outputs", {})["selected_final_invariant_lock_validation_target"] = payload["selected_final_invariant_lock_validation_target"]
    report.setdefault("outputs", {})["selected_final_provenance"] = payload["selected_final_provenance"]
    report.setdefault("final_dem_runtime", {}).update({
        "final_generation_route": contract.get("final_generation_route"),
        "runtime_engine": contract.get("runtime_engine"),
        "guidance_manifests": contract.get("guidance_manifests"),
    })
    return out_path



def write_validation_invariance_summary(cfg: Any, report: Dict[str, Any], *, final_native: Optional[Path], final_for_user: Optional[Path | str], final_provenance: Optional[Path | str], logger: Optional[logging.Logger] = None, enforce_hard_fail: bool = True) -> Optional[Path]:
    log = logger or logging.getLogger(__name__)
    try:
        manifest_path = Path(cfg.out_dir) / "final_outputs.json"
        if not manifest_path.exists():
            return None
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            log.debug("[VALIDATION] Failed reading final outputs manifest", exc_info=True)
            return None
        if not isinstance(manifest_payload, dict):
            return None
        selected_final = manifest_payload.get("selected_final_depth") or manifest_payload.get("final_depth_user") or manifest_payload.get("final_depth_native")
        if not selected_final:
            log.info("[VALIDATION] Skipping validation/invariance summary because no selected final depth exists in final_outputs manifest.")
            report.setdefault("validation", {}).update({
                "status": "skipped_no_selected_final_depth",
                "final_outputs_manifest": str(manifest_path),
            })
            return None
        payload = run_validation_invariance_framework(
            final_outputs_manifest=manifest_path,
            overlap_identity_evaluation=report.get("seams", {}).get("overlap_identity_evaluation") if isinstance(report.get("seams", {}), dict) else None,
            validation_truth=str(getattr(cfg, "validation_truth", None)) if getattr(cfg, "validation_truth", None) else None,
            case_specs=list(getattr(cfg, "validation_case_specs", []) or []),
            case_manifest=str(getattr(cfg, "validation_case_manifest", None)) if getattr(cfg, "validation_case_manifest", None) else None,
            guidance_baseline_case=str(getattr(cfg, "validation_guidance_baseline_case", "baseline_cudem_interpolation")),
            guidance_target_case=str(getattr(cfg, "validation_guidance_target_case", "selected_final")),
            require_guidance_non_degradation=bool(getattr(cfg, "validation_require_guidance_non_degradation", False)),
            guidance_rmse_tolerance=float(getattr(cfg, "validation_guidance_rmse_tolerance", 0.0) or 0.0),
        )
        if payload.get("status") == "not_run":
            report.setdefault("optional_validation", {})["validation_invariance"] = payload
            log.debug("[VALIDATION] not run; no validation truth/cases or overlap identity were configured")
            return None

        out_path = Path(cfg.out_dir) / "validation_invariance_summary.json"
        write_json(out_path, payload)
        report.setdefault("outputs", {})["validation_invariance_summary"] = str(out_path)
        report.setdefault("validation", {}).update(payload)
        sci_path = Path(cfg.out_dir) / "scientific_validation_summary.json"
        sci_payload = write_scientific_validation_summary(
            out_path=sci_path,
            final_outputs_manifest=manifest_path,
            overlap_identity_evaluation=report.get("seams", {}).get("overlap_identity_evaluation") if isinstance(report.get("seams", {}), dict) else None,
            validation_truth=str(getattr(cfg, "validation_truth", None)) if getattr(cfg, "validation_truth", None) else None,
            case_specs=list(getattr(cfg, "validation_case_specs", []) or []),
            case_manifest=str(getattr(cfg, "validation_case_manifest", None)) if getattr(cfg, "validation_case_manifest", None) else None,
            guidance_baseline_case=str(getattr(cfg, "validation_guidance_baseline_case", "baseline_cudem_interpolation")),
            guidance_target_case=str(getattr(cfg, "validation_guidance_target_case", "selected_final")),
            guidance_rmse_tolerance=float(getattr(cfg, "validation_guidance_rmse_tolerance", 0.0) or 0.0),
        )
        if sci_payload.get("status") == "not_run":
            report.setdefault("optional_validation", {})["scientific_validation"] = sci_payload
        else:
            report.setdefault("outputs", {})["scientific_validation_summary"] = str(sci_path)
            report.setdefault("validation", {})["scientific_validation_summary"] = sci_payload
        if payload.get("all_hard_invariants_ok") is False:
            reasons = "; ".join(payload.get("hard_failures", []))
            if enforce_hard_fail:
                raise RuntimeError(f"Validation/invariance framework hard-failed: {reasons}")
            log.info("[VALIDATION] Recorded pre-seam hard invariant failure for later enforcement: %s", reasons)
        return out_path
    except RuntimeError:
        raise
    except Exception:
        log.debug("[VALIDATION] Failed writing validation/invariance summary", exc_info=True)
        return None





def evaluate_overlap_identity_checks(overlap_checks: list[dict] | None, *, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Evaluate overlap identity results and decide whether they pass the stability contract."""
    checks = list(overlap_checks or [])
    failures = []
    ok_count = 0
    skipped_no_valid_count = 0
    for check in checks:
        status = check.get('status')
        if status == 'no_valid':
            skipped_no_valid_count += 1
            continue
        if status != 'ok':
            failures.append({
                'artifact': check.get('artifact'),
                'neighbor_io_manifest': check.get('neighbor_io_manifest'),
                'status': status,
                'reason': f'non-ok status: {status}',
            })
            continue
        ok_count += 1
        max_abs = check.get('max_abs')
        if max_abs is not None and float(max_abs) > float(tolerance):
            failures.append({
                'artifact': check.get('artifact'),
                'neighbor_io_manifest': check.get('neighbor_io_manifest'),
                'status': status,
                'max_abs': float(max_abs),
                'tolerance': float(tolerance),
                'reason': f'max_abs {float(max_abs):.12g} exceeds tolerance {float(tolerance):.12g}',
            })
    all_ok = None
    if ok_count > 0:
        all_ok = len(failures) == 0
    return {
        'checked': len(checks),
        'ok_count': int(ok_count),
        'skipped_no_valid_count': int(skipped_no_valid_count),
        'tolerance': float(tolerance),
        'all_ok': all_ok,
        'failures': failures,
    }

def write_river_stability_summary(cfg: Any, report: Dict[str, Any], *, logger: Optional[logging.Logger] = None) -> Optional[Path]:
    log = logger or logging.getLogger(__name__)
    try:
        river_guidance = report.get('river', {}).get('guidance', {}) if isinstance(report.get('river', {}), dict) else {}
        outputs = report.get('outputs', {}) if isinstance(report.get('outputs', {}), dict) else {}
        scaffold_path = river_guidance.get('scaffold_domains') or outputs.get('river_scaffold_domains')
        trusted_summary_path = outputs.get('river_trusted_interior_summary')
        scaffold_payload = None
        trusted_payload = None
        if scaffold_path and Path(scaffold_path).exists():
            scaffold_payload = __import__('json').loads(Path(scaffold_path).read_text(encoding='utf-8'))
        if trusted_summary_path and Path(trusted_summary_path).exists():
            trusted_payload = __import__('json').loads(Path(trusted_summary_path).read_text(encoding='utf-8'))
        seam_results = report.get('seams', {}).get('adjacent_tile_comparisons', []) if isinstance(report.get('seams', {}), dict) else []
        overlap_checks = report.get('seams', {}).get('overlap_identity_checks', []) if isinstance(report.get('seams', {}), dict) else []
        trusted_checks = report.get('seams', {}).get('trusted_interior_identity_checks', []) if isinstance(report.get('seams', {}), dict) else []
        nested = nested_aoi_relationship(str(cfg.aoi), scaffold_payload.get('solve_aoi')) if scaffold_payload and scaffold_payload.get('solve_aoi') else None
        overlap_eval = evaluate_overlap_identity_checks(overlap_checks)
        trusted_eval = evaluate_overlap_identity_checks(trusted_checks)
        payload = {
            'aoi': str(cfg.aoi),
            'scaffold_contract': scaffold_payload,
            'trusted_interior_contract': trusted_payload,
            'nested_aoi_relationship_to_solve_domain': nested,
            'adjacent_tile_seam_checks': seam_results,
            'overlap_identity_checks': overlap_checks,
            'overlap_identity_evaluation': overlap_eval,
            'all_overlap_identity_ok': overlap_eval.get('all_ok'),
            'trusted_interior_identity_checks': trusted_checks,
            'trusted_interior_identity_evaluation': trusted_eval,
            'all_trusted_interior_identity_ok': trusted_eval.get('all_ok'),
        }
        out = Path(cfg.out_dir) / 'river_stability_summary.json'
        write_json(out, payload)
        report.setdefault('outputs', {})['river_stability_summary'] = str(out)
        return out
    except Exception:
        log.debug('[STABILITY] Failed writing river stability summary', exc_info=True)
        return None
