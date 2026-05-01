from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.json_io import write_json
from output_products import build_final_output_contract


def _grid_pixel_size_m(transform, crs, ref_lat_deg: Optional[float] = None) -> tuple[float, float]:
    dx = abs(float(getattr(transform, "a", 0.0) or 0.0))
    dy = abs(float(getattr(transform, "e", 0.0) or 0.0))
    px = max((dx + dy) / 2.0, 0.0)
    is_geographic = bool(getattr(crs, "is_geographic", False))
    if not is_geographic:
        return max(dx, 1.0), max(dy, 1.0)
    lat = float(ref_lat_deg if ref_lat_deg is not None else 0.0)
    lat_rad = np.deg2rad(lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2.0 * lat_rad) + 1.175 * np.cos(4.0 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3.0 * lat_rad)
    dx_m = dx * max(abs(m_per_deg_lon), 1.0)
    dy_m = dy * max(abs(m_per_deg_lat), 1.0)
    return max(dx_m, 1.0), max(dy_m, 1.0)


def _read_raster(path: Path):
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        transform = ds.transform
        crs = ds.crs
    return arr, nodata, transform, crs


def _float_array(arr: np.ndarray, nodata: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if nodata is not None:
        try:
            out[np.isclose(out, np.float32(nodata))] = np.nan
        except (TypeError, ValueError):
            pass
    return out


def _class_summary(cls: np.ndarray, codes: Dict[str, str], families: Dict[str, str], *, domain: np.ndarray, pixel_area_m2: float) -> Dict[str, Any]:
    total = int(np.count_nonzero(domain))
    by_class: Dict[str, Any] = {}
    by_family: Dict[str, Any] = {}
    known_codes: set[int] = set()
    for code_str, label in (codes or {}).items():
        try:
            code = int(code_str)
        except (TypeError, ValueError):
            continue
        known_codes.add(code)
        mask = domain & (cls == code)
        count = int(np.count_nonzero(mask))
        if count <= 0:
            continue
        fam = str((families or {}).get(str(code)) or label)
        entry = {
            "label": str(label),
            "family": fam,
            "count": count,
            "area_m2": float(count * pixel_area_m2),
            "fraction_of_domain": float(count / max(total, 1)),
        }
        by_class[str(code)] = entry
        fam_entry = by_family.setdefault(fam, {"count": 0, "area_m2": 0.0})
        fam_entry["count"] += count
        fam_entry["area_m2"] += float(count * pixel_area_m2)
    unknown_mask = domain & np.isfinite(cls)
    if known_codes:
        unknown_mask &= ~np.isin(cls, list(known_codes))
    unknown_count = int(np.count_nonzero(unknown_mask))
    for fam, entry in by_family.items():
        entry["fraction_of_domain"] = float(entry["count"] / max(total, 1))
    summary = {"total_count": total, "pixel_area_m2": float(pixel_area_m2), "by_class": by_class, "by_family": by_family}
    if unknown_count > 0:
        summary["unknown_codes"] = {
            "count": unknown_count,
            "area_m2": float(unknown_count * pixel_area_m2),
            "fraction_of_domain": float(unknown_count / max(total, 1)),
            "codes": sorted(int(v) for v in np.unique(cls[unknown_mask]).tolist()),
        }
    return summary


def _continuous_summary(arr: np.ndarray, *, domain: np.ndarray) -> Optional[Dict[str, Any]]:
    vals = np.asarray(arr, dtype=np.float32)[domain & np.isfinite(arr)]
    if vals.size <= 0:
        return None
    return {
        "count": int(vals.size),
        "mean": float(np.nanmean(vals)),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
    }


def write_support_provenance_summary(
    cfg: Any,
    report: Dict[str, Any],
    *,
    final_native: Optional[Path],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    log = logger or logging.getLogger(__name__)
    try:
        contract = build_final_output_contract(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)
        final_path = contract.get("selected_final_native") or contract.get("selected_final_depth")
        support_artifacts = contract.get("support_artifacts", {}) if isinstance(contract.get("support_artifacts", {}), dict) else {}
        support_path = support_artifacts.get("support_class")
        prov_path = contract.get("selected_final_provenance")
        if not final_path or not support_path or not prov_path:
            return None

        final_arr, final_nodata, transform, crs = _read_raster(Path(final_path))
        final_arr = _float_array(final_arr, final_nodata)
        support_arr, _, _, _ = _read_raster(Path(support_path))
        prov_arr, _, _, _ = _read_raster(Path(prov_path))

        domain = np.isfinite(final_arr)
        if not np.any(domain):
            return None
        ref_lat = (cfg.tile_bbox[1] + cfg.tile_bbox[3]) / 2.0 if getattr(cfg, "tile_bbox", None) else None
        pixel_size_x_m, pixel_size_y_m = _grid_pixel_size_m(transform, crs, ref_lat_deg=ref_lat)
        pixel_area_m2 = float(max(pixel_size_x_m, 1.0) * max(pixel_size_y_m, 1.0))

        policy = report.get("authoritative_base", {}).get("policy", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
        support_codes = policy.get("support_class_codes", {}) if isinstance(policy.get("support_class_codes", {}), dict) else {}
        support_families = policy.get("support_class_families", {}) if isinstance(policy.get("support_class_families", {}), dict) else {}
        regime_codes = policy.get("regime_class_codes", {}) if isinstance(policy.get("regime_class_codes", {}), dict) else {}
        prov_codes = policy.get("provenance_class_codes", {}) if isinstance(policy.get("provenance_class_codes", {}), dict) else {}
        prov_families = policy.get("provenance_class_families", {}) if isinstance(policy.get("provenance_class_families", {}), dict) else {}

        summary: Dict[str, Any] = {
            "selected_final_depth": str(final_path),
            "selected_final_provenance": str(prov_path),
            "pixel_size_x_m": float(max(pixel_size_x_m, 1.0)),
            "pixel_size_y_m": float(max(pixel_size_y_m, 1.0)),
            "pixel_area_m2": pixel_area_m2,
            "support_class_summary": _class_summary(support_arr, support_codes, support_families, domain=domain, pixel_area_m2=pixel_area_m2),
            "provenance_class_summary": _class_summary(prov_arr, prov_codes, prov_families, domain=domain, pixel_area_m2=pixel_area_m2),
            "confidence_artifacts": {},
            "guidance_influence_summary": {},
            "regime_class_summary": {},
        }

        guidance_path = support_artifacts.get("guidance_influence")
        if guidance_path:
            g_arr, g_nodata, _, _ = _read_raster(Path(guidance_path))
            g_arr = _float_array(g_arr, g_nodata)
            g_stats = _continuous_summary(g_arr, domain=domain)
            if g_stats:
                summary["guidance_influence_summary"]["guidance_influence"] = g_stats
                nonzero = domain & np.isfinite(g_arr) & (g_arr > 0)
                summary["guidance_influence_summary"]["guided_fraction_of_domain"] = float(np.count_nonzero(nonzero) / max(np.count_nonzero(domain), 1))


        regime_sources_present = []
        final_regime = support_artifacts.get("regime_class")
        if final_regime:
            arr, _, _, _ = _read_raster(Path(final_regime))
            summary["regime_class_summary"]["final_regime_class"] = _class_summary(np.asarray(arr, dtype=np.uint8), regime_codes, regime_codes, domain=domain, pixel_area_m2=pixel_area_m2)
            regime_sources_present.append("final_regime_class")
        for key in ("sdb_regime_class", "river_regime_class"):
            cand = support_artifacts.get(key)
            if not cand:
                continue
            arr, _, _, _ = _read_raster(Path(cand))
            regime_summary = _class_summary(np.asarray(arr, dtype=np.uint8), regime_codes, regime_codes, domain=domain, pixel_area_m2=pixel_area_m2)
            summary["regime_class_summary"][key] = regime_summary
            regime_sources_present.append(key)
        summary["regime_sources_present"] = regime_sources_present

        for key in ("coastal_sdb_confidence", "river_scaffold_confidence"):
            cand = support_artifacts.get(key)
            if not cand:
                continue
            arr, nodata, _, _ = _read_raster(Path(cand))
            arr = _float_array(arr, nodata)
            stats = _continuous_summary(arr, domain=domain)
            if stats:
                positive = domain & np.isfinite(arr) & (arr > 0)
                stats["positive_fraction_of_domain"] = float(np.count_nonzero(positive) / max(np.count_nonzero(domain), 1))
                summary["confidence_artifacts"][key] = stats

        out_path = Path(cfg.out_dir) / "support_provenance_summary.json"
        write_json(out_path, summary)
        report.setdefault("outputs", {})["support_provenance_summary"] = str(out_path)
        report.setdefault("final_dem_runtime", {}).setdefault("support_reporting", {}).update({
            "support_provenance_summary": str(out_path),
            "has_support_class_summary": True,
            "has_provenance_class_summary": True,
            "confidence_artifact_keys": sorted(summary["confidence_artifacts"].keys()),
        })
        return out_path
    except ImportError:
        log.debug("[SUPPORT_REPORTING] rasterio unavailable; skipping support/provenance summary", exc_info=True)
        return None
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        log.debug("[SUPPORT_REPORTING] Failed writing support/provenance summary", exc_info=True)
        return None


__all__ = ["write_support_provenance_summary"]
