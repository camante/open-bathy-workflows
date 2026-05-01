from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.json_io import write_json
from output_products import build_final_output_contract
from reporting.provenance_reporting import _class_summary, _grid_pixel_size_m, _read_raster, _float_array
from river_primary_surface_contract import RIVER_PRIMARY_SURFACE_SOURCE_NAMES
from provenance_schema import provenance_class_code_from_name, provenance_schema_summary
from support_classes import SupportClass, RegimeClass, support_class_code_from_name, support_schema_summary


def _crosstab(a: np.ndarray, b: np.ndarray, *, domain: np.ndarray, a_codes: Dict[str, str], b_codes: Dict[str, str], pixel_area_m2: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    valid = domain & np.isfinite(a) & np.isfinite(b)
    total = int(np.count_nonzero(valid))
    for a_code_str, a_label in (a_codes or {}).items():
        try:
            a_code = int(a_code_str)
        except (TypeError, ValueError):
            continue
        a_mask = valid & (a == a_code)
        if not np.any(a_mask):
            continue
        row: Dict[str, Any] = {"label": str(a_label), "count": int(np.count_nonzero(a_mask)), "by_class": {}}
        for b_code_str, b_label in (b_codes or {}).items():
            try:
                b_code = int(b_code_str)
            except (TypeError, ValueError):
                continue
            mask = a_mask & (b == b_code)
            count = int(np.count_nonzero(mask))
            if count <= 0:
                continue
            row["by_class"][str(b_code)] = {
                "label": str(b_label),
                "count": count,
                "area_m2": float(count * pixel_area_m2),
                "fraction_of_domain": float(count / max(total, 1)),
                "fraction_of_row": float(count / max(row["count"], 1)),
            }
        out[str(a_code)] = row
    return {"total_count": total, "by_class": out}


def _safe_read_array(path_value: Any) -> Optional[np.ndarray]:
    try:
        if not path_value:
            return None
        arr, _, _, _ = _read_raster(Path(path_value))
        return np.asarray(arr)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return None


def _binary_crosswalk(
    arr: Optional[np.ndarray],
    *,
    domain: np.ndarray,
    support_arr: np.ndarray,
    support_codes: Dict[str, str],
    prov_arr: np.ndarray,
    prov_codes: Dict[str, str],
    pixel_area_m2: float,
) -> Dict[str, Any]:
    if arr is None:
        return {"available": False}
    arr_u8 = (np.asarray(arr) > 0).astype(np.uint8)
    labels = {"0": "absent", "1": "present"}
    return {
        "available": True,
        "support": _crosstab(arr_u8, np.asarray(support_arr), domain=domain, a_codes=labels, b_codes=support_codes, pixel_area_m2=pixel_area_m2),
        "provenance": _crosstab(arr_u8, np.asarray(prov_arr), domain=domain, a_codes=labels, b_codes=prov_codes, pixel_area_m2=pixel_area_m2),
    }


def write_final_support_regime_audit(
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
        ab_outputs = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
        river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
        support_path = support_artifacts.get("support_class") or ab_outputs.get("support_class")
        regime_path = support_artifacts.get("regime_class") or ab_outputs.get("regime_class")
        prov_path = contract.get("selected_final_provenance")
        if not final_path or not support_path or not regime_path or not prov_path:
            return None

        final_arr, final_nodata, transform, crs = _read_raster(Path(final_path))
        final_arr = _float_array(final_arr, final_nodata)
        support_arr, _, _, _ = _read_raster(Path(support_path))
        regime_arr, _, _, _ = _read_raster(Path(regime_path))
        prov_arr, _, _, _ = _read_raster(Path(prov_path))
        domain = np.isfinite(final_arr)
        if not np.any(domain):
            return None

        ref_lat = (cfg.tile_bbox[1] + cfg.tile_bbox[3]) / 2.0 if getattr(cfg, "tile_bbox", None) else None
        dx_m, dy_m = _grid_pixel_size_m(transform, crs, ref_lat_deg=ref_lat)
        pixel_area_m2 = float(max(dx_m, 1.0) * max(dy_m, 1.0))

        policy = report.get("authoritative_base", {}).get("policy", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
        support_codes = policy.get("support_class_codes", {}) if isinstance(policy.get("support_class_codes", {}), dict) else {}
        support_families = policy.get("support_class_families", {}) if isinstance(policy.get("support_class_families", {}), dict) else {}
        regime_codes = policy.get("regime_class_codes", {}) if isinstance(policy.get("regime_class_codes", {}), dict) else {}
        prov_codes = policy.get("provenance_class_codes", {}) if isinstance(policy.get("provenance_class_codes", {}), dict) else {}
        prov_families = policy.get("provenance_class_families", {}) if isinstance(policy.get("provenance_class_families", {}), dict) else {}

        support_summary = _class_summary(np.asarray(support_arr), support_codes, support_families, domain=domain, pixel_area_m2=pixel_area_m2)
        regime_summary = _class_summary(np.asarray(regime_arr), regime_codes, regime_codes, domain=domain, pixel_area_m2=pixel_area_m2)
        provenance_summary = _class_summary(np.asarray(prov_arr), prov_codes, prov_families, domain=domain, pixel_area_m2=pixel_area_m2)

        support_regime = _crosstab(np.asarray(support_arr), np.asarray(regime_arr), domain=domain, a_codes=support_codes, b_codes=regime_codes, pixel_area_m2=pixel_area_m2)
        provenance_regime = _crosstab(np.asarray(prov_arr), np.asarray(regime_arr), domain=domain, a_codes=prov_codes, b_codes=regime_codes, pixel_area_m2=pixel_area_m2)

        locked_code = int(SupportClass.AUTHORITATIVE_LOCKED)
        locked_prov_code = provenance_class_code_from_name("authoritative_locked")
        estuary_code = int(RegimeClass.ESTUARY_TRANSITION)
        river_code = int(RegimeClass.RIVER_CHANNEL)
        sdb_code = int(SupportClass.GUIDANCE_CONDITIONED_SDB)
        river_guided_code = int(SupportClass.GUIDANCE_CONDITIONED_RIVER)
        scaffold_code = int(SupportClass.SCAFFOLD_INFERRED)
        low_conf_code = int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)

        estuary_scaffold_count = int(np.count_nonzero(domain & (regime_arr == estuary_code) & (support_arr == scaffold_code)))
        river_lock_path = river_outputs.get("channel_surface_authoritative_lock_applied")
        river_lock_arr = _safe_read_array(river_lock_path)
        river_lock_checks: Dict[str, Any] = {}
        if river_lock_arr is not None:
            river_lock_mask = domain & (np.asarray(river_lock_arr) > 0)
            river_lock_checks = {
                "runtime_river_authoritative_lock_pixels": int(np.count_nonzero(river_lock_mask)),
                "runtime_river_authoritative_lock_pixels_with_authoritative_support_class": int(np.count_nonzero(river_lock_mask & (support_arr == locked_code))),
                "runtime_river_authoritative_lock_pixels_without_authoritative_support_class": int(np.count_nonzero(river_lock_mask & (support_arr != locked_code))),
                "runtime_river_authoritative_lock_pixels_with_nonlocked_provenance": int(np.count_nonzero(river_lock_mask & (prov_arr != locked_prov_code))),
                "runtime_river_authoritative_lock_pixels_in_river_regime": int(np.count_nonzero(river_lock_mask & (regime_arr == river_code))),
            }
        elif river_lock_path:
            river_lock_checks = {"runtime_river_authoritative_lock_pixels": 0, "runtime_river_authoritative_lock_read_error": True}

        river_domain = domain & (regime_arr == river_code)
        river_primary_source_path = support_artifacts.get("river_primary_surface_source_class") or ab_outputs.get("river_primary_surface_source_class")
        river_primary_source_arr = _safe_read_array(river_primary_source_path)
        xs_participation_arr = _safe_read_array(river_outputs.get("channel_surface_xs_participation"))
        authoritative_participation_arr = _safe_read_array(river_outputs.get("channel_surface_authoritative_participation"))
        prediction_admissibility_arr = _safe_read_array(river_outputs.get("channel_surface_prediction_admissibility"))
        low_support_caution_arr = _safe_read_array(river_outputs.get("channel_surface_low_support_caution"))

        river_crosswalk: Dict[str, Any] = {
            "domain_pixels": int(np.count_nonzero(river_domain)),
            "artifacts": {
                "river_primary_surface_source_class": str(river_primary_source_path) if river_primary_source_path else None,
                "channel_surface_xs_participation": str(river_outputs.get("channel_surface_xs_participation")) if river_outputs.get("channel_surface_xs_participation") else None,
                "channel_surface_authoritative_participation": str(river_outputs.get("channel_surface_authoritative_participation")) if river_outputs.get("channel_surface_authoritative_participation") else None,
                "channel_surface_authoritative_lock_applied": str(river_lock_path) if river_lock_path else None,
                "channel_surface_prediction_admissibility": str(river_outputs.get("channel_surface_prediction_admissibility")) if river_outputs.get("channel_surface_prediction_admissibility") else None,
                "channel_surface_low_support_caution": str(river_outputs.get("channel_surface_low_support_caution")) if river_outputs.get("channel_surface_low_support_caution") else None,
            },
            "support_by_primary_surface_source": {"available": False},
            "provenance_by_primary_surface_source": {"available": False},
            "support_by_xs_participation": {"available": False},
            "provenance_by_xs_participation": {"available": False},
            "support_by_authoritative_participation": {"available": False},
            "provenance_by_authoritative_participation": {"available": False},
            "support_by_runtime_authoritative_lock": {"available": False},
            "provenance_by_runtime_authoritative_lock": {"available": False},
            "support_by_prediction_admissibility": {"available": False},
            "provenance_by_prediction_admissibility": {"available": False},
            "support_by_low_support_caution": {"available": False},
            "provenance_by_low_support_caution": {"available": False},
        }
        if river_primary_source_arr is not None:
            river_primary_codes = {str(k): str(v) for k, v in RIVER_PRIMARY_SURFACE_SOURCE_NAMES.items()}
            river_crosswalk["support_by_primary_surface_source"] = {
                "available": True,
                "support": _crosstab(np.asarray(river_primary_source_arr), np.asarray(support_arr), domain=river_domain, a_codes=river_primary_codes, b_codes=support_codes, pixel_area_m2=pixel_area_m2),
            }
            river_crosswalk["provenance_by_primary_surface_source"] = {
                "available": True,
                "provenance": _crosstab(np.asarray(river_primary_source_arr), np.asarray(prov_arr), domain=river_domain, a_codes=river_primary_codes, b_codes=prov_codes, pixel_area_m2=pixel_area_m2),
            }

        xs_crosswalk = _binary_crosswalk(
            xs_participation_arr,
            domain=river_domain,
            support_arr=np.asarray(support_arr),
            support_codes=support_codes,
            prov_arr=np.asarray(prov_arr),
            prov_codes=prov_codes,
            pixel_area_m2=pixel_area_m2,
        )
        river_crosswalk["support_by_xs_participation"] = {"available": xs_crosswalk.get("available", False), "support": xs_crosswalk.get("support")}
        river_crosswalk["provenance_by_xs_participation"] = {"available": xs_crosswalk.get("available", False), "provenance": xs_crosswalk.get("provenance")}

        auth_crosswalk = _binary_crosswalk(
            authoritative_participation_arr,
            domain=river_domain,
            support_arr=np.asarray(support_arr),
            support_codes=support_codes,
            prov_arr=np.asarray(prov_arr),
            prov_codes=prov_codes,
            pixel_area_m2=pixel_area_m2,
        )
        river_crosswalk["support_by_authoritative_participation"] = {"available": auth_crosswalk.get("available", False), "support": auth_crosswalk.get("support")}
        river_crosswalk["provenance_by_authoritative_participation"] = {"available": auth_crosswalk.get("available", False), "provenance": auth_crosswalk.get("provenance")}

        lock_crosswalk = _binary_crosswalk(
            river_lock_arr,
            domain=river_domain,
            support_arr=np.asarray(support_arr),
            support_codes=support_codes,
            prov_arr=np.asarray(prov_arr),
            prov_codes=prov_codes,
            pixel_area_m2=pixel_area_m2,
        )
        river_crosswalk["support_by_runtime_authoritative_lock"] = {"available": lock_crosswalk.get("available", False), "support": lock_crosswalk.get("support")}
        river_crosswalk["provenance_by_runtime_authoritative_lock"] = {"available": lock_crosswalk.get("available", False), "provenance": lock_crosswalk.get("provenance")}

        pred_adm_crosswalk = _binary_crosswalk(
            prediction_admissibility_arr,
            domain=river_domain,
            support_arr=np.asarray(support_arr),
            support_codes=support_codes,
            prov_arr=np.asarray(prov_arr),
            prov_codes=prov_codes,
            pixel_area_m2=pixel_area_m2,
        )
        river_crosswalk["support_by_prediction_admissibility"] = {"available": pred_adm_crosswalk.get("available", False), "support": pred_adm_crosswalk.get("support")}
        river_crosswalk["provenance_by_prediction_admissibility"] = {"available": pred_adm_crosswalk.get("available", False), "provenance": pred_adm_crosswalk.get("provenance")}

        caution_crosswalk = _binary_crosswalk(
            low_support_caution_arr,
            domain=river_domain,
            support_arr=np.asarray(support_arr),
            support_codes=support_codes,
            prov_arr=np.asarray(prov_arr),
            prov_codes=prov_codes,
            pixel_area_m2=pixel_area_m2,
        )
        river_crosswalk["support_by_low_support_caution"] = {"available": caution_crosswalk.get("available", False), "support": caution_crosswalk.get("support")}
        river_crosswalk["provenance_by_low_support_caution"] = {"available": caution_crosswalk.get("available", False), "provenance": caution_crosswalk.get("provenance")}

        suspicious = {
            "locked_pixels_with_nonlocked_provenance": int(np.count_nonzero(domain & (support_arr == locked_code) & (prov_arr != locked_prov_code))),
            "estuary_pixels_using_scaffold_inferred": estuary_scaffold_count,
            "estuary_pixels_using_channel_structure_guidance": estuary_scaffold_count,
            "river_pixels_without_river_support_class": int(np.count_nonzero(domain & (regime_arr == river_code) & ~np.isin(support_arr, [locked_code, river_guided_code, scaffold_code, low_conf_code]))),
            "nearshore_pixels_using_river_guidance": int(np.count_nonzero(domain & (regime_arr == int(RegimeClass.NEARSHORE_WATER)) & np.isin(support_arr, [river_guided_code, scaffold_code]))),
            **river_lock_checks,
        }

        support_family_counts = support_summary.get("by_family", {}) if isinstance(support_summary.get("by_family"), dict) else {}
        audit = {
            "selected_final_depth": str(final_path),
            "support_schema": support_schema_summary(),
            "provenance_schema": provenance_schema_summary(),
            "selected_final_provenance": str(prov_path),
            "pixel_area_m2": pixel_area_m2,
            "total_domain_pixels": int(np.count_nonzero(domain)),
            "support_class_summary": support_summary,
            "regime_class_summary": regime_summary,
            "provenance_class_summary": provenance_summary,
            "support_by_regime": support_regime,
            "provenance_by_regime": provenance_regime,
            "river_support_crosswalk": river_crosswalk,
            "late_stage_contract_checks": suspicious,
            "headline": {
                "authoritative_locked_pixels": int(np.count_nonzero(domain & (support_arr == locked_code))),
                "anchored_or_guided_pixels": int(sum(v.get("count", 0) for k, v in support_family_counts.items() if k in {"anchored_interpolation", "guidance_conditioned", "scaffold_inferred"})),
                "low_confidence_continuous_fill_pixels": int(np.count_nonzero(domain & (support_arr == low_conf_code))),
                "sdb_guided_pixels": int(np.count_nonzero(domain & (support_arr == sdb_code))),
                "river_guided_pixels": int(np.count_nonzero(domain & (support_arr == river_guided_code))),
                "channel_structure_guidance_pixels": int(np.count_nonzero(domain & (support_arr == scaffold_code))),
            },
            "final_generation_route": report.get("final_dem_runtime", {}).get("final_generation_route"),
        }

        out_path = Path(cfg.out_dir) / "final_support_regime_audit.json"
        write_json(out_path, audit)
        report.setdefault("outputs", {})["final_support_regime_audit"] = str(out_path)
        report.setdefault("final_dem_runtime", {}).setdefault("support_reporting", {}).update({
            "final_support_regime_audit": str(out_path),
            "late_stage_contract_checks": suspicious,
            "river_support_crosswalk": {
                "domain_pixels": river_crosswalk["domain_pixels"],
                "artifacts": river_crosswalk["artifacts"],
            },
        })
        return out_path
    except ImportError:
        log.debug("[FINAL_SUPPORT_AUDIT] rasterio unavailable; skipping", exc_info=True)
        return None
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        log.debug("[FINAL_SUPPORT_AUDIT] Failed writing final support/regime audit", exc_info=True)
        return None


__all__ = ["write_final_support_regime_audit"]
