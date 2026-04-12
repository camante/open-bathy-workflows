from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyproj import CRS, Transformer

from support_classes import (
    SUPPORT_CLASS_CODE_TO_NAME,
    support_class_code_from_name,
    support_class_family_or_raise,
    support_class_is_final_authoritative,
    support_class_label_or_raise,
)
from authoritative_river_roles import code_to_role, ROLE_BED_CORE, ROLE_BED_INNER, ROLE_BANK_MARGIN
from river_withheld_support import attach_support_point_keys, summarize_role_counts
import logging

log = logging.getLogger(__name__)


def _assert_same_grid(*, reference, candidate, candidate_label: str) -> None:
    if reference.crs != candidate.crs:
        raise ValueError(
            f"Science-evaluation benchmark requires aligned rasters; CRS mismatch for {candidate_label}: "
            f"{candidate.crs} != {reference.crs}"
        )
    if reference.width != candidate.width or reference.height != candidate.height:
        raise ValueError(
            f"Science-evaluation benchmark requires aligned rasters; shape mismatch for {candidate_label}: "
            f"{candidate.width}x{candidate.height} != {reference.width}x{reference.height}"
        )
    if candidate.transform != reference.transform:
        raise ValueError(
            f"Science-evaluation benchmark requires aligned rasters; transform mismatch for {candidate_label}"
        )


def _support_class_code(name: str) -> Optional[int]:
    try:
        return support_class_code_from_name(name)
    except ValueError:
        return None


def _support_class_labels(values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in np.asarray(values):
        try:
            labels.append(support_class_label_or_raise(int(round(float(value)))))
        except (TypeError, ValueError):
            labels.append("unknown")
    return labels


def _is_non_authoritative_support_label(label: str) -> bool:
    try:
        code = support_class_code_from_name(str(label))
    except ValueError:
        return False
    return not support_class_is_final_authoritative(code)


def _is_unsupported_support_code(code: int) -> bool:
    try:
        family = support_class_family_or_raise(int(code))
    except ValueError:
        return False
    return family in {"guidance_conditioned", "scaffold_inferred", "low_confidence_continuous_fill"}


def _read_json_file(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        log.debug("benchmark_workflow_stage: failed reading json %s", path, exc_info=True)
        return {}




def _receipt_path(report: dict, section: str, key: str, out_dir: Path) -> Optional[Path]:
    section_dict = report.get(section, {}) if isinstance(report.get(section, {}), dict) else {}
    outputs = section_dict.get("outputs", {}) if isinstance(section_dict.get("outputs", {}), dict) else {}
    return _resolve_report_path(outputs.get(key), out_dir)


def _classify_river_lateral_accountability(*, weakest_role: Optional[str], thalweg_p95: float, inner_p95: float, bank_p95: float, section_p95: float) -> dict[str, Any]:
    weakest_role_p95 = np.nan
    role_map = {
        "thalweg": thalweg_p95,
        "inner_shape": inner_p95,
        "bank_edge": bank_p95,
    }
    if weakest_role in role_map and np.isfinite(role_map[weakest_role]):
        weakest_role_p95 = float(role_map[weakest_role])
    else:
        finite_candidates = [(name, value) for name, value in role_map.items() if np.isfinite(value)]
        if finite_candidates:
            weakest_role, weakest_role_p95 = max(finite_candidates, key=lambda kv: kv[1])
    bank_minus_inner = float(bank_p95 - inner_p95) if np.isfinite(bank_p95) and np.isfinite(inner_p95) else np.nan
    inner_minus_thalweg = float(inner_p95 - thalweg_p95) if np.isfinite(inner_p95) and np.isfinite(thalweg_p95) else np.nan
    if weakest_role == "bank_edge" and (not np.isfinite(bank_minus_inner) or bank_minus_inner > 0.05):
        dominant_issue = "bank_vs_inner_shape"
        reason = "Bank-edge agreement is worse than inner-shape agreement, so margin behavior remains the dominant lateral discrepancy."
    elif weakest_role == "inner_shape" and (not np.isfinite(inner_minus_thalweg) or inner_minus_thalweg > 0.05):
        dominant_issue = "inner_shape_width_propagation"
        reason = "Inner-shape agreement is worse than thalweg agreement, so the backbone signal is not spreading across the channel width strongly enough."
    elif weakest_role == "thalweg":
        dominant_issue = "thalweg_fit"
        reason = "Thalweg agreement is the weakest role, so longitudinal backbone preservation remains the dominant discrepancy."
    elif weakest_role == "inner_shape":
        dominant_issue = "inner_shape_width_propagation"
        reason = "Inner-shape agreement is the weakest role, so channel-width reconstruction remains the dominant lateral discrepancy."
    elif weakest_role == "bank_edge":
        dominant_issue = "bank_vs_inner_shape"
        reason = "Bank-edge agreement is the weakest role, so bank-margin behavior remains the dominant lateral discrepancy."
    else:
        dominant_issue = None
        reason = "No finite role-agreement comparisons were available."
    return {
        "available": bool((weakest_role is not None) or bool(np.isfinite(section_p95))),
        "weakest_role": weakest_role,
        "weakest_role_p95_abs_error_m": float(weakest_role_p95) if np.isfinite(weakest_role_p95) else np.nan,
        "dominant_issue": dominant_issue,
        "reason": reason,
        "thalweg_p95_abs_error_m": float(thalweg_p95) if np.isfinite(thalweg_p95) else np.nan,
        "inner_shape_p95_abs_error_m": float(inner_p95) if np.isfinite(inner_p95) else np.nan,
        "bank_edge_p95_abs_error_m": float(bank_p95) if np.isfinite(bank_p95) else np.nan,
        "section_target_p95_abs_error_m": float(section_p95) if np.isfinite(section_p95) else np.nan,
        "bank_minus_inner_p95_abs_error_m": bank_minus_inner,
        "inner_minus_thalweg_p95_abs_error_m": inner_minus_thalweg,
    }


def _compute_river_receipt_triage_summary(*, report: dict, out_dir: Path, river_scientific_summary: dict[str, Any]) -> dict[str, Any]:
    def _p95_from_bucket(bucket: Any) -> float:
        if not isinstance(bucket, dict):
            return np.nan
        abs_err = bucket.get("abs_error_m", {}) if isinstance(bucket.get("abs_error_m", {}), dict) else {}
        value = abs_err.get("p95", np.nan)
        return float(value) if value is not None and np.isfinite(value) else np.nan

    centerline = river_scientific_summary.get("centerline_agreement", {}) if isinstance(river_scientific_summary, dict) else {}
    backbone_path = _receipt_path(report, "river", "backbone_smoothing_summary", out_dir)
    transition_path = _receipt_path(report, "river", "channel_surface_authoritative_transition_summary", out_dir)
    smoothing_path = _receipt_path(report, "river", "channel_surface_longitudinal_smoothing_summary", out_dir)
    role_path = _receipt_path(report, "river", "channel_surface_role_agreement_summary", out_dir)
    section_target_path = _receipt_path(report, "river", "channel_surface_section_target_agreement_summary", out_dir)

    backbone = _read_json_file(backbone_path)
    transition = _read_json_file(transition_path)
    longitudinal_smoothing = _read_json_file(smoothing_path)
    role = _read_json_file(role_path)
    section_target = _read_json_file(section_target_path)

    role_groups = role.get("by_role_semantic_group", {}) if isinstance(role.get("by_role_semantic_group", {}), dict) else {}
    thalweg_p95 = _p95_from_bucket(role.get("thalweg_agreement", {})) if isinstance(role, dict) else np.nan
    inner_p95 = _p95_from_bucket(role.get("inner_shape_agreement", {})) if isinstance(role, dict) else np.nan
    bank_p95 = _p95_from_bucket(role.get("bank_edge_agreement", {})) if isinstance(role, dict) else np.nan
    if not np.isfinite(thalweg_p95):
        thalweg_p95 = _p95_from_bucket(role_groups.get("thalweg", {}))
    if not np.isfinite(inner_p95):
        inner_p95 = _p95_from_bucket(role_groups.get("inner_shape", {}))
    if not np.isfinite(bank_p95):
        bank_p95 = _p95_from_bucket(role_groups.get("bank_edge", {}))

    weakest_role = str(role.get("weakest_role")) if isinstance(role.get("weakest_role"), str) and role.get("weakest_role") else None
    role_candidates = [(name, value) for name, value in (("thalweg", thalweg_p95), ("inner_shape", inner_p95), ("bank_edge", bank_p95)) if np.isfinite(value)]
    if weakest_role is None and role_candidates:
        weakest_role = max(role_candidates, key=lambda kv: kv[1])[0]

    centerline_p95 = float(centerline.get("p95_abs_error_m", np.nan)) if isinstance(centerline, dict) else np.nan
    roughness = centerline.get("roughness", {}) if isinstance(centerline.get("roughness", {}), dict) else {}
    centerline_rough = float(roughness.get("roughness_ratio_p95_abs", np.nan)) if isinstance(roughness, dict) else np.nan

    section_abs = section_target.get("abs_error_m", {}) if isinstance(section_target.get("abs_error_m", {}), dict) else {}
    section_p95 = float(section_abs.get("p95", np.nan)) if isinstance(section_abs, dict) else np.nan
    if not np.isfinite(section_p95):
        section_p95 = float(section_target.get("weakest_role_class_p95_abs_error_m", np.nan)) if isinstance(section_target, dict) else np.nan

    lateral_accountability = _classify_river_lateral_accountability(
        weakest_role=weakest_role,
        thalweg_p95=thalweg_p95,
        inner_p95=inner_p95,
        bank_p95=bank_p95,
        section_p95=section_p95,
    )

    focus_signals: list[tuple[str, float]] = []
    if np.isfinite(centerline_p95):
        focus_signals.append(("centerline_alignment", centerline_p95))
    if np.isfinite(centerline_rough) and centerline_rough > 1.0:
        focus_signals.append(("centerline_roughness", centerline_rough))
    if np.isfinite(section_p95):
        focus_signals.append(("section_target_fit", section_p95))
    if lateral_accountability.get("dominant_issue") and np.isfinite(lateral_accountability.get("weakest_role_p95_abs_error_m", np.nan)):
        focus_signals.append((str(lateral_accountability.get("dominant_issue")), float(lateral_accountability.get("weakest_role_p95_abs_error_m"))))

    transition_nonzero = int(transition.get("nonzero_cell_count", 0)) if isinstance(transition, dict) else 0
    transition_dist = transition.get("weight_distribution", {}) if isinstance(transition.get("weight_distribution", {}), dict) else {}
    transition_mean = float(transition_dist.get("mean", np.nan)) if isinstance(transition_dist, dict) else np.nan
    transition_p95 = float(transition_dist.get("p95", np.nan)) if isinstance(transition_dist, dict) else np.nan
    smoothing_changed = int(longitudinal_smoothing.get("changed_count", 0)) if isinstance(longitudinal_smoothing, dict) else 0
    smoothing_eligible = int(longitudinal_smoothing.get("eligible_count", 0)) if isinstance(longitudinal_smoothing, dict) else 0
    smoothing_applied = int(longitudinal_smoothing.get("applied_count", 0)) if isinstance(longitudinal_smoothing, dict) else 0
    smoothing_adjust = longitudinal_smoothing.get("abs_adjustment_m", {}) if isinstance(longitudinal_smoothing.get("abs_adjustment_m", {}), dict) else {}
    smoothing_adjust_mean = float(smoothing_adjust.get("mean", np.nan)) if isinstance(smoothing_adjust, dict) else np.nan
    if smoothing_eligible > 0 and smoothing_changed == 0 and np.isfinite(centerline_rough) and centerline_rough > 1.05:
        focus_signals.append(("longitudinal_smoothing_activation", centerline_rough))

    primary_focus = None
    if focus_signals:
        scored: list[tuple[str, float]] = []
        for name, value in focus_signals:
            if not np.isfinite(value):
                continue
            score = float(value)
            if name == "centerline_roughness" and score <= 1.05:
                continue
            if name == "section_target_fit" and score <= 0.10:
                continue
            if name == "bank_vs_inner_shape" and score <= 0.10:
                continue
            if name == "inner_shape_width_propagation" and score <= 0.10:
                continue
            if name == "thalweg_fit" and score <= 0.10:
                continue
            scored.append((name, score))
        if scored:
            primary_focus = max(scored, key=lambda kv: kv[1])[0]

    suggested_actions = {
        "centerline_alignment": "Inspect backbone smoothing and thalweg-led render where final surface still drifts from reconciled centerline target.",
        "centerline_roughness": "Inspect unsupported reaches for residual along-channel oscillation; tighten support-aware backbone smoothing before changing lateral shape again.",
        "longitudinal_smoothing_activation": "Inspect the rendered longitudinal smoothing receipt; smoothing had eligible support-aware candidates but did not change the final weak-support thalweg render.",
        "section_target_fit": "Inspect local section-target construction and authoritative reconciliation weights where cross-channel target fit remains weak.",
        "bank_vs_inner_shape": "Inspect bank-edge semantics and margin damping if bank agreement error exceeds inner-shape error; banks may still be over-driving the rendered margins.",
        "inner_shape_width_propagation": "Inspect backbone-led inner-shape reconstruction if inner-shape error exceeds thalweg error; the centerline signal is still not spreading across the channel width strongly enough.",
        "thalweg_fit": "Inspect thalweg/backbone preservation in weak-support rebuild and render paths.",
        None: "No single dominant failure mode identified from current receipts.",
    }

    return {
        "available": True,
        "receipts": {
            "centerline_agreement": {
                "available": bool(centerline.get("available", False)) if isinstance(centerline, dict) else False,
                "mean_abs_error_m": float(centerline.get("mean_abs_error_m", np.nan)) if isinstance(centerline, dict) else np.nan,
                "p95_abs_error_m": centerline_p95,
                "roughness_ratio_p95_abs": centerline_rough,
            },
            "backbone_smoothing": {
                "available": bool(backbone.get("available", False)) if isinstance(backbone, dict) else False,
                "adjusted_station_count": int(backbone.get("adjusted_station_count", 0)) if isinstance(backbone, dict) else 0,
                "eligible_station_count": int(backbone.get("eligible_station_count", 0)) if isinstance(backbone, dict) else 0,
                "candidate_station_count": int(backbone.get("candidate_station_count", 0)) if isinstance(backbone, dict) else 0,
                "delta_abs_p95_m": float((backbone.get("delta_abs_m", {}) if isinstance(backbone.get("delta_abs_m", {}), dict) else {}).get("p95", np.nan)) if isinstance(backbone, dict) else np.nan,
                "weight_mean": float((backbone.get("weight_summary", {}) if isinstance(backbone.get("weight_summary", {}), dict) else {}).get("mean", np.nan)) if isinstance(backbone, dict) else np.nan,
                "reason": str(backbone.get("reason", "unknown")) if isinstance(backbone, dict) else "unavailable",
                "effectiveness_warning": backbone.get("effectiveness_warning") if isinstance(backbone, dict) else None,
            },
            "authoritative_transition": {
                "available": bool(transition.get("available", False)) if isinstance(transition, dict) else False,
                "nonzero_cell_count": transition_nonzero,
                "weight_mean": transition_mean,
                "weight_p95": transition_p95,
            },
            "longitudinal_smoothing": {
                "available": bool(longitudinal_smoothing.get("available", False)) if isinstance(longitudinal_smoothing, dict) else False,
                "eligible_count": smoothing_eligible,
                "applied_count": smoothing_applied,
                "changed_count": smoothing_changed,
                "abs_adjustment_mean_m": smoothing_adjust_mean,
                "reason": str(longitudinal_smoothing.get("reason", "unknown")) if isinstance(longitudinal_smoothing, dict) else "unavailable",
            },
            "role_agreement": {
                "available": bool(role.get("available", False)) if isinstance(role, dict) else False,
                "reason": str(role.get("reason", "unknown")) if isinstance(role, dict) else "unavailable",
                "weakest_role": weakest_role,
                "weakest_role_p95_abs_error_m": lateral_accountability.get("weakest_role_p95_abs_error_m", np.nan),
                "thalweg_p95_abs_error_m": thalweg_p95,
                "inner_shape_p95_abs_error_m": inner_p95,
                "bank_edge_p95_abs_error_m": bank_p95,
                "lateral_failure_mode": lateral_accountability.get("dominant_issue"),
                "lateral_failure_reason": lateral_accountability.get("reason"),
            },
            "section_target_agreement": {
                "available": bool(section_target.get("available", False)) if isinstance(section_target, dict) else False,
                "reason": str(section_target.get("reason", "unknown")) if isinstance(section_target, dict) else "unavailable",
                "comparison_node_count": int(section_target.get("comparison_node_count", 0)) if isinstance(section_target, dict) else 0,
                "weakest_role_class": section_target.get("weakest_role_class") if isinstance(section_target, dict) else None,
                "weakest_role_class_p95_abs_error_m": (lambda _v: float(_v) if _v is not None and np.isfinite(_v) else np.nan)(section_target.get("weakest_role_class_p95_abs_error_m", np.nan)) if isinstance(section_target, dict) else np.nan,
                "p95_abs_error_m": section_p95,
            },
        },
        "lateral_accountability": lateral_accountability,
        "primary_focus": primary_focus,
        "suggested_next_action": suggested_actions.get(primary_focus, "No single dominant failure mode identified from current receipts."),
        "artifacts": {
            "backbone_smoothing_summary": str(backbone_path) if backbone_path else None,
            "authoritative_transition_summary": str(transition_path) if transition_path else None,
            "channel_surface_longitudinal_smoothing_summary": str(smoothing_path) if smoothing_path else None,
            "role_agreement_summary": str(role_path) if role_path else None,
            "section_target_agreement_summary": str(section_target_path) if section_target_path else None,
        },
    }


def _resolve_authoritative_role_artifact(*, report: dict, cfg: Any, out_dir: Path, key: str, cfg_attr: str) -> Optional[Path]:
    auth_section = report.get("authoritative_base", {}).get("river_guidance", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    role_contract = _resolve_report_path(auth_section.get("role_contract"), out_dir)
    contract_payload = _read_json_file(role_contract)
    artifacts = contract_payload.get("artifacts", {}) if isinstance(contract_payload, dict) else {}
    cfg_value = getattr(cfg, cfg_attr, None)
    cfg_path = None
    if cfg_value:
        try:
            cfg_path = Path(str(cfg_value))
        except Exception:
            cfg_path = None
    candidates = [
        _resolve_report_path(auth_section.get(key), out_dir),
        _resolve_report_path(artifacts.get(key), out_dir),
        cfg_path if cfg_path and cfg_path.exists() else None,
    ]
    return _first_existing(*candidates)


def _authoritative_role_confidence_bin(value: float) -> Optional[str]:
    if not np.isfinite(value):
        return None
    if value >= 0.80:
        return "high"
    if value >= 0.50:
        return "medium"
    return "low"

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    log.debug("benchmark_workflow_stage: suppressed exception", exc_info=True)
    gpd = None


def _infer_col(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(str(cand).lower())
        if hit is not None:
            return str(hit)
    raise ValueError(f"Could not find {label} column. Tried {list(candidates)}. Columns={list(df.columns)}")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".gpkg", ".shp", ".geojson"}:
        if gpd is None:
            raise ValueError(f"Benchmark holdout format {suffix} requires geopandas: {path}")
        gdf = gpd.read_file(path)
        if gdf.empty:
            return pd.DataFrame()
        if gdf.geometry is not None:
            geom = gdf.geometry
            if getattr(geom, "x", None) is not None and getattr(geom, "y", None) is not None:
                gdf = gdf.copy()
                gdf["x"] = geom.x
                gdf["y"] = geom.y
        if gdf.crs is not None and gdf.crs.to_epsg() is not None and "crs" not in gdf.columns:
            gdf = gdf.copy()
            gdf["crs"] = f"EPSG:{gdf.crs.to_epsg()}"
        return pd.DataFrame(gdf.drop(columns=[c for c in ["geometry"] if c in gdf.columns]))
    raise ValueError(f"Unsupported benchmark holdout format: {path}")


def _sample_raster(raster_path: Path, x: np.ndarray, y: np.ndarray, src_epsg: int) -> np.ndarray:
    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        dst_epsg = ds.crs.to_epsg()
        if dst_epsg is None:
            raise ValueError(f"Raster CRS has no EPSG code: {raster_path} ({ds.crs})")
        if src_epsg != dst_epsg:
            tx = Transformer.from_crs(f"EPSG:{src_epsg}", ds.crs, always_xy=True)
            xs, ys = tx.transform(x, y)
        else:
            xs, ys = x, y
        vals = np.array([v[0] for v in ds.sample(list(zip(xs, ys)))], dtype="float64")
        nodata = ds.nodata
        if nodata is not None:
            vals[np.isclose(vals, nodata)] = np.nan
        return vals


def _sample_mask(raster_path: Optional[Path], x: np.ndarray, y: np.ndarray, src_epsg: int) -> Optional[np.ndarray]:
    if raster_path is None:
        return None
    vals = _sample_raster(raster_path, x, y, src_epsg=src_epsg)
    return np.isfinite(vals) & (vals != 0)




def _candidate_rows_available(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            _ = f.readline()
            for line in f:
                if line.strip():
                    return True
        return False
    except Exception:
        log.debug("benchmark_workflow_stage: failed candidate row availability check for %s", path, exc_info=True)
        return False



def _resolve_auto_holdout_candidate(*, args, report: dict, out_dir: Path) -> Optional[Path]:
    explicit = getattr(args, "benchmark_holdout", None)
    if explicit:
        p = Path(str(explicit)).resolve()
        return p if p.exists() else None

    auth = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base"), dict) else {}
    activation = report.get("method_activation_truth", {}) if isinstance(report.get("method_activation_truth"), dict) else {}

    river_active = bool(((activation.get("river") or {}).get("effective_should_run")))
    sdb_active = bool(((activation.get("sdb") or {}).get("effective_should_run")))
    priority = ["river_guidance", "sdb_guidance"] if river_active and not sdb_active else ["sdb_guidance", "river_guidance"] if sdb_active and not river_active else ["river_guidance", "sdb_guidance"]

    for sec_name in priority:
        sec = auth.get(sec_name, {}) if isinstance(auth.get(sec_name), dict) else {}
        cand = sec.get("path")
        if not cand:
            continue
        p = Path(str(cand))
        if not p.is_absolute():
            p = (out_dir / p).resolve()
        if not p.exists():
            continue
        explicit_support = sec.get("explicit_support") if isinstance(sec.get("explicit_support"), dict) else {}
        support_pixels = explicit_support.get("support_pixels")
        if support_pixels is not None and int(support_pixels) <= 0:
            continue
        if p.suffix.lower() == ".csv" and not _candidate_rows_available(p):
            continue
        return p
    return None


def _try_parse_epsg(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and int(value) > 0:
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("EPSG:"):
        text = text.split(":", 1)[1]
    try:
        epsg = int(float(text))
    except Exception:
        return None
    return epsg if epsg > 0 else None


def _resolve_report_candidate_path(path_value: Any, out_dir: Optional[Path]) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(str(path_value))
    if not p.is_absolute() and out_dir is not None:
        p = (out_dir / p).resolve()
    try:
        return p.resolve()
    except Exception:
        return p


def _resolve_candidate_points_epsg(*, candidate_path: Path, cfg, report: dict, out_dir: Optional[Path] = None) -> Optional[int]:
    auth = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base"), dict) else {}
    auth_inputs = auth.get("inputs", {}) if isinstance(auth.get("inputs"), dict) else {}
    working_srs_epsg = _try_parse_epsg(getattr(cfg, "working_srs", None))
    for section_name in ("river_guidance", "sdb_guidance"):
        sec = auth.get(section_name, {}) if isinstance(auth.get(section_name), dict) else {}
        sec_path = _resolve_report_candidate_path(sec.get("path"), out_dir)
        if sec_path is None:
            continue
        try:
            same = sec_path == Path(candidate_path).resolve()
        except Exception:
            same = False
        if not same and sec_path.name == Path(candidate_path).name:
            same = True
        if not same:
            continue
        for key in ("points_epsg", "epsg", "source_epsg", "working_epsg", "crs", "out_crs"):
            epsg = _try_parse_epsg(sec.get(key))
            if epsg is not None:
                return epsg
        contract_path = None
        explicit_support = sec.get("explicit_support") if isinstance(sec.get("explicit_support"), dict) else {}
        if explicit_support:
            contract_path = explicit_support.get("contract")
        if contract_path:
            try:
                payload = json.loads(Path(str(contract_path)).read_text(encoding="utf-8"))
                for key in ("out_crs", "crs", "points_epsg", "epsg", "source_epsg", "working_epsg"):
                    epsg = _try_parse_epsg(payload.get(key))
                    if epsg is not None:
                        return epsg
            except Exception:
                log.debug("benchmark_workflow_stage: failed to parse candidate contract CRS %s", contract_path, exc_info=True)
        # River and SDB authoritative support CSVs are written in the workflow working CRS.
        # Prefer that over the source authoritative raster CRS when no explicit point CRS exists.
        if working_srs_epsg is not None:
            return working_srs_epsg
        source_raster = auth_inputs.get("source") or getattr(cfg, "authoritative_base", None)
        if source_raster:
            try:
                with rasterio.open(str(source_raster)) as ds:
                    epsg = ds.crs.to_epsg() if ds.crs is not None else None
                if epsg is not None:
                    return int(epsg)
            except Exception:
                log.debug("benchmark_workflow_stage: failed to resolve candidate EPSG from source raster %s", source_raster, exc_info=True)
        break
    if working_srs_epsg is not None:
        return working_srs_epsg
    return None


def _infer_points_epsg_from_df(df: pd.DataFrame, x_name: str, y_name: str, *, baseline_raster: Path, final_raster: Path, explicit_epsg: Optional[int]) -> int:
    if explicit_epsg and int(explicit_epsg) > 0:
        return int(explicit_epsg)
    if "crs" in df.columns:
        vals = [str(v) for v in df["crs"].dropna().unique().tolist()]
        if len(vals) == 1 and vals[0].upper().startswith("EPSG:"):
            try:
                return int(vals[0].split(":", 1)[1])
            except Exception:
                log.debug("Failed to parse CRS from column value %s", vals[0], exc_info=True)
    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(df[y_name], errors="coerce").to_numpy(dtype="float64")
    return _infer_points_epsg(x=x, y=y, x_name=x_name, y_name=y_name, baseline_raster=baseline_raster, final_raster=final_raster)


def _transform_xy_to_raster_crs(x: np.ndarray, y: np.ndarray, src_epsg: int, raster_path: Path) -> tuple[np.ndarray, np.ndarray, Any]:
    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        dst_crs = ds.crs
        dst_epsg = dst_crs.to_epsg()
        if dst_epsg is None:
            raise ValueError(f"Raster CRS has no EPSG code: {raster_path} ({ds.crs})")
        if src_epsg != dst_epsg:
            tx = Transformer.from_crs(f"EPSG:{src_epsg}", dst_crs, always_xy=True)
            xx, yy = tx.transform(x, y)
        else:
            xx, yy = x, y
    return np.asarray(xx, dtype="float64"), np.asarray(yy, dtype="float64"), dst_crs




def _crs_is_projected(crs: Any) -> bool:
    if crs is None:
        return False
    val = getattr(crs, "is_projected", None)
    if isinstance(val, bool):
        return val
    try:
        return bool(CRS.from_user_input(crs).is_projected)
    except Exception:
        return False



def _crs_to_string(crs: Any) -> str:
    if crs is None:
        return ""
    val = getattr(crs, "to_string", None)
    if callable(val):
        try:
            return str(val())
        except Exception:
            pass
    try:
        return str(CRS.from_user_input(crs))
    except Exception:
        return str(crs)

def _estimate_template_pixel_size_m(template_path: Path) -> float:
    with rasterio.open(template_path) as ds:
        px = abs(float(ds.transform.a)) or 1.0
        py = abs(float(ds.transform.e)) or px
        if ds.crs is not None and _crs_is_projected(ds.crs):
            return max(px, py, 1.0)
        if ds.crs is None:
            return max(px, py, 1.0)
        cx = 0.5 * (float(ds.bounds.left) + float(ds.bounds.right))
        cy = 0.5 * (float(ds.bounds.bottom) + float(ds.bounds.top))
        metric_crs = CRS.from_epsg(3857)
        tx = Transformer.from_crs(ds.crs, metric_crs, always_xy=True)
        x0, y0 = tx.transform(cx, cy)
        x1, y1 = tx.transform(cx + px, cy)
        x2, y2 = tx.transform(cx, cy + py)
    return max(abs(float(x1) - float(x0)), abs(float(y2) - float(y0)), 1.0)


def _xy_in_metric_block_crs(x: np.ndarray, y: np.ndarray, src_epsg: int) -> tuple[np.ndarray, np.ndarray, str]:
    src_crs = CRS.from_epsg(int(src_epsg))
    if _crs_is_projected(src_crs):
        return np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"), _crs_to_string(src_crs)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise ValueError("Cannot build metric holdout blocks from non-finite coordinates")
    mean_lon = float(np.nanmean(x[finite]))
    mean_lat = float(np.nanmean(y[finite]))
    zone = int(math.floor((mean_lon + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    epsg = 32600 + zone if mean_lat >= 0.0 else 32700 + zone
    metric_crs = CRS.from_epsg(epsg)
    tx = Transformer.from_crs(src_crs, metric_crs, always_xy=True)
    mx, my = tx.transform(x, y)
    return np.asarray(mx, dtype="float64"), np.asarray(my, dtype="float64"), _crs_to_string(metric_crs)


def _build_auto_holdout(*, candidate_path: Path, bench_dir: Path, baseline_raster: Path, final_raster: Path, points_epsg_arg: Optional[int], fallback_points_epsg: Optional[int], x_col: Optional[str], y_col: Optional[str], z_col: Optional[str], holdout_frac: float, holdout_min_points: int, holdout_seed: int, logger) -> tuple[Path, int, str, str, str, dict[str, Any]]:
    df = _read_table(candidate_path)
    if df.empty:
        raise ValueError(f"Auto-holdout candidate pool is empty: {candidate_path}")
    x_name = x_col if x_col and x_col in df.columns else _infer_col(df, ["x", "X", "easting", "lon", "longitude"], "x")
    y_name = y_col if y_col and y_col in df.columns else _infer_col(df, ["y", "Y", "northing", "lat", "latitude"], "y")
    z_name = z_col if z_col and z_col in df.columns else _infer_col(df, ["z", "Z", "depth_m", "z_m", "depth", "elevation_m", "elev_m", "bed_z_m", "elevation", "bed_elevation_m"], "z/depth")
    points_epsg = _infer_points_epsg_from_df(df, x_name, y_name, baseline_raster=baseline_raster, final_raster=final_raster, explicit_epsg=(points_epsg_arg if points_epsg_arg and int(points_epsg_arg) > 0 else fallback_points_epsg))

    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(df[y_name], errors="coerce").to_numpy(dtype="float64")
    z = pd.to_numeric(df[z_name], errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    base = df.loc[finite].copy().reset_index(drop=True)
    if base.empty:
        raise ValueError(f"Auto-holdout candidate pool has no finite x/y/z rows: {candidate_path}")
    x = pd.to_numeric(base[x_name], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(base[y_name], errors="coerce").to_numpy(dtype="float64")
    xx, yy, block_crs = _xy_in_metric_block_crs(x, y, points_epsg)

    pixel_size_m = _estimate_template_pixel_size_m(baseline_raster)
    block = max(float(pixel_size_m * 25.0), 250.0)
    xmin = float(np.nanmin(xx))
    ymin = float(np.nanmin(yy))

    def _block_ids_for_size(block_size: float) -> np.ndarray:
        bx = np.floor((xx - xmin) / block_size).astype(int)
        by = np.floor((yy - ymin) / block_size).astype(int)
        return np.array([f"{a}_{b}" for a, b in zip(bx, by)], dtype=object)

    block_ids = _block_ids_for_size(block)
    uniq, counts = np.unique(block_ids, return_counts=True)
    target = max(int(math.ceil(len(base) * float(holdout_frac))), int(holdout_min_points))
    target = min(target, len(base))
    if len(uniq) <= 1 and len(base) > target:
        span_x = float(np.nanmax(xx) - xmin)
        span_y = float(np.nanmax(yy) - ymin)
        dominant_span = max(span_x, span_y)
        if dominant_span > 0.0:
            desired_blocks = max(int(math.ceil(1.0 / max(float(holdout_frac), 1.0e-6))), 4)
            adaptive_block = max(min(block, dominant_span / float(desired_blocks)), 1.0)
            adaptive_block_ids = _block_ids_for_size(adaptive_block)
            adaptive_uniq, adaptive_counts = np.unique(adaptive_block_ids, return_counts=True)
            if len(adaptive_uniq) > len(uniq):
                block = adaptive_block
                block_ids = adaptive_block_ids
                uniq, counts = adaptive_uniq, adaptive_counts

    base["_bench_block_id"] = block_ids
    rng = np.random.default_rng(int(holdout_seed))
    order = rng.permutation(len(uniq))
    chosen = []
    running = 0
    for idx in order:
        chosen.append(uniq[idx])
        running += int(counts[idx])
        if running >= target:
            break
    hold = base[base["_bench_block_id"].isin(chosen)].copy()
    hold["benchmark_holdout"] = 1
    hold["benchmark_holdout_method"] = "spatial_blocks"
    hold["benchmark_holdout_seed"] = int(holdout_seed)
    hold["benchmark_points_epsg"] = int(points_epsg)
    hold_path = bench_dir / "auto_holdout_points.csv"
    hold.to_csv(hold_path, index=False)
    receipt = {
        "candidate_path": str(candidate_path),
        "candidate_rows": int(len(df)),
        "finite_candidate_rows": int(len(base)),
        "points_epsg": int(points_epsg),
        "method": "spatial_blocks",
        "block_size_in_raster_crs_units": float(block),
        "block_size_m": float(block),
        "block_crs": str(block_crs),
        "template_pixel_size_m": float(pixel_size_m),
        "holdout_frac_requested": float(holdout_frac),
        "holdout_min_points": int(holdout_min_points),
        "holdout_seed": int(holdout_seed),
        "available_block_count": int(len(uniq)),
        "selected_block_count": int(len(chosen)),
        "holdout_rows": int(len(hold)),
        "holdout_fraction_realized": float(len(hold) / max(len(base), 1)),
        "x_col": x_name,
        "y_col": y_name,
        "z_col": z_name,
        "path": str(hold_path),
    }
    receipt_path = bench_dir / "auto_holdout_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    logger.info(
        "[BENCHMARK] Auto-holdout selected %s/%s rows (%.1f%%) using %s/%s spatial blocks (block=%.1fm, %s) from %s into %s",
        len(hold),
        len(base),
        100.0 * float(len(hold) / max(len(base), 1)),
        len(chosen),
        len(uniq),
        float(block),
        str(block_crs),
        candidate_path,
        hold_path,
    )
    return hold_path, points_epsg, x_name, y_name, z_name, receipt




def _select_spatial_blocks(*, xx: np.ndarray, yy: np.ndarray, target_count: int, seed: int, template_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    pixel_size_m = _estimate_template_pixel_size_m(template_path)
    block = max(float(pixel_size_m * 25.0), 250.0)
    xmin = float(np.nanmin(xx))
    ymin = float(np.nanmin(yy))

    def _block_ids_for_size(block_size: float) -> np.ndarray:
        bx = np.floor((xx - xmin) / block_size).astype(int)
        by = np.floor((yy - ymin) / block_size).astype(int)
        return np.array([f"{a}_{b}" for a, b in zip(bx, by)], dtype=object)

    block_ids = _block_ids_for_size(block)
    uniq, counts = np.unique(block_ids, return_counts=True)
    if len(uniq) <= 1 and len(xx) > target_count:
        span_x = float(np.nanmax(xx) - xmin)
        span_y = float(np.nanmax(yy) - ymin)
        dominant_span = max(span_x, span_y)
        if dominant_span > 0.0:
            desired_blocks = max(int(math.ceil(1.0 / max(float(target_count) / max(float(len(xx)), 1.0), 1.0e-6))), 4)
            adaptive_block = max(min(block, dominant_span / float(desired_blocks)), 1.0)
            adaptive_ids = _block_ids_for_size(adaptive_block)
            adaptive_uniq, adaptive_counts = np.unique(adaptive_ids, return_counts=True)
            if len(adaptive_uniq) > len(uniq):
                block = adaptive_block
                block_ids = adaptive_ids
                uniq, counts = adaptive_uniq, adaptive_counts

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(uniq))
    chosen: list[Any] = []
    running = 0
    for idx in order:
        chosen.append(uniq[idx])
        running += int(counts[idx])
        if running >= target_count:
            break
    mask = np.isin(block_ids, np.asarray(chosen, dtype=object))
    return mask, {
        "block_size_m": float(block),
        "available_block_count": int(len(uniq)),
        "selected_block_count": int(len(chosen)),
    }


def _build_river_withheld_support_plan(*, candidate_path: Path, bench_dir: Path, baseline_raster: Path, final_raster: Path, points_epsg_arg: Optional[int], fallback_points_epsg: Optional[int], x_col: Optional[str], y_col: Optional[str], z_col: Optional[str], holdout_frac: float, holdout_min_points: int, holdout_seed: int, logger) -> tuple[Optional[Path], dict[str, Any], Optional[int], Optional[str], Optional[str], Optional[str]]:
    receipt_path = bench_dir / "river_withheld_support_receipt.json"
    plan_path = bench_dir / "river_withheld_support_points.csv"
    df = _read_table(candidate_path)
    if df.empty:
        receipt = {"available": False, "reason": "candidate_pool_empty", "candidate_path": str(candidate_path)}
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return None, receipt, None, None, None, None

    x_name = x_col if x_col and x_col in df.columns else _infer_col(df, ["x", "X", "easting", "lon", "longitude"], "x")
    y_name = y_col if y_col and y_col in df.columns else _infer_col(df, ["y", "Y", "northing", "lat", "latitude"], "y")
    z_name = z_col if z_col and z_col in df.columns else _infer_col(df, ["z", "Z", "depth_m", "z_m", "depth", "elevation_m", "elev_m", "bed_z_m", "elevation", "bed_elevation_m"], "z/depth")
    points_epsg = _infer_points_epsg_from_df(df, x_name, y_name, baseline_raster=baseline_raster, final_raster=final_raster, explicit_epsg=(points_epsg_arg if points_epsg_arg and int(points_epsg_arg) > 0 else fallback_points_epsg))

    work = df.copy()
    x = pd.to_numeric(work[x_name], errors="coerce")
    y = pd.to_numeric(work[y_name], errors="coerce")
    z = pd.to_numeric(work[z_name], errors="coerce")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    work = work.loc[finite].copy().reset_index(drop=True)
    if work.empty:
        receipt = {"available": False, "reason": "no_finite_candidate_rows", "candidate_path": str(candidate_path)}
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return None, receipt, points_epsg, x_name, y_name, z_name

    work = attach_support_point_keys(work, key_col="support_point_key", x_col=x_name, y_col=y_name, z_col=z_name, role_col="authoritative_role")
    roles = work.get("authoritative_role", pd.Series("", index=work.index)).fillna("").astype(str)
    eligible = roles.isin({ROLE_BED_CORE, ROLE_BED_INNER})
    if "inside_channel_mask" in work.columns:
        eligible &= pd.to_numeric(work["inside_channel_mask"], errors="coerce").fillna(0.0) > 0.5
    if "inside_river_guidance_domain" in work.columns:
        eligible &= pd.to_numeric(work["inside_river_guidance_domain"], errors="coerce").fillna(0.0) > 0.5
    if "inside_estuary_clip" in work.columns:
        eligible &= pd.to_numeric(work["inside_estuary_clip"], errors="coerce").fillna(0.0) < 0.5
    if "role_confidence" in work.columns:
        eligible &= pd.to_numeric(work["role_confidence"], errors="coerce").fillna(0.0) >= 0.5

    eligible_df = work.loc[eligible].copy().reset_index(drop=True)
    if eligible_df.empty:
        receipt = {
            "available": False,
            "reason": "no_eligible_river_support_rows",
            "candidate_path": str(candidate_path),
            "candidate_rows": int(len(df)),
            "finite_candidate_rows": int(len(work)),
            "eligible_role_counts": summarize_role_counts(work),
        }
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return None, receipt, points_epsg, x_name, y_name, z_name

    mainstem_mask = pd.Series(False, index=eligible_df.index)
    if "inside_mainstem_mask" in eligible_df.columns:
        mainstem_mask = pd.to_numeric(eligible_df["inside_mainstem_mask"], errors="coerce").fillna(0.0) > 0.5

    target = max(int(math.ceil(len(eligible_df) * float(holdout_frac))), min(int(holdout_min_points), len(eligible_df)))
    target = min(target, len(eligible_df))
    selected_parts = []
    selection_meta: dict[str, Any] = {}
    strata = [("mainstem", eligible_df.loc[mainstem_mask].copy()), ("non_mainstem", eligible_df.loc[~mainstem_mask].copy())]
    total_eligible = max(len(eligible_df), 1)
    remaining_target = target
    for idx, (name, sub) in enumerate(strata):
        if sub.empty:
            selection_meta[name] = {"eligible_rows": 0, "selected_rows": 0}
            continue
        if idx == len(strata) - 1:
            sub_target = max(0, remaining_target)
        else:
            sub_target = int(round(target * (len(sub) / total_eligible)))
            if sub_target <= 0 and remaining_target > 0:
                sub_target = min(1, len(sub))
        sub_target = min(max(sub_target, 0), len(sub), remaining_target if remaining_target > 0 else len(sub))
        if sub_target <= 0:
            selection_meta[name] = {"eligible_rows": int(len(sub)), "selected_rows": 0}
            continue
        sx = pd.to_numeric(sub[x_name], errors="coerce").to_numpy(dtype="float64")
        sy = pd.to_numeric(sub[y_name], errors="coerce").to_numpy(dtype="float64")
        mx, my, block_crs = _xy_in_metric_block_crs(sx, sy, points_epsg)
        mask, meta = _select_spatial_blocks(xx=mx, yy=my, target_count=sub_target, seed=int(holdout_seed) + idx, template_path=baseline_raster)
        picked = sub.loc[mask].copy()
        if not picked.empty:
            picked["benchmark_withheld_support"] = 1
            picked["benchmark_withheld_support_stratum"] = name
            picked["benchmark_withheld_support_seed"] = int(holdout_seed)
            picked["benchmark_withheld_support_points_epsg"] = int(points_epsg)
            selected_parts.append(picked)
        selection_meta[name] = {
            "eligible_rows": int(len(sub)),
            "selected_rows": int(len(picked)),
            "block_crs": str(block_crs),
            **meta,
        }
        remaining_target -= int(len(picked))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else eligible_df.iloc[0:0].copy()
    if selected.empty:
        receipt = {"available": False, "reason": "selection_empty", "candidate_path": str(candidate_path), "eligible_rows": int(len(eligible_df))}
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return None, receipt, points_epsg, x_name, y_name, z_name

    selected.to_csv(plan_path, index=False)
    receipt = {
        "available": True,
        "candidate_path": str(candidate_path),
        "path": str(plan_path),
        "candidate_rows": int(len(df)),
        "finite_candidate_rows": int(len(work)),
        "eligible_rows": int(len(eligible_df)),
        "withheld_rows": int(len(selected)),
        "withheld_fraction_realized": float(len(selected) / max(len(eligible_df), 1)),
        "points_epsg": int(points_epsg),
        "x_col": x_name,
        "y_col": y_name,
        "z_col": z_name,
        "eligible_role_counts": summarize_role_counts(eligible_df),
        "withheld_role_counts": summarize_role_counts(selected),
        "strata": selection_meta,
        "support_point_key_column": "support_point_key",
        "status": "planned",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    logger.info("[BENCHMARK][RIVER] Withheld-support plan selected %d/%d eligible river support rows into %s", int(len(selected)), int(len(eligible_df)), plan_path)
    return plan_path, receipt, points_epsg, x_name, y_name, z_name


def _infer_points_epsg(*, x: np.ndarray, y: np.ndarray, x_name: str, y_name: str, baseline_raster: Path, final_raster: Path) -> int:
    x_lower = str(x_name).lower()
    y_lower = str(y_name).lower()
    looks_lonlat_name = x_lower in {"lon", "longitude"} or y_lower in {"lat", "latitude"}
    finite = np.isfinite(x) & np.isfinite(y)
    if np.any(finite):
        xf = x[finite]
        yf = y[finite]
        looks_lonlat_range = (np.nanmin(xf) >= -180.0 and np.nanmax(xf) <= 180.0 and np.nanmin(yf) >= -90.0 and np.nanmax(yf) <= 90.0)
        if looks_lonlat_name or looks_lonlat_range:
            return 4326

    for rp in (baseline_raster, final_raster):
        with rasterio.open(rp) as ds:
            if ds.crs is None:
                continue
            epsg = ds.crs.to_epsg()
            if epsg is None:
                continue
            if np.any(finite):
                xf = x[finite]
                yf = y[finite]
                left, bottom, right, top = ds.bounds
                if np.nanmin(xf) >= left and np.nanmax(xf) <= right and np.nanmin(yf) >= bottom and np.nanmax(yf) <= top:
                    return int(epsg)
    raise ValueError(
        "Could not infer benchmark point CRS automatically. Provide --benchmark-points-epsg, "
        f"or use lon/lat-like columns. x_col={x_name} y_col={y_name}"
    )


def _align_raster_to_template(src_path: Optional[Path], template_path: Path, *, dtype: str = "float64", nodata_value: float = np.nan, resampling: Resampling = Resampling.nearest) -> Optional[np.ndarray]:
    if src_path is None:
        return None
    with rasterio.open(template_path) as tds:
        out = np.full((tds.height, tds.width), nodata_value, dtype=dtype)
        with rasterio.open(src_path) as sds:
            src = sds.read(1)
            src_nodata = sds.nodata
            if src_nodata is not None and np.issubdtype(src.dtype, np.floating):
                src = src.astype("float64", copy=False)
                src[np.isclose(src, src_nodata)] = np.nan
            reproject(
                source=src,
                destination=out,
                src_transform=sds.transform,
                src_crs=sds.crs,
                dst_transform=tds.transform,
                dst_crs=tds.crs,
                src_nodata=src_nodata,
                dst_nodata=nodata_value,
                resampling=resampling,
            )
    return out


def _raster_diff_stats(diff: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool) & np.isfinite(diff)
    n = int(np.count_nonzero(m))
    if n == 0:
        return {
            "n": 0,
            "changed_pixels": 0,
            "changed_frac": np.nan,
            "mean_diff": np.nan,
            "mean_abs_diff": np.nan,
            "p95_abs_diff": np.nan,
            "max_abs_diff": np.nan,
        }
    vals = diff[m]
    abs_vals = np.abs(vals)
    changed = abs_vals > 1.0e-6
    return {
        "n": n,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frac": float(np.count_nonzero(changed) / float(n)),
        "mean_diff": float(np.mean(vals)),
        "mean_abs_diff": float(np.mean(abs_vals)),
        "p95_abs_diff": float(np.percentile(abs_vals, 95.0)),
        "max_abs_diff": float(np.max(abs_vals)),
    }




def _scored_metrics_from_df(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {"n": 0}
    obs = pd.to_numeric(df["obs"], errors="coerce").to_numpy(dtype="float64")
    baseline = pd.to_numeric(df["baseline"], errors="coerce").to_numpy(dtype="float64")
    final = pd.to_numeric(df["final"], errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(obs) & np.isfinite(baseline) & np.isfinite(final)
    if not np.any(finite):
        return {"n": 0}
    obs = obs[finite]
    baseline = baseline[finite]
    final = final[finite]
    baseline_metrics = _metrics(obs, baseline)
    final_metrics = _metrics(obs, final)
    return {
        "n": int(obs.size),
        "baseline": baseline_metrics,
        "final": final_metrics,
        "delta_final_minus_baseline": _delta_metrics(final_metrics, baseline_metrics),
        "improvement": _improvement_summary(obs, baseline, final),
    }


def _compute_hard_river_benchmark(scored_df: pd.DataFrame) -> dict[str, Any]:
    if scored_df is None or scored_df.empty:
        return {"available": False, "reason": "scored_df_empty"}
    if "river_mask" not in scored_df.columns:
        return {"available": False, "reason": "river_mask_missing"}

    work = scored_df.loc[scored_df["river_mask"].fillna(False).astype(bool)].copy()
    if work.empty:
        return {"available": False, "reason": "no_river_holdout_points"}

    rows: list[dict[str, Any]] = []

    def _add_group(name: str, mask: pd.Series) -> None:
        sub = work.loc[mask.fillna(False).astype(bool)].copy()
        metrics = _scored_metrics_from_df(sub)
        row = {"group": name, **metrics}
        rows.append(row)

    _add_group("river_all", pd.Series(True, index=work.index))

    if "support_class_label" in work.columns:
        for label in sorted(x for x in work["support_class_label"].dropna().astype(str).unique() if x and x != "<NA>"):
            _add_group(f"river_support::{label}", work["support_class_label"].astype(str) == label)

    if "authoritative_role_label" in work.columns:
        for label in sorted(x for x in work["authoritative_role_label"].dropna().astype(str).unique() if x and x != "<NA>"):
            _add_group(f"river_authoritative_role::{label}", work["authoritative_role_label"].astype(str) == label)

    if {"prediction_measured_anchor_fraction", "prediction_structure_only_fraction"}.issubset(work.columns):
        measured = pd.to_numeric(work["prediction_measured_anchor_fraction"], errors="coerce").fillna(0.0)
        struct_only = pd.to_numeric(work["prediction_structure_only_fraction"], errors="coerce").fillna(0.0)
        _add_group("river_regime::weak_structure_dominated", (measured <= 0.15) & (struct_only >= 0.5))
        _add_group("river_regime::measured_anchor_influenced", measured >= 0.35)

    if "prediction_low_support_caution" in work.columns:
        caution = pd.to_numeric(work["prediction_low_support_caution"], errors="coerce").fillna(0).astype(int)
        _add_group("river_regime::low_support_caution", caution > 0)

    if "prediction_support_confidence" in work.columns:
        conf = pd.to_numeric(work["prediction_support_confidence"], errors="coerce")
        _add_group("river_regime::low_support_confidence", conf < 0.35)
        _add_group("river_regime::higher_support_confidence", conf >= 0.65)

    if "support_class_label" in work.columns:
        hard_mask = work["support_class_label"].map(_is_non_authoritative_support_label)
        _add_group("river_hard_problem", hard_mask)

    available_rows = [r for r in rows if int(r.get("n", 0)) > 0]
    if not available_rows:
        return {"available": False, "reason": "no_nonempty_river_groups"}

    support_counts = {}
    if "support_class_label" in work.columns:
        support_counts = {str(k): int(v) for k, v in work["support_class_label"].astype(str).value_counts(dropna=False).to_dict().items()}

    hard_summary = next((r for r in available_rows if r.get("group") == "river_hard_problem"), None)
    # Detect the pathological case: all river holdout points are authoritative_locked,
    # so the hard-problem group is empty and this benchmark cannot evaluate unsupported
    # river improvement.  Flag this explicitly so the reader knows to consult the
    # support_distance_binned_roughness and zone_diff metrics instead.
    all_locked = bool(support_counts) and all(_support_class_code(str(k)) is not None and support_class_is_final_authoritative(_support_class_code(str(k))) for k in support_counts.keys())
    hard_problem_blind = hard_summary is None or int(hard_summary.get("n", 0)) == 0
    return {
        "available": True,
        "river_point_count": int(len(work)),
        "river_support_class_counts": support_counts,
        "hard_problem_group": hard_summary,
        "hard_problem_evaluation_blind": hard_problem_blind,
        "hard_problem_blind_reason": (
            "All river holdout points are authoritative_locked — the auto-holdout "
            "draws from authoritative data, so non-authoritative river zones "
            "(guidance_conditioned_river, scaffold_inferred) have zero holdout coverage. "
            "Use support_distance_binned_roughness and zone_diff metrics to evaluate "
            "unsupported river improvement."
        ) if hard_problem_blind else None,
        "groups": available_rows,
    }
def _compute_zone_diff_summary(*, baseline_raster: Path, final_raster: Path, support_class_raster: Optional[Path], river_mask_raster: Optional[Path], estuary_mask_raster: Optional[Path]) -> dict[str, Any]:
    final_arr = _align_raster_to_template(final_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.bilinear)
    base_arr = _align_raster_to_template(baseline_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.bilinear)
    if final_arr is None or base_arr is None:
        return {}
    diff = final_arr - base_arr
    finite = np.isfinite(final_arr) & np.isfinite(base_arr)
    zones: dict[str, dict[str, Any]] = {
        "overall": _raster_diff_stats(diff, finite),
    }
    support_arr = _align_raster_to_template(support_class_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.nearest)
    if support_arr is not None:
        valid_support = finite & np.isfinite(support_arr)
        for code in sorted(int(c) for c in np.unique(support_arr[np.isfinite(support_arr)])):
            name = SUPPORT_CLASS_CODE_TO_NAME.get(code, f"support_class_{code}")
            zones[f"support_class::{name}"] = _raster_diff_stats(diff, valid_support & np.isclose(support_arr, code))
        auth_mask = valid_support & np.isclose(support_arr, 1)
        auth_stats = _raster_diff_stats(diff, auth_mask)
        zones["authoritative_locked"] = auth_stats
        zones["authoritative_locked_invariant"] = {
            "n": auth_stats["n"],
            "changed_pixels": auth_stats["changed_pixels"],
            "passes": bool(auth_stats["changed_pixels"] == 0),
            "max_abs_diff": auth_stats["max_abs_diff"],
            "tolerance": 1.0e-6,
        }
        sdb_mask = valid_support & np.isclose(support_arr, 3)
        river_zone_mask = valid_support & np.isin(support_arr, [4, 5])
        low_conf_mask = valid_support & np.isclose(support_arr, 6)
        zones["sdb_zone"] = _raster_diff_stats(diff, sdb_mask)
        zones["river_zone"] = _raster_diff_stats(diff, river_zone_mask)
        zones["low_confidence_fill_zone"] = _raster_diff_stats(diff, low_conf_mask)
    river_mask_arr = _align_raster_to_template(river_mask_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.nearest)
    if river_mask_arr is not None:
        zones["river_mask"] = _raster_diff_stats(diff, finite & np.isfinite(river_mask_arr) & (river_mask_arr != 0))
    estuary_mask_arr = _align_raster_to_template(estuary_mask_raster, final_raster, dtype="float64", nodata_value=np.nan, resampling=Resampling.nearest)
    if estuary_mask_arr is not None:
        zones["estuary_mask"] = _raster_diff_stats(diff, finite & np.isfinite(estuary_mask_arr) & (estuary_mask_arr != 0))
    return zones


def _write_zone_diff_csv(path: Path, zone_summary: dict[str, Any]) -> None:
    rows=[]
    for name, stats in zone_summary.items():
        if not isinstance(stats, dict):
            continue
        row={"zone": name}
        row.update(stats)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)



def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "mean_abs": np.nan,
            "median_abs": np.nan,
            "p95_abs": np.nan,
            "max_abs": np.nan,
        }
    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "mean_abs": float(np.mean(abs_arr)),
        "median_abs": float(np.median(abs_arr)),
        "p95_abs": float(np.percentile(abs_arr, 95.0)),
        "max_abs": float(np.max(abs_arr)),
    }

def _weighted_mean(items: list[dict[str, Any]], value_key: str, weight_key: str = "n") -> float:
    vals: list[float] = []
    weights: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get(value_key)
        weight = item.get(weight_key, 0)
        try:
            value_f = float(value)
            weight_f = float(weight)
        except Exception:
            continue
        if np.isfinite(value_f) and np.isfinite(weight_f) and weight_f > 0:
            vals.append(value_f)
            weights.append(weight_f)
    if not vals:
        return np.nan
    return float(np.average(np.asarray(vals, dtype="float64"), weights=np.asarray(weights, dtype="float64")))


def _compute_support_aware_science_summary(
    *,
    river_scientific_summary: dict[str, Any],
    science_evaluation_summary: dict[str, Any],
    zone_diff_summary: dict[str, Any],
    hard_river_summary: dict[str, Any],
) -> dict[str, Any]:
    bins = river_scientific_summary.get("support_distance_binned_roughness") if isinstance(river_scientific_summary, dict) else None
    class_stats = science_evaluation_summary.get("by_support_class") if isinstance(science_evaluation_summary, dict) else None
    auth_inv = zone_diff_summary.get("authoritative_locked_invariant", {}) if isinstance(zone_diff_summary, dict) else {}
    if not isinstance(bins, list) and not isinstance(class_stats, dict):
        return {"available": False}

    transition_labels = {"0_100m_near_anchor", "100_300m_transitional"}
    unsupported_labels = {"300_500m_weak_support", "500_1000m_unsupported", "1000m_plus_far_unsupported"}
    transition_rows = [b for b in (bins or []) if isinstance(b, dict) and b.get("available") and b.get("bin") in transition_labels]
    unsupported_rows = [b for b in (bins or []) if isinstance(b, dict) and b.get("available") and b.get("bin") in unsupported_labels]

    unsupported_class_labels = [
        "guidance_conditioned_river",
        "scaffold_inferred",
        "low_confidence_continuous_fill",
    ]
    unsupported_class_rows = [class_stats.get(label) for label in unsupported_class_labels if isinstance(class_stats, dict) and isinstance(class_stats.get(label), dict)]
    unsupported_class_rows = [r for r in unsupported_class_rows if int(r.get("n", 0)) > 0]
    component_stats = river_scientific_summary.get("unsupported_by_component_class", {}) if isinstance(river_scientific_summary, dict) else {}
    component_roughness_stats = river_scientific_summary.get("unsupported_roughness_by_component_class", {}) if isinstance(river_scientific_summary, dict) else {}
    component_curvature_stats = river_scientific_summary.get("unsupported_curvature_by_component_class", {}) if isinstance(river_scientific_summary, dict) else {}
    component_tendency_stats = river_scientific_summary.get("unsupported_section_tendency_by_component_class", {}) if isinstance(river_scientific_summary, dict) else {}
    component_relief_stats = river_scientific_summary.get("unsupported_inner_relief_by_component_class", {}) if isinstance(river_scientific_summary, dict) else {}
    component_reconciliation_stats = river_scientific_summary.get("unsupported_reconciliation_by_component_class", {}) if isinstance(river_scientific_summary, dict) else {}
    component_labels = ["unsupported_mainstem", "unsupported_side_component", "tiny_detached_component"]
    component_rows = [component_stats.get(label) for label in component_labels if isinstance(component_stats, dict) and isinstance(component_stats.get(label), dict)]
    component_rows = [r for r in component_rows if int(r.get("n", 0)) > 0]
    tendency_rows = [component_tendency_stats.get(label) for label in component_labels if isinstance(component_tendency_stats, dict) and isinstance(component_tendency_stats.get(label), dict)]
    tendency_rows = [r for r in tendency_rows if int(r.get("n", 0)) > 0]
    relief_rows = [component_relief_stats.get(label) for label in component_labels if isinstance(component_relief_stats, dict) and isinstance(component_relief_stats.get(label), dict)]
    relief_rows = [r for r in relief_rows if int(r.get("n", 0)) > 0]
    reconciliation_rows = [component_reconciliation_stats.get(label) for label in component_labels if isinstance(component_reconciliation_stats, dict) and isinstance(component_reconciliation_stats.get(label), dict)]
    reconciliation_rows = [r for r in reconciliation_rows if int(r.get("n", 0)) > 0]
    admiss_stats = science_evaluation_summary.get("by_prediction_admissibility", {}) if isinstance(science_evaluation_summary, dict) else {}

    transition_quality = {
        "available": bool(transition_rows),
        "station_count": int(sum(int(r.get("n", 0)) for r in transition_rows)),
        "baseline_mean_abs_step": _weighted_mean(transition_rows, "baseline_mean_abs_step"),
        "final_mean_abs_step": _weighted_mean(transition_rows, "final_mean_abs_step"),
        "delta_mean_abs_step": _weighted_mean(transition_rows, "delta_mean_abs_step"),
        "baseline_p95_abs_step": _weighted_mean(transition_rows, "baseline_p95_abs_step"),
        "final_p95_abs_step": _weighted_mean(transition_rows, "final_p95_abs_step"),
        "delta_p95_abs_step": _weighted_mean(transition_rows, "delta_p95_abs_step"),
        "improved_fraction": _weighted_mean([{"n": int(r.get("n", 0)), "flag": 1.0 if bool(r.get("roughness_improved")) else 0.0} for r in transition_rows], "flag"),
    }
    unsupported_roughness = {
        "available": bool(unsupported_rows),
        "station_count": int(sum(int(r.get("n", 0)) for r in unsupported_rows)),
        "baseline_mean_abs_step": _weighted_mean(unsupported_rows, "baseline_mean_abs_step"),
        "final_mean_abs_step": _weighted_mean(unsupported_rows, "final_mean_abs_step"),
        "delta_mean_abs_step": _weighted_mean(unsupported_rows, "delta_mean_abs_step"),
        "baseline_p95_abs_step": _weighted_mean(unsupported_rows, "baseline_p95_abs_step"),
        "final_p95_abs_step": _weighted_mean(unsupported_rows, "final_p95_abs_step"),
        "delta_p95_abs_step": _weighted_mean(unsupported_rows, "delta_p95_abs_step"),
        "improved_fraction": _weighted_mean([{"n": int(r.get("n", 0)), "flag": 1.0 if bool(r.get("roughness_improved")) else 0.0} for r in unsupported_rows], "flag"),
    }
    unsupported_science_grid = {
        "available": bool(unsupported_class_rows),
        "cell_count": int(sum(int(r.get("n", 0)) for r in unsupported_class_rows)),
        "mean_abs_delta_m": _weighted_mean(unsupported_class_rows, "mean_abs_delta_m"),
        "median_abs_delta_m": _weighted_mean(unsupported_class_rows, "median_abs_delta_m"),
        "p95_abs_delta_m": _weighted_mean(unsupported_class_rows, "p95_abs_delta_m"),
        "active_fraction": _weighted_mean(unsupported_class_rows, "active_fraction"),
        "by_support_class": {label: class_stats.get(label) for label in unsupported_class_labels if isinstance(class_stats, dict) and label in class_stats},
    }
    unsupported_by_component_class = {
        "available": bool(component_rows),
        "point_count": int(sum(int(r.get("n", 0)) for r in component_rows)),
        "mean_abs_delta_m": _weighted_mean(component_rows, "mean_abs_delta_m"),
        "median_abs_delta_m": _weighted_mean(component_rows, "median_abs_delta_m"),
        "p95_abs_delta_m": _weighted_mean(component_rows, "p95_abs_delta_m"),
        "active_fraction": _weighted_mean(component_rows, "active_fraction"),
        "by_component_class": {label: component_stats.get(label) for label in component_labels if isinstance(component_stats, dict) and label in component_stats},
    }
    unsupported_by_prediction_admissibility = {
        "available": bool(admiss_stats),
        "groups": {str(k): v for k, v in admiss_stats.items() if isinstance(v, dict)} if isinstance(admiss_stats, dict) else {},
    }
    unsupported_section_tendency = {
        "available": bool(tendency_rows),
        "station_count": int(sum(int(r.get("n", 0)) for r in tendency_rows)),
        "mean_tendency_depth_fraction": _weighted_mean(tendency_rows, "mean_tendency_depth_fraction"),
        "median_tendency_depth_fraction": _weighted_mean(tendency_rows, "median_tendency_depth_fraction"),
        "mean_tendency_confidence": _weighted_mean(tendency_rows, "mean_tendency_confidence"),
        "active_fraction": _weighted_mean(tendency_rows, "active_fraction"),
        "simplified_fraction": _weighted_mean(tendency_rows, "simplified_fraction"),
        "by_component_class": {label: component_tendency_stats.get(label) for label in component_labels if isinstance(component_tendency_stats, dict) and label in component_tendency_stats},
    }
    unsupported_inner_relief = {
        "available": bool(relief_rows),
        "station_count": int(sum(int(r.get("n", 0)) for r in relief_rows)),
        "mean_inner_relief_m": _weighted_mean(relief_rows, "mean_inner_relief_m"),
        "median_inner_relief_m": _weighted_mean(relief_rows, "median_inner_relief_m"),
        "p95_inner_relief_m": _weighted_mean(relief_rows, "p95_inner_relief_m"),
        "max_inner_relief_m": _weighted_mean(relief_rows, "max_inner_relief_m"),
        "by_component_class": {label: component_relief_stats.get(label) for label in component_labels if isinstance(component_relief_stats, dict) and label in component_relief_stats},
    }
    unsupported_reconciliation = {
        "available": bool(reconciliation_rows),
        "point_count": int(sum(int(r.get("n", 0)) for r in reconciliation_rows)),
        "mean_reconciliation_weight": _weighted_mean(reconciliation_rows, "mean_reconciliation_weight"),
        "p95_reconciliation_weight": _weighted_mean(reconciliation_rows, "p95_reconciliation_weight"),
        "mean_abs_reconciliation_delta_m": _weighted_mean(reconciliation_rows, "mean_abs_reconciliation_delta_m"),
        "p95_abs_reconciliation_delta_m": _weighted_mean(reconciliation_rows, "p95_abs_reconciliation_delta_m"),
        "active_fraction": _weighted_mean(reconciliation_rows, "active_fraction"),
        "by_component_class": {label: component_reconciliation_stats.get(label) for label in component_labels if isinstance(component_reconciliation_stats, dict) and label in component_reconciliation_stats},
    }
    mainstem_stats = component_stats.get("unsupported_mainstem") if isinstance(component_stats, dict) else None
    side_stats = component_stats.get("unsupported_side_component") if isinstance(component_stats, dict) else None
    mainstem_rough = component_roughness_stats.get("unsupported_mainstem") if isinstance(component_roughness_stats, dict) else None
    side_rough = component_roughness_stats.get("unsupported_side_component") if isinstance(component_roughness_stats, dict) else None
    mainstem_curv = component_curvature_stats.get("unsupported_mainstem") if isinstance(component_curvature_stats, dict) else None
    side_curv = component_curvature_stats.get("unsupported_side_component") if isinstance(component_curvature_stats, dict) else None
    mainstem_tend = component_tendency_stats.get("unsupported_mainstem") if isinstance(component_tendency_stats, dict) else None
    side_tend = component_tendency_stats.get("unsupported_side_component") if isinstance(component_tendency_stats, dict) else None
    mainstem_relief = component_relief_stats.get("unsupported_mainstem") if isinstance(component_relief_stats, dict) else None
    side_relief = component_relief_stats.get("unsupported_side_component") if isinstance(component_relief_stats, dict) else None
    unsupported_mainstem_vs_side_component = {
        "available": all(isinstance(x, dict) and int(x.get("n", 0)) > 0 for x in (mainstem_stats, side_stats, mainstem_rough, side_rough, mainstem_curv, side_curv, mainstem_tend, side_tend, mainstem_relief, side_relief)),
        "mainstem_point_count": int(mainstem_stats.get("n", 0)) if isinstance(mainstem_stats, dict) else 0,
        "side_point_count": int(side_stats.get("n", 0)) if isinstance(side_stats, dict) else 0,
        "mainstem_mean_abs_delta_m": float(mainstem_stats.get("mean_abs_delta_m", np.nan)) if isinstance(mainstem_stats, dict) else np.nan,
        "side_mean_abs_delta_m": float(side_stats.get("mean_abs_delta_m", np.nan)) if isinstance(side_stats, dict) else np.nan,
        "mainstem_final_mean_abs_step": float(mainstem_rough.get("final_mean_abs_step", np.nan)) if isinstance(mainstem_rough, dict) else np.nan,
        "side_final_mean_abs_step": float(side_rough.get("final_mean_abs_step", np.nan)) if isinstance(side_rough, dict) else np.nan,
        "mainstem_final_p95_abs_second_diff": float(mainstem_curv.get("final_p95_abs_second_diff", np.nan)) if isinstance(mainstem_curv, dict) else np.nan,
        "side_final_p95_abs_second_diff": float(side_curv.get("final_p95_abs_second_diff", np.nan)) if isinstance(side_curv, dict) else np.nan,
        "mainstem_mean_tendency_depth_fraction": float(mainstem_tend.get("mean_tendency_depth_fraction", np.nan)) if isinstance(mainstem_tend, dict) else np.nan,
        "side_mean_tendency_depth_fraction": float(side_tend.get("mean_tendency_depth_fraction", np.nan)) if isinstance(side_tend, dict) else np.nan,
        "mainstem_mean_inner_relief_m": float(mainstem_relief.get("mean_inner_relief_m", np.nan)) if isinstance(mainstem_relief, dict) else np.nan,
        "side_mean_inner_relief_m": float(side_relief.get("mean_inner_relief_m", np.nan)) if isinstance(side_relief, dict) else np.nan,
        "mainstem_is_more_expressive": bool(float(mainstem_tend.get("mean_tendency_depth_fraction", np.nan)) > float(side_tend.get("mean_tendency_depth_fraction", np.nan))) if isinstance(mainstem_tend, dict) and isinstance(side_tend, dict) and np.isfinite(float(mainstem_tend.get("mean_tendency_depth_fraction", np.nan))) and np.isfinite(float(side_tend.get("mean_tendency_depth_fraction", np.nan))) else False,
        "mainstem_has_more_inner_relief": bool(float(mainstem_relief.get("mean_inner_relief_m", np.nan)) > float(side_relief.get("mean_inner_relief_m", np.nan))) if isinstance(mainstem_relief, dict) and isinstance(side_relief, dict) and np.isfinite(float(mainstem_relief.get("mean_inner_relief_m", np.nan))) and np.isfinite(float(side_relief.get("mean_inner_relief_m", np.nan))) else False,
        "side_is_rougher": bool(float(side_rough.get("final_mean_abs_step", np.nan)) > float(mainstem_rough.get("final_mean_abs_step", np.nan))) if isinstance(mainstem_rough, dict) and isinstance(side_rough, dict) and np.isfinite(float(mainstem_rough.get("final_mean_abs_step", np.nan))) and np.isfinite(float(side_rough.get("final_mean_abs_step", np.nan))) else False,
        "side_is_more_curved": bool(float(side_curv.get("final_p95_abs_second_diff", np.nan)) > float(mainstem_curv.get("final_p95_abs_second_diff", np.nan))) if isinstance(mainstem_curv, dict) and isinstance(side_curv, dict) and np.isfinite(float(mainstem_curv.get("final_p95_abs_second_diff", np.nan))) and np.isfinite(float(side_curv.get("final_p95_abs_second_diff", np.nan))) else False,
    }
    mainstem_recon = component_reconciliation_stats.get("unsupported_mainstem") if isinstance(component_reconciliation_stats, dict) else None
    side_recon = component_reconciliation_stats.get("unsupported_side_component") if isinstance(component_reconciliation_stats, dict) else None
    unsupported_mainstem_vs_side_reconciliation = {
        "available": isinstance(mainstem_recon, dict) and isinstance(side_recon, dict) and int(mainstem_recon.get("n", 0)) > 0 and int(side_recon.get("n", 0)) > 0,
        "mainstem_point_count": int(mainstem_recon.get("n", 0)) if isinstance(mainstem_recon, dict) else 0,
        "side_point_count": int(side_recon.get("n", 0)) if isinstance(side_recon, dict) else 0,
        "mainstem_mean_reconciliation_weight": float(mainstem_recon.get("mean_reconciliation_weight", np.nan)) if isinstance(mainstem_recon, dict) else np.nan,
        "side_mean_reconciliation_weight": float(side_recon.get("mean_reconciliation_weight", np.nan)) if isinstance(side_recon, dict) else np.nan,
        "mainstem_p95_reconciliation_weight": float(mainstem_recon.get("p95_reconciliation_weight", np.nan)) if isinstance(mainstem_recon, dict) else np.nan,
        "side_p95_reconciliation_weight": float(side_recon.get("p95_reconciliation_weight", np.nan)) if isinstance(side_recon, dict) else np.nan,
        "mainstem_mean_abs_reconciliation_delta_m": float(mainstem_recon.get("mean_abs_reconciliation_delta_m", np.nan)) if isinstance(mainstem_recon, dict) else np.nan,
        "side_mean_abs_reconciliation_delta_m": float(side_recon.get("mean_abs_reconciliation_delta_m", np.nan)) if isinstance(side_recon, dict) else np.nan,
        "mainstem_active_fraction": float(mainstem_recon.get("active_fraction", np.nan)) if isinstance(mainstem_recon, dict) else np.nan,
        "side_active_fraction": float(side_recon.get("active_fraction", np.nan)) if isinstance(side_recon, dict) else np.nan,
        "mainstem_carries_more_reconciliation": bool(float(mainstem_recon.get("mean_reconciliation_weight", np.nan)) > float(side_recon.get("mean_reconciliation_weight", np.nan))) if isinstance(mainstem_recon, dict) and isinstance(side_recon, dict) and np.isfinite(float(mainstem_recon.get("mean_reconciliation_weight", np.nan))) and np.isfinite(float(side_recon.get("mean_reconciliation_weight", np.nan))) else False,
        "mainstem_has_stronger_reconciliation_delta": bool(float(mainstem_recon.get("mean_abs_reconciliation_delta_m", np.nan)) >= float(side_recon.get("mean_abs_reconciliation_delta_m", np.nan))) if isinstance(mainstem_recon, dict) and isinstance(side_recon, dict) and np.isfinite(float(mainstem_recon.get("mean_abs_reconciliation_delta_m", np.nan))) and np.isfinite(float(side_recon.get("mean_abs_reconciliation_delta_m", np.nan))) else False,
    }

    hard_blind = bool(hard_river_summary.get("hard_problem_evaluation_blind")) if isinstance(hard_river_summary, dict) else False
    return {
        "available": True,
        "evaluation_mode": "proxy_grid_and_profile" if hard_blind else "holdout_and_proxy_grid",
        "hard_problem_holdout_blind": hard_blind,
        "authoritative_preservation": {
            "available": isinstance(auth_inv, dict) and auth_inv.get("n", 0) > 0,
            "passes": bool(auth_inv.get("passes", False)),
            "changed_pixels": int(auth_inv.get("changed_pixels", 0)) if isinstance(auth_inv, dict) else 0,
            "max_abs_diff": float(auth_inv.get("max_abs_diff", np.nan)) if isinstance(auth_inv, dict) else np.nan,
        },
        "transition_quality": transition_quality,
        "unsupported_roughness": unsupported_roughness,
        "unsupported_science_grid": unsupported_science_grid,
        "unsupported_by_component_class": unsupported_by_component_class,
        "unsupported_by_prediction_admissibility": unsupported_by_prediction_admissibility,
        "unsupported_section_tendency": unsupported_section_tendency,
        "unsupported_inner_relief": unsupported_inner_relief,
        "unsupported_reconciliation": unsupported_reconciliation,
        "unsupported_mainstem_vs_side_component": unsupported_mainstem_vs_side_component,
        "unsupported_mainstem_vs_side_reconciliation": unsupported_mainstem_vs_side_reconciliation,
    }


def _sample_raster_from_xy_crs(raster_path: Path, x: np.ndarray, y: np.ndarray, src_crs: Any) -> np.ndarray:
    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        if src_crs is None:
            raise ValueError(f"Source CRS missing while sampling raster: {raster_path}")
        if str(src_crs) != str(ds.crs):
            tx = Transformer.from_crs(src_crs, ds.crs, always_xy=True)
            xs, ys = tx.transform(x, y)
        else:
            xs, ys = x, y
        vals = np.array([v[0] for v in ds.sample(list(zip(xs, ys)))], dtype="float64")
        nodata = ds.nodata
        if nodata is not None:
            vals[np.isclose(vals, nodata)] = np.nan
        return vals


def _compute_longitudinal_profile_coverage_summary(coverage_csv_path: Optional[Path], coverage_summary_path: Optional[Path]) -> dict[str, Any]:
    if coverage_summary_path is not None and Path(coverage_summary_path).exists():
        try:
            payload = json.loads(Path(coverage_summary_path).read_text(encoding="utf-8"))
            coverage = payload.get("coverage_summary")
            if isinstance(coverage, dict):
                return {"available": True, "source": str(coverage_summary_path), "coverage_summary": coverage}
        except Exception:
            log.debug("Failed to read longitudinal profile summary %s", coverage_summary_path, exc_info=True)
    if coverage_csv_path is None or not Path(coverage_csv_path).exists():
        return {"available": False, "reason": "coverage_artifacts_missing"}
    try:
        df = pd.read_csv(coverage_csv_path)
    except Exception:
        log.debug("Failed to read longitudinal profile coverage CSV %s", coverage_csv_path, exc_info=True)
        return {"available": False, "reason": "coverage_csv_read_failed"}
    if df.empty:
        return {"available": False, "reason": "coverage_csv_empty"}
    classes = df.get("profile_support_class")
    if classes is None:
        return {"available": False, "reason": "coverage_csv_missing_profile_support_class"}
    coverage_by_class = {str(k): int(v) for k, v in classes.astype(str).value_counts(dropna=False).to_dict().items()}
    support_present = pd.to_numeric(df.get("profile_support_present"), errors="coerce").fillna(0).astype(bool)
    unsupported = ~support_present
    out = {
        "coverage_by_class": coverage_by_class,
        "unsupported_station_count": int(unsupported.sum()),
        "supported_station_count": int(support_present.sum()),
        "unsupported_fraction": float(unsupported.mean()) if len(df) else np.nan,
        "max_support_source_count": int(pd.to_numeric(df.get("profile_support_source_count"), errors="coerce").max()) if "profile_support_source_count" in df.columns else 0,
        "min_support_source_count": int(pd.to_numeric(df.get("profile_support_source_count"), errors="coerce").min()) if "profile_support_source_count" in df.columns else 0,
    }
    # Include measured-support-distance summary when available
    if "profile_measured_support_distance_m" in df.columns:
        dist = pd.to_numeric(df["profile_measured_support_distance_m"], errors="coerce").to_numpy(dtype=float)
        finite_dist = dist[np.isfinite(dist)]
        if finite_dist.size > 0:
            out["measured_support_distance_summary"] = {
                "median_m": float(np.median(finite_dist)),
                "p75_m": float(np.percentile(finite_dist, 75)),
                "p95_m": float(np.percentile(finite_dist, 95)),
                "max_m": float(np.max(finite_dist)),
                "fraction_gt_500m": float(np.mean(finite_dist > 500.0)),
                "fraction_gt_1000m": float(np.mean(finite_dist > 1000.0)),
            }
    if "profile_authoritative_role" in df.columns:
        out["authoritative_role_counts"] = {str(k): int(v) for k, v in df["profile_authoritative_role"].astype(str).value_counts(dropna=False).to_dict().items()}
    if "profile_authoritative_bed_support_present" in df.columns:
        bed = pd.to_numeric(df["profile_authoritative_bed_support_present"], errors="coerce").fillna(0).astype(bool)
        out["authoritative_bed_supported_station_count"] = int(bed.sum())
        out["authoritative_bed_supported_fraction"] = float(bed.mean()) if len(df) else np.nan
    if "profile_authoritative_bank_margin_present" in df.columns:
        bank = pd.to_numeric(df["profile_authoritative_bank_margin_present"], errors="coerce").fillna(0).astype(bool)
        out["authoritative_bank_margin_station_count"] = int(bank.sum())
        out["authoritative_bank_margin_fraction"] = float(bank.mean()) if len(df) else np.nan
    if "profile_authoritative_bed_support_distance_m" in df.columns:
        dist = pd.to_numeric(df["profile_authoritative_bed_support_distance_m"], errors="coerce").to_numpy(dtype=float)
        finite_dist = dist[np.isfinite(dist)]
        if finite_dist.size > 0:
            out["authoritative_bed_support_distance_summary"] = {
                "median_m": float(np.median(finite_dist)),
                "p75_m": float(np.percentile(finite_dist, 75)),
                "p95_m": float(np.percentile(finite_dist, 95)),
                "max_m": float(np.max(finite_dist)),
                "fraction_gt_500m": float(np.mean(finite_dist > 500.0)),
                "fraction_gt_1000m": float(np.mean(finite_dist > 1000.0)),
            }
    if "profile_far_from_authoritative_bed_support" in df.columns:
        far = pd.to_numeric(df["profile_far_from_authoritative_bed_support"], errors="coerce").fillna(0).astype(bool)
        out["far_from_authoritative_bed_station_count"] = int(far.sum())
        out["far_from_authoritative_bed_fraction"] = float(far.mean()) if len(df) else np.nan
    return {"available": True, "source": str(coverage_csv_path), "coverage_summary": out}


def _compute_river_scientific_summary(*, baseline_raster: Path, final_raster: Path, profile_points_path: Optional[Path], coverage_csv_path: Optional[Path], coverage_summary_path: Optional[Path], station_targets_path: Optional[Path], bench_dir: Path) -> tuple[dict[str, Any], Optional[Path]]:
    coverage_info = _compute_longitudinal_profile_coverage_summary(coverage_csv_path, coverage_summary_path)
    if profile_points_path is None or not Path(profile_points_path).exists():
        return ({
            "available": False,
            "reason": "longitudinal_profile_points_missing",
            "coverage": coverage_info,
        }, None)
    if gpd is None:
        return ({
            "available": False,
            "reason": "geopandas_unavailable",
            "coverage": coverage_info,
        }, None)

    try:
        pts = gpd.read_file(profile_points_path)
    except Exception:
        log.debug("Failed to read longitudinal profile points %s", profile_points_path, exc_info=True)
        return ({
            "available": False,
            "reason": "longitudinal_profile_points_read_failed",
            "coverage": coverage_info,
        }, None)
    if pts.empty or pts.geometry is None:
        return ({
            "available": False,
            "reason": "longitudinal_profile_points_empty",
            "coverage": coverage_info,
        }, None)
    if pts.crs is None:
        return ({
            "available": False,
            "reason": "longitudinal_profile_points_missing_crs",
            "coverage": coverage_info,
        }, None)

    pts = pts.copy()
    profile_col = _infer_col(pd.DataFrame(pts.drop(columns=[c for c in ["geometry"] if c in pts.columns])), ["profile_id", "component_id", "levelpathi"], "profile/component id")
    station_col = _infer_col(pd.DataFrame(pts.drop(columns=[c for c in ["geometry"] if c in pts.columns])), ["station_m", "station"], "station")
    pts[profile_col] = pts[profile_col].astype(str)
    pts[station_col] = pd.to_numeric(pts[station_col], errors="coerce")
    pts = pts.loc[pts.geometry.notna() & (~pts.geometry.is_empty) & np.isfinite(pts[station_col].to_numpy(dtype=float))].copy()
    if pts.empty:
        return ({
            "available": False,
            "reason": "longitudinal_profile_points_no_finite_station_geometry",
            "coverage": coverage_info,
        }, None)

    x = pts.geometry.x.to_numpy(dtype="float64")
    y = pts.geometry.y.to_numpy(dtype="float64")
    baseline = _sample_raster_from_xy_crs(baseline_raster, x, y, pts.crs)
    final = _sample_raster_from_xy_crs(final_raster, x, y, pts.crs)
    finite = np.isfinite(baseline) & np.isfinite(final)
    pts = pts.loc[finite].copy()
    baseline = baseline[finite]
    final = final[finite]
    if pts.empty:
        return ({
            "available": False,
            "reason": "longitudinal_profile_points_no_finite_samples",
            "coverage": coverage_info,
        }, None)

    pts["baseline_sample_m"] = baseline
    pts["final_sample_m"] = final
    pts["final_minus_baseline_m"] = final - baseline

    if coverage_csv_path is not None and Path(coverage_csv_path).exists():
        try:
            cov = pd.read_csv(coverage_csv_path)
            if not cov.empty and "profile_support_class" in cov.columns and "station_m" in cov.columns:
                cov = cov.copy()
                cov[profile_col] = cov[_infer_col(cov, [profile_col, "profile_id", "component_id", "levelpathi"], "profile/component id")].astype(str)
                cov["_station_key"] = pd.to_numeric(cov["station_m"], errors="coerce").round(6)
                pts["_station_key"] = pd.to_numeric(pts[station_col], errors="coerce").round(6)
                merge_cols = [profile_col, "_station_key"]
                extra_cols = [c for c in ["profile_support_class", "profile_support_source_count", "profile_support_present", "profile_authoritative_anchor_present", "profile_xs_support_present", "profile_centerline_support_present", "profile_bank_support_present", "profile_wse_support_present", "profile_measured_support_distance_m", "profile_far_from_measured_support", "profile_authoritative_role", "profile_authoritative_bed_support_present", "profile_authoritative_bank_margin_present", "profile_authoritative_bed_support_distance_m", "profile_far_from_authoritative_bed_support"] if c in cov.columns]
                pts = pts.merge(cov[merge_cols + extra_cols].drop_duplicates(merge_cols), on=merge_cols, how="left")
        except Exception:
            log.debug("Failed to merge longitudinal profile coverage CSV %s", coverage_csv_path, exc_info=True)

    station_targets = None
    if station_targets_path is not None and Path(station_targets_path).exists():
        try:
            station_targets = pd.read_csv(station_targets_path)
        except Exception:
            log.debug("Failed to read station targets %s", station_targets_path, exc_info=True)
            station_targets = None

    per_profile_rows: list[dict[str, Any]] = []
    baseline_steps_all = []
    final_steps_all = []
    change_steps_all = []
    baseline_second_all = []
    final_second_all = []
    change_second_all = []
    for pid, sub in pts.groupby(profile_col):
        sub = sub.sort_values(station_col)
        stations = pd.to_numeric(sub[station_col], errors="coerce").to_numpy(dtype="float64")
        base_vals = pd.to_numeric(sub["baseline_sample_m"], errors="coerce").to_numpy(dtype="float64")
        final_vals = pd.to_numeric(sub["final_sample_m"], errors="coerce").to_numpy(dtype="float64")
        valid = np.isfinite(stations) & np.isfinite(base_vals) & np.isfinite(final_vals)
        stations = stations[valid]
        base_vals = base_vals[valid]
        final_vals = final_vals[valid]
        row: dict[str, Any] = {
            "profile_id": str(pid),
            "point_count": int(valid.sum()),
            "station_min_m": float(np.nanmin(stations)) if stations.size else np.nan,
            "station_max_m": float(np.nanmax(stations)) if stations.size else np.nan,
            "station_step_median": float(np.nanmedian(np.diff(stations))) if stations.size >= 2 else np.nan,
        }
        if stations.size >= 2:
            base_step = np.diff(base_vals)
            final_step = np.diff(final_vals)
            change_step = np.diff(final_vals - base_vals)
            baseline_steps_all.append(base_step)
            final_steps_all.append(final_step)
            change_steps_all.append(change_step)
            base_step_summary = _distribution_summary(base_step)
            final_step_summary = _distribution_summary(final_step)
            change_step_summary = _distribution_summary(change_step)
            row.update({
                "step_count": int(base_step.size),
                "baseline_mean_abs_step_m": base_step_summary["mean_abs"],
                "final_mean_abs_step_m": final_step_summary["mean_abs"],
                "change_step_p95_abs_m": change_step_summary["p95_abs"],
                "baseline_p95_abs_step_m": base_step_summary["p95_abs"],
                "final_p95_abs_step_m": final_step_summary["p95_abs"],
            })
            if stations.size >= 3:
                base_second = np.diff(base_step)
                final_second = np.diff(final_step)
                change_second = np.diff(final_vals - base_vals, n=2)
                baseline_second_all.append(base_second)
                final_second_all.append(final_second)
                change_second_all.append(change_second)
                base_second_summary = _distribution_summary(base_second)
                final_second_summary = _distribution_summary(final_second)
                change_second_summary = _distribution_summary(change_second)
                row.update({
                    "curvature_count": int(base_second.size),
                    "baseline_p95_abs_second_diff_m": base_second_summary["p95_abs"],
                    "final_p95_abs_second_diff_m": final_second_summary["p95_abs"],
                    "change_second_diff_p95_abs_m": change_second_summary["p95_abs"],
                })
        if "profile_support_class" in sub.columns:
            profile_classes = sub["profile_support_class"].copy()
            profile_classes = profile_classes[profile_classes.notna()]
            if not profile_classes.empty:
                vc = profile_classes.astype(str).value_counts(dropna=False).to_dict()
                row["profile_support_class_counts"] = json.dumps({str(k): int(v) for k, v in vc.items()}, sort_keys=True)
        per_profile_rows.append(row)

    def _concat(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.asarray([], dtype="float64")
        return np.concatenate([np.asarray(p, dtype="float64") for p in parts if np.asarray(p).size > 0])

    baseline_steps = _concat(baseline_steps_all)
    final_steps = _concat(final_steps_all)
    change_steps = _concat(change_steps_all)
    baseline_second = _concat(baseline_second_all)
    final_second = _concat(final_second_all)
    change_second = _concat(change_second_all)

    source_counts = None
    for source_col in ("generalized_longitudinal_bed_source", "active_core_support_source", "longitudinal_profile_source", "network_backbone_source"):
        if source_col in pts.columns:
            source_series = pts[source_col].copy()
            source_series = source_series[source_series.notna()]
            source_series = source_series[source_series.astype(str).str.len() > 0]
            if not source_series.empty:
                source_counts = {str(k): int(v) for k, v in source_series.astype(str).value_counts(dropna=False).to_dict().items()}
                break

    unsupported_component_class_summary = {}
    unsupported_component_class_roughness = {}
    unsupported_component_class_curvature = {}
    unsupported_section_tendency_by_component_class = {}
    unsupported_inner_relief_by_component_class = {}
    unsupported_reconciliation_by_component_class = {}
    if "component_support_class" in pts.columns:
        unsupported_labels = {"unsupported_mainstem", "unsupported_side_component", "tiny_detached_component"}
        pts_comp = pts.copy()
        pts_comp["component_support_class"] = pts_comp["component_support_class"].astype(object)
        pts_comp = pts_comp.loc[pts_comp["component_support_class"].notna() & pts_comp["component_support_class"].astype(str).isin(unsupported_labels)].copy()
        if not pts_comp.empty and ("authoritative_reconciliation_weight" in pts_comp.columns or "authoritative_reconciliation_delta_m" in pts_comp.columns):
            recon_weight = pd.to_numeric(pts_comp.get("authoritative_reconciliation_weight"), errors="coerce") if "authoritative_reconciliation_weight" in pts_comp.columns else pd.Series(np.nan, index=pts_comp.index)
            recon_delta = pd.to_numeric(pts_comp.get("authoritative_reconciliation_delta_m"), errors="coerce") if "authoritative_reconciliation_delta_m" in pts_comp.columns else pd.Series(np.nan, index=pts_comp.index)
            for comp_class, sub in pts_comp.groupby("component_support_class"):
                idx = sub.index
                sub_weight = pd.to_numeric(recon_weight.loc[idx], errors="coerce").to_numpy(dtype=float)
                sub_delta = pd.to_numeric(recon_delta.loc[idx], errors="coerce").to_numpy(dtype=float)
                valid_weight = np.isfinite(sub_weight)
                valid_delta = np.isfinite(sub_delta)
                unsupported_reconciliation_by_component_class[str(comp_class)] = {
                    "n": int(len(sub)),
                    "available": bool(valid_weight.any() or valid_delta.any()),
                    "mean_reconciliation_weight": float(np.nanmean(sub_weight)) if valid_weight.any() else np.nan,
                    "p95_reconciliation_weight": float(np.nanpercentile(sub_weight[valid_weight], 95)) if valid_weight.any() else np.nan,
                    "mean_abs_reconciliation_delta_m": float(np.nanmean(np.abs(sub_delta))) if valid_delta.any() else np.nan,
                    "p95_abs_reconciliation_delta_m": float(np.nanpercentile(np.abs(sub_delta[valid_delta]), 95)) if valid_delta.any() else np.nan,
                    "active_fraction": float(np.count_nonzero(np.nan_to_num(sub_weight, nan=0.0) > 0.05) / max(int(len(sub)), 1)),
                }
    if station_targets is not None and not station_targets.empty and "component_support_class" in station_targets.columns:
        st = station_targets.copy()
        st["component_support_class"] = st["component_support_class"].astype(object)
        st = st.loc[st["component_support_class"].notna() & (st["component_support_class"].astype(str).str.len() > 0)].copy()
        if not st.empty:
            unsupported_labels = {"unsupported_mainstem", "unsupported_side_component", "tiny_detached_component"}
            st = st.loc[st["component_support_class"].astype(str).isin(unsupported_labels)].copy()
        if not st.empty:
            relief = pd.to_numeric(st.get("section_tendency_inner_relief_m"), errors="coerce") if "section_tendency_inner_relief_m" in st.columns else pd.Series(np.nan, index=st.index)
            depth_frac = pd.to_numeric(st.get("section_tendency_depth_m"), errors="coerce") if "section_tendency_depth_m" in st.columns else pd.Series(np.nan, index=st.index)
            conf = pd.to_numeric(st.get("section_tendency_confidence"), errors="coerce") if "section_tendency_confidence" in st.columns else pd.Series(np.nan, index=st.index)
            source = st.get("section_tendency_source", pd.Series("", index=st.index)).astype(str)
            for comp_class, sub in st.groupby("component_support_class"):
                idx = sub.index
                sub_relief = pd.to_numeric(relief.loc[idx], errors="coerce").to_numpy(dtype=float)
                sub_depth = pd.to_numeric(depth_frac.loc[idx], errors="coerce").to_numpy(dtype=float)
                sub_conf = pd.to_numeric(conf.loc[idx], errors="coerce").to_numpy(dtype=float)
                sub_source = source.loc[idx].astype(str)
                valid_relief = np.isfinite(sub_relief)
                valid_depth = np.isfinite(sub_depth)
                valid_conf = np.isfinite(sub_conf)
                unsupported_section_tendency_by_component_class[str(comp_class)] = {
                    "n": int(len(sub)),
                    "available": bool(valid_depth.any() or valid_conf.any()),
                    "mean_tendency_depth_fraction": float(np.nanmean(sub_depth)) if valid_depth.any() else np.nan,
                    "median_tendency_depth_fraction": float(np.nanmedian(sub_depth)) if valid_depth.any() else np.nan,
                    "mean_tendency_confidence": float(np.nanmean(sub_conf)) if valid_conf.any() else np.nan,
                    "active_fraction": float(np.count_nonzero(np.nan_to_num(sub_relief, nan=0.0) > 0.05) / max(int(len(sub)), 1)),
                    "simplified_fraction": float(np.count_nonzero(sub_source.str.contains("weak_component_simplified_tendency", regex=False)) / max(int(len(sub)), 1)),
                }
                unsupported_inner_relief_by_component_class[str(comp_class)] = {
                    "n": int(len(sub)),
                    "available": bool(valid_relief.any()),
                    "mean_inner_relief_m": float(np.nanmean(sub_relief)) if valid_relief.any() else np.nan,
                    "median_inner_relief_m": float(np.nanmedian(sub_relief)) if valid_relief.any() else np.nan,
                    "p95_inner_relief_m": float(np.nanpercentile(sub_relief[valid_relief], 95.0)) if valid_relief.any() else np.nan,
                    "max_inner_relief_m": float(np.nanmax(sub_relief)) if valid_relief.any() else np.nan,
                }
    if "component_support_class" in pts.columns:
        if "profile_measured_support_distance_m" in pts.columns:
            dist_series = pd.to_numeric(pts["profile_measured_support_distance_m"], errors="coerce")
            unsupported_pts = pts.loc[dist_series >= 300.0].copy()
        elif "profile_far_from_authoritative_bed_support" in pts.columns:
            unsupported_pts = pts.loc[pd.to_numeric(pts["profile_far_from_authoritative_bed_support"], errors="coerce").fillna(0).astype(bool)].copy()
        else:
            unsupported_pts = pts.iloc[0:0].copy()
        if not unsupported_pts.empty:
            class_series = unsupported_pts["component_support_class"].astype(object)
            unsupported_pts = unsupported_pts.loc[class_series.notna() & (class_series.astype(str).str.len() > 0)].copy()
        if not unsupported_pts.empty:
            for comp_class, sub in unsupported_pts.groupby("component_support_class"):
                comp_vals = pd.to_numeric(sub["final_minus_baseline_m"], errors="coerce").to_numpy(dtype=float)
                comp_abs = np.abs(comp_vals)
                unsupported_component_class_summary[str(comp_class)] = {
                    "n": int(len(sub)),
                    "mean_delta_m": float(np.nanmean(comp_vals)),
                    "mean_abs_delta_m": float(np.nanmean(comp_abs)),
                    "median_abs_delta_m": float(np.nanmedian(comp_abs)),
                    "p95_abs_delta_m": float(np.nanpercentile(comp_abs, 95.0)),
                    "active_fraction": float(np.count_nonzero(comp_abs > 0.001) / max(int(len(sub)), 1)),
                }
                comp_base_steps = []
                comp_final_steps = []
                comp_base_second = []
                comp_final_second = []
                for pid2, sub2 in sub.groupby(profile_col):
                    sub2 = sub2.sort_values(station_col)
                    bv2 = pd.to_numeric(sub2["baseline_sample_m"], errors="coerce").to_numpy(dtype="float64")
                    fv2 = pd.to_numeric(sub2["final_sample_m"], errors="coerce").to_numpy(dtype="float64")
                    valid2 = np.isfinite(bv2) & np.isfinite(fv2)
                    bv2 = bv2[valid2]
                    fv2 = fv2[valid2]
                    if bv2.size >= 2:
                        comp_base_steps.append(np.diff(bv2))
                        comp_final_steps.append(np.diff(fv2))
                    if bv2.size >= 3:
                        comp_base_second.append(np.diff(bv2, n=2))
                        comp_final_second.append(np.diff(fv2, n=2))
                b_steps = _concat(comp_base_steps)
                f_steps = _concat(comp_final_steps)
                b_second = _concat(comp_base_second)
                f_second = _concat(comp_final_second)
                rough_row = {"n": int(len(sub)), "available": bool(b_steps.size > 0 and f_steps.size > 0)}
                if b_steps.size > 0 and f_steps.size > 0:
                    b_sum = _distribution_summary(b_steps)
                    f_sum = _distribution_summary(f_steps)
                    rough_row.update({
                        "baseline_mean_abs_step": b_sum["mean_abs"],
                        "final_mean_abs_step": f_sum["mean_abs"],
                        "delta_mean_abs_step": f_sum["mean_abs"] - b_sum["mean_abs"],
                        "baseline_p95_abs_step": b_sum["p95_abs"],
                        "final_p95_abs_step": f_sum["p95_abs"],
                        "delta_p95_abs_step": f_sum["p95_abs"] - b_sum["p95_abs"],
                        "roughness_improved": f_sum["mean_abs"] < b_sum["mean_abs"],
                    })
                unsupported_component_class_roughness[str(comp_class)] = rough_row
                curv_row = {"n": int(len(sub)), "available": bool(b_second.size > 0 and f_second.size > 0)}
                if b_second.size > 0 and f_second.size > 0:
                    b_sum2 = _distribution_summary(b_second)
                    f_sum2 = _distribution_summary(f_second)
                    curv_row.update({
                        "baseline_mean_abs_second_diff": b_sum2["mean_abs"],
                        "final_mean_abs_second_diff": f_sum2["mean_abs"],
                        "delta_mean_abs_second_diff": f_sum2["mean_abs"] - b_sum2["mean_abs"],
                        "baseline_p95_abs_second_diff": b_sum2["p95_abs"],
                        "final_p95_abs_second_diff": f_sum2["p95_abs"],
                        "delta_p95_abs_second_diff": f_sum2["p95_abs"] - b_sum2["p95_abs"],
                        "curvature_improved": f_sum2["mean_abs"] < b_sum2["mean_abs"],
                    })
                unsupported_component_class_curvature[str(comp_class)] = curv_row

    table_path = bench_dir / "benchmark_river_longitudinal_metrics.csv"
    pd.DataFrame(per_profile_rows).to_csv(table_path, index=False)

    # ---- Support-distance-binned roughness metrics (the hard problem evaluator) ----
    # These bins let you see whether unsupported reaches (far from authoritative
    # data) are getting smoother or rougher relative to baseline.
    distance_bin_metrics: list[dict[str, Any]] = []
    if "profile_measured_support_distance_m" in pts.columns:
        dist_col = pd.to_numeric(pts["profile_measured_support_distance_m"], errors="coerce").to_numpy(dtype=float)
        # Bin edges: [0, 100), [100, 300), [300, 500), [500, 1000), [1000, inf)
        bin_edges = [(0, 100, "0_100m_near_anchor"), (100, 300, "100_300m_transitional"),
                     (300, 500, "300_500m_weak_support"), (500, 1000, "500_1000m_unsupported"),
                     (1000, np.inf, "1000m_plus_far_unsupported")]
        for lo, hi, label in bin_edges:
            bin_mask_pts = (dist_col >= lo) & (dist_col < hi)
            n_bin = int(bin_mask_pts.sum())
            if n_bin < 3:
                distance_bin_metrics.append({"bin": label, "n": n_bin, "available": False})
                continue
            # Recompute step/curvature metrics for stations in this distance bin
            bin_base_steps = []
            bin_final_steps = []
            for pid, sub in pts.loc[bin_mask_pts].groupby(profile_col):
                sub = sub.sort_values(station_col)
                bv = pd.to_numeric(sub["baseline_sample_m"], errors="coerce").to_numpy(dtype="float64")
                fv = pd.to_numeric(sub["final_sample_m"], errors="coerce").to_numpy(dtype="float64")
                valid = np.isfinite(bv) & np.isfinite(fv)
                if valid.sum() >= 2:
                    bin_base_steps.append(np.diff(bv[valid]))
                    bin_final_steps.append(np.diff(fv[valid]))
            b_steps = _concat(bin_base_steps)
            f_steps = _concat(bin_final_steps)
            bin_row: dict[str, Any] = {"bin": label, "n": n_bin, "available": True}
            if b_steps.size > 0 and f_steps.size > 0:
                b_sum = _distribution_summary(b_steps)
                f_sum = _distribution_summary(f_steps)
                bin_row["baseline_mean_abs_step"] = b_sum["mean_abs"]
                bin_row["final_mean_abs_step"] = f_sum["mean_abs"]
                bin_row["delta_mean_abs_step"] = f_sum["mean_abs"] - b_sum["mean_abs"]
                bin_row["baseline_p95_abs_step"] = b_sum["p95_abs"]
                bin_row["final_p95_abs_step"] = f_sum["p95_abs"]
                bin_row["delta_p95_abs_step"] = f_sum["p95_abs"] - b_sum["p95_abs"]
                bin_row["roughness_improved"] = f_sum["mean_abs"] < b_sum["mean_abs"]
            distance_bin_metrics.append(bin_row)

    centerline_agreement_summary: dict[str, Any] | None = None
    centerline_target_col = None
    for candidate in ("generalized_longitudinal_bed_reconciled_elevation_m", "network_backbone_elevation_m", "longitudinal_profile_elevation_m"):
        if candidate in pts.columns:
            candidate_vals = pd.to_numeric(pts[candidate], errors="coerce").to_numpy(dtype="float64")
            if np.any(np.isfinite(candidate_vals)):
                centerline_target_col = candidate
                pts["centerline_target_m"] = candidate_vals
                target_vals = candidate_vals
                target_valid = np.isfinite(target_vals) & np.isfinite(final)
                if np.any(target_valid):
                    abs_err = np.abs(final[target_valid] - target_vals[target_valid])
                    centerline_agreement_summary = {
                        "available": True,
                        "target_field": candidate,
                        "count": int(np.count_nonzero(target_valid)),
                        "mean_abs_error_m": float(np.mean(abs_err)),
                        "p95_abs_error_m": float(np.percentile(abs_err, 95.0)),
                        "max_abs_error_m": float(np.max(abs_err)),
                        "bias_m": float(np.mean(final[target_valid] - target_vals[target_valid])),
                    }
                    target_step_parts = []
                    final_step_parts = []
                    by_component_class = {}
                    group_col = "component_support_class" if "component_support_class" in pts.columns else None
                    for pid, sub in pts.loc[target_valid].groupby(profile_col):
                        sub = sub.sort_values(station_col)
                        tv = pd.to_numeric(sub["centerline_target_m"], errors="coerce").to_numpy(dtype="float64")
                        fv = pd.to_numeric(sub["final_sample_m"], errors="coerce").to_numpy(dtype="float64")
                        valid_pair = np.isfinite(tv) & np.isfinite(fv)
                        if np.count_nonzero(valid_pair) >= 2:
                            target_step_parts.append(np.diff(tv[valid_pair]))
                            final_step_parts.append(np.diff(fv[valid_pair]))
                    target_steps = _concat(target_step_parts)
                    final_target_steps = _concat(final_step_parts)
                    target_step_summary = _distribution_summary(target_steps)
                    final_target_step_summary = _distribution_summary(final_target_steps)
                    target_mean_abs = float(target_step_summary.get("mean_abs", np.nan))
                    target_p95_abs = float(target_step_summary.get("p95_abs", np.nan))
                    final_mean_abs = float(final_target_step_summary.get("mean_abs", np.nan))
                    final_p95_abs = float(final_target_step_summary.get("p95_abs", np.nan))
                    centerline_agreement_summary["roughness"] = {
                        "target": target_step_summary,
                        "final": final_target_step_summary,
                        "roughness_ratio_mean_abs": float(final_mean_abs / target_mean_abs) if target_steps.size and np.isfinite(target_mean_abs) and target_mean_abs > 0 else np.nan,
                        "roughness_ratio_p95_abs": float(final_p95_abs / target_p95_abs) if target_steps.size and np.isfinite(target_p95_abs) and target_p95_abs > 0 else np.nan,
                    }
                    if group_col is not None:
                        cls_series = pts.loc[target_valid, group_col].astype(object)
                        sub_pts = pts.loc[target_valid].copy()
                        sub_pts[group_col] = cls_series
                        sub_pts = sub_pts.loc[sub_pts[group_col].notna() & (sub_pts[group_col].astype(str).str.len() > 0)].copy()
                        for comp_class, sub in sub_pts.groupby(group_col):
                            tv = pd.to_numeric(sub["centerline_target_m"], errors="coerce").to_numpy(dtype="float64")
                            fv = pd.to_numeric(sub["final_sample_m"], errors="coerce").to_numpy(dtype="float64")
                            valid_pair = np.isfinite(tv) & np.isfinite(fv)
                            if not np.any(valid_pair):
                                continue
                            abs_err_cls = np.abs(fv[valid_pair] - tv[valid_pair])
                            row = {
                                "count": int(np.count_nonzero(valid_pair)),
                                "mean_abs_error_m": float(np.mean(abs_err_cls)),
                                "p95_abs_error_m": float(np.percentile(abs_err_cls, 95.0)),
                                "max_abs_error_m": float(np.max(abs_err_cls)),
                                "bias_m": float(np.mean(fv[valid_pair] - tv[valid_pair])),
                            }
                            target_cls_steps_parts = []
                            final_cls_steps_parts = []
                            for pid, sub2 in sub.groupby(profile_col):
                                sub2 = sub2.sort_values(station_col)
                                tv2 = pd.to_numeric(sub2["centerline_target_m"], errors="coerce").to_numpy(dtype="float64")
                                fv2 = pd.to_numeric(sub2["final_sample_m"], errors="coerce").to_numpy(dtype="float64")
                                valid2 = np.isfinite(tv2) & np.isfinite(fv2)
                                if np.count_nonzero(valid2) >= 2:
                                    target_cls_steps_parts.append(np.diff(tv2[valid2]))
                                    final_cls_steps_parts.append(np.diff(fv2[valid2]))
                            target_cls_steps = _concat(target_cls_steps_parts)
                            final_cls_steps = _concat(final_cls_steps_parts)
                            target_cls_summary = _distribution_summary(target_cls_steps)
                            final_cls_summary = _distribution_summary(final_cls_steps)
                            target_cls_mean_abs = float(target_cls_summary.get("mean_abs", np.nan))
                            target_cls_p95_abs = float(target_cls_summary.get("p95_abs", np.nan))
                            final_cls_mean_abs = float(final_cls_summary.get("mean_abs", np.nan))
                            final_cls_p95_abs = float(final_cls_summary.get("p95_abs", np.nan))
                            row["roughness_ratio_mean_abs"] = float(final_cls_mean_abs / target_cls_mean_abs) if target_cls_steps.size and np.isfinite(target_cls_mean_abs) and target_cls_mean_abs > 0 else np.nan
                            row["roughness_ratio_p95_abs"] = float(final_cls_p95_abs / target_cls_p95_abs) if target_cls_steps.size and np.isfinite(target_cls_p95_abs) and target_cls_p95_abs > 0 else np.nan
                            by_component_class[str(comp_class)] = row
                    centerline_agreement_summary["by_component_class"] = by_component_class if by_component_class else None
                break
    authoritative_role_roughness: list[dict[str, Any]] = []
    if "profile_authoritative_role" in pts.columns:
        role_series = pts["profile_authoritative_role"].astype(str)
        for role_label in sorted(x for x in role_series.dropna().unique().tolist() if str(x)):
            role_mask_pts = role_series.astype(str) == str(role_label)
            n_role = int(role_mask_pts.sum())
            if n_role < 3:
                authoritative_role_roughness.append({"authoritative_role": str(role_label), "n": n_role, "available": False})
                continue
            role_base_steps = []
            role_final_steps = []
            for pid, sub in pts.loc[role_mask_pts].groupby(profile_col):
                sub = sub.sort_values(station_col)
                bv = pd.to_numeric(sub["baseline_sample_m"], errors="coerce").to_numpy(dtype="float64")
                fv = pd.to_numeric(sub["final_sample_m"], errors="coerce").to_numpy(dtype="float64")
                valid = np.isfinite(bv) & np.isfinite(fv)
                if valid.sum() >= 2:
                    role_base_steps.append(np.diff(bv[valid]))
                    role_final_steps.append(np.diff(fv[valid]))
            b_steps = _concat(role_base_steps)
            f_steps = _concat(role_final_steps)
            row = {"authoritative_role": str(role_label), "n": n_role, "available": True}
            if b_steps.size > 0 and f_steps.size > 0:
                b_sum = _distribution_summary(b_steps)
                f_sum = _distribution_summary(f_steps)
                row["baseline_mean_abs_step"] = b_sum["mean_abs"]
                row["final_mean_abs_step"] = f_sum["mean_abs"]
                row["delta_mean_abs_step"] = f_sum["mean_abs"] - b_sum["mean_abs"]
                row["baseline_p95_abs_step"] = b_sum["p95_abs"]
                row["final_p95_abs_step"] = f_sum["p95_abs"]
                row["delta_p95_abs_step"] = f_sum["p95_abs"] - b_sum["p95_abs"]
                row["roughness_improved"] = f_sum["mean_abs"] < b_sum["mean_abs"]
            authoritative_role_roughness.append(row)

    baseline_step_summary = _distribution_summary(baseline_steps)
    final_step_summary = _distribution_summary(final_steps)
    change_step_summary = _distribution_summary(change_steps)
    baseline_second_summary = _distribution_summary(baseline_second)
    final_second_summary = _distribution_summary(final_second)
    change_second_summary = _distribution_summary(change_second)

    summary = {
        "available": True,
        "profile_points_path": str(profile_points_path),
        "coverage": coverage_info,
        "counts": {
            "profile_count": int(len(per_profile_rows)),
            "profile_point_count": int(len(pts)),
            "step_count": int(baseline_steps.size),
            "curvature_count": int(baseline_second.size),
        },
        "longitudinal_source_counts": source_counts,
        "adjacent_step_metrics": {
            "baseline": baseline_step_summary,
            "final": final_step_summary,
            "change_final_minus_baseline": change_step_summary,
            "delta_final_minus_baseline_mean_abs": float(final_step_summary["mean_abs"] - baseline_step_summary["mean_abs"]) if baseline_steps.size or final_steps.size else np.nan,
            "delta_final_minus_baseline_p95_abs": float(final_step_summary["p95_abs"] - baseline_step_summary["p95_abs"]) if baseline_steps.size or final_steps.size else np.nan,
        },
        "curvature_metrics": {
            "baseline": baseline_second_summary,
            "final": final_second_summary,
            "change_final_minus_baseline": change_second_summary,
            "delta_final_minus_baseline_mean_abs": float(final_second_summary["mean_abs"] - baseline_second_summary["mean_abs"]) if baseline_second.size or final_second.size else np.nan,
            "delta_final_minus_baseline_p95_abs": float(final_second_summary["p95_abs"] - baseline_second_summary["p95_abs"]) if baseline_second.size or final_second.size else np.nan,
        },
        "support_distance_binned_roughness": distance_bin_metrics if distance_bin_metrics else None,
        "authoritative_role_binned_roughness": authoritative_role_roughness if authoritative_role_roughness else None,
        "unsupported_by_component_class": unsupported_component_class_summary if unsupported_component_class_summary else None,
        "unsupported_roughness_by_component_class": unsupported_component_class_roughness if unsupported_component_class_roughness else None,
        "unsupported_curvature_by_component_class": unsupported_component_class_curvature if unsupported_component_class_curvature else None,
        "unsupported_section_tendency_by_component_class": unsupported_section_tendency_by_component_class if unsupported_section_tendency_by_component_class else None,
        "unsupported_inner_relief_by_component_class": unsupported_inner_relief_by_component_class if unsupported_inner_relief_by_component_class else None,
        "unsupported_reconciliation_by_component_class": unsupported_reconciliation_by_component_class if unsupported_reconciliation_by_component_class else None,
        "centerline_agreement": centerline_agreement_summary or {"available": False, "target_field": centerline_target_col},
        "per_profile_csv": str(table_path),
    }
    return summary, table_path


def _metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if obs.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan}
    resid = pred - obs
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"n": int(obs.size), "rmse": rmse, "mae": mae, "bias": bias, "r2": r2}


def _delta_metrics(final_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rmse": float(final_metrics["rmse"] - baseline_metrics["rmse"]),
        "mae": float(final_metrics["mae"] - baseline_metrics["mae"]),
        "bias": float(final_metrics["bias"] - baseline_metrics["bias"]),
        "r2": float(final_metrics["r2"] - baseline_metrics["r2"]) if np.isfinite(final_metrics["r2"]) and np.isfinite(baseline_metrics["r2"]) else np.nan,
    }


def _improvement_summary(obs: np.ndarray, baseline: np.ndarray, final: np.ndarray) -> dict[str, Any]:
    if obs.size == 0:
        return {
            "count": 0,
            "improved_count": 0,
            "improved_fraction": np.nan,
            "worsened_count": 0,
            "worsened_fraction": np.nan,
            "unchanged_count": 0,
            "unchanged_fraction": np.nan,
            "mean_abs_error_delta_final_minus_baseline": np.nan,
        }
    base_abs = np.abs(baseline - obs)
    final_abs = np.abs(final - obs)
    tol = 1e-9
    improved = final_abs < (base_abs - tol)
    worsened = final_abs > (base_abs + tol)
    unchanged = ~(improved | worsened)
    n = int(obs.size)
    return {
        "count": n,
        "improved_count": int(np.count_nonzero(improved)),
        "improved_fraction": float(np.count_nonzero(improved) / n) if n > 0 else np.nan,
        "worsened_count": int(np.count_nonzero(worsened)),
        "worsened_fraction": float(np.count_nonzero(worsened) / n) if n > 0 else np.nan,
        "unchanged_count": int(np.count_nonzero(unchanged)),
        "unchanged_fraction": float(np.count_nonzero(unchanged) / n) if n > 0 else np.nan,
        "mean_abs_error_delta_final_minus_baseline": float(np.nanmean(final_abs - base_abs)),
    }


def _paired_metrics(obs: np.ndarray, baseline: np.ndarray, final: np.ndarray) -> dict[str, Any]:
    baseline_metrics = _metrics(obs, baseline)
    final_metrics = _metrics(obs, final)
    return {
        "baseline": baseline_metrics,
        "final": final_metrics,
        "delta_final_minus_baseline": _delta_metrics(final_metrics, baseline_metrics),
        "improvement": _improvement_summary(obs, baseline, final),
    }


def _summarize_group_metrics(df: pd.DataFrame, *, group_col: str, labels: Optional[dict[Any, str]] = None) -> dict[str, Any]:
    if group_col not in df.columns:
        return {"available": False}
    work = df.loc[df[group_col].notna()].copy()
    if work.empty:
        return {"available": False}
    groups: dict[str, Any] = {}
    for raw_value in sorted(work[group_col].unique().tolist(), key=lambda v: str(v)):
        sub = work.loc[work[group_col] == raw_value]
        obs = pd.to_numeric(sub["obs"], errors="coerce").to_numpy(dtype=float)
        baseline = pd.to_numeric(sub["baseline"], errors="coerce").to_numpy(dtype=float)
        final = pd.to_numeric(sub["final"], errors="coerce").to_numpy(dtype=float)
        key = str(raw_value)
        label = labels.get(raw_value) if isinstance(labels, dict) else None
        if label is None:
            label = key
        groups[key] = {"label": str(label), **_paired_metrics(obs, baseline, final)}
    return {"available": True, "group_col": group_col, "n": int(len(work)), "groups": groups}


def _confidence_bin(value: float) -> Optional[str]:
    if not np.isfinite(value):
        return None
    if value >= 0.75:
        return "high"
    if value >= 0.40:
        return "medium"
    return "low"


def _measured_anchor_bin(value: float) -> Optional[str]:
    if not np.isfinite(value):
        return None
    if value >= 0.75:
        return "anchor_dominant"
    if value >= 0.05:
        return "mixed"
    return "none"


def _structure_only_bin(value: float) -> Optional[str]:
    if not np.isfinite(value):
        return None
    if value >= 0.75:
        return "structure_dominant"
    if value >= 0.25:
        return "mixed"
    return "low"


def _build_science_evaluation_grid(
    *,
    baseline_raster: Path,
    final_raster: Path,
    support_class_raster: Optional[Path],
    river_mask_raster: Optional[Path],
    prediction_admissibility_raster: Optional[Path],
    bench_dir: Path,
    grid_step: int = 3,
    logger,
) -> dict[str, Any]:
    """Sample a regular grid in non-authoritative river zones to evaluate science effect.

    Unlike the authoritative-invariance holdout, these points have no ground truth.
    We compare baseline vs final to measure whether the river science is active and
    whether it changes the surface in guidance-conditioned zones.
    """
    result: dict[str, Any] = {"available": False}
    if support_class_raster is None or not support_class_raster.exists():
        return result
    with rasterio.open(baseline_raster) as bds, rasterio.open(final_raster) as fds, rasterio.open(support_class_raster) as sds:
        _assert_same_grid(reference=bds, candidate=fds, candidate_label="final_raster")
        _assert_same_grid(reference=bds, candidate=sds, candidate_label="support_class_raster")
        base_arr = bds.read(1).astype("float64")
        final_arr = fds.read(1).astype("float64")
        sc_arr = sds.read(1).astype("float64")
        if bds.nodata is not None:
            base_arr[np.isclose(base_arr, bds.nodata)] = np.nan
        if fds.nodata is not None:
            final_arr[np.isclose(final_arr, fds.nodata)] = np.nan
        if sds.nodata is not None:
            sc_arr[np.isclose(sc_arr, sds.nodata)] = np.nan
        transform = bds.transform

    river_arr = None
    if river_mask_raster is not None and river_mask_raster.exists():
        river_arr = _align_raster_to_template(
            river_mask_raster,
            baseline_raster,
            dtype="float64",
            nodata_value=0.0,
            resampling=Resampling.nearest,
        )
        if river_arr is not None:
            river_arr[~np.isfinite(river_arr)] = 0.0

    admiss_arr = None
    if prediction_admissibility_raster is not None and prediction_admissibility_raster.exists():
        admiss_arr = _align_raster_to_template(
            prediction_admissibility_raster,
            baseline_raster,
            dtype="float64",
            nodata_value=np.nan,
            resampling=Resampling.nearest,
        )

    auth_code = _support_class_code("authoritative_locked")
    if auth_code is None:
        raise ValueError("Could not resolve support-class code for authoritative_locked")

    populated = np.isfinite(base_arr) & np.isfinite(final_arr) & np.isfinite(sc_arr)
    non_auth = populated & (~np.isclose(sc_arr, float(auth_code)))
    if river_arr is not None:
        non_auth = non_auth & (river_arr > 0)

    rows_grid, cols_grid = np.where(non_auth)
    if rows_grid.size == 0:
        result["available"] = True
        result["reason"] = "no_non_authoritative_river_cells"
        result["non_authoritative_cell_count"] = 0
        return result

    step = max(int(grid_step), 1)
    mask = np.zeros(non_auth.shape, dtype=bool)
    mask[::step, ::step] = True
    grid_mask = non_auth & mask
    gr, gc = np.where(grid_mask)
    if gr.size == 0:
        gr, gc = rows_grid, cols_grid

    base_vals = base_arr[gr, gc]
    final_vals = final_arr[gr, gc]
    sc_vals = sc_arr[gr, gc]
    admiss_vals = admiss_arr[gr, gc] if admiss_arr is not None else None
    delta = final_vals - base_vals
    abs_delta = np.abs(delta)

    class_summaries = {}
    for code in sorted(np.unique(sc_vals[np.isfinite(sc_vals)])):
        code_int = int(round(code))
        cls_mask = np.isclose(sc_vals, code)
        cls_name = str(SUPPORT_CLASS_CODE_TO_NAME.get(code_int, f"unknown_{code_int}"))
        n = int(np.count_nonzero(cls_mask))
        if n == 0:
            continue
        cls_delta = delta[cls_mask]
        cls_abs = abs_delta[cls_mask]
        class_summaries[cls_name] = {
            "n": n,
            "mean_delta_m": float(np.nanmean(cls_delta)),
            "mean_abs_delta_m": float(np.nanmean(cls_abs)),
            "median_abs_delta_m": float(np.nanmedian(cls_abs)),
            "p95_abs_delta_m": float(np.nanpercentile(cls_abs, 95.0)),
            "max_abs_delta_m": float(np.nanmax(cls_abs)),
            "active_fraction": float(np.count_nonzero(cls_abs > 0.001) / max(n, 1)),
        }

    xs, ys = rasterio.transform.xy(transform, gr, gc, offset="center")
    grid_df = pd.DataFrame({
        "x": np.asarray(xs, dtype=float),
        "y": np.asarray(ys, dtype=float),
        "baseline": base_vals,
        "final": final_vals,
        "delta": delta,
        "abs_delta": abs_delta,
        "support_class": sc_vals.astype(int),
        "support_class_label": _support_class_labels(sc_vals),
        "benchmark_stratum": "science_evaluation",
    })
    if admiss_vals is not None:
        grid_df["prediction_admissibility"] = pd.Series(admiss_vals).round().astype("Int64")
        grid_df["prediction_admissibility_label"] = grid_df["prediction_admissibility"].map({0: "inadmissible", 1: "admissible"})
    grid_path = bench_dir / "benchmark_science_evaluation_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    active_count = int(np.count_nonzero(abs_delta > 0.001))
    unsupported_mask = np.array([_is_unsupported_support_code(int(round(v))) for v in sc_vals], dtype=bool)
    unsupported_summary = {
        "n": int(np.count_nonzero(unsupported_mask)),
        "mean_delta_m": float(np.nanmean(delta[unsupported_mask])) if np.any(unsupported_mask) else np.nan,
        "mean_abs_delta_m": float(np.nanmean(abs_delta[unsupported_mask])) if np.any(unsupported_mask) else np.nan,
        "median_abs_delta_m": float(np.nanmedian(abs_delta[unsupported_mask])) if np.any(unsupported_mask) else np.nan,
        "p95_abs_delta_m": float(np.nanpercentile(abs_delta[unsupported_mask], 95.0)) if np.any(unsupported_mask) else np.nan,
        "max_abs_delta_m": float(np.nanmax(abs_delta[unsupported_mask])) if np.any(unsupported_mask) else np.nan,
        "active_fraction": float(np.count_nonzero(abs_delta[unsupported_mask] > 0.001) / max(int(np.count_nonzero(unsupported_mask)), 1)) if np.any(unsupported_mask) else np.nan,
    }
    admiss_summaries = {}
    if admiss_vals is not None:
        admiss_series = pd.Series(admiss_vals).round().astype("Int64")
        for raw_value in sorted([v for v in admiss_series.dropna().unique().tolist()]):
            comp_v = admiss_series == raw_value
            if hasattr(comp_v, 'fillna'):
                comp_v = comp_v.fillna(False)
            try:
                mask_v = comp_v.to_numpy(dtype=bool, na_value=False)
            except TypeError:
                mask_v = np.asarray(comp_v.fillna(False) if hasattr(comp_v, 'fillna') else comp_v, dtype=bool)
            n_v = int(np.count_nonzero(mask_v))
            if n_v == 0:
                continue
            cls_abs = abs_delta[mask_v]
            cls_delta = delta[mask_v]
            label = "admissible" if int(raw_value) == 1 else "inadmissible"
            admiss_summaries[str(int(raw_value))] = {
                "label": label,
                "n": n_v,
                "mean_delta_m": float(np.nanmean(cls_delta)),
                "mean_abs_delta_m": float(np.nanmean(cls_abs)),
                "median_abs_delta_m": float(np.nanmedian(cls_abs)),
                "p95_abs_delta_m": float(np.nanpercentile(cls_abs, 95.0)),
                "max_abs_delta_m": float(np.nanmax(cls_abs)),
                "active_fraction": float(np.count_nonzero(cls_abs > 0.001) / max(n_v, 1)),
            }

    result = {
        "available": True,
        "non_authoritative_cell_count": int(np.count_nonzero(non_auth)),
        "grid_point_count": int(gr.size),
        "grid_step": step,
        "active_count": active_count,
        "active_fraction": float(active_count / max(gr.size, 1)),
        "overall": {
            "mean_delta_m": float(np.nanmean(delta)),
            "mean_abs_delta_m": float(np.nanmean(abs_delta)),
            "median_abs_delta_m": float(np.nanmedian(abs_delta)),
            "p95_abs_delta_m": float(np.nanpercentile(abs_delta, 95.0)),
            "max_abs_delta_m": float(np.nanmax(abs_delta)),
        },
        "unsupported_river": unsupported_summary,
        "by_support_class": class_summaries,
        "by_prediction_admissibility": admiss_summaries,
        "grid_path": str(grid_path),
    }
    logger.info(
        "[BENCHMARK] Science evaluation grid: %d points, %d active (%.1f%%), mean_abs_delta=%.4f m",
        gr.size, active_count, 100.0 * active_count / max(gr.size, 1), float(np.nanmean(abs_delta)),
    )
    return result


def _compute_support_aware_validation_summary(scored_df: pd.DataFrame) -> dict[str, Any]:
    if scored_df.empty:
        return {"available": False}
    river_df = scored_df.loc[scored_df.get("river_mask", pd.Series(False, index=scored_df.index)).astype(bool)].copy() if "river_mask" in scored_df.columns else scored_df.copy()
    if river_df.empty:
        return {"available": False, "river_holdout_count": 0}
    work = river_df.copy()
    if "prediction_support_confidence" in work.columns:
        work["prediction_support_confidence_bin"] = [
            _confidence_bin(float(v)) if pd.notna(v) else None for v in pd.to_numeric(work["prediction_support_confidence"], errors="coerce")
        ]
    if "prediction_measured_anchor_fraction" in work.columns:
        work["prediction_measured_anchor_fraction_bin"] = [
            _measured_anchor_bin(float(v)) if pd.notna(v) else None for v in pd.to_numeric(work["prediction_measured_anchor_fraction"], errors="coerce")
        ]
    if "prediction_structure_only_fraction" in work.columns:
        work["prediction_structure_only_fraction_bin"] = [
            _structure_only_bin(float(v)) if pd.notna(v) else None for v in pd.to_numeric(work["prediction_structure_only_fraction"], errors="coerce")
        ]
    if "authoritative_role_confidence" in work.columns:
        work["authoritative_role_confidence_bin"] = [
            _authoritative_role_confidence_bin(float(v)) if pd.notna(v) else None for v in pd.to_numeric(work["authoritative_role_confidence"], errors="coerce")
        ]
    bool_labels = {0: "absent", 1: "present"}
    summary = {
        "available": True,
        "river_holdout_count": int(len(work)),
        "overall_river_holdout": _paired_metrics(
            pd.to_numeric(work["obs"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(work["baseline"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(work["final"], errors="coerce").to_numpy(dtype=float),
        ),
        "by_prediction_admissibility": _summarize_group_metrics(work, group_col="prediction_admissibility", labels=bool_labels),
        "by_low_support_caution": _summarize_group_metrics(work, group_col="prediction_low_support_caution", labels=bool_labels),
        "by_prediction_support_confidence_bin": _summarize_group_metrics(work, group_col="prediction_support_confidence_bin"),
        "by_prediction_measured_anchor_fraction_bin": _summarize_group_metrics(work, group_col="prediction_measured_anchor_fraction_bin"),
        "by_prediction_structure_only_fraction_bin": _summarize_group_metrics(work, group_col="prediction_structure_only_fraction_bin"),
        "by_support_class_in_river": _summarize_group_metrics(work, group_col="support_class", labels=SUPPORT_CLASS_CODE_TO_NAME),
    }
    if "provenance" in work.columns:
        summary["by_provenance_in_river"] = _summarize_group_metrics(work, group_col="provenance")
    if "authoritative_role_label" in work.columns:
        summary["by_authoritative_role_in_river"] = _summarize_group_metrics(work, group_col="authoritative_role_label")
    if "authoritative_role_confidence_bin" in work.columns:
        summary["by_authoritative_role_confidence_bin"] = _summarize_group_metrics(work, group_col="authoritative_role_confidence_bin")
    if "authoritative_bed_support_present" in work.columns:
        summary["by_authoritative_bed_support_present"] = _summarize_group_metrics(work, group_col="authoritative_bed_support_present", labels=bool_labels)
    if "authoritative_bank_margin_present" in work.columns:
        summary["by_authoritative_bank_margin_present"] = _summarize_group_metrics(work, group_col="authoritative_bank_margin_present", labels=bool_labels)
    return summary


def _compute_authoritative_role_validation_summary(scored_df: pd.DataFrame) -> dict[str, Any]:
    if scored_df.empty or "authoritative_role_label" not in scored_df.columns:
        return {"available": False}
    work = scored_df.copy()
    if "river_mask" in work.columns:
        work = work.loc[work["river_mask"].fillna(False).astype(bool)].copy()
    if work.empty:
        return {"available": False, "river_holdout_count": 0}
    if "authoritative_role_confidence" in work.columns:
        work["authoritative_role_confidence_bin"] = [
            _authoritative_role_confidence_bin(float(v)) if pd.notna(v) else None for v in pd.to_numeric(work["authoritative_role_confidence"], errors="coerce")
        ]
    role_counts = {str(k): int(v) for k, v in work["authoritative_role_label"].astype(str).value_counts(dropna=False).to_dict().items()}
    summary = {
        "available": True,
        "river_holdout_count": int(len(work)),
        "authoritative_role_counts": role_counts,
        "by_authoritative_role": _summarize_group_metrics(work, group_col="authoritative_role_label"),
    }
    if "authoritative_role_confidence_bin" in work.columns:
        summary["by_role_confidence_bin"] = _summarize_group_metrics(work, group_col="authoritative_role_confidence_bin")
    if "authoritative_bed_support_present" in work.columns:
        summary["by_authoritative_bed_support_present"] = _summarize_group_metrics(work, group_col="authoritative_bed_support_present", labels={0: "absent", 1: "present"})
    if "authoritative_bank_margin_present" in work.columns:
        summary["by_authoritative_bank_margin_present"] = _summarize_group_metrics(work, group_col="authoritative_bank_margin_present", labels={0: "absent", 1: "present"})
    return summary




def _build_river_benchmark_mode_summary(*, support_aware_science: dict[str, Any], river_receipt_triage: dict[str, Any], hard_river_summary: dict[str, Any], auto_holdout_receipt: Optional[dict[str, Any]], withheld_support_receipt: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if not isinstance(support_aware_science, dict) or not support_aware_science.get("available"):
        return {"available": False, "reason": "support_aware_science_unavailable"}

    hard_blind = bool(support_aware_science.get("hard_problem_holdout_blind", False))
    unsupported = support_aware_science.get("unsupported_roughness", {}) if isinstance(support_aware_science.get("unsupported_roughness", {}), dict) else {}
    transition = support_aware_science.get("transition_quality", {}) if isinstance(support_aware_science.get("transition_quality", {}), dict) else {}
    grid = support_aware_science.get("unsupported_science_grid", {}) if isinstance(support_aware_science.get("unsupported_science_grid", {}), dict) else {}
    comp = support_aware_science.get("unsupported_mainstem_vs_side_component", {}) if isinstance(support_aware_science.get("unsupported_mainstem_vs_side_component", {}), dict) else {}
    lateral = river_receipt_triage.get("lateral_accountability", {}) if isinstance(river_receipt_triage.get("lateral_accountability", {}), dict) else {}
    hard_group = hard_river_summary.get("hard_problem_group", {}) if isinstance(hard_river_summary, dict) and isinstance(hard_river_summary.get("hard_problem_group", {}), dict) else {}
    support_counts = hard_river_summary.get("river_support_class_counts", {}) if isinstance(hard_river_summary, dict) and isinstance(hard_river_summary.get("river_support_class_counts", {}), dict) else {}

    withheld_applied = bool(isinstance(withheld_support_receipt, dict) and withheld_support_receipt.get("applied"))
    canonical_mode = "withheld_support_primary" if withheld_applied else "support_aware_primary"
    canonical_decision_basis = "withheld_support_direct_test" if withheld_applied else "support_aware_science_and_receipts"
    legacy_holdout_basis = "support_aware_primary" if hard_blind else "mixed_holdout_and_support_aware"
    legacy_holdout_signal = "support_aware_science_and_receipts" if hard_blind else "hard_river_holdout_plus_support_aware_science"

    unsupported_delta = float(unsupported.get("delta_mean_abs_step", np.nan)) if np.isfinite(float(unsupported.get("delta_mean_abs_step", np.nan))) else np.nan
    transition_delta = float(transition.get("delta_mean_abs_step", np.nan)) if np.isfinite(float(transition.get("delta_mean_abs_step", np.nan))) else np.nan
    mainstem_step = float(comp.get("mainstem_final_mean_abs_step", np.nan)) if np.isfinite(float(comp.get("mainstem_final_mean_abs_step", np.nan))) else np.nan
    side_step = float(comp.get("side_final_mean_abs_step", np.nan)) if np.isfinite(float(comp.get("side_final_mean_abs_step", np.nan))) else np.nan
    mainstem_delta = float(comp.get("mainstem_mean_abs_delta_m", np.nan)) if np.isfinite(float(comp.get("mainstem_mean_abs_delta_m", np.nan))) else np.nan
    side_delta = float(comp.get("side_mean_abs_delta_m", np.nan)) if np.isfinite(float(comp.get("side_mean_abs_delta_m", np.nan))) else np.nan
    lateral_issue = lateral.get("dominant_issue") if isinstance(lateral.get("dominant_issue"), str) else None

    priority_domain = "unsupported_river"
    priority_reason = "Use unsupported-river support-aware diagnostics as the primary benchmark signal."
    if comp.get("available") and np.isfinite(mainstem_step) and np.isfinite(side_step):
        if mainstem_step >= side_step:
            priority_domain = "unsupported_mainstem"
            priority_reason = "Unsupported mainstem shows the larger final longitudinal roughness burden and should drive river benchmark decisions."
        else:
            priority_domain = "unsupported_side_component"
            priority_reason = "Unsupported side components are rougher than unsupported mainstem and should drive river benchmark decisions."
    elif np.isfinite(unsupported_delta) and unsupported_delta > 0.02:
        priority_domain = "unsupported_river"
        priority_reason = "Unsupported river roughness regressed relative to baseline, so support-aware river diagnostics should drive benchmark decisions."
    elif np.isfinite(transition_delta) and transition_delta > 0.02:
        priority_domain = "authoritative_transition"
        priority_reason = "Authoritative transition engagement regressed relative to baseline, so transition diagnostics should drive benchmark decisions."
    elif lateral_issue:
        priority_domain = str(lateral_issue)
        priority_reason = "Holdout alone is not sufficient; use the lateral-accountability receipts as the primary river benchmark tie-breaker."

    unsupported_class_counts = grid.get("by_support_class", {}) if isinstance(grid.get("by_support_class", {}), dict) else {}
    unsupported_holdout_count = int(hard_group.get("n", 0) or 0)
    unsupported_grid_count = int(grid.get("cell_count", 0) or 0)
    holdout_coverage = {
        "river_point_count": int(hard_river_summary.get("river_point_count", 0) or 0) if isinstance(hard_river_summary, dict) else 0,
        "unsupported_holdout_point_count": unsupported_holdout_count,
        "unsupported_grid_cell_count": unsupported_grid_count,
        "support_class_counts": support_counts,
        "unsupported_support_class_counts": {str(k): v for k, v in unsupported_class_counts.items()},
    }
    holdout_drilldown_available = (not hard_blind) and bool(auto_holdout_receipt is not None)

    return {
        "available": True,
        "evaluation_basis": canonical_mode,
        "active_river_benchmark_mode": canonical_mode,
        "active_river_benchmark_decision_basis": canonical_decision_basis,
        "primary_decision_signal": canonical_decision_basis,
        "legacy_holdout_evaluation_basis": legacy_holdout_basis,
        "legacy_holdout_primary_decision_signal": legacy_holdout_signal,
        "holdout_drilldown_available": holdout_drilldown_available,
        "holdout_drilldown_reason": (
            "Hard-problem holdout coverage exists, but canonical workflow still treats support-aware river diagnostics as the primary evaluation path. Use holdout only as a drilldown."
            if holdout_drilldown_available else None
        ),
        "hard_problem_holdout_blind": hard_blind,
        "hard_problem_blind_reason": support_aware_science.get("hard_problem_blind_reason") if isinstance(support_aware_science.get("hard_problem_blind_reason"), str) else (hard_river_summary.get("hard_problem_blind_reason") if isinstance(hard_river_summary, dict) else None),
        "river_specific_withheld_support_benchmark_recommended": bool(hard_blind and not withheld_applied),
        "withheld_support_benchmark_reason": (
            "Auto-holdout and generic holdout do not directly measure unsupported river reaches; use the river_withheld_support_points.csv plan to remove usable in-channel support before channel-surface generation on the next run."
            if hard_blind and not withheld_applied else None
        ),
        "withheld_support_plan_available": bool(isinstance(withheld_support_receipt, dict) and withheld_support_receipt.get("available")),
        "withheld_support_plan_path": withheld_support_receipt.get("path") if isinstance(withheld_support_receipt, dict) else None,
        "withheld_support_cli_flag": (
            f'--river-withheld-support-csv "{withheld_support_receipt.get("path")}"'
            if isinstance(withheld_support_receipt, dict) and withheld_support_receipt.get("path")
            else ('--river-withheld-support-csv <path-to-benchmark/river_withheld_support_points.csv>' if hard_blind and not withheld_applied else None)
        ),
        "auto_holdout_used": bool(auto_holdout_receipt is not None),
        "priority_domain": priority_domain,
        "priority_reason": priority_reason,
        "unsupported_delta_mean_abs_step": unsupported_delta,
        "transition_delta_mean_abs_step": transition_delta,
        "mainstem_final_mean_abs_step": mainstem_step,
        "side_final_mean_abs_step": side_step,
        "mainstem_mean_abs_delta_m": mainstem_delta,
        "side_mean_abs_delta_m": side_delta,
        "holdout_support_coverage": holdout_coverage,
    }


def _build_primary_river_benchmark_focus(*, support_aware_science: dict[str, Any], river_receipt_triage: dict[str, Any], hard_river_summary: dict[str, Any], benchmark_mode_summary: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if not isinstance(support_aware_science, dict) or not support_aware_science.get("available"):
        return {"available": False, "reason": "support_aware_science_unavailable"}
    unsupported = support_aware_science.get("unsupported_roughness", {}) if isinstance(support_aware_science.get("unsupported_roughness", {}), dict) else {}
    transition = support_aware_science.get("transition_quality", {}) if isinstance(support_aware_science.get("transition_quality", {}), dict) else {}
    grid = support_aware_science.get("unsupported_science_grid", {}) if isinstance(support_aware_science.get("unsupported_science_grid", {}), dict) else {}
    triage_primary = river_receipt_triage.get("primary_focus") if isinstance(river_receipt_triage, dict) else None
    triage_action = river_receipt_triage.get("suggested_next_action") if isinstance(river_receipt_triage, dict) else None
    lateral = river_receipt_triage.get("lateral_accountability", {}) if isinstance(river_receipt_triage.get("lateral_accountability", {}), dict) else {}
    hard_blind = bool(support_aware_science.get("hard_problem_holdout_blind", False))
    unsupported_delta = unsupported.get("delta_mean_abs_step", np.nan)
    transition_delta = transition.get("delta_mean_abs_step", np.nan)
    centerline_p95 = np.nan
    smoothing_changed = 0
    smoothing_eligible = 0
    lateral_issue = lateral.get("dominant_issue") if isinstance(lateral.get("dominant_issue"), str) else None
    lateral_role = lateral.get("weakest_role") if isinstance(lateral.get("weakest_role"), str) else None
    lateral_p95 = lateral.get("weakest_role_p95_abs_error_m", np.nan)
    if isinstance(river_receipt_triage, dict):
        receipts = river_receipt_triage.get("receipts", {}) if isinstance(river_receipt_triage.get("receipts", {}), dict) else {}
        center = receipts.get("centerline_agreement", {}) if isinstance(receipts.get("centerline_agreement", {}), dict) else {}
        centerline_p95 = center.get("p95_abs_error_m", np.nan)
        smoothing = receipts.get("longitudinal_smoothing", {}) if isinstance(receipts.get("longitudinal_smoothing", {}), dict) else {}
        smoothing_changed = int(smoothing.get("changed_count", 0) or 0)
        smoothing_eligible = int(smoothing.get("eligible_count", 0) or 0)
    if np.isfinite(unsupported_delta) and unsupported_delta > 0.02:
        focus = "unsupported_roughness"
        action = "Focus on weak-support longitudinal smoothness and transition engagement before relying on holdout RMSE."
    elif np.isfinite(transition_delta) and transition_delta > 0.02:
        focus = "authoritative_transition"
        action = "Inspect authoritative interior-bed transition candidates and verify nonzero taper weights in the active render path."
    elif smoothing_eligible > 0 and smoothing_changed == 0 and np.isfinite(centerline_p95) and centerline_p95 > 0.5:
        focus = "longitudinal_smoothing_activation"
        action = "Inspect the channel-surface longitudinal smoothing receipt and effect summary; eligible weak-support reaches did not produce longitudinal smoothing changes."
    elif lateral_issue and np.isfinite(lateral_p95) and lateral_p95 > 0.35:
        focus = lateral_issue
        action = triage_action or "Inspect role-agreement and section-target receipts to isolate whether thalweg, inner-shape, or bank-edge is still driving the river error."
    elif np.isfinite(centerline_p95) and centerline_p95 > 0.5:
        focus = "centerline_alignment"
        action = triage_action or "Inspect centerline agreement, backbone smoothing, and thalweg render effect receipts."
    else:
        focus = triage_primary or lateral_issue or "unsupported_river"
        action = triage_action or "Inspect support-aware river diagnostics before using holdout metrics as the main decision signal."
    mode_summary = benchmark_mode_summary if isinstance(benchmark_mode_summary, dict) else {}
    if hard_blind:
        action = f"{action} Holdout is blind to unsupported river, so prioritize support-aware diagnostics."
    return {
        "available": True,
        "hard_problem_holdout_blind": hard_blind,
        "evaluation_basis": mode_summary.get("evaluation_basis") if mode_summary else ("support_aware_primary" if hard_blind else "mixed_holdout_and_support_aware"),
        "primary_decision_signal": mode_summary.get("primary_decision_signal") if mode_summary else ("support_aware_science_and_receipts" if hard_blind else "hard_river_holdout_plus_support_aware_science"),
        "priority_domain": mode_summary.get("priority_domain") if mode_summary else None,
        "priority_reason": mode_summary.get("priority_reason") if mode_summary else None,
        "river_specific_withheld_support_benchmark_recommended": bool(mode_summary.get("river_specific_withheld_support_benchmark_recommended", False)) if mode_summary else bool(hard_blind),
        "withheld_support_benchmark_reason": mode_summary.get("withheld_support_benchmark_reason") if mode_summary else None,
        "withheld_support_plan_available": bool(mode_summary.get("withheld_support_plan_available", False)) if mode_summary else False,
        "withheld_support_plan_path": mode_summary.get("withheld_support_plan_path") if mode_summary else None,
        "hard_problem_blind_reason": support_aware_science.get("hard_problem_blind_reason") if isinstance(support_aware_science.get("hard_problem_blind_reason"), str) else (hard_river_summary.get("hard_problem_blind_reason") if isinstance(hard_river_summary, dict) else None),
        "primary_focus": focus,
        "suggested_next_action": action,
        "unsupported_delta_mean_abs_step": float(unsupported_delta) if np.isfinite(unsupported_delta) else np.nan,
        "transition_delta_mean_abs_step": float(transition_delta) if np.isfinite(transition_delta) else np.nan,
        "unsupported_station_count": int(unsupported.get("station_count", 0)) if unsupported else 0,
        "transition_station_count": int(transition.get("station_count", 0)) if transition else 0,
        "unsupported_grid_cell_count": int(grid.get("cell_count", 0)) if grid else 0,
        "holdout_support_coverage": mode_summary.get("holdout_support_coverage") if mode_summary else None,
        "triage_primary_focus": triage_primary,
        "triage_weakest_role": lateral_role,
        "triage_weakest_role_p95_abs_error_m": float(lateral_p95) if np.isfinite(lateral_p95) else np.nan,
        "longitudinal_smoothing_changed_count": smoothing_changed,
        "longitudinal_smoothing_eligible_count": smoothing_eligible,
    }


def _build_active_river_evaluation_summary(*, benchmark_mode_summary: Optional[dict[str, Any]] = None, river_primary_focus_summary: Optional[dict[str, Any]] = None, river_receipt_triage_summary: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    mode_summary = benchmark_mode_summary if isinstance(benchmark_mode_summary, dict) else {}
    focus_summary = river_primary_focus_summary if isinstance(river_primary_focus_summary, dict) else {}
    triage_summary = river_receipt_triage_summary if isinstance(river_receipt_triage_summary, dict) else {}
    if not mode_summary.get("available") and not focus_summary.get("available"):
        return {"available": False, "reason": "benchmark_mode_and_focus_unavailable"}

    lateral = triage_summary.get("lateral_accountability", {}) if isinstance(triage_summary.get("lateral_accountability", {}), dict) else {}
    dominant_issue = lateral.get("dominant_issue") if isinstance(lateral.get("dominant_issue"), str) else None
    weakest_role = lateral.get("weakest_role") if isinstance(lateral.get("weakest_role"), str) else focus_summary.get("triage_weakest_role")
    next_action = focus_summary.get("suggested_next_action") if isinstance(focus_summary.get("suggested_next_action"), str) else triage_summary.get("suggested_next_action")
    drilldowns = {
        "benchmark_river_mode_summary_json": "benchmark_river_mode_summary.json",
        "benchmark_river_receipt_triage_summary_json": "benchmark_river_receipt_triage_summary.json",
        "benchmark_river_primary_focus_summary_json": "benchmark_river_primary_focus_summary.json",
    }
    return {
        "available": True,
        "artifact_role": "diagnostic_only",
        "active_river_benchmark_mode": mode_summary.get("active_river_benchmark_mode") or mode_summary.get("evaluation_basis"),
        "active_river_benchmark_decision_basis": mode_summary.get("active_river_benchmark_decision_basis") or focus_summary.get("primary_decision_signal"),
        "primary_decision_signal": mode_summary.get("primary_decision_signal") or focus_summary.get("primary_decision_signal"),
        "priority_domain": mode_summary.get("priority_domain") or focus_summary.get("priority_domain"),
        "priority_reason": mode_summary.get("priority_reason") or focus_summary.get("priority_reason"),
        "hard_problem_holdout_blind": bool(mode_summary.get("hard_problem_holdout_blind", focus_summary.get("hard_problem_holdout_blind", False))),
        "hard_problem_blind_reason": mode_summary.get("hard_problem_blind_reason") or focus_summary.get("hard_problem_blind_reason"),
        "withheld_support_plan_available": bool(mode_summary.get("withheld_support_plan_available", False)),
        "withheld_support_plan_path": mode_summary.get("withheld_support_plan_path"),
        "withheld_support_cli_flag": mode_summary.get("withheld_support_cli_flag"),
        "river_specific_withheld_support_benchmark_recommended": bool(mode_summary.get("river_specific_withheld_support_benchmark_recommended", focus_summary.get("river_specific_withheld_support_benchmark_recommended", False))),
        "primary_focus": focus_summary.get("primary_focus"),
        "suggested_next_action": next_action,
        "dominant_remaining_issue": dominant_issue,
        "weakest_lateral_role": weakest_role,
        "triage_weakest_role_p95_abs_error_m": focus_summary.get("triage_weakest_role_p95_abs_error_m"),
        "diagnostic_drilldown_receipts": drilldowns,
        "diagnostic_receipt_roles": {name: "diagnostic_only" for name in drilldowns.values()},
    }
def _write_markdown_summary(path: Path, *, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    baseline = overall["baseline"]
    final = overall["final"]
    delta = overall["delta_final_minus_baseline"]
    zone_summary = payload.get("zone_diff_summary", {}) if isinstance(payload.get("zone_diff_summary", {}), dict) else {}
    auth_inv = zone_summary.get("authoritative_locked_invariant", {}) if isinstance(zone_summary.get("authoritative_locked_invariant", {}), dict) else {}
    river_focus = payload.get("river_primary_focus_summary", {}) if isinstance(payload.get("river_primary_focus_summary", {}), dict) else {}
    river_mode = payload.get("river_benchmark_mode_summary", {}) if isinstance(payload.get("river_benchmark_mode_summary", {}), dict) else {}
    active_eval = payload.get("river_active_evaluation_summary", {}) if isinstance(payload.get("river_active_evaluation_summary", {}), dict) else {}
    lines = [
        "# Workflow Benchmark Summary",
        "",
        f"Holdout: `{payload['inputs']['holdout']}`",
        f"Baseline raster: `{payload['inputs']['baseline_raster']}`",
        f"Final raster: `{payload['inputs']['final_raster']}`",
        "",
    ]
    if active_eval.get("available"):
        lines.extend([
            "## Active River Evaluation Summary",
            "",
            f"- Active benchmark mode: {active_eval.get('active_river_benchmark_mode')}",
            f"- Decision basis: {active_eval.get('active_river_benchmark_decision_basis')}",
            f"- Primary decision signal: {active_eval.get('primary_decision_signal')}",
            f"- Primary focus: {active_eval.get('primary_focus')}",
            f"- Suggested next action: {active_eval.get('suggested_next_action')}",
            f"- Dominant remaining issue: {active_eval.get('dominant_remaining_issue')}",
            f"- Weakest lateral role: {active_eval.get('weakest_lateral_role')}",
            f"- Priority domain: {active_eval.get('priority_domain')}",
            f"- Priority reason: {active_eval.get('priority_reason')}",
            f"- Hard-problem holdout blind: {active_eval.get('hard_problem_holdout_blind')}",
            f"- River-specific withheld-support benchmark recommended: {active_eval.get('river_specific_withheld_support_benchmark_recommended')}",
            f"- Withheld-support plan available: {active_eval.get('withheld_support_plan_available')}",
            "",
        ])
        if active_eval.get("hard_problem_holdout_blind"):
            lines.extend([
                "**WARNING: Holdout is blind to unsupported-river performance.**",
                f"Reason: {active_eval.get('hard_problem_blind_reason', 'unknown')}",
                "Use the active river evaluation summary and support-aware receipts below as the primary decision signal for river changes.",
                "",
            ])
    elif river_focus.get("available"):
        lines.extend([
            "## Primary River Evaluation Focus",
            "",
            f"- Primary focus: {river_focus.get('primary_focus')}",
            f"- Suggested next action: {river_focus.get('suggested_next_action')}",
            f"- Evaluation basis: {river_focus.get('evaluation_basis')}",
            f"- Primary decision signal: {river_focus.get('primary_decision_signal')}",
            "",
        ])
    if river_mode.get("available"):
        cover = river_mode.get("holdout_support_coverage", {}) if isinstance(river_mode.get("holdout_support_coverage", {}), dict) else {}
        lines.extend([
            "## River Benchmark Mode",
            "",
            f"- Evaluation basis: {river_mode.get('evaluation_basis')}",
            f"- Primary decision signal: {river_mode.get('primary_decision_signal')}",
            f"- Priority domain: {river_mode.get('priority_domain')}",
            f"- Priority reason: {river_mode.get('priority_reason')}",
            f"- River holdout points: {cover.get('river_point_count')}",
            f"- Unsupported-river holdout points: {cover.get('unsupported_holdout_point_count')}",
            f"- Unsupported-river science-grid cells: {cover.get('unsupported_grid_cell_count')}",
            f"- River-specific withheld-support benchmark recommended: {river_mode.get('river_specific_withheld_support_benchmark_recommended')}",
            f"- Withheld-support plan available: {river_mode.get('withheld_support_plan_available')}",
            "",
        ])
        if river_mode.get("river_specific_withheld_support_benchmark_recommended"):
            lines.extend([
                f"Reason: {river_mode.get('withheld_support_benchmark_reason')}",
                f"Plan path: {river_mode.get('withheld_support_plan_path')}",
                f"CLI flag: {river_mode.get('withheld_support_cli_flag')}",
                "",
            ])
    lines.extend([
        "## Overall Holdout Comparison",
        "",
        f"- Points evaluated: {baseline['n']}",
        f"- Baseline RMSE: {baseline['rmse']:.3f} m",
        f"- Final RMSE: {final['rmse']:.3f} m",
        f"- Delta RMSE (final-baseline): {delta['rmse']:.3f} m",
        f"- Baseline MAE: {baseline['mae']:.3f} m",
        f"- Final MAE: {final['mae']:.3f} m",
        f"- Delta MAE (final-baseline): {delta['mae']:.3f} m",
        f"- Baseline bias: {baseline['bias']:.3f} m",
        f"- Final bias: {final['bias']:.3f} m",
        f"- Baseline R²: {baseline['r2']:.3f}",
        f"- Final R²: {final['r2']:.3f}",
        "",
        "## Raster Difference By Zone",
        "",
    ])
    if auth_inv:
        lines.extend([
            f"- Authoritative locked invariant passes: **{auth_inv.get('passes')}**",
            f"- Authoritative locked changed pixels: {auth_inv.get('changed_pixels')}",
            f"- Authoritative locked max abs diff: {auth_inv.get('max_abs_diff')}",
            "",
        ])
    for zone_name in ("authoritative_locked", "river_zone", "sdb_zone", "estuary_mask", "river_mask", "low_confidence_fill_zone"):
        stats = zone_summary.get(zone_name)
        if isinstance(stats, dict) and stats.get("n", 0) > 0:
            lines.extend([
                f"### {zone_name}",
                f"- Pixels: {stats['n']}",
                f"- Changed pixels: {stats['changed_pixels']} ({100.0 * stats['changed_frac']:.2f}%)",
                f"- Mean diff (final-baseline): {stats['mean_diff']:.3f} m",
                f"- Mean abs diff: {stats['mean_abs_diff']:.3f} m",
                f"- P95 abs diff: {stats['p95_abs_diff']:.3f} m",
                f"- Max abs diff: {stats['max_abs_diff']:.3f} m",
                "",
            ])
    river_science = payload.get("river_scientific_summary", {}) if isinstance(payload.get("river_scientific_summary", {}), dict) else {}
    if river_science.get("available"):
        step = river_science.get("adjacent_step_metrics", {}) if isinstance(river_science.get("adjacent_step_metrics", {}), dict) else {}
        curv = river_science.get("curvature_metrics", {}) if isinstance(river_science.get("curvature_metrics", {}), dict) else {}
        coverage = river_science.get("coverage", {}) if isinstance(river_science.get("coverage", {}), dict) else {}
        coverage_summary = coverage.get("coverage_summary", {}) if isinstance(coverage.get("coverage_summary", {}), dict) else {}
        lines.extend([
            "## River Scientific Diagnostics",
            "",
            f"- Longitudinal profile points sampled: {river_science.get('counts', {}).get('profile_point_count')}",
            f"- Profiles sampled: {river_science.get('counts', {}).get('profile_count')}",
            f"- Final adjacent-step mean abs change: {step.get('final', {}).get('mean_abs', np.nan):.3f} m",
            f"- Baseline adjacent-step mean abs change: {step.get('baseline', {}).get('mean_abs', np.nan):.3f} m",
            f"- Delta adjacent-step p95 abs (final-baseline): {step.get('delta_final_minus_baseline_p95_abs', np.nan):.3f} m",
            f"- Final curvature p95 abs: {curv.get('final', {}).get('p95_abs', np.nan):.3f} m",
            f"- Baseline curvature p95 abs: {curv.get('baseline', {}).get('p95_abs', np.nan):.3f} m",
            f"- Delta curvature p95 abs (final-baseline): {curv.get('delta_final_minus_baseline_p95_abs', np.nan):.3f} m",
            f"- Unsupported longitudinal stations: {coverage_summary.get('unsupported_station_count')}",
            f"- Authoritative bed-supported stations: {coverage_summary.get('authoritative_bed_supported_station_count')}",
            f"- Authoritative bank-margin stations: {coverage_summary.get('authoritative_bank_margin_station_count')}",
            "",
        ])
        by_comp_rough = river_science.get("unsupported_roughness_by_component_class", {}) if isinstance(river_science.get("unsupported_roughness_by_component_class", {}), dict) else {}
        by_comp_curv = river_science.get("unsupported_curvature_by_component_class", {}) if isinstance(river_science.get("unsupported_curvature_by_component_class", {}), dict) else {}
        by_comp_tend = river_science.get("unsupported_section_tendency_by_component_class", {}) if isinstance(river_science.get("unsupported_section_tendency_by_component_class", {}), dict) else {}
        by_comp_relief = river_science.get("unsupported_inner_relief_by_component_class", {}) if isinstance(river_science.get("unsupported_inner_relief_by_component_class", {}), dict) else {}
        by_comp_recon = river_science.get("unsupported_reconciliation_by_component_class", {}) if isinstance(river_science.get("unsupported_reconciliation_by_component_class", {}), dict) else {}
        if by_comp_rough:
            lines.extend(["### Unsupported roughness by component class", ""])
            for key, stats in by_comp_rough.items():
                if not isinstance(stats, dict):
                    continue
                lines.append(f"- {key}: n={stats.get('n')}, final mean abs step={stats.get('final_mean_abs_step', np.nan):.3f} m, delta p95 abs step={stats.get('delta_p95_abs_step', np.nan):.3f} m")
            lines.append("")
        if by_comp_curv:
            lines.extend(["### Unsupported curvature by component class", ""])
            for key, stats in by_comp_curv.items():
                if not isinstance(stats, dict):
                    continue
                lines.append(f"- {key}: n={stats.get('n')}, final p95 abs second diff={stats.get('final_p95_abs_second_diff', np.nan):.3f} m, delta p95 abs second diff={stats.get('delta_p95_abs_second_diff', np.nan):.3f} m")
            lines.append("")
        if by_comp_tend:
            lines.extend(["### Unsupported section tendency by component class", ""])
            for key, stats in by_comp_tend.items():
                if not isinstance(stats, dict):
                    continue
                lines.append(f"- {key}: n={stats.get('n')}, mean depth fraction={stats.get('mean_tendency_depth_fraction', np.nan):.3f}, mean confidence={stats.get('mean_tendency_confidence', np.nan):.3f}, simplified fraction={stats.get('simplified_fraction', np.nan):.3f}")
            lines.append("")
        if by_comp_relief:
            lines.extend(["### Unsupported inner relief by component class", ""])
            for key, stats in by_comp_relief.items():
                if not isinstance(stats, dict):
                    continue
                lines.append(f"- {key}: n={stats.get('n')}, mean inner relief={stats.get('mean_inner_relief_m', np.nan):.3f} m, p95 inner relief={stats.get('p95_inner_relief_m', np.nan):.3f} m")
            lines.append("")
        if by_comp_recon:
            lines.extend(["### Unsupported reconciliation by component class", ""])
            for key, stats in by_comp_recon.items():
                if not isinstance(stats, dict):
                    continue
                lines.append(f"- {key}: n={stats.get('n')}, mean reconciliation weight={stats.get('mean_reconciliation_weight', np.nan):.3f}, p95 reconciliation weight={stats.get('p95_reconciliation_weight', np.nan):.3f}, mean abs reconciliation delta={stats.get('mean_abs_reconciliation_delta_m', np.nan):.3f} m")
            lines.append("")
    hard_river = payload.get("hard_river_benchmark_summary", {}) if isinstance(payload.get("hard_river_benchmark_summary", {}), dict) else {}
    if hard_river.get("available"):
        hard_problem = hard_river.get("hard_problem_group", {}) if isinstance(hard_river.get("hard_problem_group", {}), dict) else {}
        hard_improve = hard_problem.get("improvement", {}) if isinstance(hard_problem.get("improvement", {}), dict) else {}
        hard_delta = hard_problem.get("delta_final_minus_baseline", {}) if isinstance(hard_problem.get("delta_final_minus_baseline", {}), dict) else {}
        lines.extend([
            "## Hard River Problem Benchmark",
            "",
            f"- River points in holdout: {hard_river.get('river_point_count')}",
            f"- Hard-problem points: {hard_problem.get('n')}",
            f"- Hard-problem final better fraction: {hard_improve.get('improved_fraction', np.nan):.3f}" if hard_problem else "- Hard-problem final better fraction: nan",
            f"- Hard-problem delta RMSE (final-baseline): {hard_delta.get('rmse', np.nan):.3f}" if hard_problem else "- Hard-problem delta RMSE (final-baseline): nan",
            "",
        ])
        if hard_river.get("hard_problem_evaluation_blind"):
            lines.extend([
                "**WARNING: Hard-problem evaluation is blind.**",
                f"Reason: {hard_river.get('hard_problem_blind_reason', 'unknown')}",
                "",
            ])
    # ---- Support-distance-binned roughness: the real hard-problem evaluator ----
    binned = river_science.get("support_distance_binned_roughness") if river_science.get("available") else None
    if binned and isinstance(binned, list) and len(binned) > 0:
        lines.extend([
            "## Longitudinal Roughness by Distance to Authoritative Data",
            "",
            "This table shows whether the workflow is making unsupported river reaches",
            "smoother (negative delta = improvement) or rougher (positive delta = regression).",
            "",
            "| Distance Bin | N | Baseline mean|abs step | Final mean|abs step | Delta | Improved? |",
            "|---|---|---|---|---|---|",
        ])
        for b in binned:
            if not b.get("available", False):
                lines.append(f"| {b['bin']} | {b['n']} | — | — | — | — |")
            else:
                lines.append(
                    f"| {b['bin']} | {b['n']} "
                    f"| {b.get('baseline_mean_abs_step', np.nan):.4f} "
                    f"| {b.get('final_mean_abs_step', np.nan):.4f} "
                    f"| {b.get('delta_mean_abs_step', np.nan):+.4f} "
                    f"| {'YES' if b.get('roughness_improved') else 'NO'} |"
                )
        lines.append("")
    support_aware = payload.get("support_aware_validation_summary", {}) if isinstance(payload.get("support_aware_validation_summary", {}), dict) else {}
    if support_aware.get("available"):
        overall_river = support_aware.get("overall_river_holdout", {}) if isinstance(support_aware.get("overall_river_holdout", {}), dict) else {}
        improve = overall_river.get("improvement", {}) if isinstance(overall_river.get("improvement", {}), dict) else {}
        caution = support_aware.get("by_low_support_caution", {}) if isinstance(support_aware.get("by_low_support_caution", {}), dict) else {}
        caution_groups = caution.get("groups", {}) if isinstance(caution.get("groups", {}), dict) else {}
        caution_present = caution_groups.get("1", {}) if "1" in caution_groups else caution_groups.get("present", {})
        role_summary = payload.get("authoritative_role_validation_summary", {}) if isinstance(payload.get("authoritative_role_validation_summary", {}), dict) else {}
        lines.extend([
            "## Support-Aware River Holdout Diagnostics",
            "",
            f"- River holdout points: {support_aware.get('river_holdout_count')}",
            f"- Final better than baseline fraction in river holdout: {improve.get('improved_fraction', np.nan):.3f}",
        ])
        if isinstance(caution_present, dict) and caution_present:
            c_improve = caution_present.get("improvement", {}) if isinstance(caution_present.get("improvement", {}), dict) else {}
            lines.append(f"- Low-support caution improved fraction: {c_improve.get('improved_fraction', np.nan):.3f}")
        if role_summary.get("available"):
            role_counts = role_summary.get("authoritative_role_counts", {}) if isinstance(role_summary.get("authoritative_role_counts", {}), dict) else {}
            if role_counts:
                lines.append(f"- Holdout authoritative role counts: {json.dumps(role_counts, sort_keys=True)}")
        lines.extend([
            "",
        ])
    support_science = payload.get("support_aware_science_summary", {}) if isinstance(payload.get("support_aware_science_summary", {}), dict) else {}
    if support_science.get("available"):
        trans = support_science.get("transition_quality", {}) if isinstance(support_science.get("transition_quality", {}), dict) else {}
        unsup = support_science.get("unsupported_roughness", {}) if isinstance(support_science.get("unsupported_roughness", {}), dict) else {}
        grid = support_science.get("unsupported_science_grid", {}) if isinstance(support_science.get("unsupported_science_grid", {}), dict) else {}
        by_comp = support_science.get("unsupported_by_component_class", {}) if isinstance(support_science.get("unsupported_by_component_class", {}), dict) else {}
        by_adm = support_science.get("unsupported_by_prediction_admissibility", {}) if isinstance(support_science.get("unsupported_by_prediction_admissibility", {}), dict) else {}
        by_tend = support_science.get("unsupported_section_tendency", {}) if isinstance(support_science.get("unsupported_section_tendency", {}), dict) else {}
        by_relief = support_science.get("unsupported_inner_relief", {}) if isinstance(support_science.get("unsupported_inner_relief", {}), dict) else {}
        by_recon = support_science.get("unsupported_reconciliation", {}) if isinstance(support_science.get("unsupported_reconciliation", {}), dict) else {}
        ms_side_comp = support_science.get("unsupported_mainstem_vs_side_component", {}) if isinstance(support_science.get("unsupported_mainstem_vs_side_component", {}), dict) else {}
        ms_side_recon = support_science.get("unsupported_mainstem_vs_side_reconciliation", {}) if isinstance(support_science.get("unsupported_mainstem_vs_side_reconciliation", {}), dict) else {}
        lines.extend([
            "## Support-Aware Science Evaluation",
            "",
            f"- Evaluation mode: {support_science.get('evaluation_mode')}",
            f"- Hard-problem holdout blind: {support_science.get('hard_problem_holdout_blind')}",
            f"- Unsupported roughness station count: {unsup.get('station_count')}",
            f"- Unsupported delta mean|abs step (final-baseline): {unsup.get('delta_mean_abs_step', np.nan):+.3f} m",
            f"- Unsupported delta p95|abs step (final-baseline): {unsup.get('delta_p95_abs_step', np.nan):+.3f} m",
            f"- Transition delta mean|abs step (final-baseline): {trans.get('delta_mean_abs_step', np.nan):+.3f} m",
            f"- Unsupported science-grid cell count: {grid.get('cell_count')}",
            f"- Unsupported science-grid mean abs delta: {grid.get('mean_abs_delta_m', np.nan):.3f} m",
            f"- Unsupported science-grid active fraction: {grid.get('active_fraction', np.nan):.3f}",
            f"- Unsupported component-class point count: {by_comp.get('point_count')}",
            f"- Unsupported component-class mean abs delta: {by_comp.get('mean_abs_delta_m', np.nan):.3f} m",
            f"- Unsupported section-tendency station count: {by_tend.get('station_count')}",
            f"- Unsupported mean tendency depth fraction: {by_tend.get('mean_tendency_depth_fraction', np.nan):.3f}",
            f"- Unsupported mean inner relief: {by_relief.get('mean_inner_relief_m', np.nan):.3f} m",
            f"- Unsupported mean reconciliation weight: {by_recon.get('mean_reconciliation_weight', np.nan):.3f}",
            f"- Unsupported mean abs reconciliation delta: {by_recon.get('mean_abs_reconciliation_delta_m', np.nan):.3f} m",
            f"- Unsupported admissibility groups present: {len(by_adm.get('groups', {}))}",
            "",
        ])
        if ms_side_comp.get("available"):
            lines.extend([
                "### Unsupported mainstem vs unsupported side component",
                "",
                f"- Mainstem mean abs delta: {ms_side_comp.get('mainstem_mean_abs_delta_m', np.nan):.3f} m",
                f"- Side mean abs delta: {ms_side_comp.get('side_mean_abs_delta_m', np.nan):.3f} m",
                f"- Mainstem final mean abs step: {ms_side_comp.get('mainstem_final_mean_abs_step', np.nan):.3f} m",
                f"- Side final mean abs step: {ms_side_comp.get('side_final_mean_abs_step', np.nan):.3f} m",
                f"- Mainstem final p95 abs second diff: {ms_side_comp.get('mainstem_final_p95_abs_second_diff', np.nan):.3f} m",
                f"- Side final p95 abs second diff: {ms_side_comp.get('side_final_p95_abs_second_diff', np.nan):.3f} m",
                f"- Mainstem mean tendency depth fraction: {ms_side_comp.get('mainstem_mean_tendency_depth_fraction', np.nan):.3f}",
                f"- Side mean tendency depth fraction: {ms_side_comp.get('side_mean_tendency_depth_fraction', np.nan):.3f}",
                f"- Mainstem mean inner relief: {ms_side_comp.get('mainstem_mean_inner_relief_m', np.nan):.3f} m",
                f"- Side mean inner relief: {ms_side_comp.get('side_mean_inner_relief_m', np.nan):.3f} m",
                f"- Mainstem is more expressive: {ms_side_comp.get('mainstem_is_more_expressive')}",
                f"- Side is rougher: {ms_side_comp.get('side_is_rougher')}",
                f"- Side is more curved: {ms_side_comp.get('side_is_more_curved')}",
                "",
            ])
        if ms_side_recon.get("available"):
            lines.extend([
                "### Unsupported mainstem vs unsupported side-component reconciliation",
                "",
                f"- Mainstem mean reconciliation weight: {ms_side_recon.get('mainstem_mean_reconciliation_weight', np.nan):.3f}",
                f"- Side mean reconciliation weight: {ms_side_recon.get('side_mean_reconciliation_weight', np.nan):.3f}",
                f"- Mainstem mean abs reconciliation delta: {ms_side_recon.get('mainstem_mean_abs_reconciliation_delta_m', np.nan):.3f} m",
                f"- Side mean abs reconciliation delta: {ms_side_recon.get('side_mean_abs_reconciliation_delta_m', np.nan):.3f} m",
                f"- Mainstem carries more reconciliation: {ms_side_recon.get('mainstem_carries_more_reconciliation')}",
                "",
            ])

    lines.extend([
        "## Sampling",
        "",
        f"- Input rows: {payload['counts']['input_rows']}",
        f"- Finite point rows: {payload['counts']['finite_point_rows']}",
        f"- Points sampled on both rasters: {payload['counts']['points_after_sampling']}",
        f"- Points dropped by raster nodata/non-overlap: {payload['counts']['dropped_after_sampling']}",
        "",
        f"Detailed table: `{payload['table_csv']}`",
        f"Scored holdout points: `{payload.get('scored_points_csv', '')}`",
        f"Zone diff table: `{payload.get('zone_diff_csv', '')}`",
        f"River science summary: `{payload.get('river_scientific_summary_json', '')}`",
        f"Support-aware validation summary: `{payload.get('support_aware_validation_summary_json', '')}`",
        f"Support-aware science summary: `{payload.get('support_aware_science_summary_json', '')}`",
        f"Active river evaluation summary: `{payload.get('river_active_evaluation_summary_json', '')}`",
        f"River benchmark mode summary: `{payload.get('river_benchmark_mode_summary_json', '')}`",
        f"River receipt triage summary: `{payload.get('river_receipt_triage_summary_json', '')}`",
        f"River primary focus summary: `{payload.get('river_primary_focus_summary_json', '')}`",
        f"Authoritative role validation summary: `{payload.get('authoritative_role_validation_summary_json', '')}`",
    ])
    triage = payload.get("river_receipt_triage_summary", {}) if isinstance(payload.get("river_receipt_triage_summary", {}), dict) else {}
    lateral = triage.get("lateral_accountability", {}) if isinstance(triage.get("lateral_accountability", {}), dict) else {}
    if triage.get("available"):
        role_receipts = triage.get("receipts", {}) if isinstance(triage.get("receipts", {}), dict) else {}
        role_receipt = role_receipts.get("role_agreement", {}) if isinstance(role_receipts.get("role_agreement", {}), dict) else {}
        section_receipt = role_receipts.get("section_target_agreement", {}) if isinstance(role_receipts.get("section_target_agreement", {}), dict) else {}
        lines.extend([
            "",
            "## River Receipt Triage",
            "",
            f"- Primary focus: {triage.get('primary_focus')}",
            f"- Suggested next action: {triage.get('suggested_next_action')}",
            f"- Dominant lateral issue: {lateral.get('dominant_issue')}",
            f"- Lateral diagnosis: {lateral.get('reason')}",
            f"- Weakest role: {role_receipt.get('weakest_role')}",
            f"- Weakest-role p95 abs error: {role_receipt.get('weakest_role_p95_abs_error_m', np.nan):.3f} m",
            f"- Thalweg / inner-shape / bank-edge p95 abs error: {role_receipt.get('thalweg_p95_abs_error_m', np.nan):.3f} / {role_receipt.get('inner_shape_p95_abs_error_m', np.nan):.3f} / {role_receipt.get('bank_edge_p95_abs_error_m', np.nan):.3f} m",
            f"- Section-target weakest role class: {section_receipt.get('weakest_role_class')}",
            f"- Section-target weakest-role-class p95 abs error: {section_receipt.get('weakest_role_class_p95_abs_error_m', np.nan):.3f} m",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_existing(*paths: Optional[Path]) -> Optional[Path]:
    for p in paths:
        if p is None:
            continue
        pp = Path(p)
        if pp.exists():
            return pp
    return None




def _resolve_baseline_raster(*, args, cfg, report: dict, out_dir: Path) -> Optional[Path]:
    baseline_override = getattr(args, "benchmark_baseline_raster", None)
    if baseline_override:
        p = Path(str(baseline_override)).resolve()
        return p if p.exists() else None

    final_outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    auth_section = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base"), dict) else {}
    auth_outputs = auth_section.get("outputs", {}) if isinstance(auth_section.get("outputs"), dict) else {}
    auth_auto = report.get("authoritative_base_auto", {}) if isinstance(report.get("authoritative_base_auto"), dict) else {}

    candidates = [
        _resolve_report_path(final_outputs.get("baseline_cudem_interpolation_aligned_to_final"), out_dir),
        _resolve_report_path(final_outputs.get("baseline_cudem_interpolation"), out_dir),
        _resolve_report_path(auth_outputs.get("baseline_cudem_interpolation"), out_dir),
    ]
    auto_native = auth_auto.get("baseline_cudem_interpolation")
    if auto_native:
        auto_path = Path(str(auto_native))
        candidates.append(auto_path if auto_path.exists() else None)

    auth_cfg = getattr(cfg, "authoritative_base", None)
    if auth_cfg:
        try:
            sibling = Path(str(auth_cfg)).with_name("cudem_baseline_interpolation.tif")
        except (TypeError, ValueError, OSError):
            sibling = None
        candidates.append(sibling if sibling and sibling.exists() else None)

    candidates.extend([
        out_dir / "comparison_package" / "baseline_cudem_interpolation_aligned_to_final.tif",
        _resolve_report_path(auth_outputs.get("aligned_authoritative_base"), out_dir),
        _resolve_report_path(auth_outputs.get("authoritative_aligned"), out_dir),
        out_dir / "combined" / "authoritative_base_aligned.tif",
    ])
    return _first_existing(*candidates)

def _resolve_report_path(v: Any, out_dir: Path) -> Optional[Path]:
    if not v:
        return None
    p = Path(str(v))
    if not p.is_absolute():
        p = (out_dir / p).resolve()
    return p if p.exists() else None


def run_workflow_benchmark(*, cfg, args, report: dict, logger, final_path: Optional[Path]) -> Optional[dict[str, Any]]:
    auto_holdout = bool(getattr(args, "benchmark_auto_holdout", False))
    explicit_holdout = getattr(args, "benchmark_holdout", None)
    if not explicit_holdout and not auto_holdout:
        return None

    points_epsg_arg = int(getattr(args, "benchmark_points_epsg", 0) or 0)

    out_dir = Path(cfg.out_dir)
    bench_dir = out_dir / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    final_outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    auth_outputs = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}

    final_override = getattr(args, "benchmark_final_raster", None)

    baseline_raster = _resolve_baseline_raster(args=args, cfg=cfg, report=report, out_dir=out_dir)
    final_raster = _first_existing(
        Path(str(final_override)).resolve() if final_override else None,
        Path(final_path).resolve() if final_path else None,
        _resolve_report_path(final_outputs.get("selected_final_depth"), out_dir),
        out_dir / "combined" / "bathy_combined_depth_conditioned.tif",
    )

    if baseline_raster is None:
        raise FileNotFoundError("Could not resolve benchmark baseline raster")
    if final_raster is None:
        raise FileNotFoundError("Could not resolve benchmark final raster")

    auto_holdout_receipt = None
    generic_candidate_path = None
    if auto_holdout:
        generic_candidate_path = _resolve_auto_holdout_candidate(args=args, report=report, out_dir=out_dir)
        if generic_candidate_path is None:
            raise FileNotFoundError("Could not resolve auto-holdout candidate pool from --benchmark-holdout or authoritative support report")
        holdout_path, points_epsg, inferred_x, inferred_y, inferred_z, auto_holdout_receipt = _build_auto_holdout(
            candidate_path=generic_candidate_path,
            bench_dir=bench_dir,
            baseline_raster=baseline_raster,
            final_raster=final_raster,
            points_epsg_arg=points_epsg_arg if points_epsg_arg > 0 else None,
            fallback_points_epsg=_resolve_candidate_points_epsg(candidate_path=generic_candidate_path, cfg=cfg, report=report, out_dir=out_dir),
            x_col=getattr(args, "benchmark_x_col", None),
            y_col=getattr(args, "benchmark_y_col", None),
            z_col=getattr(args, "benchmark_z_col", None),
            holdout_frac=float(getattr(args, "benchmark_holdout_frac", 0.2) or 0.2),
            holdout_min_points=int(getattr(args, "benchmark_holdout_min_points", 2000) or 2000),
            holdout_seed=int(getattr(args, "benchmark_holdout_seed", 42) or 42),
            logger=logger,
        )
    else:
        holdout_path = Path(str(explicit_holdout)).resolve()
        if not holdout_path.exists():
            raise FileNotFoundError(f"Benchmark holdout not found: {holdout_path}")
        points_epsg = points_epsg_arg
        inferred_x = inferred_y = inferred_z = None

    river_withheld_support_path = None
    river_withheld_support_receipt = None
    auth_section = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    river_auth_section = auth_section.get("river_guidance", {}) if isinstance(auth_section.get("river_guidance", {}), dict) else {}
    river_candidate_path = _resolve_report_path(river_auth_section.get("path"), out_dir)
    if river_candidate_path is not None and river_candidate_path.exists():
        river_withheld_support_path, river_withheld_support_receipt, _, _, _, _ = _build_river_withheld_support_plan(
            candidate_path=river_candidate_path,
            bench_dir=bench_dir,
            baseline_raster=baseline_raster,
            final_raster=final_raster,
            points_epsg_arg=points_epsg_arg if points_epsg_arg > 0 else None,
            fallback_points_epsg=_resolve_candidate_points_epsg(candidate_path=river_candidate_path, cfg=cfg, report=report, out_dir=out_dir),
            x_col=getattr(args, "benchmark_x_col", None),
            y_col=getattr(args, "benchmark_y_col", None),
            z_col=getattr(args, "benchmark_z_col", None),
            holdout_frac=float(getattr(args, "benchmark_holdout_frac", 0.2) or 0.2),
            holdout_min_points=int(getattr(args, "benchmark_holdout_min_points", 2000) or 2000),
            holdout_seed=int(getattr(args, "benchmark_holdout_seed", 42) or 42),
            logger=logger,
        )

    support_class_raster = _first_existing(
        _resolve_report_path(final_outputs.get("support_class"), out_dir),
        out_dir / "combined" / "support_class.tif",
    )
    provenance_raster = _first_existing(
        _resolve_report_path(final_outputs.get("selected_final_provenance"), out_dir),
        out_dir / "combined" / "bathy_combined_depth_conditioned_provenance.tif",
    )
    river_mask_raster = _first_existing(
        _resolve_report_path(river_outputs.get("river_channel_mask"), out_dir),
        out_dir / "derived_cache" / str(getattr(cfg, "run_id", "")) / "river" / "work" / "river_channel_mask.tif",
    )
    estuary_mask_raster = _first_existing(
        _resolve_report_path(river_outputs.get("estuary_clip_mask"), out_dir),
        out_dir / "derived_cache" / str(getattr(cfg, "run_id", "")) / "river" / "work" / "estuary_clip_mask.tif",
    )
    longitudinal_profile_points = _first_existing(
        _resolve_report_path(river_outputs.get("longitudinal_profile_points"), out_dir),
        out_dir / "river" / "river_longitudinal_profile_points.gpkg",
    )
    longitudinal_profile_coverage = _first_existing(
        _resolve_report_path(river_outputs.get("longitudinal_profile_coverage"), out_dir),
        out_dir / "river" / "river_longitudinal_profile_coverage.csv",
    )
    longitudinal_profile_summary = _first_existing(
        _resolve_report_path(river_outputs.get("longitudinal_profile_summary"), out_dir),
        out_dir / "river" / "river_longitudinal_profile_summary.json",
    )
    station_targets_path = _first_existing(
        _resolve_report_path(river_outputs.get("station_targets"), out_dir),
        out_dir / "river" / "river_station_targets.csv",
    )
    reach_attributes_csv = _first_existing(
        _resolve_report_path(river_outputs.get("reach_attributes"), out_dir),
        out_dir / "river" / "river_reach_attributes.csv",
    )
    reach_attributes_summary_json = _first_existing(
        _resolve_report_path(river_outputs.get("reach_attributes_summary"), out_dir),
        out_dir / "river" / "river_reach_attributes_summary.json",
    )
    prediction_support_confidence_raster = _first_existing(
        _resolve_report_path(river_outputs.get("channel_surface_prediction_support_confidence"), out_dir),
        out_dir / "river" / "river_channel_surface_prediction_support_confidence.tif",
    )
    measured_anchor_fraction_raster = _first_existing(
        _resolve_report_path(river_outputs.get("channel_surface_measured_anchor_fraction"), out_dir),
        _resolve_report_path(river_outputs.get("channel_surface_prediction_measured_anchor_fraction"), out_dir),
        out_dir / "river" / "river_channel_surface_measured_anchor_fraction.tif",
        out_dir / "river" / "river_channel_surface_prediction_measured_anchor_fraction.tif",
    )
    structure_only_fraction_raster = _first_existing(
        _resolve_report_path(river_outputs.get("channel_surface_structure_only_fraction"), out_dir),
        _resolve_report_path(river_outputs.get("channel_surface_prediction_structure_only_fraction"), out_dir),
        out_dir / "river" / "river_channel_surface_structure_only_fraction.tif",
        out_dir / "river" / "river_channel_surface_prediction_structure_only_fraction.tif",
    )
    low_support_caution_raster = _first_existing(
        _resolve_report_path(river_outputs.get("channel_surface_low_support_caution"), out_dir),
        out_dir / "river" / "river_channel_surface_low_support_caution.tif",
    )
    prediction_admissibility_raster = _first_existing(
        _resolve_report_path(river_outputs.get("channel_surface_prediction_admissibility"), out_dir),
        out_dir / "river" / "river_channel_surface_prediction_admissibility.tif",
    )
    authoritative_role_code_raster = _resolve_authoritative_role_artifact(
        report=report, cfg=cfg, out_dir=out_dir, key="role_code_raster", cfg_attr="river_authoritative_role_code_raster"
    )
    authoritative_role_confidence_raster = _resolve_authoritative_role_artifact(
        report=report, cfg=cfg, out_dir=out_dir, key="role_confidence_raster", cfg_attr="river_authoritative_role_confidence_raster"
    )
    authoritative_distance_to_bank_raster = _resolve_authoritative_role_artifact(
        report=report, cfg=cfg, out_dir=out_dir, key="distance_to_bank_raster", cfg_attr="river_authoritative_distance_to_bank_raster"
    )
    authoritative_normalized_channel_position_raster = _resolve_authoritative_role_artifact(
        report=report, cfg=cfg, out_dir=out_dir, key="normalized_channel_position_raster", cfg_attr="river_authoritative_normalized_channel_position_raster"
    )

    df = _read_table(holdout_path)
    x_name = inferred_x or getattr(args, "benchmark_x_col", None) or _infer_col(df, ["x", "X", "easting", "lon", "longitude"], "x")
    y_name = inferred_y or getattr(args, "benchmark_y_col", None) or _infer_col(df, ["y", "Y", "northing", "lat", "latitude"], "y")
    z_name = inferred_z or getattr(args, "benchmark_z_col", None) or _infer_col(df, ["z", "Z", "depth_m", "z_m", "depth", "elevation_m", "elev_m", "bed_z_m", "elevation", "bed_elevation_m"], "z/depth")

    input_rows = int(len(df))
    x = pd.to_numeric(df[x_name], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(df[y_name], errors="coerce").to_numpy(dtype="float64")
    obs = pd.to_numeric(df[z_name], errors="coerce").to_numpy(dtype="float64")

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(obs)
    finite_point_rows = int(np.count_nonzero(valid))
    x = x[valid]
    y = y[valid]
    obs = obs[valid]

    points_epsg = points_epsg if auto_holdout else (points_epsg_arg if points_epsg_arg > 0 else _infer_points_epsg(
        x=x, y=y, x_name=x_name, y_name=y_name, baseline_raster=baseline_raster, final_raster=final_raster
    ))
    logger.info(f"[BENCHMARK] Using point CRS EPSG:{points_epsg} for holdout sampling")

    baseline = _sample_raster(baseline_raster, x, y, src_epsg=points_epsg)
    final = _sample_raster(final_raster, x, y, src_epsg=points_epsg)
    finite = np.isfinite(obs) & np.isfinite(baseline) & np.isfinite(final)
    sampled_rows = int(np.count_nonzero(finite))
    dropped_after_sampling = int(obs.size - sampled_rows)
    x = x[finite]
    y = y[finite]
    obs = obs[finite]
    baseline = baseline[finite]
    final = final[finite]
    if sampled_rows == 0:
        raise ValueError("Benchmark produced zero overlapping finite samples between holdout points and both rasters")

    support_class = _sample_raster(support_class_raster, x, y, src_epsg=points_epsg) if support_class_raster else None
    provenance = _sample_raster(provenance_raster, x, y, src_epsg=points_epsg) if provenance_raster else None
    river_mask = _sample_mask(river_mask_raster, x, y, src_epsg=points_epsg) if river_mask_raster else None
    estuary_mask = _sample_mask(estuary_mask_raster, x, y, src_epsg=points_epsg) if estuary_mask_raster else None
    prediction_support_confidence = _sample_raster(prediction_support_confidence_raster, x, y, src_epsg=points_epsg) if prediction_support_confidence_raster else None
    prediction_measured_anchor_fraction = _sample_raster(measured_anchor_fraction_raster, x, y, src_epsg=points_epsg) if measured_anchor_fraction_raster else None
    prediction_structure_only_fraction = _sample_raster(structure_only_fraction_raster, x, y, src_epsg=points_epsg) if structure_only_fraction_raster else None
    prediction_low_support_caution = _sample_mask(low_support_caution_raster, x, y, src_epsg=points_epsg) if low_support_caution_raster else None
    prediction_admissibility = _sample_mask(prediction_admissibility_raster, x, y, src_epsg=points_epsg) if prediction_admissibility_raster else None
    authoritative_role_code = _sample_raster(authoritative_role_code_raster, x, y, src_epsg=points_epsg) if authoritative_role_code_raster else None
    authoritative_role_confidence = _sample_raster(authoritative_role_confidence_raster, x, y, src_epsg=points_epsg) if authoritative_role_confidence_raster else None
    authoritative_distance_to_bank = _sample_raster(authoritative_distance_to_bank_raster, x, y, src_epsg=points_epsg) if authoritative_distance_to_bank_raster else None
    authoritative_normalized_channel_position = _sample_raster(authoritative_normalized_channel_position_raster, x, y, src_epsg=points_epsg) if authoritative_normalized_channel_position_raster else None

    scored_df = pd.DataFrame({
        "x": x,
        "y": y,
        "obs": obs,
        "baseline": baseline,
        "final": final,
        "baseline_error": baseline - obs,
        "final_error": final - obs,
        "baseline_abs_error": np.abs(baseline - obs),
        "final_abs_error": np.abs(final - obs),
        "final_better_than_baseline": (np.abs(final - obs) < np.abs(baseline - obs)),
    })
    if support_class is not None:
        support_class_int = pd.Series(support_class).round().astype("Int64")
        scored_df["support_class"] = support_class_int
        scored_df["support_class_label"] = support_class_int.map({int(k): v for k, v in SUPPORT_CLASS_CODE_TO_NAME.items()})
    if provenance is not None:
        scored_df["provenance"] = pd.Series(provenance).round().astype("Int64")
    if river_mask is not None:
        scored_df["river_mask"] = river_mask.astype(bool)
    if estuary_mask is not None:
        scored_df["estuary_mask"] = estuary_mask.astype(bool)
    if prediction_support_confidence is not None:
        scored_df["prediction_support_confidence"] = prediction_support_confidence
    if prediction_measured_anchor_fraction is not None:
        scored_df["prediction_measured_anchor_fraction"] = prediction_measured_anchor_fraction
    if prediction_structure_only_fraction is not None:
        scored_df["prediction_structure_only_fraction"] = prediction_structure_only_fraction
    if prediction_low_support_caution is not None:
        scored_df["prediction_low_support_caution"] = prediction_low_support_caution.astype(int)
    if prediction_admissibility is not None:
        scored_df["prediction_admissibility"] = prediction_admissibility.astype(int)
    if authoritative_role_code is not None:
        role_code_int = pd.Series(authoritative_role_code).round().astype("Int64")
        scored_df["authoritative_role_code"] = role_code_int
        scored_df["authoritative_role_label"] = role_code_int.map(lambda v: code_to_role(int(v)) if pd.notna(v) else None)
        scored_df["authoritative_bed_support_present"] = scored_df["authoritative_role_label"].isin([ROLE_BED_CORE, ROLE_BED_INNER]).astype(int)
        scored_df["authoritative_bank_margin_present"] = scored_df["authoritative_role_label"].eq(ROLE_BANK_MARGIN).astype(int)
    if authoritative_role_confidence is not None:
        scored_df["authoritative_role_confidence"] = authoritative_role_confidence
    if authoritative_distance_to_bank is not None:
        scored_df["authoritative_distance_to_bank_m"] = authoritative_distance_to_bank
    if authoritative_normalized_channel_position is not None:
        scored_df["authoritative_normalized_channel_position"] = authoritative_normalized_channel_position

    scored_points_csv = bench_dir / "benchmark_holdout_scored.csv"
    scored_df.to_csv(scored_points_csv, index=False)
    support_aware_summary = _compute_support_aware_validation_summary(scored_df)
    support_aware_summary_path = bench_dir / "benchmark_support_aware_validation_summary.json"
    support_aware_summary_path.write_text(json.dumps(support_aware_summary, indent=2), encoding="utf-8")
    authoritative_role_validation_summary = _compute_authoritative_role_validation_summary(scored_df)
    authoritative_role_validation_summary_path = bench_dir / "benchmark_authoritative_role_validation_summary.json"
    authoritative_role_validation_summary_path.write_text(json.dumps(authoritative_role_validation_summary, indent=2), encoding="utf-8")
    hard_river_summary = _compute_hard_river_benchmark(scored_df)
    hard_river_summary_path = bench_dir / "benchmark_hard_river_summary.json"
    hard_river_summary_path.write_text(json.dumps(hard_river_summary, indent=2), encoding="utf-8")
    hard_river_table_path = bench_dir / "benchmark_hard_river_groups.csv"
    pd.DataFrame(hard_river_summary.get("groups", []) if isinstance(hard_river_summary, dict) else []).to_csv(hard_river_table_path, index=False)

    rows: list[dict[str, Any]] = []
    overall_baseline = _metrics(obs, baseline)
    overall_final = _metrics(obs, final)
    rows.append({"group": "overall", "model": "baseline", **overall_baseline})
    rows.append({"group": "overall", "model": "final", **overall_final})

    def _append_group(name: str, mask: np.ndarray) -> None:
        rows.append({"group": name, "model": "baseline", **_metrics(obs[mask], baseline[mask])})
        rows.append({"group": name, "model": "final", **_metrics(obs[mask], final[mask])})

    if river_mask is not None:
        _append_group("river_mask", river_mask)
    if estuary_mask is not None:
        _append_group("estuary_mask", estuary_mask)
    if support_class is not None:
        for cls in sorted(np.unique(support_class[np.isfinite(support_class)])):
            _append_group(f"support_class_{int(cls)}", np.isclose(support_class, cls))
    if provenance is not None:
        for cls in sorted(np.unique(provenance[np.isfinite(provenance)])):
            _append_group(f"provenance_{int(cls)}", np.isclose(provenance, cls))

    table_csv = bench_dir / "benchmark_table.csv"
    summary_json = bench_dir / "benchmark_summary.json"
    zone_diff_csv = bench_dir / "benchmark_zone_diff_table.csv"
    pd.DataFrame(rows).to_csv(table_csv, index=False)

    zone_diff_summary = _compute_zone_diff_summary(
        baseline_raster=baseline_raster,
        final_raster=final_raster,
        support_class_raster=support_class_raster,
        river_mask_raster=river_mask_raster,
        estuary_mask_raster=estuary_mask_raster,
    )
    _write_zone_diff_csv(zone_diff_csv, zone_diff_summary)

    river_scientific_summary, river_science_table_path = _compute_river_scientific_summary(
        baseline_raster=baseline_raster,
        final_raster=final_raster,
        profile_points_path=longitudinal_profile_points,
        coverage_csv_path=longitudinal_profile_coverage,
        coverage_summary_path=longitudinal_profile_summary,
        station_targets_path=station_targets_path,
        bench_dir=bench_dir,
    )
    river_science_summary_path = bench_dir / "benchmark_river_science_summary.json"
    river_science_summary_path.write_text(json.dumps(river_scientific_summary, indent=2), encoding="utf-8")
    centerline_agreement_path = bench_dir / "benchmark_centerline_agreement_summary.json"
    centerline_agreement_path.write_text(json.dumps((river_scientific_summary or {}).get("centerline_agreement", {"available": False}), indent=2), encoding="utf-8")

    science_evaluation_summary = _build_science_evaluation_grid(
        baseline_raster=baseline_raster,
        final_raster=final_raster,
        support_class_raster=support_class_raster,
        river_mask_raster=river_mask_raster,
        prediction_admissibility_raster=prediction_admissibility_raster,
        bench_dir=bench_dir,
        grid_step=3,
        logger=logger,
    )
    science_evaluation_path = bench_dir / "benchmark_science_evaluation_summary.json"
    science_evaluation_path.write_text(json.dumps(science_evaluation_summary, indent=2), encoding="utf-8")
    support_aware_science_summary = _compute_support_aware_science_summary(
        river_scientific_summary=river_scientific_summary,
        science_evaluation_summary=science_evaluation_summary,
        zone_diff_summary=zone_diff_summary,
        hard_river_summary=hard_river_summary,
    )
    support_aware_science_path = bench_dir / "benchmark_support_aware_science_summary.json"
    support_aware_science_path.write_text(json.dumps(support_aware_science_summary, indent=2), encoding="utf-8")
    river_receipt_triage_summary = _compute_river_receipt_triage_summary(
        report=report,
        out_dir=out_dir,
        river_scientific_summary=river_scientific_summary,
    )
    river_receipt_triage_path = bench_dir / "benchmark_river_receipt_triage_summary.json"
    river_receipt_triage_path.write_text(json.dumps(river_receipt_triage_summary, indent=2), encoding="utf-8")
    river_benchmark_mode_summary = _build_river_benchmark_mode_summary(
        support_aware_science=support_aware_science_summary,
        river_receipt_triage=river_receipt_triage_summary,
        hard_river_summary=hard_river_summary,
        auto_holdout_receipt=auto_holdout_receipt,
        withheld_support_receipt=river_withheld_support_receipt,
    )
    river_benchmark_mode_path = bench_dir / "benchmark_river_mode_summary.json"
    river_benchmark_mode_path.write_text(json.dumps(river_benchmark_mode_summary, indent=2), encoding="utf-8")
    river_primary_focus_summary = _build_primary_river_benchmark_focus(
        support_aware_science=support_aware_science_summary,
        river_receipt_triage=river_receipt_triage_summary,
        hard_river_summary=hard_river_summary,
        benchmark_mode_summary=river_benchmark_mode_summary,
    )
    river_primary_focus_path = bench_dir / "benchmark_river_primary_focus_summary.json"
    river_primary_focus_path.write_text(json.dumps(river_primary_focus_summary, indent=2), encoding="utf-8")
    river_active_evaluation_summary = _build_active_river_evaluation_summary(
        benchmark_mode_summary=river_benchmark_mode_summary,
        river_primary_focus_summary=river_primary_focus_summary,
        river_receipt_triage_summary=river_receipt_triage_summary,
    )
    river_active_evaluation_path = bench_dir / "benchmark_river_active_evaluation_summary.json"
    river_active_evaluation_path.write_text(json.dumps(river_active_evaluation_summary, indent=2), encoding="utf-8")

    markdown_path = bench_dir / "benchmark_summary.md"
    payload = {
        "inputs": {
            "holdout": str(holdout_path),
            "auto_holdout": bool(auto_holdout),
            "baseline_raster": str(baseline_raster),
            "final_raster": str(final_raster),
            "support_class_raster": str(support_class_raster) if support_class_raster else None,
            "provenance_raster": str(provenance_raster) if provenance_raster else None,
            "river_mask_raster": str(river_mask_raster) if river_mask_raster else None,
            "estuary_mask_raster": str(estuary_mask_raster) if estuary_mask_raster else None,
            "authoritative_role_code_raster": str(authoritative_role_code_raster) if authoritative_role_code_raster else None,
            "authoritative_role_confidence_raster": str(authoritative_role_confidence_raster) if authoritative_role_confidence_raster else None,
            "authoritative_distance_to_bank_raster": str(authoritative_distance_to_bank_raster) if authoritative_distance_to_bank_raster else None,
            "authoritative_normalized_channel_position_raster": str(authoritative_normalized_channel_position_raster) if authoritative_normalized_channel_position_raster else None,
            "points_epsg": points_epsg,
            "x_col": x_name,
            "y_col": y_name,
            "z_col": z_name,
            "river_withheld_support_plan": str(river_withheld_support_path) if river_withheld_support_path else None,
        },
        "counts": {
            "input_rows": input_rows,
            "finite_point_rows": finite_point_rows,
            "points_after_sampling": int(obs.size),
            "dropped_after_sampling": dropped_after_sampling,
        },
        "overall": {
            "baseline": overall_baseline,
            "final": overall_final,
            "delta_final_minus_baseline": _delta_metrics(overall_final, overall_baseline),
            "improvement": _improvement_summary(obs, baseline, final),
        },
        "table_csv": str(table_csv),
        "scored_points_csv": str(scored_points_csv),
        "zone_diff_csv": str(zone_diff_csv),
        "zone_diff_summary": zone_diff_summary,
        "river_scientific_summary": river_scientific_summary,
        "river_scientific_summary_json": str(river_science_summary_path),
        "river_longitudinal_metrics_csv": str(river_science_table_path) if river_science_table_path else None,
        "river_reach_attributes_csv": str(reach_attributes_csv) if reach_attributes_csv else None,
        "river_reach_attributes_summary_json": str(reach_attributes_summary_json) if reach_attributes_summary_json else None,
        "support_aware_validation_summary": support_aware_summary,
        "support_aware_validation_summary_json": str(support_aware_summary_path),
        "authoritative_role_validation_summary": authoritative_role_validation_summary,
        "authoritative_role_validation_summary_json": str(authoritative_role_validation_summary_path),
        "hard_river_benchmark_summary": hard_river_summary,
        "hard_river_benchmark_summary_json": str(hard_river_summary_path),
        "hard_river_benchmark_groups_csv": str(hard_river_table_path),
        "science_evaluation_summary": science_evaluation_summary,
        "science_evaluation_summary_json": str(science_evaluation_path),
        "support_aware_science_summary": support_aware_science_summary,
        "support_aware_science_summary_json": str(support_aware_science_path),
        "river_receipt_triage_summary": river_receipt_triage_summary,
        "river_receipt_triage_summary_json": str(river_receipt_triage_path),
        "river_benchmark_mode_summary": river_benchmark_mode_summary,
        "river_benchmark_mode_summary_json": str(river_benchmark_mode_path),
        "river_active_evaluation_summary": river_active_evaluation_summary,
        "river_active_evaluation_summary_json": str(river_active_evaluation_path),
        "river_withheld_support_receipt": river_withheld_support_receipt,
        "river_withheld_support_receipt_json": str(bench_dir / "river_withheld_support_receipt.json"),
        "river_withheld_support_points": str(river_withheld_support_path) if river_withheld_support_path else None,
        "river_primary_focus_summary": river_primary_focus_summary,
        "river_primary_focus_summary_json": str(river_primary_focus_path),
        "markdown_summary": str(markdown_path),
        "auto_holdout_receipt": auto_holdout_receipt,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown_summary(markdown_path, payload=payload)
    auth_inv = zone_diff_summary.get("authoritative_locked_invariant", {}) if isinstance(zone_diff_summary, dict) else {}
    if isinstance(auth_inv, dict) and auth_inv.get("n", 0) > 0 and not bool(auth_inv.get("passes", False)):
        raise RuntimeError(
            f"Benchmark invariant failed: final raster differs from baseline on authoritative_locked cells "
            f"(changed_pixels={auth_inv.get('changed_pixels')}, max_abs_diff={auth_inv.get('max_abs_diff')})"
        )
    report.setdefault("benchmark", {})["workflow_benchmark"] = payload
    report.setdefault("outputs", {})["benchmark_summary_json"] = str(summary_json)
    report.setdefault("outputs", {})["benchmark_table_csv"] = str(table_csv)
    report.setdefault("outputs", {})["benchmark_scored_points_csv"] = str(scored_points_csv)
    report.setdefault("outputs", {})["benchmark_summary_md"] = str(markdown_path)
    report.setdefault("outputs", {})["benchmark_zone_diff_csv"] = str(zone_diff_csv)
    report.setdefault("outputs", {})["benchmark_river_science_summary_json"] = str(river_science_summary_path)
    report.setdefault("outputs", {})["benchmark_support_aware_validation_summary_json"] = str(support_aware_summary_path)
    report.setdefault("outputs", {})["benchmark_support_aware_science_summary_json"] = str(support_aware_science_path)
    report.setdefault("outputs", {})["benchmark_river_receipt_triage_summary_json"] = str(river_receipt_triage_path)
    report.setdefault("outputs", {})["benchmark_river_mode_summary_json"] = str(river_benchmark_mode_path)
    report.setdefault("outputs", {})["benchmark_river_active_evaluation_summary_json"] = str(river_active_evaluation_path)
    report.setdefault("outputs", {})["benchmark_river_primary_focus_summary_json"] = str(river_primary_focus_path)
    report.setdefault("outputs", {})["benchmark_river_withheld_support_receipt_json"] = str(bench_dir / "river_withheld_support_receipt.json")
    if river_withheld_support_path is not None:
        report.setdefault("outputs", {})["benchmark_river_withheld_support_points"] = str(river_withheld_support_path)
    report.setdefault("outputs", {})["benchmark_hard_river_summary_json"] = str(hard_river_summary_path)
    report.setdefault("outputs", {})["benchmark_hard_river_groups_csv"] = str(hard_river_table_path)
    report.setdefault("outputs", {})["benchmark_authoritative_role_validation_summary_json"] = str(authoritative_role_validation_summary_path)
    report.setdefault("outputs", {})["benchmark_science_evaluation_summary_json"] = str(science_evaluation_path)
    if science_evaluation_summary.get("grid_path"):
        report.setdefault("outputs", {})["benchmark_science_evaluation_grid_csv"] = str(science_evaluation_summary["grid_path"])
    if river_science_table_path is not None:
        report.setdefault("outputs", {})["benchmark_river_longitudinal_metrics_csv"] = str(river_science_table_path)
    if reach_attributes_csv is not None:
        report.setdefault("outputs", {})["river_reach_attributes_csv"] = str(reach_attributes_csv)
    if reach_attributes_summary_json is not None:
        report.setdefault("outputs", {})["river_reach_attributes_summary_json"] = str(reach_attributes_summary_json)
    if auto_holdout_receipt is not None:
        report.setdefault("outputs", {})["benchmark_auto_holdout_receipt_json"] = str(bench_dir / "auto_holdout_receipt.json")
        report.setdefault("outputs", {})["benchmark_auto_holdout_points"] = str(bench_dir / "auto_holdout_points.csv")
    if isinstance(hard_river_summary, dict) and hard_river_summary.get("available"):
        hard_group = hard_river_summary.get("hard_problem_group") or {}
        logger.info("[BENCHMARK][RIVER] hard_problem_n=%s final_better_frac=%s final_delta_rmse=%s", hard_group.get("n"), ((hard_group.get("improvement") or {}).get("fraction_points_final_better")), ((hard_group.get("delta_final_minus_baseline") or {}).get("rmse")))
    logger.info("[BENCHMARK] Workflow benchmark written: %s ; %s ; %s", summary_json, table_csv, markdown_path)
    return payload
