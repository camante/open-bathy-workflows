from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from river_anchor_policy import build_anchor_policy_table
from river_section_tendency import (
    build_section_from_thalweg,
    classify_section_tendency_family,
    compute_tendency_confidence,
    compute_tendency_depth_fraction,
)

SCHEMA_VERSION = 3

_CANONICAL_COLUMNS = [
    "component_id",
    "component_support_class",
    "station_m",
    "profile_inside_fluvial_monotone_domain",
    "station_support_regime",
    "station_authoritative_bed_support_class",
    "station_authoritative_bed_support_distance_m",
    "active_core_support_z_m",
    "active_core_support_source",
    "active_core_support_bank_reference_m",
    "active_core_support_bank_offset_m",
    "active_core_support_model_elevation_m",
    "generalized_longitudinal_bed_base_elevation_m",
    "generalized_longitudinal_bed_reconciled_elevation_m",
    "generalized_longitudinal_bed_local_auth_taper_m",
    "generalized_longitudinal_bed_local_auth_reconciliation_weight",
    "authoritative_reconciliation_delta_m",
    "authoritative_reconciliation_weight",
    "authoritative_reconciliation_confidence",
    "authoritative_reconciliation_support_distance_m",
    "authoritative_reconciliation_exact_anchor",
    "left_bank_fit_z_m",
    "right_bank_fit_z_m",
    "bank_pair_fit_z_m",
    "bank_profile_active_source",
    "authoritative_anchor_present",
    "authoritative_anchor_source",
    "authoritative_anchor_curve_present",
    "true_measured_xs_qualified",
    "hard_anchor_present",
    "target_left_bank_z_m",
    "target_right_bank_z_m",
    "target_effective_channel_width_m",
    "target_thalweg_z_m",
    "target_left_inner_z_m",
    "target_right_inner_z_m",
    "section_tendency_family",
    "section_tendency_depth_m",
    "section_tendency_inner_relief_m",
    "section_tendency_confidence",
    "section_tendency_source",
    "xs_residual_allowed",
    "xs_realism_allowed",
    "rebuild_allowed",
    "post_rebuild_monotone_required",
    "target_source_class",
    "target_policy_reason",
    "anchor_class",
    "anchor_exact",
    "anchor_locks_core",
    "anchor_blocks_rebuild",
]


def canonical_station_target_columns() -> list[str]:
    return list(_CANONICAL_COLUMNS)


def _float_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float32", name=name)
    return pd.to_numeric(frame[name], errors="coerce").astype("float32")


def _bool_series(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool, name=name)
    return frame[name].fillna(default).astype(bool)


def _str_series(frame: pd.DataFrame, name: str, default: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object", name=name)
    vals = frame[name].fillna(default).astype(str).str.strip()
    vals = vals.where(vals.ne(""), default)
    return pd.Series(vals, index=frame.index, dtype="object", name=name)


def _first_float_series(frame: pd.DataFrame, *names: str) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float32")
    for name in names:
        vals = _float_series(frame, name)
        out = out.where(np.isfinite(out), vals)
    return out.astype("float32")


def _estimate_effective_channel_width_m(frame: pd.DataFrame, left_bank: pd.Series, right_bank: pd.Series) -> pd.Series:
    auth_dist = _float_series(frame, "authoritative_distance_to_bank_m")
    width = (auth_dist * np.float32(2.0)).astype("float32")
    width = width.where(np.isfinite(width) & width.gt(0.0))
    fallback = pd.Series(np.float32(20.0), index=frame.index, dtype="float32")
    fallback = fallback.where(~_str_series(frame, "channel_support_class", "unsupported").eq("bank_margin_only"), np.float32(28.0))
    fallback = fallback.where(~_bool_series(frame, "authoritative_bed_support_present"), np.float32(14.0))
    width = width.where(np.isfinite(width), fallback)
    width = width.clip(lower=np.float32(4.0), upper=np.float32(150.0))
    return width.astype("float32")


def _build_tendency_targets(
    frame: pd.DataFrame,
    thalweg: pd.Series,
    left_bank: pd.Series,
    right_bank: pd.Series,
    bed_support_class: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    width = _estimate_effective_channel_width_m(frame, left_bank, right_bank)
    component_class = _str_series(frame, "component_support_class", "unknown")
    reconciliation_conf = _float_series(frame, "authoritative_reconciliation_confidence")
    support_distance = _float_series(frame, "station_authoritative_bed_support_distance_m")
    if support_distance.isna().all():
        support_distance = _float_series(frame, "authoritative_reconciliation_support_distance_m")
    families: list[str] = []
    depth_m: list[float] = []
    relief_m: list[float] = []
    confidence: list[float] = []
    left_inner_vals: list[float] = []
    right_inner_vals: list[float] = []
    for idx in frame.index:
        support_cls = str(bed_support_class.loc[idx] or "none")
        width_here = float(width.loc[idx]) if np.isfinite(width.loc[idx]) else float("nan")
        fam = classify_section_tendency_family(width_here, support_cls)
        comp_cls = str(component_class.loc[idx] or "unknown")
        recon_conf = float(reconciliation_conf.loc[idx]) if np.isfinite(reconciliation_conf.loc[idx]) else float("nan")
        support_dist = float(support_distance.loc[idx]) if np.isfinite(support_distance.loc[idx]) else float("nan")
        frac = compute_tendency_depth_fraction(width_here, fam, support_cls, comp_cls, recon_conf, support_dist)
        thal = float(thalweg.loc[idx]) if np.isfinite(thalweg.loc[idx]) else float("nan")
        left_cap = float(left_bank.loc[idx]) if np.isfinite(left_bank.loc[idx]) else float("nan")
        right_cap = float(right_bank.loc[idx]) if np.isfinite(right_bank.loc[idx]) else float("nan")
        left_inner, right_inner, relief = build_section_from_thalweg(thal, width_here, fam, frac, bank_caps=(left_cap, right_cap))
        families.append(fam)
        depth_m.append(float(relief) if np.isfinite(relief) else float("nan"))
        relief_m.append(float(relief) if np.isfinite(relief) else float("nan"))
        conf = compute_tendency_confidence(width_here, fam, support_cls, comp_cls, recon_conf, support_dist)
        confidence.append(float(conf))
        left_inner_vals.append(float(left_inner) if np.isfinite(left_inner) else float("nan"))
        right_inner_vals.append(float(right_inner) if np.isfinite(right_inner) else float("nan"))
    return (
        width.astype("float32"),
        pd.Series(families, index=frame.index, dtype="object"),
        pd.Series(depth_m, index=frame.index, dtype="float32"),
        pd.Series(relief_m, index=frame.index, dtype="float32"),
        pd.Series(confidence, index=frame.index, dtype="float32"),
        pd.Series(left_inner_vals, index=frame.index, dtype="float32"),
        pd.Series(right_inner_vals, index=frame.index, dtype="float32"),
    )


def build_station_target_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    target = pd.DataFrame(index=work.index)
    target["component_id"] = _str_series(work, "component_id", "main")
    target["component_support_class"] = _str_series(work, "component_support_class", "unknown")
    target["station_m"] = _float_series(work, "station_m")

    support_class = _str_series(work, "channel_support_class", "unsupported")
    support_regime = _str_series(work, "graph_solver_support_class", "unsupported")
    bed_support_class = pd.Series(
        np.where(
            _bool_series(work, "authoritative_bed_support_present"),
            "bed_supported",
            np.where(_bool_series(work, "authoritative_bank_margin_present"), "bank_margin_only", "none"),
        ),
        index=work.index,
        dtype="object",
    )
    bed_support_distance = _float_series(work, "station_authoritative_bed_support_distance_m")
    if bed_support_distance.isna().all():
        bed_support_distance = _float_series(work, "profile_authoritative_bed_support_distance_m")

    target["profile_inside_fluvial_monotone_domain"] = _bool_series(work, "profile_inside_fluvial_monotone_domain", True)
    target["station_support_regime"] = support_regime
    target["station_authoritative_bed_support_class"] = bed_support_class
    target["station_authoritative_bed_support_distance_m"] = bed_support_distance

    active_core = _first_float_series(work, "generalized_longitudinal_bed_reconciled_elevation_m", "generalized_longitudinal_bed_elevation_m", "generalized_longitudinal_bed_z_m")
    active_core = active_core.where(np.isfinite(active_core), _float_series(work, "active_core_support_z_m"))
    active_source = _str_series(work, "generalized_longitudinal_bed_source", "active_core_support")
    active_source = pd.Series(
        np.where(np.isfinite(active_core), active_source.astype(str).to_numpy(dtype=object), "missing"),
        index=work.index,
        dtype="object",
    )
    bank_ref = _float_series(work, "bank_pair_fit_z_m")
    bank_ref = bank_ref.where(np.isfinite(bank_ref), _float_series(work, "resolved_stage_control_z_m"))
    bank_ref = bank_ref.where(np.isfinite(bank_ref), _float_series(work, "bank_low_stage_z_m"))
    bank_ref_source = pd.Series(
        np.where(
            np.isfinite(_float_series(work, "bank_pair_fit_z_m")),
            "bank_pair_fit",
            np.where(
                np.isfinite(_float_series(work, "resolved_stage_control_z_m")),
                _str_series(work, "resolved_stage_control_source", "resolved_stage_control"),
                np.where(np.isfinite(_float_series(work, "bank_low_stage_z_m")), "bank_low_stage", "missing"),
            ),
        ),
        index=work.index,
        dtype="object",
    )
    active_model = _first_float_series(work, "generalized_longitudinal_bed_reconciled_elevation_m", "generalized_longitudinal_bed_elevation_m", "generalized_longitudinal_bed_z_m")
    active_model = active_model.where(np.isfinite(active_model), _float_series(work, "active_core_support_model_elevation_m"))
    active_model = active_model.where(np.isfinite(active_model), active_core)
    bank_offset = pd.Series(bank_ref - active_model, index=work.index, dtype="float32")
    generalized_base = _first_float_series(work, "generalized_longitudinal_bed_base_elevation_m", "generalized_longitudinal_bed_elevation_m", "generalized_longitudinal_bed_z_m")
    generalized_reconciled = _first_float_series(work, "generalized_longitudinal_bed_reconciled_elevation_m", "generalized_longitudinal_bed_elevation_m", "generalized_longitudinal_bed_z_m")
    reconciliation_delta = _first_float_series(work, "authoritative_reconciliation_delta_m", "generalized_longitudinal_bed_local_auth_taper_m")
    reconciliation_weight = _first_float_series(work, "authoritative_reconciliation_weight", "generalized_longitudinal_bed_local_auth_reconciliation_weight")
    reconciliation_confidence = _float_series(work, "authoritative_reconciliation_confidence")
    reconciliation_support_distance = _float_series(work, "authoritative_reconciliation_support_distance_m")
    reconciliation_exact_anchor = _bool_series(work, "authoritative_reconciliation_exact_anchor")
    local_auth_taper = _float_series(work, "generalized_longitudinal_bed_local_auth_taper_m")
    local_auth_reconciliation_weight = _float_series(work, "generalized_longitudinal_bed_local_auth_reconciliation_weight")

    target["active_core_support_z_m"] = active_core
    target["active_core_support_source"] = active_source
    target["active_core_support_bank_reference_m"] = bank_ref.astype("float32")
    target["active_core_support_bank_offset_m"] = bank_offset.astype("float32")
    target["active_core_support_model_elevation_m"] = active_model.astype("float32")
    target["generalized_longitudinal_bed_base_elevation_m"] = generalized_base.astype("float32")
    target["generalized_longitudinal_bed_reconciled_elevation_m"] = generalized_reconciled.astype("float32")
    target["generalized_longitudinal_bed_local_auth_taper_m"] = local_auth_taper.astype("float32")
    target["generalized_longitudinal_bed_local_auth_reconciliation_weight"] = local_auth_reconciliation_weight.astype("float32")
    target["authoritative_reconciliation_delta_m"] = reconciliation_delta.astype("float32")
    target["authoritative_reconciliation_weight"] = reconciliation_weight.astype("float32")
    target["authoritative_reconciliation_confidence"] = reconciliation_confidence.astype("float32")
    target["authoritative_reconciliation_support_distance_m"] = reconciliation_support_distance.astype("float32")
    target["authoritative_reconciliation_exact_anchor"] = reconciliation_exact_anchor.astype(bool)

    left_bank = _float_series(work, "left_bank_fit_z_m")
    right_bank = _float_series(work, "right_bank_fit_z_m")
    if left_bank.isna().all():
        left_bank = bank_ref.copy()
    else:
        left_bank = left_bank.where(np.isfinite(left_bank), bank_ref)
    if right_bank.isna().all():
        right_bank = bank_ref.copy()
    else:
        right_bank = right_bank.where(np.isfinite(right_bank), bank_ref)

    target["left_bank_fit_z_m"] = left_bank.astype("float32")
    target["right_bank_fit_z_m"] = right_bank.astype("float32")
    target["bank_pair_fit_z_m"] = _float_series(work, "bank_pair_fit_z_m")
    target["bank_profile_active_source"] = bank_ref_source

    anchor_present = _bool_series(work, "authoritative_anchor_present")
    anchor_curve_present = _bool_series(work, "authoritative_anchor_curve_present")
    hard_anchor = anchor_present | _bool_series(work, "true_measured_xs_qualified")
    anchor_source = _str_series(work, "authoritative_anchor_source", "none")
    anchor_source = pd.Series(
        np.where(
            hard_anchor & anchor_source.eq("none").to_numpy(),
            np.where(anchor_present.to_numpy(), "authoritative_anchor", "true_measured_xs"),
            anchor_source,
        ),
        index=work.index,
        dtype="object",
    )

    target["authoritative_anchor_present"] = anchor_present
    target["authoritative_anchor_source"] = anchor_source
    target["authoritative_anchor_curve_present"] = anchor_curve_present
    target["true_measured_xs_qualified"] = _bool_series(work, "true_measured_xs_qualified")
    target["hard_anchor_present"] = hard_anchor

    thalweg = generalized_reconciled.where(np.isfinite(generalized_reconciled), active_core)
    thalweg = thalweg.where(np.isfinite(thalweg), _float_series(work, "backbone_z_m"))
    thalweg = thalweg.where(np.isfinite(thalweg), _float_series(work, "longitudinal_profile_z_m"))
    thalweg = thalweg.where(np.isfinite(thalweg), _float_series(work, "authoritative_hard_bed_z_m"))
    local_reconciled = np.isfinite(reconciliation_delta.to_numpy(dtype=float)) & (np.abs(reconciliation_delta.to_numpy(dtype=float)) > 1.0e-6)
    (
        effective_width,
        tendency_family,
        tendency_depth_m,
        tendency_inner_relief_m,
        tendency_confidence,
        left_inner,
        right_inner,
    ) = _build_tendency_targets(work, thalweg.astype("float32"), left_bank.astype("float32"), right_bank.astype("float32"), bed_support_class)

    target["target_left_bank_z_m"] = left_bank.astype("float32")
    target["target_right_bank_z_m"] = right_bank.astype("float32")
    target["target_effective_channel_width_m"] = effective_width.astype("float32")
    target["target_thalweg_z_m"] = thalweg.astype("float32")
    target["target_left_inner_z_m"] = left_inner.astype("float32")
    target["target_right_inner_z_m"] = right_inner.astype("float32")
    target["section_tendency_family"] = tendency_family.astype("object")
    target["section_tendency_depth_m"] = tendency_depth_m.astype("float32")
    target["section_tendency_inner_relief_m"] = tendency_inner_relief_m.astype("float32")
    target["section_tendency_confidence"] = tendency_confidence.astype("float32")
    weak_component = target["component_support_class"].isin(["unsupported_side_component", "tiny_detached_component"]).to_numpy()
    unsupported_mainstem = target["component_support_class"].eq("unsupported_mainstem").to_numpy()
    target["section_tendency_source"] = pd.Series(
        np.where(
            bed_support_class.eq("bed_supported").to_numpy(),
            "authoritative_section_derived",
            np.where(
                weak_component & local_reconciled,
                "weak_component_simplified_tendency_local_authoritative_reconciliation",
                np.where(
                    weak_component,
                    "weak_component_simplified_tendency",
                    np.where(
                        unsupported_mainstem & local_reconciled,
                        "mainstem_calibrated_tendency_local_authoritative_reconciliation",
                        np.where(
                            unsupported_mainstem,
                            "mainstem_calibrated_tendency",
                            np.where(
                                local_reconciled,
                                "support_class_width_depth_tendency_local_authoritative_reconciliation",
                                "support_class_width_depth_tendency",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        index=work.index,
        dtype="object",
    )

    residual_allowed = (~_str_series(work, "residual_shape_mode", "none").isin(["suppressed_longitudinal_core", "suppressed_bank_margin", "none"]))
    residual_allowed &= (~bed_support_distance.ge(500.0).fillna(False))
    residual_allowed &= (~bed_support_class.eq("bank_margin_only"))
    realism_allowed = residual_allowed | _bool_series(work, "true_measured_xs_qualified")
    rebuild_allowed = (~hard_anchor) | bed_support_class.eq("bank_margin_only")
    post_monotone_required = target["profile_inside_fluvial_monotone_domain"] & (~hard_anchor)

    target["xs_residual_allowed"] = residual_allowed.astype(bool)
    target["xs_realism_allowed"] = realism_allowed.astype(bool)
    target["rebuild_allowed"] = rebuild_allowed.astype(bool)
    target["post_rebuild_monotone_required"] = post_monotone_required.astype(bool)

    target_source = np.where(
        hard_anchor.to_numpy(),
        "hard_anchor_controlled",
        np.where(
            local_reconciled,
            "generalized_longitudinal_section_local_authoritative_reconciliation",
            np.where(
                np.isfinite(active_core).to_numpy(),
                "generalized_longitudinal_section",
                "legacy_backbone_fallback",
            ),
        ),
    )
    reason = np.where(
        hard_anchor.to_numpy(),
        "hard_anchor_present",
        np.where(
            local_reconciled,
            "local_authoritative_reconciliation_applied",
            np.where(
                bed_support_class.eq("bank_margin_only").to_numpy(),
                "bank_margin_boundary_with_longitudinal_core",
                np.where(
                    _bool_series(work, "true_measured_xs_qualified").to_numpy(),
                    "true_measured_xs_qualified",
                    np.where(
                        bed_support_distance.ge(500.0).fillna(False).to_numpy(),
                        "far_from_bed_support_longitudinal_target",
                        "generalized_longitudinal_station_target",
                    ),
                ),
            ),
        ),
    )
    target["target_source_class"] = pd.Series(target_source, index=work.index, dtype="object")
    target["target_policy_reason"] = pd.Series(reason, index=work.index, dtype="object")

    anchor_policy_df, anchor_policy_summary = build_anchor_policy_table(target)
    for col in ("anchor_class", "anchor_exact", "anchor_locks_core", "anchor_blocks_rebuild"):
        target[col] = anchor_policy_df[col].to_numpy()

    target = target[canonical_station_target_columns()].copy()
    target = target.sort_values(["component_id", "station_m"]).reset_index(drop=True)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "station_count": int(len(target)),
        "component_count": int(target["component_id"].nunique(dropna=True)),
        "target_source_class_counts": {str(k): int(v) for k, v in target["target_source_class"].astype(str).value_counts(dropna=False).items()},
        "station_support_regime_counts": {str(k): int(v) for k, v in target["station_support_regime"].astype(str).value_counts(dropna=False).items()},
        "xs_residual_allowed_count": int(target["xs_residual_allowed"].sum()),
        "xs_realism_allowed_count": int(target["xs_realism_allowed"].sum()),
        "rebuild_allowed_count": int(target["rebuild_allowed"].sum()),
        "post_rebuild_monotone_required_count": int(target["post_rebuild_monotone_required"].sum()),
        "hard_anchor_present_count": int(target["hard_anchor_present"].sum()),
        "anchor_class_counts": dict(anchor_policy_summary.get("anchor_class_counts", {})),
        "anchor_locks_core_count": int(anchor_policy_summary.get("anchor_locks_core_count", 0)),
        "anchor_blocks_rebuild_count": int(anchor_policy_summary.get("anchor_blocks_rebuild_count", 0)),
        "bank_pair_fit_target_count": int(np.count_nonzero(np.isfinite(pd.to_numeric(target["bank_pair_fit_z_m"], errors="coerce").to_numpy(dtype=float)))),
        "active_core_target_count": int(np.count_nonzero(np.isfinite(pd.to_numeric(target["active_core_support_z_m"], errors="coerce").to_numpy(dtype=float)))),
        "section_tendency_family_counts": {str(k): int(v) for k, v in target["section_tendency_family"].astype(str).value_counts(dropna=False).items()},
        "section_tendency_source_counts": {str(k): int(v) for k, v in target["section_tendency_source"].astype(str).value_counts(dropna=False).items()},
    }
    return target, summary


def write_station_target_artifacts(*, river_dir: str | Path, target_df: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, str]:
    river_dir = Path(river_dir)
    target_path = river_dir / "river_station_targets.csv"
    summary_path = river_dir / "river_station_targets_summary.json"
    target_df.to_csv(target_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "station_targets": str(target_path),
        "station_targets_summary": str(summary_path),
    }
