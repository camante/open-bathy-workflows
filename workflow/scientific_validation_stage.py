from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from validation_invariance_framework import run_validation_invariance_framework


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_delta_block(base: Optional[Mapping[str, Any]], target: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(base, Mapping) or not isinstance(target, Mapping):
        return None
    out: Dict[str, Any] = {
        "baseline_count": int(base.get("count", 0) or 0),
        "target_count": int(target.get("count", 0) or 0),
    }
    for key in ("mean_error", "mae", "rmse"):
        b = base.get(key)
        t = target.get(key)
        out[f"baseline_{key}"] = float(b) if b is not None else None
        out[f"target_{key}"] = float(t) if t is not None else None
        out[f"delta_{key}"] = (float(t) - float(b)) if b is not None and t is not None else None
    return out



def _compare_metric_groups(*, baseline: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "overall": _metric_delta_block(baseline.get("overall") if isinstance(baseline, Mapping) else None,
                                       target.get("overall") if isinstance(target, Mapping) else None),
        "by_family": {},
        "by_class": {},
    }
    baseline_by_family = baseline.get("by_family", {}) if isinstance(baseline, Mapping) and isinstance(baseline.get("by_family", {}), Mapping) else {}
    target_by_family = target.get("by_family", {}) if isinstance(target, Mapping) and isinstance(target.get("by_family", {}), Mapping) else {}
    for key in sorted(set(baseline_by_family) | set(target_by_family)):
        block = _metric_delta_block(baseline_by_family.get(key), target_by_family.get(key))
        if block:
            result["by_family"][str(key)] = block

    baseline_by_class = baseline.get("by_class", {}) if isinstance(baseline, Mapping) and isinstance(baseline.get("by_class", {}), Mapping) else {}
    target_by_class = target.get("by_class", {}) if isinstance(target, Mapping) and isinstance(target.get("by_class", {}), Mapping) else {}
    for key in sorted(set(baseline_by_class) | set(target_by_class)):
        block = _metric_delta_block(baseline_by_class.get(key), target_by_class.get(key))
        if block:
            label = None
            if isinstance(target_by_class.get(key), Mapping):
                label = target_by_class.get(key, {}).get("label")
            if label is None and isinstance(baseline_by_class.get(key), Mapping):
                label = baseline_by_class.get(key, {}).get("label")
            if label is not None:
                block["label"] = label
            result["by_class"][str(key)] = block
    return result



def _summarize_seam_overlap(overlap_identity_evaluation: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    payload = dict(overlap_identity_evaluation or {})
    failures = payload.get("failures", []) if isinstance(payload.get("failures", []), list) else []
    return {
        "evaluated": bool(payload),
        "all_ok": payload.get("all_ok"),
        "checked": int(payload.get("checked", 0) or 0),
        "failures": failures,
        "failure_count": len(failures),
    }



def run_scientific_validation_stage(*,
    final_outputs_manifest: str | Path,
    overlap_identity_evaluation: Optional[Mapping[str, Any]] = None,
    validation_truth: Optional[str | Path] = None,
    case_specs=None,
    case_manifest: Optional[str | Path] = None,
    guidance_baseline_case: str = "baseline_cudem_interpolation",
    guidance_target_case: str = "selected_final",
    guidance_rmse_tolerance: float = 0.0,
) -> Dict[str, Any]:
    invariance = run_validation_invariance_framework(
        final_outputs_manifest=final_outputs_manifest,
        overlap_identity_evaluation=overlap_identity_evaluation,
        validation_truth=validation_truth,
        case_specs=case_specs,
        case_manifest=case_manifest,
        guidance_baseline_case=guidance_baseline_case,
        guidance_target_case=guidance_target_case,
        guidance_rmse_tolerance=guidance_rmse_tolerance,
        require_guidance_non_degradation=False,
    )
    payload: Dict[str, Any] = {
        "final_outputs_manifest": str(final_outputs_manifest),
        "validation_truth": str(validation_truth) if validation_truth else None,
        "support_class_metrics": invariance.get("selected_final_support_metrics"),
        "provenance_class_metrics": invariance.get("selected_final_provenance_metrics"),
        "guidance_non_degradation": invariance.get("guidance_non_degradation"),
        "seam_and_nested_aoi": _summarize_seam_overlap(overlap_identity_evaluation),
        "authoritative_lock_invariant": invariance.get("authoritative_lock_invariant"),
        "ablation_cases": invariance.get("ablation_cases"),
    }
    ablation = invariance.get("ablation_results", {}) if isinstance(invariance.get("ablation_results", {}), Mapping) else {}
    cases = ablation.get("cases", {}) if isinstance(ablation.get("cases", {}), Mapping) else {}
    baseline = cases.get(str(guidance_baseline_case)) if isinstance(cases.get(str(guidance_baseline_case)), Mapping) else None
    target = cases.get(str(guidance_target_case)) if isinstance(cases.get(str(guidance_target_case)), Mapping) else None
    if baseline and target:
        payload["baseline_comparison"] = {
            "baseline_case": str(guidance_baseline_case),
            "target_case": str(guidance_target_case),
            "support_metrics": _compare_metric_groups(
                baseline=baseline.get("support_metrics", {}) if isinstance(baseline.get("support_metrics", {}), Mapping) else {},
                target=target.get("support_metrics", {}) if isinstance(target.get("support_metrics", {}), Mapping) else {},
            ),
            "provenance_metrics": _compare_metric_groups(
                baseline=baseline.get("provenance_metrics", {}) if isinstance(baseline.get("provenance_metrics", {}), Mapping) else {},
                target=target.get("provenance_metrics", {}) if isinstance(target.get("provenance_metrics", {}), Mapping) else {},
            ),
        }
    else:
        payload["baseline_comparison"] = None
    summary_flags = {
        "has_support_class_metrics": bool(payload.get("support_class_metrics")),
        "has_provenance_class_metrics": bool(payload.get("provenance_class_metrics")),
        "has_baseline_comparison": payload.get("baseline_comparison") is not None,
        "has_seam_and_nested_aoi_summary": payload.get("seam_and_nested_aoi", {}).get("evaluated") is True,
        "guidance_non_degradation_ok": (payload.get("guidance_non_degradation") or {}).get("ok"),
        "authoritative_lock_ok": (payload.get("authoritative_lock_invariant") or {}).get("ok"),
    }
    payload["summary_flags"] = summary_flags
    return payload



def write_scientific_validation_summary(*,
    out_path: str | Path,
    final_outputs_manifest: str | Path,
    overlap_identity_evaluation: Optional[Mapping[str, Any]] = None,
    validation_truth: Optional[str | Path] = None,
    case_specs=None,
    case_manifest: Optional[str | Path] = None,
    guidance_baseline_case: str = "baseline_cudem_interpolation",
    guidance_target_case: str = "selected_final",
    guidance_rmse_tolerance: float = 0.0,
) -> Dict[str, Any]:
    payload = run_scientific_validation_stage(
        final_outputs_manifest=final_outputs_manifest,
        overlap_identity_evaluation=overlap_identity_evaluation,
        validation_truth=validation_truth,
        case_specs=case_specs,
        case_manifest=case_manifest,
        guidance_baseline_case=guidance_baseline_case,
        guidance_target_case=guidance_target_case,
        guidance_rmse_tolerance=guidance_rmse_tolerance,
    )
    out_path = Path(out_path)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


__all__ = [
    "run_scientific_validation_stage",
    "write_scientific_validation_summary",
]
