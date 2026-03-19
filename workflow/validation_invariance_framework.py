from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from support_classes import SupportClass
from validation_runner import (
    compute_provenance_class_metrics,
    compute_support_class_metrics,
    run_ablation_matrix,
)


def _parse_case_spec(spec: str) -> tuple[str, str]:
    text = str(spec or "").strip()
    if not text or "=" not in text:
        raise ValueError(f"Validation case must be NAME=PATH, got: {spec!r}")
    name, path = text.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Validation case must be NAME=PATH, got: {spec!r}")
    return name, path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_float_raster(path: str | Path, *, reference: Optional[str | Path] = None) -> np.ndarray:
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("rasterio is required for validation/invariance framework") from exc

    src_path = Path(path)
    if not src_path.exists():
        raise FileNotFoundError(f"Raster not found: {src_path}")
    if reference is None:
        with rasterio.open(src_path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr[np.isclose(arr, np.float32(nodata))] = np.nan
            return arr

    ref_path = Path(reference)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {ref_path}")

    with rasterio.open(ref_path) as ref_ds, rasterio.open(src_path) as src_ds:
        out = np.full((ref_ds.height, ref_ds.width), np.nan, dtype=np.float32)
        if (
            src_ds.crs == ref_ds.crs
            and src_ds.transform == ref_ds.transform
            and src_ds.width == ref_ds.width
            and src_ds.height == ref_ds.height
        ):
            out = src_ds.read(1).astype(np.float32)
        else:
            reproject(
                source=rasterio.band(src_ds, 1),
                destination=out,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                src_nodata=src_ds.nodata,
                dst_transform=ref_ds.transform,
                dst_crs=ref_ds.crs,
                dst_nodata=np.float32(np.nan),
                resampling=Resampling.bilinear,
            )
        if src_ds.nodata is not None:
            out[np.isclose(out, np.float32(src_ds.nodata))] = np.nan
        return out


def _load_class_raster(path: str | Path, *, reference: str | Path) -> np.ndarray:
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("rasterio is required for validation/invariance framework") from exc

    src_path = Path(path)
    ref_path = Path(reference)
    if not src_path.exists():
        raise FileNotFoundError(f"Class raster not found: {src_path}")
    with rasterio.open(ref_path) as ref_ds, rasterio.open(src_path) as src_ds:
        out = np.zeros((ref_ds.height, ref_ds.width), dtype=np.int32)
        if (
            src_ds.crs == ref_ds.crs
            and src_ds.transform == ref_ds.transform
            and src_ds.width == ref_ds.width
            and src_ds.height == ref_ds.height
        ):
            out = src_ds.read(1).astype(np.int32)
        else:
            reproject(
                source=rasterio.band(src_ds, 1),
                destination=out,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                src_nodata=src_ds.nodata,
                dst_transform=ref_ds.transform,
                dst_crs=ref_ds.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
        return out


def evaluate_authoritative_lock_invariant(*, pred: np.ndarray, authoritative: np.ndarray, support_class: Optional[np.ndarray] = None, tolerance: float = 1e-6) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    authoritative = np.asarray(authoritative, dtype=np.float32)
    if pred.shape != authoritative.shape:
        raise ValueError("pred and authoritative must have matching shapes")
    if support_class is not None:
        support_class = np.asarray(support_class)
        if support_class.shape != pred.shape:
            raise ValueError("support_class must match pred shape")
        domain = np.isfinite(pred) & np.isfinite(authoritative) & (support_class == int(SupportClass.AUTHORITATIVE_LOCKED))
    else:
        domain = np.isfinite(pred) & np.isfinite(authoritative)
    diff = pred - authoritative
    vals = np.abs(diff[domain])
    checked = int(np.count_nonzero(domain))
    if checked <= 0:
        return {
            "checked": 0,
            "tolerance": float(tolerance),
            "ok": None,
            "max_abs": None,
            "mean_abs": None,
        }
    max_abs = float(np.nanmax(vals))
    mean_abs = float(np.nanmean(vals))
    return {
        "checked": checked,
        "tolerance": float(tolerance),
        "ok": bool(max_abs <= float(tolerance)),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


def evaluate_guidance_non_degradation(*, ablation: Mapping[str, Any], baseline_case: str, target_case: str, rmse_tolerance: float = 0.0) -> Dict[str, Any]:
    cases = dict((ablation or {}).get("cases") or {})
    baseline = cases.get(str(baseline_case))
    target = cases.get(str(target_case))
    if not baseline or not target:
        return {
            "baseline_case": str(baseline_case),
            "target_case": str(target_case),
            "evaluated": False,
            "ok": None,
            "reason": "missing baseline or target case",
            "families": {},
        }
    families = ["guidance_conditioned", "scaffold_inferred", "low_confidence_continuous_fill"]
    out_families: Dict[str, Any] = {}
    failures = []
    for family in families:
        b = (baseline.get("support_metrics", {}).get("by_family", {}) or {}).get(family)
        t = (target.get("support_metrics", {}).get("by_family", {}) or {}).get(family)
        if not b or not t:
            continue
        b_rmse = b.get("rmse")
        t_rmse = t.get("rmse")
        if b_rmse is None or t_rmse is None:
            continue
        delta = float(t_rmse) - float(b_rmse)
        ok = bool(delta <= float(rmse_tolerance))
        out_families[family] = {
            "baseline_rmse": float(b_rmse),
            "target_rmse": float(t_rmse),
            "delta_rmse": delta,
            "tolerance": float(rmse_tolerance),
            "ok": ok,
        }
        if not ok:
            failures.append({"family": family, "delta_rmse": delta, "tolerance": float(rmse_tolerance)})
    return {
        "baseline_case": str(baseline_case),
        "target_case": str(target_case),
        "evaluated": True,
        "ok": len(failures) == 0 if out_families else None,
        "families": out_families,
        "failures": failures,
    }


def _gather_case_paths(final_outputs: Mapping[str, Any], *, case_specs=None, case_manifest=None) -> Dict[str, str]:
    cases: Dict[str, str] = {}
    selected = final_outputs.get("selected_final_depth") or final_outputs.get("final_depth_native") or final_outputs.get("final_depth_user")
    if selected:
        cases["selected_final"] = str(selected)
    baseline = final_outputs.get("baseline_cudem_interpolation") or final_outputs.get("baseline_cudem_interpolation_aligned_to_final")
    if baseline:
        cases["baseline_cudem_interpolation"] = str(baseline)
    if case_manifest:
        manifest_payload = _read_json(Path(case_manifest))
        if not isinstance(manifest_payload, dict):
            raise ValueError("validation case manifest must be a JSON object mapping case names to raster paths")
        for name, path in manifest_payload.items():
            if path:
                cases[str(name)] = str(path)
    for spec in list(case_specs or []):
        name, path = _parse_case_spec(spec)
        cases[name] = path
    return cases


def run_validation_invariance_framework(*,
    final_outputs_manifest: str | Path,
    overlap_identity_evaluation: Optional[Mapping[str, Any]] = None,
    validation_truth: Optional[str | Path] = None,
    case_specs=None,
    case_manifest: Optional[str | Path] = None,
    guidance_baseline_case: str = "baseline_cudem_interpolation",
    guidance_target_case: str = "selected_final",
    require_guidance_non_degradation: bool = False,
    guidance_rmse_tolerance: float = 0.0,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    final_outputs = _read_json(Path(final_outputs_manifest))
    selected_final = final_outputs.get("selected_final_depth") or final_outputs.get("final_depth_native") or final_outputs.get("final_depth_user")
    invariant_final = final_outputs.get("selected_final_native") or final_outputs.get("final_depth_native") or selected_final
    support_path = final_outputs.get("support_class")
    provenance_path = final_outputs.get("final_provenance_native") or final_outputs.get("selected_final_provenance")
    authoritative_base = final_outputs.get("conditioned_authoritative_base") or final_outputs.get("authoritative_base")
    if not selected_final:
        raise RuntimeError("validation/invariance framework requires selected_final_depth in final_outputs manifest")

    pred = _load_float_raster(invariant_final)
    support = _load_class_raster(support_path, reference=invariant_final) if support_path else None
    authoritative = _load_float_raster(authoritative_base, reference=invariant_final) if authoritative_base else None
    provenance = _load_class_raster(provenance_path, reference=invariant_final) if provenance_path else None

    authoritative_invariant = evaluate_authoritative_lock_invariant(
        pred=pred,
        authoritative=authoritative if authoritative is not None else np.full_like(pred, np.nan),
        support_class=support,
        tolerance=tolerance,
    ) if authoritative_base else {"checked": 0, "ok": None, "reason": "no authoritative_base available"}

    payload: Dict[str, Any] = {
        "selected_final_depth": str(selected_final),
        "invariant_evaluation_depth": str(invariant_final),
        "support_class": str(support_path) if support_path else None,
        "final_provenance_native": str(provenance_path) if provenance_path else None,
        "authoritative_base": str(authoritative_base) if authoritative_base else None,
        "authoritative_lock_invariant": authoritative_invariant,
        "aoi_stability_invariant": dict(overlap_identity_evaluation or {}),
        "validation_truth": str(validation_truth) if validation_truth else None,
    }

    if validation_truth:
        truth = _load_float_raster(validation_truth, reference=invariant_final)
        if support is not None:
            payload["selected_final_support_metrics"] = compute_support_class_metrics(pred=pred, truth=truth, support_class=support)
        if provenance is not None:
            payload["selected_final_provenance_metrics"] = compute_provenance_class_metrics(pred=pred, truth=truth, provenance_class=provenance)
        if support is not None and provenance is not None:
            cases = _gather_case_paths(final_outputs, case_specs=case_specs, case_manifest=case_manifest)
            loaded_cases = {name: _load_float_raster(path, reference=invariant_final) for name, path in cases.items()}
            payload["ablation_cases"] = {name: path for name, path in cases.items()}
            payload["ablation_results"] = run_ablation_matrix(
                truth=truth,
                support_class=support,
                provenance_class=provenance,
                cases=loaded_cases,
            )
            payload["guidance_non_degradation"] = evaluate_guidance_non_degradation(
                ablation=payload["ablation_results"],
                baseline_case=guidance_baseline_case,
                target_case=guidance_target_case,
                rmse_tolerance=guidance_rmse_tolerance,
            )

    hard_failures = []
    if authoritative_invariant.get("ok") is False:
        hard_failures.append(
            f"authoritative lock invariant failed: max_abs {authoritative_invariant.get('max_abs')} exceeds tolerance {authoritative_invariant.get('tolerance')}"
        )
    aoi_eval = payload.get("aoi_stability_invariant") or {}
    if aoi_eval.get("all_ok") is False:
        hard_failures.append("AOI-stability invariant failed: overlap identity evaluation reported failures")
    gd = payload.get("guidance_non_degradation") or {}
    if require_guidance_non_degradation and gd.get("ok") is False:
        hard_failures.append(
            f"guidance non-degradation failed for {gd.get('target_case')} relative to {gd.get('baseline_case')}"
        )
    payload["hard_failures"] = hard_failures
    payload["all_hard_invariants_ok"] = len(hard_failures) == 0
    return payload


__all__ = [
    "evaluate_authoritative_lock_invariant",
    "evaluate_guidance_non_degradation",
    "run_validation_invariance_framework",
]
