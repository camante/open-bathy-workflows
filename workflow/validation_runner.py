from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from provenance_schema import PROVENANCE_CLASS_CODE_TO_NAME
from support_classes import SUPPORT_CLASS_CODE_TO_NAME


def _finite_domain(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    return np.isfinite(pred) & np.isfinite(truth)


def _metric_block(errors: np.ndarray) -> Dict[str, Any]:
    if errors.size <= 0:
        return {"count": 0, "mean_error": None, "mae": None, "rmse": None}
    return {
        "count": int(errors.size),
        "mean_error": float(np.nanmean(errors)),
        "mae": float(np.nanmean(np.abs(errors))),
        "rmse": float(np.sqrt(np.nanmean(np.square(errors)))),
    }


def compute_support_class_metrics(*, pred: np.ndarray, truth: np.ndarray, support_class: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    support_class = np.asarray(support_class)
    if pred.shape != truth.shape or pred.shape != support_class.shape:
        raise ValueError("pred, truth, and support_class must have matching shapes")
    domain = _finite_domain(pred, truth)
    errors = pred - truth
    out: Dict[str, Any] = {"overall": _metric_block(errors[domain]), "by_class": {}, "by_family": {}}
    for code, label in SUPPORT_CLASS_CODE_TO_NAME.items():
        mask = domain & (support_class == int(code))
        block = _metric_block(errors[mask])
        if block["count"] <= 0:
            continue
        out["by_class"][str(int(code))] = {"label": label, **block}
        family = label if "guidance_conditioned" not in label else "guidance_conditioned"
        fam = out["by_family"].setdefault(family, [])
        fam.append(errors[mask])
    out["by_family"] = {fam: _metric_block(np.concatenate(vals) if vals else np.array([], dtype=np.float32)) for fam, vals in out["by_family"].items()}
    return out


def compute_provenance_class_metrics(*, pred: np.ndarray, truth: np.ndarray, provenance_class: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    provenance_class = np.asarray(provenance_class)
    if pred.shape != truth.shape or pred.shape != provenance_class.shape:
        raise ValueError("pred, truth, and provenance_class must have matching shapes")
    domain = _finite_domain(pred, truth)
    errors = pred - truth
    out: Dict[str, Any] = {"overall": _metric_block(errors[domain]), "by_class": {}}
    for code, label in PROVENANCE_CLASS_CODE_TO_NAME.items():
        mask = domain & (provenance_class == int(code))
        block = _metric_block(errors[mask])
        if block["count"] <= 0:
            continue
        out["by_class"][str(int(code))] = {"label": label, **block}
    return out


def run_ablation_matrix(*, truth: np.ndarray, support_class: np.ndarray, provenance_class: np.ndarray, cases: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float32)
    support_class = np.asarray(support_class)
    provenance_class = np.asarray(provenance_class)
    result: Dict[str, Any] = {"cases": {}}
    for name, pred in cases.items():
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != truth.shape:
            raise ValueError(f"Case {name} shape {pred.shape} does not match truth shape {truth.shape}")
        result["cases"][str(name)] = {
            "support_metrics": compute_support_class_metrics(pred=pred, truth=truth, support_class=support_class),
            "provenance_metrics": compute_provenance_class_metrics(pred=pred, truth=truth, provenance_class=provenance_class),
        }
    return result


__all__ = [
    "compute_support_class_metrics",
    "compute_provenance_class_metrics",
    "run_ablation_matrix",
]
