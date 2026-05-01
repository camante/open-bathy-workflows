from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.nodata_utils import array_valid_mask

from provenance_schema import PROVENANCE_CLASS_CODE_TO_NAME, PROVENANCE_CLASS_FAMILY
from support_classes import SUPPORT_CLASS_CODE_TO_NAME, SUPPORT_CLASS_FAMILY, class_is_authoritative_locked


def _safe_stats(arr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    vals = np.asarray(arr, dtype="float32")[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95.0)),
        "max": float(np.max(vals)),
    }



def _class_breakdown(arr: np.ndarray, codes: Dict[int, str], families: Dict[int, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_class": {}, "by_family": {}}
    arr_u8 = np.asarray(arr, dtype="uint8")
    unique, counts = np.unique(arr_u8, return_counts=True)
    for code, count in zip(unique.tolist(), counts.tolist()):
        if int(code) == 0:
            continue
        label = codes.get(int(code), f"unknown_{int(code)}")
        family = families.get(int(code), label)
        out["by_class"][str(int(code))] = {"label": label, "count": int(count)}
        cur = out["by_family"].setdefault(family, {"count": 0})
        cur["count"] += int(count)
    return out



def summarize_conditioning_audit(
    *,
    auth: np.ndarray,
    conditioned: np.ndarray,
    support: np.ndarray,
    provenance: np.ndarray,
    guidance_influence: Optional[np.ndarray] = None,
    conditioned_uncertainty: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    auth = np.asarray(auth, dtype="float32")
    conditioned = np.asarray(conditioned, dtype="float32")
    support = np.asarray(support, dtype="uint8")
    provenance = np.asarray(provenance, dtype="uint8")
    locked = array_valid_mask(auth)
    gap = ~locked
    conditioned_valid = array_valid_mask(conditioned)
    changed_locked = locked & conditioned_valid & (~np.isclose(auth, conditioned, equal_nan=True))
    guidance = np.asarray(guidance_influence, dtype="float32") if guidance_influence is not None else None
    sigma = np.asarray(conditioned_uncertainty, dtype="float32") if conditioned_uncertainty is not None else None
    guided = (np.nan_to_num(guidance, nan=0.0) > 1e-6) if guidance is not None else np.zeros_like(locked, dtype=bool)
    dominant_guided = (np.nan_to_num(guidance, nan=0.0) >= 0.5) if guidance is not None else np.zeros_like(locked, dtype=bool)

    support_codes = {int(k): v for k, v in SUPPORT_CLASS_CODE_TO_NAME.items()}
    prov_codes = {int(k): v for k, v in PROVENANCE_CLASS_CODE_TO_NAME.items()}

    auth_locked_mask = np.zeros_like(support, dtype=bool)
    for code in np.unique(support):
        if class_is_authoritative_locked(int(code)):
            auth_locked_mask |= support == int(code)

    out: Dict[str, Any] = {
        "contract_version": 1,
        "authoritative_lock": {
            "locked_cell_count": int(np.count_nonzero(locked)),
            "support_authoritative_locked_cell_count": int(np.count_nonzero(auth_locked_mask)),
            "unchanged_locked_cell_count": int(np.count_nonzero(locked & conditioned_valid & np.isclose(auth, conditioned, equal_nan=True))),
            "changed_locked_cell_count": int(np.count_nonzero(changed_locked)),
            "lock_preserved": bool(np.count_nonzero(changed_locked) == 0),
        },
        "guidance_impact": {
            "guided_fraction_of_domain": float(np.count_nonzero(guided) / guided.size) if guided.size else 0.0,
            "dominant_guided_fraction_of_domain": float(np.count_nonzero(dominant_guided) / dominant_guided.size) if dominant_guided.size else 0.0,
            "guided_gap_fraction": float(np.count_nonzero(guided & gap) / max(np.count_nonzero(gap), 1)),
        },
        "gap_fill": {
            "authoritative_gap_cell_count": int(np.count_nonzero(gap)),
            "conditioned_finite_gap_count": int(np.count_nonzero(gap & conditioned_valid)),
            "remaining_gap_nodata_count": int(np.count_nonzero(gap & ~conditioned_valid)),
            "continuous_fill_achieved": bool(np.count_nonzero(gap & ~conditioned_valid) == 0),
        },
        "uncertainty_summary": {
            "overall": _safe_stats(sigma, np.isfinite(sigma)) if sigma is not None else {"count": 0, "mean": None, "median": None, "p95": None, "max": None},
            "guided": _safe_stats(sigma, guided) if sigma is not None else {"count": 0, "mean": None, "median": None, "p95": None, "max": None},
            "authoritative_locked": _safe_stats(sigma, auth_locked_mask) if sigma is not None else {"count": 0, "mean": None, "median": None, "p95": None, "max": None},
            "gap_only": _safe_stats(sigma, gap) if sigma is not None else {"count": 0, "mean": None, "median": None, "p95": None, "max": None},
        },
        "support_classes": _class_breakdown(support, support_codes, SUPPORT_CLASS_FAMILY),
        "provenance_classes": _class_breakdown(provenance, prov_codes, PROVENANCE_CLASS_FAMILY),
    }
    return out



def write_conditioning_audit(
    path: Path | str,
    *,
    auth: np.ndarray,
    conditioned: np.ndarray,
    support: np.ndarray,
    provenance: np.ndarray,
    guidance_influence: Optional[np.ndarray] = None,
    conditioned_uncertainty: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    out = Path(path)
    payload = summarize_conditioning_audit(
        auth=auth,
        conditioned=conditioned,
        support=support,
        provenance=provenance,
        guidance_influence=guidance_influence,
        conditioned_uncertainty=conditioned_uncertainty,
    )
    if extra:
        payload.update(extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


__all__ = ["summarize_conditioning_audit", "write_conditioning_audit"]
