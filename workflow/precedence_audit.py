from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from nodata_utils import array_valid_mask

from provenance_schema import PROVENANCE_CLASS_CODE_TO_NAME, PROVENANCE_CLASS_FAMILY
from support_classes import SUPPORT_CLASS_CODE_TO_NAME, SUPPORT_CLASS_FAMILY


def _counts(arr: np.ndarray, mapping: Dict[int, str], families: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_code": {}, "by_name": {}, "by_family": {}}
    unique, counts = np.unique(arr.astype("uint8"), return_counts=True)
    for code, count in zip(unique.tolist(), counts.tolist()):
        if int(code) == 0:
            continue
        name = mapping.get(int(code), f"unknown_{int(code)}")
        out["by_code"][str(int(code))] = int(count)
        out["by_name"][name] = int(count)
        if families is not None:
            fam = families.get(int(code), name)
            out["by_family"][fam] = int(out["by_family"].get(fam, 0)) + int(count)
    return out


def summarize_precedence_audit(
    *,
    auth: np.ndarray,
    conditioned: np.ndarray,
    support: np.ndarray,
    provenance: np.ndarray,
    guidance_influence: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    auth = np.asarray(auth, dtype="float32")
    conditioned = np.asarray(conditioned, dtype="float32")
    support = np.asarray(support, dtype="uint8")
    provenance = np.asarray(provenance, dtype="uint8")
    locked = array_valid_mask(auth)
    gap = ~locked
    conditioned_valid = array_valid_mask(conditioned)
    changed_locked = locked & conditioned_valid & (~np.isclose(auth, conditioned, equal_nan=True))
    guidance_arr = None if guidance_influence is None else np.asarray(guidance_influence, dtype="float32")
    guidance_on_locked = 0
    if guidance_arr is not None:
        guidance_on_locked = int(np.count_nonzero(locked & (np.nan_to_num(guidance_arr, nan=0.0) > 1e-6)))
    return {
        "authoritative_lock": {
            "locked_cell_count": int(np.count_nonzero(locked)),
            "changed_locked_cell_count": int(np.count_nonzero(changed_locked)),
            "lock_preserved": bool(np.count_nonzero(changed_locked) == 0),
            "guidance_nonzero_on_locked_count": guidance_on_locked,
        },
        "gap_fill": {
            "authoritative_gap_cell_count": int(np.count_nonzero(gap)),
            "conditioned_finite_gap_count": int(np.count_nonzero(gap & conditioned_valid)),
            "remaining_gap_nodata_count": int(np.count_nonzero(gap & ~conditioned_valid)),
            "continuous_fill_achieved": bool(np.count_nonzero(gap & ~conditioned_valid) == 0),
        },
        "support_classes": _counts(support, SUPPORT_CLASS_CODE_TO_NAME, SUPPORT_CLASS_FAMILY),
        "provenance_classes": _counts(provenance, PROVENANCE_CLASS_CODE_TO_NAME, PROVENANCE_CLASS_FAMILY),
    }


def validate_precedence_audit(audit: Dict[str, Any]) -> None:
    lock = audit.get("authoritative_lock", {})
    gap = audit.get("gap_fill", {})
    if not isinstance(lock, dict) or not isinstance(gap, dict):
        raise ValueError("Precedence audit missing required sections")
    changed_locked = int(lock.get("changed_locked_cell_count", 0))
    if changed_locked != 0:
        raise ValueError("Changed locked authoritative cells detected")
    if not bool(lock.get("lock_preserved", False)):
        raise ValueError("Authoritative lock preservation failed")
    if int(lock.get("guidance_nonzero_on_locked_count", 0)) != 0:
        raise ValueError("Guidance influence present on authoritative locked cells")
    if int(gap.get("remaining_gap_nodata_count", 0)) < 0:
        raise ValueError("Invalid remaining gap count in precedence audit")


def write_precedence_audit(path: Path | str, audit: Dict[str, Any], *, extra: Optional[Dict[str, Any]] = None) -> Path:
    validate_precedence_audit(audit)
    out = Path(path)
    payload = dict(audit)
    if extra:
        payload.update(extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


__all__ = [
    "summarize_precedence_audit",
    "validate_precedence_audit",
    "write_precedence_audit",
]
