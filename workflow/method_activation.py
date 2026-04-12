from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from activation_truth_utils import count_mask_value_pixels


@dataclass
class MethodActivationTruth:
    method_name: str
    requested: bool
    requested_reason: Optional[str]
    domain_summary_should_run: Optional[bool]
    validated_mask_should_run: Optional[bool]
    validated_mask_pixels: Optional[int]
    effective_should_run: bool
    effective_reason: str
    candidate_domain_mask: Optional[str]
    active_domain_mask: Optional[str]
    semantic_valid: Optional[bool]
    semantic_reason: Optional[str]
    active_guidance_product: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _requested_methods(cfg: Any) -> List[str]:
    return [
        str(m).strip().lower()
        for m in (getattr(cfg, "methods_requested", None) or getattr(cfg, "methods", []) or [])
        if str(m).strip()
    ]


def _load_domain_summary(summary_path: Any) -> Dict[str, Any]:
    if summary_path is None:
        raise RuntimeError("shared_domain_summary_missing")
    summary_file = Path(summary_path)
    if not summary_file.exists():
        raise RuntimeError(f"shared_domain_summary_missing: path={summary_file}")
    try:
        with summary_file.open("r", encoding="utf-8") as f:
            summary = json.load(f) or {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"shared_domain_summary_read_failed: path={summary_file} error={exc}") from exc
    if not isinstance(summary, dict):
        raise RuntimeError(f"shared_domain_summary_invalid_type: path={summary_file} type={type(summary).__name__}")
    return summary


def _as_bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def determine_method_activation_truth(cfg: Any, report: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], Dict[str, MethodActivationTruth]]:
    requested = _requested_methods(cfg)
    requested_set = set(requested)
    summary_path = getattr(cfg, "domain_review_summary", None)
    summary = _load_domain_summary(summary_path)
    derived = summary.get("derived_activation", {})
    if not isinstance(derived, dict):
        raise RuntimeError(f"shared_domain_summary_missing_activation: path={summary_path}")

    river_summary_should_run = bool(derived.get("river_should_run", False))
    sdb_summary_should_run = bool(derived.get("sdb_should_run", False))

    validated_river_mask = getattr(cfg, "river_guidance_domain_mask", None)
    validated_sdb_mask = getattr(cfg, "validated_sdb_guidance_domain_mask", None) or getattr(cfg, "sdb_guidance_domain_mask", None)
    validated_river_pixels = count_mask_value_pixels(Path(validated_river_mask), value=1) if validated_river_mask else None
    validated_sdb_pixels = getattr(cfg, "validated_sdb_guidance_domain_pixels", None)
    if validated_sdb_pixels is None and validated_sdb_mask:
        validated_sdb_pixels = count_mask_value_pixels(Path(validated_sdb_mask), value=0)

    activation_mismatch: Dict[str, Any] = {}
    river_effective_should_run = river_summary_should_run
    sdb_effective_should_run = sdb_summary_should_run

    river_validated_should_run = None if validated_river_pixels is None else bool(int(validated_river_pixels) > 0)
    sdb_validated_should_run = None if validated_sdb_pixels is None else bool(int(validated_sdb_pixels) > 0)

    if river_validated_should_run is not None and river_validated_should_run != river_summary_should_run:
        activation_mismatch["river"] = {
            "summary_should_run": bool(river_summary_should_run),
            "validated_should_run": bool(river_validated_should_run),
            "validated_pixels": int(validated_river_pixels),
            "validated_mask": str(validated_river_mask),
        }
        river_effective_should_run = river_validated_should_run
    if sdb_validated_should_run is not None and sdb_validated_should_run != sdb_summary_should_run:
        activation_mismatch["sdb"] = {
            "summary_should_run": bool(sdb_summary_should_run),
            "validated_should_run": bool(sdb_validated_should_run),
            "validated_pixels": int(validated_sdb_pixels),
            "validated_mask": str(validated_sdb_mask) if validated_sdb_mask else None,
        }
        sdb_effective_should_run = sdb_validated_should_run

    effective: List[str] = []
    skipped: Dict[str, str] = {}
    for method in requested:
        if method == "river":
            if river_effective_should_run:
                effective.append(method)
            else:
                skipped[method] = "shared_domain_empty"
        elif method == "sdb":
            if sdb_effective_should_run:
                effective.append(method)
            else:
                skipped[method] = "shared_domain_empty"
        elif method == "fuse":
            if ({"river", "sdb"} & requested_set) and (river_effective_should_run or sdb_effective_should_run):
                effective.append(method)
            elif not ({"river", "sdb"} & requested_set):
                effective.append(method)
            else:
                skipped[method] = "no_active_water_methods"
        else:
            effective.append(method)

    truths: Dict[str, MethodActivationTruth] = {
        "river": MethodActivationTruth(
            method_name="river",
            requested="river" in requested_set,
            requested_reason="requested" if "river" in requested_set else None,
            domain_summary_should_run=river_summary_should_run,
            validated_mask_should_run=river_validated_should_run,
            validated_mask_pixels=int(validated_river_pixels) if validated_river_pixels is not None else None,
            effective_should_run=river_effective_should_run,
            effective_reason="validated_execution_mask" if "river" in activation_mismatch else ("domain_summary" if river_effective_should_run else "shared_domain_empty"),
            candidate_domain_mask=None,
            active_domain_mask=str(validated_river_mask) if validated_river_mask else None,
            semantic_valid=None,
            semantic_reason=None,
            active_guidance_product=None,
        ),
        "sdb": MethodActivationTruth(
            method_name="sdb",
            requested="sdb" in requested_set,
            requested_reason="requested" if "sdb" in requested_set else None,
            domain_summary_should_run=sdb_summary_should_run,
            validated_mask_should_run=sdb_validated_should_run,
            validated_mask_pixels=int(validated_sdb_pixels) if validated_sdb_pixels is not None else None,
            effective_should_run=sdb_effective_should_run,
            effective_reason="validated_execution_mask" if "sdb" in activation_mismatch else ("domain_summary" if sdb_effective_should_run else "shared_domain_empty"),
            candidate_domain_mask=None,
            active_domain_mask=str(validated_sdb_mask) if validated_sdb_mask else None,
            semantic_valid=None,
            semantic_reason=None,
            active_guidance_product=None,
        ),
    }

    meta = {
        "activation_source": "shared_domains",
        "requested": requested,
        "effective": list(effective),
        "skipped": skipped,
        "domain_summary": str(summary_path) if summary_path else None,
        "derived_activation": {
            "river_should_run": river_effective_should_run,
            "sdb_should_run": sdb_effective_should_run,
        },
        "validated_execution_masks": {
            "river_guidance_domain_mask": str(validated_river_mask) if validated_river_mask else None,
            "river_guidance_domain_pixels": int(validated_river_pixels) if validated_river_pixels is not None else None,
            "sdb_guidance_domain_mask": str(validated_sdb_mask) if validated_sdb_mask else None,
            "sdb_guidance_domain_pixels": int(validated_sdb_pixels) if validated_sdb_pixels is not None else None,
        },
        "activation_mismatch": activation_mismatch,
        "method_activation_truth": {name: truth.to_dict() for name, truth in truths.items()},
    }
    report.setdefault("guidance_domains", {})["activation"] = meta
    report["method_activation_truth"] = {name: truth.to_dict() for name, truth in truths.items()}
    return effective, meta, truths
