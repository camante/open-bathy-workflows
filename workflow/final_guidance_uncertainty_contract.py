from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

REQUIRED_SUPPORT_OUTPUTS: Dict[str, Dict[str, str]] = {
    "support_class": {"kind": "raster", "units": "code", "role": "structural"},
    "support_distance": {"kind": "raster", "units": "meters", "role": "diagnostic"},
    "guidance_influence": {"kind": "raster", "units": "0_to_1", "role": "diagnostic"},
    "anchor_uncertainty": {"kind": "raster", "units": "vertical_meters", "role": "structural"},
    "guidance_uncertainty": {"kind": "raster", "units": "vertical_meters", "role": "structural"},
    "conditioned_uncertainty": {"kind": "raster", "units": "vertical_meters", "role": "structural"},
    "conditioned_depth": {"kind": "raster", "units": "vertical_meters", "role": "structural"},
    "conditioned_provenance": {"kind": "raster", "units": "code", "role": "structural"},
}

OPTIONAL_SUPPORT_OUTPUTS: Dict[str, Dict[str, str]] = {
    "support_density": {"kind": "raster", "units": "0_to_1", "role": "diagnostic"},
    "regime_class": {"kind": "raster", "units": "code", "role": "structural"},
    "coastal_sdb_confidence": {"kind": "raster", "units": "0_to_1", "role": "diagnostic"},
    "river_anchor_distance": {"kind": "raster", "units": "meters", "role": "diagnostic"},
    "river_anchor_density": {"kind": "raster", "units": "0_to_1", "role": "diagnostic"},
    "river_scaffold_confidence": {"kind": "raster", "units": "0_to_1", "role": "diagnostic"},
    "river_bank_distance": {"kind": "raster", "units": "meters", "role": "diagnostic"},
    "river_bank_influence": {"kind": "raster", "units": "0_to_1", "role": "diagnostic"},
    "river_bank_elevation": {"kind": "raster", "units": "vertical_meters", "role": "diagnostic"},
}


def _existing_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    try:
        p = Path(str(value))
    except (TypeError, ValueError, OSError):
        return None
    return p if p.exists() else None



def build_guidance_uncertainty_contract(*, outputs: Dict[str, Any], support_note: Optional[str] = None) -> Dict[str, Any]:
    present_required: Dict[str, str] = {}
    missing_required = []
    for key, meta in REQUIRED_SUPPORT_OUTPUTS.items():
        p = _existing_path(outputs.get(key))
        if p is None:
            missing_required.append(key)
        else:
            present_required[key] = str(p)

    present_optional: Dict[str, str] = {}
    for key in OPTIONAL_SUPPORT_OUTPUTS:
        p = _existing_path(outputs.get(key))
        if p is not None:
            present_optional[key] = str(p)

    artifact_specs = {**REQUIRED_SUPPORT_OUTPUTS, **OPTIONAL_SUPPORT_OUTPUTS}
    return {
        "contract_version": 1,
        "mode": "authoritative_first_guidance_uncertainty_conditioning",
        "required_outputs": REQUIRED_SUPPORT_OUTPUTS,
        "optional_outputs": OPTIONAL_SUPPORT_OUTPUTS,
        "present_required": present_required,
        "present_optional": present_optional,
        "missing_required": missing_required,
        "ok": len(missing_required) == 0,
        "support_note": support_note,
        "artifact_specs": artifact_specs,
    }



def validate_guidance_uncertainty_contract(*, outputs: Dict[str, Any], support_note: Optional[str] = None) -> Dict[str, Any]:
    payload = build_guidance_uncertainty_contract(outputs=outputs, support_note=support_note)
    if payload["missing_required"]:
        raise ValueError(f"Missing required conditioning outputs: {', '.join(payload['missing_required'])}")
    return payload



def write_guidance_uncertainty_contract(path: Path | str, *, outputs: Dict[str, Any], support_note: Optional[str] = None) -> Path:
    out = Path(path)
    payload = validate_guidance_uncertainty_contract(outputs=outputs, support_note=support_note)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


__all__ = [
    "REQUIRED_SUPPORT_OUTPUTS",
    "OPTIONAL_SUPPORT_OUTPUTS",
    "build_guidance_uncertainty_contract",
    "validate_guidance_uncertainty_contract",
    "write_guidance_uncertainty_contract",
]
