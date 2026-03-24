from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import rasterio


REQUIRED = {
    "centerline_stationing": "centerline_stationing_coordinate",
    "centerline_elevation": "centerline_longitudinal_tendency",
    "longitudinal_profile": "station_indexed_longitudinal_profile_table",
    "longitudinal_profile_elevation": "rasterized_longitudinal_profile_elevation",
    "longitudinal_profile_uncertainty": "rasterized_longitudinal_profile_uncertainty",
}
OPTIONAL = {
    "xs_support_elevation": "cross_stream_support_elevation",
    "authoritative_support_depth": "authoritative_anchor_depth",
    "bank_elevation_xs": "xs_longitudinal_bank_elevation",
}


def _existing(value: Any) -> Optional[Path]:
    if value is None:
        return None
    try:
        p = Path(str(value))
    except (TypeError, ValueError, OSError):
        return None
    return p if p.exists() else None


def _is_raster(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _grid_signature(path: Path) -> Dict[str, Any]:
    with rasterio.open(path) as ds:
        return {
            "shape": [int(ds.height), int(ds.width)],
            "crs": str(ds.crs) if ds.crs is not None else None,
            "transform": list(tuple(ds.transform)[:6]),
        }


def build_river_longitudinal_profile_contract(*, outputs: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contract_version": 1,
        "phase": "phase_c_scaffold_vertical_transition",
        "mode": "network_aware_hydraulic_backbone_v1",
        "required": REQUIRED,
        "optional": OPTIONAL,
        "present": {},
        "errors": [],
        "grid_reference": None,
        "ok": False,
        "notes": [
            "This contract validates the scaffold-owned station-indexed longitudinal bed-profile object and its network-aware 1D backbone export.",
            "The profile object is represented by a station-indexed table plus rasterized elevation and uncertainty derivatives.",
            "The hydraulic backbone export encodes component smoothing plus junction continuity using the scaffold graph when available.",
        ],
    }
    present_required = {}
    for key in REQUIRED:
        path = _existing(outputs.get(key))
        if path is None:
            payload["errors"].append(f"missing_{key}")
        else:
            present_required[key] = path
            payload["present"][key] = str(path)
    for key in OPTIONAL:
        path = _existing(outputs.get(key))
        if path is not None:
            payload["present"][key] = str(path)
    ref = present_required.get("centerline_stationing") or present_required.get("longitudinal_profile_elevation") or present_required.get("centerline_elevation")
    if ref is not None:
        ref_sig = _grid_signature(ref)
        payload["grid_reference"] = ref_sig
        for key, path_str in payload["present"].items():
            path_obj = Path(path_str)
            if not _is_raster(path_obj):
                continue
            sig = _grid_signature(path_obj)
            if sig != ref_sig:
                payload["errors"].append(f"{key}_grid_mismatch")
    payload["ok"] = not payload["errors"]
    return payload


def validate_river_longitudinal_profile_contract(*, outputs: Dict[str, Any]) -> Dict[str, Any]:
    payload = build_river_longitudinal_profile_contract(outputs=outputs)
    if not payload["ok"]:
        raise ValueError("river_longitudinal_profile_contract_failed: " + ", ".join(payload["errors"]))
    return payload


def write_river_longitudinal_profile_contract(path: Path | str, *, outputs: Dict[str, Any]) -> Path:
    out = Path(path)
    payload = validate_river_longitudinal_profile_contract(outputs=outputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


__all__ = [
    "build_river_longitudinal_profile_contract",
    "validate_river_longitudinal_profile_contract",
    "write_river_longitudinal_profile_contract",
]
