from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import rasterio


EXPECTED_TAGS: Dict[str, Dict[str, str]] = {
    "conditioned_depth": {
        "VALUE_TYPE": "elevation",
        "UNITS": "meters",
        "SIGN_CONVENTION": "relative_to_datum",
        "ROLE": "conditioned_final_depth",
    },
    "source_candidate": {
        "VALUE_TYPE": "elevation",
        "UNITS": "meters",
        "SIGN_CONVENTION": "relative_to_datum",
        "ROLE": "guidance_surface_diagnostic",
    },
    "river_bank_elevation": {
        "VALUE_TYPE": "elevation",
        "UNITS": "meters",
        "SIGN_CONVENTION": "relative_to_datum",
        "ROLE": "river_bank_elevation_guidance",
    },
    "anchor_uncertainty": {
        "VALUE_TYPE": "uncertainty",
        "UNITS": "meters",
        "ROLE": "anchor_uncertainty",
    },
    "guidance_uncertainty": {
        "VALUE_TYPE": "uncertainty",
        "UNITS": "meters",
        "ROLE": "guidance_uncertainty",
    },
    "conditioned_uncertainty": {
        "VALUE_TYPE": "uncertainty",
        "UNITS": "meters",
        "ROLE": "conditioned_uncertainty",
    },
    "support_distance": {
        "VALUE_TYPE": "distance",
        "UNITS": "meters",
        "ROLE": "support_distance",
    },
    "river_anchor_distance": {
        "VALUE_TYPE": "distance",
        "UNITS": "meters",
        "ROLE": "river_anchor_distance",
    },
    "river_bank_distance": {
        "VALUE_TYPE": "distance",
        "UNITS": "meters",
        "ROLE": "river_bank_distance",
    },
    "guidance_influence": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "guidance_influence",
    },
    "support_density": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "support_density",
    },
    "coastal_sdb_confidence": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "coastal_sdb_confidence",
    },
    "river_anchor_density": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "river_anchor_density",
    },
    "river_scaffold_confidence": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "river_scaffold_confidence",
    },
    "river_bank_influence": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "river_bank_influence",
    },
    "river_bank_continuity_weight": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "river_bank_continuity_weight",
    },
    "river_bank_graph_confidence": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "river_bank_graph_confidence",
    },
    "river_bank_confluence_damping": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "river_bank_confluence_damping",
    },
    "river_bank_estuary_side_decay": {
        "VALUE_TYPE": "fraction",
        "UNITS": "unitless",
        "ROLE": "river_bank_estuary_side_decay",
    },
    "support_class": {
        "VALUE_TYPE": "classification",
        "UNITS": "code",
        "ROLE": "support_class",
    },
    "regime_class": {
        "VALUE_TYPE": "classification",
        "UNITS": "code",
        "ROLE": "regime_class",
    },
    "conditioned_provenance": {
        "VALUE_TYPE": "classification",
        "UNITS": "code",
        "ROLE": "conditioned_provenance",
    },
}

VERTICAL_KEYS = {"conditioned_depth", "source_candidate", "river_bank_elevation"}
OPTIONAL_KEYS = set(EXPECTED_TAGS) - {"conditioned_depth", "conditioned_uncertainty", "conditioned_provenance", "support_class", "support_distance", "guidance_influence", "anchor_uncertainty", "guidance_uncertainty"}


def _existing_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    try:
        p = Path(str(value))
    except (TypeError, ValueError, OSError):
        return None
    return p if p.exists() else None


def build_final_vertical_semantics_contract(*, outputs: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contract_version": 1,
        "required": sorted(set(EXPECTED_TAGS) - OPTIONAL_KEYS),
        "optional": sorted(OPTIONAL_KEYS),
        "present": {},
        "checks": {},
        "errors": [],
        "ok": False,
        "vertical_reference": None,
    }
    vertical_datums = {}
    vertical_epsg = {}
    for key, expected in EXPECTED_TAGS.items():
        path = _existing_path(outputs.get(key))
        if path is None:
            if key not in OPTIONAL_KEYS:
                payload["errors"].append(f"missing_{key}")
            continue
        payload["present"][key] = str(path)
        with rasterio.open(path) as ds:
            tags = {str(k): str(v) for k, v in (ds.tags() or {}).items()}
        observed = {k: tags.get(k) for k in expected}
        tag_errors = [f"{key}_tag_{name}_mismatch" for name, value in expected.items() if observed.get(name) != value]
        check = {
            "path": str(path),
            "expected_tags": expected,
            "observed_tags": observed,
            "tag_errors": tag_errors,
        }
        if key in VERTICAL_KEYS:
            check["vertical_datum"] = tags.get("VERTICAL_DATUM")
            check["vertical_datum_epsg"] = tags.get("VERTICAL_DATUM_EPSG")
            check["vertical_semantics"] = tags.get("VERTICAL_SEMANTICS")
            vertical_datums[key] = tags.get("VERTICAL_DATUM")
            vertical_epsg[key] = tags.get("VERTICAL_DATUM_EPSG")
        payload["checks"][key] = check
        payload["errors"].extend(tag_errors)
    present_vertical_datums = {k: v for k, v in vertical_datums.items() if v}
    present_vertical_epsg = {k: v for k, v in vertical_epsg.items() if v}
    if present_vertical_datums:
        unique_datum = sorted(set(present_vertical_datums.values()))
        if len(unique_datum) > 1:
            payload["errors"].append("vertical_datum_mismatch")
        payload["vertical_reference"] = {
            "vertical_datum": unique_datum[0] if len(unique_datum) == 1 else None,
            "vertical_datum_epsg": sorted(set(present_vertical_epsg.values()))[0] if len(set(present_vertical_epsg.values())) == 1 and present_vertical_epsg else None,
            "contributors": sorted(present_vertical_datums),
        }
    if present_vertical_epsg and len(set(present_vertical_epsg.values())) > 1:
        payload["errors"].append("vertical_datum_epsg_mismatch")
    payload["ok"] = not payload["errors"]
    return payload


def validate_final_vertical_semantics_contract(*, outputs: Dict[str, Any]) -> Dict[str, Any]:
    payload = build_final_vertical_semantics_contract(outputs=outputs)
    if not payload["ok"]:
        raise ValueError("final_vertical_semantics_contract_failed: " + ", ".join(payload["errors"]))
    return payload


def write_final_vertical_semantics_contract(path: Path | str, *, outputs: Dict[str, Any]) -> Path:
    out = Path(path)
    payload = validate_final_vertical_semantics_contract(outputs=outputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


__all__ = [
    "EXPECTED_TAGS",
    "build_final_vertical_semantics_contract",
    "validate_final_vertical_semantics_contract",
    "write_final_vertical_semantics_contract",
]
