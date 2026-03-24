from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from rasterio.transform import Affine


def _existing_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    try:
        p = Path(str(value))
    except (TypeError, ValueError, OSError):
        return None
    return p if p.exists() else None


def _transform_matches(a: Affine, b: Affine, tol: float = 1e-9) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(tuple(a)[:6], tuple(b)[:6]))


def _finite_data(ds, arr: np.ndarray) -> np.ndarray:
    nodata = ds.nodata
    data = np.asarray(arr)
    mask = np.isfinite(data)
    if nodata is not None:
        mask &= ~np.isclose(data, nodata)
    return data[mask]


def _classify_semantics(name: str) -> str:
    if name in {"support_distance", "river_anchor_distance", "river_bank_distance"}:
        return "nonnegative"
    if name in {"anchor_uncertainty", "guidance_uncertainty", "conditioned_uncertainty"}:
        return "nonnegative"
    if name in {"guidance_influence", "support_density", "coastal_sdb_confidence", "river_anchor_density", "river_scaffold_confidence", "river_bank_influence"}:
        return "unit_interval"
    return "free"


def _validate_semantics(name: str, data: np.ndarray) -> list[str]:
    errs: list[str] = []
    if data.size == 0:
        return errs
    sem = _classify_semantics(name)
    if sem == "nonnegative":
        if np.nanmin(data) < -1e-6:
            errs.append(f"{name}_contains_negative_values")
    elif sem == "unit_interval":
        if np.nanmin(data) < -1e-6 or np.nanmax(data) > 1.0 + 1e-6:
            errs.append(f"{name}_outside_0_1_range")
    return errs


def build_final_output_layer_contract(*, outputs: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "conditioned_depth",
        "conditioned_provenance",
        "support_class",
        "support_distance",
        "guidance_influence",
        "anchor_uncertainty",
        "guidance_uncertainty",
        "conditioned_uncertainty",
    ]
    optional = [
        "regime_class",
        "support_density",
        "coastal_sdb_confidence",
        "river_anchor_distance",
        "river_anchor_density",
        "river_scaffold_confidence",
        "river_bank_distance",
        "river_bank_influence",
        "river_bank_elevation",
    ]
    missing_required = [k for k in required if _existing_path(outputs.get(k)) is None]
    present = {k: str(_existing_path(outputs.get(k))) for k in required + optional if _existing_path(outputs.get(k)) is not None}
    payload: Dict[str, Any] = {
        "contract_version": 1,
        "required_rasters": required,
        "optional_rasters": optional,
        "present": present,
        "missing_required": missing_required,
        "ok": len(missing_required) == 0,
        "reference_raster": None,
        "grid_checks": {},
        "semantic_checks": {},
        "errors": [],
    }
    ref_path = _existing_path(outputs.get("conditioned_depth"))
    if ref_path is None:
        payload["errors"].extend(missing_required)
        payload["ok"] = False
        return payload
    payload["reference_raster"] = str(ref_path)
    with rasterio.open(ref_path) as ref_ds:
        ref_shape = (ref_ds.height, ref_ds.width)
        ref_crs = str(ref_ds.crs) if ref_ds.crs is not None else None
        ref_transform = ref_ds.transform
        for name, path_str in present.items():
            path = Path(path_str)
            with rasterio.open(path) as ds:
                shape_ok = (ds.height, ds.width) == ref_shape
                crs_ok = (str(ds.crs) if ds.crs is not None else None) == ref_crs
                transform_ok = _transform_matches(ds.transform, ref_transform)
                checks = {
                    "shape_ok": shape_ok,
                    "crs_ok": crs_ok,
                    "transform_ok": transform_ok,
                    "dtype": ds.dtypes[0],
                }
                payload["grid_checks"][name] = checks
                if not (shape_ok and crs_ok and transform_ok):
                    payload["errors"].append(f"{name}_grid_mismatch")
                data = _finite_data(ds, ds.read(1))
                sem_errors = _validate_semantics(name, data)
                payload["semantic_checks"][name] = {
                    "finite_count": int(data.size),
                    "min": float(np.nanmin(data)) if data.size else None,
                    "max": float(np.nanmax(data)) if data.size else None,
                    "errors": sem_errors,
                }
                payload["errors"].extend(sem_errors)
    payload["ok"] = not payload["errors"] and not payload["missing_required"]
    return payload


def validate_final_output_layer_contract(*, outputs: Dict[str, Any]) -> Dict[str, Any]:
    payload = build_final_output_layer_contract(outputs=outputs)
    if not payload["ok"]:
        raise ValueError("final_output_layer_contract_failed: " + ", ".join(payload["errors"] or payload["missing_required"]))
    return payload


def write_final_output_layer_contract(path: Path | str, *, outputs: Dict[str, Any]) -> Path:
    out = Path(path)
    payload = validate_final_output_layer_contract(outputs=outputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


__all__ = [
    "build_final_output_layer_contract",
    "validate_final_output_layer_contract",
    "write_final_output_layer_contract",
]
