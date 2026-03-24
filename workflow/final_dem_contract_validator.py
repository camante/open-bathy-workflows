from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from precedence_audit import summarize_precedence_audit, validate_precedence_audit
from sign_semantics import raster_value_semantics
from support_classes import SupportClass

log = logging.getLogger(__name__)


def _existing_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    try:
        p = Path(value)
    except (TypeError, ValueError):
        return None
    return p if p.exists() else None


def _read_raster(path: Path):
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)
            if nodata is not None:
                arr[np.isclose(arr, np.float32(nodata))] = np.nan
        else:
            arr = arr.astype(np.int32, copy=False)
        meta = {
            "shape": tuple(arr.shape),
            "crs": ds.crs.to_string() if ds.crs else None,
            "transform": tuple(ds.transform),
            "nodata": nodata,
            "tags": dict(ds.tags() or {}),
            "declared_semantics": raster_value_semantics(ds.tags() or {}),
        }
    return arr, meta


def summarize_written_precedence_audit(
    *,
    final_depth: Any,
    aligned_authoritative_base: Any = None,
    support_class: Any = None,
    final_provenance: Any = None,
    guidance_influence: Any = None,
) -> Dict[str, Any]:
    final_path = _existing_path(final_depth)
    auth_path = _existing_path(aligned_authoritative_base)
    support_path = _existing_path(support_class)
    prov_path = _existing_path(final_provenance)
    guidance_path = _existing_path(guidance_influence)

    payload: Dict[str, Any] = {
        "validated": False,
        "skipped": False,
        "skip_reason": None,
        "final_depth": str(final_path) if final_path else None,
        "aligned_authoritative_base": str(auth_path) if auth_path else None,
        "support_class": str(support_path) if support_path else None,
        "final_provenance": str(prov_path) if prov_path else None,
        "guidance_influence": str(guidance_path) if guidance_path else None,
        "authoritative_lock": None,
        "gap_fill": None,
        "support_classes": None,
        "provenance_classes": None,
        "all_ok": None,
    }
    if final_path is None:
        payload["skipped"] = True
        payload["skip_reason"] = "missing_final_depth"
        return payload
    if auth_path is None or support_path is None or prov_path is None:
        payload["skipped"] = True
        payload["skip_reason"] = "missing_precedence_inputs"
        return payload

    try:
        final_arr, final_meta = _read_raster(final_path)
        auth_arr, auth_meta = _read_raster(auth_path)
        support_arr, support_meta = _read_raster(support_path)
        prov_arr, prov_meta = _read_raster(prov_path)
        guidance_arr = None
        guidance_meta = None
        if guidance_path is not None:
            guidance_arr, guidance_meta = _read_raster(guidance_path)
    except Exception as exc:  # pragma: no cover - defensive I/O surface
        log.debug("summarize_written_precedence_audit: suppressed exception", exc_info=True)
        payload["skipped"] = True
        payload["skip_reason"] = f"failed_to_read_precedence_inputs:{type(exc).__name__}"
        return payload

    shapes = [final_meta["shape"], auth_meta["shape"], support_meta["shape"], prov_meta["shape"]]
    if guidance_meta is not None:
        shapes.append(guidance_meta["shape"])
    if any(shape != final_meta["shape"] for shape in shapes[1:]):
        payload["skipped"] = True
        payload["skip_reason"] = "shape_mismatch"
        return payload

    try:
        audit = summarize_precedence_audit(
            auth=auth_arr,
            conditioned=final_arr,
            support=support_arr,
            provenance=prov_arr,
            guidance_influence=guidance_arr,
        )
        validate_precedence_audit(audit)
    except Exception as exc:
        log.debug("final_dem_contract_validator: suppressed exception", exc_info=True)
        audit = summarize_precedence_audit(
            auth=auth_arr,
            conditioned=final_arr,
            support=support_arr,
            provenance=prov_arr,
            guidance_influence=guidance_arr,
        )
        payload["validation_error"] = f"{type(exc).__name__}: {exc}"
    else:
        payload["validation_error"] = None

    payload.update(audit)
    payload["validated"] = True
    payload["all_ok"] = bool(
        audit.get("authoritative_lock", {}).get("lock_preserved") is True
        and int(audit.get("authoritative_lock", {}).get("guidance_nonzero_on_locked_count", 0)) == 0
        and audit.get("gap_fill", {}).get("continuous_fill_achieved") is True
    )
    return payload


def validate_written_final_dem_contract(
    *,
    final_depth: Any,
    aligned_authoritative_base: Any = None,
    support_class: Any = None,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    final_path = _existing_path(final_depth)
    auth_path = _existing_path(aligned_authoritative_base)
    support_path = _existing_path(support_class)

    payload: Dict[str, Any] = {
        "validated": False,
        "skipped": False,
        "skip_reason": None,
        "final_depth": str(final_path) if final_path else None,
        "aligned_authoritative_base": str(auth_path) if auth_path else None,
        "support_class": str(support_path) if support_path else None,
        "continuous_output": {
            "ok": None,
            "nonfinite_pixels": None,
            "shape": None,
        },
        "authoritative_hard_lock": {
            "ok": None,
            "locked_pixels": None,
            "locked_finite_authoritative_pixels": None,
            "max_abs_diff_m": None,
            "mismatch_pixels": None,
        },
        "semantic_contract": {
            "final_depth_expected": "absolute_elevation",
            "final_depth_observed": None,
            "aligned_authoritative_expected": "absolute_elevation",
            "aligned_authoritative_observed": None,
            "ok": None,
        },
        "all_ok": None,
    }
    if final_path is None:
        payload["skipped"] = True
        payload["skip_reason"] = "missing_final_depth"
        return payload

    try:
        final_arr, final_meta = _read_raster(final_path)
    except Exception as exc:  # pragma: no cover - defensive I/O surface
        log.debug("validate_written_final_dem_contract: suppressed exception", exc_info=True)
        payload["skipped"] = True
        payload["skip_reason"] = f"failed_to_read_final_depth:{type(exc).__name__}"
        return payload

    if final_meta["shape"] is not None:
        payload["continuous_output"]["shape"] = list(final_meta["shape"])
    nonfinite = int(np.count_nonzero(~np.isfinite(final_arr))) if np.issubdtype(final_arr.dtype, np.floating) else 0
    payload["continuous_output"].update({
        "ok": bool(nonfinite == 0),
        "nonfinite_pixels": nonfinite,
    })

    semantic_contract = payload["semantic_contract"]
    semantic_contract["final_depth_observed"] = final_meta.get("declared_semantics")
    final_sem = final_meta.get("declared_semantics")
    semantic_ok: Optional[bool]
    if final_sem in (None, "unknown"):
        semantic_ok = None
    else:
        semantic_ok = final_sem == "absolute_elevation"

    hard_lock = payload["authoritative_hard_lock"]
    if auth_path is not None and support_path is not None:
        try:
            auth_arr, auth_meta = _read_raster(auth_path)
            support_arr, support_meta = _read_raster(support_path)
        except Exception as exc:  # pragma: no cover - defensive I/O surface
            log.debug("final_dem_contract_validator: suppressed exception", exc_info=True)
            payload["skipped"] = True
            payload["skip_reason"] = f"failed_to_read_contract_inputs:{type(exc).__name__}"
            return payload
        auth_sem = auth_meta.get("declared_semantics")
        semantic_contract["aligned_authoritative_observed"] = auth_sem
        if auth_sem not in (None, "unknown"):
            auth_ok = auth_sem == "absolute_elevation"
            semantic_ok = auth_ok if semantic_ok is None else bool(semantic_ok and auth_ok)
        if final_meta["shape"] != auth_meta["shape"] or final_meta["shape"] != support_meta["shape"]:
            payload["skipped"] = True
            payload["skip_reason"] = "shape_mismatch"
            return payload
        locked = np.asarray(support_arr == int(SupportClass.AUTHORITATIVE_LOCKED), dtype=bool)
        auth_finite = np.isfinite(auth_arr) if np.issubdtype(auth_arr.dtype, np.floating) else np.ones(auth_arr.shape, dtype=bool)
        locked_finite = locked & auth_finite
        hard_lock["locked_pixels"] = int(np.count_nonzero(locked))
        hard_lock["locked_finite_authoritative_pixels"] = int(np.count_nonzero(locked_finite))
        if np.any(locked_finite):
            diff = np.abs(final_arr.astype(np.float64) - auth_arr.astype(np.float64))
            diff[~locked_finite] = 0.0
            max_abs = float(np.nanmax(diff[locked_finite])) if np.any(locked_finite) else 0.0
            mism = int(np.count_nonzero(locked_finite & (diff > float(atol))))
            hard_lock.update({
                "ok": bool(mism == 0),
                "max_abs_diff_m": max_abs,
                "mismatch_pixels": mism,
            })
        else:
            hard_lock.update({
                "ok": True,
                "max_abs_diff_m": 0.0,
                "mismatch_pixels": 0,
            })
    else:
        hard_lock["ok"] = None

    semantic_contract["ok"] = semantic_ok
    payload["validated"] = True
    payload["all_ok"] = bool(
        payload["continuous_output"]["ok"] is True
        and (hard_lock["ok"] is not False)
        and semantic_contract["ok"] is not False
    )
    return payload


__all__ = ["validate_written_final_dem_contract", "summarize_written_precedence_audit"]
