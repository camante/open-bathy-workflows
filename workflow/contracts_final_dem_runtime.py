from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio

from contract_enforcement import ContractResult, ContractSuiteResult
from final_dem_contract_validator import (
    summarize_written_precedence_audit,
    validate_written_final_dem_contract,
)
from provenance_schema import ProvenanceClass
from support_classes import SupportClass


def _existing_path(value: Any) -> Path | None:
    if not value:
        return None
    try:
        path = Path(value)
    except (TypeError, ValueError, OSError):
        return None
    return path if path.exists() else None




def _write_debug_mask(path: Path, mask: np.ndarray, template_path: Path) -> str | None:
    if not np.any(mask):
        return None
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(mask.astype("uint8"), 1)
    return str(path)

def _profile(path: Path) -> Dict[str, Any]:
    with rasterio.open(path) as ds:
        return {
            "width": ds.width,
            "height": ds.height,
            "crs": ds.crs.to_string() if ds.crs else None,
            "transform": tuple(ds.transform),
        }


def _same_grid(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a.get("width"), a.get("height"), a.get("crs"), a.get("transform")) == (
        b.get("width"), b.get("height"), b.get("crs"), b.get("transform")
    )


def _guidance_allowed_support_mask(support_arr: np.ndarray) -> np.ndarray:
    allowed = {
        int(SupportClass.GUIDANCE_CONDITIONED_SDB),
        int(SupportClass.GUIDANCE_CONDITIONED_RIVER),
        int(SupportClass.SCAFFOLD_INFERRED),
        int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL),
    }
    mask = np.zeros(support_arr.shape, dtype=bool)
    for code in allowed:
        mask |= support_arr == code
    return mask


def _guidance_allowed_provenance_mask(prov_arr: np.ndarray) -> np.ndarray:
    allowed = {
        int(ProvenanceClass.SDB_CONDITIONED_FILL),
        int(ProvenanceClass.RIVER_CONDITIONED_FILL),
        int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL),
        int(ProvenanceClass.LOW_CONFIDENCE_FILL),
    }
    mask = np.zeros(prov_arr.shape, dtype=bool)
    for code in allowed:
        mask |= prov_arr == code
    return mask


def _expected_provenance_from_support(support_arr: np.ndarray) -> np.ndarray:
    out = np.zeros(support_arr.shape, dtype=np.uint8)
    mapping = {
        int(SupportClass.AUTHORITATIVE_LOCKED): int(ProvenanceClass.AUTHORITATIVE_LOCKED),
        int(SupportClass.ANCHORED_INTERPOLATION): int(ProvenanceClass.ANCHORED_INTERPOLATION),
        int(SupportClass.GUIDANCE_CONDITIONED_SDB): int(ProvenanceClass.SDB_CONDITIONED_FILL),
        int(SupportClass.GUIDANCE_CONDITIONED_RIVER): int(ProvenanceClass.RIVER_CONDITIONED_FILL),
        int(SupportClass.SCAFFOLD_INFERRED): int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL),
        int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL): int(ProvenanceClass.LOW_CONFIDENCE_FILL),
    }
    for support_code, prov_code in mapping.items():
        out[support_arr == support_code] = prov_code
    return out


def run_final_dem_runtime_contracts(*, report: Dict[str, Any], final_depth: Any = None, debug_dir: Path | None = None) -> ContractSuiteResult:
    suite = ContractSuiteResult(stage="final_dem")
    ab = report.get("authoritative_base", {}) if isinstance(report.get("authoritative_base"), dict) else {}
    outputs = ab.get("outputs", {}) if isinstance(ab.get("outputs"), dict) else {}

    final_path = _existing_path(final_depth) or _existing_path(outputs.get("conditioned_depth"))
    auth_path = _existing_path(outputs.get("aligned_authoritative_base"))
    support_path = _existing_path(outputs.get("support_class"))
    prov_path = _existing_path(outputs.get("conditioned_provenance"))
    guidance_influence_path = _existing_path(outputs.get("guidance_influence"))

    required = {
        "final_depth": final_path,
        "aligned_authoritative_base": auth_path,
        "support_class": support_path,
        "final_provenance": prov_path,
    }
    missing = sorted(name for name, path in required.items() if path is None)
    suite.add(ContractResult(
        name="required_outputs_exist",
        stage=suite.stage,
        severity="error",
        passed=not missing,
        message="All required final DEM contract artifacts exist." if not missing else f"Missing final DEM artifacts: {', '.join(missing)}",
        metrics={"missing_count": len(missing)},
        artifact_paths={k: str(v) for k, v in required.items() if v is not None},
    ))
    if missing:
        return suite

    profiles = {
        "final_depth": _profile(final_path),
        "aligned_authoritative_base": _profile(auth_path),
        "support_class": _profile(support_path),
        "final_provenance": _profile(prov_path),
    }
    grid_ok = all(_same_grid(profiles["final_depth"], p) for name, p in profiles.items() if name != "final_depth")
    suite.add(ContractResult(
        name="aligned_grids",
        stage=suite.stage,
        severity="error",
        passed=grid_ok,
        message="Final depth, authoritative base, support class, and provenance align to one grid." if grid_ok else "Final DEM contract rasters are not grid-aligned.",
    ))

    final_contract = validate_written_final_dem_contract(
        final_depth=final_path,
        aligned_authoritative_base=auth_path,
        support_class=support_path,
    )
    continuous = final_contract.get("continuous_output", {}) if isinstance(final_contract.get("continuous_output"), dict) else {}
    continuous_artifacts = {}
    if debug_dir is not None and not bool(continuous.get("ok", False)):
        with rasterio.open(final_path) as ds:
            final_arr = ds.read(1)
            nonfinite_mask = ~np.isfinite(final_arr)
        pth = _write_debug_mask(debug_dir / "final_dem_nonfinite_pixels.tif", nonfinite_mask, final_path)
        if pth:
            continuous_artifacts["nonfinite_mask"] = pth
    suite.add(ContractResult(
        name="continuous_output",
        stage=suite.stage,
        severity="error",
        passed=bool(continuous.get("ok", False)),
        message="Final DEM is continuous with no non-finite pixels." if continuous.get("ok", False) else "Final DEM contains non-finite pixels.",
        metrics={"nonfinite_pixels": int(continuous.get("nonfinite_pixels") or 0)},
        artifact_paths=continuous_artifacts,
    ))

    hard_lock = final_contract.get("authoritative_hard_lock", {}) if isinstance(final_contract.get("authoritative_hard_lock"), dict) else {}
    hard_lock_artifacts = {}
    if debug_dir is not None and not bool(hard_lock.get("ok", False)):
        with rasterio.open(final_path) as ds_f, rasterio.open(auth_path) as ds_a, rasterio.open(support_path) as ds_s:
            final_arr = ds_f.read(1)
            auth_arr = ds_a.read(1)
            support_arr_dbg = ds_s.read(1)
        locked = support_arr_dbg == 1
        finite_auth = np.isfinite(auth_arr)
        mismatch = locked & finite_auth & (~np.isfinite(final_arr) | (np.abs(final_arr - auth_arr) > 0.0))
        pth = _write_debug_mask(debug_dir / "final_dem_authoritative_lock_mismatch.tif", mismatch, final_path)
        if pth:
            hard_lock_artifacts["mismatch_mask"] = pth
    suite.add(ContractResult(
        name="authoritative_hard_lock",
        stage=suite.stage,
        severity="error",
        passed=bool(hard_lock.get("ok", False)),
        message="Authoritative-locked cells match aligned authoritative base exactly." if hard_lock.get("ok", False) else "Authoritative-locked cells changed relative to aligned authoritative base.",
        metrics={
            "locked_pixels": int(hard_lock.get("locked_pixels") or 0),
            "locked_finite_authoritative_pixels": int(hard_lock.get("locked_finite_authoritative_pixels") or 0),
            "mismatch_pixels": int(hard_lock.get("mismatch_pixels") or 0),
            "max_abs_diff_m": float(hard_lock.get("max_abs_diff_m") or 0.0),
        },
        artifact_paths=hard_lock_artifacts,
    ))

    precedence = summarize_written_precedence_audit(
        final_depth=final_path,
        aligned_authoritative_base=auth_path,
        support_class=support_path,
        final_provenance=prov_path,
        guidance_influence=guidance_influence_path,
    )
    skipped = bool(precedence.get("skipped", False))
    all_ok = bool(precedence.get("all_ok", False))
    sev = "warning" if skipped else "error"
    msg = "Precedence audit confirms authoritative lock preservation and continuous gap fill."
    if skipped:
        msg = f"Precedence audit skipped: {precedence.get('skip_reason') or 'unknown'}"
    elif not all_ok:
        msg = "Precedence audit found a final-route precedence violation."
    suite.add(ContractResult(
        name="precedence_audit",
        stage=suite.stage,
        severity=sev,
        passed=all_ok if not skipped else False,
        message=msg,
        metrics={
            "skipped": skipped,
            "skip_reason": precedence.get("skip_reason"),
            "guidance_nonzero_on_locked_count": int(((precedence.get("authoritative_lock") or {}).get("guidance_nonzero_on_locked_count", 0)) or 0),
            "continuous_fill_achieved": bool(((precedence.get("gap_fill") or {}).get("continuous_fill_achieved", False))),
        },
    ))

    with rasterio.open(support_path) as ds:
        support_arr = ds.read(1)
    support_codes = sorted(int(v) for v in np.unique(support_arr))
    suite.add(ContractResult(
        name="support_class_present",
        stage=suite.stage,
        severity="warning",
        passed=bool(np.count_nonzero(support_arr) > 0),
        message="Support-class raster has nonzero classified coverage." if np.count_nonzero(support_arr) > 0 else "Support-class raster is entirely zero.",
        metrics={"support_codes": support_codes[:32], "unique_count": len(support_codes)},
    ))

    with rasterio.open(prov_path) as ds:
        prov_arr = ds.read(1).astype(np.uint8, copy=False)
    expected_prov = _expected_provenance_from_support(support_arr)
    support_prov_mismatch = (support_arr > 0) & (prov_arr > 0) & (prov_arr != expected_prov)
    support_prov_artifacts = {}
    if debug_dir is not None and np.any(support_prov_mismatch):
        pth = _write_debug_mask(debug_dir / "final_dem_support_provenance_mismatch.tif", support_prov_mismatch, final_path)
        if pth:
            support_prov_artifacts["mismatch_mask"] = pth
    suite.add(ContractResult(
        name="support_provenance_alignment",
        stage=suite.stage,
        severity="error",
        passed=not bool(np.any(support_prov_mismatch)),
        message="Support-class and provenance codes follow the final-route contract mapping." if not np.any(support_prov_mismatch) else "Support-class and provenance codes disagree with the final-route contract mapping.",
        metrics={"mismatch_pixels": int(np.count_nonzero(support_prov_mismatch))},
        artifact_paths=support_prov_artifacts,
    ))

    if guidance_influence_path is None:
        suite.add(ContractResult(
            name="guidance_influence_present",
            stage=suite.stage,
            severity="warning",
            passed=False,
            message="Guidance-influence raster is missing; influence eligibility contracts were skipped.",
        ))
        return suite

    guidance_profile = _profile(guidance_influence_path)
    guidance_grid_ok = _same_grid(profiles["final_depth"], guidance_profile)
    suite.add(ContractResult(
        name="guidance_influence_grid_alignment",
        stage=suite.stage,
        severity="error",
        passed=guidance_grid_ok,
        message="Guidance-influence raster aligns to the final DEM grid." if guidance_grid_ok else "Guidance-influence raster is not grid-aligned with the final DEM.",
    ))
    if not guidance_grid_ok:
        return suite

    with rasterio.open(guidance_influence_path) as ds:
        guidance_arr = ds.read(1).astype(np.float32, copy=False)
        nodata = ds.nodata
    if nodata is not None:
        guidance_arr[np.isclose(guidance_arr, np.float32(nodata))] = np.nan
    guidance_nonfinite = ~np.isfinite(guidance_arr)
    guidance_nonfinite_artifacts = {}
    if debug_dir is not None and np.any(guidance_nonfinite):
        pth = _write_debug_mask(debug_dir / "final_dem_guidance_influence_nonfinite.tif", guidance_nonfinite, final_path)
        if pth:
            guidance_nonfinite_artifacts["nonfinite_mask"] = pth
    suite.add(ContractResult(
        name="guidance_influence_finite",
        stage=suite.stage,
        severity="error",
        passed=not bool(np.any(guidance_nonfinite)),
        message="Guidance-influence raster contains only finite values." if not np.any(guidance_nonfinite) else "Guidance-influence raster contains non-finite pixels.",
        metrics={"nonfinite_pixels": int(np.count_nonzero(guidance_nonfinite))},
        artifact_paths=guidance_nonfinite_artifacts,
    ))

    guidance_nonzero = np.nan_to_num(guidance_arr, nan=0.0) > 1e-6
    allowed_support = _guidance_allowed_support_mask(support_arr)
    guidance_support_violation = guidance_nonzero & (~allowed_support)
    guidance_support_artifacts = {}
    if debug_dir is not None and np.any(guidance_support_violation):
        pth = _write_debug_mask(debug_dir / "final_dem_guidance_influence_outside_guidance_support.tif", guidance_support_violation, final_path)
        if pth:
            guidance_support_artifacts["violation_mask"] = pth
    suite.add(ContractResult(
        name="guidance_influence_support_eligibility",
        stage=suite.stage,
        severity="error",
        passed=not bool(np.any(guidance_support_violation)),
        message="Nonzero guidance influence is confined to guidance-eligible support classes." if not np.any(guidance_support_violation) else "Nonzero guidance influence appears outside guidance-eligible support classes.",
        metrics={"violation_pixels": int(np.count_nonzero(guidance_support_violation))},
        artifact_paths=guidance_support_artifacts,
    ))

    allowed_prov = _guidance_allowed_provenance_mask(prov_arr)
    guidance_prov_violation = guidance_nonzero & (~allowed_prov)
    guidance_prov_artifacts = {}
    if debug_dir is not None and np.any(guidance_prov_violation):
        pth = _write_debug_mask(debug_dir / "final_dem_guidance_influence_outside_guidance_provenance.tif", guidance_prov_violation, final_path)
        if pth:
            guidance_prov_artifacts["violation_mask"] = pth
    suite.add(ContractResult(
        name="guidance_influence_provenance_eligibility",
        stage=suite.stage,
        severity="error",
        passed=not bool(np.any(guidance_prov_violation)),
        message="Nonzero guidance influence is confined to guidance-derived provenance classes." if not np.any(guidance_prov_violation) else "Nonzero guidance influence appears outside guidance-derived provenance classes.",
        metrics={"violation_pixels": int(np.count_nonzero(guidance_prov_violation))},
        artifact_paths=guidance_prov_artifacts,
    ))
    return suite
