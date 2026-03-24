from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

try:
    import rasterio
except Exception:  # pragma: no cover - test will skip without rasterio
    rasterio = None


pytestmark = pytest.mark.heavy


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_reference_spec() -> dict:
    spec_path = os.environ.get("MERRIMACK_REFERENCE_SPEC_JSON")
    if spec_path:
        p = Path(spec_path)
    else:
        p = _repo_root() / "tests" / "data" / "merrimack_reference_spec.json"
    return json.loads(p.read_text())


def _resolve_output_dir() -> Path:
    out_dir = os.environ.get("MERRIMACK_REFERENCE_OUTPUT_DIR")
    if not out_dir:
        pytest.skip("Set MERRIMACK_REFERENCE_OUTPUT_DIR to run the Merrimack reference test.")
    return Path(out_dir)


def _resolve_final_raster(spec: dict, *, base: Path) -> Path:
    direct = os.environ.get("MERRIMACK_REFERENCE_FINAL_RASTER")
    if direct:
        p = Path(direct)
        if p.exists():
            return p
        raise FileNotFoundError(f"MERRIMACK_REFERENCE_FINAL_RASTER does not exist: {p}")

    for rel in spec.get("final_raster_candidates", []):
        candidate = base / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve final raster beneath {base} using candidates {spec.get('final_raster_candidates', [])}"
    )


def _first_existing(base: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        cand = base / rel
        if cand.exists():
            return cand
    return None


@pytest.mark.heavy
def test_merrimack_reference_real_output_contract():
    if rasterio is None:
        pytest.skip("rasterio is required for the Merrimack reference test")

    spec = _load_reference_spec()
    base = _resolve_output_dir()
    raster_path = _resolve_final_raster(spec, base=base)
    checks = spec.get("checks", [])
    assert checks, "Reference spec must define at least one sampling check"

    with rasterio.open(raster_path) as ds:
        tags = {str(k).upper(): str(v) for k, v in ds.tags().items()}
        assert tags.get("VALUE_TYPE") in {"elevation", "bed_elevation", "bathymetric_elevation", "bottom_elevation"}, (
            f"Final Merrimack reference raster should be tagged as elevation semantics, got {tags!r}"
        )
        for check in checks:
            xy = [(float(check["x"]), float(check["y"]))]
            val = float(next(ds.sample(xy))[0])
            assert np.isfinite(val), f"{check['name']}: sampled value is not finite"
            lo = float(check["min"])
            hi = float(check["max"])
            assert lo <= val <= hi, (
                f"{check['name']}: sampled value {val} outside expected range [{lo}, {hi}] at ({check['x']}, {check['y']})"
            )

        # Sign consistency checks — catch sign-convention regressions
        for sc in spec.get("sign_checks", []):
            for pt in sc.get("points", []):
                xy = [(float(pt["x"]), float(pt["y"]))]
                val = float(next(ds.sample(xy))[0])
                assert np.isfinite(val), f"{sc['name']}: sampled value at ({pt['x']}, {pt['y']}) is not finite"
                if sc.get("expect_negative"):
                    assert val < 0.0, (
                        f"{sc['name']}: expected negative elevation at ({pt['x']}, {pt['y']}) but got {val:.3f} — "
                        f"possible sign flip. {sc.get('note', '')}"
                    )

        # Monotonicity checks — catch depth-cap and conditioning regressions
        for mc in spec.get("monotonicity_checks", []):
            up_xy = [(float(mc["upstream"]["x"]), float(mc["upstream"]["y"]))]
            dn_xy = [(float(mc["downstream"]["x"]), float(mc["downstream"]["y"]))]
            up_val = float(next(ds.sample(up_xy))[0])
            dn_val = float(next(ds.sample(dn_xy))[0])
            assert np.isfinite(up_val) and np.isfinite(dn_val), (
                f"{mc['name']}: upstream or downstream sample is not finite"
            )
            if mc.get("expect_downstream_deeper"):
                assert dn_val < up_val, (
                    f"{mc['name']}: downstream ({dn_val:.3f}) should be deeper (more negative) "
                    f"than upstream ({up_val:.3f}). {mc.get('note', '')}"
                )


    support_path = _first_existing(base, [
        "combined/support_class.tif",
        "support_class.tif",
    ])
    uncertainty_path = _first_existing(base, [
        "combined/conditioned_uncertainty.tif",
        "conditioned_uncertainty.tif",
    ])
    precedence_audit = _first_existing(base, [
        "combined/authoritative_precedence_audit.json",
        "authoritative_precedence_audit.json",
    ])
    conditioning_audit = _first_existing(base, [
        "combined/conditioning_audit.json",
        "conditioning_audit.json",
    ])
    contract_path = _first_existing(base, [
        "combined/guidance_uncertainty_contract.json",
        "guidance_uncertainty_contract.json",
        "combined/final_output_layer_contract.json",
        "final_output_layer_contract.json",
    ])
    vertical_contract_path = _first_existing(base, [
        "combined/final_vertical_semantics_contract.json",
        "final_vertical_semantics_contract.json",
    ])

    assert support_path is not None, "Expected support_class.tif in completed Merrimack output"
    assert uncertainty_path is not None, "Expected conditioned_uncertainty.tif in completed Merrimack output"
    assert precedence_audit is not None, "Expected authoritative_precedence_audit.json in completed Merrimack output"
    assert conditioning_audit is not None, "Expected conditioning_audit.json in completed Merrimack output"
    assert contract_path is not None, "Expected guidance_uncertainty_contract.json in completed Merrimack output"
    assert vertical_contract_path is not None, "Expected final_vertical_semantics_contract.json in completed Merrimack output"

    with rasterio.open(support_path) as ds:
        support_sample = int(next(ds.sample([(float(checks[0]["x"]), float(checks[0]["y"]))]))[0])
        assert support_sample >= 0

    with rasterio.open(uncertainty_path) as ds:
        unc_val = float(next(ds.sample([(float(checks[0]["x"]), float(checks[0]["y"]))]))[0])
        assert np.isfinite(unc_val), "Sampled conditioned uncertainty must be finite"
        assert unc_val >= 0.0, "Conditioned uncertainty should be non-negative"

    precedence = json.loads(precedence_audit.read_text())
    conditioning = json.loads(conditioning_audit.read_text())
    contract = json.loads(contract_path.read_text())
    assert precedence["authoritative_lock"]["lock_preserved"] is True
    assert conditioning["authoritative_lock"]["lock_preserved"] is True
    assert conditioning["gap_fill"]["continuous_fill_achieved"] is True
    assert contract["ok"] is True
