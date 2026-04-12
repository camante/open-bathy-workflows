import tempfile
from pathlib import Path

import numpy as np
import pytest

from final_dem_contract_validator import validate_written_final_dem_contract

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin


def _write_tif(path: Path, arr: np.ndarray, *, nodata=-9999.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = "float32" if np.issubdtype(arr.dtype, np.floating) else "uint8"
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": from_origin(-71.0, 43.0, 1.0, 1.0),
        "nodata": nodata if np.issubdtype(arr.dtype, np.floating) else 0,
        "compress": "deflate",
    }
    out = arr.copy()
    if np.issubdtype(out.dtype, np.floating):
        out = out.astype(np.float32)
        out[~np.isfinite(out)] = np.float32(nodata)
    else:
        out = out.astype(np.uint8)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(out, 1)
    return path


def test_validate_written_final_dem_contract_accepts_continuous_authoritative_locked_output():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        final = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        auth = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
        support = np.array([[1, 2], [2, 1]], dtype=np.uint8)
        final_p = _write_tif(td / "final.tif", final)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        result = validate_written_final_dem_contract(
            final_depth=final_p,
            aligned_authoritative_base=auth_p,
            support_class=support_p,
        )
        assert result["validated"] is True
        assert result["all_ok"] is True
        assert result["continuous_output"]["ok"] is True
        assert result["authoritative_hard_lock"]["ok"] is True
        assert result["authoritative_hard_lock"]["locked_finite_authoritative_pixels"] == 2


def test_validate_written_final_dem_contract_flags_nonfinite_and_locked_mismatch():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        final = np.array([[1.5, 2.0], [np.nan, 4.0]], dtype=np.float32)
        auth = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
        support = np.array([[1, 2], [2, 1]], dtype=np.uint8)
        final_p = _write_tif(td / "final.tif", final)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        result = validate_written_final_dem_contract(
            final_depth=final_p,
            aligned_authoritative_base=auth_p,
            support_class=support_p,
        )
        assert result["validated"] is True
        assert result["all_ok"] is False
        assert result["continuous_output"]["nonfinite_pixels"] == 1
        assert result["authoritative_hard_lock"]["ok"] is False
        assert result["authoritative_hard_lock"]["mismatch_pixels"] == 1
        assert result["authoritative_hard_lock"]["max_abs_diff_m"] == pytest.approx(0.5)


def test_summarize_written_precedence_audit_accepts_locked_continuous_output():
    from final_dem_contract_validator import summarize_written_precedence_audit

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        final = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        auth = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
        support = np.array([[1, 3], [6, 1]], dtype=np.uint8)
        prov = np.array([[10, 30], [60, 10]], dtype=np.uint8)
        guidance = np.array([[0.0, 0.5], [0.25, 0.0]], dtype=np.float32)
        final_p = _write_tif(td / "final.tif", final)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        prov_p = _write_tif(td / "prov.tif", prov)
        guidance_p = _write_tif(td / "guidance.tif", guidance)

        audit = summarize_written_precedence_audit(
            final_depth=final_p,
            aligned_authoritative_base=auth_p,
            support_class=support_p,
            final_provenance=prov_p,
            guidance_influence=guidance_p,
        )
        assert audit["validated"] is True
        assert audit["all_ok"] is True
        assert audit["authoritative_lock"]["changed_locked_cell_count"] == 0
        assert audit["authoritative_lock"]["guidance_nonzero_on_locked_count"] == 0
        assert audit["gap_fill"]["continuous_fill_achieved"] is True


def test_summarize_written_precedence_audit_flags_locked_change_and_guidance_on_locked():
    from final_dem_contract_validator import summarize_written_precedence_audit

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        final = np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float32)
        auth = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
        support = np.array([[1, 3], [6, 1]], dtype=np.uint8)
        prov = np.array([[10, 30], [60, 10]], dtype=np.uint8)
        guidance = np.array([[0.1, 0.5], [0.25, 0.0]], dtype=np.float32)
        final_p = _write_tif(td / "final.tif", final)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        prov_p = _write_tif(td / "prov.tif", prov)
        guidance_p = _write_tif(td / "guidance.tif", guidance)

        audit = summarize_written_precedence_audit(
            final_depth=final_p,
            aligned_authoritative_base=auth_p,
            support_class=support_p,
            final_provenance=prov_p,
            guidance_influence=guidance_p,
        )
        assert audit["validated"] is True
        assert audit["all_ok"] is False
        assert audit["authoritative_lock"]["changed_locked_cell_count"] == 1
        assert audit["authoritative_lock"]["guidance_nonzero_on_locked_count"] == 1
        assert "Changed locked authoritative cells" in audit["validation_error"]


def test_validate_written_final_dem_lineage_accepts_v2_locked_surface(tmp_path):
    from final_dem_contract_validator import validate_written_final_dem_lineage

    final_p = _write_tif(tmp_path / "final.tif", np.array([[1.0]], dtype=np.float32))
    locked = tmp_path / "river_primary_surface_authoritative_applied.tif"
    locked.write_text("x", encoding="utf-8")
    report = {
        "river": {
            "outputs": {
                "primary_river_guidance_surface": str(locked),
                "river_primary_surface_authoritative_applied": str(locked),
            },
        },
        "final_dem_contract": {
            "final_route_contract": {
                "river_v2_final_route_contract": {
                    "active": True,
                    "active_stage": "river_primary_surface_authoritative_applied",
                    "active_river_guidance_surface": str(locked),
                    "legacy_river_final_route_participation_blocked": True,
                    "runtime_enforced": True,
                }
            }
        },
    }
    payload = validate_written_final_dem_lineage(report=report, final_depth=final_p)
    assert payload["validated"] is True
    assert payload["all_ok"] is True


def test_validate_written_final_dem_lineage_flags_mismatch(tmp_path):
    from final_dem_contract_validator import validate_written_final_dem_lineage

    final_p = _write_tif(tmp_path / "final.tif", np.array([[1.0]], dtype=np.float32))
    locked = tmp_path / "river_primary_surface_authoritative_applied.tif"
    locked.write_text("x", encoding="utf-8")
    other = tmp_path / "legacy_surface.tif"
    other.write_text("x", encoding="utf-8")
    report = {
        "river": {
            "outputs": {
                "primary_river_guidance_surface": str(other),
                "river_primary_surface_authoritative_applied": str(locked),
            },
        },
        "final_dem_contract": {
            "final_route_contract": {
                "river_v2_final_route_contract": {
                    "active": True,
                    "active_stage": "river_primary_surface_authoritative_applied",
                    "active_river_guidance_surface": str(locked),
                    "legacy_river_final_route_participation_blocked": True,
                    "runtime_enforced": True,
                }
            }
        },
    }
    payload = validate_written_final_dem_lineage(report=report, final_depth=final_p)
    assert payload["validated"] is True
    assert payload["all_ok"] is False
    assert payload["primary_surface_matches_active"] is False
