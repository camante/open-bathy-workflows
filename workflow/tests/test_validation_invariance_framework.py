import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from final_reporting import write_explicit_final_outputs_manifest, write_validation_invariance_summary
from support_classes import SupportClass
from provenance_schema import ProvenanceClass
from validation_invariance_framework import (
    evaluate_authoritative_lock_invariant,
    evaluate_guidance_non_degradation,
    run_validation_invariance_framework,
)

def _rasterio_available():
    try:
        import rasterio  # noqa: F401
        from rasterio.transform import from_origin as _from_origin  # noqa: F401
        return True
    except Exception:
        return False


def _from_origin(*args, **kwargs):
    from rasterio.transform import from_origin
    return from_origin(*args, **kwargs)


def _write_tif(path: Path, arr: np.ndarray, *, nodata: float = -9999.0):
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": _from_origin(-71.0, 43.0, 1.0, 1.0),
        "nodata": nodata,
        "compress": "deflate",
    }
    out = arr.astype(np.float32).copy()
    out[~np.isfinite(out)] = np.float32(nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    return path


def test_authoritative_lock_invariant_detects_mismatch():
    pred = np.array([[1.0, 2.0], [3.0, 4.1]], dtype=np.float32)
    auth = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    support = np.array([
        [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
        [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.AUTHORITATIVE_LOCKED)],
    ], dtype=np.uint8)
    out = evaluate_authoritative_lock_invariant(pred=pred, authoritative=auth, support_class=support, tolerance=1e-6)
    assert out["ok"] is False
    assert out["checked"] == 3
    assert out["max_abs"] > 0.09


def test_guidance_non_degradation_reports_family_failure():
    out = evaluate_guidance_non_degradation(
        ablation={
            "cases": {
                "baseline": {"support_metrics": {"by_family": {"guidance_conditioned": {"rmse": 1.0}}}},
                "target": {"support_metrics": {"by_family": {"guidance_conditioned": {"rmse": 1.25}}}},
            }
        },
        baseline_case="baseline",
        target_case="target",
        rmse_tolerance=0.1,
    )
    assert out["ok"] is False
    assert out["failures"][0]["family"] == "guidance_conditioned"


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio required")
def test_run_validation_framework_writes_metrics_and_fails_on_authoritative_violation():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        truth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        final = np.array([[1.0, 2.0], [3.0, 4.1]], dtype=np.float32)
        baseline = np.array([[1.1, 2.2], [3.1, 4.2]], dtype=np.float32)
        auth = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        support = np.array([
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.AUTHORITATIVE_LOCKED)],
        ], dtype=np.float32)
        prov = np.array([
            [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.SDB_CONDITIONED_FILL)],
            [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.AUTHORITATIVE_LOCKED)],
        ], dtype=np.float32)

        final_p = _write_tif(td / "final.tif", final)
        baseline_p = _write_tif(td / "baseline.tif", baseline)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        prov_p = _write_tif(td / "prov.tif", prov)
        truth_p = _write_tif(td / "truth.tif", truth)

        cfg = SimpleNamespace(
            out_dir=td,
            authoritative_base=auth_p,
            aoi="-71/-69/41/43",
            tile_bbox=None,
        )
        report = {
            "authoritative_base": {"outputs": {"support_class": str(support_p)}},
            "outputs": {},
        }
        write_explicit_final_outputs_manifest(cfg, report, final_native=final_p, final_for_user=None, final_provenance=prov_p)
        payload = run_validation_invariance_framework(
            final_outputs_manifest=td / "final_outputs.json",
            validation_truth=truth_p,
            case_specs=[f"baseline={baseline_p}"],
            guidance_baseline_case="baseline",
            guidance_target_case="selected_final",
        )
        assert payload["selected_final_support_metrics"]["overall"]["count"] == 4
        assert payload["authoritative_lock_invariant"]["ok"] is False
        assert payload["all_hard_invariants_ok"] is False
        assert payload["ablation_results"]["cases"]["baseline"]["support_metrics"]["overall"]["count"] == 4


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio required")
def test_write_validation_summary_raises_on_hard_invariant_failure():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        auth = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
        final = np.array([[1.2, 2.0], [3.0, 4.0]], dtype=np.float32)
        support = np.array([
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
            [int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.GUIDANCE_CONDITIONED_RIVER)],
        ], dtype=np.float32)
        prov = np.array([
            [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.SDB_CONDITIONED_FILL)],
            [int(ProvenanceClass.RIVER_CONDITIONED_FILL), int(ProvenanceClass.RIVER_CONDITIONED_FILL)],
        ], dtype=np.float32)
        final_p = _write_tif(td / "final.tif", final)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        prov_p = _write_tif(td / "prov.tif", prov)

        cfg = SimpleNamespace(
            out_dir=td,
            authoritative_base=auth_p,
            aoi="-71/-69/41/43",
            tile_bbox=None,
            validation_truth=None,
            validation_case_specs=[],
            validation_case_manifest=None,
            validation_guidance_baseline_case="baseline_cudem_interpolation",
            validation_guidance_target_case="selected_final",
            validation_require_guidance_non_degradation=False,
            validation_guidance_rmse_tolerance=0.0,
        )
        report = {
            "authoritative_base": {"outputs": {"support_class": str(support_p)}},
            "outputs": {},
            "seams": {},
        }
        write_explicit_final_outputs_manifest(cfg, report, final_native=final_p, final_for_user=None, final_provenance=prov_p)
        with pytest.raises(RuntimeError):
            write_validation_invariance_summary(cfg, report, final_native=final_p, final_for_user=None, final_provenance=prov_p)
        payload = json.loads((td / "validation_invariance_summary.json").read_text())
        assert payload["authoritative_lock_invariant"]["ok"] is False


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio required")
def test_validation_uses_native_final_for_authoritative_invariant_when_user_delivery_exists():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        auth = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
        final_native = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        final_user = np.array([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32)
        support = np.array([
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
            [int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.GUIDANCE_CONDITIONED_RIVER)],
        ], dtype=np.float32)
        prov = np.array([
            [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.SDB_CONDITIONED_FILL)],
            [int(ProvenanceClass.RIVER_CONDITIONED_FILL), int(ProvenanceClass.RIVER_CONDITIONED_FILL)],
        ], dtype=np.float32)
        final_native_p = _write_tif(td / "final_native.tif", final_native)
        final_user_p = _write_tif(td / "final_user.tif", final_user)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        prov_p = _write_tif(td / "prov.tif", prov)

        cfg = SimpleNamespace(
            out_dir=td,
            authoritative_base=auth_p,
            aoi="-71/-69/41/43",
            tile_bbox=None,
        )
        report = {
            "authoritative_base": {"outputs": {"support_class": str(support_p), "aligned_authoritative_base": str(auth_p)}},
            "outputs": {},
        }
        write_explicit_final_outputs_manifest(cfg, report, final_native=final_native_p, final_for_user=final_user_p, final_provenance=prov_p)
        payload = run_validation_invariance_framework(final_outputs_manifest=td / "final_outputs.json")
        assert payload["selected_final_depth"] == str(final_user_p)
        assert payload["invariant_evaluation_depth"] == str(final_native_p)
        assert payload["authoritative_lock_invariant"]["ok"] is True



@pytest.mark.skipif(not _rasterio_available(), reason="rasterio required")
def test_validation_prefers_selected_final_invariant_when_present():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        auth = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
        final_native = np.array([[99.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        comparison = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        final_user = np.array([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32)
        support = np.array([
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
            [int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.GUIDANCE_CONDITIONED_RIVER)],
        ], dtype=np.float32)
        prov = np.array([
            [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.SDB_CONDITIONED_FILL)],
            [int(ProvenanceClass.RIVER_CONDITIONED_FILL), int(ProvenanceClass.RIVER_CONDITIONED_FILL)],
        ], dtype=np.float32)
        final_native_p = _write_tif(td / "final_native.tif", final_native)
        comparison_p = _write_tif(td / "comparison.tif", comparison)
        final_user_p = _write_tif(td / "final_user.tif", final_user)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        prov_p = _write_tif(td / "prov.tif", prov)

        cfg = SimpleNamespace(out_dir=td, authoritative_base=auth_p, aoi="-71/-69/41/43", tile_bbox=None)
        report = {"authoritative_base": {"outputs": {"support_class": str(support_p), "aligned_authoritative_base": str(auth_p)}}, "outputs": {}}
        write_explicit_final_outputs_manifest(cfg, report, final_native=final_native_p, final_for_user=final_user_p, final_provenance=prov_p)
        payload = json.loads((td / "final_outputs.json").read_text())
        payload["selected_final_invariant"] = str(comparison_p)
        (td / "final_outputs.json").write_text(json.dumps(payload), encoding="utf-8")
        result = run_validation_invariance_framework(final_outputs_manifest=td / "final_outputs.json")
        assert result["selected_final_depth"] == str(final_user_p)
        assert result["invariant_evaluation_depth"] == str(comparison_p)
        assert result["authoritative_lock_invariant"]["ok"] is True


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio required")
def test_write_validation_summary_skips_when_no_selected_final_depth_exists():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = SimpleNamespace(
            out_dir=td,
            authoritative_base=None,
            aoi="-71/-69/41/43",
            tile_bbox=None,
            validation_truth=None,
            validation_case_specs=[],
            validation_case_manifest=None,
            validation_guidance_baseline_case="baseline_cudem_interpolation",
            validation_guidance_target_case="selected_final",
            validation_require_guidance_non_degradation=False,
            validation_guidance_rmse_tolerance=0.0,
        )
        report = {"outputs": {}, "seams": {}}
        write_explicit_final_outputs_manifest(cfg, report, final_native=None, final_for_user=None, final_provenance=None)
        out = write_validation_invariance_summary(cfg, report, final_native=None, final_for_user=None, final_provenance=None)
        assert out is None
        assert report["validation"]["status"] == "skipped_no_selected_final_depth"


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio required")
def test_validation_uses_nested_precomputed_lock_validation_receipt():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        auth = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
        final_native = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        support = np.array([
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_RIVER)],
            [int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.GUIDANCE_CONDITIONED_RIVER)],
        ], dtype=np.float32)
        final_native_p = _write_tif(td / "final_native.tif", final_native)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        receipt = td / "final_route_authoritative_lock_validation.json"
        receipt.write_text(json.dumps({
            "artifacts": {"conditioned_depth": str(final_native_p)},
            "written_artifact_validation": {
                "validated": True,
                "skipped": False,
                "authoritative_hard_lock": {
                    "ok": True,
                    "mismatch_pixels": 0,
                    "max_abs_diff_m": 0.0,
                    "locked_finite_authoritative_pixels": 1,
                },
            },
        }), encoding="utf-8")
        final_outputs = {
            "selected_final_depth": str(final_native_p),
            "selected_final_invariant": str(final_native_p),
            "selected_final_invariant_lock_validation": str(receipt),
            "selected_final_invariant_lock_validation_target": str(final_native_p),
            "conditioned_authoritative_base": str(auth_p),
            "support_class": str(support_p),
        }
        (td / "final_outputs.json").write_text(json.dumps(final_outputs), encoding="utf-8")
        payload = run_validation_invariance_framework(final_outputs_manifest=td / "final_outputs.json")
        assert payload["authoritative_lock_invariant"]["ok"] is True
        assert payload["authoritative_lock_invariant"]["source"] == "final_dem_contract_validator"
