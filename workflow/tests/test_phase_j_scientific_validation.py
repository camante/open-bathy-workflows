import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from final_reporting import write_explicit_final_outputs_manifest, write_validation_invariance_summary
from provenance_schema import ProvenanceClass
from scientific_validation_stage import run_scientific_validation_stage
from support_classes import SupportClass


def _rasterio_available():
    try:
        import rasterio  # noqa: F401
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


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio required")
def test_phase_j_scientific_validation_reports_baseline_deltas_and_seams():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        truth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        final = np.array([[1.0, 2.1], [2.8, 4.0]], dtype=np.float32)
        baseline = np.array([[1.2, 2.4], [3.4, 4.2]], dtype=np.float32)
        auth = np.array([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
        support = np.array([
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
            [int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)],
        ], dtype=np.float32)
        prov = np.array([
            [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.SDB_CONDITIONED_FILL)],
            [int(ProvenanceClass.RIVER_CONDITIONED_FILL), int(ProvenanceClass.LOW_CONFIDENCE_FILL)],
        ], dtype=np.float32)

        final_p = _write_tif(td / "final.tif", final)
        baseline_p = _write_tif(td / "baseline.tif", baseline)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        prov_p = _write_tif(td / "prov.tif", prov)
        truth_p = _write_tif(td / "truth.tif", truth)

        cfg = SimpleNamespace(out_dir=td, authoritative_base=auth_p, aoi="-71/-69/41/43", tile_bbox=None)
        report = {"authoritative_base": {"outputs": {"support_class": str(support_p), "aligned_authoritative_base": str(auth_p)}}, "outputs": {}}
        write_explicit_final_outputs_manifest(cfg, report, final_native=final_p, final_for_user=None, final_provenance=prov_p)
        payload = run_scientific_validation_stage(
            final_outputs_manifest=td / "final_outputs.json",
            overlap_identity_evaluation={"all_ok": True, "checked": 2, "failures": []},
            validation_truth=truth_p,
            case_specs=[f"baseline_cudem_interpolation={baseline_p}"],
        )
        assert payload["summary_flags"]["has_support_class_metrics"] is True
        assert payload["summary_flags"]["has_baseline_comparison"] is True
        assert payload["seam_and_nested_aoi"]["all_ok"] is True
        assert payload["baseline_comparison"]["support_metrics"]["overall"]["delta_rmse"] < 0.0


@pytest.mark.skipif(not _rasterio_available(), reason="rasterio required")
def test_final_reporting_writes_scientific_validation_summary():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        truth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        final = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        auth = np.array([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
        support = np.array([
            [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
            [int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.GUIDANCE_CONDITIONED_RIVER)],
        ], dtype=np.float32)
        prov = np.array([
            [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.SDB_CONDITIONED_FILL)],
            [int(ProvenanceClass.RIVER_CONDITIONED_FILL), int(ProvenanceClass.RIVER_CONDITIONED_FILL)],
        ], dtype=np.float32)
        baseline = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float32)
        final_p = _write_tif(td / "final.tif", final)
        auth_p = _write_tif(td / "auth.tif", auth)
        support_p = _write_tif(td / "support.tif", support)
        prov_p = _write_tif(td / "prov.tif", prov)
        truth_p = _write_tif(td / "truth.tif", truth)
        baseline_p = _write_tif(td / "baseline.tif", baseline)
        cfg = SimpleNamespace(
            out_dir=td,
            authoritative_base=auth_p,
            aoi="-71/-69/41/43",
            tile_bbox=None,
            validation_truth=truth_p,
            validation_case_specs=[f"baseline_cudem_interpolation={baseline_p}"],
            validation_case_manifest=None,
            validation_guidance_baseline_case="baseline_cudem_interpolation",
            validation_guidance_target_case="selected_final",
            validation_require_guidance_non_degradation=False,
            validation_guidance_rmse_tolerance=0.0,
        )
        report = {
            "authoritative_base": {"outputs": {"support_class": str(support_p), "aligned_authoritative_base": str(auth_p)}},
            "outputs": {},
            "seams": {"overlap_identity_evaluation": {"all_ok": True, "checked": 1, "failures": []}},
        }
        write_explicit_final_outputs_manifest(cfg, report, final_native=final_p, final_for_user=None, final_provenance=prov_p)
        write_validation_invariance_summary(cfg, report, final_native=final_p, final_for_user=None, final_provenance=prov_p, enforce_hard_fail=True)
        sci = td / "scientific_validation_summary.json"
        assert sci.exists()
        payload = json.loads(sci.read_text())
        assert payload["summary_flags"]["has_support_class_metrics"] is True
        assert report["outputs"]["scientific_validation_summary"] == str(sci)
