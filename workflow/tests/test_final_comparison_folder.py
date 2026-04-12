from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rasterio
from rasterio.transform import from_origin

from bathy_main import _populate_final_comparison_folder_best_effort, _write_final_output_receipt


def _write_raster(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype="float32", crs="EPSG:4326", transform=from_origin(0.0, 2.0, 1.0, 1.0), nodata=np.nan) as dst:
        dst.write(data.astype("float32"), 1)


def test_final_comparison_folder_uses_baseline_interpolated_dem_for_authoritative_named_output(tmp_path: Path):
    out_dir = tmp_path / "out"
    baseline = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    measured_only = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    enhanced = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    _write_raster(out_dir / "_external" / "cudem_baseline_interpolation.tif", baseline)
    _write_raster(out_dir / "combined" / "authoritative_base_aligned.tif", measured_only)
    _write_raster(out_dir / "combined" / "DEM_enhanced.tif", enhanced)
    cfg = SimpleNamespace(out_dir=str(out_dir))
    report = {"outputs": {"combined_warped": str(out_dir / "combined" / "DEM_enhanced.tif"), "authoritative_base_aligned": str(out_dir / "combined" / "authoritative_base_aligned.tif")}}
    _populate_final_comparison_folder_best_effort(cfg, report)
    final_auth = out_dir / "final" / "authoritative_base_aligned.tif"
    assert final_auth.exists()
    with rasterio.open(final_auth) as ds:
        arr = ds.read(1)
    assert np.allclose(arr, baseline)
    assert report["outputs"]["final_folder_authoritative_base_source"].endswith("_external/cudem_baseline_interpolation.tif")


def test_ensure_final_baseline_outputs_fills_missing_baseline_artifacts(tmp_path):
    from bathy_main import _ensure_final_baseline_outputs_best_effort
    out_dir = tmp_path
    (out_dir / "_external").mkdir(parents=True, exist_ok=True)
    baseline = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    _write_raster(out_dir / "_external" / "cudem_baseline_interpolation.tif", baseline)
    report = {"outputs": {}}
    cfg = SimpleNamespace(out_dir=out_dir)
    _ensure_final_baseline_outputs_best_effort(cfg, report)
    final_auth = out_dir / "final" / "authoritative_base_aligned.tif"
    final_hs = out_dir / "final" / "authoritative_base_aligned_hillshade.tif"
    assert final_auth.exists()
    assert final_hs.exists()
    assert report["outputs"]["final_folder_authoritative_base_source"].endswith("_external/cudem_baseline_interpolation.tif")


def test_write_final_output_receipt_creates_strict_contract_artifacts(tmp_path: Path):
    out_dir = tmp_path / "out"
    baseline = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    enhanced = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    _write_raster(out_dir / "_external" / "cudem_baseline_interpolation.tif", baseline)
    _write_raster(out_dir / "combined" / "DEM_enhanced.tif", enhanced)
    cfg = SimpleNamespace(out_dir=str(out_dir))
    report = {"outputs": {"combined_warped": str(out_dir / "combined" / "DEM_enhanced.tif")}}
    receipt = _write_final_output_receipt(cfg, report)
    assert receipt.exists()
    assert (out_dir / "final" / "authoritative_base_aligned.tif").exists()
    assert (out_dir / "final" / "authoritative_base_aligned_hillshade.tif").exists()
    assert (out_dir / "final" / "DEM_enhanced.tif").exists()
    assert (out_dir / "final" / "DEM_enhanced_hillshade.tif").exists()
    assert report["outputs"]["final_output_receipt"].endswith("final_output_receipt.json")
