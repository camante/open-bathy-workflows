from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from nested_aoi_regression import run_nested_aoi_regression


def _write_tif(path: Path, arr: np.ndarray, *, crs: str = "EPSG:4326") -> Path:
    transform = from_origin(0.0, float(arr.shape[0]), 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as ds:
        ds.write(arr.astype("float32"), 1)
    return path


def test_run_nested_aoi_regression_writes_overlap_and_trusted_metrics(tmp_path: Path) -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    trusted = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    current_depth = _write_tif(tmp_path / "current_depth.tif", arr)
    current_support = _write_tif(tmp_path / "current_support.tif", arr + 10.0)
    current_trusted = _write_tif(tmp_path / "current_trusted.tif", trusted)
    current_base = _write_tif(tmp_path / "current_base.tif", arr + 20.0)
    current_recon = _write_tif(tmp_path / "current_recon.tif", arr + 21.0)
    current_recon_delta = _write_tif(tmp_path / "current_recon_delta.tif", arr + 22.0)
    current_recon_inf = _write_tif(tmp_path / "current_recon_inf.tif", arr + 23.0)
    neighbor_depth = _write_tif(tmp_path / "neighbor_depth.tif", arr)
    neighbor_support = _write_tif(tmp_path / "neighbor_support.tif", arr + 10.0)
    neighbor_trusted = _write_tif(tmp_path / "neighbor_trusted.tif", trusted)
    neighbor_base = _write_tif(tmp_path / "neighbor_base.tif", arr + 20.0)
    neighbor_recon = _write_tif(tmp_path / "neighbor_recon.tif", arr + 21.0)
    neighbor_recon_delta = _write_tif(tmp_path / "neighbor_recon_delta.tif", arr + 22.0)
    neighbor_recon_inf = _write_tif(tmp_path / "neighbor_recon_inf.tif", arr + 23.0)

    current_manifest = tmp_path / "current_final_outputs.json"
    neighbor_manifest = tmp_path / "neighbor_final_outputs.json"
    current_manifest.write_text(json.dumps({
        "aoi": "small",
        "selected_final_depth": str(current_depth),
        "support_class": str(current_support),
        "river_trusted_interior": str(current_trusted),
        "river_generalized_longitudinal_bed_base": str(current_base),
        "river_generalized_longitudinal_bed_reconciled": str(current_recon),
        "river_longitudinal_profile_local_authoritative_reconciliation": str(current_recon_delta),
        "river_longitudinal_profile_local_authoritative_reconciliation_influence": str(current_recon_inf),
    }), encoding="utf-8")
    neighbor_manifest.write_text(json.dumps({
        "aoi": "large",
        "selected_final_depth": str(neighbor_depth),
        "support_class": str(neighbor_support),
        "river_trusted_interior": str(neighbor_trusted),
        "river_generalized_longitudinal_bed_base": str(neighbor_base),
        "river_generalized_longitudinal_bed_reconciled": str(neighbor_recon),
        "river_longitudinal_profile_local_authoritative_reconciliation": str(neighbor_recon_delta),
        "river_longitudinal_profile_local_authoritative_reconciliation_influence": str(neighbor_recon_inf),
    }), encoding="utf-8")

    payload = run_nested_aoi_regression(
        current_final_outputs_manifest=current_manifest,
        neighbor_final_outputs_manifests=[neighbor_manifest],
    )

    assert payload["neighbor_count"] == 1
    assert payload["all_overlap_identity_ok"] is True
    assert payload["all_trusted_interior_identity_ok"] is True
    artifacts = {item["artifact"] for item in payload["overlap_identity_checks"]}
    assert "selected_final_depth" in artifacts
    assert "support_class" in artifacts
    assert "river_generalized_longitudinal_bed_reconciled" in artifacts
    assert "river_longitudinal_profile_local_authoritative_reconciliation" in artifacts
    trusted_artifacts = {item["artifact"] for item in payload["trusted_interior_identity_checks"]}
    assert "trusted_interior::selected_final_depth" in trusted_artifacts
    assert "trusted_interior::river_generalized_longitudinal_bed_reconciled" in trusted_artifacts
