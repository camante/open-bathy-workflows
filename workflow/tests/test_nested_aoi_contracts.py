from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from nested_aoi_contracts import evaluate_nested_aoi_contracts
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
        dtype=str(arr.dtype),
        crs=crs,
        transform=transform,
        nodata=255 if arr.dtype.kind in ("i", "u") else -9999.0,
    ) as ds:
        ds.write(arr, 1)
    return path


def test_contract_eval_flags_missing_required_artifact(tmp_path: Path) -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    support = np.array([[1, 4], [4, 4]], dtype=np.uint8)
    trusted = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    current_depth = _write_tif(tmp_path / "current_depth.tif", arr)
    current_support = _write_tif(tmp_path / "current_support.tif", support)
    current_trusted = _write_tif(tmp_path / "current_trusted.tif", trusted)
    neighbor_depth = _write_tif(tmp_path / "neighbor_depth.tif", arr)
    neighbor_support = _write_tif(tmp_path / "neighbor_support.tif", support)
    neighbor_trusted = _write_tif(tmp_path / "neighbor_trusted.tif", trusted)

    current_manifest = tmp_path / "current_final_outputs.json"
    neighbor_manifest = tmp_path / "neighbor_final_outputs.json"
    current_manifest.write_text(json.dumps({
        "aoi": "current",
        "selected_final_depth": str(current_depth),
        "support_class": str(current_support),
        "final_provenance_native": str(current_depth),
        "river_trusted_interior": str(current_trusted),
    }), encoding="utf-8")
    neighbor_manifest.write_text(json.dumps({
        "aoi": "neighbor",
        "selected_final_depth": str(neighbor_depth),
        "support_class": str(neighbor_support),
        "river_trusted_interior": str(neighbor_trusted),
    }), encoding="utf-8")

    payload = run_nested_aoi_regression(
        current_final_outputs_manifest=current_manifest,
        neighbor_final_outputs_manifests=[neighbor_manifest],
    )
    contracts = evaluate_nested_aoi_contracts(
        current_final_outputs_manifest=current_manifest,
        nested_payload=payload,
        overlap_tolerance=1.0e-6,
        trusted_tolerance=1.0e-6,
    )

    assert contracts["all_required_overlap_artifacts_present"] is False
    assert contracts["all_overlap_contracts_ok"] is False
    assert contracts["comparisons"][0]["overlap_status"] == "missing_required_artifacts"


def test_contract_eval_reports_first_class_failure(tmp_path: Path) -> None:
    current_depth_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    neighbor_depth_arr = np.array([[1.0, 2.5], [3.0, 4.0]], dtype=np.float32)
    support = np.array([[1, 4], [4, 4]], dtype=np.uint8)
    trusted = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    current_depth = _write_tif(tmp_path / "current_depth.tif", current_depth_arr)
    current_support = _write_tif(tmp_path / "current_support.tif", support)
    current_trusted = _write_tif(tmp_path / "current_trusted.tif", trusted)
    current_prov = _write_tif(tmp_path / "current_prov.tif", support)
    neighbor_depth = _write_tif(tmp_path / "neighbor_depth.tif", neighbor_depth_arr)
    neighbor_support = _write_tif(tmp_path / "neighbor_support.tif", support)
    neighbor_trusted = _write_tif(tmp_path / "neighbor_trusted.tif", trusted)
    neighbor_prov = _write_tif(tmp_path / "neighbor_prov.tif", support)

    current_manifest = tmp_path / "current_final_outputs.json"
    neighbor_manifest = tmp_path / "neighbor_final_outputs.json"
    current_manifest.write_text(json.dumps({
        "aoi": "current",
        "selected_final_depth": str(current_depth),
        "support_class": str(current_support),
        "final_provenance_native": str(current_prov),
        "river_trusted_interior": str(current_trusted),
    }), encoding="utf-8")
    neighbor_manifest.write_text(json.dumps({
        "aoi": "neighbor",
        "selected_final_depth": str(neighbor_depth),
        "support_class": str(neighbor_support),
        "final_provenance_native": str(neighbor_prov),
        "river_trusted_interior": str(neighbor_trusted),
    }), encoding="utf-8")

    payload = run_nested_aoi_regression(
        current_final_outputs_manifest=current_manifest,
        neighbor_final_outputs_manifests=[neighbor_manifest],
        overlap_tolerance=1.0e-6,
        trusted_tolerance=1.0e-6,
    )
    contracts = evaluate_nested_aoi_contracts(
        current_final_outputs_manifest=current_manifest,
        nested_payload=payload,
        overlap_tolerance=1.0e-6,
        trusted_tolerance=1.0e-6,
    )

    assert contracts["all_required_overlap_artifacts_present"] is True
    assert contracts["all_overlap_contracts_ok"] is False
    failures = contracts["first_overlap_class_failures"]
    assert failures
    assert failures[0]["support_class_code"] == 4
    assert failures[0]["support_class_name"] == "guidance_conditioned_river"
