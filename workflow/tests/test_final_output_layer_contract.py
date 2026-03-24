from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from final_output_layer_contract import write_final_output_layer_contract


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float = -9999.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=str(arr.dtype),
        crs="EPSG:4326",
        transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)
    return str(path)


def test_final_output_layer_contract_writes_when_grid_and_semantics_are_valid(tmp_path: Path):
    arr = np.ones((3, 3), dtype=np.float32)
    outputs = {
        "conditioned_depth": _write_raster(tmp_path / "conditioned_depth.tif", arr),
        "conditioned_provenance": _write_raster(tmp_path / "conditioned_provenance.tif", np.ones((3, 3), dtype=np.uint8), nodata=0),
        "support_class": _write_raster(tmp_path / "support_class.tif", np.ones((3, 3), dtype=np.uint8), nodata=0),
        "support_distance": _write_raster(tmp_path / "support_distance.tif", arr),
        "guidance_influence": _write_raster(tmp_path / "guidance_influence.tif", np.full((3, 3), 0.5, dtype=np.float32)),
        "anchor_uncertainty": _write_raster(tmp_path / "anchor_uncertainty.tif", arr),
        "guidance_uncertainty": _write_raster(tmp_path / "guidance_uncertainty.tif", arr),
        "conditioned_uncertainty": _write_raster(tmp_path / "conditioned_uncertainty.tif", arr),
    }
    out = write_final_output_layer_contract(tmp_path / "layer_contract.json", outputs=outputs)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["grid_checks"]["support_distance"]["shape_ok"] is True


def test_final_output_layer_contract_fails_on_grid_mismatch(tmp_path: Path):
    outputs = {
        "conditioned_depth": _write_raster(tmp_path / "conditioned_depth.tif", np.ones((3, 3), dtype=np.float32)),
        "conditioned_provenance": _write_raster(tmp_path / "conditioned_provenance.tif", np.ones((3, 3), dtype=np.uint8), nodata=0),
        "support_class": _write_raster(tmp_path / "support_class.tif", np.ones((3, 3), dtype=np.uint8), nodata=0),
        "support_distance": _write_raster(tmp_path / "support_distance.tif", np.ones((4, 4), dtype=np.float32)),
        "guidance_influence": _write_raster(tmp_path / "guidance_influence.tif", np.full((3, 3), 0.5, dtype=np.float32)),
        "anchor_uncertainty": _write_raster(tmp_path / "anchor_uncertainty.tif", np.ones((3, 3), dtype=np.float32)),
        "guidance_uncertainty": _write_raster(tmp_path / "guidance_uncertainty.tif", np.ones((3, 3), dtype=np.float32)),
        "conditioned_uncertainty": _write_raster(tmp_path / "conditioned_uncertainty.tif", np.ones((3, 3), dtype=np.float32)),
    }
    with pytest.raises(ValueError, match="support_distance_grid_mismatch"):
        write_final_output_layer_contract(tmp_path / "layer_contract.json", outputs=outputs)


def test_final_output_layer_contract_fails_on_negative_uncertainty(tmp_path: Path):
    outputs = {
        "conditioned_depth": _write_raster(tmp_path / "conditioned_depth.tif", np.ones((3, 3), dtype=np.float32)),
        "conditioned_provenance": _write_raster(tmp_path / "conditioned_provenance.tif", np.ones((3, 3), dtype=np.uint8), nodata=0),
        "support_class": _write_raster(tmp_path / "support_class.tif", np.ones((3, 3), dtype=np.uint8), nodata=0),
        "support_distance": _write_raster(tmp_path / "support_distance.tif", np.ones((3, 3), dtype=np.float32)),
        "guidance_influence": _write_raster(tmp_path / "guidance_influence.tif", np.full((3, 3), 0.5, dtype=np.float32)),
        "anchor_uncertainty": _write_raster(tmp_path / "anchor_uncertainty.tif", np.ones((3, 3), dtype=np.float32)),
        "guidance_uncertainty": _write_raster(tmp_path / "guidance_uncertainty.tif", np.ones((3, 3), dtype=np.float32)),
        "conditioned_uncertainty": _write_raster(tmp_path / "conditioned_uncertainty.tif", -np.ones((3, 3), dtype=np.float32)),
    }
    with pytest.raises(ValueError, match="conditioned_uncertainty_contains_negative_values"):
        write_final_output_layer_contract(tmp_path / "layer_contract.json", outputs=outputs)
