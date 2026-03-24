from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from river_longitudinal_profile_contract import write_river_longitudinal_profile_contract


def _write_raster(path: Path, arr: np.ndarray) -> str:
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
        nodata=-9999.0,
    ) as ds:
        ds.write(arr, 1)
    return str(path)


def test_river_longitudinal_profile_contract_writes_when_required_artifacts_exist(tmp_path: Path):
    table = tmp_path / "river_longitudinal_profile.csv"
    table.write_text("profile_id,station_m,bed_elevation_m\nmain,0,1.0\n", encoding="utf-8")
    outputs = {
        "centerline_stationing": _write_raster(tmp_path / "centerline_stationing.tif", np.ones((3, 3), dtype=np.float32)),
        "centerline_elevation": _write_raster(tmp_path / "centerline_elevation.tif", np.ones((3, 3), dtype=np.float32)),
        "longitudinal_profile": str(table),
        "longitudinal_profile_elevation": _write_raster(tmp_path / "longitudinal_profile_elevation.tif", np.ones((3, 3), dtype=np.float32)),
        "longitudinal_profile_uncertainty": _write_raster(tmp_path / "longitudinal_profile_uncertainty.tif", np.ones((3, 3), dtype=np.float32)),
        "xs_support_elevation": _write_raster(tmp_path / "xs_support_elevation.tif", np.ones((3, 3), dtype=np.float32)),
        "hydraulic_backbone": str(tmp_path / "river_hydraulic_backbone.csv"),
    }
    Path(outputs["hydraulic_backbone"]).write_text("profile_id,station_m,network_backbone_elevation_m\nmain,0,1.0\n", encoding="utf-8")
    out = write_river_longitudinal_profile_contract(tmp_path / "river_longitudinal_profile_contract.json", outputs=outputs)
    payload = json.loads(out.read_text())
    assert payload["ok"] is True
    assert payload["mode"] == "network_aware_hydraulic_backbone_v1"


def test_river_longitudinal_profile_contract_fails_when_required_artifact_missing(tmp_path: Path):
    outputs = {
        "centerline_stationing": _write_raster(tmp_path / "centerline_stationing.tif", np.ones((3, 3), dtype=np.float32)),
    }
    with pytest.raises(ValueError, match="missing_centerline_elevation"):
        write_river_longitudinal_profile_contract(tmp_path / "river_longitudinal_profile_contract.json", outputs=outputs)
