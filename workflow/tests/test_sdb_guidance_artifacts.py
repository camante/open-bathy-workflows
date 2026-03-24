import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
import pytest

from sdb_guidance import (
    build_sdb_guidance_manifest,
    guidance_artifact_paths,
    rasterize_sdb_guide_points_to_template,
    write_sdb_guidance_bounds,
    write_sdb_guidance_manifest,
)


def _write_raster(path: Path, arr: np.ndarray, nodata: float = -9999.0) -> None:
    from rasterio.transform import from_origin
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0.0, float(arr.shape[0]), 1.0, 1.0),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)


def test_write_sdb_guidance_bounds(tmp_path: Path):
    depth = tmp_path / "depth.tif"
    unc = tmp_path / "depth_uncertainty.tif"
    _write_raster(depth, np.array([[2.0, 3.0], [4.0, np.nan]], dtype=np.float32))
    _write_raster(unc, np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32))
    outputs = write_sdb_guidance_bounds(depth, unc)
    lower = Path(outputs["lower_bound_raster"])
    upper = Path(outputs["upper_bound_raster"])
    assert lower.exists() and upper.exists()
    with rasterio.open(lower) as ds:
        arr = ds.read(1)
        assert np.isclose(arr[0, 0], 1.5)
        assert np.isclose(arr[1, 0], 2.5)
    with rasterio.open(upper) as ds:
        arr = ds.read(1)
        assert np.isclose(arr[0, 1], 4.0)
        assert ds.nodata == -9999.0


def test_build_and_write_sdb_guidance_manifest(tmp_path: Path):
    depth = tmp_path / "pred_depth.tif"
    _write_raster(depth, np.ones((2, 2), dtype=np.float32))
    paths = guidance_artifact_paths(depth)
    for key in ("guide_points", "guidance_weight_raster", "trusted_interior_raster", "admissibility_raster"):
        p = paths[key]
        if p.suffix == ".gpkg":
            p.write_text("placeholder", encoding="utf-8")
        else:
            _write_raster(p, np.ones((2, 2), dtype=np.float32))
    args = SimpleNamespace(authoritative_base="/tmp/auth_base.tif")
    manifest = build_sdb_guidance_manifest(out_root=tmp_path, depth_raster=depth, args=args)
    assert manifest["guidance_only"] is True
    assert manifest["artifacts"]["depth_raster_role"] == "diagnostic_only"
    assert manifest["final_route_contract"]["diagnostic_only_artifacts"] == ["depth_raster"]
    assert "guide_points" in manifest["final_route_contract"]["allowed_structural_artifacts"]
    assert manifest["artifact_roles"]["guide_points"] == "sparse_guidance_points"
    manifest_path = write_sdb_guidance_manifest(out_root=tmp_path, depth_raster=depth, args=args)
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["artifacts"]["guide_points"].endswith("_guide_points.gpkg")


@pytest.mark.skipif(__import__("importlib").util.find_spec("geopandas") is None, reason="geopandas not installed")
def test_rasterize_sdb_guide_points_to_template(tmp_path: Path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point
    depth = tmp_path / "pred_depth.tif"
    _write_raster(depth, np.ones((3, 3), dtype=np.float32))
    gpkg = guidance_artifact_paths(depth)["guide_points"]
    gdf = gpd.GeoDataFrame(
        pd.DataFrame({"depth_m": [2.0, 4.0]}),
        geometry=[Point(0.5, 2.5), Point(1.5, 1.5)],
        crs="EPSG:4326",
    )
    gdf.to_file(gpkg, driver="GPKG")
    arr = rasterize_sdb_guide_points_to_template(gpkg, depth)
    assert arr is not None
    assert np.isclose(arr[0, 0], 2.0)
    assert np.isclose(arr[1, 1], 4.0)
    assert np.isnan(arr[2, 2])
