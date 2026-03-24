from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from alignment import TiePoints, residuals_against_points
from contracts_sign_semantics_runtime import run_sign_semantics_stage_contracts


def _write_raster(path: Path, arr: np.ndarray, tags: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-70, 43, 0.01, 0.01),
        nodata=-9999.0,
    ) as ds:
        ds.write(arr.astype("float32"), 1)
        ds.update_tags(**tags)


def test_stage_contract_warns_on_auto_soundings(tmp_path: Path):
    cfg = SimpleNamespace(out_dir=str(tmp_path), river_soundings_mode="auto")
    report = {}
    suite = run_sign_semantics_stage_contracts(cfg, report, contracts_dir=tmp_path / "contracts")
    assert suite["warn"] >= 1
    assert any(r["name"] == "river_soundings_mode_explicit" and r["status"] == "warning" for r in suite["results"])


def test_alignment_semantic_contract_rejects_elevation_raster(tmp_path: Path):
    rast = tmp_path / "elev.tif"
    _write_raster(rast, np.full((10, 10), 5.0, dtype=np.float32), {"VALUE_TYPE": "elevation", "SIGN_CONVENTION": "relative_to_datum"})
    pts = TiePoints(
        lon=np.array([-69.995, -69.985]),
        lat=np.array([42.995, 42.985]),
        depth_m=np.array([1.0, 2.0]),
        source=None,
        depth_positive_down=True,
    )
    out = residuals_against_points(str(rast), pts)
    assert out["status"] == "skip"
    assert out["reason"] == "semantic_mismatch"
    assert out["semantic_contract"]["status"] == "error"
