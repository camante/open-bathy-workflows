from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from benchmark_workflow_stage import prepare_benchmark_holdout_support_exclusion


def _write_raster(path: Path, arr: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, float(arr.shape[0]), 1, 1),
        "nodata": -9999.0,
        "tiled": False,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype("float32"), 1)


class _Log:
    def info(self, *args, **kwargs):
        return None



def test_prepare_benchmark_holdout_exclusion_removes_holdout_from_active_support(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    auth = tmp_path / "authoritative_base.tif"
    arr = -np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    _write_raster(auth, arr)

    support = pd.DataFrame(
        {
            "x": np.repeat(np.arange(0.5, 4.0, 1.0), 4),
            "y": np.tile(np.arange(3.5, -0.5, -1.0), 4),
            "depth_m": -np.arange(1, 17, dtype=float),
            "source": "authoritative_base",
        }
    )
    sdb_support = tmp_path / "sdb_support.csv"
    river_support = tmp_path / "river_support.csv"
    support.to_csv(sdb_support, index=False)
    support.to_csv(river_support, index=False)

    cfg = SimpleNamespace(
        out_dir=out_dir,
        authoritative_base=auth,
        sdb_authoritative_extra_xyz=sdb_support,
        river_authoritative_soundings=river_support,
        river_authoritative_bed=None,
        extra_xyz_crs="EPSG:4326",
        working_srs="EPSG:4326",
    )
    args = SimpleNamespace(
        benchmark_holdout=None,
        benchmark_auto_holdout=True,
        benchmark_holdout_frac=0.25,
        benchmark_holdout_min_points=2,
        benchmark_holdout_seed=7,
        benchmark_points_epsg=4326,
        benchmark_x_col=None,
        benchmark_y_col=None,
        benchmark_z_col=None,
    )
    report = {"authoritative_base": {"sdb_guidance": {"path": str(sdb_support)}, "river_guidance": {"path": str(river_support)}}}

    receipt = prepare_benchmark_holdout_support_exclusion(cfg=cfg, args=args, report=report, logger=_Log())

    assert receipt is not None
    holdout_path = out_dir / "benchmark" / "auto_holdout_points.csv"
    assert holdout_path.exists()
    holdout = pd.read_csv(holdout_path)
    assert "support_id" in holdout.columns
    holdout_ids = set(holdout["support_id"].dropna().astype(str))
    assert holdout_ids

    active_sdb = pd.read_csv(Path(cfg.sdb_authoritative_extra_xyz))
    active_river = pd.read_csv(Path(cfg.river_authoritative_soundings))
    assert holdout_ids.isdisjoint(set(active_sdb["support_id"].dropna().astype(str)))
    assert holdout_ids.isdisjoint(set(active_river["support_id"].dropna().astype(str)))

    used = receipt["used_support"]
    assert used["sdb_training_support"]["holdout_count_removed"] > 0
    assert used["river_soundings_support"]["holdout_count_removed"] > 0

    bed_path = Path(cfg.river_authoritative_bed)
    assert bed_path.exists()
    with rasterio.open(bed_path) as ds:
        bed = ds.read(1)
        nodata = ds.nodata
    for row, col in holdout[["support_row", "support_col"]].dropna().astype(int).itertuples(index=False):
        assert np.isclose(bed[row, col], nodata)
