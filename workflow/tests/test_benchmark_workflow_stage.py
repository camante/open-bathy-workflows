from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from benchmark_workflow_stage import run_workflow_benchmark


def _write_raster(path: Path, arr: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 3, 1, 1),
        "nodata": -9999.0,
        "tiled": False,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype("float32"), 1)


def test_run_workflow_benchmark_basic(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()
    river_work = out_dir / "derived_cache" / "run1" / "river" / "work"
    river_work.mkdir(parents=True)

    baseline = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    final = baseline.copy()
    final[1:, :] = final[1:, :] + 1
    support = np.array([[1, 1, 0], [0, 2, 2], [0, 0, 2]], dtype=np.float32)
    prov = np.array([[5, 5, 0], [0, 6, 6], [0, 0, 6]], dtype=np.float32)
    river_mask = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    estuary_mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)
    _write_raster(combined / 'support_class.tif', support)
    _write_raster(combined / 'bathy_combined_depth_conditioned_provenance.tif', prov)
    _write_raster(river_work / 'river_channel_mask.tif', river_mask)
    _write_raster(river_work / 'estuary_clip_mask.tif', estuary_mask)

    holdout = pd.DataFrame({'x': [0.5, 1.5, 0.5], 'y': [2.5, 1.5, 1.5], 'z': [1.0, 5.0, 4.0]})
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    report = {
        'outputs': {
            'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif'),
            'support_class': str(combined / 'support_class.tif'),
            'selected_final_provenance': str(combined / 'bathy_combined_depth_conditioned_provenance.tif'),
        },
        'authoritative_base': {'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')}},
        'river': {'outputs': {
            'river_channel_mask': str(river_work / 'river_channel_mask.tif'),
            'estuary_clip_mask': str(river_work / 'estuary_clip_mask.tif'),
        }},
    }
    args = SimpleNamespace(
        benchmark_holdout=str(holdout_path),
        benchmark_points_epsg=4326,
        benchmark_x_col='x',
        benchmark_y_col='y',
        benchmark_z_col='z',
        benchmark_baseline_raster=None,
        benchmark_final_raster=None,
    )
    cfg = SimpleNamespace(out_dir=out_dir, run_id='run1')

    class _Log:
        def info(self, *args, **kwargs):
            return None

    payload = run_workflow_benchmark(cfg=cfg, args=args, report=report, logger=_Log(), final_path=combined / 'bathy_combined_depth_conditioned.tif')
    assert payload is not None
    assert (out_dir / 'benchmark' / 'benchmark_summary.json').exists()
    assert (out_dir / 'benchmark' / 'benchmark_table.csv').exists()
    assert (out_dir / 'benchmark' / 'benchmark_summary.md').exists()
    assert payload['overall']['baseline']['n'] == 3
    assert payload['counts']['input_rows'] == 3
    assert payload['counts']['dropped_after_sampling'] == 0


def test_run_workflow_benchmark_infers_epsg4326(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()

    baseline = np.array([[1, 2], [3, 4]], dtype=np.float32)
    final = baseline + 0.5
    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)

    holdout = pd.DataFrame({'lon': [0.5, 1.5], 'lat': [2.5, 1.5], 'depth_m': [1.0, 4.0]})
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    report = {
        'outputs': {'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif')},
        'authoritative_base': {'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')}},
        'river': {'outputs': {}},
    }
    args = SimpleNamespace(
        benchmark_holdout=str(holdout_path),
        benchmark_points_epsg=None,
        benchmark_x_col=None,
        benchmark_y_col=None,
        benchmark_z_col=None,
        benchmark_baseline_raster=None,
        benchmark_final_raster=None,
    )
    cfg = SimpleNamespace(out_dir=out_dir, run_id='run1')

    class _Log:
        def info(self, *args, **kwargs):
            return None

    payload = run_workflow_benchmark(cfg=cfg, args=args, report=report, logger=_Log(), final_path=combined / 'bathy_combined_depth_conditioned.tif')
    assert payload is not None
    assert payload['inputs']['points_epsg'] == 4326
    assert payload['overall']['baseline']['n'] == 2


def test_run_workflow_benchmark_supports_gpkg_holdout(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import Point

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()

    baseline = np.array([[1, 2], [3, 4]], dtype=np.float32)
    final = baseline + 0.25
    _write_raster(combined / "authoritative_base_aligned.tif", baseline)
    _write_raster(combined / "bathy_combined_depth_conditioned.tif", final)

    gdf = gpd.GeoDataFrame({"depth_m": [1.0, 4.0]}, geometry=[Point(0.5, 2.5), Point(1.5, 1.5)], crs="EPSG:4326")
    holdout_path = tmp_path / "holdout.gpkg"
    gdf.to_file(holdout_path, driver="GPKG")

    report = {
        "outputs": {"selected_final_depth": str(combined / "bathy_combined_depth_conditioned.tif")},
        "authoritative_base": {"outputs": {"authoritative_aligned": str(combined / "authoritative_base_aligned.tif")}},
        "river": {"outputs": {}},
    }
    args = SimpleNamespace(
        benchmark_holdout=str(holdout_path),
        benchmark_points_epsg=None,
        benchmark_x_col=None,
        benchmark_y_col=None,
        benchmark_z_col=None,
        benchmark_baseline_raster=None,
        benchmark_final_raster=None,
    )
    cfg = SimpleNamespace(out_dir=out_dir, run_id="run1")

    class _Log:
        def info(self, *args, **kwargs):
            return None

    payload = run_workflow_benchmark(cfg=cfg, args=args, report=report, logger=_Log(), final_path=combined / "bathy_combined_depth_conditioned.tif")
    assert payload is not None
    assert payload["overall"]["baseline"]["n"] == 2

def test_run_workflow_benchmark_auto_holdout_from_authoritative_support(tmp_path: Path):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    combined = out_dir / 'combined'
    combined.mkdir()

    baseline = np.arange(100, dtype=np.float32).reshape(10, 10)
    final = baseline + 0.5
    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)

    cand = pd.DataFrame({
        'x': np.repeat(np.arange(0.5, 10.0, 1.0), 10),
        'y': np.tile(np.arange(9.5, -0.5, -1.0), 10),
        'z': np.arange(100, dtype=float),
    })
    cand_path = tmp_path / 'authoritative_support.csv'
    cand.to_csv(cand_path, index=False)

    report = {
        'outputs': {'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif')},
        'authoritative_base': {
            'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')},
            'sdb_guidance': {'path': str(cand_path)},
        },
        'river': {'outputs': {}},
    }
    args = SimpleNamespace(
        benchmark_holdout=None,
        benchmark_auto_holdout=True,
        benchmark_holdout_frac=0.2,
        benchmark_holdout_min_points=10,
        benchmark_holdout_seed=42,
        benchmark_points_epsg=4326,
        benchmark_x_col=None,
        benchmark_y_col=None,
        benchmark_z_col=None,
        benchmark_baseline_raster=None,
        benchmark_final_raster=None,
    )
    cfg = SimpleNamespace(out_dir=out_dir, run_id='run1')

    class _Log:
        def info(self, *args, **kwargs):
            return None

    payload = run_workflow_benchmark(cfg=cfg, args=args, report=report, logger=_Log(), final_path=combined / 'bathy_combined_depth_conditioned.tif')
    assert payload is not None
    assert payload['inputs']['auto_holdout'] is True
    assert (out_dir / 'benchmark' / 'auto_holdout_points.csv').exists()
    assert (out_dir / 'benchmark' / 'auto_holdout_receipt.json').exists()
    assert payload['counts']['points_after_sampling'] > 0


def test_run_workflow_benchmark_fails_when_authoritative_locked_changes(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()

    baseline = np.array([[1, 2], [3, 4]], dtype=np.float32)
    final = baseline + 1.0
    support = np.array([[1, 0], [0, 0]], dtype=np.float32)
    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)
    _write_raster(combined / 'support_class.tif', support)

    holdout = pd.DataFrame({'x': [0.5, 1.5], 'y': [2.5, 1.5], 'z': [1.0, 4.0]})
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    report = {
        'outputs': {
            'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif'),
            'support_class': str(combined / 'support_class.tif'),
        },
        'authoritative_base': {'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')}},
        'river': {'outputs': {}},
    }
    args = SimpleNamespace(
        benchmark_holdout=str(holdout_path),
        benchmark_points_epsg=4326,
        benchmark_x_col='x',
        benchmark_y_col='y',
        benchmark_z_col='z',
        benchmark_baseline_raster=None,
        benchmark_final_raster=None,
    )
    cfg = SimpleNamespace(out_dir=out_dir, run_id='run1')

    class _Log:
        def info(self, *args, **kwargs):
            return None

    import pytest
    with pytest.raises(RuntimeError, match='authoritative_locked'):
        run_workflow_benchmark(cfg=cfg, args=args, report=report, logger=_Log(), final_path=combined / 'bathy_combined_depth_conditioned.tif')


def test_run_workflow_benchmark_prefers_baseline_cudem_interpolation(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()

    measured_only = np.array([[10, 10], [10, 10]], dtype=np.float32)
    baseline_cudem = np.array([[1, 2], [3, 4]], dtype=np.float32)
    final = baseline_cudem + 1.0
    _write_raster(combined / 'authoritative_base_aligned.tif', measured_only)
    baseline_cudem_path = tmp_path / 'cudem_baseline_interpolation.tif'
    _write_raster(baseline_cudem_path, baseline_cudem)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)

    holdout = pd.DataFrame({'x': [0.5, 1.5], 'y': [2.5, 1.5], 'z': [1.0, 4.0]})
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    report = {
        'outputs': {'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif')},
        'authoritative_base': {'outputs': {'aligned_authoritative_base': str(combined / 'authoritative_base_aligned.tif')}},
        'authoritative_base_auto': {'baseline_cudem_interpolation': str(baseline_cudem_path)},
        'river': {'outputs': {}},
    }
    args = SimpleNamespace(
        benchmark_holdout=str(holdout_path),
        benchmark_auto_holdout=False,
        benchmark_points_epsg=4326,
        benchmark_x_col='x',
        benchmark_y_col='y',
        benchmark_z_col='z',
        benchmark_baseline_raster=None,
        benchmark_final_raster=None,
    )
    cfg = SimpleNamespace(out_dir=out_dir, run_id='run1', authoritative_base=combined / 'authoritative_base_aligned.tif')

    class _Log:
        def info(self, *args, **kwargs):
            return None

    payload = run_workflow_benchmark(cfg=cfg, args=args, report=report, logger=_Log(), final_path=combined / 'bathy_combined_depth_conditioned.tif')
    assert payload is not None
    assert payload['inputs']['baseline_raster'] == str(baseline_cudem_path)
    assert payload['overall']['baseline']['rmse'] == 0.0
    assert payload['overall']['final']['rmse'] > 0.0
