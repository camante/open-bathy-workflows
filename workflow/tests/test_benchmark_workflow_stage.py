from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

from benchmark_workflow_stage import (
    run_workflow_benchmark,
    _build_river_benchmark_mode_summary,
    _build_river_withheld_support_plan,
    _compute_support_aware_science_summary,
    _compute_river_scientific_summary,
)


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


def test_run_workflow_benchmark_writes_river_scientific_receipts(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import Point

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()
    river_dir = out_dir / "river"
    river_dir.mkdir()

    baseline = np.array([[1, 2, 3, 4]], dtype=np.float32)
    final = np.array([[1, 2, 5, 6]], dtype=np.float32)
    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)

    holdout = pd.DataFrame({'x': [0.5, 3.5], 'y': [2.5, 2.5], 'z': [1.0, 4.0]})
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    profile_points = gpd.GeoDataFrame(
        {
            'profile_id': ['reach_a'] * 4,
            'station_m': [0.0, 10.0, 20.0, 30.0],
            'longitudinal_profile_source': ['centerline'] * 4,
            'generalized_longitudinal_bed_reconciled_elevation_m': [1.0, 2.0, 3.0, 4.0],
            'component_support_class': ['unsupported_mainstem'] * 4,
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5), Point(3.5, 2.5)],
        crs='EPSG:4326',
    )
    profile_points_path = river_dir / 'river_longitudinal_profile_points.gpkg'
    profile_points.to_file(profile_points_path, driver='GPKG')

    coverage = pd.DataFrame({
        'profile_id': ['reach_a'] * 4,
        'station_m': [0.0, 10.0, 20.0, 30.0],
        'profile_support_class': ['centerline_supported', 'centerline_supported', 'xs_supported', 'xs_supported'],
        'profile_support_source_count': [2, 2, 3, 3],
        'profile_support_present': [True, True, True, True],
        'profile_authoritative_anchor_present': [False, False, False, False],
        'profile_xs_support_present': [False, False, True, True],
        'profile_centerline_support_present': [True, True, True, True],
        'profile_bank_support_present': [True, True, True, True],
        'profile_wse_support_present': [False, False, False, False],
    })
    coverage_path = river_dir / 'river_longitudinal_profile_coverage.csv'
    coverage.to_csv(coverage_path, index=False)

    report = {
        'outputs': {'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif')},
        'authoritative_base': {'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')}},
        'river': {'outputs': {
            'longitudinal_profile_points': str(profile_points_path),
            'longitudinal_profile_coverage': str(coverage_path),
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
    assert payload['river_scientific_summary']['available'] is True
    assert (out_dir / 'benchmark' / 'benchmark_river_science_summary.json').exists()
    assert (out_dir / 'benchmark' / 'benchmark_river_longitudinal_metrics.csv').exists()
    assert payload['river_scientific_summary']['adjacent_step_metrics']['delta_final_minus_baseline_p95_abs'] > 0.0
    assert payload['river_scientific_summary']['curvature_metrics']['delta_final_minus_baseline_p95_abs'] > 0.0
    assert payload['river_scientific_summary']['coverage']['coverage_summary']['unsupported_station_count'] == 0
    assert payload['river_scientific_summary']['centerline_agreement']['available'] is True
    assert payload['river_scientific_summary']['centerline_agreement']['target_field'] == 'generalized_longitudinal_bed_reconciled_elevation_m'
    assert (out_dir / 'benchmark' / 'benchmark_centerline_agreement_summary.json').exists()


def test_run_workflow_benchmark_writes_support_aware_validation_receipts(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()
    river_dir = out_dir / "river"
    river_dir.mkdir()

    baseline = np.array([[5, 5], [5, 5]], dtype=np.float32)
    final = np.array([[4, 7], [5, 5]], dtype=np.float32)
    river_mask = np.array([[1, 1], [0, 0]], dtype=np.float32)
    pred_conf = np.array([[0.95, 0.20], [np.nan, np.nan]], dtype=np.float32)
    measured = np.array([[0.90, 0.05], [np.nan, np.nan]], dtype=np.float32)
    structure = np.array([[0.05, 0.90], [np.nan, np.nan]], dtype=np.float32)
    caution = np.array([[0, 1], [0, 0]], dtype=np.float32)
    admiss = np.array([[1, 0], [0, 0]], dtype=np.float32)

    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)
    _write_raster(river_dir / 'river_channel_mask.tif', river_mask)
    _write_raster(river_dir / 'river_channel_surface_prediction_support_confidence.tif', pred_conf)
    _write_raster(river_dir / 'river_channel_surface_measured_anchor_fraction.tif', measured)
    _write_raster(river_dir / 'river_channel_surface_structure_only_fraction.tif', structure)
    _write_raster(river_dir / 'river_channel_surface_low_support_caution.tif', caution)
    _write_raster(river_dir / 'river_channel_surface_prediction_admissibility.tif', admiss)

    holdout = pd.DataFrame({
        'x': [0.5, 1.5],
        'y': [2.5, 2.5],
        'z': [4.0, 6.0],
    })
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    report = {
        'outputs': {'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif')},
        'authoritative_base': {'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')}},
        'river': {'outputs': {
            'river_channel_mask': str(river_dir / 'river_channel_mask.tif'),
            'channel_surface_prediction_support_confidence': str(river_dir / 'river_channel_surface_prediction_support_confidence.tif'),
            'channel_surface_measured_anchor_fraction': str(river_dir / 'river_channel_surface_measured_anchor_fraction.tif'),
            'channel_surface_structure_only_fraction': str(river_dir / 'river_channel_surface_structure_only_fraction.tif'),
            'channel_surface_low_support_caution': str(river_dir / 'river_channel_surface_low_support_caution.tif'),
            'channel_surface_prediction_admissibility': str(river_dir / 'river_channel_surface_prediction_admissibility.tif'),
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
    assert (out_dir / 'benchmark' / 'benchmark_holdout_scored.csv').exists()
    assert (out_dir / 'benchmark' / 'benchmark_support_aware_validation_summary.json').exists()
    support_aware = payload['support_aware_validation_summary']
    assert support_aware['available'] is True
    assert support_aware['river_holdout_count'] == 2
    assert support_aware['by_prediction_admissibility']['groups']['0']['improvement']['improved_count'] == 0
    assert support_aware['by_prediction_admissibility']['groups']['1']['improvement']['improved_count'] == 1
    assert support_aware['by_low_support_caution']['groups']['1']['baseline']['n'] == 1
    assert support_aware['by_prediction_support_confidence_bin']['groups']['high']['baseline']['n'] == 1
    assert support_aware['by_prediction_structure_only_fraction_bin']['groups']['structure_dominant']['baseline']['n'] == 1


def test_run_workflow_benchmark_auto_holdout_prefers_active_river_pool_when_sdb_empty(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()

    baseline = np.arange(25, dtype=np.float32).reshape(5, 5)
    final = baseline + 0.25
    _write_raster(combined / "authoritative_base_aligned.tif", baseline)
    _write_raster(combined / "bathy_combined_depth_conditioned.tif", final)

    empty_sdb = tmp_path / "authoritative_sdb_support_points.csv"
    empty_sdb.write_text("x,y,depth_m,source\n", encoding="utf-8")
    river_cand = pd.DataFrame({
        "x": [0.5, 1.5, 2.5, 3.5],
        "y": [4.5, 3.5, 2.5, 1.5],
        "depth_m": [1.0, 2.0, 3.0, 4.0],
    })
    river_cand_path = tmp_path / "authoritative_river_soundings.csv"
    river_cand.to_csv(river_cand_path, index=False)

    report = {
        "outputs": {"selected_final_depth": str(combined / "bathy_combined_depth_conditioned.tif")},
        "authoritative_base": {
            "outputs": {"authoritative_aligned": str(combined / "authoritative_base_aligned.tif")},
            "sdb_guidance": {"path": str(empty_sdb), "explicit_support": {"support_pixels": 0}},
            "river_guidance": {"path": str(river_cand_path)},
        },
        "method_activation_truth": {
            "river": {"effective_should_run": True},
            "sdb": {"effective_should_run": False},
        },
        "river": {"outputs": {}},
    }
    args = SimpleNamespace(
        benchmark_holdout=None,
        benchmark_auto_holdout=True,
        benchmark_holdout_frac=0.5,
        benchmark_holdout_min_points=2,
        benchmark_holdout_seed=7,
        benchmark_points_epsg=4326,
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
    receipt = (out_dir / "benchmark" / "auto_holdout_receipt.json").read_text(encoding="utf-8")
    assert "authoritative_river_soundings.csv" in receipt


def test_run_workflow_benchmark_auto_holdout_infers_projected_candidate_epsg_from_report(tmp_path: Path):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    combined = out_dir / 'combined'
    combined.mkdir()

    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    profile = {
        'driver': 'GTiff', 'height': 10, 'width': 10, 'count': 1, 'dtype': 'float32',
        'crs': 'EPSG:32619', 'transform': from_origin(320000, 4750000, 10, 10), 'nodata': -9999.0, 'tiled': False,
    }
    baseline_path = combined / 'authoritative_base_aligned.tif'
    final_path = combined / 'bathy_combined_depth_conditioned.tif'
    with rasterio.open(baseline_path, 'w', **profile) as ds:
        ds.write(arr, 1)
    with rasterio.open(final_path, 'w', **profile) as ds:
        ds.write((arr + 0.5).astype('float32'), 1)

    xs = 320005.0 + np.repeat(np.arange(10) * 10.0, 10)
    ys = 4749995.0 - np.tile(np.arange(10) * 10.0, 10)
    cand = pd.DataFrame({'x': xs, 'y': ys, 'depth_m': np.arange(100, dtype=float)})
    cand_path = tmp_path / 'authoritative_river_soundings.csv'
    cand.to_csv(cand_path, index=False)

    source_raster = tmp_path / 'source_authoritative_base.tif'
    with rasterio.open(source_raster, 'w', **profile) as ds:
        ds.write(arr, 1)

    report = {
        'outputs': {'selected_final_depth': str(final_path)},
        'authoritative_base': {
            'outputs': {'authoritative_aligned': str(baseline_path)},
            'inputs': {'source': str(source_raster)},
            'river_guidance': {'path': str(cand_path), 'source': 'authoritative_base'},
        },
        'method_activation_truth': {'river': {'effective_should_run': True}, 'sdb': {'effective_should_run': False}},
        'river': {'outputs': {}},
    }
    args = SimpleNamespace(
        benchmark_holdout=None,
        benchmark_auto_holdout=True,
        benchmark_holdout_frac=0.2,
        benchmark_holdout_min_points=10,
        benchmark_holdout_seed=42,
        benchmark_points_epsg=None,
        benchmark_x_col=None,
        benchmark_y_col=None,
        benchmark_z_col=None,
        benchmark_baseline_raster=None,
        benchmark_final_raster=None,
    )
    cfg = SimpleNamespace(out_dir=out_dir, run_id='run1', authoritative_base=str(source_raster))

    class _Log:
        def info(self, *args, **kwargs):
            return None

    payload = run_workflow_benchmark(cfg=cfg, args=args, report=report, logger=_Log(), final_path=final_path)
    assert payload is not None
    assert payload['inputs']['points_epsg'] == 32619
    assert (out_dir / 'benchmark' / 'auto_holdout_points.csv').exists()


def test_run_workflow_benchmark_auto_holdout_prefers_working_srs_over_source_raster_crs(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()

    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    projected_profile = {
        'driver': 'GTiff', 'height': 10, 'width': 10, 'count': 1, 'dtype': 'float32',
        'crs': 'EPSG:4326', 'transform': from_origin(-72.0, 43.5, 0.1, 0.1), 'nodata': -9999.0, 'tiled': False,
    }
    baseline_path = combined / 'authoritative_base_aligned.tif'
    final_path = combined / 'bathy_combined_depth_conditioned.tif'
    with rasterio.open(baseline_path, 'w', **projected_profile) as ds:
        ds.write(arr, 1)
    with rasterio.open(final_path, 'w', **projected_profile) as ds:
        ds.write((arr + 0.25).astype('float32'), 1)

    lon = -71.95 + np.repeat(np.arange(10) * 0.05, 10)
    lat = 43.45 - np.tile(np.arange(10) * 0.05, 10)
    tx = Transformer.from_crs('EPSG:4326', 'EPSG:32619', always_xy=True)
    xs, ys = tx.transform(lon, lat)
    cand = pd.DataFrame({'x': xs, 'y': ys, 'depth_m': np.arange(100, dtype=float)})
    cand_path = tmp_path / 'authoritative_river_soundings.csv'
    cand.to_csv(cand_path, index=False)

    source_raster = tmp_path / 'source_authoritative_base.tif'
    with rasterio.open(source_raster, 'w', **projected_profile) as ds:
        ds.write(arr, 1)

    report = {
        'outputs': {'selected_final_depth': str(final_path)},
        'authoritative_base': {
            'outputs': {'authoritative_aligned': str(baseline_path)},
            'inputs': {'source': str(source_raster)},
            'river_guidance': {'path': str(cand_path), 'source': 'authoritative_base'},
        },
        'method_activation_truth': {'river': {'effective_should_run': True}, 'sdb': {'effective_should_run': False}},
        'river': {'outputs': {}},
    }
    args = SimpleNamespace(
        benchmark_holdout=None,
        benchmark_auto_holdout=True,
        benchmark_holdout_frac=0.2,
        benchmark_holdout_min_points=10,
        benchmark_holdout_seed=42,
        benchmark_points_epsg=None,
        benchmark_x_col=None,
        benchmark_y_col=None,
        benchmark_z_col=None,
        benchmark_baseline_raster=None,
        benchmark_final_raster=None,
    )
    cfg = SimpleNamespace(out_dir=out_dir, run_id='run1', authoritative_base=str(source_raster), working_srs='EPSG:32619')

    class _Log:
        def info(self, *args, **kwargs):
            return None

    payload = run_workflow_benchmark(cfg=cfg, args=args, report=report, logger=_Log(), final_path=final_path)
    assert payload is not None
    assert payload['inputs']['points_epsg'] == 32619
    receipt = pd.read_csv(out_dir / 'benchmark' / 'auto_holdout_points.csv')
    assert int(receipt['benchmark_points_epsg'].iloc[0]) == 32619


def test_run_workflow_benchmark_writes_hard_river_summary(tmp_path: Path):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    combined = out_dir / 'combined'
    combined.mkdir()
    river_work = out_dir / 'derived_cache' / 'run1' / 'river' / 'work'
    river_work.mkdir(parents=True)

    baseline = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    final = baseline.copy()
    final[1:, :] = final[1:, :] + np.array([[0.0, -0.5, 0.0], [-0.25, 0.25, 0.0]], dtype=np.float32)
    support = np.array([[1, 1, 0], [4, 5, 0], [6, 4, 0]], dtype=np.float32)
    river_mask = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
    conf = np.array([[0.9, 0.9, 0.0], [0.2, 0.3, 0.0], [0.2, 0.8, 0.0]], dtype=np.float32)
    measured = np.array([[1.0, 1.0, 0.0], [0.05, 0.1, 0.0], [0.0, 0.4, 0.0]], dtype=np.float32)
    struct_only = np.array([[0.0, 0.0, 0.0], [0.8, 0.7, 0.0], [0.9, 0.2, 0.0]], dtype=np.float32)
    caution = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)
    _write_raster(combined / 'support_class.tif', support)
    _write_raster(river_work / 'river_channel_mask.tif', river_mask)
    _write_raster(river_work / 'river_channel_surface_prediction_support_confidence.tif', conf)
    _write_raster(river_work / 'river_channel_surface_prediction_measured_anchor_fraction.tif', measured)
    _write_raster(river_work / 'river_channel_surface_prediction_structure_only_fraction.tif', struct_only)
    _write_raster(river_work / 'river_channel_surface_low_support_caution.tif', caution)

    holdout = pd.DataFrame({'x': [0.5, 1.5, 0.5, 1.5], 'y': [1.5, 1.5, 0.5, 0.5], 'z': [4.1, 5.2, 6.8, 7.7]})
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    report = {
        'outputs': {
            'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif'),
            'support_class': str(combined / 'support_class.tif'),
        },
        'authoritative_base': {'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')}},
        'river': {'outputs': {
            'river_channel_mask': str(river_work / 'river_channel_mask.tif'),
            'channel_surface_prediction_support_confidence': str(river_work / 'river_channel_surface_prediction_support_confidence.tif'),
            'channel_surface_prediction_measured_anchor_fraction': str(river_work / 'river_channel_surface_prediction_measured_anchor_fraction.tif'),
            'channel_surface_prediction_structure_only_fraction': str(river_work / 'river_channel_surface_prediction_structure_only_fraction.tif'),
            'channel_surface_low_support_caution': str(river_work / 'river_channel_surface_low_support_caution.tif'),
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
    assert payload['hard_river_benchmark_summary']['available'] is True
    assert payload['hard_river_benchmark_summary']['hard_problem_group']['n'] >= 1
    assert (out_dir / 'benchmark' / 'benchmark_hard_river_summary.json').exists()
    assert (out_dir / 'benchmark' / 'benchmark_hard_river_groups.csv').exists()


def test_run_workflow_benchmark_writes_authoritative_role_validation_summary(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()
    river_dir = out_dir / "river"
    river_dir.mkdir()

    baseline = np.array([[5, 5], [5, 5]], dtype=np.float32)
    final = np.array([[4, 7], [5, 5]], dtype=np.float32)
    river_mask = np.array([[1, 1], [0, 0]], dtype=np.float32)
    role_code = np.array([[3, 1], [0, 0]], dtype=np.float32)
    role_conf = np.array([[0.9, 0.7], [0.0, 0.0]], dtype=np.float32)
    dist_bank = np.array([[6.0, 1.0], [np.nan, np.nan]], dtype=np.float32)
    norm_pos = np.array([[0.8, 0.1], [np.nan, np.nan]], dtype=np.float32)

    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)
    _write_raster(river_dir / 'river_channel_mask.tif', river_mask)
    _write_raster(river_dir / 'authoritative_role_code.tif', role_code)
    _write_raster(river_dir / 'authoritative_role_confidence.tif', role_conf)
    _write_raster(river_dir / 'authoritative_distance_to_bank_m.tif', dist_bank)
    _write_raster(river_dir / 'authoritative_normalized_channel_position.tif', norm_pos)

    holdout = pd.DataFrame({
        'x': [0.5, 1.5],
        'y': [2.5, 2.5],
        'z': [4.0, 6.0],
    })
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    report = {
        'outputs': {'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif')},
        'authoritative_base': {
            'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')},
            'river_guidance': {
                'role_code_raster': str(river_dir / 'authoritative_role_code.tif'),
                'role_confidence_raster': str(river_dir / 'authoritative_role_confidence.tif'),
                'distance_to_bank_raster': str(river_dir / 'authoritative_distance_to_bank_m.tif'),
                'normalized_channel_position_raster': str(river_dir / 'authoritative_normalized_channel_position.tif'),
            },
        },
        'river': {'outputs': {'river_channel_mask': str(river_dir / 'river_channel_mask.tif')}},
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
    assert (out_dir / 'benchmark' / 'benchmark_authoritative_role_validation_summary.json').exists()
    role_summary = payload['authoritative_role_validation_summary']
    assert role_summary['available'] is True
    assert role_summary['authoritative_role_counts']['authoritative_bed_core'] == 1
    assert role_summary['authoritative_role_counts']['authoritative_bank_margin'] == 1
    scored = pd.read_csv(out_dir / 'benchmark' / 'benchmark_holdout_scored.csv')
    assert set(scored['authoritative_role_label']) == {'authoritative_bed_core', 'authoritative_bank_margin'}
    assert 'authoritative_distance_to_bank_m' in scored.columns
    assert payload['support_aware_validation_summary']['by_authoritative_role_in_river']['available'] is True


def test_compute_support_aware_science_summary_prefers_proxy_when_holdout_blind():
    river_scientific_summary = {
        "support_distance_binned_roughness": [
            {"bin": "0_100m_near_anchor", "available": True, "n": 5, "delta_mean_abs_step": -0.01, "delta_p95_abs_step": -0.02, "baseline_mean_abs_step": 0.10, "final_mean_abs_step": 0.09, "baseline_p95_abs_step": 0.20, "final_p95_abs_step": 0.18, "roughness_improved": True},
            {"bin": "100_300m_transitional", "available": True, "n": 5, "delta_mean_abs_step": 0.00, "delta_p95_abs_step": 0.01, "baseline_mean_abs_step": 0.11, "final_mean_abs_step": 0.11, "baseline_p95_abs_step": 0.21, "final_p95_abs_step": 0.22, "roughness_improved": False},
            {"bin": "500_1000m_unsupported", "available": True, "n": 10, "delta_mean_abs_step": 0.05, "delta_p95_abs_step": 0.08, "baseline_mean_abs_step": 0.12, "final_mean_abs_step": 0.17, "baseline_p95_abs_step": 0.22, "final_p95_abs_step": 0.30, "roughness_improved": False},
        ],
        "unsupported_by_component_class": {
            "unsupported_mainstem": {"n": 7, "mean_abs_delta_m": 0.45, "median_abs_delta_m": 0.4, "p95_abs_delta_m": 0.8, "active_fraction": 0.8},
            "unsupported_side_component": {"n": 6, "mean_abs_delta_m": 0.55, "median_abs_delta_m": 0.5, "p95_abs_delta_m": 0.9, "active_fraction": 0.95},
            "tiny_detached_component": {"n": 4, "mean_abs_delta_m": 0.8, "median_abs_delta_m": 0.75, "p95_abs_delta_m": 1.1, "active_fraction": 1.0},
        },
        "unsupported_roughness_by_component_class": {
            "unsupported_mainstem": {"n": 7, "final_mean_abs_step": 0.14, "final_p95_abs_step": 0.28},
            "unsupported_side_component": {"n": 6, "final_mean_abs_step": 0.22, "final_p95_abs_step": 0.44},
        },
        "unsupported_curvature_by_component_class": {
            "unsupported_mainstem": {"n": 7, "final_mean_abs_second_diff": 0.05, "final_p95_abs_second_diff": 0.11},
            "unsupported_side_component": {"n": 6, "final_mean_abs_second_diff": 0.09, "final_p95_abs_second_diff": 0.21},
        },
        "unsupported_section_tendency_by_component_class": {
            "unsupported_mainstem": {"n": 7, "mean_tendency_depth_fraction": 0.62, "mean_tendency_confidence": 0.58, "active_fraction": 0.86, "simplified_fraction": 0.10},
            "unsupported_side_component": {"n": 6, "mean_tendency_depth_fraction": 0.44, "mean_tendency_confidence": 0.41, "active_fraction": 0.72, "simplified_fraction": 0.75},
        },
        "unsupported_inner_relief_by_component_class": {
            "unsupported_mainstem": {"n": 7, "mean_inner_relief_m": 1.25, "median_inner_relief_m": 1.1, "p95_inner_relief_m": 2.0, "max_inner_relief_m": 2.3},
            "unsupported_side_component": {"n": 6, "mean_inner_relief_m": 0.72, "median_inner_relief_m": 0.68, "p95_inner_relief_m": 1.1, "max_inner_relief_m": 1.2},
        },
        "unsupported_reconciliation_by_component_class": {
            "unsupported_mainstem": {"n": 7, "mean_reconciliation_weight": 0.42, "p95_reconciliation_weight": 0.73, "mean_abs_reconciliation_delta_m": 0.38, "p95_abs_reconciliation_delta_m": 0.7, "active_fraction": 0.86},
            "unsupported_side_component": {"n": 6, "mean_reconciliation_weight": 0.21, "p95_reconciliation_weight": 0.41, "mean_abs_reconciliation_delta_m": 0.22, "p95_abs_reconciliation_delta_m": 0.5, "active_fraction": 0.52},
        },
    }
    science_evaluation_summary = {
        "by_support_class": {
            "guidance_conditioned_river": {"n": 20, "mean_abs_delta_m": 0.3, "median_abs_delta_m": 0.25, "p95_abs_delta_m": 0.6, "active_fraction": 0.8},
            "scaffold_inferred": {"n": 10, "mean_abs_delta_m": 0.4, "median_abs_delta_m": 0.35, "p95_abs_delta_m": 0.7, "active_fraction": 0.9},
        },
        "by_prediction_admissibility": {
            "0": {"label": "inadmissible", "n": 8, "mean_abs_delta_m": 0.2, "median_abs_delta_m": 0.2, "p95_abs_delta_m": 0.3, "active_fraction": 0.5},
            "1": {"label": "admissible", "n": 22, "mean_abs_delta_m": 0.45, "median_abs_delta_m": 0.4, "p95_abs_delta_m": 0.8, "active_fraction": 0.95},
        },
    }
    zone_diff_summary = {
        "authoritative_locked_invariant": {"n": 15, "passes": True, "changed_pixels": 0, "max_abs_diff": 0.0}
    }
    hard_river_summary = {"hard_problem_evaluation_blind": True}

    summary = _compute_support_aware_science_summary(
        river_scientific_summary=river_scientific_summary,
        science_evaluation_summary=science_evaluation_summary,
        zone_diff_summary=zone_diff_summary,
        hard_river_summary=hard_river_summary,
    )

    assert summary["available"] is True
    assert summary["evaluation_mode"] == "proxy_grid_and_profile"
    assert summary["authoritative_preservation"]["passes"] is True
    assert summary["unsupported_roughness"]["station_count"] == 10
    assert summary["unsupported_science_grid"]["cell_count"] == 30
    assert summary["unsupported_by_component_class"]["point_count"] == 17
    assert summary["unsupported_by_prediction_admissibility"]["groups"]["1"]["label"] == "admissible"
    assert summary["unsupported_reconciliation"]["available"] is True
    assert summary["unsupported_mainstem_vs_side_component"]["available"] is True
    assert summary["unsupported_mainstem_vs_side_component"]["mainstem_is_more_expressive"] is True
    assert summary["unsupported_mainstem_vs_side_component"]["side_is_rougher"] is True
    assert summary["unsupported_mainstem_vs_side_component"]["side_is_more_curved"] is True
    assert summary["unsupported_mainstem_vs_side_reconciliation"]["available"] is True
    assert summary["unsupported_mainstem_vs_side_reconciliation"]["mainstem_carries_more_reconciliation"] is True


def test_run_workflow_benchmark_writes_support_aware_science_summary(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()
    river_work = out_dir / "derived_cache" / "run1" / "river" / "work"
    river_work.mkdir(parents=True)

    baseline = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    final = baseline.copy()
    final[1:, :] = final[1:, :] + 1
    support = np.array([[1, 1, 1], [4, 5, 6], [4, 5, 6]], dtype=np.float32)
    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)
    _write_raster(combined / 'support_class.tif', support)
    _write_raster(river_work / 'river_channel_mask.tif', np.ones_like(baseline, dtype=np.float32))
    _write_raster(river_work / 'river_channel_surface_prediction_admissibility.tif', np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float32))

    profile = pd.DataFrame({
        'component_id': [1,1,1,1,1,1],
        'station_m': [0,1,2,3,4,5],
        'x': [0.5,1.5,2.5,0.5,1.5,2.5],
        'y': [2.5,2.5,2.5,1.5,1.5,1.5],
        'profile_measured_support_distance_m': [50, 150, 600, 50, 150, 600],
        'profile_support_class': ['authoritative_bed_support','transitional','unsupported','authoritative_bed_support','transitional','unsupported'],
        'component_support_class': ['anchored_mainstem','anchored_mainstem','unsupported_side_component','anchored_mainstem','anchored_mainstem','tiny_detached_component'],
    })
    profile.to_csv(river_work / 'river_longitudinal_profile_points.csv', index=False)
    summary_payload = {
        'coverage_summary': {
            'unsupported_station_count': 2,
            'authoritative_bed_supported_station_count': 2,
            'authoritative_bank_margin_station_count': 0,
        }
    }
    (river_work / 'river_longitudinal_profile_summary.json').write_text(__import__('json').dumps(summary_payload))

    holdout = pd.DataFrame({'x': [0.5, 1.5], 'y': [2.5, 2.5], 'z': [1.0, 2.0]})
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    report = {
        'outputs': {
            'selected_final_depth': str(combined / 'bathy_combined_depth_conditioned.tif'),
            'support_class': str(combined / 'support_class.tif'),
        },
        'authoritative_base': {'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')}},
        'river': {'outputs': {
            'river_channel_mask': str(river_work / 'river_channel_mask.tif'),
            'longitudinal_profile_points_csv': str(river_work / 'river_longitudinal_profile_points.csv'),
            'longitudinal_profile_summary_json': str(river_work / 'river_longitudinal_profile_summary.json'),
            'channel_surface_prediction_admissibility': str(river_work / 'river_channel_surface_prediction_admissibility.tif'),
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
    assert payload['support_aware_science_summary']['available'] is True
    assert payload['support_aware_science_summary']['unsupported_by_component_class']['point_count'] == 0
    assert payload['support_aware_science_summary']['unsupported_by_prediction_admissibility']['groups']['1']['label'] == 'admissible'
    assert (out_dir / 'benchmark' / 'benchmark_support_aware_science_summary.json').exists()


def test_compute_river_scientific_summary_reports_unsupported_roughness_and_curvature_by_component_class(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    baseline = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    final = baseline.copy()
    final[1, 0] += 0.5
    final[1, 1] += 1.0
    final[1, 2] += 1.5
    final[2, 0] += 0.2
    final[2, 1] += 0.4
    final[2, 2] += 0.6
    _write_raster(out_dir / 'baseline.tif', baseline)
    _write_raster(out_dir / 'final.tif', final)

    profile = gpd.GeoDataFrame({
        'component_id': ['a','a','a','b','b','b'],
        'station_m': [0,1,2,0,1,2],
        'profile_measured_support_distance_m': [600,600,600,600,600,600],
        'component_support_class': ['unsupported_side_component']*3 + ['tiny_detached_component']*3,
        'geometry': gpd.points_from_xy([0.5,1.5,2.5,0.5,1.5,2.5], [1.5,1.5,1.5,0.5,0.5,0.5]),
    }, crs='EPSG:4326')
    profile_path = out_dir / 'profile.gpkg'
    profile.to_file(profile_path, driver='GPKG')

    coverage_summary = {'coverage_summary': {'unsupported_station_count': 6, 'authoritative_bed_supported_station_count': 0, 'authoritative_bank_margin_station_count': 0}}
    cov_path = out_dir / 'coverage_summary.json'
    cov_path.write_text(__import__('json').dumps(coverage_summary))

    summary, table_path = _compute_river_scientific_summary(
        baseline_raster=out_dir / 'baseline.tif',
        final_raster=out_dir / 'final.tif',
        profile_points_path=profile_path,
        coverage_csv_path=None,
        coverage_summary_path=cov_path,
        station_targets_path=None,
        bench_dir=out_dir,
    )

    assert summary['available'] is True
    assert table_path is not None and table_path.exists()
    assert 'unsupported_roughness_by_component_class' in summary
    assert 'unsupported_curvature_by_component_class' in summary
    assert summary['unsupported_roughness_by_component_class']['unsupported_side_component']['available'] is True
    assert summary['unsupported_curvature_by_component_class']['tiny_detached_component']['available'] is True


def test_compute_river_scientific_summary_reports_section_tendency_and_inner_relief_by_component_class(tmp_path: Path):
    out_dir = tmp_path / "out2"
    out_dir.mkdir()
    baseline = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    final = baseline.copy()
    _write_raster(out_dir / 'baseline.tif', baseline)
    _write_raster(out_dir / 'final.tif', final)

    profile = gpd.GeoDataFrame({
        'component_id': ['a','a','a','b','b','b'],
        'station_m': [0,1,2,0,1,2],
        'profile_measured_support_distance_m': [600,600,600,600,600,600],
        'component_support_class': ['unsupported_side_component']*3 + ['tiny_detached_component']*3,
        'geometry': gpd.points_from_xy([0.5,1.5,2.5,0.5,1.5,2.5], [1.5,1.5,1.5,0.5,0.5,0.5]),
    }, crs='EPSG:4326')
    profile_path = out_dir / 'profile.gpkg'
    profile.to_file(profile_path, driver='GPKG')

    station_targets = pd.DataFrame({
        'component_support_class': ['unsupported_side_component']*2 + ['tiny_detached_component']*2,
        'section_tendency_depth_m': [0.22, 0.18, 0.10, 0.08],
        'section_tendency_inner_relief_m': [1.2, 0.9, 0.4, 0.3],
        'section_tendency_confidence': [0.45, 0.40, 0.20, 0.15],
        'section_tendency_source': ['support_class_width_depth_tendency', 'weak_component_simplified_tendency', 'weak_component_simplified_tendency', 'weak_component_simplified_tendency_local_authoritative_reconciliation'],
    })
    station_targets_path = out_dir / 'river_station_targets.csv'
    station_targets.to_csv(station_targets_path, index=False)

    coverage_summary = {'coverage_summary': {'unsupported_station_count': 6, 'authoritative_bed_supported_station_count': 0, 'authoritative_bank_margin_station_count': 0}}
    cov_path = out_dir / 'coverage_summary.json'
    cov_path.write_text(__import__('json').dumps(coverage_summary))

    summary, _ = _compute_river_scientific_summary(
        baseline_raster=out_dir / 'baseline.tif',
        final_raster=out_dir / 'final.tif',
        profile_points_path=profile_path,
        coverage_csv_path=None,
        coverage_summary_path=cov_path,
        station_targets_path=station_targets_path,
        bench_dir=out_dir,
    )

    assert summary['unsupported_section_tendency_by_component_class']['unsupported_side_component']['available'] is True
    assert summary['unsupported_inner_relief_by_component_class']['tiny_detached_component']['available'] is True
    assert summary['unsupported_section_tendency_by_component_class']['tiny_detached_component']['simplified_fraction'] > 0.5
    assert summary['unsupported_inner_relief_by_component_class']['unsupported_side_component']['mean_inner_relief_m'] > summary['unsupported_inner_relief_by_component_class']['tiny_detached_component']['mean_inner_relief_m']


def test_compute_river_scientific_summary_reports_reconciliation_by_component_class(tmp_path: Path):
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    baseline_path = bench_dir / "baseline.tif"
    final_path = bench_dir / "final.tif"
    base = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    final = base.copy()
    final[1, 1] += 0.5
    _write_raster(baseline_path, base)
    _write_raster(final_path, final)

    profile = gpd.GeoDataFrame({
        'component_id': ['a','a','b','b'],
        'station_m': [0, 1, 0, 1],
        'component_support_class': ['unsupported_mainstem','unsupported_mainstem','unsupported_side_component','unsupported_side_component'],
        'authoritative_reconciliation_weight': [0.6, 0.4, 0.2, 0.1],
        'authoritative_reconciliation_delta_m': [0.5, 0.3, 0.2, 0.1],
    }, geometry=gpd.points_from_xy([0.5,1.5,0.5,1.5],[2.5,2.5,1.5,1.5]), crs='EPSG:4326')
    profile_path = bench_dir / 'profile.gpkg'
    profile.to_file(profile_path, driver='GPKG')

    summary, _ = _compute_river_scientific_summary(
        bench_dir=bench_dir,
        baseline_raster=baseline_path,
        final_raster=final_path,
        profile_points_path=profile_path,
        coverage_csv_path=None,
        coverage_summary_path=None,
        station_targets_path=None,
    )
    recon = summary['unsupported_reconciliation_by_component_class']
    assert recon['unsupported_mainstem']['mean_reconciliation_weight'] > recon['unsupported_side_component']['mean_reconciliation_weight']
    assert recon['unsupported_mainstem']['mean_abs_reconciliation_delta_m'] > recon['unsupported_side_component']['mean_abs_reconciliation_delta_m']


def test_run_workflow_benchmark_writes_river_receipt_triage_summary(tmp_path: Path):
    out_dir = tmp_path / "out"
    combined = out_dir / "combined"
    river_work = out_dir / "river"
    combined.mkdir(parents=True)
    river_work.mkdir(parents=True)

    final = np.array([[1.0, 1.3, 1.6], [2.0, 2.3, 2.6], [3.0, 3.3, 3.6]], dtype=np.float32)
    baseline = final - 0.2
    support = np.zeros_like(final, dtype=np.float32)
    prov = np.zeros_like(final, dtype=np.float32)
    river_mask = np.ones_like(final, dtype=np.float32)
    estuary_mask = np.zeros_like(final, dtype=np.float32)
    transform = from_origin(0, 3, 1, 1)
    _write_raster(combined / 'bathy_combined_depth_conditioned.tif', final)
    _write_raster(combined / 'authoritative_base_aligned.tif', baseline)
    _write_raster(combined / 'support_class.tif', support)
    _write_raster(combined / 'bathy_combined_depth_conditioned_provenance.tif', prov)
    _write_raster(river_work / 'river_channel_mask.tif', river_mask)
    _write_raster(river_work / 'estuary_clip_mask.tif', estuary_mask)

    holdout = pd.DataFrame({'x': [0.5, 1.5, 2.5], 'y': [2.5, 1.5, 0.5], 'z': [1.0, 2.3, 3.6]})
    holdout_path = tmp_path / 'holdout.csv'
    holdout.to_csv(holdout_path, index=False)

    longitudinal = pd.DataFrame({
        'profile_id': ['p1', 'p1', 'p1'],
        'station_m': [0.0, 10.0, 20.0],
        'x': [0.5, 1.5, 2.5],
        'y': [2.5, 1.5, 0.5],
        'generalized_longitudinal_bed_reconciled_elevation_m': [1.0, 2.1, 3.3],
        'component_support_class': ['unsupported_mainstem'] * 3,
    })
    longitudinal_path = river_work / 'river_longitudinal_profile_points.csv'
    longitudinal.to_csv(longitudinal_path, index=False)
    (river_work / 'river_longitudinal_profile_coverage.csv').write_text('profile_id,station_count\np1,3\n', encoding='utf-8')
    coverage_summary = {'available': True, 'coverage_summary': {'unsupported_station_count': 3, 'authoritative_bed_supported_station_count': 0, 'authoritative_bank_margin_station_count': 0}}
    (river_work / 'river_longitudinal_profile_coverage_summary.json').write_text(json.dumps(coverage_summary), encoding='utf-8')

    role_summary = {
        'available': True,
        'reason': 'prepared',
        'by_role_semantic_group': {
            'thalweg': {'count': 3, 'abs_error_m': {'p95': 0.2}},
            'inner_shape': {'count': 3, 'abs_error_m': {'p95': 0.35}},
            'bank_edge': {'count': 3, 'abs_error_m': {'p95': 0.6}},
        },
    }
    transition_summary = {'available': True, 'nonzero_cell_count': 4, 'weight_distribution': {'mean': 0.3, 'p95': 0.8}}
    smoothing_summary = {'available': True, 'adjusted_station_count': 2, 'delta_abs_m': {'p95': 0.12}, 'weight_summary': {'mean': 0.45}}
    longitudinal_smoothing_summary = {'available': True, 'eligible_count': 3, 'applied_count': 2, 'changed_count': 2, 'abs_adjustment_m': {'mean': 0.09}, 'reason': 'applied'}
    section_summary = {'available': True, 'reason': 'prepared', 'comparison_node_count': 5, 'abs_error_m': {'p95': 0.25}}
    (river_work / 'river_channel_surface_role_agreement_summary.json').write_text(json.dumps(role_summary), encoding='utf-8')
    (river_work / 'river_channel_surface_authoritative_transition_summary.json').write_text(json.dumps(transition_summary), encoding='utf-8')
    (river_work / 'river_backbone_smoothing_summary.json').write_text(json.dumps(smoothing_summary), encoding='utf-8')
    (river_work / 'river_channel_surface_longitudinal_smoothing_summary.json').write_text(json.dumps(longitudinal_smoothing_summary), encoding='utf-8')
    (river_work / 'river_channel_surface_section_target_agreement_summary.json').write_text(json.dumps(section_summary), encoding='utf-8')

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
            'river_longitudinal_profile_points': str(longitudinal_path),
            'river_longitudinal_profile_coverage': str(river_work / 'river_longitudinal_profile_coverage.csv'),
            'river_longitudinal_profile_coverage_summary': str(river_work / 'river_longitudinal_profile_coverage_summary.json'),
            'channel_surface_role_agreement_summary': str(river_work / 'river_channel_surface_role_agreement_summary.json'),
            'channel_surface_authoritative_transition_summary': str(river_work / 'river_channel_surface_authoritative_transition_summary.json'),
            'backbone_smoothing_summary': str(river_work / 'river_backbone_smoothing_summary.json'),
            'channel_surface_longitudinal_smoothing_summary': str(river_work / 'river_channel_surface_longitudinal_smoothing_summary.json'),
            'channel_surface_section_target_agreement_summary': str(river_work / 'river_channel_surface_section_target_agreement_summary.json'),
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
    triage = payload['river_receipt_triage_summary']
    assert triage['available'] is True
    assert triage['receipts']['role_agreement']['weakest_role'] == 'bank_edge'
    assert triage['lateral_accountability']['dominant_issue'] == 'bank_vs_inner_shape'
    assert triage['lateral_accountability']['weakest_role'] == 'bank_edge'
    assert triage['receipts']['authoritative_transition']['nonzero_cell_count'] == 4
    assert triage['receipts']['longitudinal_smoothing']['changed_count'] == 2
    assert triage['receipts']['role_agreement']['reason'] == 'prepared'
    assert triage['receipts']['section_target_agreement']['reason'] == 'prepared'
    mode = payload['river_benchmark_mode_summary']
    assert mode['available'] is True
    assert mode['evaluation_basis'] == 'support_aware_primary'
    assert mode['primary_decision_signal'] == 'support_aware_science_and_receipts'
    assert mode['river_specific_withheld_support_benchmark_recommended'] is True
    assert mode['holdout_support_coverage']['unsupported_holdout_point_count'] == 0
    assert '--river-withheld-support-csv' in str(mode.get('withheld_support_cli_flag'))
    active_eval = payload['river_active_evaluation_summary']
    assert active_eval['available'] is True
    assert active_eval['artifact_role'] == 'diagnostic_only'
    assert active_eval['active_river_benchmark_mode'] == 'support_aware_primary'
    assert active_eval['active_river_benchmark_decision_basis'] == 'support_aware_science_and_receipts'
    assert active_eval['diagnostic_receipt_roles']['benchmark_river_mode_summary.json'] == 'diagnostic_only'
    assert active_eval['hard_problem_holdout_blind'] is True
    focus = payload['river_primary_focus_summary']
    assert focus['available'] is True
    assert focus['hard_problem_holdout_blind'] is True
    assert focus['evaluation_basis'] == 'support_aware_primary'
    assert focus['primary_decision_signal'] == 'support_aware_science_and_receipts'
    assert focus['river_specific_withheld_support_benchmark_recommended'] is True
    assert focus['primary_focus'] in {'unsupported_roughness', 'authoritative_transition', 'centerline_alignment', 'bank_vs_inner_shape', 'inner_shape_width_propagation', 'thalweg_fit', 'section_target_fit', 'unsupported_river'}
    assert (out_dir / 'benchmark' / 'benchmark_river_receipt_triage_summary.json').exists()
    assert (out_dir / 'benchmark' / 'benchmark_river_mode_summary.json').exists()
    assert (out_dir / 'benchmark' / 'benchmark_river_active_evaluation_summary.json').exists()
    assert (out_dir / 'benchmark' / 'benchmark_river_primary_focus_summary.json').exists()


def test_build_river_benchmark_mode_summary_keeps_support_aware_primary_and_demotes_holdout_to_drilldown():
    support_aware = {
        'available': True,
        'hard_problem_holdout_blind': False,
        'unsupported_roughness': {'delta_mean_abs_step': -0.03, 'station_count': 42},
        'transition_quality': {'delta_mean_abs_step': 0.01, 'station_count': 8},
        'unsupported_science_grid': {
            'cell_count': 30,
            'by_support_class': {'guidance_conditioned_river': {'n': 20}, 'scaffold_inferred': {'n': 10}},
        },
        'unsupported_mainstem_vs_side_component': {
            'available': True,
            'mainstem_final_mean_abs_step': 0.45,
            'side_final_mean_abs_step': 0.30,
            'mainstem_mean_abs_delta_m': 0.22,
            'side_mean_abs_delta_m': 0.15,
        },
    }
    triage = {'lateral_accountability': {'dominant_issue': 'inner_shape_width_propagation'}}
    hard = {
        'available': True,
        'river_point_count': 18,
        'hard_problem_evaluation_blind': False,
        'hard_problem_group': {'n': 7},
        'river_support_class_counts': {'guidance_conditioned_river': 4, 'scaffold_inferred': 3, 'authoritative_locked': 11},
    }
    summary = _build_river_benchmark_mode_summary(
        support_aware_science=support_aware,
        river_receipt_triage=triage,
        hard_river_summary=hard,
        auto_holdout_receipt={'path': 'auto_holdout_points.csv'},
    )
    assert summary['available'] is True
    assert summary['evaluation_basis'] == 'support_aware_primary'
    assert summary['active_river_benchmark_mode'] == 'support_aware_primary'
    assert summary['active_river_benchmark_decision_basis'] == 'support_aware_science_and_receipts'
    assert summary['primary_decision_signal'] == 'support_aware_science_and_receipts'
    assert summary['legacy_holdout_evaluation_basis'] == 'mixed_holdout_and_support_aware'
    assert summary['legacy_holdout_primary_decision_signal'] == 'hard_river_holdout_plus_support_aware_science'
    assert summary['holdout_drilldown_available'] is True
    assert 'drilldown' in str(summary.get('holdout_drilldown_reason')).lower()
    assert summary['priority_domain'] == 'unsupported_mainstem'
    assert summary['river_specific_withheld_support_benchmark_recommended'] is False
    assert summary['holdout_support_coverage']['unsupported_holdout_point_count'] == 7


def test_build_river_withheld_support_plan_prefers_interior_river_support(tmp_path: Path):
    candidate = pd.DataFrame({
        'x': [0.5, 1.5, 2.5, 3.5],
        'y': [2.5, 2.5, 2.5, 2.5],
        'depth_m': [-1.0, -2.0, -3.0, -4.0],
        'authoritative_role': ['authoritative_bed_inner', 'authoritative_bed_core', 'authoritative_bank_margin', 'authoritative_bed_inner'],
        'inside_channel_mask': [1, 1, 1, 0],
        'inside_river_guidance_domain': [1, 1, 1, 1],
        'inside_mainstem_mask': [1, 0, 1, 0],
        'inside_estuary_clip': [0, 0, 0, 0],
        'role_confidence': [0.9, 0.8, 0.9, 0.9],
    })
    candidate_path = tmp_path / 'candidate.csv'
    candidate.to_csv(candidate_path, index=False)
    bench_dir = tmp_path / 'bench'
    bench_dir.mkdir()
    baseline = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32)
    final = baseline.copy()
    _write_raster(tmp_path / 'baseline.tif', baseline)
    _write_raster(tmp_path / 'final.tif', final)

    class _Log:
        def info(self, *args, **kwargs):
            return None

    plan_path, receipt, points_epsg, x_name, y_name, z_name = _build_river_withheld_support_plan(
        candidate_path=candidate_path,
        bench_dir=bench_dir,
        baseline_raster=tmp_path / 'baseline.tif',
        final_raster=tmp_path / 'final.tif',
        points_epsg_arg=4326,
        fallback_points_epsg=None,
        x_col='x',
        y_col='y',
        z_col='depth_m',
        holdout_frac=0.5,
        holdout_min_points=1,
        holdout_seed=7,
        logger=_Log(),
    )
    assert plan_path is not None and Path(plan_path).exists()
    planned = pd.read_csv(plan_path)
    assert set(planned['authoritative_role'].astype(str)).issubset({'authoritative_bed_inner', 'authoritative_bed_core'})
    assert planned['inside_channel_mask'].astype(int).min() == 1
    assert receipt['available'] is True
    assert receipt['support_point_key_column'] == 'support_point_key'
    assert receipt['withheld_rows'] >= 1
    assert points_epsg == 4326
    assert (x_name, y_name, z_name) == ('x', 'y', 'depth_m')


def test_run_workflow_benchmark_writes_river_withheld_support_plan(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "combined"
    combined.mkdir()
    river_work = out_dir / "derived_cache" / "run1" / "river" / "work"
    river_work.mkdir(parents=True)
    cache_support = out_dir / 'cache_support.csv'
    pd.DataFrame({
        'x': [0.5, 1.5, 0.5, 1.5],
        'y': [1.5, 1.5, 0.5, 0.5],
        'depth_m': [4.0, 5.0, 6.0, 7.0],
        'authoritative_role': ['authoritative_bed_inner', 'authoritative_bed_core', 'authoritative_bed_inner', 'authoritative_bank_margin'],
        'inside_channel_mask': [1, 1, 1, 1],
        'inside_river_guidance_domain': [1, 1, 1, 1],
        'inside_mainstem_mask': [1, 1, 0, 0],
        'inside_estuary_clip': [0, 0, 0, 0],
        'role_confidence': [0.9, 0.9, 0.9, 0.9],
    }).to_csv(cache_support, index=False)

    baseline = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    final = baseline.copy()
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
        'authoritative_base': {
            'outputs': {'authoritative_aligned': str(combined / 'authoritative_base_aligned.tif')},
            'river_guidance': {'path': str(cache_support)},
        },
        'river': {'outputs': {
            'river_channel_mask': str(river_work / 'river_channel_mask.tif'),
            'estuary_clip_mask': str(river_work / 'estuary_clip_mask.tif'),
        }},
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
        benchmark_holdout_frac=0.5,
        benchmark_holdout_min_points=1,
        benchmark_holdout_seed=5,
    )
    cfg = SimpleNamespace(out_dir=out_dir, run_id='run1')

    class _Log:
        def info(self, *args, **kwargs):
            return None

    payload = run_workflow_benchmark(cfg=cfg, args=args, report=report, logger=_Log(), final_path=combined / 'bathy_combined_depth_conditioned.tif')
    receipt = payload['river_withheld_support_receipt']
    assert receipt['available'] is True
    assert Path(payload['river_withheld_support_receipt_json']).exists()
    assert Path(payload['river_withheld_support_points']).exists()
    mode = payload['river_benchmark_mode_summary']
    assert mode['withheld_support_plan_available'] is True
    assert mode['withheld_support_plan_path'] == payload['river_withheld_support_points']
    assert '--river-withheld-support-csv' in str(mode.get('withheld_support_cli_flag'))
