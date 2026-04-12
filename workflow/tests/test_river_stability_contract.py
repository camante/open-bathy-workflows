import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from canonical_river_scaffold import nested_aoi_relationship
from trusted_interior import summarize_trusted_export_region, trusted_overlap_identity
from seam_metrics import compute_array_overlap_identity_metrics, compute_vector_overlap_identity_metrics, compute_raster_trusted_interior_identity_metrics, compute_vector_trusted_interior_identity_metrics
from final_reporting import write_river_stability_summary, evaluate_overlap_identity_checks


def test_nested_aoi_relationship_contains_export():
    rel = nested_aoi_relationship('-71/-70.75/42.75/43', '-71.1/-70.6/42.6/43.1')
    assert rel['contains_export_aoi'] is True
    assert rel['overlap_fraction_of_export'] == 1.0


def test_trusted_export_summary_reports_contract():
    channel = np.ones((5, 5), dtype=np.uint8)
    estuary = np.zeros((5, 5), dtype=np.uint8)
    estuary[2, 2] = 1
    trusted = np.ones((5, 5), dtype=np.uint8)
    trusted[0, :] = 0
    trusted[:, 0] = 0
    trusted[2, 2] = 0
    summary = summarize_trusted_export_region(channel=channel, trusted_export_region=trusted, estuary_transition=estuary, edge_buffer_px=1)
    assert summary['trusted_excludes_estuary_transition'] is True
    assert summary['trusted_subset_of_channel'] is True
    assert summary['edge_buffer_px'] == 1


def test_trusted_overlap_identity_perfect_match():
    a = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    b = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    stats = trusted_overlap_identity(trusted_a=a, trusted_b=b)
    assert stats['jaccard'] == 1.0
    assert stats['exact_identity_fraction'] == 1.0


def test_array_overlap_identity_metrics_detect_difference():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.5]], dtype=np.float32)
    stats = compute_array_overlap_identity_metrics(a, b)
    assert stats['status'] == 'ok'
    assert stats['max_abs'] == 0.5
    assert stats['exact_identity_fraction'] < 1.0


def test_write_river_stability_summary_collects_contracts(tmp_path):
    scaffold = tmp_path / 'scaffold.json'
    scaffold.write_text(json.dumps({'solve_aoi': '-71.1/-70.6/42.6/43.1'}), encoding='utf-8')
    trusted = tmp_path / 'trusted.json'
    trusted.write_text(json.dumps({'trusted_export_pixels': 10}), encoding='utf-8')
    cfg = SimpleNamespace(out_dir=tmp_path, aoi='-71/-70.75/42.75/43')
    report = {
        'river': {'guidance': {'scaffold_domains': str(scaffold)}},
        'outputs': {'river_trusted_interior_summary': str(trusted)},
        'seams': {
            'adjacent_tile_comparisons': [{'status': 'ok'}],
            'overlap_identity_checks': [{'status': 'ok', 'max_abs': 0.0}],
        },
    }
    out = write_river_stability_summary(cfg, report)
    payload = json.loads(Path(out).read_text(encoding='utf-8'))
    assert payload['nested_aoi_relationship_to_solve_domain']['contains_export_aoi'] is True
    assert payload['all_overlap_identity_ok'] is True


def test_evaluate_overlap_identity_checks_flags_failure():
    out = evaluate_overlap_identity_checks([
        {'status': 'ok', 'artifact': 'final_depth', 'neighbor_io_manifest': 'n1', 'max_abs': 0.0},
        {'status': 'ok', 'artifact': 'support_class', 'neighbor_io_manifest': 'n1', 'max_abs': 0.25},
    ])
    assert out['all_ok'] is False
    assert len(out['failures']) == 1
    assert out['failures'][0]['artifact'] == 'support_class'


def test_write_river_stability_summary_collects_failed_overlap_eval(tmp_path):
    scaffold = tmp_path / 'scaffold.json'
    scaffold.write_text(json.dumps({'solve_aoi': '-71.1/-70.6/42.6/43.1'}), encoding='utf-8')
    trusted = tmp_path / 'trusted.json'
    trusted.write_text(json.dumps({'trusted_export_pixels': 10}), encoding='utf-8')
    cfg = SimpleNamespace(out_dir=tmp_path, aoi='-71/-70.75/42.75/43')
    report = {
        'river': {'guidance': {'scaffold_domains': str(scaffold)}},
        'outputs': {'river_trusted_interior_summary': str(trusted)},
        'seams': {
            'adjacent_tile_comparisons': [{'status': 'ok'}],
            'overlap_identity_checks': [{'status': 'ok', 'artifact': 'final_depth', 'neighbor_io_manifest': 'n1', 'max_abs': 0.01}],
        },
    }
    out = write_river_stability_summary(cfg, report)
    payload = json.loads(Path(out).read_text(encoding='utf-8'))
    assert payload['all_overlap_identity_ok'] is False
    assert payload['overlap_identity_evaluation']['failures'][0]['artifact'] == 'final_depth'


def test_vector_overlap_identity_metrics_detect_difference(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point

    a = gpd.GeoDataFrame({
        'component_id': ['c1', 'c1'],
        'station_m': [0.0, 100.0],
        'graph_backbone_z_m': [1.0, 2.0],
        'graph_solution_mode': ['prior_driven', 'junction_constrained'],
    }, geometry=[Point(0, 0), Point(1, 0)], crs='EPSG:4326')
    b = gpd.GeoDataFrame({
        'component_id': ['c1', 'c1'],
        'station_m': [0.0, 100.0],
        'graph_backbone_z_m': [1.0, 2.5],
        'graph_solution_mode': ['prior_driven', 'regularization_driven'],
    }, geometry=[Point(0, 0), Point(1, 0)], crs='EPSG:4326')
    pa = tmp_path / 'a.gpkg'
    pb = tmp_path / 'b.gpkg'
    a.to_file(pa, driver='GPKG')
    b.to_file(pb, driver='GPKG')
    stats = compute_vector_overlap_identity_metrics(pa, pb, compare_fields=('graph_backbone_z_m', 'graph_solution_mode'))
    assert stats['status'] == 'ok'
    assert stats['n_valid'] == 2
    assert stats['max_abs'] == 1.0
    assert stats['field_metrics']['graph_backbone_z_m']['max_abs'] == 0.5
    assert stats['field_metrics']['graph_solution_mode']['max_abs'] == 1.0
    assert stats['exact_identity_fraction'] < 1.0



def test_raster_trusted_interior_identity_metrics_restricts_to_common_trusted(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(0, 2, 1, 1)
    profile = dict(driver='GTiff', height=2, width=2, count=1, dtype='float32', crs='EPSG:4326', transform=transform, nodata=-9999.0)
    a = tmp_path / 'a.tif'
    b = tmp_path / 'b.tif'
    ta = tmp_path / 'ta.tif'
    tb = tmp_path / 'tb.tif'
    with rasterio.open(a, 'w', **profile) as ds:
        ds.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 1)
    with rasterio.open(b, 'w', **profile) as ds:
        ds.write(np.array([[9.0, 2.0], [3.0, 7.0]], dtype=np.float32), 1)
    mprof = dict(profile, dtype='uint8', nodata=0)
    with rasterio.open(ta, 'w', **mprof) as ds:
        ds.write(np.array([[0, 1], [1, 0]], dtype=np.uint8), 1)
    with rasterio.open(tb, 'w', **mprof) as ds:
        ds.write(np.array([[0, 1], [1, 0]], dtype=np.uint8), 1)
    stats = compute_raster_trusted_interior_identity_metrics(a, b, ta, tb)
    assert stats['status'] == 'ok'
    assert stats['trusted_overlap_pixels'] == 2
    assert stats['max_abs'] == 0.0
    assert stats['exact_identity_fraction'] == 1.0


def test_vector_trusted_interior_identity_metrics_filters_nontrusted_rows(tmp_path):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point

    profile = dict(driver='GTiff', height=2, width=2, count=1, dtype='uint8', crs='EPSG:4326', transform=from_origin(0, 2, 1, 1), nodata=0)
    ta = tmp_path / 'ta.tif'
    tb = tmp_path / 'tb.tif'
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    with rasterio.open(ta, 'w', **profile) as ds:
        ds.write(mask, 1)
    with rasterio.open(tb, 'w', **profile) as ds:
        ds.write(mask, 1)
    a = gpd.GeoDataFrame({
        'component_id': ['c1', 'c1'],
        'station_m': [0.0, 100.0],
        'graph_backbone_z_m': [1.0, 2.0],
        'graph_solution_mode': ['prior_driven', 'junction_constrained'],
    }, geometry=[Point(0.5, 1.5), Point(0.5, 0.5)], crs='EPSG:4326')
    b = gpd.GeoDataFrame({
        'component_id': ['c1', 'c1'],
        'station_m': [0.0, 100.0],
        'graph_backbone_z_m': [5.0, 2.0],
        'graph_solution_mode': ['regularization_driven', 'junction_constrained'],
    }, geometry=[Point(0.5, 1.5), Point(0.5, 0.5)], crs='EPSG:4326')
    pa = tmp_path / 'a.gpkg'; pb = tmp_path / 'b.gpkg'
    a.to_file(pa, driver='GPKG'); b.to_file(pb, driver='GPKG')
    stats = compute_vector_trusted_interior_identity_metrics(pa, pb, ta, tb, compare_fields=('graph_backbone_z_m', 'graph_solution_mode'))
    assert stats['status'] == 'ok'
    assert stats['trusted_overlap_rows'] == 1
    assert stats['field_metrics']['graph_backbone_z_m']['max_abs'] == 0.0
    assert stats['field_metrics']['graph_solution_mode']['max_abs'] == 0.0



def test_evaluate_overlap_identity_checks_no_valid_does_not_report_true():
    from final_reporting import evaluate_overlap_identity_checks

    stats = evaluate_overlap_identity_checks([
        {"status": "no_valid", "artifact": "trusted_interior::river_graph_backbone_diagnostics"}
    ])
    assert stats["checked"] == 1
    assert stats["ok_count"] == 0
    assert stats["skipped_no_valid_count"] == 1
    assert stats["all_ok"] is None


def test_evaluate_overlap_identity_checks_reports_true_only_for_real_ok_checks():
    from final_reporting import evaluate_overlap_identity_checks

    stats = evaluate_overlap_identity_checks([
        {"status": "no_valid", "artifact": "trusted_interior::river_graph_backbone_diagnostics"},
        {"status": "ok", "artifact": "trusted_interior::river_channel_surface_support_class", "max_abs": 0.0},
    ])
    assert stats["checked"] == 2
    assert stats["ok_count"] == 1
    assert stats["skipped_no_valid_count"] == 1
    assert stats["all_ok"] is True
