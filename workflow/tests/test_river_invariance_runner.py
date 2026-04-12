import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from river_invariance_runner import run_nested_aoi_river_invariance_test


def _write_raster(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    dtype = arr.dtype
    nodata = -9999.0 if np.issubdtype(dtype, np.floating) else 0
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=str(dtype),
        crs='EPSG:32619',
        transform=from_origin(0.0, 3.0, 1.0, 1.0),
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)
    return path


def _write_gpkg(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:32619')
    gdf.to_file(path, driver='GPKG')
    return path


def _write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_nested_aoi_runner_compares_graph_scaffold_and_surface(tmp_path: Path):
    trusted_small = _write_raster(tmp_path / 'small' / 'river_trusted_interior.tif', np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8))
    trusted_large = _write_raster(tmp_path / 'large' / 'river_trusted_interior.tif', np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8))
    surf_small = _write_raster(tmp_path / 'small' / 'surface.tif', np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    surf_large = _write_raster(tmp_path / 'large' / 'surface.tif', np.array([[5.0, 2.0, 9.0], [8.0, 3.0, 7.0], [6.0, 5.0, 4.0]], dtype=np.float32))
    mode_small = _write_raster(tmp_path / 'small' / 'mode.tif', np.array([[0, 4, 0], [0, 4, 0], [0, 0, 0]], dtype=np.uint8))

    base_small = _write_raster(tmp_path / 'small' / 'bed_base.tif', np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    base_large = _write_raster(tmp_path / 'large' / 'bed_base.tif', np.array([[9.0, 1.0, 8.0], [7.0, 2.0, 6.0], [5.0, 4.0, 3.0]], dtype=np.float32))
    recon_small = _write_raster(tmp_path / 'small' / 'bed_reconciled.tif', np.array([[0.0, 1.2, 0.0], [0.0, 2.1, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    recon_large = _write_raster(tmp_path / 'large' / 'bed_reconciled.tif', np.array([[5.0, 1.2, 9.0], [8.0, 2.1, 7.0], [6.0, 5.0, 4.0]], dtype=np.float32))
    recon_delta_small = _write_raster(tmp_path / 'small' / 'recon_delta.tif', np.array([[0.0, 0.2, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    recon_delta_large = _write_raster(tmp_path / 'large' / 'recon_delta.tif', np.array([[5.0, 0.2, 9.0], [8.0, 0.1, 7.0], [6.0, 5.0, 4.0]], dtype=np.float32))
    recon_inf_small = _write_raster(tmp_path / 'small' / 'recon_inf.tif', np.array([[0.0, 0.7, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    recon_inf_large = _write_raster(tmp_path / 'large' / 'recon_inf.tif', np.array([[5.0, 0.7, 9.0], [8.0, 0.8, 7.0], [6.0, 5.0, 4.0]], dtype=np.float32))
    mode_large = _write_raster(tmp_path / 'large' / 'mode.tif', np.array([[9, 4, 8], [7, 4, 6], [5, 4, 3]], dtype=np.uint8))

    graph_small = _write_gpkg(
        tmp_path / 'small' / 'graph.gpkg',
        [
            {'component_id': 'A', 'station_m': 10.0, 'graph_backbone_z_m': 2.0, 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 2.5)},
            {'component_id': 'A', 'station_m': 20.0, 'graph_backbone_z_m': 3.0, 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 1.5)},
        ],
    )
    graph_large = _write_gpkg(
        tmp_path / 'large' / 'graph.gpkg',
        [
            {'component_id': 'A', 'station_m': 10.0, 'graph_backbone_z_m': 2.0, 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 2.5)},
            {'component_id': 'A', 'station_m': 20.0, 'graph_backbone_z_m': 3.0, 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 1.5)},
            {'component_id': 'B', 'station_m': 99.0, 'graph_backbone_z_m': 99.0, 'graph_solution_mode': 'regularization_driven', 'graph_solver_support_class': 'unsupported', 'graph_unsupported_regime': 'long_gap_stiffened', 'geometry': Point(0.5, 2.5)},
        ],
    )
    nodes_small = _write_gpkg(
        tmp_path / 'small' / 'nodes.gpkg',
        [
            {'component_id': 'A', 'station_m': 10.0, 'bed_z_m': 2.0, 'z_source': 'graph_backbone', 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 2.5)},
        ],
    )
    nodes_large = _write_gpkg(
        tmp_path / 'large' / 'nodes.gpkg',
        [
            {'component_id': 'A', 'station_m': 10.0, 'bed_z_m': 2.0, 'z_source': 'graph_backbone', 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 2.5)},
            {'component_id': 'B', 'station_m': 90.0, 'bed_z_m': 5.0, 'z_source': 'graph_backbone', 'graph_solution_mode': 'regularization_driven', 'graph_solver_support_class': 'unsupported', 'graph_unsupported_regime': 'long_gap_stiffened', 'geometry': Point(0.5, 0.5)},
        ],
    )

    small_manifest = _write_manifest(tmp_path / 'small' / 'final_outputs.json', {
        'aoi': 'small',
        'river_trusted_interior': str(trusted_small),
        'river_channel_surface': str(surf_small),
        'river_channel_surface_graph_mode': str(mode_small),
        'river_generalized_longitudinal_bed_base': str(base_small),
        'river_generalized_longitudinal_bed_reconciled': str(recon_small),
        'river_longitudinal_profile_local_authoritative_reconciliation': str(recon_delta_small),
        'river_longitudinal_profile_local_authoritative_reconciliation_influence': str(recon_inf_small),
        'river_graph_backbone_diagnostics': str(graph_small),
        'support_artifacts': {'river_channel_scaffold_nodes': str(nodes_small)},
    })
    large_manifest = _write_manifest(tmp_path / 'large' / 'final_outputs.json', {
        'aoi': 'large',
        'river_trusted_interior': str(trusted_large),
        'river_channel_surface': str(surf_large),
        'river_channel_surface_graph_mode': str(mode_large),
        'river_generalized_longitudinal_bed_base': str(base_large),
        'river_generalized_longitudinal_bed_reconciled': str(recon_large),
        'river_longitudinal_profile_local_authoritative_reconciliation': str(recon_delta_large),
        'river_longitudinal_profile_local_authoritative_reconciliation_influence': str(recon_inf_large),
        'river_graph_backbone_diagnostics': str(graph_large),
        'support_artifacts': {'river_channel_scaffold_nodes': str(nodes_large)},
    })

    out = tmp_path / 'summary.json'
    payload = run_nested_aoi_river_invariance_test(
        small_final_outputs_manifest=small_manifest,
        large_final_outputs_manifest=large_manifest,
        out_path=out,
    )
    assert out.exists()
    assert payload['all_trusted_interior_identity_ok'] is True
    artifacts = {c['artifact']: c for c in payload['trusted_interior_identity_checks']}
    assert artifacts['trusted_interior::river_channel_surface']['status'] == 'ok'
    assert artifacts['trusted_interior::river_generalized_longitudinal_bed_reconciled']['status'] == 'ok'
    assert artifacts['trusted_interior::river_longitudinal_profile_local_authoritative_reconciliation']['status'] == 'ok'
    assert artifacts['trusted_interior::river_graph_backbone_diagnostics']['status'] == 'ok'
    assert artifacts['trusted_interior::river_channel_scaffold_nodes']['status'] == 'ok'


def test_nested_aoi_runner_fails_when_interior_graph_differs(tmp_path: Path):
    trusted = _write_raster(tmp_path / 'trusted.tif', np.array([[0, 1], [0, 1]], dtype=np.uint8))
    surf_a = _write_raster(tmp_path / 'a_surface.tif', np.array([[0.0, 2.0], [0.0, 3.0]], dtype=np.float32))
    surf_b = _write_raster(tmp_path / 'b_surface.tif', np.array([[0.0, 2.0], [0.0, 3.0]], dtype=np.float32))
    graph_a = _write_gpkg(
        tmp_path / 'a_graph.gpkg',
        [{'component_id': 'A', 'station_m': 10.0, 'graph_backbone_z_m': 2.0, 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 1.5)}],
    )
    graph_b = _write_gpkg(
        tmp_path / 'b_graph.gpkg',
        [{'component_id': 'A', 'station_m': 10.0, 'graph_backbone_z_m': 9.0, 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 1.5)}],
    )
    nodes_a = _write_gpkg(
        tmp_path / 'a_nodes.gpkg',
        [{'component_id': 'A', 'station_m': 10.0, 'bed_z_m': 2.0, 'z_source': 'graph_backbone', 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 1.5)}],
    )
    nodes_b = _write_gpkg(
        tmp_path / 'b_nodes.gpkg',
        [{'component_id': 'A', 'station_m': 10.0, 'bed_z_m': 2.0, 'z_source': 'graph_backbone', 'graph_solution_mode': 'prior_driven', 'graph_solver_support_class': 'graph_backbone', 'graph_unsupported_regime': 'supported', 'geometry': Point(1.5, 1.5)}],
    )

    ma = _write_manifest(tmp_path / 'a.json', {
        'aoi': 'small',
        'river_trusted_interior': str(trusted),
        'river_channel_surface': str(surf_a),
        'river_graph_backbone_diagnostics': str(graph_a),
        'support_artifacts': {'river_channel_scaffold_nodes': str(nodes_a)},
    })
    mb = _write_manifest(tmp_path / 'b.json', {
        'aoi': 'large',
        'river_trusted_interior': str(trusted),
        'river_channel_surface': str(surf_b),
        'river_graph_backbone_diagnostics': str(graph_b),
        'support_artifacts': {'river_channel_scaffold_nodes': str(nodes_b)},
    })

    payload = run_nested_aoi_river_invariance_test(
        small_final_outputs_manifest=ma,
        large_final_outputs_manifest=mb,
    )
    assert payload['all_trusted_interior_identity_ok'] is False
    failures = payload['trusted_interior_identity_evaluation']['failures']
    assert any(f['artifact'] == 'trusted_interior::river_graph_backbone_diagnostics' for f in failures)
