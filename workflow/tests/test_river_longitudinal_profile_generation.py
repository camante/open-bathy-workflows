from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from river_longitudinal_profile import _apply_active_core_support, _apply_bank_longitudinal_reference, _assign_profile_values_to_points, _classify_component_support_regime, _pava_isotonic, _solve_component_profile, build_and_write_longitudinal_profile
from river_bank_longitudinal_fit import fit_bank_longitudinal_points


def _write_raster(path: Path, arr: np.ndarray, *, nodata: float = -9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=str(arr.dtype),
        crs="EPSG:32619",
        transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)
    return path


def test_apply_active_core_support_handles_missing_authoritative_support_columns():
    import pandas as pd

    profile = pd.DataFrame({
        "profile_id": ["a", "a", "a"],
        "station_m": [0.0, 10.0, 20.0],
        "network_backbone_elevation_m": [5.0, 4.5, 4.0],
        "network_backbone_uncertainty_m": [0.2, 0.2, 0.2],
        "bank_pair_fit_z_m": [8.0, 8.0, 8.0],
        "bank_profile_monotone_m": [8.0, 8.0, 8.0],
        "bank_elevation_m": [8.0, 8.0, 8.0],
        "authoritative_anchor_present": [False, False, False],
        "profile_inside_fluvial_monotone_domain": [True, True, True],
    })
    updated, diagnostics = _apply_active_core_support(profile, {"a": {"station_direction": "increasing_station_downstream"}})
    assert not updated.empty
    assert "a" in diagnostics
    assert np.isfinite(updated["active_core_support_elevation_m"].to_numpy(dtype=float)).all()


def test_apply_active_core_support_exports_explicit_authoritative_reconciliation_fields():
    import pandas as pd

    profile = pd.DataFrame({
        "profile_id": ["a", "a", "a"],
        "station_m": [0.0, 10.0, 20.0],
        "network_backbone_elevation_m": [5.0, 4.5, 4.0],
        "network_backbone_uncertainty_m": [0.2, 0.2, 0.2],
        "authoritative_bed_support_point_z_m": [np.nan, 4.1, np.nan],
        "profile_authoritative_bed_support_distance_m": [np.nan, 15.0, 300.0],
        "authoritative_anchor_present": [False, False, False],
        "profile_inside_fluvial_monotone_domain": [True, True, True],
    })
    updated, diagnostics = _apply_active_core_support(profile, {"a": {"station_direction": "increasing_station_downstream"}})
    assert "authoritative_reconciliation_delta_m" in updated.columns
    assert "generalized_longitudinal_bed_reconciled_elevation_m" in updated.columns
    assert abs(float(updated.loc[1, "authoritative_reconciliation_delta_m"])) > 0.0
    assert diagnostics["a"]["authoritative_reconciliation_observation_count"] == 1




def test_apply_active_core_support_uses_point_distance_fallback_for_reconciliation():
    import pandas as pd

    profile = pd.DataFrame({
        "profile_id": ["a", "a", "a"],
        "station_m": [0.0, 10.0, 20.0],
        "network_backbone_elevation_m": [5.0, 4.5, 4.0],
        "network_backbone_uncertainty_m": [0.2, 0.2, 0.2],
        "authoritative_bed_support_point_z_m": [np.nan, 4.1, np.nan],
        "authoritative_bed_support_point_distance_m": [np.nan, 15.0, 300.0],
        "authoritative_anchor_present": [False, False, False],
        "profile_inside_fluvial_monotone_domain": [True, True, True],
    })
    updated, diagnostics = _apply_active_core_support(profile, {"a": {"station_direction": "increasing_station_downstream"}})
    assert abs(float(updated.loc[1, "authoritative_reconciliation_delta_m"])) > 0.0
    assert diagnostics["a"]["authoritative_reconciliation_observation_count"] == 1
def test_bank_longitudinal_reference_does_not_write_back_into_bed():
    import pandas as pd

    profile = pd.DataFrame({
        "profile_id": ["a", "a", "a"],
        "station_m": [0.0, 10.0, 20.0],
        "bed_elevation_m": [4.0, 4.0, 4.0],
        "bank_elevation_m": [20.0, 20.0, 20.0],
        "bank_pair_fit_z_m": [20.0, 20.0, 20.0],
        "authoritative_anchor_elevation_m": [np.nan, np.nan, np.nan],
        "authoritative_anchor_present": [False, False, False],
        "xs_support_elevation_m": [3.8, 3.8, 3.8],
        "centerline_elevation_m": [4.0, 4.0, 4.0],
        "bank_weight": [1.0, 1.0, 1.0],
        "xs_support_weight": [1.0, 1.0, 1.0],
        "centerline_weight": [1.0, 1.0, 1.0],
        "component_spread_m": [0.0, 0.0, 0.0],
        "dominant_source": ["centerline", "centerline", "centerline"],
    })
    updated, diagnostics = _apply_bank_longitudinal_reference(profile, {"a": {"station_direction": "increasing_station_downstream"}})
    assert diagnostics["a"]["bed_adjustment_max_m"] == 0.0
    assert np.allclose(updated["bed_elevation_m"].to_numpy(dtype=float), [4.0, 4.0, 4.0])
    assert np.isfinite(updated["bank_bed_reference_m"].to_numpy(dtype=float)).all()


def test_build_and_write_longitudinal_profile_writes_profile_and_rasters(tmp_path: Path):
    river_dir = tmp_path / "river"
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
            "centerline_z_m": [5.0, 5.5, 6.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts.to_file(cpts, driver="GPKG")

    base = np.array([[np.nan, np.nan, np.nan], [5.0, 5.5, 6.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", base),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[0.0, 10.0, 20.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", np.array([[np.nan, np.nan, np.nan], [4.8, 5.3, 5.8], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", np.array([[np.nan, np.nan, np.nan], [5.2, 5.7, 6.2], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        authoritative_support_depth_path=_write_raster(river_dir / "river_authoritative_support_depth.tif", np.array([[np.nan, np.nan, np.nan], [-1.0, -1.0, -1.0], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )
    assert Path(outputs["longitudinal_profile"]).exists()
    assert Path(outputs["longitudinal_profile_elevation"]).exists()
    assert Path(outputs["longitudinal_profile_uncertainty"]).exists()
    assert Path(outputs["hydraulic_backbone"]).exists()
    assert Path(outputs["authoritative_reconciliation_field"]).exists()
    assert Path(outputs["authoritative_reconciliation_points"]).exists()


def test_longitudinal_profile_uses_direct_authoritative_dem_anchors(tmp_path: Path):
    river_dir = tmp_path / "river"
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts.to_file(cpts, driver="GPKG")

    ones = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    center = np.array([[5.0, 5.5, 6.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    auth_dem = np.array([[4.2, np.nan, 4.8], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    auth_support = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[0.0, 10.0, 20.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", center),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", center),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        authoritative_support_depth_path=_write_raster(river_dir / "river_authoritative_support_depth.tif", np.array([[np.nan, np.nan, np.nan], [-1.0, -1.0, -1.0], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        authoritative_bed_elevation_path=_write_raster(river_dir / "river_authoritative_bed.tif", auth_dem),
        authoritative_support_mask_path=_write_raster(river_dir / "river_authoritative_support.tif", auth_support),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )
    profile = __import__("pandas").read_csv(outputs["hydraulic_backbone"])
    assert np.isclose(profile.loc[profile["station_m"] == 0.0, "network_backbone_elevation_m"].iloc[0], 4.2)
    assert np.isclose(profile.loc[profile["station_m"] == 20.0, "network_backbone_elevation_m"].iloc[0], 4.8)


def test_longitudinal_profile_prefers_direct_authoritative_centerline_points_and_reports_station_metadata(tmp_path: Path):
    import json
    import pandas as pd

    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
            "centerline_z_m": [5.0, 5.5, 6.0],
            "centerline_sample_source": ["authoritative_support_point", "authoritative_baseline_dem", "authoritative_support_point"],
            "centerline_authoritative_support_applied": [True, False, True],
            "centerline_authoritative_support_distance_m": [2.0, np.nan, 3.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    noisy_center = np.array([[9.0, 9.0, 9.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    half = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", noisy_center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", half),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[0.0, 10.0, 20.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", np.full((3, 3), np.nan, dtype=np.float32)),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", np.zeros((3, 3), dtype=np.float32)),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", np.full((3, 3), np.nan, dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", np.zeros((3, 3), dtype=np.float32)),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", np.ones((3, 3), dtype=np.float32)),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", np.ones((3, 3), dtype=np.float32)),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", np.ones((3, 3), dtype=np.float32)),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", np.ones((3, 3), dtype=np.float32)),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )

    profile = pd.read_csv(outputs["longitudinal_profile"]).sort_values("station_m").reset_index(drop=True)
    assert np.isclose(profile.loc[0, "centerline_elevation_m"], 5.0)
    assert np.isclose(profile.loc[1, "centerline_elevation_m"], 9.0)
    assert np.isclose(profile.loc[2, "centerline_elevation_m"], 6.0)
    assert np.isclose(profile.loc[0, "centerline_weight"], 0.75)
    assert np.isclose(profile.loc[1, "centerline_weight"], 0.5)
    assert bool(profile.loc[0, "centerline_authoritative_support_station_present"])
    assert str(profile.loc[0, "centerline_sample_source"]) == "authoritative_support_point"

    summary = json.loads(Path(outputs["longitudinal_profile_summary"]).read_text(encoding="utf-8"))
    coverage_summary = summary["coverage_summary"]
    assert coverage_summary["centerline_authoritative_support_station_count"] == 2
    assert coverage_summary["centerline_sample_source_counts"]["authoritative_support_point"] == 2


def test_hydraulic_backbone_nodes_use_component_specific_endpoints(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 2, 2],
            "station_m": [0.0, 10.0, 0.0, 10.0],
        },
        geometry=[Point(0.5, 3.5), Point(1.5, 3.5), Point(10.5, 1.5), Point(11.5, 1.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    base = np.full((4, 12), np.nan, dtype=np.float32)
    base[0, 0:2] = [5.0, 5.5]
    base[2, 10:12] = [2.0, 2.5]
    ones = np.where(np.isfinite(base), 1.0, 0.0).astype(np.float32)
    corr = np.where(np.isfinite(base), 1, 0).astype(np.uint8)

    edges = gpd.GeoDataFrame(
        {
            "component_id": [1, 2],
            "from_node": [100, 200],
            "to_node": [101, 201],
            "s_m_from": [0.0, 0.0],
            "s_m_to": [10.0, 10.0],
        },
        geometry=[Point(0, 0), Point(0, 0)],
        crs="EPSG:32619",
    )
    edges_path = river_dir / "network_edges.gpkg"
    edges.to_file(edges_path, driver="GPKG")

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", base),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", base),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", base),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", base),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
        network_edges_path=edges_path,
    )

    nodes = gpd.read_file(outputs["hydraulic_backbone_nodes"])
    comp1_start = nodes[(nodes["component_id"] == "1") & (nodes["position"] == "start")].iloc[0]
    comp2_start = nodes[(nodes["component_id"] == "2") & (nodes["position"] == "start")].iloc[0]
    assert np.isclose(comp1_start.geometry.x, 0.5)
    assert np.isclose(comp2_start.geometry.x, 10.5)


def test_longitudinal_profile_drops_detached_tiny_component_from_raster(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1, 2],
            "station_m": [0.0, 10.0, 20.0, 0.0],
            "centerline_z_m": [5.0, 5.5, 6.0, -3000.0],
        },
        geometry=[Point(0.5, 3.5), Point(1.5, 3.5), Point(2.5, 3.5), Point(4.5, 0.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    base = np.full((5, 5), np.nan, dtype=np.float32)
    base[1, 0:3] = [5.0, 5.5, 6.0]
    base[4, 4] = -3000.0
    ones = np.where(np.isfinite(base), 1.0, 0.0).astype(np.float32)
    corr = np.where(np.isfinite(base), 1, 0).astype(np.uint8)

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", base),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.where(np.isfinite(base), np.array([[np.nan]*5, [0.0, 10.0, 20.0, np.nan, np.nan], [np.nan]*5, [np.nan]*5, [np.nan, np.nan, np.nan, np.nan, 0.0]], dtype=np.float32), np.nan)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", base),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", base),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )

    with rasterio.open(outputs["longitudinal_profile_elevation"]) as ds:
        arr = ds.read(1, masked=True).filled(np.nan)
    assert np.isnan(arr[4, 4])
    assert np.isfinite(arr[1, 1])


def test_longitudinal_profile_does_not_let_bank_only_shape_raise_bed(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
            "centerline_z_m": [4.0, 4.0, 4.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    center = np.array([[np.nan, np.nan, np.nan], [4.0, 4.0, 4.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs = np.array([[np.nan, np.nan, np.nan], [3.8, 3.8, 3.8], [np.nan, np.nan, np.nan]], dtype=np.float32)
    bank = np.array([[np.nan, np.nan, np.nan], [20.0, 20.0, 20.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[0.0, 10.0, 20.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", xs),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", bank),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )

    profile = __import__("pandas").read_csv(outputs["hydraulic_backbone"])
    assert np.all(profile["network_backbone_elevation_m"].to_numpy(dtype=float) < 5.0)


def test_bank_longitudinal_fit_monotone_and_profile_outputs(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
            "centerline_z_m": [4.0, 4.0, 4.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    bank_points = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1, 1, 1, 1],
            "river_id": ["r1"] * 6,
            "side": ["left", "left", "left", "right", "right", "right"],
            "s_center_m": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            "bank_z_raw_m": [10.0, 15.0, 11.0, 12.0, 17.0, 13.0],
            "bank_z_m": [10.0, 15.0, 11.0, 12.0, 17.0, 13.0],
            "bank_z_final_m": [10.0, 15.0, 11.0, 12.0, 17.0, 13.0],
            "bank_longitudinal_ref_m": [10.0, 14.0, 11.0, 12.0, 16.0, 13.0],
            "continuity_weight": [1.0] * 6,
        },
        geometry=[Point(0.0, 2.0), Point(1.0, 2.0), Point(2.0, 2.0), Point(0.0, 3.0), Point(1.0, 3.0), Point(2.0, 3.0)],
        crs="EPSG:32619",
    )
    bank_points_path = river_dir / "river_bank_points.gpkg"
    bank_points.to_file(bank_points_path, driver="GPKG")

    fit_points, fit_diag = fit_bank_longitudinal_points(bank_points, endpoint_meta={"1": {"station_direction": "increasing_station_downstream"}})
    left_fit = fit_points.loc[fit_points["side"] == "left"].sort_values("s_center_m")["bank_fit_monotone_m"].to_numpy(dtype=float)
    right_fit = fit_points.loc[fit_points["side"] == "right"].sort_values("s_center_m")["bank_fit_monotone_m"].to_numpy(dtype=float)
    assert np.all(np.diff(left_fit) <= 1.0e-6)
    assert np.all(np.diff(right_fit) <= 1.0e-6)
    assert fit_diag["1:left"]["post_monotone_violation_count"] == 0
    assert fit_diag["1:right"]["post_monotone_violation_count"] == 0

    center = np.array([[np.nan, np.nan, np.nan], [4.0, 4.0, 4.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    xs = np.array([[np.nan, np.nan, np.nan], [3.8, 3.8, 3.8], [np.nan, np.nan, np.nan]], dtype=np.float32)
    bank = np.array([[np.nan, np.nan, np.nan], [9.0, 15.0, 10.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[0.0, 10.0, 20.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", xs),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", bank),
        bank_points_path=bank_points_path,
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )

    assert Path(outputs["bank_longitudinal_fit_points"]).exists()
    assert Path(outputs["bank_longitudinal_fit_summary"]).exists()
    assert Path(outputs["left_bank_fit_elevation"]).exists()
    assert Path(outputs["right_bank_fit_elevation"]).exists()
    assert Path(outputs["bank_pair_fit_elevation"]).exists()

    profile = __import__("pandas").read_csv(outputs["hydraulic_backbone"])
    assert "bank_pair_fit_z_m" in profile.columns
    assert np.all(np.diff(profile["bank_pair_fit_z_m"].to_numpy(dtype=float)) <= 1.0e-6)
    assert set(profile["bank_profile_active_source"].astype(str).unique()) == {"bank_pair_fit"}


def test_row_support_reference_handles_all_nan_rows_without_warning():
    import pandas as pd
    from river_longitudinal_profile import _row_support_reference

    profile = pd.DataFrame(
        {
            "authoritative_anchor_elevation_m": [np.nan, 4.0],
            "bank_bed_reference_m": [np.nan, 4.1],
            "centerline_elevation_m": [np.nan, np.nan],
            "xs_support_elevation_m": [np.nan, 3.9],
            "bank_elevation_m": [np.nan, 4.2],
        }
    )
    ref, spread = _row_support_reference(profile)
    assert np.isnan(ref[0])
    assert np.isnan(spread[0])
    assert np.isfinite(ref[1])
    assert np.isfinite(spread[1])


def test_longitudinal_profile_writes_coverage_artifact_and_summary(tmp_path: Path):
    import pandas as pd
    import json

    river_dir = tmp_path / "river"
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
            "centerline_z_m": [5.0, 5.5, 6.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts.to_file(cpts, driver="GPKG")

    center = np.array([[5.0, np.nan, 6.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[0.0, 10.0, 20.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_elevation_path=None,
        xs_support_weight_path=None,
        bank_elevation_path=None,
        bank_influence_path=None,
        bank_graph_confidence_path=None,
        bank_continuity_weight_path=None,
        bank_confluence_damping_path=None,
        bank_estuary_side_decay_path=None,
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )
    coverage = pd.read_csv(outputs["longitudinal_profile_coverage"])
    assert "profile_support_class" in coverage.columns
    assert (coverage["profile_support_class"] == "unsupported").sum() >= 1
    summary = json.loads((river_dir / "river_longitudinal_profile_summary.json").read_text())
    assert summary["coverage_summary"]["unsupported_station_count"] >= 1



def test_longitudinal_profile_ingests_roleaware_authoritative_support_points(tmp_path: Path):
    import pandas as pd

    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
            "centerline_z_m": [4.0, 3.8, 3.6],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(7.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    base = np.full((3, 8), np.nan, dtype=np.float32)
    base[0, 0] = 4.0
    base[0, 1] = 3.8
    base[0, 7] = 3.6
    ones = np.where(np.isfinite(base), 1.0, 0.0).astype(np.float32)
    corr = np.where(np.isfinite(base), 1, 0).astype(np.uint8)

    support_csv = river_dir / "authoritative_river_support.csv"
    support_csv.write_text(
        "x,y,depth_m,source,authoritative_role,role_confidence,distance_to_bank_m,component_half_width_est_m,normalized_channel_position\n"
        "0.5,2.5,4.0,authoritative_base,authoritative_bed_core,0.90,3.0,4.0,0.80\n"
        "1.5,2.5,3.8,authoritative_base,authoritative_bank_margin,0.70,0.5,4.0,0.10\n",
        encoding="utf-8",
    )

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", base),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(
            river_dir / "river_centerline_stationing_m.tif",
            np.array([[0.0, 10.0, np.nan, np.nan, np.nan, np.nan, np.nan, 20.0], [np.nan] * 8, [np.nan] * 8], dtype=np.float32),
        ),
        xs_support_elevation_path=None,
        xs_support_weight_path=None,
        bank_elevation_path=None,
        bank_influence_path=None,
        bank_graph_confidence_path=None,
        bank_continuity_weight_path=None,
        bank_confluence_damping_path=None,
        bank_estuary_side_decay_path=None,
        authoritative_support_points_path=support_csv,
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )

    profile = pd.read_csv(outputs["longitudinal_profile"])
    coverage = pd.read_csv(outputs["longitudinal_profile_coverage"])

    assert set(profile["profile_authoritative_role"].astype(str)) >= {"authoritative_bed_core", "authoritative_bank_margin"}
    first = coverage.loc[coverage["station_m"] == 0.0].iloc[0]
    second = coverage.loc[coverage["station_m"] == 10.0].iloc[0]
    third = coverage.loc[coverage["station_m"] == 20.0].iloc[0]
    assert bool(first["profile_authoritative_bed_support_present"]) is True
    assert second["profile_authoritative_role"] == "authoritative_bank_margin"
    assert float(first["profile_authoritative_bed_support_distance_m"]) == 0.0
    assert float(third["profile_authoritative_bed_support_distance_m"]) > 5.0
    assert third["profile_authoritative_role"] == "authoritative_overbank_or_ambiguous"


def test_longitudinal_profile_exports_explicit_monotone_active_core_support(tmp_path: Path):
    import pandas as pd

    river_dir = tmp_path / "river_active"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1, 1],
            "station_m": [0.0, 10.0, 20.0, 30.0],
            "centerline_z_m": [5.0, 4.4, 4.8, 3.7],
        },
        geometry=[Point(0.5, 3.5), Point(1.5, 3.5), Point(2.5, 3.5), Point(3.5, 3.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    center = np.array([[np.nan, np.nan, np.nan, np.nan], [5.0, 4.4, 4.8, 3.7], [np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", center),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[np.nan]*4, [0.0, 10.0, 20.0, 30.0], [np.nan]*4, [np.nan]*4], dtype=np.float32)),
        xs_support_elevation_path=None,
        xs_support_weight_path=None,
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", center + 0.8),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )

    profile = pd.read_csv(outputs["hydraulic_backbone"])
    active = profile["active_core_support_elevation_m"].to_numpy(dtype=float)
    assert np.all(np.diff(active[np.isfinite(active)]) <= 1.0e-6)
    assert Path(outputs["active_core_support_elevation"]).exists()
    pts_gdf = gpd.read_file(outputs["longitudinal_profile_points"])
    assert "active_core_support_elevation_m" in pts_gdf.columns


def test_pava_isotonic_honors_weights():
    y = np.array([0.0, 10.0, 1.0], dtype=float)
    unweighted = _pava_isotonic(y, increasing=True)
    weighted = _pava_isotonic(y, increasing=True, weights=np.array([1.0, 1000.0, 1.0], dtype=float))
    assert np.all(np.diff(weighted) >= -1.0e-9)
    assert weighted[1] > 9.9
    assert weighted[2] > 9.9
    assert weighted[1] > unweighted[1]


def test_longitudinal_profile_promotes_nearby_roleaware_bed_support_to_anchor(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1, 1],
            "station_m": [0.0, 10.0, 20.0],
            "centerline_z_m": [5.0, 5.5, 6.0],
        },
        geometry=[Point(0.5, 2.5), Point(1.5, 2.5), Point(2.5, 2.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    base = np.array([[np.nan, np.nan, np.nan], [5.0, 5.5, 6.0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    ones = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    corr = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    auth_dem = np.full((3, 3), np.nan, dtype=np.float32)
    auth_support = np.zeros((3, 3), dtype=np.float32)
    support_csv = river_dir / "authoritative_river_soundings.csv"
    support_csv.write_text(
        "x,y,depth_m,source,authoritative_role,role_confidence,distance_to_bank_m,component_half_width_est_m,normalized_channel_position,inside_channel_mask,inside_river_guidance_domain,inside_mainstem_mask,inside_estuary_clip\n"
        "0.5000000000,2.5000000000,4.200000,authoritative_base,authoritative_bed_inner,0.95,3.000,12.000,0.50,1,1,1,0\n"
        "2.5000000000,2.5000000000,4.800000,authoritative_base,authoritative_bed_core,0.95,4.000,12.000,0.50,1,1,1,0\n",
        encoding="utf-8",
    )

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", base),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[0.0, 10.0, 20.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", base),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", np.array([[np.nan, np.nan, np.nan], [5.2, 5.7, 6.2], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        authoritative_support_depth_path=_write_raster(river_dir / "river_authoritative_support_depth.tif", np.array([[np.nan, np.nan, np.nan], [-1.0, -1.0, -1.0], [np.nan, np.nan, np.nan]], dtype=np.float32)),
        authoritative_bed_elevation_path=_write_raster(river_dir / "river_authoritative_bed.tif", auth_dem),
        authoritative_support_mask_path=_write_raster(river_dir / "river_authoritative_support.tif", auth_support),
        authoritative_support_points_path=support_csv,
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )
    profile = __import__("pandas").read_csv(outputs["longitudinal_profile"])
    assert bool(profile.loc[profile["station_m"] == 0.0, "authoritative_anchor_present"].iloc[0])
    assert profile.loc[profile["station_m"] == 0.0, "authoritative_anchor_source"].iloc[0] == "authoritative_bed_support_point"
    assert np.isclose(profile.loc[profile["station_m"] == 20.0, "authoritative_anchor_elevation_m"].iloc[0], 4.8)


def test_longitudinal_profile_uses_explicit_estuary_clip_for_monotone_domain(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1, 1],
            "station_m": [0.0, 10.0],
            "centerline_z_m": [5.0, 4.8],
        },
        geometry=[Point(0.5, 1.5), Point(1.5, 1.5)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    base = np.array([[np.nan, np.nan], [5.0, 4.8]], dtype=np.float32)
    ones = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    corr = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    support_csv = river_dir / "authoritative_river_soundings.csv"
    support_csv.write_text(
        "x,y,depth_m,source,authoritative_role,role_confidence,distance_to_bank_m,component_half_width_est_m,normalized_channel_position,inside_channel_mask,inside_river_guidance_domain,inside_mainstem_mask,inside_estuary_clip\n"
        "0.5000000000,1.5000000000,4.900000,authoritative_base,authoritative_bank_margin,0.80,2.000,10.000,0.80,1,1,1,1\n",
        encoding="utf-8",
    )

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", base),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.array([[np.nan, np.nan], [0.0, 10.0]], dtype=np.float32)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", base),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", base),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)),
        authoritative_support_points_path=support_csv,
        authoritative_support_depth_path=_write_raster(river_dir / "river_authoritative_support_depth.tif", base),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )
    profile = __import__("pandas").read_csv(outputs["longitudinal_profile"])
    assert not bool(profile.loc[0, "profile_inside_fluvial_monotone_domain"])
    assert profile.loc[0, "profile_fluvial_monotone_domain_source"] == "explicit_estuary_clip"


def test_load_bank_points_defaults_missing_continuity_weight(tmp_path: Path):
    bank_points_path = tmp_path / "bank_points.gpkg"
    gdf = gpd.GeoDataFrame(
        {
            "component_id": [1, 1],
            "side": ["left", "right"],
            "s_center_m": [0.0, 10.0],
            "bank_z_m": [5.0, 4.8],
        },
        geometry=[Point(0.5, 0.5), Point(1.5, 0.5)],
        crs="EPSG:32619",
    )
    gdf.to_file(bank_points_path, driver="GPKG")
    loaded = __import__("river_bank_longitudinal_fit").load_bank_points(bank_points_path)
    assert "continuity_weight" in loaded.columns
    assert np.allclose(loaded["continuity_weight"].to_numpy(dtype=float), 1.0)



def test_longitudinal_profile_builds_stationized_authoritative_bed_anchor_curve(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir(parents=True, exist_ok=True)
    stations = [float(i * 10) for i in range(10)]
    pts = gpd.GeoDataFrame(
        {
            "component_id": [1] * 10,
            "station_m": stations,
            "centerline_z_m": [6.0, 5.8, 5.6, 5.5, 5.7, 5.3, 5.1, 4.9, 4.7, 4.5],
        },
        geometry=[Point(0.5 + i, 2.5) for i in range(10)],
        crs="EPSG:32619",
    )
    cpts = river_dir / "river_centerline_points.gpkg"
    pts.to_file(cpts, driver="GPKG")

    base = np.full((3, 10), np.nan, dtype=np.float32)
    center = np.array([6.0, 5.8, 5.6, 5.5, 5.7, 5.3, 5.1, 4.9, 4.7, 4.5], dtype=np.float32)
    base[0, :] = center
    ones = np.zeros((3, 10), dtype=np.float32)
    ones[0, :] = 1.0
    corr = np.zeros((3, 10), dtype=np.uint8)
    corr[0, :] = 1
    auth_dem = np.full((3, 10), np.nan, dtype=np.float32)
    auth_dem[0, :] = np.array([5.2, 5.0, 4.9, 4.8, 4.95, 4.6, 4.4, 4.3, 4.1, 4.0], dtype=np.float32)
    auth_support = np.zeros((3, 10), dtype=np.float32)
    auth_support[0, :] = 1.0

    outputs = build_and_write_longitudinal_profile(
        river_dir=river_dir,
        centerline_points_path=cpts,
        centerline_elevation_path=_write_raster(river_dir / "river_centerline_elevation.tif", base),
        centerline_influence_path=_write_raster(river_dir / "river_centerline_influence.tif", ones),
        centerline_stationing_path=_write_raster(river_dir / "river_centerline_stationing_m.tif", np.where(np.isfinite(base), np.array([stations, [np.nan]*10, [np.nan]*10], dtype=np.float32), np.nan)),
        xs_support_elevation_path=_write_raster(river_dir / "river_xs_support_elevation.tif", base),
        xs_support_weight_path=_write_raster(river_dir / "river_xs_support_weight.tif", ones),
        bank_elevation_path=_write_raster(river_dir / "river_bank_elevation_xs.tif", base + 1.0),
        bank_influence_path=_write_raster(river_dir / "river_bank_influence.tif", ones),
        bank_graph_confidence_path=_write_raster(river_dir / "river_bank_graph_confidence.tif", ones),
        bank_continuity_weight_path=_write_raster(river_dir / "river_bank_continuity_weight.tif", ones),
        bank_confluence_damping_path=_write_raster(river_dir / "river_bank_confluence_damping.tif", ones),
        bank_estuary_side_decay_path=_write_raster(river_dir / "river_bank_estuary_side_decay.tif", ones),
        authoritative_support_depth_path=_write_raster(river_dir / "river_authoritative_support_depth.tif", auth_dem - 1.0),
        authoritative_bed_elevation_path=_write_raster(river_dir / "river_authoritative_bed.tif", auth_dem),
        authoritative_support_mask_path=_write_raster(river_dir / "river_authoritative_support.tif", auth_support),
        corridor_mask_path=_write_raster(river_dir / "river_corridor_mask.tif", corr, nodata=0),
    )
    curve = __import__("pandas").read_csv(outputs["authoritative_bed_anchor_curve"])
    summary = __import__("json").loads(Path(outputs["authoritative_bed_anchor_curve_summary"]).read_text())
    profile = __import__("pandas").read_csv(outputs["longitudinal_profile"])
    assert len(curve) < 10
    assert summary["components"]["1"]["curve_anchor_rows"] < summary["components"]["1"]["raw_anchor_rows"]
    assert set(curve.columns) >= {"profile_id", "station_m", "authoritative_anchor_curve_elevation_m"}
    assert int(profile["authoritative_anchor_present"].sum()) == len(curve)


def test_active_core_support_uses_bank_pair_fit_and_anchor_curve_for_smooth_unsupported_bed():
    import pandas as pd

    profile = pd.DataFrame(
        {
            "profile_id": ["1"] * 5,
            "station_m": [0.0, 10.0, 20.0, 30.0, 40.0],
            "network_backbone_elevation_m": [8.5, 8.9, 7.2, 7.8, 6.0],
            "network_backbone_uncertainty_m": [0.30] * 5,
            "network_backbone_source": ["network_component_solve"] * 5,
            "bank_pair_fit_z_m": [10.0, 9.6, 9.2, 8.8, 8.4],
            "bank_profile_monotone_m": [10.0, 9.6, 9.2, 8.8, 8.4],
            "bank_elevation_m": [10.1, 9.7, 9.25, 8.85, 8.45],
            "authoritative_anchor_present": [True, False, False, False, True],
            "authoritative_anchor_elevation_m": [8.0, np.nan, np.nan, np.nan, 6.1],
            "authoritative_anchor_source": ["authoritative_bed_core_anchor_curve", "none", "none", "none", "authoritative_bed_core_anchor_curve"],
            "authoritative_anchor_curve_present": [True, False, False, False, True],
            "profile_inside_fluvial_monotone_domain": [True] * 5,
            "profile_authoritative_bed_support_distance_m": [0.0, 10.0, 20.0, 10.0, 0.0],
        }
    )
    out, diagnostics = _apply_active_core_support(
        profile,
        endpoint_meta={"1": {"station_direction": "increasing_station_downstream"}},
    )
    active = out["active_core_support_elevation_m"].to_numpy(dtype=float)
    expected_model = np.array([8.0, 7.525, 7.05, 6.575, 6.1], dtype=float)
    noisy_backbone = profile["network_backbone_elevation_m"].to_numpy(dtype=float)
    assert np.all(np.diff(active) <= 1.0e-6)
    assert np.isclose(active[0], 8.0)
    assert np.isclose(active[-1], 6.1)
    assert np.nanmax(np.abs(active - expected_model)) < np.nanmax(np.abs(noisy_backbone - expected_model))
    assert np.nanmean(np.abs(active - expected_model)) < 0.5
    assert str(out["active_core_support_source"].iloc[0]).startswith("authoritative")
    assert diagnostics["1"]["generalized_bed_rows"] == 5
    assert diagnostics["1"]["post_monotone_violation_count"] == 0


def test_active_core_support_offset_fit_smooths_far_unsupported_reaches_more_strongly():
    import pandas as pd

    profile = pd.DataFrame(
        {
            "profile_id": ["1"] * 7,
            "station_m": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "network_backbone_elevation_m": [8.0, 7.7, 7.6, 7.4, 7.1, 6.9, 6.7],
            "network_backbone_uncertainty_m": [0.30] * 7,
            "network_backbone_source": ["network_component_solve"] * 7,
            "bank_pair_fit_z_m": [10.5, 10.0, 9.6, 9.0, 8.6, 8.1, 7.7],
            "bank_profile_monotone_m": [10.5, 10.0, 9.6, 9.0, 8.6, 8.1, 7.7],
            "bank_elevation_m": [10.5, 10.0, 9.6, 9.0, 8.6, 8.1, 7.7],
            "authoritative_anchor_present": [True, False, False, False, False, False, True],
            "authoritative_anchor_elevation_m": [8.2, np.nan, np.nan, np.nan, np.nan, np.nan, 5.9],
            "authoritative_anchor_source": ["authoritative_bed_core_anchor_curve", "none", "none", "none", "none", "none", "authoritative_bed_core_anchor_curve"],
            "authoritative_anchor_curve_present": [True, False, False, False, False, False, True],
            "profile_inside_fluvial_monotone_domain": [True] * 7,
            "profile_authoritative_bed_support_distance_m": [0.0, 50.0, 150.0, 400.0, 700.0, 1000.0, 0.0],
        }
    )
    out, diagnostics = _apply_active_core_support(
        profile,
        endpoint_meta={"1": {"station_direction": "increasing_station_downstream"}},
    )
    active = out["active_core_support_elevation_m"].to_numpy(dtype=float)
    backbone = profile["network_backbone_elevation_m"].to_numpy(dtype=float)
    assert np.all(np.diff(active) <= 1.0e-6)
    assert float(np.nanmax(np.abs(np.diff(active[1:-1])))) <= float(np.nanmax(np.abs(np.diff(backbone[1:-1]))))
    assert np.isclose(active[0], 8.2)
    assert np.isclose(active[-1], 5.9)
    assert diagnostics["1"]["generalized_bed_rows"] == 7


def test_bank_fit_handoff_normalizes_component_ids_between_bank_points_and_profile(tmp_path: Path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    from river_bank_longitudinal_fit import load_bank_points, fit_bank_longitudinal_points, sample_bank_fit_to_profile

    bank_points = gpd.GeoDataFrame(
        pd.DataFrame({
            "component_id": [1, 1, 1, 1],
            "river_id": ["r"] * 4,
            "side": ["left", "left", "right", "right"],
            "s_center_m": [0.0, 100.0, 0.0, 100.0],
            "bank_z_m": [10.0, 9.0, 10.5, 9.5],
            "bank_z_raw_m": [10.0, 9.0, 10.5, 9.5],
        }),
        geometry=[Point(0,0), Point(1,0), Point(0,1), Point(1,1)],
        crs="EPSG:32619",
    )
    path = tmp_path / "bank_points.gpkg"
    bank_points.to_file(path, driver="GPKG")
    loaded = load_bank_points(path)
    fit_points, _ = fit_bank_longitudinal_points(loaded, endpoint_meta={"1": {"station_direction": "increasing_station_downstream"}})
    profile = pd.DataFrame({
        "profile_id": ["1.0", "1.0", "1.0"],
        "station_m": [0.0, 50.0, 100.0],
    })
    sampled, diag = sample_bank_fit_to_profile(profile, fit_points)
    assert np.count_nonzero(np.isfinite(sampled["bank_pair_fit_z_m"].to_numpy(dtype=float))) == 3
    assert diag["1"]["pair_fit_rows"] == 3


def test_active_core_support_stabilizes_far_from_support_roughness():
    import pandas as pd

    profile = pd.DataFrame({
        "profile_id": ["a"] * 8,
        "station_m": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
        "network_backbone_elevation_m": [8.0, 7.5, 7.35, 6.9, 6.75, 6.1, 6.0, 5.7],
        "network_backbone_uncertainty_m": [0.25] * 8,
        "authoritative_anchor_present": [True, False, False, False, False, False, False, True],
        "authoritative_anchor_elevation_m": [8.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5.7],
        "authoritative_anchor_source": ["authoritative_bed_core_anchor_curve", "none", "none", "none", "none", "none", "none", "authoritative_bed_core_anchor_curve"],
        "profile_inside_fluvial_monotone_domain": [True] * 8,
        "authoritative_bed_support_point_distance_m": [0.0, 50.0, 200.0, 450.0, 700.0, 900.0, 1100.0, 0.0],
    })
    updated, diagnostics = _apply_active_core_support(profile, {"a": {"station_direction": "increasing_station_downstream"}})
    out = updated["active_core_support_elevation_m"].to_numpy(dtype=float)
    base = profile["network_backbone_elevation_m"].to_numpy(dtype=float)
    far_slice = slice(2, 7)
    assert np.nanmean(np.abs(np.diff(out[far_slice]))) < np.nanmean(np.abs(np.diff(base[far_slice])))
    assert diagnostics["a"]["post_monotone_violation_count"] == 0


def test_active_core_support_treats_missing_support_distance_as_far_unsupported():
    import pandas as pd

    profile = pd.DataFrame({
        "profile_id": ["a"] * 6,
        "station_m": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        "network_backbone_elevation_m": [7.0, 6.8, 6.2, 6.0, 5.5, 5.0],
        "network_backbone_uncertainty_m": [0.25] * 6,
        "authoritative_anchor_present": [True, False, False, False, False, True],
        "authoritative_anchor_elevation_m": [7.0, np.nan, np.nan, np.nan, np.nan, 5.0],
        "authoritative_anchor_source": ["authoritative_bed_core_anchor_curve", "none", "none", "none", "none", "authoritative_bed_core_anchor_curve"],
        "profile_inside_fluvial_monotone_domain": [True] * 6,
        "authoritative_bed_support_point_distance_m": [0.0, np.nan, np.nan, np.nan, np.nan, 0.0],
    })
    updated, _ = _apply_active_core_support(profile, {"a": {"station_direction": "increasing_station_downstream"}})
    out = updated["active_core_support_elevation_m"].to_numpy(dtype=float)
    assert np.all(np.diff(out) <= 1.0e-6)
    assert np.isclose(out[0], 7.0)
    assert np.isclose(out[-1], 5.0)


def test_assign_profile_values_to_points_carries_support_semantics_without_exact_station_match():
    import pandas as pd

    centerline_points = gpd.GeoDataFrame(
        {
            "component_id": [1, 1],
            "station_m": [10.0000004, 19.9999996],
        },
        geometry=[Point(0.0, 0.0), Point(1.0, 0.0)],
        crs="EPSG:32619",
    )
    profile = pd.DataFrame({
        "profile_id": ["1", "1"],
        "station_m": [10.0, 20.0],
        "bed_elevation_m": [5.0, 4.8],
        "uncertainty_m": [0.1, 0.1],
        "network_backbone_elevation_m": [5.0, 4.8],
        "network_backbone_uncertainty_m": [0.1, 0.1],
        "network_backbone_source": ["centerline", "centerline"],
        "active_core_support_elevation_m": [4.9, 4.7],
        "active_core_support_uncertainty_m": [0.1, 0.1],
        "active_core_support_source": ["generalized_longitudinal_bed", "generalized_longitudinal_bed_local_authoritative_reconciliation"],
        "generalized_longitudinal_bed_source": ["generalized_longitudinal_bed", "generalized_longitudinal_bed_local_authoritative_reconciliation"],
        "dominant_source": ["centerline", "centerline"],
        "profile_support_class": ["centerline_supported", "unsupported"],
        "profile_authoritative_role": ["authoritative_bed_core", "authoritative_bank_margin"],
        "profile_measured_support_distance_m": [25.0, 1200.0],
        "profile_authoritative_bed_support_distance_m": [15.0, 800.0],
    })
    pts = _assign_profile_values_to_points(centerline_points, profile)
    assert pts.loc[0, "profile_support_class"] == "centerline_supported"
    assert pts.loc[1, "profile_support_class"] == "unsupported"
    assert pts.loc[1, "active_core_support_source"] == "generalized_longitudinal_bed_local_authoritative_reconciliation"
    assert pts.loc[1, "generalized_longitudinal_bed_source"] == "generalized_longitudinal_bed_local_authoritative_reconciliation"
    assert float(pts.loc[1, "profile_authoritative_bed_support_distance_m"]) == 800.0


def test_classify_component_support_regime_marks_tiny_detached_component():
    import pandas as pd

    sub = pd.DataFrame({
        "station_m": [0.0, 50.0, 100.0],
        "profile_authoritative_bed_support_present": [False, False, False],
        "authoritative_anchor_present": [False, False, False],
    })
    meta = _classify_component_support_regime(sub)
    assert meta["component_class"] == "tiny_detached_component"


def test_solve_component_profile_uses_unsupported_side_component_regime():
    import pandas as pd

    sub = pd.DataFrame({
        "station_m": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        "bed_elevation_m": [5.0, 4.7, 4.9, 4.2, 4.4, 3.7, 3.9],
        "uncertainty_m": [0.2] * 7,
        "authoritative_anchor_present": [False] * 7,
        "authoritative_anchor_elevation_m": [np.nan] * 7,
        "profile_authoritative_bed_support_present": [False] * 7,
        "authoritative_bed_support_point_distance_m": [1200.0] * 7,
        "junction_hierarchy_weight": [1.0] * 7,
        "stream_order_proxy": [1.0] * 7,
        "drainage_area_proxy": [5.0] * 7,
    })
    solved, solved_unc, meta = _solve_component_profile(sub)
    assert meta["component_class"] == "unsupported_side_component"
    assert meta["far_support_smoothing_applied"] is True
    assert np.isfinite(solved).all()
    assert np.isfinite(solved_unc).all()
