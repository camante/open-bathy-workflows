from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Point

from hydrologic_solve_domain import build_hydrologic_solve_domain_artifacts, select_major_system_component


def test_select_major_system_component_prefers_drainage_and_length():
    edges = gpd.GeoDataFrame(
        {
            "component_id": ["1", "1", "2"],
            "length_m": [100.0, 120.0, 500.0],
            "totdasqkm": [20.0, 20.0, 5.0],
            "streamorde": [6, 6, 4],
            "root_node": [1, 1, 10],
            "s_m_min": [0.0, 100.0, 0.0],
            "from_node": [1, 2, 10],
            "to_node": [2, 3, 11],
            "geometry": [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)]), LineString([(10, 0), (15, 0)])],
        },
        crs="EPSG:32619",
    )
    major_component, rankings = select_major_system_component(edges)
    assert major_component == "1"
    assert rankings[0]["component_id"] == "1"


def test_build_hydrologic_solve_domain_artifacts_writes_major_system_and_anchors():
    edges = gpd.GeoDataFrame(
        {
            "component_id": ["1", "1", "2"],
            "length_m": [100.0, 120.0, 500.0],
            "totdasqkm": [20.0, 20.0, 5.0],
            "streamorde": [6, 6, 4],
            "root_node": [1, 1, 10],
            "s_m_min": [0.0, 100.0, 0.0],
            "from_node": [1, 2, 10],
            "to_node": [2, 3, 11],
            "river_id": ["a", "b", "c"],
            "geometry": [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)]), LineString([(10, 0), (15, 0)])],
        },
        crs="EPSG:32619",
    )
    nodes = gpd.GeoDataFrame(
        {"node_id": [1, 2, 3, 10, 11], "geometry": [Point(0, 0), Point(1, 0), Point(2, 0), Point(10, 0), Point(15, 0)]},
        crs="EPSG:32619",
    )
    artifacts = build_hydrologic_solve_domain_artifacts(
        solve_network=edges,
        nodes=nodes,
        export_aoi="0/1/2/3",
        solve_aoi="0/1/2/3",
        scaffold_aoi="0/1/2/3",
        solve_halo_km=2.0,
        trusted_halo_m=60.0,
        solve_domain_role="solve",
        export_domain_role="export",
        scaffold_domain_role="scaffold",
        solve_domain_rationale="because stable",
    )
    assert artifacts.hydrologic_manifest["major_system_id"] == "1"
    assert len(artifacts.major_system_network) == 2
    assert len(artifacts.outlet_anchors) == 2
    assert artifacts.hydrologic_manifest["outlet_anchor_layer_name"] == "outlet_anchors"
