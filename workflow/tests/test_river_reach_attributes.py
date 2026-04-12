from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from river_reach_attributes import build_and_write_reach_attributes


def test_build_and_write_reach_attributes_with_topology_truth(tmp_path: Path):
    import geopandas as gpd
    from shapely.geometry import LineString

    river_dir = tmp_path / "river"
    river_dir.mkdir()

    profile = pd.DataFrame(
        {
            "profile_id": ["A", "A", "A", "A", "B", "B"],
            "station_m": [0.0, 100.0, 250.0, 600.0, 0.0, 200.0],
            "profile_support_class": [
                "authoritative_anchor",
                "authoritative_anchor",
                "xs_support",
                "xs_support",
                "centerline_only",
                "centerline_only",
            ],
            "profile_support_source_count": [3, 3, 2, 2, 1, 1],
            "profile_support_present": [1, 1, 1, 1, 1, 1],
            "profile_authoritative_anchor_present": [1, 1, 0, 0, 0, 0],
            "profile_xs_support_present": [0, 0, 1, 1, 0, 0],
            "profile_centerline_support_present": [1, 1, 1, 1, 1, 1],
            "profile_bank_support_present": [1, 1, 0, 0, 1, 1],
            "profile_wse_support_present": [0, 0, 0, 0, 0, 0],
            "profile_authoritative_role": [
                "authoritative_bed_core",
                "authoritative_bed_inner",
                "authoritative_bank_margin",
                "authoritative_overbank_or_ambiguous",
                "authoritative_bank_margin",
                "authoritative_overbank_or_ambiguous",
            ],
            "profile_authoritative_bed_support_present": [1, 1, 0, 0, 0, 0],
            "profile_authoritative_bank_margin_present": [0, 0, 1, 0, 1, 0],
            "profile_authoritative_bed_core_present": [1, 0, 0, 0, 0, 0],
            "profile_authoritative_ambiguous_present": [0, 0, 0, 1, 0, 1],
            "profile_authoritative_bed_support_distance_m": [0.0, 0.0, 150.0, 350.0, 999.0, 1200.0],
            "profile_far_from_authoritative_bed_support": [0, 0, 0, 0, 1, 1],
            "network_backbone_elevation_m": [10.0, 9.0, 8.7, 8.0, 6.0, 5.5],
            "network_junction_adjustment_m": [0.0, 0.0, 0.2, 0.0, 0.15, 0.0],
            "junction_hierarchy_weight": [1, 1, 0.8, 1, 0.7, 1],
            "junction_wse_weight": [1, 1, 1, 1, 1, 1],
            "drainage_area_proxy": [10, 10, 11, 12, 8, 8],
            "stream_order_proxy": [3, 3, 3, 3, 2, 2],
            "network_backbone_source": [
                "network_component_solve",
                "network_component_solve",
                "network_junction_flow_aware_solve",
                "network_component_solve",
                "network_junction_flow_aware_solve",
                "network_component_solve",
            ],
        }
    )
    profile_path = river_dir / "river_longitudinal_profile.csv"
    profile.to_csv(profile_path, index=False)

    long_summary = {
        "network_component_count": 2,
        "junction_count": 1,
    }
    long_summary_path = river_dir / "river_longitudinal_profile_summary.json"
    long_summary_path.write_text(json.dumps(long_summary), encoding="utf-8")

    edges = gpd.GeoDataFrame(
        {
            "component_id": ["A", "B"],
            "from_node": ["n1", "n2"],
            "to_node": ["n2", "n3"],
        },
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs="EPSG:4326",
    )
    edges_path = river_dir / "river_hydraulic_backbone_edges.gpkg"
    edges.to_file(edges_path, driver="GPKG")

    outputs = build_and_write_reach_attributes(
        river_dir=river_dir,
        longitudinal_profile_path=profile_path,
        longitudinal_profile_summary_path=long_summary_path,
        hydraulic_backbone_edges_path=edges_path,
        segment_length_m=200.0,
    )

    assert Path(outputs["reach_attributes"]).exists()
    assert Path(outputs["reach_components"]).exists()
    assert Path(outputs["reach_attributes_summary"]).exists()

    reach_df = pd.read_csv(outputs["reach_attributes"])
    assert not reach_df.empty
    assert set(["reach_id", "reach_role", "component_has_topology_junction"]).issubset(reach_df.columns)
    assert (reach_df["component_has_topology_junction"].astype(bool)).any()
    assert (reach_df["reach_role"].astype(str) == "junction_adjusted").any()
    assert "authoritative_bed_support_fraction" in reach_df.columns
    assert "authoritative_bank_margin_fraction" in reach_df.columns
    assert "primary_authoritative_role" in reach_df.columns

    summary = json.loads(Path(outputs["reach_attributes_summary"]).read_text(encoding="utf-8"))
    assert summary["component_count"] == 2
    assert summary["topology_truth"]["available"] is True
    assert summary["topology_truth"]["junction_node_count"] == 1
    assert summary["topology_truth"]["junction_count_matches_longitudinal_summary"] is True
    assert summary["authoritative_bed_support_fraction_summary"]["n"] > 0
    assert summary["primary_authoritative_role_counts"]["authoritative_bank_margin"] >= 1
