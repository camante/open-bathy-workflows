from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

from simple_river_bundle_b import run_simple_river_bundle_b
from simple_river_stage_contract import STAGE_RIVER_CENTERLINE, STAGE_CENTERLINE_WSE_PROXY, STAGE_CENTERLINE_AUTHORITATIVE_BED, STAGE_CENTERLINE_OBSERVED_OFFSET, mark_stage_implemented, simple_river_stage_status_placeholder


def test_mark_stage_implemented_updates_river_centerline_status():
    status = simple_river_stage_status_placeholder()
    updated = mark_stage_implemented(status, stage_id=STAGE_RIVER_CENTERLINE, output_artifact='x.gpkg', record_count=3, receipt_path='x.json')
    assert updated[STAGE_RIVER_CENTERLINE]['implemented'] is True
    assert updated[STAGE_RIVER_CENTERLINE]['output_artifact'] == 'x.gpkg'


def test_phase1_runner_returns_centerline_outputs(tmp_path):
    gdf = gpd.GeoDataFrame([
        {'point_id': 'a', 'station_m': 0.0, 'geometry': Point(0, 0), 'reach_id': 'r1'},
        {'point_id': 'b', 'station_m': 10.0, 'geometry': Point(1, 0), 'reach_id': 'r1'},
    ], geometry='geometry', crs='EPSG:32619')
    status = simple_river_stage_status_placeholder()
    result = run_simple_river_bundle_b(
        river_context={'centerline_points_gdf': gdf, 'input_artifacts': []},
        out_dir=str(tmp_path),
        stage_status=status,
        upto_stage=STAGE_RIVER_CENTERLINE,
    )
    assert (tmp_path / 'river_centerline_points.gpkg').exists()
    assert (tmp_path / 'river_centerline_points_receipt.json').exists()
    assert result['simple_river_stage_status'][STAGE_RIVER_CENTERLINE]['implemented'] is True
    assert 'centerline' in result['simple_river_stage_outputs']



def test_phase2_runner_returns_wse_outputs(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame({
        "point_id": ["a", "b"],
        "station_m": [0.0, 5.0],
        "bank_wse_proxy_monotone_m": [3.0, 2.0],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    status = simple_river_stage_status_placeholder()
    result = run_simple_river_bundle_b(river_context={"centerline_points_gdf": gdf}, out_dir=str(tmp_path), stage_status=status, upto_stage=STAGE_CENTERLINE_WSE_PROXY)
    assert result["simple_river_stage_status"][STAGE_RIVER_CENTERLINE]["implemented"] is True
    assert result["simple_river_stage_status"][STAGE_CENTERLINE_WSE_PROXY]["implemented"] is True
    assert Path(result["wse_proxy"]["output_artifact"]).exists()
    assert sorted(result['simple_river_stage_outputs'].keys()) == ['centerline', 'wse_proxy']



def test_phase3_runner_returns_authoritative_bed_outputs(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame({
        "point_id": ["a", "b"],
        "station_m": [0.0, 5.0],
        "bank_wse_proxy_monotone_m": [3.0, 2.0],
        "centerline_z_m": [1.5, 0.5],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    status = simple_river_stage_status_placeholder()
    result = run_simple_river_bundle_b(river_context={"centerline_points_gdf": gdf}, out_dir=str(tmp_path), stage_status=status, upto_stage=STAGE_CENTERLINE_AUTHORITATIVE_BED)
    assert result["simple_river_stage_status"][STAGE_RIVER_CENTERLINE]["implemented"] is True
    assert result["simple_river_stage_status"][STAGE_CENTERLINE_WSE_PROXY]["implemented"] is True
    assert result["simple_river_stage_status"][STAGE_CENTERLINE_AUTHORITATIVE_BED]["implemented"] is True
    assert Path(result["authoritative_bed"]["output_artifact"]).exists()
    assert sorted(result['simple_river_stage_outputs'].keys()) == ['authoritative_bed', 'centerline', 'wse_proxy']



def test_phase4_runner_returns_observed_offset_outputs(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame({
        "point_id": ["a", "b"],
        "station_m": [0.0, 5.0],
        "bank_wse_proxy_monotone_m": [3.0, 2.0],
        "centerline_z_m": [1.5, 0.5],
    }, geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:4326")
    status = simple_river_stage_status_placeholder()
    result = run_simple_river_bundle_b(river_context={"centerline_points_gdf": gdf}, out_dir=str(tmp_path), stage_status=status, upto_stage=STAGE_CENTERLINE_OBSERVED_OFFSET)
    assert result["simple_river_stage_status"][STAGE_RIVER_CENTERLINE]["implemented"] is True
    assert result["simple_river_stage_status"][STAGE_CENTERLINE_WSE_PROXY]["implemented"] is True
    assert result["simple_river_stage_status"][STAGE_CENTERLINE_AUTHORITATIVE_BED]["implemented"] is True
    assert result["simple_river_stage_status"][STAGE_CENTERLINE_OBSERVED_OFFSET]["implemented"] is True
    assert Path(result["observed_offset"]["output_artifact"]).exists()
    assert sorted(result['simple_river_stage_outputs'].keys()) == ['authoritative_bed', 'centerline', 'observed_offset', 'wse_proxy']
