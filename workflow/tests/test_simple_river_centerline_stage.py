import geopandas as gpd
from shapely.geometry import Point

from simple_river_centerline_stage import centerline_field_schema, validate_centerline_points


def _gdf(rows):
    return gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:32619')


def test_centerline_field_schema_has_required_fields():
    schema = centerline_field_schema()
    assert 'point_id' in schema
    assert 'station_m' in schema
    assert 'geometry' in schema


def test_validate_centerline_points_accepts_minimal_valid_gdf():
    gdf = _gdf([
        {'point_id': 'a', 'station_m': 0.0, 'geometry': Point(0, 0), 'reach_id': 'r1'},
        {'point_id': 'b', 'station_m': 10.0, 'geometry': Point(1, 0), 'reach_id': 'r1'},
    ])
    result = validate_centerline_points(gdf)
    assert result['valid'] is True
    assert result['station_monotonic_by_reach'] is True


def test_validate_centerline_points_rejects_missing_station():
    gdf = _gdf([{'point_id': 'a', 'geometry': Point(0, 0)}])
    result = validate_centerline_points(gdf)
    assert result['valid'] is False


def test_validate_centerline_points_rejects_duplicate_point_id():
    gdf = _gdf([
        {'point_id': 'a', 'station_m': 0.0, 'geometry': Point(0, 0)},
        {'point_id': 'a', 'station_m': 1.0, 'geometry': Point(1, 0)},
    ])
    result = validate_centerline_points(gdf)
    assert result['point_id_unique'] is False
    assert result['valid'] is False


def test_validate_centerline_points_rejects_non_monotonic_stationing():
    gdf = _gdf([
        {'point_id': 'a', 'station_m': 10.0, 'geometry': Point(0, 0), 'reach_id': 'r1'},
        {'point_id': 'b', 'station_m': 5.0, 'geometry': Point(1, 0), 'reach_id': 'r1'},
    ])
    result = validate_centerline_points(gdf)
    assert result['station_monotonic_by_reach'] is False
    assert result['valid'] is False
