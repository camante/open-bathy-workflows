from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from river_v2_stage_authoritative_bed import sample_authoritative_bed_to_centerline



def test_authoritative_bed_reads_nearest_support_point_from_csv(tmp_path: Path):
    csv_path = tmp_path / "support.csv"
    csv_path.write_text(
        """x,y,depth_m,authoritative_role
0.0,0.0,-2.5,authoritative_bed
10.0,0.0,-9.0,authoritative_overbank_or_ambiguous
""",
        encoding="utf-8",
    )
    gdf = gpd.GeoDataFrame(
        {'point_id': ['a'], 'station_m': [0.0]},
        geometry=[Point(1.0, 0.0)],
        crs='EPSG:4326',
    )
    out, warnings, method, diagnostics = sample_authoritative_bed_to_centerline(gdf, csv_path)
    assert method == 'nearest_authoritative_support_points'
    assert len(out) == 1
    assert np.isclose(float(out['authoritative_bed_z_m'].iloc[0]), -2.5)
    assert diagnostics['matched_point_count'] == 1
    assert diagnostics['authoritative_source_kind'] == 'csv_support_points'
    assert 'authoritative_bed_no_finite_values' not in warnings




def test_authoritative_bed_stage_requires_explicit_support_points(tmp_path):
    from river_v2_stage_authoritative_bed import sample_authoritative_bed_to_centerline
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({
        "point_id": ["p1"],
        "station_m": [0.0],
        "authoritative_bed_z_m": [1.23],
    }, geometry=[Point(0, 0)], crs="EPSG:32619")

    out, warnings, method, diagnostics = sample_authoritative_bed_to_centerline(gdf, None)
    assert len(out) == 0
    assert method == "missing_authoritative_support_points"
    assert diagnostics["authoritative_source_kind"] == "missing"
    assert diagnostics["authoritative_bed_failure_reason"] == "missing_explicit_support_points_artifact"
    assert "authoritative_bed_missing_support_points_artifact" in warnings
