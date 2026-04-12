from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from river_v2_context import RiverV2Context
from river_v2_stage_primary_surface_support import build_primary_surface_support


class _Cfg:
    river_mainstem_min_order = 5
    river_centerline_sample_spacing_m = 25.0


def _write_projected_domain_mask(path: Path):
    arr = np.ones((31, 31), dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=31,
        height=31,
        count=1,
        dtype="uint8",
        crs="EPSG:32619",
        transform=from_origin(0.0, 310.0, 10.0, 10.0),
        nodata=0,
    ) as ds:
        ds.write(arr, 1)


def test_primary_surface_domain_is_narrower_than_valid_mask_for_projected_grid(tmp_path: Path):
    centerline = gpd.GeoDataFrame(
        {"point_id": ["a", "b", "c"], "station_m": [0.0, 50.0, 100.0]},
        geometry=[Point(100, 155), Point(150, 155), Point(200, 155)],
        crs="EPSG:32619",
    )
    centerline_path = tmp_path / "centerline.gpkg"
    centerline.to_file(centerline_path, driver="GPKG")
    domain_mask = tmp_path / "domain.tif"
    _write_projected_domain_mask(domain_mask)
    ctx = RiverV2Context(
        cfg=_Cfg(),
        report={},
        out_dir=tmp_path,
        network_gpkg=tmp_path / "network.gpkg",
        river_dem_path=domain_mask,
        channel_mask_path=domain_mask,
        centerline_points_gdf=centerline,
        vertical_reference="NAVD88",
    )
    result = build_primary_surface_support(ctx, centerline_points_path=centerline_path, channel_mask_path=domain_mask)
    assert result.validation["valid"] is True
    with rasterio.open(ctx.paths.river_primary_surface_domain) as ds:
        domain = ds.read(1).astype(bool)
    assert np.count_nonzero(domain) > 0
    assert np.count_nonzero(domain) < (31 * 31)
