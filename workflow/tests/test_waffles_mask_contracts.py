from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, LineString

from bathy_main import _build_canonical_with_nhd_mask


def test_build_canonical_with_nhd_mask_adds_inland_water(tmp_path):
    ocean = tmp_path / 'ocean.tif'
    gpkg = tmp_path / 'river_network.gpkg'
    out = tmp_path / 'with_nhd.tif'

    transform = from_origin(0, 6, 1, 1)
    data = np.ones((6, 6), dtype=np.uint8)
    data[:, 0] = 0  # ocean water on left edge
    profile = {
        'driver': 'GTiff', 'height': 6, 'width': 6, 'count': 1,
        'dtype': 'uint8', 'crs': 'EPSG:32619', 'transform': transform,
        'nodata': None,
    }
    with rasterio.open(ocean, 'w', **profile) as ds:
        ds.write(data, 1)

    rivers = gpd.GeoDataFrame({'id': [1]}, geometry=[LineString([(2.5, 1.5), (2.5, 4.5)])], crs='EPSG:32619')
    nhdarea = gpd.GeoDataFrame({'id': [1]}, geometry=[Polygon([(2, 1), (3, 1), (3, 5), (2, 5)])], crs='EPSG:32619')
    rivers.to_file(gpkg, layer='rivers_clip', driver='GPKG')
    nhdarea.to_file(gpkg, layer='nhdarea_clip', driver='GPKG')

    _build_canonical_with_nhd_mask(ocean, gpkg, out)
    with rasterio.open(out) as ds:
        arr = ds.read(1)
        assert ds.nodata is None
    # Ocean remains water=0
    assert np.all(arr[:, 0] == 0)
    # Inland river corridor added as water=0
    assert np.all(arr[1:5, 2] == 0)
    # Land remains 1 somewhere away from ocean/river
    assert arr[0, 5] == 1


def test_build_canonical_with_nhd_closes_diagonal_ocean_seam(tmp_path):
    from scipy import ndimage as ndi

    ocean = tmp_path / "ocean_diag.tif"
    gpkg = tmp_path / "river_network_diag.gpkg"
    out = tmp_path / "with_nhd_diag.tif"

    transform = from_origin(0, 4, 1, 1)
    data = np.ones((4, 4), dtype=np.uint8)
    data[0, 0] = 0  # one ocean-water cell
    profile = {
        'driver': 'GTiff', 'height': 4, 'width': 4, 'count': 1,
        'dtype': 'uint8', 'crs': 'EPSG:32619', 'transform': transform,
        'nodata': None,
    }
    with rasterio.open(ocean, 'w', **profile) as ds:
        ds.write(data, 1)

    rivers = gpd.GeoDataFrame({'id': [1]}, geometry=[LineString([(1.5, 2.5), (1.5, 2.5)])], crs='EPSG:32619')
    nhdarea = gpd.GeoDataFrame({'id': [1]}, geometry=[Polygon([(1, 2), (2, 2), (2, 3), (1, 3)])], crs='EPSG:32619')
    rivers.to_file(gpkg, layer='rivers_clip', driver='GPKG')
    nhdarea.to_file(gpkg, layer='nhdarea_clip', driver='GPKG')

    _build_canonical_with_nhd_mask(ocean, gpkg, out)
    with rasterio.open(out) as ds:
        arr = ds.read(1)

    water = (arr == 0)
    labels, ncomp = ndi.label(water, structure=ndi.generate_binary_structure(2, 1))
    assert ncomp == 1
    assert water[0, 1] or water[1, 0] or water[1, 1]
