import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path = [p for p in sys.path if '/workflow/tests' not in p and not p.endswith('/workflow')]
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sdb_domain_postprocess import apply_estuary_aware_sdb_domain


def _write_raster(path: Path, arr: np.ndarray, *, nodata=None, dtype=None):
    dtype = dtype or str(arr.dtype)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs='EPSG:32619',
        transform=from_origin(0, arr.shape[0], 1, 1),
        nodata=nodata,
    ) as ds:
        ds.write(arr.astype(dtype), 1)


def test_apply_estuary_aware_sdb_domain_masks_to_ocean_plus_estuary(tmp_path: Path):
    sdb_dir = tmp_path / 'sdb'
    rasters = sdb_dir / 'rasters'
    rasters.mkdir(parents=True)

    depth = rasters / 'depth.tif'
    conf = rasters / 'depth_confidence.tif'
    adm = rasters / 'depth_admissibility.tif'
    gpkg = rasters / 'depth_guide_points.gpkg'

    depth_arr = np.array([
        [-1.0, -2.0, -3.0],
        [-4.0, -5.0, -6.0],
        [-7.0, -8.0, -9.0],
    ], dtype=np.float32)
    _write_raster(depth, depth_arr, nodata=-9999.0, dtype='float32')
    _write_raster(conf, np.ones((3, 3), dtype=np.float32), nodata=-9999.0, dtype='float32')
    _write_raster(adm, np.ones((3, 3), dtype=np.uint8), nodata=0, dtype='uint8')

    import geopandas as gpd
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame(
        {'depth_m': [-1.0, -5.0, -9.0]},
        geometry=[Point(0.5, 2.5), Point(1.5, 1.5), Point(2.5, 0.5)],
        crs='EPSG:32619',
    )
    gdf.to_file(gpkg, driver='GPKG')

    manifest = {
        'depth_raster': 'rasters/depth.tif',
        'confidence_raster': 'rasters/depth_confidence.tif',
        'admissibility_raster': 'rasters/depth_admissibility.tif',
        'guide_points': 'rasters/depth_guide_points.gpkg',
    }
    (sdb_dir / 'artifacts_sdb.json').write_text(json.dumps(manifest), encoding='utf-8')

    ocean = tmp_path / 'ocean.tif'
    ocean_arr = np.ones((3, 3), dtype=np.uint8)
    ocean_arr[0, :] = 0  # only top row is ocean water
    _write_raster(ocean, ocean_arr, nodata=1, dtype='uint8')

    estuary = tmp_path / 'estuary_clip_mask.tif'
    estuary_arr = np.zeros((3, 3), dtype=np.uint8)
    estuary_arr[1, 1] = 1
    _write_raster(estuary, estuary_arr, nodata=0, dtype='uint8')

    receipt = apply_estuary_aware_sdb_domain(
        sdb_dir=sdb_dir,
        ocean_mask_path=ocean,
        estuary_mask_path=estuary,
    )

    assert receipt is not None
    with rasterio.open(depth) as ds:
        out = ds.read(1)
        assert out[0, 0] == -1.0
        assert out[1, 1] == -5.0
        assert out[2, 2] == ds.nodata

    with rasterio.open(adm) as ds:
        out = ds.read(1)
        assert out[0, 0] == 1
        assert out[1, 1] == 1
        assert out[2, 2] == 0

    manifest2 = json.loads((sdb_dir / 'artifacts_sdb.json').read_text(encoding='utf-8'))
    assert manifest2['domain_policy']['type'] == 'ocean_plus_estuary'
    assert manifest2['land_mask'].endswith('_domain_mask.tif')

    gdf2 = gpd.read_file(gpkg)
    assert len(gdf2) == 2
