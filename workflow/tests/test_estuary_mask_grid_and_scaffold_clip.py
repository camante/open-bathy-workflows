import sys
from pathlib import Path

# Avoid repo-local shadowing and ensure real geospatial packages are imported first.
sys.path = [p for p in sys.path if '/workflow/tests' not in p and not p.endswith('/workflow')]

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

sys.path.append(str(Path(__file__).resolve().parents[1]))
from river_masking import _estimate_pixel_size_m
from river_structured_scaffold import _clip_gdf_to_allowed_mask, build_dense_bank_points


def test_estuary_pixel_size_uses_meter_equivalent_for_geographic_grid():
    transform = from_origin(-70.0, 43.0, 1.0 / 3600.0, 1.0 / 3600.0)

    class _CRS:
        is_geographic = True

    px_m = _estimate_pixel_size_m(transform, _CRS(), height=3600)
    assert 20.0 <= px_m <= 40.0


def test_clip_retained_vectors_to_estuary_first_allowed_mask():
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(0.5, 1.5), (3.5, 1.5)])],
        crs="EPSG:32619",
    )
    allowed = np.zeros((3, 4), dtype=np.uint8)
    allowed[:, :2] = 1
    clipped = _clip_gdf_to_allowed_mask(gdf, allowed, from_origin(0.0, 3.0, 1.0, 1.0))
    assert len(clipped) == 1
    minx, miny, maxx, maxy = clipped.geometry.iloc[0].bounds
    assert maxx <= 2.0 + 1e-6


def test_dense_bank_points_include_interior_island_banks(tmp_path: Path):
    poly = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Polygon(shell=[(0, 0), (30, 0), (30, 20), (0, 20)], holes=[[(10, 5), (20, 5), (20, 15), (10, 15)]])],
        crs="EPSG:32619",
    )
    bank = build_dense_bank_points(poly, spacing_m=5.0)
    assert not bank.empty
    xs = bank.geometry.x.to_numpy(dtype=float)
    ys = bank.geometry.y.to_numpy(dtype=float)
    interior_hits = ((xs >= 9.5) & (xs <= 20.5) & (ys >= 4.5) & (ys <= 15.5)).sum()
    assert interior_hits > 0
