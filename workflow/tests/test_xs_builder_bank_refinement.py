import importlib
import sys
import numpy as np
import pandas as pd

# conftest installs lightweight stubs for geospatial libs; replace the stubs
# with the real packages for this focused XS bank-picking test.
for name in [
    "shapely", "shapely.geometry", "shapely.ops", "shapely.strtree",
    "pyproj", "geopandas", "fiona",
]:
    sys.modules.pop(name, None)
import shapely
import shapely.geometry
import shapely.ops
import shapely.strtree
import pyproj
import geopandas
import fiona
sys.modules.update({
    "shapely": shapely,
    "shapely.geometry": shapely.geometry,
    "shapely.ops": shapely.ops,
    "shapely.strtree": shapely.strtree,
    "pyproj": pyproj,
    "geopandas": geopandas,
    "fiona": fiona,
})

import rasterio
if not hasattr(rasterio, "DatasetReader"):
    rasterio.DatasetReader = object

from shapely.geometry import LineString, Point, Polygon

xs_builder = importlib.import_module("xs_builder")
estimate_bank_edge_distances = xs_builder.estimate_bank_edge_distances
pick_banks = xs_builder.pick_banks


def test_estimate_bank_edge_distances_uses_polygon_crossing_containing_center():
    xs = LineString([(0.0, 0.0), (100.0, 0.0)])
    center = Point(50.0, 0.0)
    poly = Polygon([(20.0, -10.0), (80.0, -10.0), (80.0, 10.0), (20.0, 10.0)])
    left, right = estimate_bank_edge_distances(xs, center, poly)
    assert left == 20.0
    assert right == 80.0


def test_pick_banks_prefers_corridor_edge_refinement_over_endpoint_peak():
    dist = np.arange(0.0, 101.0, 1.0)
    z = np.full(dist.shape, 1.0, dtype=float)
    z[4] = 10.0
    z[96] = 11.0
    z[19:23] = [5.0, 6.0, 6.5, 6.0]
    z[78:82] = [6.0, 6.6, 6.7, 6.2]
    profile = pd.DataFrame({
        "dist_m": dist,
        "z_dem": z,
        "z_topo": np.full(dist.shape, np.nan, dtype=float),
    })
    idx_l, idx_r, meta = pick_banks(
        profile,
        bank_search_m=40.0,
        prefer_topo=True,
        expected_left_dist_m=20.0,
        expected_right_dist_m=80.0,
        bank_edge_refine_m=4.0,
        bank_quantile=0.8,
        bank_smooth_window_m=3.0,
    )
    assert meta["method"] == "corridor_edge_refined"
    assert abs(profile.loc[idx_l, "dist_m"] - 20.0) <= 2.0
    assert abs(profile.loc[idx_r, "dist_m"] - 80.0) <= 2.0


def test_pick_banks_falls_back_when_no_expected_edges_available():
    dist = np.arange(0.0, 51.0, 1.0)
    z = np.zeros(dist.shape, dtype=float)
    z[5] = 4.0
    z[-6] = 5.0
    profile = pd.DataFrame({
        "dist_m": dist,
        "z_dem": z,
        "z_topo": np.full(dist.shape, np.nan, dtype=float),
    })
    idx_l, idx_r, meta = pick_banks(profile, bank_search_m=10.0)
    assert meta["method"] == "endpoint_peak_fallback"
    assert idx_l == 5
    assert idx_r == len(dist) - 6
