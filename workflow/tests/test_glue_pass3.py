from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from xs_infer_bathy_raster import _read_layer_with_fallback, _write_soundings_subset


def test_read_layer_with_fallback_requires_explicit_layer(tmp_path: Path):
    gpkg = tmp_path / 'rivers.gpkg'
    gdf = gpd.GeoDataFrame({'id':[1]}, geometry=[LineString([(0,0),(1,1)])], crs='EPSG:4326')
    gdf.to_file(gpkg, layer='flowlines', driver='GPKG')
    with pytest.raises(RuntimeError, match="Required rivers layer 'rivers'"):
        _read_layer_with_fallback(gpkg, 'rivers', purpose='rivers')


def test_write_soundings_subset_requires_parquet_suffix(tmp_path: Path):
    gdf = gpd.GeoDataFrame(
        {'_depth_m': [1.0], '_src_file':['a.xyz']},
        geometry=[Point(0, 0)],
        crs='EPSG:4326',
    )
    with pytest.raises(RuntimeError, match='must be parquet'):
        _write_soundings_subset(tmp_path / 'subset.gpkg', gdf)


def test_write_soundings_subset_raises_when_empty_after_filter(tmp_path: Path):
    gdf = gpd.GeoDataFrame(
        {'_depth_m': [float('nan')], '_z_m':[float('nan')], '_src_file':['a.xyz']},
        geometry=[Point(0, 0)],
        crs='EPSG:4326',
    )
    with pytest.raises(RuntimeError, match='would be empty after finite filter'):
        _write_soundings_subset(tmp_path / 'subset.parquet', gdf)
