#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measured_mask_from_points.py

Create a DEM-aligned "measured bathy preserve" mask GeoTIFF from measured depth points
(sonar / bathymetric lidar / trusted survey points). This mask is intended to be used
with cudem_river_burn.py to *block* any inferred patch from overwriting measured depths.

Inputs
------
- --points : point dataset (GPKG/GeoJSON/SHP) OR CSV
- --template-raster : DEM GeoTIFF grid to match (usually your CUDEM DEM)
- --out-mask : output mask GeoTIFF (uint8)
Optional:
- --buffer-m : buffer around points (meters) before rasterizing (recommended > 0)

CSV support
-----------
If --points is a CSV, provide:
  --x-col / --y-col (or it will try lon/lat, longitude/latitude, x/y)
And optionally:
  --crs EPSG:####  (defaults to EPSG:4326)

Output
------
- uint8 GeoTIFF mask:
    1 = measured/trusted (preserve; block patch)
    0 = not measured
    nodata = 0

Example
-------
python measured_mask_from_points.py \
  --points /path/to/soundings.gpkg \
  --template-raster /path/to/cudem_dem.tif \
  --out-mask output/measured_bathy_mask.tif \
  --buffer-m 5

Then:
python cudem_river_burn.py ... --measured-mask output/measured_bathy_mask.tif
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import mapping
from pyproj import CRS

log = logging.getLogger("measured_mask_from_points")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _open_template(path: Path):
    ds = rasterio.open(path)
    if ds.crs is None:
        raise RuntimeError(f"Template raster has no CRS: {path}")
    return ds


def _load_points(points_path: Path, csv_crs: str, x_col: Optional[str], y_col: Optional[str]) -> gpd.GeoDataFrame:
    points_path = Path(points_path)
    if not points_path.exists():
        raise FileNotFoundError(str(points_path))

    if points_path.suffix.lower() == ".csv":
        df = pd.read_csv(points_path)
        if x_col is None or y_col is None:
            for xc, yc in [("lon", "lat"), ("longitude", "latitude"), ("x", "y"), ("easting", "northing")]:
                if xc in df.columns and yc in df.columns:
                    x_col, y_col = xc, yc
                    break
        if x_col is None or y_col is None:
            raise RuntimeError("CSV points require --x-col and --y-col (or lon/lat columns).")

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=csv_crs)
    else:
        gdf = gpd.read_file(points_path)
        if gdf.empty:
            raise RuntimeError(f"No features in {points_path}")
        if gdf.crs is None:
            raise RuntimeError(f"Points file has no CRS: {points_path}")

    if gdf.geometry.isna().all():
        raise RuntimeError("Points dataset has no geometries.")
    return gdf


def _rasterize_geoms(
    geoms: List,
    template_ds: rasterio.DatasetReader,
    burn_value: int = 1,
) -> np.ndarray:
    if not geoms:
        return np.zeros((template_ds.height, template_ds.width), dtype="uint8")
    shapes = [(mapping(g), burn_value) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((template_ds.height, template_ds.width), dtype="uint8")
    arr = features.rasterize(
        shapes=shapes,
        out_shape=(template_ds.height, template_ds.width),
        transform=template_ds.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return arr


def build_measured_mask(
    points: Path,
    template_raster: Path,
    out_mask: Path,
    buffer_m: float = 5.0,
    csv_crs: str = "EPSG:4326",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
) -> None:
    out_mask = Path(out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)

    with _open_template(Path(template_raster)) as tmpl:
        tcrs = CRS.from_user_input(tmpl.crs)

        gdf = _load_points(Path(points), csv_crs=csv_crs, x_col=x_col, y_col=y_col)
        if CRS.from_user_input(gdf.crs) != tcrs:
            gdf = gdf.to_crs(tcrs)

        # buffer geometries in meters (assumes projected CRS; if geographic, buffer still happens in degrees -> warn)
        if CRS.from_user_input(gdf.crs).is_geographic:
            log.warning("[CRS] Points CRS is geographic; buffering in degrees (not meters). Provide projected points or reproject first.")
        buf = float(max(0.0, buffer_m))
        geoms = [geom.buffer(buf) if buf > 0 else geom for geom in gdf.geometry if geom is not None and not geom.is_empty]

        mask = _rasterize_geoms(geoms, tmpl, burn_value=1)

        profile = tmpl.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(out_mask, "w", **profile) as dst:
            dst.write(mask, 1)

    log.info("[WRITE] %s (measured_mask pixels=%d)", str(out_mask), int((mask == 1).sum()))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "measured_mask_from_points.py – create a DEM-aligned preserve mask from measured bathy points",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--points", required=True, help="Measured/trusted bathy points (GPKG/GeoJSON/SHP) or CSV")
    p.add_argument("--template-raster", required=True, help="Template DEM GeoTIFF (CUDEM DEM) for grid alignment")
    p.add_argument("--out-mask", required=True, help="Output measured bathy preserve mask GeoTIFF (uint8)")
    p.add_argument("--buffer-m", type=float, default=5.0, help="Buffer around points before rasterizing (meters if projected CRS)")
    p.add_argument("--csv-crs", default="EPSG:4326", help="CRS for CSV points if not lon/lat")
    p.add_argument("--x-col", default=None, help="CSV X column name")
    p.add_argument("--y-col", default=None, help="CSV Y column name")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_measured_mask(
        points=Path(a.points),
        template_raster=Path(a.template_raster),
        out_mask=Path(a.out_mask),
        buffer_m=float(a.buffer_m),
        csv_crs=str(a.csv_crs),
        x_col=a.x_col,
        y_col=a.y_col,
    )


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
