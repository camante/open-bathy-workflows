#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bank_mask_from_xs.py – Create a DEM-aligned bank-preservation mask from XS outputs

Create a DEM-aligned bank-preservation mask GeoTIFF (1=bank/preserve, 0=else)
from xs_builder.py outputs.

UPDATED: Automatically handles Geographic (Degrees) vs Projected (Meters) CRS."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import mapping
from pyproj import CRS

log = logging.getLogger("bank_mask_from_xs")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _read_layer(gpkg: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg, layer=layer)
    if gdf.empty:
        raise RuntimeError(f"Layer '{layer}' is empty in {gpkg}")
    if gdf.crs is None:
        raise RuntimeError(f"Layer '{layer}' has no CRS: {gpkg}")
    return gdf


def _open_template(path: Path):
    ds = rasterio.open(path)
    if ds.crs is None:
        raise RuntimeError(f"Template raster has no CRS: {path}")
    return ds


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


def build_bank_mask(
    xs_gpkg: Path,
    template_raster: Path,
    out_mask: Path,
    mode: str = "banks",
    bank_buffer_m: float = 10.0,
    corridor_buffer_m: float = 10.0,
    xs_points_layer: str = "xs_points",
    xs_lines_layer: str = "xs_lines",
) -> None:
    xs_gpkg = Path(xs_gpkg)
    template_raster = Path(template_raster)
    out_mask = Path(out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)

    with _open_template(template_raster) as tmpl:
        tcrs = CRS.from_user_input(tmpl.crs)

        # --- AUTO-CORRECT FOR DEGREES (GCS) ---
        # If CRS is geographic, convert meters to approximate decimal degrees
        if tcrs.is_geographic:
            bounds = tmpl.bounds
            lat_center = (bounds.bottom + bounds.top) / 2.0

            # 1 degree lat approx 111,000 meters
            deg_per_m_lat = 1.0 / 111000.0
            # Longitude scaling depends on latitude
            deg_per_m_lon = 1.0 / (111000.0 * np.cos(np.radians(lat_center)))

            # Use conservative scaling factor
            scaling = max(deg_per_m_lat, deg_per_m_lon)
            bank_buf_val = bank_buffer_m * scaling
            corr_buf_val = corridor_buffer_m * scaling

            log.warning("[CRS] Template is GCS (degrees). Converted buffers: bank=%.8f deg, corridor=%.8f deg",
                        bank_buf_val, corr_buf_val)
        else:
            bank_buf_val = bank_buffer_m
            corr_buf_val = corridor_buffer_m
            log.info("[CRS] Template is Projected (meters). Using raw buffer values.")

        bank_geoms = []
        corridor_geoms = []

        if mode in ("banks", "both"):
            xsp = _read_layer(xs_gpkg, xs_points_layer)
            if CRS.from_user_input(xsp.crs) != tcrs:
                xsp = xsp.to_crs(tcrs)

            if "is_bank_left" not in xsp.columns or "is_bank_right" not in xsp.columns:
                raise RuntimeError("xs_points missing is_bank_left/is_bank_right fields (from xs_builder.py).")

            banks = xsp[(xsp["is_bank_left"] == True) | (xsp["is_bank_right"] == True)].copy()
            if banks.empty:
                raise RuntimeError("No bank points found in xs_points (is_bank_left/is_bank_right all false).")

            bank_geoms = [g.buffer(bank_buf_val) for g in banks.geometry if g is not None and not g.is_empty]
            log.info("[BANKS] bank points=%d, buffer applied=%.8f", len(banks), bank_buf_val)

        if mode in ("corridor", "both"):
            xsl = _read_layer(xs_gpkg, xs_lines_layer)
            if CRS.from_user_input(xsl.crs) != tcrs:
                xsl = xsl.to_crs(tcrs)
            corridor_geoms = [g.buffer(corr_buf_val) for g in xsl.geometry if g is not None and not g.is_empty]
            log.info("[CORRIDOR] xs_lines=%d, buffer applied=%.8f", len(xsl), corr_buf_val)

        mask = np.zeros((tmpl.height, tmpl.width), dtype="uint8")
        if bank_geoms:
            mask = np.maximum(mask, _rasterize_geoms(bank_geoms, tmpl, burn_value=1))
        if corridor_geoms:
            mask = np.maximum(mask, _rasterize_geoms(corridor_geoms, tmpl, burn_value=1))

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

    log.info("[WRITE] %s (bank_mask pixels=%d)", str(out_mask), int((mask == 1).sum()))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "bank_mask_from_xs.py – create a DEM-aligned bank-preserve mask from XS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--xs-gpkg", required=True, help="XS GeoPackage from xs_builder.py")
    p.add_argument("--template-raster", required=True, help="Template DEM GeoTIFF (CUDEM DEM) for grid alignment")
    p.add_argument("--out-mask", required=True, help="Output bank preserve mask GeoTIFF (uint8)")
    p.add_argument("--mode", choices=["banks", "corridor", "both"], default="banks",
                   help="Mask construction mode: banks (recommended), corridor, or both")
    p.add_argument("--bank-buffer-m", type=float, default=10.0, help="Buffer around bank points (meters)")
    p.add_argument("--corridor-buffer-m", type=float, default=10.0, help="Buffer around XS lines (meters)")
    p.add_argument("--xs-points-layer", default="xs_points")
    p.add_argument("--xs-lines-layer", default="xs_lines")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    build_bank_mask(
        xs_gpkg=Path(a.xs_gpkg),
        template_raster=Path(a.template_raster),
        out_mask=Path(a.out_mask),
        mode=a.mode,
        bank_buffer_m=a.bank_buffer_m,
        corridor_buffer_m=a.corridor_buffer_m,
        xs_points_layer=a.xs_points_layer,
        xs_lines_layer=a.xs_lines_layer,
    )


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
