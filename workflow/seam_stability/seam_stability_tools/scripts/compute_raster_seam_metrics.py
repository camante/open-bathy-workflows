#!/usr/bin/env python3
"""compute_raster_seam_metrics.py

Compute seam metrics between two rasters A and B across their shared boundary.

Within boundary buffer Ω:
Δ = A - B on valid overlap pixels (optional masks applied).

Metrics:
seam_bias      = median(Δ)
seam_rmse      = sqrt(mean(Δ^2))
seam_p95_abs   = p95(|Δ|)
overlap_valid_frac = N_valid_overlap / N_total_buffer

Ω is the shared boundary of raster bounds buffered by W meters.
If A CRS is geographic, buffer is computed in local UTM.

B is resampled onto A grid (bilinear default).
"""
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import box, mapping, LineString
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer


def _local_utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def _shared_boundary_line(bounds_a: Tuple[float, float, float, float],
                          bounds_b: Tuple[float, float, float, float]) -> Optional[LineString]:
    a = box(*bounds_a)
    b = box(*bounds_b)
    inter = a.boundary.intersection(b.boundary)
    if inter.is_empty:
        return None
    if inter.geom_type == "LineString":
        return inter
    if inter.geom_type in ("MultiLineString", "GeometryCollection"):
        lines = [g for g in getattr(inter, "geoms", []) if g.geom_type == "LineString"]
        if not lines:
            return None
        return max(lines, key=lambda g: g.length)
    return None


def _buffer_boundary(boundary: LineString, src_crs: CRS, buffer_m: float):
    if src_crs.is_geographic:
        cx, cy = boundary.centroid.x, boundary.centroid.y
        proj = _local_utm_crs(cx, cy)
    else:
        proj = src_crs

    to_proj = Transformer.from_crs(src_crs, proj, always_xy=True).transform
    to_src  = Transformer.from_crs(proj, src_crs, always_xy=True).transform
    boundary_proj = shp_transform(to_proj, boundary)
    poly_proj = boundary_proj.buffer(buffer_m)
    return shp_transform(to_src, poly_proj)


def _geometry_mask(geom, out_shape, transform):
    return ~features.geometry_mask([mapping(geom)], out_shape=out_shape, transform=transform, invert=False)


def _read_float(ds):
    arr = ds.read(1).astype("float32", copy=False)
    if ds.nodata is not None:
        arr = np.where(arr == ds.nodata, np.nan, arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Raster A (reference grid)")
    ap.add_argument("--b", required=True, help="Raster B (resampled to A)")
    ap.add_argument("--mask-a", default=None, help="Optional mask for A (1=in domain)")
    ap.add_argument("--mask-b", default=None, help="Optional mask for B (1=in domain)")
    ap.add_argument("--buffer-m", type=float, default=200.0)
    ap.add_argument("--resampling", default="bilinear", choices=["nearest", "bilinear", "cubic"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    resamp = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "cubic": Resampling.cubic}[args.resampling]
    out_csv = Path(args.out); out_csv.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.a) as da, rasterio.open(args.b) as db:
        if da.crs is None or db.crs is None:
            raise SystemExit("Both rasters must have CRS.")

        boundary = _shared_boundary_line(da.bounds, db.bounds)
        if boundary is None:
            raise SystemExit("No shared boundary detected (tiles may not be adjacent).")

        seam_poly = _buffer_boundary(boundary, CRS.from_user_input(da.crs), args.buffer_m)
        buf_mask = _geometry_mask(seam_poly, (da.height, da.width), da.transform)

        if args.mask_a:
            with rasterio.open(args.mask_a) as ma:
                mask_a = ma.read(1, out_shape=(da.height, da.width), resampling=Resampling.nearest).astype(bool)
        else:
            mask_a = np.ones((da.height, da.width), dtype=bool)

        if args.mask_b:
            with rasterio.open(args.mask_b) as mb:
                mask_b = mb.read(1, out_shape=(da.height, da.width), resampling=Resampling.nearest).astype(bool)
        else:
            mask_b = np.ones((da.height, da.width), dtype=bool)

        A = _read_float(da)
        B = np.full((da.height, da.width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(db, 1),
            destination=B,
            src_transform=db.transform,
            src_crs=db.crs,
            dst_transform=da.transform,
            dst_crs=da.crs,
            resampling=resamp,
            src_nodata=db.nodata,
            dst_nodata=np.nan,
        )

        N_total = int(buf_mask.sum())
        keep = buf_mask & mask_a & mask_b & np.isfinite(A) & np.isfinite(B)
        N_valid = int(keep.sum())

        if N_total == 0:
            raise SystemExit("Boundary buffer produced zero pixels in A grid.")

        if N_valid == 0:
            row = dict(seam_bias=np.nan, seam_rmse=np.nan, seam_p95_abs=np.nan,
                       overlap_valid_frac=0.0, n_total_buffer=N_total, n_valid_overlap=0,
                       buffer_m=args.buffer_m, a=args.a, b=args.b)
        else:
            d = (A[keep] - B[keep]).astype("float64")
            row = dict(
                seam_bias=float(np.nanmedian(d)),
                seam_rmse=float(np.sqrt(np.nanmean(d**2))),
                seam_p95_abs=float(np.nanpercentile(np.abs(d), 95)),
                overlap_valid_frac=float(N_valid / N_total),
                n_total_buffer=N_total,
                n_valid_overlap=N_valid,
                buffer_m=args.buffer_m,
                a=args.a,
                b=args.b,
            )

        pd.DataFrame([row]).to_csv(out_csv, index=False)

        if args.out_json:
            payload = {
                "metrics": row,
                "definitions": {
                    "seam_bias": "median(A - B) within Ω",
                    "seam_rmse": "sqrt(mean((A - B)^2)) within Ω",
                    "seam_p95_abs": "p95(|A - B|) within Ω",
                    "overlap_valid_frac": "N_valid_overlap / N_total_buffer_pixels",
                },
            }
            Path(args.out_json).write_text(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
