#!/usr/bin/env python3
"""compute_longitudinal_seam_metrics.py

Compute longitudinal hydraulic plausibility metrics along centerlines.

Sample bed elevation at spacing Δx and compute slope:
slope_i = (z_{i+1} - z_i) / Δx

Report:
- median_slope
- uphill_fraction = fraction(slope_i > +0.001)  # 0.1 m / 100 m
- slope_spike_p99 = p99(|slope_i|)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from pyproj import CRS, Transformer


def _local_utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def _sample_points(line: LineString, spacing_m: float):
    length = line.length
    if length <= 0:
        return []
    dists = np.arange(0, length + 1e-9, spacing_m)
    return [line.interpolate(float(d)) for d in dists]


def _sample_raster(ds, xs, ys):
    vals = np.array([v[0] for v in ds.sample(zip(xs, ys))], dtype="float32")
    if ds.nodata is not None:
        vals = np.where(vals == ds.nodata, np.nan, vals)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed-elev", required=True)
    ap.add_argument("--centerline", required=True)
    ap.add_argument("--spacing-m", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-field", default=None)
    args = ap.parse_args()

    out_csv = Path(args.out); out_csv.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(args.centerline)
    if gdf.empty:
        raise SystemExit("Centerline has no features.")
    if gdf.crs is None:
        raise SystemExit("Centerline CRS is required.")

    gcrs = CRS.from_user_input(gdf.crs)
    centroid = gdf.geometry.unary_union.centroid
    if gcrs.is_geographic:
        lon, lat = centroid.x, centroid.y
    else:
        to_ll = Transformer.from_crs(gcrs, CRS.from_epsg(4326), always_xy=True).transform
        lon, lat = to_ll(centroid.x, centroid.y)
    proj = _local_utm_crs(lon, lat)
    gdfp = gdf.to_crs(proj)

    rows = []
    with rasterio.open(args.bed_elev) as ds:
        raster_crs = CRS.from_user_input(ds.crs)
        to_raster = Transformer.from_crs(proj, raster_crs, always_xy=True).transform

        for idx, feat in gdfp.iterrows():
            geom = feat.geometry
            if geom is None or geom.is_empty:
                continue

            fid = feat[args.id_field] if (args.id_field and args.id_field in feat) else idx

            parts = [geom] if isinstance(geom, LineString) else (list(geom.geoms) if isinstance(geom, MultiLineString) else [])
            for part_i, line in enumerate(parts):
                pts = _sample_points(line, args.spacing_m)
                if len(pts) < 2:
                    continue

                xs, ys = [], []
                for p in pts:
                    x, y = to_raster(p.x, p.y)
                    xs.append(x); ys.append(y)

                z = _sample_raster(ds, xs, ys).astype("float64")
                keep = np.isfinite(z)
                if keep.sum() < 2:
                    continue

                valid_idx = np.where(keep)[0]
                z_valid = z[keep]
                dx = (np.diff(valid_idx) * args.spacing_m).astype("float64")
                dz = np.diff(z_valid)
                slope = dz / dx

                rows.append(dict(
                    feature_id=fid,
                    part=part_i,
                    n_samples=int(z_valid.size),
                    spacing_m=args.spacing_m,
                    median_slope=float(np.nanmedian(slope)),
                    uphill_fraction=float(np.mean(slope > 0.001)),
                    slope_spike_p99=float(np.nanpercentile(np.abs(slope), 99)),
                ))

    if not rows:
        raise SystemExit("No valid longitudinal samples (check coverage/nodata).")

    pd.DataFrame(rows).to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()
