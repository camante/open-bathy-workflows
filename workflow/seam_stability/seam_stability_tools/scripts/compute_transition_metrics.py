#!/usr/bin/env python3
"""compute_transition_metrics.py

Compute coastal↔river transition metrics within a transition zone mask Ω.

Δ = SDB - River on valid overlap pixels in Ω

Metrics:
transition_bias    = median(Δ)
transition_rmse    = sqrt(mean(Δ^2))
transition_p95_abs = p95(|Δ|)
overlap_valid_frac = N_valid_overlap / N_total_zone
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def _read_float(ds):
    arr = ds.read(1).astype("float32", copy=False)
    if ds.nodata is not None:
        arr = np.where(arr == ds.nodata, np.nan, arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdb", required=True)
    ap.add_argument("--river", required=True)
    ap.add_argument("--transition-mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--resampling", default="bilinear", choices=["nearest", "bilinear", "cubic"])
    args = ap.parse_args()

    resamp = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "cubic": Resampling.cubic}[args.resampling]
    out_csv = Path(args.out); out_csv.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.sdb) as dsdb, rasterio.open(args.river) as dr, rasterio.open(args.transition_mask) as dm:
        S = _read_float(dsdb)
        M = dm.read(1, out_shape=(dsdb.height, dsdb.width), resampling=Resampling.nearest).astype(bool)

        R = np.full((dsdb.height, dsdb.width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(dr, 1),
            destination=R,
            src_transform=dr.transform, src_crs=dr.crs,
            dst_transform=dsdb.transform, dst_crs=dsdb.crs,
            resampling=resamp,
            src_nodata=dr.nodata,
            dst_nodata=np.nan,
        )

        N_total = int(M.sum())
        keep = M & np.isfinite(S) & np.isfinite(R)
        N_valid = int(keep.sum())

        if N_total == 0:
            raise SystemExit("Transition mask has zero pixels in SDB grid.")

        if N_valid == 0:
            row = dict(transition_bias=np.nan, transition_rmse=np.nan, transition_p95_abs=np.nan,
                       overlap_valid_frac=0.0, n_total_zone=N_total, n_valid_overlap=0,
                       sdb=args.sdb, river=args.river, transition_mask=args.transition_mask)
        else:
            d = (S[keep] - R[keep]).astype("float64")
            row = dict(
                transition_bias=float(np.nanmedian(d)),
                transition_rmse=float(np.sqrt(np.nanmean(d**2))),
                transition_p95_abs=float(np.nanpercentile(np.abs(d), 95)),
                overlap_valid_frac=float(N_valid / N_total),
                n_total_zone=N_total,
                n_valid_overlap=N_valid,
                sdb=args.sdb,
                river=args.river,
                transition_mask=args.transition_mask,
            )

        pd.DataFrame([row]).to_csv(out_csv, index=False)

        if args.out_json:
            payload = {
                "metrics": row,
                "definitions": {
                    "transition_bias": "median(SDB - River) within Ω",
                    "transition_rmse": "sqrt(mean((SDB - River)^2)) within Ω",
                    "transition_p95_abs": "p95(|SDB - River|) within Ω",
                    "overlap_valid_frac": "N_valid_overlap / N_total_transition_zone_pixels",
                },
            }
            Path(args.out_json).write_text(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
