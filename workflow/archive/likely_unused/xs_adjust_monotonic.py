#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xs_adjust_monotonic.py – Enforce a downstream-monotonic bed profile along river stationing

Enforce a downstream-monotonic bed profile along river stationing, producing
adjusted columns:
  - depth_adj_m
  - z_bed_adj_m

This is meant to run on the xs_bathy_points layer produced by xs_infer_bathy_raster.py.

Expected columns (best case)
----------------------------
Required:
  - xs_id
  - depth_pred_m OR z_bed_pred_m (one is required)
Optional but strongly recommended:
  - component_id (for grouping)
  - s_center_m   (for correct downstream ordering)

If component_id / s_center_m are missing, the script will fall back to a single group
and order by xs_id, which is usually *not* physically meaningful. It will still run,
but logs a warning.

Monotonic rule
--------------
We enforce that (downstream) depth does not decrease (i.e., bed does not rise),
allowing small violations up to epsilon.
"""

from __future__ import annotations

import argparse
import logging


# Logging is configured by entrypoints (e.g., bathy_main.py / sdb_main.py).
# Standalone scripts configure logging in __main__.

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd


log = logging.getLogger("xs_adjust_monotonic")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "xs_adjust_monotonic.py – enforce downstream-monotonic bed profile along stationing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-gpkg", required=True)
    p.add_argument("--in-layer", default="xs_bathy_points")
    p.add_argument("--out-gpkg", required=True)
    p.add_argument("--out-layer", default="xs_bathy_points_monotonic")
    p.add_argument("--epsilon-m", type=float, default=0.02, help="Allowed upstream increase (m) tolerated as noise.")
    p.add_argument("--group-cols", default="reach_id,component_id", help="Comma-separated grouping columns to try in order.")
    p.add_argument("--order-col", default="s_center_m", help="Ordering column (downstream increasing).")
    return p.parse_args()


def _ensure_depth(df: pd.DataFrame) -> pd.Series:
    if "depth_pred_m" in df.columns:
        return pd.to_numeric(df["depth_pred_m"], errors="coerce")
    if "z_bed_pred_m" in df.columns and "wse_m" in df.columns:
        wse = pd.to_numeric(df["wse_m"], errors="coerce")
        z = pd.to_numeric(df["z_bed_pred_m"], errors="coerce")
        return wse - z
    raise RuntimeError("Need depth_pred_m OR (wse_m and z_bed_pred_m) to compute depth.")


def _monotonic_adjust(depth: np.ndarray, eps: float) -> np.ndarray:
    """
    Enforce non-decreasing depth downstream:
      depth_adj[i] = max(depth[i], depth_adj[i-1] - eps)
    """
    out = depth.copy()
    last = np.nan
    for i in range(out.size):
        d = out[i]
        if not np.isfinite(d):
            continue
        if not np.isfinite(last):
            last = d
            out[i] = d
            continue
        # allow tiny upstream "rise" (depth decrease) up to eps
        min_allowed = last - eps
        if d < min_allowed:
            d = min_allowed
        out[i] = d
        last = d
    return out


def main() -> None:
    args = _parse_args()

    gpkg_in = Path(args.in_gpkg)
    layer_in = str(args.in_layer)
    gdf = gpd.read_file(gpkg_in, layer=layer_in)
    if gdf.empty:
        raise RuntimeError(f"Empty layer '{layer_in}' in {gpkg_in}")
    if "xs_id" not in gdf.columns:
        raise RuntimeError(f"Missing required column 'xs_id' in layer '{layer_in}'")

    depth = _ensure_depth(gdf).to_numpy(dtype="float64")
    gdf = gdf.copy()
    gdf["depth_pred_m"] = depth  # ensure present

    # Determine grouping
    group_candidates = [c.strip() for c in str(args.group_cols).split(",") if c.strip()]
    group_cols = [c for c in group_candidates if c in gdf.columns]

    if not group_cols:
        log.warning("No group columns found (%s). Falling back to single group.", group_candidates)
        gdf["_group"] = 0
        group_cols = ["_group"]

    order_col = str(args.order_col)
    if order_col not in gdf.columns:
        log.warning("Order column '%s' missing. Falling back to xs_id ordering.", order_col)
        gdf["_order"] = pd.to_numeric(gdf["xs_id"], errors="coerce")
        if gdf["_order"].isna().all():
            # last resort: stable sort by xs_id string
            gdf["_order"] = np.arange(len(gdf), dtype="float64")
        order_col = "_order"

    eps = float(args.epsilon_m)

    out_frames: List[gpd.GeoDataFrame] = []
    for key, sub in gdf.groupby(group_cols, dropna=False, sort=False):
        sub = sub.sort_values(order_col).copy()
        d = pd.to_numeric(sub["depth_pred_m"], errors="coerce").to_numpy(dtype="float64")
        d_adj = _monotonic_adjust(d, eps=eps)
        sub["depth_adj_m"] = d_adj

        # Prefer z_bed_pred_m adjustment if present
        if "z_bed_pred_m" in sub.columns and "wse_m" in sub.columns:
            wse = pd.to_numeric(sub["wse_m"], errors="coerce").to_numpy(dtype="float64")
            sub["z_bed_adj_m"] = wse - d_adj
        elif "z_bed_pred_m" in sub.columns:
            # If we don't know WSE, shift bed by depth delta in elevation space (approx)
            z = pd.to_numeric(sub["z_bed_pred_m"], errors="coerce").to_numpy(dtype="float64")
            dz = d_adj - d
            sub["z_bed_adj_m"] = z - dz
        else:
            sub["z_bed_adj_m"] = np.nan

        out_frames.append(sub)

    out = pd.concat(out_frames, ignore_index=True)
    out_gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)

    out_path = Path(args.out_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(out_path, layer=str(args.out_layer), driver="GPKG")
    log.info("[WRITE] %s (layer=%s, n=%d)", str(out_path), str(args.out_layer), len(out_gdf))


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
