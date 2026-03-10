#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alignment.py – Post-prediction residual alignment of an SDB raster to reference tie points.

Implements translation-only (median), depth-stratified translation, and planar Δz(x,y) alignment.

Design goals:
- Works with your existing artifacts: output/data/icesat_depths.gpkg (layers: all_depths/train_depths/test_depths)
- Can also ingest external CSV / GPKG / GeoJSON point files (lon/lat + depth columns)
- Produces:
  * aligned raster (optional, if out_path provided)
  * alignment summary dict (pre/post residual stats + gates + fit params)
  * diagnostic plot (histogram + residual vs depth) for RunReport integration
"""


import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Any

import numpy as np

log = logging.getLogger("sdb.align")


# -----------------------------------------------------------------------------
# Tie-point selection (deterministic)
# -----------------------------------------------------------------------------
SOURCE_ALIASES = {
    "atl24": {"atl24"},
    "atl03": {"atl03", "atl03_refraction", "atl03_ref"},
    "xyz": {"xyz", "extra_xyz", "sonar", "lidar", "bathy", "bathymetry", "soundings", "multibeam", "survey"},
}

def _normalize_source(value) -> str:
    if value is None:
        return "unknown"
    s = str(value).strip().lower()
    if s == "":
        return "unknown"
    for canon, aliases in SOURCE_ALIASES.items():
        if s == canon or s in aliases:
            return canon
    return "unknown"

def select_tiepoints(
    tie_df,
    mode: str = "stacked",
    *,
    source_col: str = "source",
    stacked_order = ("xyz", "atl03", "atl24"),
    per_source_max: int | None = None,
    seed: int = 42,
):
    """Deterministically select tie points by source (atl24|atl03|xyz|stacked)."""
    import pandas as _pd
    import numpy as _np

    if tie_df is None or len(tie_df) == 0:
        return tie_df

    df = tie_df.copy()
    if source_col not in df.columns:
        df["__source_norm"] = "unknown"
    else:
        df["__source_norm"] = df[source_col].map(_normalize_source)

    mode_l = str(mode or "stacked").strip().lower()
    if mode_l in ("atl24", "atl03", "xyz"):
        out = df.loc[df["__source_norm"] == mode_l].copy()
    elif mode_l == "stacked":
        parts = [df.loc[df["__source_norm"] == s].copy() for s in stacked_order]
        out = _pd.concat(parts, axis=0, ignore_index=False) if len(parts) else df.iloc[0:0].copy()
    else:
        raise ValueError(f"Unknown tiepoint selection mode: {mode}. Choose atl24|atl03|xyz|stacked")

    if per_source_max is not None and per_source_max > 0 and len(out) > 0:
        rng = _np.random.default_rng(int(seed))
        if mode_l == "stacked":
            capped = []
            for s in stacked_order:
                sub = out.loc[out["__source_norm"] == s].sort_index()
                if len(sub) <= per_source_max:
                    capped.append(sub)
                else:
                    take = rng.choice(sub.index.to_numpy(), size=int(per_source_max), replace=False)
                    capped.append(sub.loc[take])
            out = _pd.concat(capped, axis=0, ignore_index=False) if capped else out.iloc[0:0].copy()
        else:
            sub = out.sort_index()
            if len(sub) > per_source_max:
                take = rng.choice(sub.index.to_numpy(), size=int(per_source_max), replace=False)
                out = sub.loc[take]

    return out
# ----------------------------
# Helpers: I/O for tie points
# ----------------------------

def _guess_lonlat_cols(cols: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    c = {x.lower(): x for x in cols}
    lon = None
    lat = None
    for k in ("lon", "longitude", "x"):
        if k in c:
            lon = c[k]
            break
    for k in ("lat", "latitude", "y"):
        if k in c:
            lat = c[k]
            break
    return lon, lat

def _guess_depth_col(cols: Sequence[str]) -> Optional[str]:
    c = {x.lower(): x for x in cols}
    for k in ("depth_m", "depth", "z", "elev_m", "elevation_m", "height_m"):
        if k in c:
            return c[k]
    return None

def read_tie_points(
    path: str,
    *,
    layer: Optional[str] = None,
    source_name: Optional[str] = None,
    depth_positive_down: Optional[bool] = None,
) -> "TiePoints":
    """
    Read tie points from a file.

    Supported:
      - GeoPackage (GPKG): reads 'layer' if given, else tries common layers.
      - Any vector readable by geopandas (if installed).
      - CSV/TSV: expects lon/lat + depth columns.

    Returns TiePoints with lon/lat in EPSG:4326 and depth_m (float) as provided.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()

    if ext in (".csv", ".tsv"):
        import pandas as pd
        df = pd.read_csv(p) if ext == ".csv" else pd.read_csv(p, sep="\t")
        lon_col, lat_col = _guess_lonlat_cols(df.columns)
        dcol = _guess_depth_col(df.columns)
        if lon_col is None or lat_col is None or dcol is None:
            raise ValueError(f"CSV missing lon/lat/depth columns: {p}")
        out = df[[lon_col, lat_col, dcol]].copy()
        out.columns = ["longitude", "latitude", "depth_m"]
        if source_name:
            out["source"] = source_name
        return TiePoints.from_df(out, depth_positive_down=depth_positive_down)

    # vector formats
    try:
        import geopandas as gpd
    except Exception as exc:
        raise ImportError("geopandas is required to read vector tie point formats") from exc

    if ext == ".gpkg":
        # If no layer specified, try the pipeline's common layers
        layers_to_try = [layer] if layer else ["all_depths", "train_depths", "test_depths"]
        gdf = None
        last_exc = None
        for lyr in layers_to_try:
            try:
                gdf = gpd.read_file(p, layer=lyr)
                if gdf is not None and len(gdf) > 0:
                    layer = lyr
                    break
            except Exception as exc:
                last_exc = exc
                continue
        if gdf is None or len(gdf) == 0:
            raise RuntimeError(f"Could not read any tie-point layer from {p} (tried: {layers_to_try}). {last_exc}")
    else:
        gdf = gpd.read_file(p)

    if gdf is None or len(gdf) == 0:
        raise ValueError(f"No features found in tie-point file: {p}")

    # normalize geometry -> lon/lat
    if gdf.geometry is None:
        raise ValueError(f"Vector has no geometry: {p}")
    if gdf.crs is None:
        # assume lon/lat
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf_ll = gdf.to_crs("EPSG:4326")

    # columns
    dcol = _guess_depth_col(gdf_ll.columns)
    if dcol is None:
        # try in original gdf
        dcol = _guess_depth_col(gdf.columns)
    if dcol is None:
        raise ValueError(f"Could not find a depth column in {p} (columns: {list(gdf.columns)[:50]})")

    out = gdf_ll.copy()
    out["longitude"] = out.geometry.x.astype("float64")
    out["latitude"] = out.geometry.y.astype("float64")
    out["depth_m"] = out[dcol].astype("float64")
    if source_name and "source" not in out.columns:
        out["source"] = source_name

    return TiePoints.from_df(out[["longitude", "latitude", "depth_m"] + (["source"] if "source" in out.columns else [])],
                             depth_positive_down=depth_positive_down)


@dataclass
class TiePoints:
    lon: np.ndarray
    lat: np.ndarray
    depth_m: np.ndarray
    source: Optional[np.ndarray] = None
    depth_positive_down: Optional[bool] = None  # if known

    @staticmethod
    def from_df(df, *, depth_positive_down: Optional[bool] = None) -> "TiePoints":
        lon = df["longitude"].to_numpy(dtype=np.float64)
        lat = df["latitude"].to_numpy(dtype=np.float64)
        depth = df["depth_m"].to_numpy(dtype=np.float64)
        src = None
        if "source" in df.columns:
            src = df["source"].astype(str).to_numpy()
        m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(depth)
        if not np.any(m):
            raise ValueError("No finite tie points")
        return TiePoints(lon=lon[m], lat=lat[m], depth_m=depth[m], source=(src[m] if src is not None else None),
                         depth_positive_down=depth_positive_down)

    def subset_by_source(self, sources: Sequence[str]) -> "TiePoints":
        if self.source is None:
            return self
        want = set([s.lower() for s in sources])
        m = np.array([str(s).lower() in want for s in self.source], dtype=bool)
        if not np.any(m):
            return TiePoints(self.lon[:0], self.lat[:0], self.depth_m[:0], self.source[:0], self.depth_positive_down)
        return TiePoints(self.lon[m], self.lat[m], self.depth_m[m], self.source[m], self.depth_positive_down)

    def concat(self, other: "TiePoints") -> "TiePoints":
        if other is None or other.lon.size == 0:
            return self
        src = None
        if self.source is not None or other.source is not None:
            s1 = self.source if self.source is not None else np.array(["unknown"] * self.lon.size, dtype=object)
            s2 = other.source if other.source is not None else np.array(["unknown"] * other.lon.size, dtype=object)
            src = np.concatenate([s1, s2])
        dpd = self.depth_positive_down if self.depth_positive_down is not None else other.depth_positive_down
        return TiePoints(np.concatenate([self.lon, other.lon]),
                         np.concatenate([self.lat, other.lat]),
                         np.concatenate([self.depth_m, other.depth_m]),
                         src,
                         dpd)

# ----------------------------
# Residual sampling + stats
# ----------------------------

def _sample_raster_at_lonlat(raster_path: str, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    import rasterio
    from pyproj import Transformer

    with rasterio.open(raster_path) as ds:
        nod = ds.nodata
        crs = ds.crs
        if crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = tfm.transform(lon, lat)
        vals = np.fromiter((v[0] for v in ds.sample(zip(xs, ys), indexes=1)),
                           dtype=np.float64, count=len(lon))
        if nod is not None and np.isfinite(nod):
            vals[vals == float(nod)] = np.nan
        info = {
            "crs": str(crs),
            "nodata": (None if nod is None else float(nod)),
            "bounds": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
        }
        return vals, info

def _ensure_positive_down(depth: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return depth as positive-down and a flip factor to recover original sign."""
    d = np.asarray(depth, dtype=np.float64)
    m = np.isfinite(d)
    if not np.any(m):
        return d, 1.0
    flip = -1.0 if np.nanmedian(d[m]) < 0 else 1.0
    return d * flip, flip

def residuals_against_points(raster_path: str, pts: TiePoints) -> Dict[str, Any]:
    pred_raw, info = _sample_raster_at_lonlat(raster_path, pts.lon, pts.lat)

    ref_raw = pts.depth_m.astype(np.float64)

    pred_pd, pred_flip = _ensure_positive_down(pred_raw)
    ref_pd, ref_flip = _ensure_positive_down(ref_raw)

    # If both already positive-down, flips are 1. If one is negative-down, we normalize.
    m = np.isfinite(pred_pd) & np.isfinite(ref_pd)
    if not np.any(m):
        return {"status": "skip", "reason": "no valid overlapping samples", "n": 0, "raster_info": info}

    res = pred_pd[m] - ref_pd[m]  # positive = predicted deeper than reference
    out = {
        "status": "ok",
        "n": int(np.count_nonzero(m)),
        "pred_flip": float(pred_flip),
        "ref_flip": float(ref_flip),
        "residuals_m": res,   # positive-down convention
        "pred_pd_m": pred_pd[m],
        "ref_pd_m": ref_pd[m],
        "raster_info": info,
    }
    return out

def _detect_bimodal(residuals: np.ndarray, min_samples: int = 50) -> Dict[str, Any]:
    """
    Detect if residual distribution is bimodal using kernel density estimation.
    
    In practice, residual distributions can be multi-modal (e.g., mixed bottom types,
    mask edge effects, or datum/quality heterogeneity). In those cases RMSE can be
    misleading and robust summaries (MAD/quantiles) are often more informative.
    
    Returns dict with:
        - is_bimodal: bool
        - n_modes: int (number of detected peaks)
        - mode_locations: list of peak locations
        - warning: str if bimodal detected
    """
    r = np.asarray(residuals, dtype=np.float64)
    m = np.isfinite(r)
    r = r[m]
    
    if len(r) < min_samples:
        return {"is_bimodal": False, "n_modes": 0, "reason": "insufficient_samples"}
    
    try:
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks
        
        # Fit KDE
        kde = gaussian_kde(r, bw_method='scott')
        x = np.linspace(r.min(), r.max(), 200)
        density = kde(x)
        
        # Find peaks with reasonable prominence
        prominence = 0.1 * np.max(density)
        peaks, properties = find_peaks(density, prominence=prominence, distance=10)
        
        n_modes = len(peaks)
        mode_locs = [float(x[p]) for p in peaks]
        
        result = {
            "is_bimodal": n_modes >= 2,
            "n_modes": n_modes,
            "mode_locations_m": mode_locs,
        }
        
        if n_modes >= 2:
            result["warning"] = (
                "Bimodal error distribution detected. RMSE may be misleading. "
                "Consider using MAD (median absolute deviation) and quantiles instead."
            )
        
        return result
        
    except ImportError:
        return {"is_bimodal": False, "n_modes": 0, "reason": "scipy_not_available"}
    except Exception as e:
        return {"is_bimodal": False, "n_modes": 0, "reason": f"error: {str(e)}"}


def _residual_summary(res: np.ndarray) -> Dict[str, Any]:
    r = np.asarray(res, dtype=np.float64)
    m = np.isfinite(r)
    if not np.any(m):
        return {"n": 0}
    absr = np.abs(r[m])
    
    summary = {
        "n": int(np.count_nonzero(m)),
        "median_m": float(np.nanmedian(r[m])),
        "mean_m": float(np.nanmean(r[m])),
        "mad_m": float(np.nanmedian(np.abs(r[m] - np.nanmedian(r[m])))),
        "rmse_m": float(np.sqrt(np.nanmean(r[m] ** 2))),
        "p90_abs_m": float(np.nanpercentile(absr, 90)),
        "p99_abs_m": float(np.nanpercentile(absr, 99)),
    }
    
    # Add bimodal detection (helps flag when RMSE is not representative)
    bimodal_info = _detect_bimodal(r[m])
    summary["bimodal"] = bimodal_info
    
    return summary

# ----------------------------
# Fit models for Δz
# ----------------------------

def _depth_bins_from_spec(spec: str) -> np.ndarray:
    """
    spec can be:
      - '0,2,5,10,20,40'
      - 'auto' (returns a conservative set)
    """
    if spec is None or str(spec).strip().lower() == "auto":
        return np.array([0, 2, 5, 10, 20, 30, 40], dtype=np.float64)
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    vals = [float(p) for p in parts]
    vals = sorted(set(vals))
    if len(vals) < 3:
        raise ValueError("depth bin spec must have at least 3 edges")
    return np.array(vals, dtype=np.float64)

def fit_median_shift(residuals_m: np.ndarray) -> Dict[str, float]:
    dz = float(np.nanmedian(residuals_m))
    return {"dz_m": dz}

def fit_depth_stratified_shift(ref_depth_pd_m: np.ndarray, residuals_m: np.ndarray, *, bins: np.ndarray) -> Dict[str, Any]:
    """
    Compute median residual within depth bins.
    Returns edges + dz per bin.
    """
    z = np.asarray(ref_depth_pd_m, dtype=np.float64)
    r = np.asarray(residuals_m, dtype=np.float64)
    m = np.isfinite(z) & np.isfinite(r)
    z = z[m]; r = r[m]
    if z.size == 0:
        return {"edges": bins.tolist(), "dz_by_bin_m": [0.0]*(len(bins)-1), "n_by_bin": [0]*(len(bins)-1)}

    dz = []
    n_by = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        mm = (z >= lo) & (z < hi)
        n = int(np.count_nonzero(mm))
        n_by.append(n)
        dz.append(float(np.nanmedian(r[mm])) if n > 0 else 0.0)
    return {"edges": bins.tolist(), "dz_by_bin_m": dz, "n_by_bin": n_by}

def _huber_weights(u: np.ndarray, c: float) -> np.ndarray:
    a = np.abs(u)
    w = np.ones_like(u, dtype=np.float64)
    m = a > c
    w[m] = c / np.maximum(a[m], 1e-12)
    return w

def fit_planar_shift_xy(
    raster_path: str,
    pts: TiePoints,
    residuals_m: np.ndarray,
    *,
    huber_c_m: float = 1.0,
    max_iter: int = 20,
) -> Dict[str, Any]:
    """
    Fit Δz(x,y) = a + b*x + c*y in raster CRS, using robust IRLS (Huber).
    """
    import rasterio
    from pyproj import Transformer

    with rasterio.open(raster_path) as ds:
        crs = ds.crs
        if crs is None:
            raise ValueError("Raster CRS is required for planar fit")
        tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = tfm.transform(pts.lon, pts.lat)

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    r = np.asarray(residuals_m, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
    x = x[m]; y = y[m]; r = r[m]
    if x.size < 3:
        return {"status": "skip", "reason": "insufficient points for plane fit"}

    # Design matrix
    A = np.column_stack([np.ones_like(x), x, y])

    w = np.ones_like(r)
    coef = np.zeros(3, dtype=np.float64)

    for _ in range(max_iter):
        W = w[:, None]
        Aw = A * W
        rw = r * w
        try:
            coef_new, *_ = np.linalg.lstsq(Aw, rw, rcond=None)
        except Exception:
            break
        u = r - (A @ coef_new)
        # robust scale (MAD)
        s = np.nanmedian(np.abs(u - np.nanmedian(u))) * 1.4826
        s = float(s) if np.isfinite(s) and s > 1e-9 else 1.0
        w_new = _huber_weights(u / s, c=huber_c_m)
        if np.max(np.abs(coef_new - coef)) < 1e-10:
            coef = coef_new
            break
        coef = coef_new
        w = w_new

    return {"status": "ok", "a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]), "huber_c_m": float(huber_c_m)}

# ----------------------------
# Apply alignment to raster
# ----------------------------

def _write_aligned_raster(
    src_raster: str,
    dst_raster: str,
    *,
    mode: str,
    fit: Dict[str, Any],
    pred_flip: float,
    nodata_val: Optional[float] = None,
    tile_size: int = 1024,
) -> None:
    """
    Apply alignment on disk, preserving raster profile and nodata.
    The alignment is applied in positive-down space and then converted back using pred_flip.
    """
    import rasterio
    from rasterio.windows import Window

    mode = str(mode).lower()

    with rasterio.open(src_raster) as src:
        prof = src.profile.copy()
        nod = src.nodata if nodata_val is None else nodata_val
        if nod is None:
            nod = prof.get("nodata", None)
        prof.update(driver="GTiff", compress="DEFLATE", tiled=True)
        if nod is not None:
            prof["nodata"] = nod

        w = src.width
        h = src.height
        transform = src.transform

        # Precompute plane coefficients if planar
        if mode == "planar":
            a = float(fit.get("a", 0.0))
            b = float(fit.get("b", 0.0))
            c = float(fit.get("c", 0.0))

        # Depth bin info if stratified
        if mode == "depth":
            edges = np.array(fit.get("edges", []), dtype=np.float64)
            dz_by = np.array(fit.get("dz_by_bin_m", []), dtype=np.float64)

        with rasterio.open(dst_raster, "w", **prof) as dst:
            for row_off in range(0, h, tile_size):
                for col_off in range(0, w, tile_size):
                    win = Window(col_off, row_off,
                                 min(tile_size, w - col_off),
                                 min(tile_size, h - row_off))
                    arr = src.read(1, window=win).astype(np.float64)

                    # mask nodata
                    if nod is not None and np.isfinite(nod):
                        m = np.isfinite(arr) & (arr != float(nod))
                    else:
                        m = np.isfinite(arr)

                    if not np.any(m):
                        dst.write(arr.astype(prof["dtype"]), 1, window=win)
                        continue

                    # normalize to positive-down
                    arr_pd = arr * pred_flip

                    if mode == "median":
                        dz = float(fit.get("dz_m", 0.0))
                        arr_pd[m] = arr_pd[m] - dz

                    elif mode == "depth":
                        # assign dz based on depth bin of predicted depth (positive-down)
                        z = arr_pd
                        dz_map = np.zeros_like(z, dtype=np.float64)
                        if edges.size >= 3 and dz_by.size == edges.size - 1:
                            for i in range(edges.size - 1):
                                lo, hi = edges[i], edges[i+1]
                                mm = m & (z >= lo) & (z < hi)
                                if np.any(mm):
                                    dz_map[mm] = dz_by[i]
                        arr_pd[m] = arr_pd[m] - dz_map[m]

                    elif mode == "planar":
                        # compute x,y for cell centers in raster CRS
                        # x = a0 + col*dx + row*rx; y = b0 + col*ry + row*dy
                        cols = np.arange(col_off, col_off + win.width)
                        rows = np.arange(row_off, row_off + win.height)
                        cc, rr = np.meshgrid(cols, rows)
                        xs, ys = rasterio.transform.xy(transform, rr, cc, offset="center")
                        xs = np.asarray(xs, dtype=np.float64)
                        ys = np.asarray(ys, dtype=np.float64)
                        dz = a + b * xs + c * ys
                        arr_pd[m] = arr_pd[m] - dz[m]

                    else:
                        raise ValueError(f"Unknown alignment mode: {mode}")

                    # convert back to original sign convention
                    arr_aligned = arr_pd / pred_flip

                    dst.write(arr_aligned.astype(prof["dtype"]), 1, window=win)

def _plot_residuals(pre: np.ndarray, post: np.ndarray, ref_depth_pd: np.ndarray, out_png: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre = np.asarray(pre, dtype=np.float64)
    post = np.asarray(post, dtype=np.float64)
    z = np.asarray(ref_depth_pd, dtype=np.float64)

    m = np.isfinite(pre) & np.isfinite(post) & np.isfinite(z)
    if np.count_nonzero(m) < 5:
        return

    pre = pre[m]; post = post[m]; z = z[m]

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.hist(pre, bins=50)
    ax1.set_title("Pre residuals (pred-ref)")
    ax1.set_xlabel("m"); ax1.set_ylabel("count")

    ax2.hist(post, bins=50)
    ax2.set_title("Post residuals (pred-ref)")
    ax2.set_xlabel("m"); ax2.set_ylabel("count")

    ax3.scatter(z, pre, s=5, alpha=0.35, label="pre")
    ax3.scatter(z, post, s=5, alpha=0.35, label="post")
    ax3.set_title("Residual vs depth")
    ax3.set_xlabel("Reference depth (m, +down)")
    ax3.set_ylabel("Residual (m)")
    ax3.legend(loc="best", markerscale=2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# ----------------------------
# Main API
# ----------------------------

def align_raster_to_tie_points(
    raster_path: str,
    tie_points: TiePoints,
    *,
    mode: str = "median",
    out_raster_path: Optional[str] = None,
    min_points: int = 100,
    depth_bins: str = "auto",
    planar_huber_c_m: float = 1.0,
    max_iter: int = 20,
    max_abs_residual_m_for_fit: Optional[float] = None,
    plots_dir: Optional[str] = None,
    plot_name: str = "Alignment_Residuals.png",
    title: str = "SDB residual alignment",
) -> Dict[str, Any]:
    """
    Align a raster to tie points.

    Returns a dict suitable to embed into predict_report.json and run_report.json.
    """
    mode = str(mode).lower().strip()
    if mode in ("none", "", "off", "false"):
        return {"status": "skip", "reason": "alignment disabled"}

    if tie_points is None or tie_points.lon.size == 0:
        return {"status": "skip", "reason": "no tie points"}

    # pre residuals
    pre = residuals_against_points(raster_path, tie_points)
    if pre.get("status") != "ok":
        return {"status": "skip", "reason": pre.get("reason", "no residuals"), "pre": pre}

    n = int(pre["n"])
    if n < int(min_points):
        return {"status": "skip", "reason": f"insufficient tie points (n={n} < min_points={min_points})", "pre": {"summary": _residual_summary(pre["residuals_m"])}}

    res = np.asarray(pre["residuals_m"], dtype=np.float64)
    ref_pd = np.asarray(pre["ref_pd_m"], dtype=np.float64)
    pred_flip = float(pre.get("pred_flip", 1.0))

    # Optional outlier gate for fit
    if max_abs_residual_m_for_fit is not None:
        mm = np.isfinite(res) & (np.abs(res) <= float(max_abs_residual_m_for_fit))
        if np.count_nonzero(mm) >= max(10, min_points):
            res_fit = res[mm]
            ref_fit = ref_pd[mm]
        else:
            res_fit = res
            ref_fit = ref_pd
    else:
        res_fit = res
        ref_fit = ref_pd

    fit: Dict[str, Any] = {}
    gate: Dict[str, Any] = {}

    # Planar geometry gate (avoid fitting planes on single-track geometry)
    if mode == "planar":
        try:
            import rasterio
            from pyproj import Transformer
            with rasterio.open(raster_path) as ds:
                b = ds.bounds
                rx = b.right - b.left
                ry = b.top - b.bottom
                tfm = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                xs, ys = tfm.transform(tie_points.lon, tie_points.lat)
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            xspan = float(np.nanmax(xs) - np.nanmin(xs))
            yspan = float(np.nanmax(ys) - np.nanmin(ys))
            # require some spread in both dimensions
            gate["raster_span_xy"] = [float(rx), float(ry)]
            gate["tie_span_xy"] = [float(xspan), float(yspan)]
            if xspan < 0.15 * rx or yspan < 0.15 * ry:
                gate["planar_allowed"] = False
                gate["planar_reason"] = "tie points lack 2D coverage (likely single track); falling back to median"
                mode = "median"
            else:
                gate["planar_allowed"] = True
        except Exception as exc:
            gate["planar_allowed"] = False
            gate["planar_reason"] = f"planar gate failed: {exc}; falling back to median"
            mode = "median"

    if mode == "median":
        fit = {"mode": "median", **fit_median_shift(res_fit)}
    elif mode in ("depth", "depth_stratified", "stratified"):
        bins = _depth_bins_from_spec(depth_bins)
        fit = {"mode": "depth", **fit_depth_stratified_shift(ref_fit, res_fit, bins=bins)}
    elif mode == "planar":
        pf = fit_planar_shift_xy(raster_path, tie_points, res_fit, huber_c_m=planar_huber_c_m, max_iter=max_iter)
        if pf.get("status") != "ok":
            # fallback
            fit = {"mode": "median", **fit_median_shift(res_fit), "fallback_from": "planar", "fallback_reason": pf.get("reason", "planar fit failed")}
            mode = "median"
        else:
            fit = {"mode": "planar", **pf}
    else:
        return {"status": "skip", "reason": f"unknown align mode: {mode}"}

    # Apply
    out_r = out_raster_path if out_raster_path else raster_path
    did_write_new = False
    if out_raster_path:
        _write_aligned_raster(raster_path, out_raster_path, mode=fit["mode"], fit=fit, pred_flip=pred_flip)
        did_write_new = True
    else:
        # in-place is dangerous; require explicit out_raster_path
        return {"status": "skip", "reason": "out_raster_path required (no in-place writes)", "pre": {"summary": _residual_summary(res)}}

    # Post residuals
    post = residuals_against_points(out_r, tie_points)
    post_summary = _residual_summary(np.asarray(post.get("residuals_m", np.array([]))))

    pre_summary = _residual_summary(res)

    # Plot
    plot_path = None
    if plots_dir:
        try:
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            plot_path = str(Path(plots_dir) / plot_name)
            _plot_residuals(res, np.asarray(post.get("residuals_m", np.array([]))), ref_pd, plot_path, title=title)
        except Exception as exc:
            log.warning("Plot generation failed: %s", exc)
            plot_path = None

    out = {
        "status": "ok",
        "mode": fit.get("mode"),
        "min_points": int(min_points),
        "n_tie_points": int(n),
        "pre": {"summary": pre_summary},
        "post": {"summary": post_summary},
        "fit": {k: v for k, v in fit.items() if k not in ("status",)},
        "gate": gate,
        "outputs": {
            "aligned_raster": str(out_r),
            "residual_plot": plot_path,
        },
        "notes": [
            "Residuals computed in +down convention (positive residual means predicted deeper than reference).",
            "Alignment applied in +down space and converted back to original raster sign convention."
        ]
    }
    out["wrote_new_raster"] = bool(did_write_new)
    return out

def align_to_atl24(
    raster_path: str,
    icesat_gpkg_path: str,
    *,
    mode: str = "median",
    tiepoint_mode: str = "stacked",
    tiepoint_source_col: str = "source",
    tiepoint_per_source_max: int | None = None,
    tiepoint_seed: int = 42,
    allow_empty: bool = False,
    out_raster_path: Optional[str] = None,
    min_points: int = 100,
    depth_bins: str = "auto",
    plots_dir: Optional[str] = None,
    plot_name: str = "Alignment_ATL24_Residuals.png",
) -> Dict[str, Any]:
    """
    Convenience wrapper: load ATL24 points from the pipeline gpkg and align.
    """
    tp_all = read_tie_points(icesat_gpkg_path, layer="all_depths")

    # __TIEPOINT_SELECTION_APPLIED__
    tp_sel = select_tiepoints(
        tp_all,
        mode=tiepoint_mode,
        source_col=tiepoint_source_col,
        per_source_max=tiepoint_per_source_max,
        seed=tiepoint_seed,
    )
    if tp_sel is None or len(tp_sel) < 1:
        msg = f"[ALIGN] No tie points after selection mode='{tiepoint_mode}'."
        if allow_empty:
            log.warning(msg + " Skipping alignment.")
            return {
                "applied": False,
                "reason": "no_tie_points_after_selection",
                "tiepoint_mode": tiepoint_mode,
                "n_tie_points": 0,
            }
        raise ValueError(msg + " Use allow_empty=True to skip instead.")
    # tiepoint selection (deterministic)
    tp_use = tp_sel
    tie_used = str(tiepoint_mode)

    # Prefer ATL24-only tie points when using the default 'stacked' mode and there are
    # enough ATL24 points to meet min_points. This improves alignment stability by
    # avoiding mixed-source vertical behavior.
    mode_l = str(tiepoint_mode or "").strip().lower()
    if mode_l in ("stacked", "auto", "") and getattr(tp_all, "source", None) is not None:
        tp_24 = tp_all.subset_by_source(["atl24"])
        if tp_24 is not None and getattr(tp_24, "lon", None) is not None and tp_24.lon.size >= min_points:
            tp_use = select_tiepoints(
                tp_all,
                mode="atl24",
                source_col=tiepoint_source_col,
                per_source_max=tiepoint_per_source_max,
                seed=tiepoint_seed,
            )
            if tp_use is None or len(tp_use) < 1:
                tp_use = tp_24
            tie_used = "atl24"
    out = align_raster_to_tie_points(
        raster_path,
        tp_use,
        mode=mode,
        out_raster_path=out_raster_path,
        min_points=min_points,
        depth_bins=depth_bins,
        plots_dir=plots_dir,
        plot_name=plot_name,
        title=f"SDB alignment ({tie_used})",
    )
    out["tie_points_source"] = tie_used
    return out
