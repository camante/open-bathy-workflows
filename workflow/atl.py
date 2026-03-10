#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
atl.py – ICESat-2 (ATL03/ATL24) utilities for the Open Bathy Workflows.

Handles data discovery and download via NASA Harmony/CMR (cache-first), training-point
extraction from ATL03/ATL24 photons with optional land masking, refraction correction
for ATL03 underwater photons, and ingestion of extra XYZ soundings.

Examples:
    # ATL03 photons -> training points CSV
    python atl.py \
      --aoi "-74.5/-74.25/40.25/40.5" \
      --start 2025-01-01 --end 2026-01-01 \
      --product atl03 \
      --cache-dir cache/icesat2 \
      --out-csv output/atl03_points.csv

    # ATL24 bathymetry points -> training points CSV
    python atl.py \
      --aoi "-74.5/-74.25/40.25/40.5" \
      --start 2025-01-01 --end 2026-01-01 \
      --product atl24 \
      --cache-dir cache/icesat2 \
      --out-csv output/atl24_points.csv

Notes
-----
- AOI format is **W/E/S/N** (lon/lat degrees), matching the rest of the workflow.
- This module is imported by higher-level entry points (e.g., `sdb_main.py`),
  so avoid adding heavy imports at module import time unless necessary.
"""

import os
import sys
import shutil
import zipfile
import time
import logging
import hashlib
import argparse
import textwrap
import shlex
from pathlib import Path
from datetime import datetime, date, timezone
from typing import List, Dict, Tuple, Optional, Any, Union

def _lazy_requests():
    """Import requests only when needed.

    Rationale: this module is imported by orchestration scripts; keeping heavy
    imports lazy improves startup time and avoids import-time issues on mixed
    HPC stacks.
    """
    try:
        import requests  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("The 'requests' package is required for ATL downloads.") from e
    return requests


def _lazy_h5py():
    """Import h5py only when needed (ATL HDF5 readers)."""
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("The 'h5py' package is required to read ATL HDF5 files.") from e
    return h5py
import numpy as np
import pandas as pd

# Ensure GeoPandas remains usable on pandas>=2.0 even if GeoPandas lags.
import compat_pandas  # noqa: F401

# -----------------------------------------------------------------------------
# Optional exact-match caching utilities
# -----------------------------------------------------------------------------
try:
    from cache_utils import (
        canonical_json,
        fingerprint_file,
        fingerprint_code,
        artifact_cache_key,
        meta_payload,
        read_meta,
        write_meta,
        cache_hit,
    )
    _CACHE_UTILS_AVAILABLE = True
except Exception:  # pragma: no cover
    _CACHE_UTILS_AVAILABLE = False


# -----------------------------------------------------------------------------
# Optional run reporting (flight recorder)
# -----------------------------------------------------------------------------
try:
    from log_report import RunReport
except Exception:  # pragma: no cover
    RunReport = None  # type: ignore

def _rr_add(rr, key, value):
    """Best-effort add to RunReport without ever breaking the pipeline."""
    try:
        if rr is not None:
            rr.add(key, value)
    except Exception:
        log.debug("ignored", exc_info=True)

def _rr_artifact(rr, kind: str, path: str):
    """Record an artifact path in the run report."""
    try:
        if rr is not None and hasattr(rr, "record_artifact"):
            rr.record_artifact(kind, path)
    except Exception:
        log.debug("run-recorder add failed", exc_info=True)


def _df_depth_summary(df: pd.DataFrame, depth_col: str = "depth_m") -> Dict[str, Any]:
    """Return depth stats for quick 'funnel' debugging (never raises)."""
    out: Dict[str, Any] = {"n": int(len(df)) if df is not None else 0}
    try:
        if df is None or df.empty or depth_col not in df.columns:
            return out
        v = pd.to_numeric(df[depth_col], errors="coerce").to_numpy()
        v = v[np.isfinite(v)]
        out["finite_n"] = int(v.size)
        if v.size == 0:
            return out
        qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        qv = np.nanpercentile(v, qs)
        for q, val in zip(qs, qv):
            out[f"p{q:02d}"] = float(val)
        out["neg_n"] = int(np.count_nonzero(v < 0))
        out["zero_n"] = int(np.count_nonzero(v == 0))
        # A small, fixed histogram (0..40m, 1m bins) for quick sanity.
        edges = np.arange(0.0, 41.0, 1.0)
        h, _ = np.histogram(v, bins=edges)
        out["hist_0_40_1m"] = h.tolist()
    except Exception:
        return out
    return out

def _log_depth_funnel(stage: str, df: pd.DataFrame, *, rr=None, depth_col: str = "depth_m") -> None:
    """Log + RunReport a compact depth summary for a dataframe stage."""
    try:
        s = _df_depth_summary(df, depth_col=depth_col)
        # Human log line (compact)
        if "finite_n" in s and s.get("finite_n", 0) > 0:
            log.info(
                f"[Funnel] {stage}: n={s.get('n')} finite={s.get('finite_n')} "
                f"p50={s.get('p50', float('nan')):.2f} "
                f"p95={s.get('p95', float('nan')):.2f} "
                f"max={s.get('p100', float('nan')):.2f}"
            )
        else:
            log.info("%s: n=%s (no '%s' or no finite values)", stage, s.get('n'), depth_col)
        # Machine log
        _rr_add(rr, f"funnel.{stage}.n", int(s.get("n", 0)))
        if "finite_n" in s:
            _rr_add(rr, f"funnel.{stage}.finite_n", int(s.get("finite_n", 0)))
        for k in ("p00","p01","p05","p25","p50","p75","p95","p99","p100","neg_n","zero_n"):
            if k in s:
                _rr_add(rr, f"funnel.{stage}.depth.{k}", s[k])
        if "hist_0_40_1m" in s:
            _rr_add(rr, f"funnel.{stage}.depth.hist_0_40_1m", s["hist_0_40_1m"])
    except Exception:
        log.debug("ignored", exc_info=True)

# Heavy geospatial imports are lazy: this module may be used in environments
# where GDAL/GeoPandas are optional. Import them inside functions that need them.

try:
    from harmony import Client as HarmonyClient, BBox, Request, Collection
except ImportError:
    log = logging.getLogger("sdb.atl")


def transform_xyz_dataframe_crs(
    df: pd.DataFrame,
    *,
    src_srs: str,
    dst_srs: str,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    z_col: str = "depth_m",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Transform a lon/lat dataframe between CRSs using PROJ via pyproj.

    IMPORTANT: This performs HORIZONTAL-ONLY transformation (lon/lat).
    The z_col (depth_m) is NOT transformed because depth is a relative measurement
    (water depth below surface), not a geodetic/orthometric height.

    Applying vertical datum transforms (e.g., EGM2008 -> NAVD88) to depth values
    would corrupt them by introducing location-dependent biases.

    Notes:
    - Only longitude and latitude are transformed between horizontal datums
    - depth_m remains unchanged (it's relative to water surface, not a datum)
    - If the transform fails, the input df is returned unchanged.
    """
    if df is None or len(df) == 0:
        return df
    if (not src_srs) or (not dst_srs):
        return df

    # Extract just the horizontal CRS components for transformation
    # Strip any vertical component (e.g., "EPSG:4326+5703" -> "EPSG:4326")
    def _horizontal_crs(crs_str: str) -> str:
        s = str(crs_str).strip()
        if "+" in s:
            # Compound CRS like "EPSG:4326+5703" - take first part
            return s.split("+")[0].strip()
        return s

    src_h = _horizontal_crs(src_srs)
    dst_h = _horizontal_crs(dst_srs)

    if src_h.lower() == dst_h.lower():
        return df

    logx = logger or log
    try:
        from pyproj import Transformer
        t = Transformer.from_crs(src_h, dst_h, always_xy=True)
        xs = df[lon_col].to_numpy(dtype=float)
        ys = df[lat_col].to_numpy(dtype=float)
        x2, y2 = t.transform(xs, ys)
        out = df.copy()
        out[lon_col] = x2
        out[lat_col] = y2
        # depth_m (z_col) is not transformed: it's relative depth, not geodetic height
        logx.info(f"[ATL][CRS] Transformed {len(out)} points (horizontal only): {src_h} -> {dst_h}")
        logx.info(f"[ATL][CRS] depth_m NOT transformed (it's relative depth, not geodetic height)")
        return out
    except Exception as e:
        logx.warning(f"[ATL][CRS] Could not transform ATL dataframe ({src_h} -> {dst_h}): {e}")
        return df

# -----------------------------------------------------------------------------
# Constants & Logging
# -----------------------------------------------------------------------------

ATL_CACHE_VERSION = "atl_mod_v1_cache"
ATL24_CONFIDENCE_DEFAULT = 0.9


def _atl_exact_cache_key(*, stage: str, params: Dict[str, Any], inputs: Dict[str, Any], cache_code_strict: bool = False, cache_ignore_code: bool = True) -> str:
    """Compute exact-match cache key for derived ATL training points."""
    if not _CACHE_UTILS_AVAILABLE:
        return ""
    code_fp = ""
    if not cache_ignore_code:
        try:
            code_fp = fingerprint_code(Path(__file__), strict=bool(cache_code_strict))
        except Exception:
            code_fp = ""
    return artifact_cache_key(stage=stage, params=params, inputs=inputs, code_fp=code_fp, key_len=20)


def _maybe_load_cached_training_points(
    *,
    stage: str,
    cache_dir: Optional[Union[str, Path]],
    params: Dict[str, Any],
    inputs: Dict[str, Any],
    cache_strict: bool = False,
    cache_code_strict: bool = False,
    cache_ignore_code: bool = True,
    rr=None,
):
    """Exact-match cache read for derived training points.

    Returns:
        (df_or_None, reason, key, data_path, meta_path)
    """
    if (not _CACHE_UTILS_AVAILABLE) or (cache_dir is None):
        return None, "cache_disabled", None, None, None

    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    key = _atl_exact_cache_key(stage=stage, params=params, inputs=inputs, cache_code_strict=cache_code_strict, cache_ignore_code=cache_ignore_code)
    if not key:
        return None, "cache_utils_unavailable", None, None, None

    data_path = cdir / f"{stage}_{key}.pkl"
    meta_path = cdir / f"{stage}_{key}.meta.json"

    # Standard cache check (meta-driven)
    hit, reason = cache_hit(meta_path, expected_key=key, require_exact_params=True, expected_params=params)
    _rr_add(rr, f"{stage}.cache.enabled", True)
    _rr_add(rr, f"{stage}.cache.key", key)
    _rr_add(rr, f"{stage}.cache.check", reason)

    # Legacy upgrade path: older runs may have written the pickle but not the meta.
    # Treat that as a HIT (if readable) and write meta now so future runs are clean.
    if (not hit) and reason == "no_meta" and data_path.exists() and data_path.stat().st_size > 0:
        try:
            df = pd.read_pickle(data_path)
            try:
                code_fp = fingerprint_code(Path(__file__), strict=bool(cache_code_strict))
            except Exception:
                code_fp = ""
            payload = {
                "cache_key": key,
                "stage": stage,
                "params": params,
                "inputs": inputs,
                "code_fingerprint": code_fp,
                "extra": {"rows": int(len(df)), "upgraded_from": "legacy_no_meta"},
            }
            try:
                write_meta(meta_path, payload)
                reason = "legacy_no_meta_upgraded"
            except Exception:
                reason = "legacy_no_meta"
            _rr_add(rr, f"{stage}.cache.legacy_upgrade", True)
            return df, "hit_legacy_no_meta", key, data_path, meta_path
        except Exception:
            return None, "read_failed", key, data_path, meta_path

    if hit and data_path.exists() and data_path.stat().st_size > 0:
        try:
            df = pd.read_pickle(data_path)
            return df, "hit", key, data_path, meta_path
        except Exception:
            return None, "read_failed", key, data_path, meta_path

    return None, reason, key, data_path, meta_path


def _write_cached_training_points(
    *,
    stage: str,
    key: str,
    data_path: Path,
    meta_path: Path,
    params: Dict[str, Any],
    inputs: Dict[str, Any],
    cache_code_strict: bool = False,
    df: pd.DataFrame,
    extra: Optional[Dict[str, Any]] = None,
):
    """Exact-match cache write (best-effort)."""
    if not _CACHE_UTILS_AVAILABLE:
        return
    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(data_path)
        try:
            code_fp = fingerprint_code(Path(__file__), strict=bool(cache_code_strict))
        except Exception:
            code_fp = ""
        payload = meta_payload(
            cache_key=key,
            stage=stage,
            params=params,
            inputs=inputs,
            code_fp=code_fp,
            extra=extra or {},
        )
        write_meta(meta_path, payload)
    except Exception:
        log.debug("write_meta failed for %s", meta_path, exc_info=True)

# Use centralized logging - get logger, don't configure root here
log = logging.getLogger("sdb.atl")


# -----------------------------------------------------------------------------
# Small utility helpers
# -----------------------------------------------------------------------------

def _iso(d: Any, end: bool = False) -> str:
    if isinstance(d, str):
        dt = datetime.fromisoformat(d)
    elif isinstance(d, date):
        dt = datetime(d.year, d.month, d.day)
    else:
        raise TypeError(f"Unsupported date type for _iso: {type(d)}")
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    else:
        dt = dt.replace(hour=0, minute=0, second=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_aoi_string(aoi: str) -> Tuple[float, float, float, float]:
    W, E, S, N = [float(x) for x in aoi.split("/")]
    return W, E, S, N

def _atl_cache_key(aoi_str: str, start: str, end: str, product_key: str) -> str:
    s = f"{ATL_CACHE_VERSION}|{aoi_str}|{start}|{end}|{product_key}"
    return hashlib.sha1(s.encode()).hexdigest()[:12]

def _filter_points_by_mask(
    df: pd.DataFrame,
    mask_path: str,
    *,
    # Backward-compat knobs:
    water_val: int = 0,
    invert: bool = False,
    # New semantics-aware knobs:
    mask_type: str = "auto",
    threshold: float = 0.5,
    rr=None,
) -> pd.DataFrame:
    """
    Filter dataframe points using a raster land/water mask with explicit semantics.

    Supported mask_type values:
      - "auto": infer whether 0 or 1 represents water for simple binary masks by sampling at points.
      - "water_only": keep points where sampled == water_val  (commonly waffles: 0=water, 1=land)
      - "land_binary": keep points where sampled == 0          (0=water, 1=land)
      - "land_probability": keep points where sampled <= threshold (0..1 land probability)

    Backward compatibility:
      - If you pass water_val/invert without mask_type, the default "auto" will still work and
        will fall back to equality-based logic for non-probability masks.
      - NoData is always treated as invalid (dropped) in all modes.

    This function is intentionally conservative: on failure it returns the original df.
    """
    if df is None or df.empty:
        return df

    _rr_artifact(rr, "land_mask", str(mask_path))
    _rr_add(rr, "atl.mask_filter.input_n", int(len(df)))
    _rr_add(rr, "atl.mask_filter.mask_type", str(mask_type))
    _rr_add(rr, "atl.mask_filter.water_val", int(water_val))
    _rr_add(rr, "atl.mask_filter.invert", bool(invert))
    _rr_add(rr, "atl.mask_filter.threshold", float(threshold))

    _log_depth_funnel("atl.mask_filter.input", df, rr=rr)

    log.info(
        f"[ATL-MASK] Filtering {len(df)} points against mask: {Path(mask_path).name} "
        f"(mask_type={mask_type}, water_val={water_val}, invert={invert}, threshold={threshold})"
    )

    try:
        import rasterio  # heavy import (GDAL) - keep local
        from pyproj import Transformer
        with rasterio.open(mask_path) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = transformer.transform(df.longitude.values, df.latitude.values)
            coords = list(zip(xs, ys))
            sampled = np.array([v[0] for v in src.sample(coords)])

            # Start with finite/non-nodata only.
            valid = np.isfinite(sampled)
            # nodata=0 guard: waffles masks use 0 for water, so filtering sampled==0
            # would erase all water pixels. Only exclude nodata when it's a distinct
            # non-zero sentinel that can't collide with valid 0 (water) or 1 (land).
            if src.nodata is not None and np.isfinite(src.nodata) and abs(float(src.nodata)) > 0.5:
                valid &= (sampled != src.nodata)

            sampled_v = sampled[valid]
            if sampled_v.size == 0:
                log.warning("[ATL-MASK] All sampled mask values are nodata/invalid; returning empty DataFrame.")
                _rr_add(rr, "atl.mask_filter.dropped_n", int(len(df)))
                _rr_add(rr, "atl.mask_filter.output_n", 0)
                return df.iloc[0:0].copy()

            mtype = (mask_type or "auto").strip().lower()

            # Probability land mask: keep low land probability (water).
            if mtype in ("land_probability", "land_prob", "probability"):
                keep_v = sampled_v <= float(threshold)

            # Binary / equality masks:
            else:
                wv = int(water_val)

                if mtype == "auto":
                    # AUTO inference:
                    #   - If mask looks binary (values ~0/1), infer water_val by majority at sampled points.
                    #   - If mask looks like a 0..1 land-probability surface, treat it as land_probability.
                    sv = sampled_v
                    sv_min = float(np.nanmin(sv))
                    sv_max = float(np.nanmax(sv))

                    # "Binary-ish" if all values are very close to 0 or 1.
                    binish = bool(np.all((np.isclose(sv, 0.0, atol=1e-6)) | (np.isclose(sv, 1.0, atol=1e-6))))
                    if binish:
                        frac0 = float(np.mean(np.isclose(sv, 0.0, atol=1e-6)))
                        frac1 = 1.0 - frac0
                        wv = 0 if frac0 >= frac1 else 1
                        log.info(
                            f"[ATL-MASK] auto mask_type: inferred binary water_val={wv} "
                            f"from sampled fractions (0:{frac0:.3f}, 1:{frac1:.3f})."
                        )
                        _rr_add(rr, "atl.mask_filter.auto.frac0", frac0)
                        _rr_add(rr, "atl.mask_filter.auto.frac1", frac1)
                        _rr_add(rr, "atl.mask_filter.auto.inferred_water_val", int(wv))
                    else:
                        # If it is constrained to [0,1] but not binary-ish, assume "land probability".
                        if (sv_min >= -1e-6) and (sv_max <= 1.0 + 1e-6):
                            log.info(
                                f"[ATL-MASK] auto mask_type: detected 0..1 continuous mask "
                                f"(min={sv_min:.3f}, max={sv_max:.3f}); treating as land_probability "
                                f"(keep <= {float(threshold):.2f})."
                            )
                            mtype = "land_probability"
                        else:
                            # Non-binary, non-probability: fall back to provided water_val/invert equality.
                            log.info(
                                f"[ATL-MASK] auto mask_type: non-binary mask values detected "
                                f"(min={sv_min:.3f}, max={sv_max:.3f}); using water_val/invert equality rule."
                            )

                if mtype == "land_binary":
                    # Convention: 0=water, 1=land
                    keep_v = sampled_v == 0
                else:
                    # "water_only" (waffles) or "auto"/unknown: equality on inferred/provided water_val
                    if invert:
                        keep_v = sampled_v != wv
                    else:
                        keep_v = sampled_v == wv


            # Sanity rescue: if we kept nothing, try a safe fallback for common continuous masks.
            if int(np.count_nonzero(keep_v)) == 0 and sampled_v.size > 0:
                sv_min = float(np.nanmin(sampled_v))
                sv_max = float(np.nanmax(sampled_v))
                # If the mask looks like 0..1 and user didn't explicitly request a binary convention,
                # treat it as land probability (keep <= threshold) as a last resort.
                if (sv_min >= -1e-6) and (sv_max <= 1.0 + 1e-6) and (mtype not in ("land_probability", "land_prob", "probability")):
                    keep_v = sampled_v <= float(threshold)
                    log.warning(
                        f"[ATL-MASK] Kept 0 points with initial semantics; "
                        f"falling back to land_probability keep<=threshold ({float(threshold):.2f})."
                    )
                    _rr_add(rr, "atl.mask_filter.auto.fallback_land_probability", True)
# Map keep_v back onto full-length mask
            keep = np.zeros(len(df), dtype=bool)
            keep_idx = np.flatnonzero(valid)
            keep[keep_idx] = keep_v

            dropped_count = int(len(df) - np.count_nonzero(keep))
            if dropped_count > 0:
                log.info("[ATL-MASK] Dropped %s points masked out.", dropped_count)
            _rr_add(rr, "atl.mask_filter.dropped_n", dropped_count)
            _rr_add(rr, "atl.mask_filter.output_n", int(np.count_nonzero(keep)))

            out_df = df.loc[keep].reset_index(drop=True)
            _log_depth_funnel("atl.mask_filter.output", out_df, rr=rr)
            return out_df

    except Exception as e:
        log.warning("[ATL-MASK] Failed to filter points by mask: %s. Returning original points.", e)
        _rr_add(rr, "atl.mask_filter.failed", True)
        return df


# -----------------------------------------------------------------------------
# ICESat-2 Data Acquisition
# -----------------------------------------------------------------------------

def _iter_h5_recursive(d: Path):
    """Yield non-empty .h5 files under d (recursive)."""
    if not d.exists():
        return
    for f in d.rglob("*.h5"):
        try:
            if f.is_file() and f.stat().st_size > 0:
                yield f
        except Exception:
            continue

def existing_atl_files(d: Path, product_key: str) -> List[str]:
    """Return cached ICESat-2 product files under directory `d`.

    Harmony outputs are not fully consistent across time:
      - filenames may or may not include the substring 'subset'
      - zip payloads may contain nested folders

    We therefore search recursively and prefer product-matching filenames when possible.
    """
    if not d.exists():
        return []
    product_key = (product_key or "").upper().strip()

    files = sorted(_iter_h5_recursive(d), key=lambda x: x.name)

    if product_key in ("ATL03", "ATL24"):
        key = product_key
        matched = [f for f in files if key in f.name.upper()]
        files = matched if matched else files

    return [str(f) for f in files]

def _safe_unzip(z: Path, d: Path) -> List[str]:
    """Extract zip file `z` into directory `d` safely, flattening any internal paths.

    Returns absolute paths to extracted files.
    """
    out: List[str] = []
    d.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(z, "r") as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue

            base = Path(m.filename).name
            if not base:
                continue

            dest = d / base
            if dest.exists():
                stem = dest.stem
                suf = dest.suffix
                i = 1
                while True:
                    cand = d / f"{stem}_{i}{suf}"
                    if not cand.exists():
                        dest = cand
                        break
                    i += 1

            try:
                with zf.open(m, "r") as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if dest.exists() and dest.stat().st_size > 0:
                    out.append(str(dest))
            except Exception:
                continue

    return out

def _cmr_latest_concept_id(short_name: str) -> Optional[str]:

    try:
        url = "https://cmr.earthdata.nasa.gov/search/collections.json"
        params = {"short_name": short_name, "page_size": 2000, "sort_key": "-version"}
        r = _lazy_requests().get(url, params=params, timeout=30)
        r.raise_for_status()
        items = r.json().get("feed", {}).get("entry", [])
        return items[0]["id"] if items else None
    except Exception: return None

def cmr_search_atl24_full(bbox, start, end, page_size=200) -> List[dict]:
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    params = {"short_name": "ATL24", "version": "001", "temporal": f"{_iso(start)},{_iso(end, True)}",
              "bounding_box": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}", "page_size": str(page_size), "sort_key": "-start_date"}
    r = _lazy_requests().get(url, params=params, timeout=45)
    entries = r.json().get("feed", {}).get("entry", []) or []
    return entries

def harmony_subset(product_key, bbox, start, end, out_dir) -> List[str]:
    try: from harmony import Client as HarmonyClient, BBox, Request, Collection
    except ImportError: log.info("not installed."); return []
    out_dir.mkdir(parents=True, exist_ok=True)
    product_key = product_key.upper(); collection_id = product_key
    if product_key == "ATL03":
        cid = _cmr_latest_concept_id("ATL03")
        if cid: collection_id = cid
    elif product_key == "ATL24": collection_id = "C3433822507-NSIDC_CPRD"
    log.info("submitting %s as %s", product_key, collection_id)
    client = HarmonyClient()
    req = Request(collection=Collection(collection_id), spatial=BBox(*bbox),
                  temporal={"start": datetime.fromisoformat(f"{start}T00:00:00").replace(tzinfo=timezone.utc),
                            "stop": datetime.fromisoformat(f"{end}T23:59:59").replace(tzinfo=timezone.utc)})
    job = client.submit(req)
    t0 = time.time()
    while True:
        st = client.status(job).get("status", "unknown")
        if st in ("successful", "failed", "aborted"): break
        if time.time() - t0 > 900: break
        time.sleep(5)
    if client.status(job).get("status") != "successful": return []
    saved = []
    for fut in client.download_all(job, directory=str(out_dir), overwrite=False):
        try: saved.append(fut.result())
        except Exception: log.debug("ignored", exc_info=True)
    norm = []
    for p in saved:
        P = Path(p)
        if P.suffix.lower() == ".zip":
            norm.extend(_safe_unzip(P, out_dir)); P.unlink(missing_ok=True)
        else: norm.append(str(P))
    return norm

def ensure_icesat_files_harmony_cachefirst(d, product_key, bbox, start, end, force_redl=False, rr=None):
    # RunReport: acquisition/caching trace
    _rr_add(rr, f"atl.{product_key}.request.bbox", bbox)
    _rr_add(rr, f"atl.{product_key}.request.start", str(start))
    _rr_add(rr, f"atl.{product_key}.request.end", str(end))
    _rr_add(rr, f"atl.{product_key}.request.force_redl", bool(force_redl))
    d.mkdir(parents=True, exist_ok=True); entries = []
    cached = existing_atl_files(d, product_key)
    _rr_add(rr, f"atl.{product_key}.cache.files_n", int(len(cached)))
    _rr_add(rr, f"atl.{product_key}.cache.hit", bool(cached and not force_redl))
    if cached and not force_redl:
        if product_key == "ATL24":
            try: entries = cmr_search_atl24_full(bbox, start, end)
            except Exception: entries = []
        _rr_add(rr, f"atl.{product_key}.status", "cached")
        _rr_add(rr, f"atl.{product_key}.files_n", int(len(cached)))
        _rr_add(rr, f"atl.{product_key}.cmr_entries_n", int(len(entries) if entries else 0))
        return cached, entries, "cached"
    if product_key == "ATL24":
        try:
            entries = cmr_search_atl24_full(bbox, start, end)
            if not entries:
                _rr_add(rr, f"atl.{product_key}.status", "no_cmr_results")
                _rr_add(rr, f"atl.{product_key}.cmr_entries_n", 0)
                return cached, entries, "no_cmr_results"
        except Exception: log.debug("ignored", exc_info=True)
    files = harmony_subset(product_key, bbox, start, end, d)
    _rr_add(rr, f"atl.{product_key}.harmony.files_n", int(len(files) if files else 0))
    if files:
        for fp in files:
            _rr_artifact(rr, f"{product_key}_file", str(fp))
    status = "successful" if files else "failed"
    _rr_add(rr, f"atl.{product_key}.status", status)
    _rr_add(rr, f"atl.{product_key}.files_n", int(len(files) if files else 0))
    _rr_add(rr, f"atl.{product_key}.cmr_entries_n", int(len(entries) if entries else 0))
    if files: return files, entries, status
    return existing_atl_files(d, product_key), entries, status


# -----------------------------------------------------------------------------
# ATL03 Helpers
# -----------------------------------------------------------------------------

def read_atl03_basic(h5_file, laser_num="1"):
    with _lazy_h5py().File(h5_file, "r") as f:
        orientation = f["/orbit_info/sc_orient"][0]
        orientDict = {0: "l", 1: "r", 21: "l"}
        laser = f"gt{laser_num}{orientDict.get(orientation, 'l')}"
        lat = np.asarray(f[f"/{laser}/heights/lat_ph"][...])
        lon = np.asarray(f[f"/{laser}/heights/lon_ph"][...])
        h_ph = np.asarray(f[f"/{laser}/heights/h_ph"][...])
        conf = np.asarray(f[f"/{laser}/heights/signal_conf_ph"][..., 0])
        ref_elev = np.asarray(f[f"/{laser}/geolocation/ref_elev"][...])
        ref_azimuth = np.asarray(f[f"/{laser}/geolocation/ref_azimuth"][...])
        ph_index_beg = np.asarray(f[f"/{laser}/geolocation/ph_index_beg"][...])
        seg_ph_cnt = np.asarray(f[f"/{laser}/geolocation/segment_ph_cnt"][...])
        altitude_sc = np.asarray(f[f"/{laser}/geolocation/altitude_sc"][...])
    n = int(min(len(lat), len(lon), len(h_ph), len(conf)))
    lat = lat[:n]; lon = lon[:n]; h_ph = h_ph[:n]; conf = conf[:n]
    m = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h_ph) & np.isfinite(conf)
    return (lat[m], lon[m], h_ph[m], conf[m], ref_elev, ref_azimuth, ph_index_beg, seg_ph_cnt, altitude_sc)

def _n_water_index(temp_c=20.0, wavelength_nm=532.0):
    return float(-0.000001501562500 * temp_c**2 + 0.000000107084865 * wavelength_nm**2 - 0.000042759374989 * temp_c - 0.000160475520686 * wavelength_nm + 1.398067112092424)

def apply_hybrid_refraction(depth_app, theta_air_rad, temp_c=20.0, wavelength_nm=532.0):
    n_water = _n_water_index(temp_c, wavelength_nm); n_air = 1.00029
    sin_t1 = np.sin(theta_air_rad); sin_t2 = np.clip((n_air / n_water) * sin_t1, -1.0, 1.0)
    # CORRECTED: Z_true = Z_app * (1/n) * (cos(theta_water) / cos(theta_air))
    cos_t1 = np.cos(theta_air_rad)
    cos_t2 = np.where(np.cos(np.arcsin(sin_t2)) == 0, 1e-6, np.cos(np.arcsin(sin_t2)))
    return depth_app * (cos_t2 / cos_t1) / n_water

def bin_data_safe(df, lat_res, height_res, max_bins=5000):
    df = df[np.isfinite(df["latitude"]) & np.isfinite(df["photon_height"])]
    if df.empty: return None
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    h_min, h_max = df["photon_height"].min(), df["photon_height"].max()
    lat_bins = max(1, min(int(round(abs(lat_max - lat_min) / max(lat_res, 1e-9))), max_bins))
    h_bins = max(1, min(int(round(abs(h_max - h_min) / max(height_res, 1e-6))), max_bins))
    if lat_bins <= 1 or h_bins <= 1: return None
    df = df.copy()
    df["lat_bins"] = pd.cut(df["latitude"], lat_bins, labels=np.arange(lat_bins, dtype=int), include_lowest=True)
    df["height_bins"] = pd.cut(df["photon_height"], h_bins, labels=np.round(np.linspace(h_min, h_max, h_bins), 2), include_lowest=True)
    return df

def infer_surface_from_atl03_binned(binned, percentile=75.0, surface_window_m=5.0):
    if binned is None or binned.empty: return None
    cnt_all = binned.groupby(["lat_bins", "height_bins"], observed=False).size().reset_index()
    if cnt_all.empty: return None
    max_per_lat = cnt_all.groupby("lat_bins", observed=False)[0].max()
    if len(max_per_lat) == 0: return None
    count_threshold = max(2, np.percentile(max_per_lat, percentile))
    global_h = cnt_all.loc[cnt_all[0].idxmax(), "height_bins"]
    surf_rows = []
    for _, grp in binned.groupby("lat_bins", observed=False):
        if grp.empty: continue
        cnt = grp.groupby("height_bins", observed=False).size().reset_index()
        cnt_f = cnt[(cnt[0] >= count_threshold) & (np.abs(cnt["height_bins"].astype(float) - float(global_h)) <= surface_window_m)]
        if cnt_f.empty: continue
        h_bin = cnt_f.loc[cnt_f[0].idxmax(), "height_bins"]
        sel = grp[grp["height_bins"] == h_bin]
        surf_rows.append((sel["latitude"].median(), sel["longitude"].median(), sel["photon_height"].median()))
    if not surf_rows: return None
    return pd.DataFrame(surf_rows, columns=["latitude", "longitude", "surface_h"]).dropna()

def infer_bottom_from_atl03_binned(binned, ws_height_df, height_res=0.25, percentile=90.0, min_bottom_photons_local=2):
    if binned is None or ws_height_df is None or ws_height_df.empty: return None
    surf = ws_height_df.sort_values("latitude"); binned = binned.copy(); bath_rows = []
    surf_lat = surf["latitude"].values; surf_h = surf["surface_h"].values
    for _, v in binned.groupby(["lat_bins"], observed=False):
        if v.empty: continue
        lat_mid = v["latitude"].median()
        if not np.isfinite(lat_mid): continue

        # Local surface fallback
        cnt_full = v.groupby("height_bins", observed=False).size().reset_index(name="count")
        surf_h_local = float(cnt_full.loc[cnt_full["count"].idxmax(), "height_bins"])

        idx_nn = (np.abs(surf_lat - lat_mid)).argmin()
        ws_h_global = float(surf_h[idx_nn])
        ws_h = ws_h_global if abs(ws_h_global - surf_h_local) <= (2.0 * height_res) else surf_h_local

        v_sub = v[v["photon_height"] < ws_h - (height_res * 2.0)]
        if v_sub.empty: continue

        new_df = pd.DataFrame(v_sub.groupby("height_bins", observed=False).count())
        if new_df.empty: continue
        bath_bin = new_df["latitude"].argmax()
        n_bottom = int(new_df.iloc[bath_bin]["latitude"])
        if n_bottom < min_bottom_photons_local: continue

        pts = v_sub[v_sub["height_bins"] == new_df.index[bath_bin]][["longitude", "latitude", "photon_height"]].copy()
        pts["depth_app"] = (ws_h - pts["photon_height"]).astype(np.float32)
        pts["ws_h"] = ws_h; pts["n_bottom"] = n_bottom; pts["n_subsurface"] = len(v_sub)
        pts["frac_bottom"] = float(n_bottom) / max(len(v_sub), 1)
        bath_rows.append(pts)
    if not bath_rows: return None
    return pd.concat(bath_rows, ignore_index=True)


# -----------------------------------------------------------------------------
# ATL24 Helper
# -----------------------------------------------------------------------------

def _collect_points_from_atl24_file(h5_path, conf_min, segment_length_m=5.0):
    """
    Collect bathymetry points from ATL24 file.

    ATL24 stores photon-level data, but we need segment-level aggregates
    to match ATL03's binned approach. This function:
    1. Reads bathymetry-classified photons (class_ph == 40)
    2. Groups them into along-track segments (~5m, matching ATL03 lat_res)
    3. Returns median position and depth per segment
    4. Captures delta_time for tidal correction

    This ensures ATL24 points can be properly collocated with ATL03 points
    during fusion (they'll have similar spatial density and positions).
    """
    all_segments = []

    with _lazy_h5py().File(h5_path, "r") as f:
        # Try to get granule-level time metadata
        granule_start_delta_time = None
        try:
            if "ancillary_data" in f and "atlas_sdp_gps_epoch" in f["ancillary_data"]:
                pass  # We'll use delta_time directly
            if "orbit_info" in f and "sc_orient_time" in f["orbit_info"]:
                granule_start_delta_time = f["orbit_info"]["sc_orient_time"][0]
        except Exception:
            log.debug("ignored", exc_info=True)

        for beam_key in [k for k in f.keys() if k.startswith("gt")]:
            g = f.get(beam_key)
            if g is None:
                continue

            # Check for required datasets
            required = {"lat_ph", "lon_ph", "ortho_h", "surface_h", "class_ph"}
            if not required.issubset(g.keys()):
                continue

            # Read photon-level data
            lat = g["lat_ph"][:]
            lon = g["lon_ph"][:]
            ortho_h = g["ortho_h"][:]      # Bottom elevation
            surface_h = g["surface_h"][:]  # Surface elevation
            class_ph = g["class_ph"][:]
            conf = g["confidence"][:] if "confidence" in g else None

            # Try to get delta_time for tidal correction
            delta_time = g["delta_time"][:] if "delta_time" in g else None

            # Ensure same length
            n = min(len(lat), len(lon), len(ortho_h), len(surface_h), len(class_ph))
            lat = lat[:n]
            lon = lon[:n]
            ortho_h = ortho_h[:n]
            surface_h = surface_h[:n]
            class_ph = class_ph[:n]
            if conf is not None:
                conf = conf[:n]
            if delta_time is not None:
                delta_time = delta_time[:n]

            # Filter to bathymetry photons (class 40) with valid data
            mask = (
                (class_ph == 40) &
                np.isfinite(lat) &
                np.isfinite(lon) &
                np.isfinite(ortho_h) &
                np.isfinite(surface_h)
            )
            if conf is not None:
                mask &= (conf >= conf_min)

            if not np.any(mask):
                continue

            # Extract valid photons
            lat_v = lat[mask]
            lon_v = lon[mask]
            # Compute depth and enforce negative-down convention (depth below surface = negative)
            depth_v = -np.abs(surface_h[mask] - ortho_h[mask])  # Negative depth
            conf_v = conf[mask] if conf is not None else np.ones(np.sum(mask))
            surface_h_v = surface_h[mask]
            ortho_h_v = ortho_h[mask]
            delta_time_v = delta_time[mask] if delta_time is not None else None

            # Group into ~segment_length_m-meter along-track segments.
            # Bin by along-track order, not latitude: tracks are not necessarily N-S.
            # Use delta_time when available, otherwise a PCA axis in local UTM.

            # Project to a local UTM zone for meter-based segmentation
            try:
                from pyproj import CRS, Transformer
                lon0 = float(np.median(lon_v))
                lat0 = float(np.median(lat_v))
                zone = int((lon0 + 180.0) // 6) + 1
                epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
                crs_utm = CRS.from_epsg(epsg)
                txy = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
                x_v, y_v = txy.transform(lon_v, lat_v)
                x_v = np.asarray(x_v, dtype="float64")
                y_v = np.asarray(y_v, dtype="float64")
            except Exception:
                # Fallback: approximate meters using degrees (coarse)
                x_v = lon_v.astype("float64") * 111320.0 * np.cos(np.deg2rad(np.median(lat_v)))
                y_v = lat_v.astype("float64") * 111320.0

            # Determine along-track ordering
            if delta_time_v is not None and np.isfinite(delta_time_v).any():
                order = np.argsort(delta_time_v)
            else:
                # PCA axis ordering
                xy = np.column_stack([x_v, y_v])
                xy0 = xy - np.nanmean(xy, axis=0)
                try:
                    c = np.cov(xy0.T)
                    w, v = np.linalg.eigh(c)
                    axis = v[:, int(np.argmax(w))]
                    t = xy0 @ axis
                    order = np.argsort(t)
                except Exception:
                    order = np.argsort(x_v)

            x_o = x_v[order]
            y_o = y_v[order]
            lat_o = lat_v[order]
            lon_o = lon_v[order]
            depth_o = depth_v[order]
            conf_o = conf_v[order]
            surf_o = surface_h_v[order]
            ortho_o = ortho_h_v[order]
            dt_o = delta_time_v[order] if delta_time_v is not None else None

            # Cumulative along-track distance
            dx = np.diff(x_o)
            dy = np.diff(y_o)
            ds = np.sqrt(dx*dx + dy*dy)
            cum = np.concatenate([[0.0], np.cumsum(ds)])

            seg_id = np.floor(cum / max(float(segment_length_m), 0.1)).astype(int)
            n_segs = int(seg_id.max()) + 1 if seg_id.size else 0

            for b in range(n_segs):
                mseg = (seg_id == b)
                if int(np.count_nonzero(mseg)) < 2:
                    continue

                seg_lat = float(np.median(lat_o[mseg]))
                seg_lon = float(np.median(lon_o[mseg]))
                seg_depth = float(np.median(depth_o[mseg]))
                seg_conf = float(np.median(conf_o[mseg]))
                seg_n_photons = int(np.count_nonzero(mseg))
                seg_surface_h = float(np.median(surf_o[mseg]))
                seg_ortho_h = float(np.median(ortho_o[mseg]))

                seg_delta_time = None
                if dt_o is not None:
                    try:
                        seg_delta_time = float(np.median(dt_o[mseg]))
                    except Exception:
                        seg_delta_time = None

                all_segments.append({
                    "latitude": seg_lat,
                    "longitude": seg_lon,
                    "depth_m": seg_depth,
                    "conf": seg_conf,
                    "n_photons": seg_n_photons,
                    "surface_h": seg_surface_h,
                    "ortho_h": seg_ortho_h,
                    "beam": beam_key,
                    "delta_time": seg_delta_time,
                })

    if not all_segments:
        return pd.DataFrame(columns=["latitude", "longitude", "depth_m", "conf", "source", "delta_time"])

    df = pd.DataFrame(all_segments)
    df["source"] = "atl24"

    log.info("Collected %s segment-aggregated points from %s", len(df), Path(h5_path).name)

    return df


# -----------------------------------------------------------------------------
# ICESat‑2 Main Data Collection Functions
# -----------------------------------------------------------------------------


def _fingerprint_path(p: Optional[str], cache_strict: bool = True) -> Optional[str]:
    """Stable fingerprint for a file path, or a string fallback on error."""
    if not p:
        return None
    try:
        pp = Path(p)
        if _CACHE_UTILS_AVAILABLE:
            return fingerprint_file(pp, strict=bool(cache_strict))
        st = pp.stat()
        return f"{pp.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        return str(p)



def collect_training_points_from_atl03(
    atl03_files: List[str], lat_res: float, height_res: float, aoi_str: str,
    atl03_conf_min: int, atl03_bottom_percentile: float, use_refraction: bool,
    default_temp_c: float, default_wavelength_nm: float, min_bottom_photons: int,
    min_bottom_frac: float, min_depth_m: float, max_depth_m: float, debug_atl03_qc: bool,
    land_mask_path: Optional[str] = None,
    land_mask_water_val: int = 0,
    land_mask_invert: bool = False,
    land_mask_type: str = "auto",
    land_mask_threshold: float = 0.5,
    cache_dir: Optional[str] = None,
    cache_strict: bool = False,
    cache_code_strict: bool = False,
    cache_ignore_code: bool = True,
    rr=None,
) -> pd.DataFrame:
    W, E, S, N = [float(x) for x in aoi_str.split("/")]
    all_rows = []


    # ---------------------------------------------------------------------
    # Exact-match cache (derived training points)
    # ---------------------------------------------------------------------
    params_fp = {
        "aoi_str": aoi_str,
        "lat_res": float(lat_res),
        "height_res": float(height_res),
        "atl03_conf_min": int(atl03_conf_min),
        "atl03_bottom_percentile": float(atl03_bottom_percentile),
        "use_refraction": bool(use_refraction),
        "default_temp_c": float(default_temp_c),
        "default_wavelength_nm": float(default_wavelength_nm),
        "min_bottom_photons": int(min_bottom_photons),
        "min_bottom_frac": float(min_bottom_frac),
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
        "debug_atl03_qc": bool(debug_atl03_qc),
        "land_mask_path": str(land_mask_path) if land_mask_path else None,
        "land_mask_water_val": int(land_mask_water_val),
        "land_mask_invert": bool(land_mask_invert),
        "land_mask_type": str(land_mask_type),
        "land_mask_threshold": float(land_mask_threshold),
    }


    inputs_fp = {
        "atl03_files": [_fingerprint_path(p) for p in (atl03_files or [])],
        "land_mask": _fingerprint_path(land_mask_path) if land_mask_path else None,
    }

    df_cached, cache_reason, cache_key, cache_data_path, cache_meta_path = _maybe_load_cached_training_points(
        stage="atl03_training_points",
        cache_dir=cache_dir,
        params=params_fp,
        inputs=inputs_fp,
        cache_strict=bool(cache_strict),
        cache_code_strict=bool(cache_code_strict),
        cache_ignore_code=bool(cache_ignore_code),
        rr=rr,
    )
    if isinstance(df_cached, pd.DataFrame):
        if len(df_cached) == 0:
            log.warning(
                "[ATL03_TRAINING_POINTS-CACHE] HIT but cached result is EMPTY (0 rows). "
                "If this is unexpected after a code or mask fix, delete the cache dir or pass "
                "--no-cache-ignore-code to force reprocessing. Cache: %s",
                Path(cache_data_path).name if cache_data_path else cache_key,
            )
        else:
            log.info(f"[ATL03_TRAINING_POINTS-CACHE] HIT: {Path(cache_data_path).name if cache_data_path else cache_key} ({cache_reason}, n={len(df_cached)})")
        return df_cached.reset_index(drop=True)
    else:
        if cache_dir is not None and _CACHE_UTILS_AVAILABLE:
            log.info("[ATL03_TRAINING_POINTS-CACHE] MISS: %s", cache_reason)


    for atl03_path in atl03_files:
        try:
            for laser_num in ("1", "2", "3"):
                try:
                    (lat, lon, h_ph, conf, ref_elev, _, _, _, _) = read_atl03_basic(atl03_path, laser_num)
                except Exception: continue

                m_geo = (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h_ph) & np.isfinite(conf))
                lat = lat[m_geo]; lon = lon[m_geo]; h_ph = h_ph[m_geo]; conf = conf[m_geo]

                m_aoi = (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
                lat = lat[m_aoi]; lon = lon[m_aoi]; h_ph = h_ph[m_aoi]; conf = conf[m_aoi]
                if lat.size == 0: continue

                m_conf = conf >= atl03_conf_min
                df_beam = pd.DataFrame({"latitude": lat[m_conf], "longitude": lon[m_conf], "photon_height": h_ph[m_conf]}).dropna()
                if df_beam.empty: continue

                theta_air_med = None
                if use_refraction and ref_elev.size:
                    theta_all = (np.pi / 2.0) - ref_elev.astype(np.float64); theta_all = theta_all[np.isfinite(theta_all)]
                    if theta_all.size: theta_air_med = float(np.nanmedian(theta_all))

                binned = bin_data_safe(df_beam, lat_res, height_res)
                surf_df = infer_surface_from_atl03_binned(binned)
                bath_df = infer_bottom_from_atl03_binned(binned, surf_df, height_res, atl03_bottom_percentile)
                if bath_df is None or bath_df.empty: continue

                if use_refraction and theta_air_med and np.isfinite(theta_air_med):
                    bath_df["depth_m"] = apply_hybrid_refraction(bath_df["depth_app"].values, theta_air_med, default_temp_c, default_wavelength_nm)
                else:
                    bath_df["depth_m"] = bath_df["depth_app"]

                # Enforce negative-down convention (depth below surface = negative)
                bath_df["depth_m"] = -np.abs(bath_df["depth_m"])

                bath_df = bath_df[(bath_df["depth_m"] <= -min_depth_m) & (bath_df["depth_m"] >= -max_depth_m)]
                try:
                    if len(bath_df) > 0 and "depth_m" in bath_df.columns:
                        d = bath_df["depth_m"].astype(float).to_numpy()
                        d = d[np.isfinite(d)]
                        if d.size > 0:
                            shallow_floor = -float(min_depth_m)  # negative-down convention
                            frac_near_floor = float(np.mean(np.abs(d - shallow_floor) <= 0.20))
                            p50 = float(np.nanpercentile(d, 50))
                            p95 = float(np.nanpercentile(d, 95))
                            if frac_near_floor >= 0.60 or (p95 - p50) <= 0.25:
                                log.warning(
                                    "[ATL03_QC] Depths cluster near shallow floor after ATL03 filtering (floor=%.2f m, p50=%.2f, p95=%.2f, near_floor<=0.20m=%.1f%%, n=%d). Check min-depth-atl03 / bottom-pick thresholds / water classification.",
                                    shallow_floor, p50, p95, 100.0 * frac_near_floor, int(d.size)
                                )
                except Exception:
                    log.debug("ATL03 shallow-floor QC warning failed", exc_info=True)
                bath_df = bath_df[(bath_df["n_bottom"] >= min_bottom_photons) & (bath_df["frac_bottom"] >= min_bottom_frac)]

                bath_df["granule"] = Path(atl03_path).stem; bath_df["beam"] = f"gt{laser_num}"; bath_df["source"] = "atl03"

                all_rows.append(bath_df[["longitude", "latitude", "depth_m", "ws_h", "photon_height", "n_bottom", "n_subsurface", "frac_bottom", "granule", "beam", "source"]])

        except Exception as exc:
            log.warning("[TRAIN-ATL03] failed to parse %s: %s", Path(atl03_path).name, exc)

    # If we have no valid rows, still write an empty cache entry.
    # This avoids repeated expensive parsing work across identical runs.
    if not all_rows:
        out_empty = pd.DataFrame()
        if cache_dir is not None and _CACHE_UTILS_AVAILABLE and cache_key and cache_data_path and cache_meta_path:
            log.info("[ATL03_TRAINING_POINTS-CACHE] WRITE: %s (empty)", Path(cache_data_path).name)
            _write_cached_training_points(
                stage="atl03_training_points",
                key=cache_key,
                data_path=Path(cache_data_path),
                meta_path=Path(cache_meta_path),
                params=params_fp,
                inputs=inputs_fp,
                cache_code_strict=bool(cache_code_strict),
                df=out_empty,
                extra={"rows": 0, "empty": True},
            )
        return out_empty
    out = pd.concat(all_rows, ignore_index=True)
    _log_depth_funnel("atl03.points.concat", out, rr=rr)

    # Filter by Land Mask
    if land_mask_path and os.path.exists(land_mask_path):
        out = _filter_points_by_mask(out, land_mask_path, water_val=land_mask_water_val, invert=land_mask_invert, mask_type=land_mask_type, threshold=land_mask_threshold, rr=rr)
        _log_depth_funnel("atl03.points.landmask", out, rr=rr)

    # Cache write (best-effort)
    if cache_dir is not None and _CACHE_UTILS_AVAILABLE and cache_key and cache_data_path and cache_meta_path:
        log.info("[ATL03_TRAINING_POINTS-CACHE] WRITE: %s", Path(cache_data_path).name)
        _write_cached_training_points(
            stage="atl03_training_points",
            key=cache_key,
            data_path=Path(cache_data_path),
            meta_path=Path(cache_meta_path),
            params=params_fp,
            inputs=inputs_fp,
            cache_code_strict=bool(cache_code_strict),
            df=out.reset_index(drop=True),
            extra={"rows": int(len(out)), "empty": bool(len(out) == 0)},
        )

    return out.reset_index(drop=True)


def collect_training_points_from_atl24(
    atl24_files: List[str], aoi_str: str, max_depth_m: float, train_relax_buffer: float,
    limit_train_samples: Optional[int], seed: int, atl24_conf_min: float = ATL24_CONFIDENCE_DEFAULT,
    land_mask_path: Optional[str] = None,
    land_mask_water_val: int = 0,
    land_mask_invert: bool = False,
    land_mask_type: str = "auto",
    land_mask_threshold: float = 0.5,
    cache_dir: Optional[str] = None,
    cache_strict: bool = False,
    cache_code_strict: bool = False,
    cache_ignore_code: bool = True,
    rr=None,
) -> pd.DataFrame:
    """Core ATL24 segment parsing logic."""


    # ---------------------------------------------------------------------
    # Exact-match cache (derived training points)
    # ---------------------------------------------------------------------
    params_fp = {
        "aoi_str": aoi_str,
        "max_depth_m": float(max_depth_m),
        "train_relax_buffer": float(train_relax_buffer),
        "limit_train_samples": int(limit_train_samples) if limit_train_samples is not None else None,
        "seed": int(seed),
        "atl24_conf_min": float(atl24_conf_min),
        "land_mask_path": str(land_mask_path) if land_mask_path else None,
        "land_mask_water_val": int(land_mask_water_val),
        "land_mask_invert": bool(land_mask_invert),
        "land_mask_type": str(land_mask_type),
        "land_mask_threshold": float(land_mask_threshold),
    }


    inputs_fp = {
        "atl24_files": [_fingerprint_path(p) for p in (atl24_files or [])],
        "land_mask": _fingerprint_path(land_mask_path) if land_mask_path else None,
    }

    df_cached, cache_reason, cache_key, cache_data_path, cache_meta_path = _maybe_load_cached_training_points(
        stage="atl24_training_points",
        cache_dir=cache_dir,
        params=params_fp,
        inputs=inputs_fp,
        cache_strict=bool(cache_strict),
        cache_code_strict=bool(cache_code_strict),
        cache_ignore_code=bool(cache_ignore_code),
        rr=rr,
    )
    if isinstance(df_cached, pd.DataFrame):
        if len(df_cached) == 0:
            log.warning(
                "[ATL24_TRAINING_POINTS-CACHE] HIT but cached result is EMPTY (0 rows). "
                "If this is unexpected after a code or mask fix, delete the cache dir or pass "
                "--no-cache-ignore-code to force reprocessing. Cache: %s",
                Path(cache_data_path).name if cache_data_path else cache_key,
            )
        else:
            log.info(f"[ATL24_TRAINING_POINTS-CACHE] HIT: {Path(cache_data_path).name if cache_data_path else cache_key} ({cache_reason}, n={len(df_cached)})")
        return df_cached.reset_index(drop=True)
    else:
        if cache_dir is not None and _CACHE_UTILS_AVAILABLE:
            log.info("[ATL24_TRAINING_POINTS-CACHE] MISS: %s", cache_reason)

    pts = []
    for h5 in atl24_files or []:
        try:
            df = _collect_points_from_atl24_file(h5, conf_min=atl24_conf_min)
            if not df.empty: pts.append(df)
        except Exception: log.debug("ignored", exc_info=True)

    if not pts: return pd.DataFrame()
    df_all = pd.concat(pts).reset_index(drop=True)
    _log_depth_funnel("atl24.points.concat", df_all, rr=rr)
    W, E, S, N = parse_aoi_string(aoi_str)
    train_df = df_all[(df_all.longitude >= W) & (df_all.longitude <= E) & (df_all.latitude >= S) & (df_all.latitude <= N)]
    _log_depth_funnel("atl24.points.aoi_clip", train_df, rr=rr)

    if train_df.empty and train_relax_buffer > 0:
        bw = (E-W)*train_relax_buffer; bh = (N-S)*train_relax_buffer
        train_df = df_all[(df_all.longitude >= W-bw) & (df_all.longitude <= E+bw) & (df_all.latitude >= S-bh) & (df_all.latitude <= N+bh)]
        _log_depth_funnel("atl24.points.relax_clip", train_df, rr=rr)

    if train_df.empty: return pd.DataFrame()
    _log_depth_funnel("atl24.points.pre_depth_cap", train_df, rr=rr)
    # Filter by depth (negative-down convention: -max_depth_m <= depth_m <= -min_depth)
    # depth_m is negative, so we filter: depth >= -max_depth_m (i.e., not deeper than max)
    train_df = train_df[train_df["depth_m"] >= -max_depth_m]
    _log_depth_funnel("atl24.points.post_depth_cap", train_df, rr=rr)

    if limit_train_samples and len(train_df) > limit_train_samples:
        train_df = train_df.sample(limit_train_samples, random_state=seed)
        _log_depth_funnel("atl24.points.sample_limit", train_df, rr=rr)

    # Filter by Land Mask
    if land_mask_path and os.path.exists(land_mask_path):
        train_df = _filter_points_by_mask(train_df, land_mask_path, water_val=land_mask_water_val, invert=land_mask_invert, mask_type=land_mask_type, threshold=land_mask_threshold, rr=rr)
        _log_depth_funnel("atl24.points.landmask", train_df, rr=rr)

    # Cache write (best-effort)
    if cache_dir is not None and _CACHE_UTILS_AVAILABLE and cache_key and cache_data_path and cache_meta_path:
        log.info("[ATL24_TRAINING_POINTS-CACHE] WRITE: %s", Path(cache_data_path).name)
        _write_cached_training_points(
            stage="atl24_training_points",
            key=cache_key,
            data_path=Path(cache_data_path),
            meta_path=Path(cache_meta_path),
            params=params_fp,
            inputs=inputs_fp,
            cache_code_strict=bool(cache_code_strict),
            df=train_df.reset_index(drop=True),
            extra={"rows": int(len(train_df))},
        )

    return train_df.reset_index(drop=True)

def build_gl_atl24_like_product(
    atl03_files: List[str], aoi_str: str, out_path: str,
    atl03_conf_min: int = 4, atl03_bottom_percentile: float = 90.0
) -> None:
    # This function allows users to "export" their ATL03 picks to a CSV
    # matching the format of the training data
    df = collect_training_points_from_atl03(
        atl03_files, lat_res=0.00005, height_res=0.25, aoi_str=aoi_str,
        atl03_conf_min=atl03_conf_min, atl03_bottom_percentile=atl03_bottom_percentile,
        use_refraction=True, default_temp_c=20.0, default_wavelength_nm=532.0,
        min_bottom_photons=3, min_bottom_frac=0.1, min_depth_m=0.1, max_depth_m=40.0,
        debug_atl03_qc=False
    )
    if not df.empty:
        df.to_csv(out_path, index=False)
        log.info("Exported %s ATL03 picks to %s", len(df), out_path)
    else:
        log.warning("No ATL03 picks found to export.")

def build_atl03_track_lines(atl03_files: List[str], aoi_str: str, out_shp: str):
    # Heavy geospatial deps are local to keep module import light.
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except Exception as e:
        raise RuntimeError("build_atl03_track_lines requires geopandas + shapely") from e

    # This visualizes where the tracks are
    W, E, S, N = [float(x) for x in aoi_str.split("/")]
    lines, names, beams = [], [], []
    for f in atl03_files:
        try:
            for laser in ("1", "2", "3"):
                # Just read enough to get a line
                with _lazy_h5py().File(f, "r") as h5:
                    # Quick orientation check
                    orientation = h5["/orbit_info/sc_orient"][0]
                    orientDict = {0: "l", 1: "r", 21: "l"}
                    beam_key = f"gt{laser}{orientDict.get(orientation, 'l')}"

                    lats = h5[f"/{beam_key}/heights/lat_ph"][::100] # Subsample heavily
                    lons = h5[f"/{beam_key}/heights/lon_ph"][::100]

                # Clip
                m = (lons >= W) & (lons <= E) & (lats >= S) & (lats <= N)
                if not np.any(m): continue

                # Make line
                pts = list(zip(lons[m], lats[m]))
                if len(pts) > 1:
                    lines.append(LineString(pts))
                    names.append(Path(f).name)
                    beams.append(f"gt{laser}")
        except Exception: log.debug("ignored", exc_info=True)

    if lines:
        gdf = gpd.GeoDataFrame({"granule": names, "beam": beams, "geometry": lines}, crs="EPSG:4326")
        gdf.to_file(out_shp)

def load_extra_xyz_points(xyz_files: List[str], crs: str, aoi_str: str) -> pd.DataFrame:
    """
    Load external XYZ bathymetry data from multiple files.

    Supports CSV, TXT, and GPKG formats. Reprojects from ``crs`` to EPSG:4326
    and clips to ``aoi_str`` (W/E/S/N). All rows are labelled source="extra_xyz".

    Args:
        xyz_files: List of file paths
        crs: Input CRS (e.g., "EPSG:4326", "EPSG:32617")
        aoi_str: AOI string "W/E/S/N" in EPSG:4326

    Returns:
        DataFrame with columns: longitude, latitude, depth_m, source
    """
    from pyproj import Transformer, CRS
    import geopandas as gpd
    import re

    dfs = []
    W, E, S, N = [float(x) for x in aoi_str.split("/")]

    log.info("Loading XYZ data from %s file(s)", len(xyz_files))
    log.info("Target AOI: W=%.4f, E=%.4f, S=%.4f, N=%.4f", W, E, S, N)
    log.info("Input CRS: %s", crs)

    # Setup coordinate transformation
    transformer = None
    if crs.upper() != "EPSG:4326":
        try:
            # Create transformer from input CRS to WGS84
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            log.info("Created CRS transformer: %s → EPSG:4326", crs)
        except Exception as e:
            log.error("Failed to create CRS transformer: %s", e, exc_info=True)
            log.error("Assuming data is already in EPSG:4326")
            transformer = None

    for i, f in enumerate(xyz_files):
        try:
            filepath = Path(f)
            log.info("Processing file %s/%s: %s", i+1, len(xyz_files), filepath.name)

            # Try different file formats
            tmp = None

            # Try GeoPackage/Shapefile first
            if filepath.suffix.lower() in ['.gpkg', '.shp', '.geojson']:
                try:
                    gdf = gpd.read_file(f)
                    # Extract coordinates
                    tmp = pd.DataFrame({
                        'longitude': gdf.geometry.x,
                        'latitude': gdf.geometry.y
                    })
                    # Look for depth column
                    depth_cols = [c for c in gdf.columns if 'depth' in c.lower() or 'z' in c.lower()]
                    if depth_cols:
                        tmp['depth_m'] = gdf[depth_cols[0]]
                    log.info("Loaded as vector file: %s points", len(tmp))
                except Exception as e:
                    log.debug("Not a vector file: %s", e)

            # Try CSV/TXT
            if tmp is None:
                tmp = pd.read_csv(f)

                # Try whitespace delimiter if comma fails
                if len(tmp.columns) == 1:
                    tmp = pd.read_csv(f, sep=r'\s+', header=None,
                                     names=['x', 'y', 'z'])

                log.info("Loaded as CSV/TXT: %s points, %s columns", len(tmp), len(tmp.columns))

            # Normalize column names
            tmp.columns = [c.lower() for c in tmp.columns]

            # Map columns to standard names with case-insensitive matching
            rename_map = {}
            for c in tmp.columns:
                cl = str(c).lower().strip()

                # Exact matches first (most reliable)
                if cl in ('lon', 'longitude', 'x'):
                    rename_map[c] = 'x_orig'
                elif cl in ('lat', 'latitude', 'y'):
                    rename_map[c] = 'y_orig'
                elif cl in ('z', 'depth', 'depth_m', 'z_m', 'bathy', 'bathymetry'):
                    rename_map[c] = 'depth_m'
                # Substring matching as fallback
                elif 'lon' in cl and 'x_orig' not in rename_map.values():
                    rename_map[c] = 'x_orig'
                elif 'lat' in cl and 'y_orig' not in rename_map.values():
                    rename_map[c] = 'y_orig'
                elif ('depth' in cl or 'bathy' in cl) and 'depth_m' not in rename_map.values():
                    rename_map[c] = 'depth_m'

            tmp = tmp.rename(columns=rename_map)

            # Check required columns
            if 'x_orig' not in tmp.columns or 'y_orig' not in tmp.columns:
                log.warning("Skipping %s: missing x/y columns", filepath.name)
                log.warning("Available columns: %s", list(tmp.columns))
                continue

            if 'depth_m' not in tmp.columns:
                # Safety: do not silently treat elevations as depths. If the file appears to contain
                # elevation/height (e.g., NAVD88 orthometric heights), you must convert to *depth below surface*
                # before using it as training data for a depth model.
                elev_like = any("elev" in str(c).lower() or "height" in str(c).lower() for c in tmp.columns)
                if elev_like:
                    log.warning("Skipping %s: appears to contain elevation/height but no depth column.", filepath.name)
                    log.warning("For depth modeling, supply a depth column (below-surface) or pre-convert elevations to depth.")
                else:
                    log.warning("Skipping %s: missing depth column", filepath.name)
                log.warning("Available columns: %s", list(tmp.columns))
                continue


            # Coerce depth_m to numeric early (protect against object/string depths)
            # This prevents downstream np.isfinite / sign logic failures.
            tmp['depth_m'] = pd.to_numeric(tmp['depth_m'], errors='coerce')
            n_bad = int(tmp['depth_m'].isna().sum())
            if n_bad:
                log.warning("Dropping %s rows with non-numeric depth_m in %s", n_bad, filepath.name)
                tmp = tmp.loc[tmp['depth_m'].notna()].copy()
            if len(tmp) == 0:
                log.warning("No valid depth rows remain after coercion in %s", filepath.name)
                continue

            # Apply CRS transformation if needed
            if transformer is not None:
                log.info("Transforming coordinates...")
                x_in = tmp['x_orig'].values
                y_in = tmp['y_orig'].values

                # Transform
                lon, lat = transformer.transform(x_in, y_in)
                tmp['longitude'] = lon
                tmp['latitude'] = lat

                # Sanity check: log range before/after
                log.info(f"Input range: X=[{x_in.min():.2f}, {x_in.max():.2f}], "
                        f"Y=[{y_in.min():.2f}, {y_in.max():.2f}]")
                log.info(f"Output range: Lon=[{lon.min():.4f}, {lon.max():.4f}], "
                        f"Lat=[{lat.min():.4f}, {lat.max():.4f}]")
            else:
                # No transformation needed
                tmp['longitude'] = tmp['x_orig']
                tmp['latitude'] = tmp['y_orig']

            # Ensure depths are negative (below surface)
            depth_vals = tmp['depth_m'].values
            finite_depths = depth_vals[np.isfinite(depth_vals)]

            if len(finite_depths) == 0:
                log.warning("No finite depth values in %s", filepath.name)
                continue

            pct_positive = (finite_depths > 0).sum() / len(finite_depths)
            depth_min, depth_max = finite_depths.min(), finite_depths.max()

            log.info(f"Depth statistics: "
                    f"min={depth_min:.2f}, max={depth_max:.2f}, "
                    f"pct_positive={pct_positive*100:.1f}%")

            # Infer depth sign from distribution and convert to negative-down
            if pct_positive >= 0.9:
                # Mostly positive -> assume depths below surface, convert to negative
                log.info("Converting depths to negative (below surface)")
                tmp['depth_m'] = -np.abs(tmp['depth_m'])
            elif pct_positive <= 0.1:
                # Mostly negative -> already in correct convention
                log.info("Depths already negative (correct convention)")
            else:
                # Mixed signs -> ambiguous, warn user
                log.warning(
                    f"[load_extra_xyz]   Mixed depth signs detected ({pct_positive*100:.1f}% positive). "
                    f"Not auto-converting. Please verify depth convention or add --xyz-depth-convention flag."
                )
                log.warning("Depth range: [%.2f, %.2f] m", depth_min, depth_max)

            # MAD-based outlier guard before high-weight ingestion
            depth_vals_clean = tmp['depth_m'].values
            depth_vals_clean = depth_vals_clean[np.isfinite(depth_vals_clean)]

            if len(depth_vals_clean) > 0:
                # MAD-based outlier detection (robust to extreme values)
                median_depth = np.median(depth_vals_clean)
                # Robust MAD without scipy (1.4826 scales MAD to ~sigma for normal)
                mad_raw = np.median(np.abs(depth_vals_clean - median_depth))
                mad = 1.4826 * mad_raw

                # Define reasonable depth range (configurable)
                # Default: typical bathymetry 0-50m depth
                expected_min_depth = -60.0  # 60m max depth
                expected_max_depth = 5.0    # 5m above surface (for tidal variation)

                # Check for gross outliers
                n_too_deep = (depth_vals_clean < expected_min_depth).sum()
                n_too_shallow = (depth_vals_clean > expected_max_depth).sum()

                if n_too_deep > 0 or n_too_shallow > 0:
                    log.warning(
                        f"[load_extra_xyz] QC WARNING: Potential outliers detected in {filepath.name}"
                    )
                    if n_too_deep > 0:
                        log.warning("  %s points deeper than %sm", n_too_deep, expected_min_depth)
                    if n_too_shallow > 0:
                        log.warning("  %s points shallower than %sm", n_too_shallow, expected_max_depth)

                # MAD-based outlier clipping (5-sigma equivalent)
                if mad > 0:
                    threshold = 5.0 * mad
                    lower_bound = median_depth - threshold
                    upper_bound = median_depth + threshold

                    outlier_mask = (tmp['depth_m'] < lower_bound) | (tmp['depth_m'] > upper_bound)
                    n_outliers = outlier_mask.sum()

                    if n_outliers > 0:
                        pct_outliers = 100.0 * n_outliers / len(tmp)
                        log.warning(
                            f"[load_extra_xyz] QC: Removing {n_outliers} outliers "
                            f"({pct_outliers:.1f}%) beyond 12×MAD "
                            f"(median={median_depth:.2f}, MAD={mad:.2f})"
                        )
                        tmp = tmp[~outlier_mask].copy()

                # Sanity check: AOI coordinate range
                lon_range = tmp['longitude'].max() - tmp['longitude'].min()
                lat_range = tmp['latitude'].max() - tmp['latitude'].min()

                if lon_range > 10.0 or lat_range > 10.0:
                    log.warning(
                        f"[load_extra_xyz] QC WARNING: Very large coordinate range "
                        f"(lon_range={lon_range:.2f}°, lat_range={lat_range:.2f}°). "
                        f"Check CRS transformation!"
                    )

            # Clip to AOI
            n_before = len(tmp)
            tmp = tmp[
                (tmp.longitude >= W) & (tmp.longitude <= E) &
                (tmp.latitude >= S) & (tmp.latitude <= N)
            ]
            n_after = len(tmp)

            log.info("AOI clipping: %s → %s points (%.1f%%)", n_before, n_after, n_after/max(n_before,1)*100)

            if not tmp.empty:
                # Preserve per-file provenance so authoritative subsets can be tiered differently.
                # If the input already has a meaningful 'source' column, keep it. Otherwise tag by filename.
                if "source" in tmp.columns and tmp["source"].notna().any():
                    tmp["source"] = tmp["source"].astype(str)
                else:
                    tag = filepath.stem.lower().strip()
                    # normalize common prefixes
                    tag = re.sub(r"[^a-z0-9_\-]+", "_", tag)
                    tmp["source"] = f"extra_xyz:{tag}"

                # Keep only required columns
                tmp = tmp[['longitude', 'latitude', 'depth_m', 'source']].copy()

                # Log depth statistics
                depth_stats = tmp['depth_m'].describe()
                log.info(f"Depth stats: min={depth_stats['min']:.2f}, "
                        f"median={depth_stats['50%']:.2f}, max={depth_stats['max']:.2f} m")

                dfs.append(tmp)
            else:
                log.warning("No points within AOI after clipping")

        except Exception as e:
            log.error("Failed to load %s: %s", f, e)
            import traceback
            log.debug(traceback.format_exc())

    if not dfs:
        log.warning("No XYZ data loaded from any file")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    log.info("XYZ data loaded: %d points | lon [%.4f, %.4f] | lat [%.4f, %.4f] | depth [%.2f, %.2f] m",
             len(result), result.longitude.min(), result.longitude.max(),
             result.latitude.min(), result.latitude.max(),
             result.depth_m.min(), result.depth_m.max())

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ICESat‑2 ATL03/ATL24 granules and extract training points to a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""            Examples:
              # ATL03 photons -> CSV
              python atl.py --product atl03 --aoi "-74.5/-74.25/40.25/40.5" --start 2025-01-01 --end 2026-01-01 \
                --cache-dir cache/icesat2 --out-csv output/atl03_points.csv

              # ATL24 bathymetry points -> CSV
              python atl.py --product atl24 --aoi "-74.5/-74.25/40.25/40.5" --start 2025-01-01 --end 2026-01-01 \
                --cache-dir cache/icesat2 --out-csv output/atl24_points.csv

            Notes:
              - AOI format is W/E/S/N (lon/lat degrees).
              - Use --land-mask to exclude obvious land contamination.
            """),
    )
    parser.add_argument("--aoi", required=True, help="Bounding box W/E/S/N (lon/lat degrees).")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--product", default="atl03", choices=["atl03", "atl24"], help="ICESat‑2 product to use.")
    parser.add_argument("--cache-dir", default="cache/icesat2", help="Cache directory for granules.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--land-mask", default=None, help="Optional land/water mask raster; interpreted by downstream filters.")
    args = parser.parse_args()

    w, e, s, n = [float(x) for x in args.aoi.split("/")]
    bbox = [w, s, e, n]

    cache_path = Path(args.cache_dir)
    files, _, status = ensure_icesat_files_harmony_cachefirst(cache_path, args.product.upper(), bbox, args.start, args.end)

    if args.product == "atl03":
        df = collect_training_points_from_atl03(
            files, 0.00005, 0.25, args.aoi, 4, 90.0, True, 20.0, 532.0,
            3, 0.1, 0.1, 40.0, False, land_mask_path=args.land_mask
        )
    else:
        df = collect_training_points_from_atl24(
            files, args.aoi, 40.0, 0.0, None, 42, 0.8, land_mask_path=args.land_mask
        )

    df.to_csv(args.out_csv, index=False)
    log.info("Wrote %s points to %s", len(df), args.out_csv)

if __name__ == "__main__":
    main()