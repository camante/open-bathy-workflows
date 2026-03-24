#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2_optics.py — Weighted Shared-Date Composite (STRICT default)

Updates:
1) FIXED NameError: '_aoi_grid_from_ref' is clearly defined before it is used.
2) Cache Directory: Defaults to ./cache/sentinel2/S2_{AOI}_{TIMEFRAME} if not provided.
3) QC Fix: Valid pixel check now scales by tile weight (fixes bug for multi-tile AOIs).
4) Cleanup: Deletes bad/rejected dates immediately to save space.
"""


import argparse
import logging
import math
import os
import re
import time
import calendar
import json
import hashlib
import warnings

# -----------------------------------------------------------------------------
# Optional exact-match caching utilities (cache_utils.py)
# -----------------------------------------------------------------------------
try:
    from cache_utils import (
        fingerprint_file,
        fingerprint_code,
        artifact_cache_key,
        cache_hit,
        write_meta,
    )
    _CACHE_UTILS_AVAILABLE = True
except Exception:  # pragma: no cover
    log.debug("s2_optics: suppressed exception", exc_info=True)
    _CACHE_UTILS_AVAILABLE = False

import shutil
import socket
import atexit
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def _safe_rint_to_uint8(arr: np.ndarray, fill: int = 0) -> np.ndarray:
    """Round/cast to uint8 safely (avoids RuntimeWarning on NaN)."""
    out = np.full(arr.shape, int(fill), dtype=np.uint8)
    m = np.isfinite(arr)
    if np.any(m):
        v = np.rint(arr[m]).astype(np.int32)
        v = np.clip(v, 0, 255).astype(np.uint8)
        out[m] = v
    return out


def _nanmean_stack_no_warn(stack: np.ndarray) -> np.ndarray:
    """Mean across axis=0 without noisy all-NaN RuntimeWarnings."""
    valid = np.isfinite(stack)
    den = valid.sum(axis=0)
    num = np.where(valid, stack, 0.0).sum(axis=0, dtype=np.float64)
    out = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 0)
    return out.astype(np.float32)


def _nanmedian_stack_no_warn(stack: np.ndarray) -> np.ndarray:
    """Median across axis=0 without cluttering logs for expected all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmedian(stack, axis=0)
    return np.asarray(out, dtype=np.float32)


def apply_s2_brightness_depth_filter(df, *args, **kwargs):
    """Production-facing helper exported for train.py binding.

    Delegates to train.py's maintained implementation to avoid interface drift
    without duplicating QC logic here. Imported lazily to avoid module cycles at
    import time.
    """
    from train import _LOCAL_APPLY_S2_BRIGHTNESS_DEPTH_FILTER as _impl
    return _impl(df, *args, **kwargs)


def apply_stumpf_residual_filter(df, *args, **kwargs):
    """Production-facing helper exported for train.py binding."""
    from train import _LOCAL_APPLY_STUMPF_RESIDUAL_FILTER as _impl
    return _impl(df, *args, **kwargs)
import requests
from requests.exceptions import HTTPError, RequestException

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except Exception:  # pragma: no cover - optional in lightweight test environments
    log.debug("apply_stumpf_residual_filter: suppressed exception", exc_info=True)
    rasterio = None  # type: ignore
    from_bounds = None  # type: ignore
    Resampling = None  # type: ignore
    reproject = None  # type: ignore

from scipy.ndimage import binary_dilation, distance_transform_edt

try:
    from shapely.geometry import shape, box, Polygon, MultiPolygon
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - optional in lightweight test environments
    log.debug("apply_stumpf_residual_filter: suppressed exception", exc_info=True)
    shape = box = Polygon = MultiPolygon = unary_union = None  # type: ignore

try:
    from pyproj import Geod, Transformer
except Exception:  # pragma: no cover - optional in lightweight test environments
    log.debug("apply_stumpf_residual_filter: suppressed exception", exc_info=True)
    Geod = Transformer = None  # type: ignore

# -------------------------
# Logging
# -------------------------
log = logging.getLogger("s2_optics")

# -------------------------
# Defaults
# -------------------------
DEFAULT_COLLECTION = "sentinel-2-l2a"
DEFAULT_STAC_URL = "https://earth-search.aws.element84.com/v1"  # no token
BANDS = ["B02", "B03", "B04", "B08", "SCL"]

# SCL codes to treat as bad / masked out
SCL_BAD = {0, 3, 8, 9, 10, 11}
GEOD = Geod(ellps="WGS84") if callable(Geod) else None

# -------------------------
# Data model
# -------------------------
@dataclass(frozen=True)
class Scene:
    id: str
    tile: str
    dt: str          # ISO datetime string
    cloud: float
    orbit: str
    assets: dict
    geometry: dict
    # Sun/view angles for physics-based SDB (Kim et al. 2024)
    sun_zenith: Optional[float] = None   # Solar zenith angle (degrees)
    sun_azimuth: Optional[float] = None  # Solar azimuth angle (degrees)
    view_zenith: Optional[float] = None  # Viewing zenith angle (degrees)
    view_azimuth: Optional[float] = None # Viewing azimuth angle (degrees)

# -------------------------
# Parsing helpers
# -------------------------
def parse_aoi(aoi_str: str) -> Tuple[float, float, float, float]:
    """Parse ``"W/E/S/N"`` → ``(W, E, S, N)``.  Delegates to :func:`pipeline.aoi.parse_aoi_wesn`."""
    from pipeline.aoi import parse_aoi_wesn
    return parse_aoi_wesn(aoi_str, strict=True)


def prepare_user_land_mask(mask_path: str, ref_raster: str, out_path: str) -> str:
    """Align a user-provided land/water mask to the reference raster grid.

    This helper preserves the user's original mask values and only performs
    nearest-neighbor reprojection/resampling onto the Sentinel-2 reference grid.
    It intentionally avoids interpreting the mask semantics here; downstream code
    already uses the explicit land-mask-type / water value arguments.
    """
    if rasterio is None or reproject is None or Resampling is None:
        raise RuntimeError("rasterio is required to prepare a user land mask")

    src_path = Path(mask_path)
    ref_path = Path(ref_raster)
    dst_path = Path(out_path)
    if not src_path.exists():
        raise FileNotFoundError(f"User land mask not found: {src_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {ref_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ref_path) as ref_ds, rasterio.open(src_path) as src_ds:
        profile = ref_ds.profile.copy()
        dtype = src_ds.dtypes[0]
        dst = np.zeros((ref_ds.height, ref_ds.width), dtype=np.dtype(dtype))
        src_nodata = src_ds.nodata
        if src_nodata in (0, 1):
            src_nodata = None
        dst_nodata = src_ds.nodata
        reproject(
            source=rasterio.band(src_ds, 1),
            destination=dst,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            src_nodata=src_nodata,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            dst_nodata=dst_nodata,
            resampling=Resampling.nearest,
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
        profile.update(dtype=dtype, count=1, nodata=dst_nodata, compress="deflate")
        with rasterio.open(dst_path, "w", **profile) as dst_ds:
            dst_ds.write(dst, 1)
    return str(dst_path)

# -------------------------
# Cache key helpers (AOI + time frame)
# -------------------------
def _norm_bbox_key(bbox_wesn: Tuple[float, float, float, float], ndp: int = 3) -> str:
    w, e, s, n = bbox_wesn
    return f"{w:.{ndp}f}_{e:.{ndp}f}_{s:.{ndp}f}_{n:.{ndp}f}"

def _composite_cache_key(bbox_wesn: Tuple[float, float, float, float], start_date: str, end_date: str) -> str:
    """Cache key used to validate cache hits: *only* AOI + time frame."""
    base = f"{_norm_bbox_key(bbox_wesn)}__{start_date}__{end_date}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:16]

def _read_composite_meta(out_dir: Path) -> Optional[dict]:
    meta_path = Path(out_dir) / "S2_COMPOSITE_META.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        log.debug("_read_composite_meta: suppressed exception", exc_info=True)
        return None

def _write_composite_meta(out_dir: Path, meta: dict) -> None:
    meta_path = Path(out_dir) / "S2_COMPOSITE_META.json"
    try:
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    except Exception:
        log.warning("Failed to write composite meta to %s", meta_path, exc_info=True)



def _sample_band_masked(path, *, mask, np_mod, rio_mod, max_samples=750000):
    """Read a single-band raster, apply a finite mask, and return a flat float32 sample array."""
    with rio_mod.open(str(path)) as ds:
        a = ds.read(1).astype(np_mod.float32)
        if ds.nodata is not None:
            a = np_mod.where(a == ds.nodata, np_mod.nan, a)
    if mask is not None:
        a = np_mod.where(mask, a, np_mod.nan)
    v = a[np_mod.isfinite(a)]
    if v.size > max_samples:
        idx = np_mod.random.choice(v.size, size=max_samples, replace=False)
        v = v[idx]
    return v



def _build_s2_params_fingerprint_dict(
    bbox_wesn: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    **kwargs,
) -> Dict[str, object]:
    """Return a dict of *all* output-affecting parameters for exact-match caching.

    Notes:
      - Include anything that changes which scenes/pixels are used or how mosaics are built.
      - Exclude performance-only knobs (threads) unless you truly want them to invalidate caches.
    """
    # Always include AOI + time bounds (they affect STAC query and output footprint)
    params = {
        "bbox_wesn": [float(x) for x in bbox_wesn],
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    # Merge remaining keyword args (already passed in from build_weighted_shared_date_composite)
    for k, v in kwargs.items():
        params[k] = v
    return params


def _s2_exact_cache_key(
    *,
    bbox_wesn: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    params: Dict[str, object],
    inputs: Dict[str, object],
    cache_code_strict: bool = False,
    cache_ignore_code: bool = True,
) -> str:
    """Compute an exact-match cache key for the final S2 composite outputs."""
    if not _CACHE_UTILS_AVAILABLE:
        # Fallback: old-style key (AOI+timeframe only)
        return _composite_cache_key(bbox_wesn, start_date, end_date)
    
    # If cache_ignore_code is True, don't include code fingerprint in cache key
    code_fp = ""
    if not cache_ignore_code:
        try:
            code_fp = fingerprint_code(Path(__file__), strict=bool(cache_code_strict))
        except Exception:
            log.debug("_s2_exact_cache_key: suppressed exception", exc_info=True)
            code_fp = ""
    
    return artifact_cache_key(stage="s2_composite", params=params, inputs=inputs, code_fp=code_fp, key_len=20)

def _purge_expected_outputs(expected: Dict[str, Path]) -> None:
    for p in expected.values():
        try:
            if p.exists():
                p.unlink()
        except Exception:
            log.debug("ignored", exc_info=True)


def acquire_run_lock(lock_path: Path, stale_hours: float = 6.0) -> None:
    """Acquire a self-healing run lock.

    - If the lock exists and is younger than `stale_hours`, abort (likely active run).
    - If it is older (or corrupted), delete it and proceed.
    - Always registers an atexit cleanup handler; caller should still also clean up in finally.
    """
    lock_path = Path(lock_path)
    now = time.time()

    if lock_path.exists():
        age_hours = 1e9
        try:
            meta = {}
            raw = lock_path.read_text().strip()
            # Prefer JSON meta (new format), but tolerate legacy plain-text.
            if raw.startswith("{"):
                meta = json.loads(raw)
            else:
                # legacy: lines like pid=... start=...
                for line in raw.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        meta[k.strip()] = v.strip()
            start_t = float(meta.get("start_time", meta.get("start", 0)) or 0)
            if start_t > 0:
                age_hours = (now - start_t) / 3600.0
        except Exception:
            log.debug("acquire_run_lock: suppressed exception", exc_info=True)
            age_hours = 1e9

        if age_hours < float(stale_hours):
            raise RuntimeError(
                f"[S2] Active run lock detected ({age_hours:.2f}h old): {lock_path} "
                f"(remove only if you're sure it's stale)"
            )

        log.warning("Removing stale lock (%.1fh old): %s", age_hours, lock_path)
        try:
            lock_path.unlink()
        except Exception:
            log.debug("ignored", exc_info=True)

    meta = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "start_time": now,
        "script": Path(__file__).name,
    }
    try:
        lock_path.write_text(json.dumps(meta, indent=2))
    except Exception:
        log.debug("s2_optics: suppressed exception", exc_info=True)
        # last resort: plain text
        lock_path.write_text(f"pid={os.getpid()}\nstart={now}\n")

    def _cleanup():
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            log.debug("ignored", exc_info=True)

    atexit.register(_cleanup)

def month_from_iso(dt: str) -> int:
    return int(dt[5:7])

def extract_datetime(item: dict) -> Optional[str]:
    props = item.get("properties", {}) or {}
    for k in ("datetime", "start_datetime", "end_datetime"):
        v = props.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def extract_cloud(item: dict) -> float:
    props = item.get("properties", {}) or {}
    v = props.get("eo:cloud_cover", None)
    try:
        return float(v) if v is not None else 100.0
    except Exception:
        log.debug("extract_cloud: suppressed exception", exc_info=True)
        return 100.0

def extract_tile(item: dict) -> str:
    """Extract MGRS tile (e.g., '15TXM') from common STAC conventions."""
    props = item.get("properties", {}) or {}

    for k in (
        "s2:mgrs_tile", "mgrs:tile", "sentinel:mgrs_tile", "s2:tile", "tileid", "tileId",
        "mgrs_tile", "mgrsTile", "s2:tile_id", "s2:tileid", "earthsearch:mgrid",
        "grid:code",
    ):
        v = props.get(k)
        if isinstance(v, str) and v.strip():
            vv = v.strip().upper()
            m = re.search(r"(\d{2}[A-Z]{3})", vv)
            if m:
                return m.group(1).upper()

    for v in props.values():
        if isinstance(v, str):
            vv = v.strip().upper()
            if re.fullmatch(r"\d{2}[A-Z]{3}", vv):
                return vv
            m = re.search(r"(\d{2}[A-Z]{3})", vv)
            if m and ("MGRS" in vv or "TILE" in vv or "GRID" in vv):
                return m.group(1).upper()

    iid = (item.get("id", "") or "").upper()

    m = re.search(r"_T(\d{2}[A-Z]{3})_", iid)
    if m:
        return m.group(1).upper()

    m = re.search(r"\bT(\d{2}[A-Z]{3})\b", iid)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b(\d{2}[A-Z]{3})\b", iid)
    if m:
        return m.group(1).upper()

    return "UNK"

def extract_orbit(item: dict) -> str:
    props = item.get("properties", {}) or {}
    for k in ("sat:relative_orbit", "s2:relative_orbit", "relative_orbit"):
        v = props.get(k)
        if v is None:
            continue
        try:
            return f"{int(v):03d}"
        except Exception:
            log.debug("ignored", exc_info=True)
    iid = item.get("id", "") or ""
    m = re.search(r"_R(\d{3})_", iid)
    if m:
        return m.group(1)
    return "UNK"


def extract_sun_angles(item: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract solar zenith and azimuth angles from STAC item properties.
    
    Sentinel-2 STAC items from Element84 Earth Search typically include:
    - s2:mean_solar_zenith or view:sun_elevation
    - s2:mean_solar_azimuth or view:sun_azimuth
    
    Returns (sun_zenith_deg, sun_azimuth_deg) or (None, None) if not found.
    """
    props = item.get("properties", {}) or {}
    
    sza = None
    saa = None
    
    # Solar zenith
    for k in ("s2:mean_solar_zenith", "view:sun_zenith", "sun_zenith",
              "mean_solar_zenith", "solar_zenith_angle"):
        v = props.get(k)
        if v is not None:
            try:
                sza = float(v)
                break
            except (TypeError, ValueError):
                pass
    
    # If sun elevation instead of zenith
    if sza is None:
        for k in ("view:sun_elevation", "sun_elevation", "s2:sun_elevation"):
            v = props.get(k)
            if v is not None:
                try:
                    sza = 90.0 - float(v)  # Convert elevation to zenith
                    break
                except (TypeError, ValueError):
                    pass
    
    # Solar azimuth
    for k in ("s2:mean_solar_azimuth", "view:sun_azimuth", "sun_azimuth",
              "mean_solar_azimuth", "solar_azimuth_angle"):
        v = props.get(k)
        if v is not None:
            try:
                saa = float(v)
                break
            except (TypeError, ValueError):
                pass
    
    return sza, saa


def extract_view_angles(item: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract viewing zenith and azimuth angles from STAC item properties.
    
    Returns (view_zenith_deg, view_azimuth_deg) or (None, None) if not found.
    """
    props = item.get("properties", {}) or {}
    
    vza = None
    vaa = None
    
    # Viewing zenith
    for k in ("s2:mean_viewing_zenith", "view:off_nadir", "view_zenith",
              "mean_viewing_zenith", "viewing_zenith_angle", "s2:viewing_zenith"):
        v = props.get(k)
        if v is not None:
            try:
                vza = float(v)
                break
            except (TypeError, ValueError):
                pass
    
    # Viewing azimuth
    for k in ("s2:mean_viewing_azimuth", "view:azimuth", "view_azimuth",
              "mean_viewing_azimuth", "viewing_azimuth_angle", "s2:viewing_azimuth"):
        v = props.get(k)
        if v is not None:
            try:
                vaa = float(v)
                break
            except (TypeError, ValueError):
                pass
    
    return vza, vaa

# -------------------------
# Date window chunking
# -------------------------
def _parse_date(d: str) -> Tuple[int, int, int]:
    y, m, dd = d.split("-")
    return int(y), int(m), int(dd)

def _date_to_ymd(t: Tuple[int, int, int]) -> str:
    return f"{t[0]:04d}-{t[1]:02d}-{t[2]:02d}"

def _days_in_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]

def _add_months(ymd: Tuple[int, int, int], months: int) -> Tuple[int, int, int]:
    y, m, d = ymd
    m2 = m - 1 + months
    y2 = y + m2 // 12
    m2 = (m2 % 12) + 1
    d2 = min(d, _days_in_month(y2, m2))
    return (y2, m2, d2)

def _iter_date_windows(start_date: str, end_date: str, chunk_months: int) -> Iterable[Tuple[str, str]]:
    if chunk_months <= 0:
        yield (start_date, end_date)
        return
    s = _parse_date(start_date)
    e = _parse_date(end_date)
    cur = s
    while True:
        nxt = _add_months(cur, chunk_months)
        if nxt[2] == 1:
            py, pm = nxt[0], nxt[1] - 1
            if pm == 0:
                py, pm = py - 1, 12
            we = (py, pm, _days_in_month(py, pm))
        else:
            we = (nxt[0], nxt[1], nxt[2] - 1)

        if we > e:
            we = e

        yield (_date_to_ymd(cur), _date_to_ymd(we))
        if we >= e:
            break

        if we[2] < _days_in_month(we[0], we[1]):
            cur = (we[0], we[1], we[2] + 1)
        else:
            ny, nm = we[0], we[1] + 1
            if nm > 12:
                ny, nm = ny + 1, 1
            cur = (ny, nm, 1)

# -------------------------
# STAC search (robust pagination, GET/POST)
# -------------------------
def _request_with_backoff(
    method: str,
    url: str,
    *,
    headers: dict,
    json_body: Optional[dict],
    timeout: int,
    max_retries: int = 6,
    base_delay: float = 2.0,
) -> requests.Response:
    attempt = 0
    while True:
        attempt += 1
        try:
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=timeout)
            else:
                resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)

            if resp.status_code == 429 and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                log.warning("429 rate-limit. Retry in %.1fs (attempt %s/%s)", delay, attempt, max_retries)
                time.sleep(delay)
                continue

            resp.raise_for_status()
            return resp

        except (HTTPError, RequestException) as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                log.warning("Request error: %s. Retry in %.1fs (attempt %s/%s)", e, delay, attempt, max_retries)
                time.sleep(delay)
                continue
            raise

def stac_search_window(
    stac_url: str,
    collection: str,
    bbox_wesn: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    *,
    limit: int = 200,
    max_items: int = 5000,
    timeout: int = 60,
) -> List[dict]:
    base = stac_url.rstrip("/")
    search_url = base + "/search"
    w, e, s, n = bbox_wesn
    bbox = [w, s, e, n]
    payload = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": int(min(max(1, limit), 500)),
    }
    headers = {"Content-Type": "application/json"}
    items: Dict[str, dict] = {}

    next_url = search_url
    next_method = "POST"
    page = 0

    while next_url and len(items) < max_items:
        page += 1
        t0 = time.time()
        json_body = payload if next_method == "POST" else None
        resp = _request_with_backoff(next_method, next_url, headers=headers, json_body=json_body, timeout=timeout)
        fc = resp.json()

        feats = fc.get("features", []) or []
        new = 0
        for f in feats:
            fid = f.get("id")
            if fid and fid not in items:
                items[fid] = f
                new += 1

        dt = time.time() - t0
        log.info("Page %s: got %s feats (%s new), total=%s in %.1fs", page, len(feats), new, len(items), dt)

        next_url = None
        next_method = "GET"
        payload = None
        for lk in fc.get("links", []) or []:
            if lk.get("rel") == "next" and lk.get("href"):
                next_url = lk["href"]
                next_method = (lk.get("method") or "GET").upper()
                if next_method == "POST":
                    payload = lk.get("body") or {}
                else:
                    payload = None
                break

        if not next_url:
            break

    return list(items.values())

def stac_search(
    stac_url: str,
    collection: str,
    bbox_wesn: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    *,
    chunk_months: int = 0,
    limit: int = 200,
    max_items: int = 5000,
    timeout: int = 60,
) -> List[dict]:
    union: Dict[str, dict] = {}
    windows = list(_iter_date_windows(start_date, end_date, chunk_months))
    for i, (ws, we) in enumerate(windows, 1):
        log.info("Searching window %s/%s: %s to %s", i, len(windows), ws, we)
        feats = stac_search_window(
            stac_url, collection, bbox_wesn, ws, we,
            limit=limit, max_items=max_items, timeout=timeout
        )
        for f in feats:
            fid = f.get("id")
            if fid and fid not in union:
                union[fid] = f
        log.info("Window %s/%s done: got %s items, union_total=%s", i, len(windows), len(feats), len(union))
        if len(union) >= max_items:
            break

    log.info("Retrieved %s unique items total (chunk_months=%s; max=%s).", len(union), chunk_months, max_items)
    return list(union.values())

# -------------------------
# Geometry: tile weights (FIXED: union footprints per tile)
# -------------------------
def _geodesic_area_m2(geom) -> float:
    if geom.is_empty:
        return 0.0
    if isinstance(geom, Polygon):
        lon, lat = geom.exterior.coords.xy
        area, _ = GEOD.polygon_area_perimeter(lon, lat)
        area = abs(area)
        for interior in geom.interiors:
            lon, lat = interior.coords.xy
            a2, _ = GEOD.polygon_area_perimeter(lon, lat)
            area -= abs(a2)
        return max(0.0, float(area))
    if isinstance(geom, MultiPolygon):
        return float(sum(_geodesic_area_m2(g) for g in geom.geoms))
    return 0.0

def compute_tile_weights(items_by_tile: Dict[str, List[Scene]], aoi_poly_ll) -> Dict[str, float]:
    """
    Stable weights: use UNION of all scene footprints per tile, not a single scene footprint.
    """
    raw: Dict[str, float] = {}
    for tile, scenes in items_by_tile.items():
        geoms = []
        for sc in scenes:
            try:
                geoms.append(shape(sc.geometry))
            except Exception:
                log.debug("compute_tile_weights: suppressed exception", exc_info=True)
                continue
        if not geoms:
            continue
        try:
            union_geom = unary_union(geoms)
        except Exception:
            log.debug("compute_tile_weights: suppressed exception", exc_info=True)
            continue
        inter = union_geom.intersection(aoi_poly_ll)
        raw[tile] = _geodesic_area_m2(inter)

    total = sum(raw.values())
    if total <= 0:
        n = max(1, len(raw))
        return {t: 1.0 / n for t in raw}
    return {t: raw[t] / total for t in raw}

# -------------------------
# Asset selection / download (FIXED: suffix)
# -------------------------
def _is_http_url(href: str) -> bool:
    return isinstance(href, str) and (href.startswith("http://") or href.startswith("https://"))

def _infer_suffix_from_asset(asset: dict, href: str) -> str:
    """
    Prefer type hints; fallback to href.
    """
    t = (asset or {}).get("type") or ""
    tl = str(t).lower()
    hl = str(href).lower()

    if "jp2" in tl or hl.endswith(".jp2") or ".jp2?" in hl:
        return ".jp2"
    if "tiff" in tl or "geotiff" in tl or "cog" in tl or hl.endswith(".tif") or hl.endswith(".tiff") or ".tif?" in hl:
        return ".tif"
    if hl.endswith(".jp2"):
        return ".jp2"
    if hl.endswith(".tif") or hl.endswith(".tiff"):
        return ".tif"
    return ".dat"

def pick_asset_href(assets: dict, band: str) -> Optional[Tuple[str, str]]:
    """
    Returns (href, suffix).
    Supports Earth Search keys: blue/green/red/nir/scl and common band keys.
    Only accepts HTTP(S) hrefs.
    """
    if not assets:
        return None

    band = band.upper()
    key_map = {
        "B02": ["B02", "B02_10m", "blue", "BLUE"],
        "B03": ["B03", "B03_10m", "green", "GREEN"],
        "B04": ["B04", "B04_10m", "red", "RED"],
        "B08": ["B08", "B08_10m", "nir", "NIR"],
        "SCL": ["SCL", "SCL_20m", "scl", "SCL_20M"],
    }

    candidates: List[Tuple[int, str, str]] = []
    for k in key_map.get(band, [band, band.lower()]):
        v = assets.get(k)
        if isinstance(v, dict) and v.get("href"):
            href = v["href"]
            if _is_http_url(href):
                score = 10
                if "10m" in str(k).lower() or "10m" in href.lower():
                    score += 2
                sfx = _infer_suffix_from_asset(v, href)
                candidates.append((score, href, sfx))

    if not candidates:
        bu = band.upper()
        for k, v in assets.items():
            if not isinstance(v, dict):
                continue
            href = v.get("href")
            if not href or not _is_http_url(href):
                continue
            ku = str(k).upper()
            if bu in ku or bu in href.upper():
                score = 1
                if "10M" in ku or "10M" in href.upper():
                    score += 5
                sfx = _infer_suffix_from_asset(v, href)
                candidates.append((score, href, sfx))

    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    _, href, sfx = candidates[0]
    return href, sfx

def download_file(url: str, out_path: Path, timeout: int = 180) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    log.info("Downloading %s from %s", out_path.name, url)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(out_path)
        log.debug("downloaded %s", out_path.name)
    except Exception as e:
        log.error("Failed %s: %s", out_path.name, e)
        raise RuntimeError(f"Failed downloading {url} -> {out_path}: {e}") from e

def resolve_and_download_scene(scene: Scene, cache_dir: Path, max_workers: int = 8) -> Optional[Dict[str, Path]]:
    """
    Returns dict of band->local_path if all required bands have usable http(s) hrefs.
    If any band has no http(s) href, returns None (scene skipped).
    """
    # Create a subfolder per scene inside the main cache_dir
    scene_dir = cache_dir / scene.id
    scene_dir.mkdir(parents=True, exist_ok=True)

    urls: Dict[str, Tuple[str, Path]] = {}
    for b in BANDS:
        picked = pick_asset_href(scene.assets, b)
        if not picked:
            log.warning("Scene skipped (no public href) band=%s scene=%s", b, scene.id)
            return None
        href, sfx = picked
        out = scene_dir / f"{scene.id}_{b}{sfx}"
        urls[b] = (href, out)

    log.info("Downloading %d assets for scene %s", len(urls), scene.id)

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for b, (href, out) in urls.items():
            futures.append(ex.submit(download_file, href, out))

        all_ok = True
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                log.debug("resolve_and_download_scene: suppressed exception", exc_info=True)
                all_ok = False

    if not all_ok:
        log.warning("Failed to download all required assets for scene %s. Skipping scene.", scene.id)
        return None

    log.info("Successfully downloaded all assets for scene: %s", scene.id)
    return {b: p for b, (_, p) in urls.items()}

# -------------------------
# Warping / masking / mosaicing
# -------------------------
def _aoi_grid_from_ref(ref_path: Path, aoi_poly_ll, res_m: float = 10.0):
    with rasterio.open(ref_path) as src:
        crs = src.crs
    if crs is None:
        raise RuntimeError("Reference raster has no CRS")

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    minx, miny, maxx, maxy = aoi_poly_ll.bounds
    xs, ys = transformer.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = max(1, int(math.ceil((xmax - xmin) / res_m)))
    height = max(1, int(math.ceil((ymax - ymin) / res_m)))
    transform = from_bounds(xmin, ymin, xmax, ymax, width=width, height=height)
    return crs, transform, width, height

def warp_to_grid(
    src_path: Path,
    dst_crs,
    dst_transform,
    dst_width: int,
    dst_height: int,
    resampling: Resampling,
    dtype=np.float32,
) -> np.ndarray:
    """
    Reproject one band to the AOI grid and return float32 with NaN nodata.
    """
    dst_nodata = -9999.0
    dst = np.full((dst_height, dst_width), dst_nodata, dtype=dtype)
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_nodata = src.nodata
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            init_dest_nodata=True,
            resampling=resampling,
            num_threads=2,
        )
    dst = dst.astype(np.float32, copy=False)
    dst[dst == dst_nodata] = np.nan
    return dst
def _robust_median_mad(x: np.ndarray) -> Tuple[float, float]:
    """Return (median, MAD) with a small epsilon to avoid division-by-zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, mad if mad > 1e-12 else 1e-12


def _erode_binary(mask: np.ndarray, px: int) -> np.ndarray:
    """Binary erosion without requiring binary_erosion import (uses dilation trick)."""
    if px <= 0:
        return mask.astype(bool)
    # Erode == invert(dilate(invert(mask)))
    inv = ~mask.astype(bool)
    inv_d = binary_dilation(inv, iterations=int(px))
    return ~inv_d


def load_waffles_coast_mask_to_grid(
    mask_path: str,
    dst_crs,
    dst_transform,
    dst_w: int,
    dst_h: int,
    *,
    water_value: int = 0,
    invert: bool = False,
    erode_px: int = 2,
) -> np.ndarray:
    """
    Load a coastline/land-water mask raster (e.g., Waffles output) and warp to the AOI grid.
    Returns a boolean mask where True == target-water pixels (optionally eroded inward).

    Assumptions:
      - mask is a raster with discrete values (nearest-neighbor resampling).
      - water_value identifies water pixels. Set invert=True if your mask uses opposite convention.
    """
    if not mask_path:
        raise ValueError("mask_path is empty")
    mp = Path(mask_path)
    if not mp.exists():
        raise FileNotFoundError(f"Coastline mask not found: {mask_path}")

    m = warp_to_grid(mp, dst_crs, dst_transform, dst_w, dst_h, resampling=Resampling.nearest, dtype=np.float32)
    m_int = np.where(np.isfinite(m), np.rint(m).astype(np.int16), -9999)
    water = (m_int == int(water_value))
    if invert:
        water = ~water
    # Erode water inward to avoid coastline edge contamination
    water = _erode_binary(water, int(erode_px))
    return water


def score_date_quality(
    mosaic: Dict[str, np.ndarray],
    water_mask: np.ndarray,
    *,
    b02_thresh: float,
    deepwater_nir_max: float,
    deepwater_bright_max: float,
    allow_bright_shallow_pixels: bool,
    bright_shallow_nir_max: float,
    weight_valid: float,
    weight_glint: float,
    weight_deepwater: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute quick, coastline-aware per-date QC metrics and a single score (lower is better).

    Metrics are computed ONLY within `water_mask` (your Waffles coastline water area, eroded),
    and only where SCL + spectral values are finite.

    Returns: (score, metrics_dict)
    """
    b02 = mosaic["B02"]; b03 = mosaic["B03"]; b04 = mosaic["B04"]; b08 = mosaic["B08"]
    scl = mosaic["SCL"]

    # SCL valid pixels
    scl_int = _safe_rint_to_uint8(scl, fill=0)
    v_scl = scl_valid_mask(scl_int, dilate=0)

    # Coastline-aware predictability metrics (computed over target water mask)
    try:
        scl_bad = {3, 8, 9, 10, 11}
        dm = _compute_date_metrics(b02, b03, b04, b08, scl_int, water_mask, water_mask, scl_bad, v_scl)
    except Exception:
        log.debug("score_date_quality: suppressed exception", exc_info=True)
        dm = {"coverage": valid_frac, "bad_frac": glint_frac, "nir_med": float("nan"), "red_med": float("nan")}

    finite = np.isfinite(b02) & np.isfinite(b03) & np.isfinite(b04) & np.isfinite(b08)
    base = water_mask & v_scl & finite

    if not np.any(water_mask):
        # No water area to score.
        metrics = {
        **dm,
            "water_pixels": 0.0,
            "valid_frac": 0.0,
            "glint_frac": 1.0,
            "deepwater_frac": 0.0,
        }
        return 1e9, metrics

    water_n = float(np.sum(water_mask))
    valid_n = float(np.sum(base))
    valid_frac = valid_n / water_n if water_n > 0 else 0.0

    # Glint proxy: bright blue among valid water pixels
    glint_frac = float(np.sum(base & (b02 > float(b02_thresh)))) / valid_n if valid_n > 0 else 1.0

    # Deep-water stability proxy: where you trust radiometry most for harmonization
    bright = compute_brightness(b02, b03, b04)
    deep = base & (b08 <= float(deepwater_nir_max)) & (bright <= float(deepwater_bright_max))
    deepwater_frac = float(np.sum(deep)) / valid_n if valid_n > 0 else 0.0

    # Optional: allow bright shallow pixels (for tropics), but we still track it
    if allow_bright_shallow_pixels:
        bright_shallow = base & (b02 > float(b02_thresh)) & (b08 < float(bright_shallow_nir_max))
        bright_shallow_frac = float(np.sum(bright_shallow)) / valid_n if valid_n > 0 else 0.0
    else:
        bright_shallow_frac = 0.0

    # Composite score (lower is better)
    score = (
        weight_valid * (1.0 - valid_frac)
        + weight_glint * glint_frac
        + weight_deepwater * (1.0 - deepwater_frac)
    )

    metrics = {
        "water_pixels": water_n,
        "valid_frac": valid_frac,
        "glint_frac": glint_frac,
        "deepwater_frac": deepwater_frac,
        "bright_shallow_frac": bright_shallow_frac,
    }
    return float(score), metrics


def scl_valid_mask(scl: np.ndarray, dilate: int = 1) -> np.ndarray:
    bad = np.zeros_like(scl, dtype=bool)
    for code in SCL_BAD:
        bad |= (scl == code)
    if dilate and dilate > 0:
        bad = binary_dilation(bad, iterations=int(dilate))
    return ~bad


def _compute_date_metrics(
    b02: np.ndarray, b03: np.ndarray, b04: np.ndarray, b08: np.ndarray,
    scl_int: np.ndarray,
    target_water: Optional[np.ndarray],
    target_water_inner: Optional[np.ndarray],
    scl_bad: set,
    vmask: np.ndarray
) -> Dict[str, float]:
    """Compute quick QC metrics over waffles-defined water if provided."""
    if target_water is None:
        target_water = np.isfinite(b03)
    if target_water_inner is None:
        target_water_inner = target_water

    tw = target_water.astype(bool)
    twi = target_water_inner.astype(bool)

    denom_tw = int(np.count_nonzero(tw))
    denom_twi = int(np.count_nonzero(twi))
    if denom_tw == 0 or denom_twi == 0:
        return {"coverage": 0.0, "bad_frac": 1.0, "nir_med": float("inf"), "red_med": float("inf")}

    finite = np.isfinite(b02) & np.isfinite(b03) & np.isfinite(b04) & np.isfinite(b08)
    valid_predict = twi & vmask & finite
    coverage = float(np.count_nonzero(valid_predict) / max(denom_twi, 1))

    bad = np.zeros_like(scl_int, dtype=bool)
    for code in scl_bad:
        bad |= (scl_int == int(code))
    bad_frac = float(np.count_nonzero(tw & bad) / max(denom_tw, 1))

    if np.count_nonzero(valid_predict) > 0:
        nir_med = float(np.nanmedian(b08[valid_predict]))
        red_med = float(np.nanmedian(b04[valid_predict]))
    else:
        nir_med = float("inf")
        red_med = float("inf")

    return {"coverage": coverage, "bad_frac": bad_frac, "nir_med": nir_med, "red_med": red_med}


def _robust_z_map(keys: List[str], values: List[float]) -> Dict[str, float]:
    a = np.array(values, dtype=float)
    mask = np.isfinite(a)
    if np.count_nonzero(mask) < 3:
        return {k: 0.0 for k in keys}
    a2 = a[mask]
    med = float(np.nanmedian(a2))
    mad = float(np.nanmedian(np.abs(a2 - med)))
    if not np.isfinite(mad) or mad <= 0:
        return {k: 0.0 for k in keys}
    z2 = 0.6745 * (a2 - med) / mad
    out = {}
    j = 0
    for k, ok in zip(keys, mask):
        if ok:
            out[k] = float(z2[j]); j += 1
        else:
            out[k] = 0.0
    return out

def compute_brightness(b02, b03, b04) -> np.ndarray:
    return (b02 + b03 + b04) / 3.0

def deep_water_mask(
    b08: np.ndarray,
    brightness: np.ndarray,
    nir_max: float = 0.03,
    bright_max: float = 0.15,
    b02: Optional[np.ndarray] = None,
    b02_max: Optional[float] = None,
) -> np.ndarray:
    """
    "Stable water" mask used for overlap harmonization:
    prefers deep, dark water with low glint/brightness.
    """
    m = (
        np.isfinite(b08) & np.isfinite(brightness) &
        (b08 < nir_max) & (brightness < bright_max)
    )
    if b02 is not None and b02_max is not None:
        m &= np.isfinite(b02) & (b02 < float(b02_max))
    return m

def _hedley_glint_correct(
    nir: np.ndarray,
    vis: Dict[str, np.ndarray],
    *,
    scl: Optional[np.ndarray] = None,
    scl_bad: Optional[set] = None,
    deepwater_nir_max: float = 0.03,
    deepwater_bright_max: float = 0.15,
    deepwater_b02_max: float = 0.20,
    nir_min_percentile: float = 1.0,
    min_samples: int = 5000,
    max_samples: int = 2_000_000,
    clip_min: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
    label: str = "composite",
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Apply Hedley-style sun-glint correction to visible bands.

    Corrects each visible band R_vis using:
        R_vis' = R_vis - beta * (R_nir - R_nir_min)

    beta is fit as the slope of a linear regression of R_vis vs R_nir over a
    stable-water mask (deep/dark water). R_nir_min is taken as a low percentile
    of R_nir over that same mask.

    Returns:
        (corrected_vis_dict, meta_dict)
    """
    meta = {
        "enabled": True,
        "status": "skipped",
        "label": str(label),
        "nir_min_percentile": float(nir_min_percentile),
        "deepwater_nir_max": float(deepwater_nir_max),
        "deepwater_bright_max": float(deepwater_bright_max),
        "deepwater_b02_max": float(deepwater_b02_max),
        "min_samples": int(min_samples),
        "max_samples": int(max_samples),
        "clip_min": float(clip_min),
        "betas": {},
        "n_samples": 0,
        "nir_min": None,
        "mask_mode": None,
        "reason": None,
    }

    if rng is None:
        rng = np.random.default_rng(0)

    if nir is None or not isinstance(nir, np.ndarray):
        meta["reason"] = "nir_missing"
        return vis, meta

    vis_list = [v for v in vis.values() if isinstance(v, np.ndarray)]
    if not vis_list:
        meta["reason"] = "no_vis_bands"
        return vis, meta

    brightness = np.nanmean(np.stack(vis_list, axis=0), axis=0).astype(np.float32)

    b02_arr = vis.get("B02", None)
    mask0 = deep_water_mask(
        nir.astype(np.float32),
        brightness.astype(np.float32),
        nir_max=float(deepwater_nir_max),
        bright_max=float(deepwater_bright_max),
        b02=b02_arr.astype(np.float32) if isinstance(b02_arr, np.ndarray) else None,
        b02_max=float(deepwater_b02_max) if deepwater_b02_max is not None else None,
    )

    if scl is not None and scl_bad is not None:
        try:
            scl_int = _safe_rint_to_uint8(scl, fill=0)
            vmask = scl_valid_mask(scl_int, dilate=0, scl_bad=set(scl_bad))
            mask0 &= vmask
        except Exception:
            log.warning("SCL masking failed; proceeding without SCL quality filter", exc_info=True)

    mask = mask0
    mask_mode = "strict"

    n0 = int(np.count_nonzero(mask0))
    if n0 < int(min_samples):
        mask1 = deep_water_mask(
            nir.astype(np.float32),
            brightness.astype(np.float32),
            nir_max=float(deepwater_nir_max) * 2.0,
            bright_max=float(deepwater_bright_max) * 2.0,
            b02=b02_arr.astype(np.float32) if isinstance(b02_arr, np.ndarray) else None,
            b02_max=min(float(deepwater_b02_max) * 1.5, 1.0) if deepwater_b02_max is not None else None,
        )
        if scl is not None and scl_bad is not None:
            try:
                scl_int = _safe_rint_to_uint8(scl, fill=0)
                vmask = scl_valid_mask(scl_int, dilate=0, scl_bad=set(scl_bad))
                mask1 &= vmask
            except Exception:
                log.debug("ignored", exc_info=True)
        n1 = int(np.count_nonzero(mask1))
        if n1 > n0:
            mask = mask1
            mask_mode = "relaxed"

    n = int(np.count_nonzero(mask))
    meta["mask_mode"] = mask_mode
    meta["n_samples"] = int(min(n, int(max_samples)))

    if n < int(min_samples):
        meta["reason"] = f"insufficient_samples_{n}"
        return vis, meta

    idx = np.flatnonzero(mask)
    if idx.size > int(max_samples):
        idx = rng.choice(idx, size=int(max_samples), replace=False)

    nir_s = nir.ravel()[idx].astype(np.float64)
    good = np.isfinite(nir_s)
    if np.count_nonzero(good) < int(min_samples):
        meta["reason"] = "nir_nonfinite"
        return vis, meta
    nir_s = nir_s[good]
    idx = idx[good]

    try:
        nir_min = float(np.percentile(nir_s, float(nir_min_percentile)))
    except Exception:
        log.debug("s2_optics: suppressed exception", exc_info=True)
        nir_min = float(np.nanmin(nir_s))
    meta["nir_min"] = nir_min

    x = nir_s
    x_mean = float(np.mean(x))
    x_var = float(np.var(x))
    if not np.isfinite(x_var) or x_var <= 1e-12:
        meta["reason"] = "nir_variance_too_small"
        return vis, meta

    corrected: Dict[str, np.ndarray] = {}

    for band, arr in vis.items():
        if arr is None or not isinstance(arr, np.ndarray):
            continue

        y = arr.ravel()[idx].astype(np.float64)
        y_good = np.isfinite(y)
        if np.count_nonzero(y_good) < int(min_samples):
            continue

        y2 = y[y_good]
        x2 = x[y_good]
        y_mean = float(np.mean(y2))
        beta = float(np.mean((x2 - x_mean) * (y2 - y_mean)) / x_var)
        if not np.isfinite(beta):
            continue

        meta["betas"][str(band)] = float(beta)

        out = arr.astype(np.float32, copy=True)
        m = np.isfinite(out) & np.isfinite(nir)
        out[m] = out[m] - (beta * (nir[m].astype(np.float32) - float(nir_min)))
        if clip_min is not None and float(clip_min) > 0:
            out[m] = np.maximum(out[m], float(clip_min))
        corrected[str(band)] = out

    if not meta["betas"]:
        meta["reason"] = "no_valid_betas"
        return vis, meta

    meta["status"] = "applied"
    meta["reason"] = None
    return corrected, meta

def harmonize_offsets_to_reference(
    mosaics_by_tile: Dict[str, Dict[str, np.ndarray]],
    ref_tile: str,
    max_abs_offset: float = 0.03,
    deepwater_nir_max: float = 0.03,
    deepwater_bright_max: float = 0.15,
    b02_thresh: float = 0.25,
    min_overlap_px: int = 2000,
) -> None:
    """
    (FIXED) Cache deep-water masks per tile, compute offsets band-by-band using cached masks.
    """
    if ref_tile not in mosaics_by_tile:
        return

    # Cache brightness + deepwater masks
    br: Dict[str, np.ndarray] = {}
    dw: Dict[str, np.ndarray] = {}
    for t, bands in mosaics_by_tile.items():
        br[t] = compute_brightness(bands["B02"], bands["B03"], bands["B04"])
        dw[t] = deep_water_mask(bands["B08"], br[t], deepwater_nir_max, deepwater_bright_max, b02=bands["B02"], b02_max=min(float(b02_thresh), 0.20))

    ref = mosaics_by_tile[ref_tile]

    for tile, bands in mosaics_by_tile.items():
        if tile == ref_tile:
            continue

        # Prefer deep-water overlap; fallback to any overlap if insufficient
        deep_overlap = dw[ref_tile] & dw[tile]

        for b in ["B02", "B03", "B04", "B08"]:
            a = ref[b]
            c = bands[b]
            overlap = np.isfinite(a) & np.isfinite(c)
            if overlap.sum() < min_overlap_px:
                continue

            m = overlap & deep_overlap
            if m.sum() < min_overlap_px:
                m = overlap

            off = np.nanmedian(a[m] - c[m])
            if not np.isfinite(off):
                continue

            off = float(np.clip(off, -max_abs_offset, max_abs_offset))
            bands[b] = c + off

def mosaic_tiles(
    mosaics_by_tile: Dict[str, Dict[str, np.ndarray]],
    weights: Dict[str, float],
    *,
    feather: bool = True,
    dist_cap: int = 300,          # pixels (~3 km at 10 m)
    use_sqrt_weight: bool = True,
    edge_weight_power: float = 2.0,
    edge_weight_min: float = 0.0,
    # Temporal aggregation: if >0, only use the best K dates (post-QC) when building the final median composite.
    temporal_median_k: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Edge-aware feather mosaic across tiles.
    """
    tiles = list(mosaics_by_tile.keys())
    out: Dict[str, np.ndarray] = {}

    wmap: Optional[Dict[str, np.ndarray]] = None
    if feather and len(tiles) > 1:
        wmap = {}
        cap = float(max(1, int(dist_cap)))
        for t in tiles:
            # use B03 validity as proxy for valid footprint
            valid = np.isfinite(mosaics_by_tile[t]["B03"])
            if not valid.any():
                wmap[t] = np.zeros_like(mosaics_by_tile[t]["B03"], dtype=np.float32)
                continue

            d = distance_transform_edt(valid).astype(np.float32)  # 0 at edge/outside, larger towards interior
            d = np.minimum(d, cap)

            # normalized taper 0..1
            w = (d / cap)
            if edge_weight_power and edge_weight_power != 1.0:
                w = np.power(w, float(edge_weight_power))
            if edge_weight_min and edge_weight_min > 0:
                w = np.maximum(w, float(edge_weight_min))

            wt = float(weights.get(t, 1.0))
            w *= math.sqrt(wt) if use_sqrt_weight else wt

            wmap[t] = w.astype(np.float32)

    # Spectral bands
    for b in ["B02", "B03", "B04", "B08"]:
        stack = np.stack([mosaics_by_tile[t][b] for t in tiles], axis=0).astype(np.float32)
        if wmap is not None:
            wstack = np.stack([wmap[t] for t in tiles], axis=0).astype(np.float32)
            wstack = np.where(np.isfinite(stack), wstack, 0.0)
            num = np.nansum(stack * wstack, axis=0)
            den = np.sum(wstack, axis=0)
            out[b] = np.where(den > 0, num / den, np.nan).astype(np.float32)
        else:
            out[b] = _nanmean_stack_no_warn(stack)

    # SCL: choose tile with max weight at each pixel
    scl_stack = np.stack([mosaics_by_tile[t]["SCL"].astype(np.float32) for t in tiles], axis=0)
    if wmap is not None:
        wstack = np.stack([wmap[t] for t in tiles], axis=0).astype(np.float32)
        wstack = np.where(np.isfinite(scl_stack), wstack, 0.0)
        idx = np.argmax(wstack, axis=0)

        scl_out = np.full(idx.shape, np.nan, dtype=np.float32)
        for ti, t in enumerate(tiles):
            m = idx == ti
            scl_out[m] = scl_stack[ti][m]

        out["SCL"] = np.where(np.isfinite(scl_out), np.rint(scl_out), np.nan).astype(np.float32)
    else:
        scl_med = _nanmedian_stack_no_warn(scl_stack)
        out["SCL"] = np.where(np.isfinite(scl_med), np.rint(scl_med), np.nan).astype(np.float32)

    return out

def compute_clear_water_mask(b08, brightness, nir_max=0.05, bright_max=0.18, dilate=1) -> np.ndarray:
    m = np.isfinite(b08) & np.isfinite(brightness) & (b08 < nir_max) & (brightness < bright_max)
    if dilate and dilate > 0:
        m = binary_dilation(m, iterations=int(dilate))
    return m.astype(np.uint8)

def write_geotiff(path: Path, arr: np.ndarray, crs, transform, nodata=None, dtype="float32") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        arr_w = arr[None, ...]
    else:
        arr_w = arr

    profile = {
        "driver": "GTiff",
        "height": arr_w.shape[1],
        "width": arr_w.shape[2],
        "count": arr_w.shape[0],
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "DEFLATE",
        "predictor": 2 if dtype in ("float32", "float64") else 1,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr_w.astype(dtype))

# -------------------------
# Pipeline
# -------------------------
def build_weighted_shared_date_composite(
    out_dir: Path,
    bbox_wesn: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    *,
    max_cloud: float = 20.0,
    scene_limit: int = 10,
    preferred_months: Optional[List[int]] = None,
    shared_date_mode: str = "strict",
    stac_url: str = DEFAULT_STAC_URL,
    collection: str = DEFAULT_COLLECTION,
    stac_limit: int = 200,
    stac_page_limit: Optional[int] = None,  # backward-compat alias
    stac_max_items: int = 5000,
    stac_chunk_months: int = 0,
    cache_dir: Optional[Path] = None,
    download_workers: int = 8,
    scl_dilate: int = 1,
    harmonize: bool = True,
    harmonize_max_abs_offset: float = 0.03,
    deepwater_nir_max: float = 0.03,
    deepwater_bright_max: float = 0.15,
    # Pixel-level quality gating (optional / region-tunable)
    b02_thresh: float = 0.25,
    allow_bright_shallow_pixels: bool = False,
    bright_shallow_nir_max: float = 0.03,
    apply_gl_turbidity_reject: bool = False,
    gl_nir_max: float = 0.12,
    gl_nir_green_ratio_max: float = 0.60,
    gl_red_max: float = 0.08,
    feather_dist_cap: int = 300,
    min_scene_valid_frac: float = 0.90,
    edge_weight_power: float = 2.0,
    edge_weight_min: float = 0.0,
    # Temporal aggregation: if >0, only use the best K dates (post-QC) when building the final median composite.
    temporal_median_k: int = 0,
    # Single best date mode: use only the highest-scoring date instead of median composite
    single_best_date: bool = False,
    # --- Coastline-aware date QC & outlier replacement ---
    coastline_mask_path: str = "",
    coastline_mask_water_value: int = 0,
    coastline_mask_invert: bool = False,
    coastline_erode_px: int = 2,
    date_qc_enable: bool = True,
    date_outlier_z: float = 3.5,
    date_score_weight_valid: float = 2.0,
    date_score_weight_glint: float = 1.0,
    date_score_weight_deepwater: float = 1.0,
    clean_cache: bool = False,
    cache_strict: bool = False,
    cache_code_strict: bool = False,
    cache_ignore_code: bool = True,
    **kwargs,
) -> Dict[str, str]:
    """
    Updated version with:
      1) Explicit coastline-mask logging + statistics
      2) Hard-fail if coastline_mask_path is provided but missing/unreadable
      3) Ensures outlier metrics exist by computing coastline-aware metrics via _compute_date_metrics()
         and merging into the score_date_quality() metrics dict (met.update(...)).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if stac_page_limit is not None:
        stac_limit = int(stac_page_limit)

    expected = {
        "B02": out_dir / "B02_10m.tif",
        "B03": out_dir / "B03_10m.tif",
        "B04": out_dir / "B04_10m.tif",
        "B08": out_dir / "B08_10m.tif",
        "SCL": out_dir / "SCL_10m.tif",
        "RGB": out_dir / "RGB_10m.tif",
        # Single-best-date products (same AOI grid + masking/harmonization, but no temporal aggregation)
        "B02_BEST_DATE": out_dir / "B02_10m_best_date.tif",
        "B03_BEST_DATE": out_dir / "B03_10m_best_date.tif",
        "B04_BEST_DATE": out_dir / "B04_10m_best_date.tif",
        "B08_BEST_DATE": out_dir / "B08_10m_best_date.tif",
        "SCL_BEST_DATE": out_dir / "SCL_10m_best_date.tif",
        "RGB_BEST_DATE": out_dir / "RGB_10m_best_date.tif",
        "BRIGHTNESS": out_dir / "BRIGHTNESS_10m.tif",
        "CLEAR_WATER": out_dir / "CLEAR_WATER_MASK_10m.tif",
        "BRIGHTNESS_BEST_DATE": out_dir / "BRIGHTNESS_10m_best_date.tif",
        "CLEAR_WATER_BEST_DATE": out_dir / "CLEAR_WATER_MASK_10m_best_date.tif",
    }

    # ---------------------------------------------------------------------
    # Exact-match cache validation (ALL output-affecting parameters)
    # ---------------------------------------------------------------------
    meta = _read_composite_meta(out_dir) or {}
    outputs_exist = all(p.exists() and p.stat().st_size > 0 for p in expected.values())


    # -----------------------------------------------------------
    # Glint-correction parameters (optional; passed via **kwargs)
    # -----------------------------------------------------------
    # These are used by the optional Hedley-style glint correction block.
    # They are *also* included in the cache fingerprint so changing them invalidates cache.
    glint_correct = bool(kwargs.get("glint_correct", False))
    glint_clip_min = float(kwargs.get("glint_clip_min", 1e-6))
    glint_deepwater_b02_max = float(kwargs.get("glint_deepwater_b02_max", 0.2))
    glint_nir_band = str(kwargs.get("glint_nir_band", "B08")).strip()
    glint_vis_bands = kwargs.get("glint_vis_bands", ["B02", "B03", "B04"])
    if isinstance(glint_vis_bands, str):
        glint_vis_bands = [
            b.strip()
            for b in glint_vis_bands.replace(";", ",").split(",")
            if b.strip()
        ]
    else:
        glint_vis_bands = (
            list(glint_vis_bands)
            if glint_vis_bands is not None
            else ["B02", "B03", "B04"]
        )
    glint_nir_min_percentile = float(kwargs.get("glint_nir_min_percentile", 1.0))
    glint_min_samples = int(kwargs.get("glint_min_samples", 5000))
    glint_max_samples = int(kwargs.get("glint_max_samples", 2000000))

    # Build the full parameter dict (exclude only performance-only knobs unless you want them to invalidate cache)
    params_fp = _build_s2_params_fingerprint_dict(
        bbox_wesn,
        start_date,
        end_date,
        max_cloud=float(max_cloud),
        scene_limit=int(scene_limit),
        preferred_months=list(preferred_months) if preferred_months is not None else None,
        shared_date_mode=str(shared_date_mode),
        stac_url=str(stac_url),
        collection=str(collection),
        stac_limit=int(stac_limit),
        stac_max_items=int(stac_max_items),
        stac_chunk_months=int(stac_chunk_months),
        scl_dilate=int(scl_dilate),
        harmonize=bool(harmonize),
        harmonize_max_abs_offset=float(harmonize_max_abs_offset),
        deepwater_nir_max=float(deepwater_nir_max),
        deepwater_bright_max=float(deepwater_bright_max),
        glint_correct=bool(glint_correct),
        glint_nir_band=str(glint_nir_band),
        glint_vis_bands=list(glint_vis_bands) if glint_vis_bands is not None else ["B02","B03","B04"],
        glint_nir_min_percentile=float(glint_nir_min_percentile),
        glint_deepwater_b02_max=float(glint_deepwater_b02_max),
        glint_min_samples=int(glint_min_samples),
        glint_max_samples=int(glint_max_samples),
        glint_clip_min=float(glint_clip_min),
        b02_thresh=float(b02_thresh),
        allow_bright_shallow_pixels=bool(allow_bright_shallow_pixels),
        bright_shallow_nir_max=float(bright_shallow_nir_max),
        apply_gl_turbidity_reject=bool(apply_gl_turbidity_reject),
        gl_nir_max=float(gl_nir_max),
        gl_nir_green_ratio_max=float(gl_nir_green_ratio_max),
        gl_red_max=float(gl_red_max),
        feather_dist_cap=int(feather_dist_cap),
        min_scene_valid_frac=float(min_scene_valid_frac),
        edge_weight_power=float(edge_weight_power),
        edge_weight_min=float(edge_weight_min),
        temporal_median_k=int(temporal_median_k),
        single_best_date=bool(single_best_date),
        coastline_mask_water_value=int(coastline_mask_water_value),
        coastline_mask_invert=bool(coastline_mask_invert),
        coastline_erode_px=int(coastline_erode_px),
        date_qc_enable=bool(date_qc_enable),
        date_outlier_z=float(date_outlier_z),
        date_score_weight_valid=float(date_score_weight_valid),
        date_score_weight_glint=float(date_score_weight_glint),
        date_score_weight_deepwater=float(date_score_weight_deepwater),
        clean_cache=bool(clean_cache),
    )

    # Inputs that affect outputs (files, masks). Prefer strict content hashes only if requested.
    inputs_fp: Dict[str, object] = {}
    if _CACHE_UTILS_AVAILABLE and coastline_mask_path:
        try:
            if Path(coastline_mask_path).exists():
                inputs_fp["coastline_mask"] = fingerprint_file(Path(coastline_mask_path), strict=bool(cache_strict))
        except Exception:
            log.debug("ignored", exc_info=True)

    want_key = _s2_exact_cache_key(
        bbox_wesn=bbox_wesn,
        start_date=start_date,
        end_date=end_date,
        params=params_fp,
        inputs=inputs_fp,
        cache_code_strict=bool(cache_code_strict),
        cache_ignore_code=bool(cache_ignore_code),
    )
    have_key = str(meta.get("cache_key", ""))

    if outputs_exist and have_key == want_key:
        log.info("CACHE HIT (exact params): %s", out_dir)
        ret = {k: str(v) for k, v in expected.items()}
        try:
            from log_report import raster_quickstats, mask_fraction
            rep = {"expected": {k: str(v) for k, v in expected.items()}, "rasters": {}, "masks": {}}
            for k0 in ("B02", "B03", "B04", "B08", "BRIGHTNESS"):
                p0 = expected.get(k0)
                if p0 and Path(p0).exists():
                    rep["rasters"][k0] = raster_quickstats(str(p0))
            if expected.get("CLEAR_WATER") and Path(expected["CLEAR_WATER"]).exists():
                rep["masks"]["CLEAR_WATER"] = mask_fraction(str(expected["CLEAR_WATER"]))
            if expected.get("SCL") and Path(expected["SCL"]).exists():
                rep["masks"]["SCL_water"] = mask_fraction(str(expected["SCL"]), true_values=[6])
            
            # Optical feasibility diagnostics (SDB physics sanity checks)
            try:
                import numpy as _np
                import rasterio as _rio
                _eps = 1e-6
                _mask = None
                _mask_path = None
                _mask_true_values = None
                _mask_threshold = None
                if expected.get("CLEAR_WATER") and Path(expected["CLEAR_WATER"]).exists():
                    _mask_path = str(expected["CLEAR_WATER"]); _mask_threshold = 1.0
                elif expected.get("SCL") and Path(expected["SCL"]).exists():
                    _mask_path = str(expected["SCL"]); _mask_true_values = [6]

                if _mask_path:
                    with _rio.open(_mask_path) as _mds:
                        _m = _mds.read(1)
                        if _mds.nodata is not None:
                            _m = _np.where(_m == _mds.nodata, _np.nan, _m)
                    if _mask_true_values is not None:
                        _mask = _np.isfinite(_m) & _np.isin(_m.astype(_np.int32), _mask_true_values)
                    elif _mask_threshold is not None:
                        _mask = _np.isfinite(_m) & (_m.astype(_np.float32) >= float(_mask_threshold))
                    else:
                        _mask = _np.isfinite(_m) & (_m != 0)


                od = {}
                if expected.get("B02") and Path(expected["B02"]).exists():
                    _v02 = _sample_band_masked(expected["B02"], mask=_mask, np_mod=_np, rio_mod=_rio)
                    od["n_sample_b02"] = int(_v02.size)
                    if _v02.size:
                        od["L_inf_proxy_b02_p01"] = float(_np.percentile(_v02, 1))
                        _vlog = _np.log(_np.maximum(_v02, _eps))
                        od["EDR_log_b02_p95_p05"] = float(_np.percentile(_vlog, 95) - _np.percentile(_vlog, 5))
                if expected.get("B03") and Path(expected["B03"]).exists():
                    _v03 = _sample_band_masked(expected["B03"], mask=_mask, np_mod=_np, rio_mod=_rio)
                    od["n_sample_b03"] = int(_v03.size)
                    if _v03.size:
                        od["L_inf_proxy_b03_p01"] = float(_np.percentile(_v03, 1))
                        _vlog = _np.log(_np.maximum(_v03, _eps))
                        od["EDR_log_b03_p95_p05"] = float(_np.percentile(_vlog, 95) - _np.percentile(_vlog, 5))
                _edrs = [od.get("EDR_log_b02_p95_p05"), od.get("EDR_log_b03_p95_p05")]
                _edrs = [float(x) for x in _edrs if isinstance(x, (int, float))]
                if _edrs:
                    _best = max(_edrs)
                    od["EDR_best"] = float(_best)
                    od["signal_feasibility"] = "good" if _best >= 1.0 else ("marginal" if _best >= 0.7 else "poor")
                else:
                    od["signal_feasibility"] = "unknown"
                rep["optical_diagnostics"] = od
            except Exception:
                log.debug("ignored", exc_info=True)

            ret["_report"] = rep
        except Exception:
            log.debug("ignored", exc_info=True)
        return ret

    # If outputs exist but key differs, purge and rebuild
    if outputs_exist and have_key and have_key != want_key:
        # Debug: show what's different
        # Note: metadata stores as 'params_fp' and 'inputs_fp'
        stored_params = meta.get("params_fp", meta.get("params", {})) or {}
        stored_inputs = meta.get("inputs_fp", meta.get("inputs", {})) or {}
        stored_code_fp = meta.get("code_fingerprint", "")
        
        diff_info = []
        
        # Only report differences for params that exist in BOTH stored and computed
        # (missing params in stored metadata means old cache format - don't spam the log)
        if stored_params:
            all_keys = set(params_fp.keys()) | set(stored_params.keys())
            for k in sorted(all_keys):
                v_new = params_fp.get(k)
                v_old = stored_params.get(k)
                if v_new != v_old:
                    diff_info.append(f"  param '{k}': stored={v_old!r} vs computed={v_new!r}")
        else:
            diff_info.append("  (stored metadata has no params - old cache format, rebuild required)")
        
        # Check inputs differences
        if stored_inputs:
            all_input_keys = set(inputs_fp.keys()) | set(stored_inputs.keys())
            for k in sorted(all_input_keys):
                v_new = inputs_fp.get(k)
                v_old = stored_inputs.get(k)
                if v_new != v_old:
                    diff_info.append(f"  input '{k}': stored={v_old!r} vs computed={v_new!r}")
        
        # Check code fingerprint
        if stored_code_fp and not cache_ignore_code:
            diff_info.append(f"  code_fp: stored={stored_code_fp!r} (new code ignores this)")
        
        log.warning(
            "[S2] Cache mismatch in %s: have_key=%s want_key=%s "
            "(full keys in meta). Rebuilding (purging old outputs).",
            out_dir, str(have_key)[:12], str(want_key)[:12],
        )
        if diff_info:
            log.warning("Cache key differences:\n" + "\n".join(diff_info[:15]))  # Limit to first 15
        
        _purge_expected_outputs(expected)


    lock = out_dir / ".s2_running"
    acquire_run_lock(lock, stale_hours=6.0)

    try:
        w, e, s, n = bbox_wesn
        aoi_poly_ll = box(w, s, e, n)
        pref = set(preferred_months or [])

        items = stac_search(
            stac_url=stac_url,
            collection=collection,
            bbox_wesn=bbox_wesn,
            start_date=start_date,
            end_date=end_date,
            chunk_months=stac_chunk_months,
            limit=stac_limit,
            max_items=stac_max_items,
        )

        # Filter & build scenes
        items_by_tile: Dict[str, List[Scene]] = defaultdict(list)
        dropped_cloud = dropped_dt = dropped_geom = dropped_tile = 0

        for it in items:
            dt = extract_datetime(it)
            if not dt:
                dropped_dt += 1
                continue

            cloud = extract_cloud(it)
            if cloud > max_cloud:
                dropped_cloud += 1
                continue

            geom = it.get("geometry", None)
            if not geom:
                dropped_geom += 1
                continue

            tile = extract_tile(it)
            if tile == "UNK":
                dropped_tile += 1
                continue

            # Extract sun/view angles for physics-based SDB
            sun_z, sun_a = extract_sun_angles(it)
            view_z, view_a = extract_view_angles(it)

            sc = Scene(
                id=it.get("id", ""),
                tile=tile,
                dt=dt,
                cloud=cloud,
                orbit=extract_orbit(it),
                assets=it.get("assets", {}) or {},
                geometry=geom,
                sun_zenith=sun_z,
                sun_azimuth=sun_a,
                view_zenith=view_z,
                view_azimuth=view_a,
            )
            items_by_tile[tile].append(sc)

        tiles = sorted(items_by_tile.keys())
        kept = sum(len(v) for v in items_by_tile.values())
        log.info(
            "[S2] Candidate build done: tiles=%s kept_items=%s  dropped_cloud=%s dropped_dt=%s dropped_geom=%s dropped_tile=%s",
            len(tiles), kept, dropped_cloud, dropped_dt, dropped_geom, dropped_tile,
        )

        if not tiles:
            raise RuntimeError("[S2] No usable scenes after filtering.")

        # best scene per tile per DATE_KEY (date only)
        per_tile_dt_best: Dict[str, Dict[str, Scene]] = {}
        for t, scenes in items_by_tile.items():
            best: Dict[str, Scene] = {}
            for sc in scenes:
                date_key = sc.dt[:10]
                cur = best.get(date_key)
                if cur is None or sc.cloud < cur.cloud:
                    best[date_key] = sc
            per_tile_dt_best[t] = best

        # stable tile weights
        weights = compute_tile_weights(items_by_tile, aoi_poly_ll)
        log.info("")
        log.info("Tile weights (AOI intersection fraction):")
        for t in sorted(weights.keys()):
            log.info("  - %s: %.1f%%", t, weights[t]*100)
        log.info("")

        
        # shared dates (with strict→relaxed fallback)
        MIN_STRICT_DATES = 2  # minimum acceptable strict shared dates before fallback
        dt_sets = [set(per_tile_dt_best[t].keys()) for t in tiles]
        req_mode = (shared_date_mode or "strict").lower().strip()

        shared_mode_used = req_mode
        if req_mode == "strict":
            shared_strict = set.intersection(*dt_sets) if dt_sets else set()
            if len(shared_strict) < MIN_STRICT_DATES:
                log.warning(
                    "[S2] STRICT shared-date selection produced only %d dates "
                    "(<%d). Falling back to RELAXED (union) shared-date selection.",
                    len(shared_strict), MIN_STRICT_DATES,
                )
                shared = set.union(*dt_sets) if dt_sets else set()
                shared_mode_used = "relaxed"
                mode = "union"
            else:
                shared = shared_strict
                mode = "strict"
        elif req_mode == "union":
            shared = set.union(*dt_sets) if dt_sets else set()
            mode = "union"
        else:
            raise ValueError("--shared-date-mode must be strict or union")

        if not shared:
            raise RuntimeError(
                f"[S2] No common dates found (requested shared-date-mode={shared_date_mode}; used={shared_mode_used})."
            )

        log.info(
            "[S2] Shared-date selection: requested=%s used=%s candidates=%s",
            req_mode, shared_mode_used, len(shared),
        )
        # rank by weighted cloud (month filtering at ranking stage)
        ranked = []
        final_dropped_month = 0

        if pref:
            log.info("Filtering candidates by preferred months: %s", sorted(list(pref)))
        else:
            log.info("No preferred month filtering active (all months accepted).")

        for date_key in sorted(shared):
            if pref and month_from_iso(date_key) not in pref:
                final_dropped_month += 1
                continue

            present = [t for t in tiles if date_key in per_tile_dt_best[t]]
            if mode == "strict" and len(present) != len(tiles):
                continue

            num = 0.0
            den = 0.0
            for t in present:
                c = per_tile_dt_best[t][date_key].cloud
                wt = weights.get(t, 0.0)
                num += wt * c
                den += wt
            score = num / den if den > 0 else 100.0
            ranked.append((score, date_key, present))

        ranked.sort(key=lambda x: x[0])

        log.info("Final Filter: Dropped %s dates due to preferred_months setting.", final_dropped_month)
        if not ranked:
            raise RuntimeError("[S2] No usable dates remain after applying all filters (cloud, shared-date, preferred-months).")

        log.info("Top DATES ranked by WEIGHTED cloud (shared-date-mode=%s)", mode)
        log.info("")
        log.info(" idx  w_cloud%%  tiles_present  DATE_KEY")
        log.info(" ---  --------  -------------  ------------------------")
        for i, (score, dt, present) in enumerate(ranked[:min(20, len(ranked))], 1):
            log.info(" %3d  %8.2f  %-13s  %s", i, score, ','.join(present), dt)
        log.info("")

        # Setup cache dir
        base_cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        cache_subdir_name = f"S2_{_norm_bbox_key(bbox_wesn, ndp=3)}_{start_date}_{end_date}"
        sentinel_cache_dir = base_cache_dir / "sentinel2" / cache_subdir_name
        sentinel_cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("Sentinel-2 downloads will be cached in: %s", sentinel_cache_dir)

        # Pre-flight: if user provided coastline_mask_path, it must exist (no silent fallback)
        if date_qc_enable and coastline_mask_path:
            cm = Path(coastline_mask_path)
            if not cm.exists() or cm.stat().st_size == 0:
                raise RuntimeError(f"[S2] Coastline mask path does not exist or is empty: {coastline_mask_path}")

        stacks = {b: [] for b in ["B02", "B03", "B04", "B08", "SCL"]}
        dst_grid = None
        N = int(scene_limit)

        def _build_date_mosaic(date_key: str, present_tiles: List[str]) -> Optional[Dict[str, np.ndarray]]:
            nonlocal dst_grid
            per_tile_arrays: Dict[str, Dict[str, np.ndarray]] = {}
            grid_ready = dst_grid is not None
            ref_path_for_grid = None

            for t in present_tiles:
                sc = per_tile_dt_best[t][date_key]
                local = resolve_and_download_scene(sc, cache_dir=sentinel_cache_dir, max_workers=download_workers)
                if local is None:
                    log.warning("Date %s skipped: missing assets for tile %s.", date_key, t)
                    return None

                if not grid_ready and ref_path_for_grid is None:
                    ref_path_for_grid = local["B02"]

                if not grid_ready and ref_path_for_grid is not None:
                    dst_crs, dst_transform, dst_w, dst_h = _aoi_grid_from_ref(ref_path_for_grid, aoi_poly_ll, res_m=10.0)
                    dst_grid = (dst_crs, dst_transform, dst_w, dst_h)
                    grid_ready = True

                dst_crs, dst_transform, dst_w, dst_h = dst_grid

                # Read + warp SCL
                scl = warp_to_grid(local["SCL"], dst_crs, dst_transform, dst_w, dst_h,
                                   resampling=Resampling.nearest, dtype=np.float32)
                scl_int = _safe_rint_to_uint8(scl, fill=0)
                vmask = scl_valid_mask(scl_int, dilate=scl_dilate)

                raw: Dict[str, np.ndarray] = {"SCL": scl.astype(np.float32)}

                # Read + warp reflectance bands
                for b in ["B02", "B03", "B04", "B08"]:
                    arr = warp_to_grid(local[b], dst_crs, dst_transform, dst_w, dst_h,
                                       resampling=Resampling.bilinear, dtype=np.float32)
                    finite = np.isfinite(arr)
                    if finite.any():
                        p99 = float(np.nanpercentile(arr[finite], 99))
                        if p99 > 2.0:
                            arr = arr / 10000.0
                    raw[b] = arr.astype(np.float32)

                # Per-tile pixel gating: keep non-water pixels (land) for visualization products (RGB),
                # while applying aggressive bright/turbidity gates only over SCL=Water pixels.
                base_valid = vmask & np.isfinite(raw["B02"]) & np.isfinite(raw["B03"]) & np.isfinite(raw["B04"]) & np.isfinite(raw["B08"])

                # Sentinel-2 SCL: 6 == Water
                scl_int_local = _safe_rint_to_uint8(raw["SCL"], fill=0)
                is_water = (scl_int_local == 6)

                # Start from "good SCL + finite" everywhere (keeps land reflectance)
                valid = base_valid.copy()

                # Apply water-only reflectance/turbidity gates (drives SDB predictability)
                if allow_bright_shallow_pixels:
                    water_valid = base_valid & is_water & ((raw["B02"] < b02_thresh) | (raw["B08"] < bright_shallow_nir_max))
                else:
                    water_valid = base_valid & is_water & (raw["B02"] < b02_thresh)

                # Keep land/non-water as base_valid; replace water pixels with water_valid
                valid = (base_valid & (~is_water)) | water_valid

                if apply_gl_turbidity_reject:
                    ratio = np.where(raw["B03"] > 0, raw["B08"] / raw["B03"], np.nan)
                    water_valid = water_valid & (raw["B08"] < gl_nir_max) & (ratio < gl_nir_green_ratio_max) & (raw["B04"] < gl_red_max)
                    valid = (base_valid & (~is_water)) | water_valid

                for b in ["B02", "B03", "B04", "B08"]:
                    a = raw[b].copy()
                    a[~valid] = np.nan
                    raw[b] = a.astype(np.float32)

                scl2 = raw["SCL"].copy()
                scl2[~base_valid] = np.nan
                raw["SCL"] = scl2.astype(np.float32)

                per_tile_arrays[t] = raw

            ref_tile = max(present_tiles, key=lambda tt: weights.get(tt, 0.0))
            if harmonize and len(present_tiles) > 1:
                harmonize_offsets_to_reference(
                    per_tile_arrays,
                    ref_tile=ref_tile,
                    max_abs_offset=harmonize_max_abs_offset,
                    deepwater_nir_max=deepwater_nir_max,
                    deepwater_bright_max=deepwater_bright_max,
                    b02_thresh=b02_thresh,
                )

            return mosaic_tiles(
                per_tile_arrays,
                weights=weights,
                feather=True,
                dist_cap=int(feather_dist_cap),
                use_sqrt_weight=True,
                edge_weight_power=float(edge_weight_power),
                edge_weight_min=float(edge_weight_min),
        temporal_median_k=int(temporal_median_k),
            )

        date_mosaics: Dict[str, Dict[str, np.ndarray]] = {}
        date_scores: Dict[str, float] = {}
        date_metrics: Dict[str, Dict[str, float]] = {}

        # Load these from your module (or define them earlier in your script)
        # scl_bad_set should include your "bad" SCL classes.
        scl_bad_set = set(getattr(globals(), "SCL_BAD", {3, 8, 9, 10, 11}))  # safe fallback

        # Phase 1: build initial pool and score
        initial = ranked[:N]
        log.info("Phase 1: building QC metrics for initial %s lowest-cloud dates...", len(initial))

        water_mask = None  # True where "target water" pixels are
        for cloud_score, date_key, present_tiles in initial:
            log.info("Building date mosaic for %s (cloud=%.2f%%) ...", date_key, cloud_score)
            mos = _build_date_mosaic(date_key, present_tiles)
            if mos is None:
                continue
            date_mosaics[date_key] = mos

            # Load water mask once AOI grid exists
            if water_mask is None:
                if date_qc_enable and coastline_mask_path:
                    dst_crs, dst_transform, dst_w, dst_h = dst_grid
                    water_mask = load_waffles_coast_mask_to_grid(
                        coastline_mask_path,
                        dst_crs, dst_transform, dst_w, dst_h,
                        water_value=coastline_mask_water_value,
                        invert=coastline_mask_invert,
                        erode_px=coastline_erode_px,
                    )
                    # --- NEW: log stats so it's undeniable it's being used ---
                    cnt = int(np.count_nonzero(water_mask))
                    log.info("Loaded coastline mask: %s", coastline_mask_path)
                    log.info("Coast mask semantics: water_value=%s invert=%s erode_px=%d",
                             coastline_mask_water_value, coastline_mask_invert, int(coastline_erode_px))
                    log.info("Target-water pixels: %d (%.2f%% of grid)",
                             cnt, 100.0 * float(cnt) / float(water_mask.size))
                else:
                    water_mask = np.ones(mos["B02"].shape, dtype=bool)
                    log.info("Coastline masking disabled; using AOI-only QC footprint.")

            # Score quality (uses water_mask)
            if date_qc_enable:
                s, met = score_date_quality(
                    mos, water_mask,
                    b02_thresh=b02_thresh,
                    deepwater_nir_max=deepwater_nir_max,
                    deepwater_bright_max=deepwater_bright_max,
                    allow_bright_shallow_pixels=allow_bright_shallow_pixels,
                    bright_shallow_nir_max=bright_shallow_nir_max,
                    weight_valid=date_score_weight_valid,
                    weight_glint=date_score_weight_glint,
                    weight_deepwater=date_score_weight_deepwater,
                )

                # --- NEW: ensure outlier metrics exist by computing coastline-aware metrics here ---
                scl_int_full = _safe_rint_to_uint8(mos["SCL"], fill=0)
                vmask_full = scl_valid_mask(scl_int_full, dilate=scl_dilate)
                cm = _compute_date_metrics(
                    mos["B02"], mos["B03"], mos["B04"], mos["B08"],
                    scl_int_full,
                    target_water=water_mask,
                    target_water_inner=water_mask,  # loader already eroded; if you track inner separately, pass it here
                    scl_bad=scl_bad_set,
                    vmask=vmask_full,
                )
                met.update(cm)
            else:
                s, met = float(cloud_score), {
                    "water_pixels": 0.0, "valid_frac": 0.0, "glint_frac": 0.0, "deepwater_frac": 0.0,
                    "bright_shallow_frac": 0.0,
                    "coverage": 0.0, "bad_frac": 1.0, "nir_med": float("inf"), "red_med": float("inf")
                }

            met["cloud_w"] = float(cloud_score)
            met["score"] = float(s)
            date_scores[date_key] = float(s)
            date_metrics[date_key] = met

        # Outliers among initial
        outliers: List[str] = []
        triggers_map: Dict[str, List[str]] = {}
        if date_qc_enable and len(date_scores) >= max(3, min(N, len(initial))):
            keys0 = list(date_scores.keys())
            triggers_map = {k: [] for k in keys0}

            def _flag_outliers(name: str, values: List[float], zthr: float, bad_tail: str = "both"):
                z = _robust_z_map(keys0, values)
                for k in keys0:
                    zk = float(z.get(k, 0.0))
                    if not np.isfinite(zk):
                        continue
                    if bad_tail == "high" and zk > zthr:
                        triggers_map[k].append(f"{name}(z={zk:.2f})")
                    elif bad_tail == "low" and zk < -zthr:
                        triggers_map[k].append(f"{name}(z={zk:.2f})")
                    elif bad_tail == "both" and abs(zk) > zthr:
                        triggers_map[k].append(f"{name}(z={zk:.2f})")

            # score: high worse
            _flag_outliers("score", [date_scores.get(k, 1e9) for k in keys0], float(date_outlier_z), bad_tail="high")
            # bad_frac: high worse
            _flag_outliers("bad_frac", [date_metrics.get(k, {}).get("bad_frac", 1.0) for k in keys0], float(date_outlier_z), bad_tail="high")
            # coverage: low worse (use direct coverage with low-tail)
            _flag_outliers("coverage", [date_metrics.get(k, {}).get("coverage", 0.0) for k in keys0], float(date_outlier_z), bad_tail="low")
            # nir_med/red_med: both tails can indicate odd radiometry
            _flag_outliers("nir_med", [date_metrics.get(k, {}).get("nir_med", np.nan) for k in keys0], float(date_outlier_z), bad_tail="both")
            _flag_outliers("red_med", [date_metrics.get(k, {}).get("red_med", np.nan) for k in keys0], float(date_outlier_z), bad_tail="both")

            outliers = [k for k in keys0 if triggers_map.get(k)]
            if outliers:
                log.info("Initial outliers detected (%d): %s", len(outliers), ', '.join(outliers))
                for k in outliers:
                    log.info("- %s triggers: %s", k, "; ".join(triggers_map[k]))
            else:
                log.info("No outliers detected in initial pool.")

        # Phase 2: download 2x(outliers) additional candidates and score them
        extra_n = 2 * len(outliers)
        if date_qc_enable and extra_n > 0:
            log.info("Phase 2: downloading %s additional candidate dates for outlier replacement...", extra_n)
            extras = []
            for cloud_score, date_key, present_tiles in ranked[N:]:
                if date_key in date_mosaics:
                    continue
                extras.append((cloud_score, date_key, present_tiles))
                if len(extras) >= extra_n:
                    break

            for cloud_score, date_key, present_tiles in extras:
                log.info("Building extra date mosaic for %s (cloud=%.2f%%) ...", date_key, cloud_score)
                mos = _build_date_mosaic(date_key, present_tiles)
                if mos is None:
                    continue
                date_mosaics[date_key] = mos

                if water_mask is None:
                    water_mask = np.ones(mos["B02"].shape, dtype=bool)

                s, met = score_date_quality(
                    mos, water_mask,
                    b02_thresh=b02_thresh,
                    deepwater_nir_max=deepwater_nir_max,
                    deepwater_bright_max=deepwater_bright_max,
                    allow_bright_shallow_pixels=allow_bright_shallow_pixels,
                    bright_shallow_nir_max=bright_shallow_nir_max,
                    weight_valid=date_score_weight_valid,
                    weight_glint=date_score_weight_glint,
                    weight_deepwater=date_score_weight_deepwater,
                )

                scl_int_full = _safe_rint_to_uint8(mos["SCL"], fill=0)
                vmask_full = scl_valid_mask(scl_int_full, dilate=scl_dilate)
                cm = _compute_date_metrics(
                    mos["B02"], mos["B03"], mos["B04"], mos["B08"],
                    scl_int_full,
                    target_water=water_mask,
                    target_water_inner=water_mask,
                    scl_bad=scl_bad_set,
                    vmask=vmask_full,
                )
                met.update(cm)

                met["cloud_w"] = float(cloud_score)
                met["score"] = float(s)
                date_scores[date_key] = float(s)
                date_metrics[date_key] = met

        # Pick best N by score (lower is better)
        picked = sorted(date_scores.items(), key=lambda kv: kv[1])[:N]
        selected_dates = [k for k, _ in picked if k in date_mosaics]
        used = len(selected_dates)

        if used:
            log.info("Selected %d dates after coastline-aware QC:", used)
            for i, k in enumerate(selected_dates, 1):
                met = date_metrics.get(k, {})
                log.info(
                    "  %2d) %s  score=%.4f  cov=%.3f  bad=%.3f  nir_med=%.4f  red_med=%.4f  valid=%.3f  glint=%.3f  deep=%.3f  cloud=%.2f%%",
                    i, k,
                    float(met.get("score", np.nan)),
                    float(met.get("coverage", np.nan)),
                    float(met.get("bad_frac", np.nan)),
                    float(met.get("nir_med", np.nan)),
                    float(met.get("red_med", np.nan)),
                    float(met.get("valid_frac", np.nan)),
                    float(met.get("glint_frac", np.nan)),
                    float(met.get("deepwater_frac", np.nan)),
                    float(met.get("cloud_w", np.nan)),
                )
        else:
            log.warning("No dates were successfully mosaicked/scored.")

        stack_dates = list(selected_dates)
        if single_best_date and selected_dates:
            stack_dates = [selected_dates[0]]
            best_score = date_metrics.get(selected_dates[0], {}).get('score', 'N/A')
            log.info("SINGLE BEST DATE MODE: Using only %s (score=%.3f)",
                     selected_dates[0], float(best_score) if best_score != 'N/A' else 0.0)
        elif int(temporal_median_k) > 0:
            stack_dates = stack_dates[: int(temporal_median_k)]
            log.info("Temporal median using best K dates: K=%s of %s", int(temporal_median_k), len(selected_dates))
        for k in stack_dates:
            mos = date_mosaics[k]
            for b in ["B02", "B03", "B04", "B08", "SCL"]:
                stacks[b].append(mos[b])

        # Write QC report
        try:
            # Collect sun/view angles from selected scenes
            sun_angles = {"zenith": [], "azimuth": []}
            view_angles = {"zenith": [], "azimuth": []}
            for date_key in stack_dates:
                for tile, best in per_tile_dt_best.items():
                    sc = best.get(date_key)
                    if sc:
                        if sc.sun_zenith is not None:
                            sun_angles["zenith"].append(sc.sun_zenith)
                        if sc.sun_azimuth is not None:
                            sun_angles["azimuth"].append(sc.sun_azimuth)
                        if sc.view_zenith is not None:
                            view_angles["zenith"].append(sc.view_zenith)
                        if sc.view_azimuth is not None:
                            view_angles["azimuth"].append(sc.view_azimuth)
            
            # Compute mean angles for physics
            mean_sun_zenith = float(np.mean(sun_angles["zenith"])) if sun_angles["zenith"] else None
            mean_sun_azimuth = float(np.mean(sun_angles["azimuth"])) if sun_angles["azimuth"] else None
            mean_view_zenith = float(np.mean(view_angles["zenith"])) if view_angles["zenith"] else None
            mean_view_azimuth = float(np.mean(view_angles["azimuth"])) if view_angles["azimuth"] else None

            qc_report = {
                "best_date": selected_dates[0] if selected_dates else None,
                "selected_dates": selected_dates,
                "single_best_date_mode": single_best_date,
                "dates_used_in_composite": stack_dates,
                "date_metrics": date_metrics,
                "date_outliers_initial": outliers,
                "date_outlier_triggers": triggers_map,
                "shared_date_mode_requested": str(shared_date_mode),
                "shared_date_mode_used": str(shared_mode_used),
                "coastline_mask_path": coastline_mask_path,
                "coastline_mask_water_value": int(coastline_mask_water_value),
                "coastline_mask_invert": bool(coastline_mask_invert),
                "coastline_erode_px": int(coastline_erode_px),
                # Sun/view angles for physics-based SDB (Kim et al. 2024)
                "mean_sun_zenith": mean_sun_zenith,
                "mean_sun_azimuth": mean_sun_azimuth,
                "mean_view_zenith": mean_view_zenith,
                "mean_view_azimuth": mean_view_azimuth,
            }
            (out_dir / "S2_DATE_QC.json").write_text(json.dumps(qc_report, indent=2))
        except Exception:
            log.debug("ignored", exc_info=True)

        if used == 0:
            raise RuntimeError("[S2] Could not build any mosaics with publicly downloadable assets. Try a different --stac-url or relax filters.")

        dst_crs, dst_transform, dst_w, dst_h = dst_grid

        final: Dict[str, np.ndarray] = {}
        for b in ["B02", "B03", "B04", "B08"]:
            final[b] = _nanmedian_stack_no_warn(np.stack(stacks[b], axis=0))
        scl_final = _nanmedian_stack_no_warn(np.stack(stacks["SCL"], axis=0))
        final["SCL"] = np.where(np.isfinite(scl_final), np.rint(scl_final), np.nan).astype(np.float32)

        # Optional: Hedley-style sun-glint correction (cheap when it helps).
        glint_meta_comp = None
        glint_meta_best = None
        if bool(glint_correct):
            try:
                vis_bands = list(glint_vis_bands) if glint_vis_bands is not None else ["B02", "B03", "B04"]
                vis_bands = [b for b in vis_bands if b in final]
                if glint_nir_band not in final or not vis_bands:
                    glint_meta_comp = {"enabled": True, "status": "skipped", "reason": "missing_bands", "nir_band": str(glint_nir_band), "vis_bands": vis_bands}
                else:
                    vis_dict = {b: final[b] for b in vis_bands}
                    corr, meta_g = _hedley_glint_correct(
                        final[str(glint_nir_band)],
                        vis_dict,
                        scl=final.get("SCL"),
                        scl_bad=SCL_BAD,
                        deepwater_nir_max=float(deepwater_nir_max),
                        deepwater_bright_max=float(deepwater_bright_max),
                        deepwater_b02_max=float(glint_deepwater_b02_max),
                        nir_min_percentile=float(glint_nir_min_percentile),
                        min_samples=int(glint_min_samples),
                        max_samples=int(glint_max_samples),
                        clip_min=float(glint_clip_min),
                        label="composite",
                    )
                    for b, v in corr.items():
                        final[b] = v
                    glint_meta_comp = meta_g
                    if meta_g.get("status") == "applied":
                        log.info("Applied Hedley correction to %s using %s samples (nir_min=%s)",
                                 vis_bands, meta_g.get('n_samples'), meta_g.get('nir_min'))
                    else:
                        log.info("Skipped glint correction: %s", meta_g.get('reason'))
            except Exception as exc:
                glint_meta_comp = {"enabled": True, "status": "skipped", "reason": f"exception: {exc}"}
                log.warning("Glint correction failed; continuing without it: %s", exc)

        brightness = compute_brightness(final["B02"], final["B03"], final["B04"]).astype(np.float32)
        clear = compute_clear_water_mask(final["B08"], brightness).astype(np.uint8)
        rgb = np.stack([final["B04"], final["B03"], final["B02"]], axis=0).astype(np.float32)

        write_geotiff(expected["B02"], final["B02"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")
        write_geotiff(expected["B03"], final["B03"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")
        write_geotiff(expected["B04"], final["B04"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")
        write_geotiff(expected["B08"], final["B08"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")
        write_geotiff(expected["SCL"], final["SCL"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")
        write_geotiff(expected["BRIGHTNESS"], brightness, dst_crs, dst_transform, nodata=np.nan, dtype="float32")
        write_geotiff(expected["CLEAR_WATER"], clear.astype(np.uint8), dst_crs, dst_transform, nodata=255, dtype="uint8")

        expected["RGB"].parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            expected["RGB"], "w",
            driver="GTiff",
            height=rgb.shape[1],
            width=rgb.shape[2],
            count=3,
            dtype="float32",
            crs=dst_crs,
            transform=dst_transform,
            compress="DEFLATE",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            nodata=np.nan,
        ) as dst:
            dst.write(rgb)

        # Also write a "single best date" RGB for diagnostics/visualization (no temporal aggregation).
        try:
            best_date = selected_dates[0] if selected_dates else None
            if best_date and best_date in date_mosaics:
                mos_best = date_mosaics[best_date]
                # Optional: apply the same glint correction to the best-date stack.
                if bool(glint_correct):
                    try:
                        vis_bands = list(glint_vis_bands) if glint_vis_bands is not None else ["B02", "B03", "B04"]
                        vis_bands = [b for b in vis_bands if b in mos_best]
                        if glint_nir_band not in mos_best or not vis_bands:
                            glint_meta_best = {"enabled": True, "status": "skipped", "reason": "missing_bands", "nir_band": str(glint_nir_band), "vis_bands": vis_bands}
                        else:
                            vis_dict = {b: mos_best[b] for b in vis_bands}
                            corr_b, meta_b = _hedley_glint_correct(
                                mos_best[str(glint_nir_band)],
                                vis_dict,
                                scl=mos_best.get("SCL"),
                                scl_bad=SCL_BAD,
                                deepwater_nir_max=float(deepwater_nir_max),
                                deepwater_bright_max=float(deepwater_bright_max),
                                deepwater_b02_max=float(glint_deepwater_b02_max),
                                nir_min_percentile=float(glint_nir_min_percentile),
                                min_samples=int(glint_min_samples),
                                max_samples=int(glint_max_samples),
                                clip_min=float(glint_clip_min),
                                label="best_date",
                            )
                            for b, v in corr_b.items():
                                mos_best[b] = v
                            glint_meta_best = meta_b
                            if meta_b.get("status") == "applied":
                                log.info("Best-date glint correction applied to %s using %s samples",
                                         vis_bands, meta_b.get('n_samples'))
                            else:
                                log.info("Best-date glint correction skipped: %s", meta_b.get('reason'))
                    except Exception as exc3:
                        glint_meta_best = {"enabled": True, "status": "skipped", "reason": f"exception: {exc3}"}
                        log.warning("Best-date glint correction failed; continuing without it: %s", exc3)
                # Write best-date full band stack + masks so downstream can compare
                # (best-date vs temporal composite) in training/validation.
                try:
                    write_geotiff(expected["B02_BEST_DATE"], mos_best["B02"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")
                    write_geotiff(expected["B03_BEST_DATE"], mos_best["B03"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")
                    write_geotiff(expected["B04_BEST_DATE"], mos_best["B04"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")
                    write_geotiff(expected["B08_BEST_DATE"], mos_best["B08"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")
                    write_geotiff(expected["SCL_BEST_DATE"], mos_best["SCL"], dst_crs, dst_transform, nodata=np.nan, dtype="float32")

                    bright_best = compute_brightness(mos_best["B02"], mos_best["B03"], mos_best["B04"]).astype(np.float32)
                    clear_best = compute_clear_water_mask(mos_best["B08"], bright_best).astype(np.uint8)
                    write_geotiff(expected["BRIGHTNESS_BEST_DATE"], bright_best, dst_crs, dst_transform, nodata=np.nan, dtype="float32")
                    write_geotiff(expected["CLEAR_WATER_BEST_DATE"], clear_best.astype(np.uint8), dst_crs, dst_transform, nodata=255, dtype="uint8")
                except Exception as exc2:
                    log.warning("Failed writing best-date band stack/masks: %s", exc2)

                rgb_best = np.stack([mos_best["B04"], mos_best["B03"], mos_best["B02"]], axis=0).astype(np.float32)
                out_best = expected.get("RGB_BEST_DATE")
                if out_best:
                    with rasterio.open(
                        out_best, "w",
                        driver="GTiff",
                        height=rgb_best.shape[1],
                        width=rgb_best.shape[2],
                        count=3,
                        dtype="float32",
                        crs=dst_crs,
                        transform=dst_transform,
                        compress="DEFLATE",
                        tiled=True,
                        blockxsize=512,
                        blockysize=512,
                        nodata=np.nan,
                    ) as dst2:
                        dst2.write(rgb_best)
                    log.info("Wrote single-best-date products: %s (best_date=%s)", out_best, best_date)
        except Exception as exc:
            log.warning("Failed writing RGB_10m_best_date.tif: %s", exc)

        log.info("Wrote composite to %s (used %s dates)", out_dir, used)
        _write_composite_meta(out_dir, {
            "cache_key": str(want_key) if want_key is not None else _composite_cache_key(bbox_wesn, start_date, end_date),
            "params_fp": params_fp if params_fp is not None else {},
            "inputs_fp": inputs_fp if inputs_fp is not None else {},
            "bbox_wesn": list(map(float, bbox_wesn)),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "shared_date_mode_requested": str(shared_date_mode),
            "shared_date_mode_used": str(shared_mode_used),
            "glint_correction": glint_meta_comp,
            "glint_correction_best_date": glint_meta_best,
        })

        # --- Optional cleanup: delete raw Sentinel-2 downloads after successful composite build ---
        if clean_cache:
            try:
                if sentinel_cache_dir.exists():
                    log.info("Cleanup requested. Deleting raw scene cache: %s", sentinel_cache_dir)
                    shutil.rmtree(sentinel_cache_dir, ignore_errors=True)
            except Exception as e:
                log.warning("Cleanup requested but failed to delete raw cache (%s): %s", sentinel_cache_dir, e)
        # ------------------------------------------------------------------------------

        ret = {k: str(v) for k, v in expected.items()}
        # Attach lightweight composite QC report (optional)
        try:
            from log_report import raster_quickstats, mask_fraction
            rep = {
                "expected": {k: str(v) for k, v in expected.items()},
                "rasters": {},
                "masks": {}
            }
            if expected.get("B02") and Path(expected["B02"]).exists():
                rep["rasters"]["B02"] = raster_quickstats(str(expected["B02"]))
            if expected.get("B03") and Path(expected["B03"]).exists():
                rep["rasters"]["B03"] = raster_quickstats(str(expected["B03"]))
            if expected.get("B04") and Path(expected["B04"]).exists():
                rep["rasters"]["B04"] = raster_quickstats(str(expected["B04"]))
            if expected.get("B08") and Path(expected["B08"]).exists():
                rep["rasters"]["B08"] = raster_quickstats(str(expected["B08"]))
            if expected.get("BRIGHTNESS") and Path(expected["BRIGHTNESS"]).exists():
                rep["rasters"]["BRIGHTNESS"] = raster_quickstats(str(expected["BRIGHTNESS"]))
            if expected.get("CLEAR_WATER") and Path(expected["CLEAR_WATER"]).exists():
                rep["masks"]["CLEAR_WATER"] = mask_fraction(str(expected["CLEAR_WATER"]))
            if expected.get("SCL") and Path(expected["SCL"]).exists():
                # SCL class 6 == water for S2 L2A
                rep["masks"]["SCL_water"] = mask_fraction(str(expected["SCL"]), true_values=[6])
            
            # Optical feasibility diagnostics (SDB physics sanity checks)
            try:
                import numpy as _np
                import rasterio as _rio
                _eps = 1e-6
                _mask = None
                _mask_path = None
                _mask_true_values = None
                _mask_threshold = None
                if expected.get("CLEAR_WATER") and Path(expected["CLEAR_WATER"]).exists():
                    _mask_path = str(expected["CLEAR_WATER"]); _mask_threshold = 1.0
                elif expected.get("SCL") and Path(expected["SCL"]).exists():
                    _mask_path = str(expected["SCL"]); _mask_true_values = [6]

                if _mask_path:
                    with _rio.open(_mask_path) as _mds:
                        _m = _mds.read(1)
                        if _mds.nodata is not None:
                            _m = _np.where(_m == _mds.nodata, _np.nan, _m)
                    if _mask_true_values is not None:
                        _mask = _np.isfinite(_m) & _np.isin(_m.astype(_np.int32), _mask_true_values)
                    elif _mask_threshold is not None:
                        _mask = _np.isfinite(_m) & (_m.astype(_np.float32) >= float(_mask_threshold))
                    else:
                        _mask = _np.isfinite(_m) & (_m != 0)


                od = {}
                if expected.get("B02") and Path(expected["B02"]).exists():
                    _v02 = _sample_band_masked(expected["B02"], mask=_mask, np_mod=_np, rio_mod=_rio)
                    od["n_sample_b02"] = int(_v02.size)
                    if _v02.size:
                        od["L_inf_proxy_b02_p01"] = float(_np.percentile(_v02, 1))
                        _vlog = _np.log(_np.maximum(_v02, _eps))
                        od["EDR_log_b02_p95_p05"] = float(_np.percentile(_vlog, 95) - _np.percentile(_vlog, 5))
                if expected.get("B03") and Path(expected["B03"]).exists():
                    _v03 = _sample_band_masked(expected["B03"], mask=_mask, np_mod=_np, rio_mod=_rio)
                    od["n_sample_b03"] = int(_v03.size)
                    if _v03.size:
                        od["L_inf_proxy_b03_p01"] = float(_np.percentile(_v03, 1))
                        _vlog = _np.log(_np.maximum(_v03, _eps))
                        od["EDR_log_b03_p95_p05"] = float(_np.percentile(_vlog, 95) - _np.percentile(_vlog, 5))
                _edrs = [od.get("EDR_log_b02_p95_p05"), od.get("EDR_log_b03_p95_p05")]
                _edrs = [float(x) for x in _edrs if isinstance(x, (int, float))]
                if _edrs:
                    _best = max(_edrs)
                    od["EDR_best"] = float(_best)
                    od["signal_feasibility"] = "good" if _best >= 1.0 else ("marginal" if _best >= 0.7 else "poor")
                else:
                    od["signal_feasibility"] = "unknown"
                rep["optical_diagnostics"] = od
            except Exception:
                log.debug("ignored", exc_info=True)

            ret["_report"] = rep
        except Exception:
            log.debug("ignored", exc_info=True)
        return ret

    finally:
        # always clear lock
        try:
            lock = out_dir / ".s2_running"
            if lock.exists():
                lock.unlink()
        except Exception:
            log.debug("ignored", exc_info=True)

def main():
    p = argparse.ArgumentParser(description="Sentinel-2 composite via weighted shared-date selection + feather mosaic.")
    p.add_argument("--aoi", required=True, help="AOI W/E/S/N")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out-dir", default="s2_out")
    p.add_argument("--max-cloud", type=float, default=20.0)
    p.add_argument("--scene-limit", type=int, default=10)
    p.add_argument("--preferred-months", default="", help="Comma-separated months, empty=all")
    p.add_argument("--shared-date-mode", default="strict", choices=["strict", "union"])
    p.add_argument("--stac-url", default=DEFAULT_STAC_URL)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--stac-limit", type=int, default=200)
    p.add_argument("--stac-page-limit", type=int, default=None, help="Alias for --stac-limit (backward compatibility)")
    p.add_argument("--stac-max-items", type=int, default=5000)
    p.add_argument("--stac-chunk-months", type=int, default=0)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--download-workers", type=int, default=8)
    p.add_argument("--scl-dilate", type=int, default=3)
    p.add_argument("--no-harmonize", action="store_true")
    p.add_argument("--harmonize-max-abs-offset", type=float, default=0.03)
    p.add_argument("--deepwater-nir-max", type=float, default=0.03)
    p.add_argument("--deepwater-bright-max", type=float, default=0.15)
    # Sun-glint correction (Hedley-style)
    p.add_argument("--glint-correct", action="store_true", help="Apply Hedley-style sun-glint correction to visible bands (post-composite).")
    p.add_argument("--glint-nir-band", default="B08", help="NIR band used for glint correction (default: B08).")
    p.add_argument("--glint-vis-bands", default="B02,B03,B04", help="Comma-separated visible bands to correct (default: B02,B03,B04).")
    p.add_argument("--glint-nir-min-percentile", type=float, default=1.0, help="NIR percentile over stable water used as nir_min (default: 1).")
    p.add_argument("--glint-deepwater-b02-max", type=float, default=0.20, help="Max B02 in stable-water mask for glint fitting (default: 0.20).")
    p.add_argument("--glint-min-samples", type=int, default=5000, help="Minimum stable-water samples for glint fitting.")
    p.add_argument("--glint-max-samples", type=int, default=2000000, help="Maximum samples used for glint fitting (random subset).")
    p.add_argument("--glint-clip-min", type=float, default=1e-6, help="Clip corrected reflectance to at least this value.")
    p.add_argument("--b02-thresh", type=float, default=0.25, help="B02 reflectance threshold for glint/brightness reject")
    p.add_argument("--allow-bright-shallow-pixels", action="store_true", help="Allow bright shallow pixels via NIR escape hatch")
    p.add_argument("--bright-shallow-nir-max", type=float, default=0.03, help="If --allow-bright-shallow-pixels, accept pixels with NIR < this value even if B02 is bright")
    p.add_argument("--apply-gl-turbidity-reject", action="store_true", help="Enable Great Lakes turbidity/haze reject (nir, nir/green, red thresholds)")
    p.add_argument("--gl-nir-max", type=float, default=0.12)
    p.add_argument("--gl-nir-green-ratio-max", type=float, default=0.60)
    p.add_argument("--gl-red-max", type=float, default=0.08)
    p.add_argument("--feather-dist-cap", type=int, default=300, help="Distance cap in pixels for edge feather weights (default=300)")
    p.add_argument("--min-scene-valid-frac", type=float, default=0.90, help="Reject a tile-scene if valid pixel fraction is below this (helps remove purple blocks / partial reads)")
    p.add_argument("--edge-weight-power", type=float, default=2.0, help="Power curve for tile-edge taper (higher penalizes edges more)")
    p.add_argument("--edge-weight-min", type=float, default=0.0, help="Minimum edge weight (0 disables contribution at edges)")
    p.add_argument("--temporal-median-k", type=int, default=0, help="If >0, build final temporal median using only the best K dates (post-QC). 0=use all selected dates.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    bbox = parse_aoi(args.aoi)
    pref = []
    if args.preferred_months.strip():
        pref = [int(x) for x in re.split(r"[,\s]+", args.preferred_months.strip()) if x.strip()]

    glint_vis = [x.strip() for x in re.split(r"[,\s]+", str(getattr(args, "glint_vis_bands", "")).strip()) if x.strip()]

    cache_dir = Path(args.cache_dir) if args.cache_dir.strip() else None

    build_weighted_shared_date_composite(
        out_dir=Path(args.out_dir),
        bbox_wesn=bbox,
        start_date=args.start,
        end_date=args.end,
        max_cloud=args.max_cloud,
        scene_limit=args.scene_limit,
        preferred_months=pref if pref else None,
        shared_date_mode=args.shared_date_mode,
        stac_url=args.stac_url,
        collection=args.collection,
        stac_limit=args.stac_limit,
        stac_page_limit=args.stac_page_limit,
        stac_max_items=args.stac_max_items,
        stac_chunk_months=args.stac_chunk_months,
        cache_dir=cache_dir,
        download_workers=args.download_workers,
        scl_dilate=args.scl_dilate,
        harmonize=(not args.no_harmonize),
        harmonize_max_abs_offset=args.harmonize_max_abs_offset,
        deepwater_nir_max=args.deepwater_nir_max,
        deepwater_bright_max=args.deepwater_bright_max,
        glint_correct=args.glint_correct,
        glint_nir_band=args.glint_nir_band,
        glint_vis_bands=glint_vis if glint_vis else None,
        glint_nir_min_percentile=args.glint_nir_min_percentile,
        glint_deepwater_b02_max=args.glint_deepwater_b02_max,
        glint_min_samples=args.glint_min_samples,
        glint_max_samples=args.glint_max_samples,
        glint_clip_min=args.glint_clip_min,
        b02_thresh=args.b02_thresh,
        allow_bright_shallow_pixels=args.allow_bright_shallow_pixels,
        bright_shallow_nir_max=args.bright_shallow_nir_max,
        apply_gl_turbidity_reject=args.apply_gl_turbidity_reject,
        gl_nir_max=args.gl_nir_max,
        gl_nir_green_ratio_max=args.gl_nir_green_ratio_max,
        gl_red_max=args.gl_red_max,
        feather_dist_cap=args.feather_dist_cap,
        min_scene_valid_frac=args.min_scene_valid_frac,
        edge_weight_power=args.edge_weight_power,
        edge_weight_min=args.edge_weight_min,
        temporal_median_k=args.temporal_median_k,
    )

if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
