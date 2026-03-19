"""usgs_nwis.py – Fetch USGS NWIS site metadata and discharge measurement records (RDB parser)

Lightweight helpers to fetch USGS NWIS site metadata and discharge measurement records.

Why this exists:
- Discharge *measurements* (not just daily values) often include width and area.
- Mean depth can be approximated as area/width, providing a strong local constraint
  for river bathymetry priors when no sonar/bathy lidar exists.

Design goals:
- Zero non-stdlib dependencies for web I/O (uses urllib).
- Caching of raw responses to reduce repeated requests and enable offline re-runs.
- Parsing of NWIS "RDB" format into pandas DataFrames.

Refs:
- NWIS Water Services: https://waterservices.usgs.gov/
"""

from __future__ import annotations

import io
import os
import time
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd

log = logging.getLogger("river.usgs_nwis")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_text_cached(url: str, cache_path: Optional[Path], timeout_s: int = 45) -> str:
    """Fetch URL as text, with optional filesystem cache."""
    if cache_path is not None and cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log.debug("ignored", exc_info=True)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "cudem-river-bathy/1.0 (xs_infer_bathy_raster.py)",
            "Accept": "text/plain",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    text = data.decode("utf-8", errors="replace")

    if cache_path is not None:
        try:
            _ensure_dir(cache_path.parent)
            cache_path.write_text(text, encoding="utf-8")
        except OSError:
            log.debug("ignored", exc_info=True)

    # be a polite client
    time.sleep(0.1)
    return text


def _parse_rdb(text: str) -> pd.DataFrame:
    """Parse NWIS RDB (tab-delimited with # comments and a types line)."""
    if not text:
        return pd.DataFrame()

    # Strip comment lines
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()

    # NWIS RDB includes a second line describing column types. pandas can ignore it via skiprows=[1]
    buf = io.StringIO("\n".join(lines))
    try:
        df = pd.read_csv(buf, sep="\t", dtype=str, na_values=["", "NaN", "nan"], keep_default_na=True, skiprows=[1])
    except (pd.errors.ParserError, ValueError):
        # fallback: try without skipping
        buf.seek(0)
        df = pd.read_csv(buf, sep="\t", dtype=str, na_values=["", "NaN", "nan"], keep_default_na=True)
    return df


def fetch_site_locations(
    sites: List[str],
    cache_dir: Optional[Path] = None,
    timeout_s: int = 45,
) -> pd.DataFrame:
    """Fetch site metadata for one or more sites (lat/lon, name, etc.)."""
    if not sites:
        return pd.DataFrame()

    sites_str = ",".join([str(s).strip() for s in sites if str(s).strip()])
    params = {"format": "rdb", "sites": sites_str}
    url = "https://waterservices.usgs.gov/nwis/site/?" + urllib.parse.urlencode(params)

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"site_{sites_str.replace(',', '_')}.rdb"

    try:
        text = _read_text_cached(url, cache_path, timeout_s=timeout_s)
        df = _parse_rdb(text)
    except (OSError, ValueError) as e:
        log.warning("site fetch failed: %s", e)
        return pd.DataFrame()

    # Normalize
    for c in ["site_no", "dec_lat_va", "dec_long_va", "station_nm"]:
        if c not in df.columns:
            df[c] = pd.NA
    df["site_no"] = df["site_no"].astype(str)
    df["dec_lat_va"] = pd.to_numeric(df["dec_lat_va"], errors="coerce")
    df["dec_long_va"] = pd.to_numeric(df["dec_long_va"], errors="coerce")
    return df


def fetch_discharge_measurements(
    sites: List[str],
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
    timeout_s: int = 45,
) -> pd.DataFrame:
    """Fetch discharge measurement records.

    Returns a DataFrame with normalized numeric columns when possible.
    """
    if not sites:
        return pd.DataFrame()

    sites_str = ",".join([str(s).strip() for s in sites if str(s).strip()])
    params = {
        "format": "rdb",
        "sites": sites_str,
        "startDT": str(start_date),
        "endDT": str(end_date),
    }
    url = "https://waterservices.usgs.gov/nwis/measurements/?" + urllib.parse.urlencode(params)

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"meas_{sites_str.replace(',', '_')}_{start_date}_{end_date}.rdb"

    try:
        text = _read_text_cached(url, cache_path, timeout_s=timeout_s)
        df = _parse_rdb(text)
    except (OSError, ValueError) as e:
        log.warning("measurements fetch failed: %s", e)
        return pd.DataFrame()

    if df.empty:
        return df

    # Common NWIS measurement fields:
    # - meas_dt, meas_tz_cd, agency_cd, site_no
    # - discharge_va (cfs), gage_height_va (ft), width_va (ft), area_va (sqft), velocity_va (fps)
    # But names may vary slightly; we will normalize known ones.
    rename = {}
    for cand, std in [
        ("discharge_va", "discharge"),
        ("gage_height_va", "gage_height"),
        ("width_va", "width"),
        ("area_va", "area"),
        ("velocity_va", "velocity"),
        ("meas_dt", "meas_dt"),
        ("site_no", "site_no"),
    ]:
        if cand in df.columns:
            rename[cand] = std
    df = df.rename(columns=rename)

    for c in ["site_no", "meas_dt"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Keep raw units; numeric coercion happens in normalize_units()
    return df


def normalize_units_us_to_si(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common USGS measurement units to SI.

    Assumes:
    - width: feet -> meters
    - area: square feet -> m^2
    - discharge: cfs -> m^3/s
    - gage_height: feet -> meters
    - velocity: ft/s -> m/s
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series([], dtype="float64")

    # Convert
    FT_TO_M = 0.3048
    SQFT_TO_M2 = 0.09290304
    CFS_TO_CMS = 0.028316846592

    if "width" in out.columns:
        out["width_m"] = _num("width") * FT_TO_M
    if "area" in out.columns:
        out["area_m2"] = _num("area") * SQFT_TO_M2
    if "discharge" in out.columns:
        out["discharge_cms"] = _num("discharge") * CFS_TO_CMS
    if "gage_height" in out.columns:
        out["gage_height_m"] = _num("gage_height") * FT_TO_M
    if "velocity" in out.columns:
        out["velocity_ms"] = _num("velocity") * FT_TO_M

    if "width_m" in out.columns and "area_m2" in out.columns:
        out["mean_depth_m"] = out["area_m2"] / out["width_m"]
    else:
        out["mean_depth_m"] = pd.NA

    # Basic validity
    for c in ["width_m", "area_m2", "mean_depth_m"]:
        if c in out.columns:
            out.loc[~pd.to_numeric(out[c], errors="coerce").notna(), c] = pd.NA
    return out



def compute_site_a_from_measurements(
    meas_si: pd.DataFrame,
    b: float,
    mean_to_dmax: float = 1.0,
    stat: str = "median",
    q_quantile_range: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[float], int, Dict[str, Any]]:
    """Estimate a site-specific coefficient *a* in: Dmax ≈ a * W^b.

    This is used as a *soft* prior when the workflow is anchored to USGS discharge
    *measurement* records (width + area -> mean depth).

    Args:
        meas_si: Output of normalize_units_us_to_si(); expects width_m and mean_depth_m.
        b: Width exponent used elsewhere in the workflow.
        mean_to_dmax: Conversion factor from mean depth to an approximate Dmax.
        stat: Aggregation over per-measurement a-values ('median' or 'mean').
        q_quantile_range: Optional (lo, hi) quantiles to filter measurements by discharge.

    Returns:
        (a_site or None, n_used, meta)
    """
    if meas_si is None or meas_si.empty:
        return (None, 0, {"reason": "empty"})

    df = meas_si.copy()

    w = pd.to_numeric(df.get("width_m", pd.Series([], dtype="float64")), errors="coerce")
    dmean = pd.to_numeric(df.get("mean_depth_m", pd.Series([], dtype="float64")), errors="coerce")

    mask = w.notna() & dmean.notna() & (w > 0) & (dmean > 0)

    q_lo_v = None
    q_hi_v = None
    if q_quantile_range is not None and "discharge_cms" in df.columns:
        q = pd.to_numeric(df.get("discharge_cms"), errors="coerce")
        q = q.where(q.notna() & (q > 0))
        try:
            lo, hi = float(q_quantile_range[0]), float(q_quantile_range[1])
            if 0.0 <= lo < hi <= 1.0 and q.notna().any():
                q_lo_v = float(q.quantile(lo))
                q_hi_v = float(q.quantile(hi))
                mask &= q.notna() & (q >= q_lo_v) & (q <= q_hi_v)
        except (TypeError, ValueError):
            q_lo_v = None
            q_hi_v = None

    w2 = w.loc[mask].astype("float64")
    d2 = dmean.loc[mask].astype("float64")
    if len(w2) == 0:
        return (None, 0, {"reason": "no_valid"})

    # Convert mean depth -> approximate max depth under the workflow's trapezoid model.
    try:
        mt = float(mean_to_dmax)
    except (TypeError, ValueError):
        mt = 1.0
    if not (pd.notna(mt) and mt > 0):
        mt = 1.0
    dmax = d2 * mt

    # a_i = Dmax / W^b
    try:
        bb = float(b)
    except (TypeError, ValueError):
        bb = 0.0

    a = dmax / (w2 ** bb)
    a = pd.to_numeric(a, errors="coerce")
    a = a.where(a.notna() & (a > 0)).dropna()

    n_used = int(len(a))
    if n_used == 0:
        return (None, 0, {"reason": "no_a"})

    s = str(stat or "median").strip().lower()
    if s == "mean":
        a_site = float(a.mean())
    else:
        a_site = float(a.median())

    a_mean = float(a.mean())
    a_std = float(a.std(ddof=1)) if n_used >= 2 else 0.0
    a_cv = float(a_std / a_mean) if (a_mean > 0 and pd.notna(a_mean)) else float("nan")

    meta: Dict[str, Any] = {
        "n_total": int(len(df)),
        "n_used": n_used,
        "width_med_m": float(w2.median()) if len(w2) else float("nan"),
        "a_cv": a_cv,
        "q_lo": q_lo_v,
        "q_hi": q_hi_v,
        "stat": s,
        "mean_to_dmax": float(mt),
    }
    return (a_site, n_used, meta)
