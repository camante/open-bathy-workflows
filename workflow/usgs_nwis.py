"""usgs_nwis.py

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
from typing import Dict, Optional, List, Tuple

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
        except Exception:
            pass

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
        except Exception:
            pass

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
    except Exception:
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
    except Exception as e:
        log.warning("[USGS] site fetch failed: %s", e)
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
    except Exception as e:
        log.warning("[USGS] measurements fetch failed: %s", e)
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


def _compute_slope_proxy(xs_param: pd.DataFrame, cfg: Optional[InferConfig] = None) -> pd.Series:
    """Estimate water-surface slope proxy per XS using d(WSE)/d(s) within each river_id.

    This is a *fallback* when no slope attribute is available. Because WSE is a proxy
    derived from DEM sampling, it can be noisy or biased. We therefore:
      - require a minimum number of XS per river_id
      - smooth WSE along stationing (rolling median)
      - require |corr(s, WSE)| >= threshold (else return NaN for that group)

    Returns slope (unitless, m/m).
    """
    if xs_param is None or xs_param.empty:
        return pd.Series([], dtype="float64")

    out = pd.Series(np.nan, index=xs_param.index, dtype="float64")
    if "river_id" not in xs_param.columns or "s_center_m" not in xs_param.columns or "wse_m" not in xs_param.columns:
        return out

    min_n = int(getattr(cfg, "slope_proxy_min_n", 10)) if cfg is not None else 10
    min_r = float(getattr(cfg, "slope_proxy_min_r", 0.70)) if cfg is not None else 0.70
    wwin = int(max(3, int(getattr(cfg, "smooth_window", 7)))) if cfg is not None else 7
    if wwin % 2 == 0:
        wwin += 1

    for _, g in xs_param.groupby("river_id", dropna=False):
        gg = g.copy()
        gg["s_center_m"] = pd.to_numeric(gg["s_center_m"], errors="coerce")
        gg["wse_m"] = pd.to_numeric(gg["wse_m"], errors="coerce")
        gg = gg.sort_values("s_center_m")
        gg = gg.loc[gg["s_center_m"].notna() & gg["wse_m"].notna()]
        if len(gg) < max(3, min_n):
            continue

        s = gg["s_center_m"].to_numpy(dtype=float)
        w = gg["wse_m"].to_numpy(dtype=float)

        # Require meaningful monotonic stationing
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
            continue
        if np.nanmax(np.diff(s)) <= 0:
            continue

        # Smooth WSE to reduce DEM artifacts (rolling median)
        w_s = pd.Series(w).rolling(window=wwin, center=True, min_periods=max(3, wwin // 3)).median().to_numpy()

        # Require correlation between stationing and WSE (river should generally slope)
        try:
            r = np.corrcoef(s, w_s)[0, 1]
        except Exception:
            r = np.nan
        if (not np.isfinite(r)) or (abs(float(r)) < float(min_r)):
            continue

        # Gradient-based slope
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.abs(np.gradient(w_s, s))
        # Clip to a sane range
        slope = np.clip(slope, 0.0, 0.05)

        out.loc[gg.index] = slope

    return out

