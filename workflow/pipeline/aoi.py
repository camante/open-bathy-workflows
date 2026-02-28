"""AOI / bbox utilities.

Lifted from bathy_main.py (Phase-2 refactor). No behavior changes intended.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Tuple


def parse_aoi_bbox(aoi: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """Parse AOI bbox from 'W/E/S/N' string.

    Accepts values separated by commas, slashes, or whitespace.
    Returns (W, S, E, N) as (lon_min, lat_min, lon_max, lat_max).
    """
    if not aoi:
        return None
    s = str(aoi).strip()
    parts = re.split(r"[,/\s]+", s)
    parts = [p for p in parts if p]
    if len(parts) != 4:
        return None
    try:
        w, e, s_, n = [float(p) for p in parts]
    except Exception:
        return None
    if not (w < e and s_ < n):
        return None
    return (w, s_, e, n)


def bbox_to_aoi_str(bbox: Tuple[float, float, float, float]) -> str:
    w, s, e, n = bbox
    return f"{w}/{e}/{s}/{n}"


def aoi_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    w, s, e, n = bbox
    return ((w + e) / 2.0, (s + n) / 2.0)  # (lon, lat)


def expand_bbox_km(
    bbox: Tuple[float, float, float, float], buffer_km: float
) -> Tuple[float, float, float, float]:
    """Expand lon/lat bbox by an approximate kilometer buffer.

    Uses 1 deg lat ~ 111.32 km and 1 deg lon scaled by cos(lat_center).
    """
    w, s, e, n = bbox
    if buffer_km <= 0:
        return bbox
    lon_c, lat_c = aoi_centroid(bbox)
    km_per_deg_lat = 111.32
    km_per_deg_lon = max(1e-6, km_per_deg_lat * math.cos(math.radians(lat_c)))
    dlat = buffer_km / km_per_deg_lat
    dlon = buffer_km / km_per_deg_lon
    return (w - dlon, s - dlat, e + dlon, n + dlat)



def expand_bbox_frac(
    bbox: Tuple[float, float, float, float], frac: float
) -> Tuple[float, float, float, float]:
    """Expand lon/lat bbox by a fractional amount of its width/height.

    frac=0.10 expands the bbox dimensions by 10% (i.e., 5% on each side).
    This is intended as the *single* user-controlled data acquisition buffer.
    """
    w, s, e, n = bbox
    if frac is None:
        return bbox
    try:
        frac = float(frac)
    except Exception:
        return bbox
    if frac <= 0:
        return bbox
    width = (e - w)
    height = (n - s)
    if width <= 0 or height <= 0:
        return bbox
    dw = 0.5 * frac * width
    dh = 0.5 * frac * height
    return (w - dw, s - dh, e + dw, n + dh)



def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """Return a WGS84 UTM EPSG code string (EPSG:326## or EPSG:327##) for lon/lat."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"


def aoi_center_lonlat(aoi: str) -> tuple[float, float]:
    bbox = parse_aoi_bbox(aoi)
    if not bbox:
        raise ValueError(f"Cannot parse AOI bbox from: {aoi!r}")
    lon, lat = aoi_centroid(bbox)
    return lon, lat


def buffer_aoi(aoi: str, buf_deg: float = 0) -> str:
    """Return AOI string buffered by buf_deg in degrees.

    Mirrors prior bathy_main behavior: expects 'w/e/s/n' and emits fixed
    precision formatting.
    """
    try:
        w, e, s, n = [float(x) for x in aoi.split("/")[:4]]
    except Exception as exc:
        raise ValueError(f"Invalid AOI string: {aoi!r}") from exc
    return f"{w - buf_deg:.8f}/{e + buf_deg:.8f}/{s - buf_deg:.8f}/{n + buf_deg:.8f}"
