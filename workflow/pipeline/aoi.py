"""AOI / bbox utilities.

Canonical home for AOI string parsing.  All modules that need to turn a
``"W/E/S/N"`` string into numeric bounds should import from here.

**Bbox convention used throughout this module:**

* ``(W, S, E, N)`` – used by ``parse_aoi_bbox``, ``expand_bbox_*``,
  ``aoi_centroid``, and any function that *accepts or returns* a bbox tuple.
  This matches the OGC/Shapely ``(minx, miny, maxx, maxy)`` convention.

* ``(W, E, S, N)`` – the *string* order callers type on the CLI and the order
  returned by ``parse_aoi_wesn``.  This matches the GMT / GDAL tradition.

Use ``parse_aoi_wesn`` when you need the values in CLI/string order.
Use ``parse_aoi_bbox`` when you need an OGC-style ``(minx, miny, maxx, maxy)`` bbox.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Tuple
import logging
log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Core parser – single implementation, two return conventions
# ---------------------------------------------------------------------------

def _parse_aoi_core(aoi: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """Parse an AOI string into (W, E, S, N) floats.

    Accepts values separated by commas, slashes, or whitespace.
    Returns None on any parse failure (no exceptions).
    """
    if not aoi:
        return None
    s = str(aoi).strip()
    parts = re.split(r"[,/\s]+", s)
    parts = [p for p in parts if p]
    if len(parts) != 4:
        return None
    try:
        w, e, s_, n = (float(p) for p in parts)
    except (ValueError, TypeError):
        return None
    # Auto-fix reversed bounds (user typo)
    if e < w:
        w, e = e, w
    if n < s_:
        s_, n = n, s_
    if w == e or s_ == n:
        return None
    return (w, e, s_, n)


def parse_aoi_wesn(aoi: Optional[str], *, strict: bool = False,
                   ) -> Optional[Tuple[float, float, float, float]]:
    """Parse AOI string into ``(W, E, S, N)`` – the CLI / GMT convention.

    This is the **canonical** AOI parser.  All other per-module parsers
    (``atl.parse_aoi_string``, ``s2_optics.parse_aoi``,
    ``river_network._parse_aoi``, ``bathy_main._parse_aoi_bounds_deg``)
    should delegate here.

    Parameters
    ----------
    aoi : str or None
        ``"W/E/S/N"`` string (commas, slashes, or whitespace accepted).
    strict : bool
        If *True*, raise ``ValueError`` on bad input instead of returning
        ``None``.

    Returns
    -------
    (W, E, S, N) or None
    """
    result = _parse_aoi_core(aoi)
    if result is None and strict:
        raise ValueError(f"Cannot parse AOI string: {aoi!r}  (expected W/E/S/N)")
    return result


def parse_aoi_bbox(aoi: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """Parse AOI bbox from 'W/E/S/N' string.

    Accepts values separated by commas, slashes, or whitespace.
    Returns (W, S, E, N) as (lon_min, lat_min, lon_max, lat_max)
    — the OGC / Shapely bounds convention.
    """
    wesn = _parse_aoi_core(aoi)
    if wesn is None:
        return None
    w, e, s_, n = wesn
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
        log.debug("expand_bbox_frac: suppressed exception", exc_info=True)
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
    wesn = parse_aoi_wesn(aoi, strict=True)
    w, e, s, n = wesn
    return ((w + e) / 2.0, (s + n) / 2.0)


def buffer_aoi(aoi: str, buf_deg: float = 0) -> str:
    """Return AOI string buffered by buf_deg in degrees.

    Mirrors prior bathy_main behavior: expects 'w/e/s/n' and emits fixed
    precision formatting.
    """
    w, e, s, n = parse_aoi_wesn(aoi, strict=True)
    return f"{w - buf_deg:.8f}/{e + buf_deg:.8f}/{s - buf_deg:.8f}/{n + buf_deg:.8f}"
