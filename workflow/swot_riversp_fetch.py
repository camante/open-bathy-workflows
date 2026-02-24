# -*- coding: utf-8 -*-
"""SWOT RiverSP auto-fetch helper (PO.DAAC via CMR).

This module is intentionally lightweight:
  - Uses `earthaccess` if installed to authenticate/search/download.
  - Falls back gracefully if unavailable or auth missing.

Expected auth (choose one):
  1) ~/.netrc with Earthdata credentials (recommended).
  2) Environment variables: EARTHDATA_USERNAME / EARTHDATA_PASSWORD.

The caller should treat failures as non-fatal (continue without SWOT).
"""


from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
from typing import List, Optional, Tuple

def _hash_dict(d: dict) -> str:
    b = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()

@dataclass
class RiverspFetchResult:
    files: List[str]
    cache_dir: str
    used_short_name: Optional[str] = None
    message: str = ""

def fetch_riversp(
    bbox_wesn: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    cache_root: str,
    product: str = "reach",
    short_name: Optional[str] = None,
    max_granules: int = 200,
    logger=None,
) -> RiverspFetchResult:
    """Fetch SWOT RiverSP granules covering bbox/time window.

    Parameters
    ----------
    bbox_wesn : (w, e, s, n) in degrees
    start_date, end_date : YYYY-MM-DD
    cache_root : base cache directory (will create swot/riversp/<hash>/)
    product : "reach" or "node" (affects default short_name search)
    short_name : optional PO.DAAC short name; if omitted, will search by prefix.
    max_granules : cap for downloads to avoid runaway for large AOIs/time ranges.

    Returns
    -------
    RiverspFetchResult
    """
    w, e, s, n = bbox_wesn
    params = {
        "bbox": [w, e, s, n],
        "start": start_date,
        "end": end_date,
        "product": product,
        "short_name": short_name,
    }
    h = _hash_dict(params)[:12]
    out_dir = Path(cache_root) / "swot" / "riversp" / h
    out_dir.mkdir(parents=True, exist_ok=True)

    # If we already have vector files, just return them.
    existing = sorted([str(p) for p in out_dir.glob("**/*") if p.suffix.lower() in {".gpkg", ".geojson", ".shp"}])
    if existing:
        return RiverspFetchResult(files=existing, cache_dir=str(out_dir), used_short_name=short_name, message="cache-hit")

    try:
        import earthaccess  # type: ignore
    except Exception as exc:
        msg = f"earthaccess not available ({exc}); cannot auto-fetch SWOT RiverSP"
        if logger: logger.warning(msg)
        return RiverspFetchResult(files=[], cache_dir=str(out_dir), used_short_name=short_name, message=msg)

    # Authentication: prefer netrc, then env.
    try:
        # netrc strategy won't prompt
        earthaccess.login(strategy="netrc")
    except Exception:
        u = os.environ.get("EARTHDATA_USERNAME") or os.environ.get("NASA_EARTHDATA_USERNAME")
        p = os.environ.get("EARTHDATA_PASSWORD") or os.environ.get("NASA_EARTHDATA_PASSWORD")
        if u and p:
            try:
                earthaccess.login(username=u, password=p)
            except Exception as exc:
                msg = f"Earthdata login failed: {exc}"
                if logger: logger.warning(msg)
                return RiverspFetchResult(files=[], cache_dir=str(out_dir), used_short_name=short_name, message=msg)
        else:
            msg = "Earthdata login not configured (no netrc and no EARTHDATA_USERNAME/EARTHDATA_PASSWORD)"
            if logger: logger.warning(msg)
            return RiverspFetchResult(files=[], cache_dir=str(out_dir), used_short_name=short_name, message=msg)

    # Determine short name if omitted.
    used_short = short_name
    if not used_short:
        # RiverSP reach vs node: PO.DAAC collections are typically like
        # SWOT_L2_HR_RiverSP_reach_2.0 (or similar). We'll search by keyword prefix.
        prefix = "SWOT_L2_HR_RiverSP_"
        key = f"{prefix}{product}"
        # Search datasets/collections first; pick first match (earthaccess returns best-ranked first).
        try:
            cols = earthaccess.search_datasets(keyword=key)
            if cols:
                used_short = cols[0].short_name
        except Exception:
            # fallback to common PO.DAAC short name patterns
            used_short = f"{prefix}{product}_D"  # older naming

    if not used_short:
        msg = "Unable to determine SWOT RiverSP short_name"
        if logger: logger.warning(msg)
        return RiverspFetchResult(files=[], cache_dir=str(out_dir), used_short_name=None, message=msg)

    if logger:
        logger.info(f"[SWOT] Searching RiverSP ({used_short}) for bbox={w:.4f},{s:.4f},{e:.4f},{n:.4f} time={start_date}..{end_date}")

    try:
        granules = earthaccess.search_data(
            short_name=used_short,
            bounding_box=(w, s, e, n),
            temporal=(start_date, end_date),
        )
    except Exception as exc:
        msg = f"SWOT search failed: {exc}"
        if logger: logger.warning(msg)
        return RiverspFetchResult(files=[], cache_dir=str(out_dir), used_short_name=used_short, message=msg)

    if not granules:
        msg = "No SWOT RiverSP granules found for AOI/time window"
        if logger: logger.info(f"[SWOT] {msg}")
        return RiverspFetchResult(files=[], cache_dir=str(out_dir), used_short_name=used_short, message=msg)

    if len(granules) > max_granules:
        granules = granules[:max_granules]
        if logger:
            logger.warning(f"[SWOT] Capping download to first {max_granules} granules; narrow --start/--end or AOI to reduce")

    # Download locally. earthaccess handles auth + redirects.
    try:
        earthaccess.download(granules, local_path=str(out_dir))
    except Exception as exc:
        msg = f"SWOT download failed: {exc}"
        if logger: logger.warning(msg)
        return RiverspFetchResult(files=[], cache_dir=str(out_dir), used_short_name=used_short, message=msg)

    files = sorted([str(p) for p in out_dir.glob('**/*') if p.suffix.lower() in {'.gpkg','.geojson','.shp'}])
    # If we downloaded only NetCDF/HDF, the workflow currently expects vectors; return empty and let caller decide.
    return RiverspFetchResult(files=files, cache_dir=str(out_dir), used_short_name=used_short, message="downloaded")
