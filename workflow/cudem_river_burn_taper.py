#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cudem_river_burn_taper.py

Fuses river bed patches into a base DEM with guardrails.
FIXED: GCS-Aware distance calculation and strict nodata protection.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import rasterio
from pyproj import CRS

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

log = logging.getLogger("cudem_river_burn")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def _read_band(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        return ds.read(1), ds.profile

def _get_pixel_size_m(profile: dict) -> float:
    """Calculates physical pixel size in meters, even if CRS is in degrees."""
    t, crs_val = profile.get("transform"), profile.get("crs")
    if not t or not crs_val: return 1.0
    pcrs = CRS.from_user_input(crs_val)
    dx, dy = abs(t.a), abs(t.e)
    if pcrs.is_geographic:
        # Approximate meters per degree at center latitude
        lat_center = t.f + (t.e * profile["height"] / 2.0)
        m_per_deg = 111320.0
        return float(((dx * m_per_deg * np.cos(np.radians(lat_center))) + (dy * m_per_deg)) / 2.0)
    return float((dx + dy) / 2.0)

def fuse(
    base_dem_path: Path, patch_dem_path: Path, patch_mask_path: Path,
    out_dem_path: Path, out_prov_path: Path,
    measured_mask_path: Optional[Path] = None, bank_mask_path: Optional[Path] = None,
    taper_m: Optional[float] = None, taper_source: str = "both"
) -> None:
    base, p_base = _read_band(base_dem_path)
    patch, p_patch = _read_band(patch_dem_path)
    pmask, p_pmask = _read_band(patch_mask_path)

    b_nd = p_base.get("nodata")
    p_nd = p_patch.get("nodata")

    # 1. DEFINE STRICT VALIDITY (Critical for fixing large negatives)
    base_valid = np.isfinite(base) & (base != b_nd) if b_nd is not None else np.isfinite(base)
    patch_valid = np.isfinite(patch) & (patch != p_nd) if p_nd is not None else np.isfinite(patch)
    
    # Only allow updates where the patch has valid bathy data
    allow = patch_valid & (pmask == 1)
    prov = np.zeros(base.shape, dtype=np.uint8)

    # Guards
    meas_block = None
    if measured_mask_path:
        meas, p_meas = _read_band(measured_mask_path)
        meas_block = (meas == 1)
        prov[allow & meas_block] = 2
        allow &= ~meas_block

    bank_block = None
    if bank_mask_path:
        bmk, p_bmk = _read_band(bank_mask_path)
        bank_block = (bmk == 1)
        prov[allow & bank_block] = 3
        allow &= ~bank_block

    # 2. Taper Weights (GCS Aware)
    w_blend = None
    if taper_m and taper_m > 0:
        src_mask = np.zeros(base.shape, dtype=bool)
        if taper_source in ("both", "measured") and meas_block is not None: src_mask |= meas_block
        if taper_source in ("both", "bank") and bank_block is not None: src_mask |= bank_block
        if src_mask.any():
            px_m = _get_pixel_size_m(p_base)
            dist_px = distance_transform_edt(~src_mask) if distance_transform_edt else np.zeros_like(base)
            w_blend = np.clip((dist_px * px_m) / taper_m, 0.0, 1.0).astype("float32")
            log.info(f"[TAPER] Local px size: {px_m:.4f}m. Tapering {taper_m}m.")

    # 3. FUSION LOGIC (Safe from Nodata)
    fused = base.copy()
    
    # CASE A: Patch applied where base DEM has valid data (Blended/Tapered)
    blend_idx = allow & base_valid
    if w_blend is not None:
        w = w_blend[blend_idx]
        fused[blend_idx] = (base[blend_idx] * (1.0 - w) + patch[blend_idx] * w).astype(base.dtype)
        prov[blend_idx] = np.where(w >= 0.999, 1, 5).astype("uint8")
    else:
        fused[blend_idx] = patch[blend_idx]
        prov[blend_idx] = 1

    # CASE B: Patch applied where base DEM is NODATA (GAP FILL)
    # Here we skip the math entirely to avoid using -9999
    fill_idx = allow & ~base_valid
    fused[fill_idx] = patch[fill_idx]
    prov[fill_idx] = 1

    # 4. WRITE OUTPUTS
    for path, arr, prof, dtype in [(out_dem_path, fused, p_base, p_base['dtype']), 
                                   (out_prov_path, prov, p_base, 'uint8')]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        out_prof = prof.copy()
        # Ensure uint8 output doesn't use float32 nodata
        new_nd = 0 if dtype == 'uint8' else b_nd
        out_prof.update(driver="GTiff", count=1, dtype=dtype, nodata=new_nd, compress="deflate")
        with rasterio.open(path, "w", **out_prof) as dst:
            dst.write(arr.astype(dtype), 1)

    log.info(f"[DONE] Applied: {int((prov==1).sum())} | Tapered: {int((prov==5).sum())} | Blocked: {int((prov==3).sum())}")


def _parse_args():
    p = argparse.ArgumentParser("cudem_river_burn_taper.py")
    p.add_argument("--base-dem", required=True)
    p.add_argument("--patch-dem", required=True)
    p.add_argument("--patch-mask", required=True)
    p.add_argument("--out-dem", required=True)
    p.add_argument("--out-provenance", required=True)
    p.add_argument("--measured-mask")
    p.add_argument("--bank-mask")
    p.add_argument("--taper-m", type=float)
    p.add_argument("--taper-source", choices=["bank", "measured", "both"], default="both")
    p.add_argument("--max-abs-change-m", type=float)
    return p.parse_args()

if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    a = _parse_args()
    fuse(Path(a.base_dem), Path(a.patch_dem), Path(a.patch_mask), Path(a.out_dem), Path(a.out_provenance),
         measured_mask_path=Path(a.measured_mask) if a.measured_mask else None,
         bank_mask_path=Path(a.bank_mask) if a.bank_mask else None,
         taper_m=a.taper_m, taper_source=a.taper_source)