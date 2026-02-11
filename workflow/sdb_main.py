#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdb_main.py – The Master Orchestrator for the Modular SDB Pipeline.

This script ties together the 6 core modules:
  1. sdb.atl       -> Fetch & Process ICESat-2 (Now with Land Masking)
  2. sdb.s2_optics -> Fetch Sentinel-2 & Build Masks
  3. sdb.fusion    -> Hierarchical Data Fusion (XYZ > ATL03 > ATL24)
  4. sdb.train     -> Feature Sampling & RF Training
  5. sdb.vis       -> Visual Debugging (Photon Transects)
  6. sdb.predict   -> Full-scene Inference

UPDATES:
- **FIXED: Hard-coded Override Removed**: `train_sdb_model` call now respects `validate_spatial` flag.
- **FIXED: Auto-Depth Logic**: Allows "interpolation limit" calc via Stratified Random Split.
- **FIXED: Config Override**: Regional config applies correctly without forcing spatial validation.
"""

import sys
import re
import os
import logging

# IMPORTANT: Configure logging FIRST, before importing other pipeline modules
try:
    from logging_config import setup_logging
    setup_logging()
except ImportError:
    # Fallback if logging_config.py is not present
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

import argparse
import json
import traceback
import importlib.util
import inspect
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple

# Optional run reporting (flight recorder)
try:
    from log_report import RunReport, raster_quickstats, mask_fraction
except Exception:  # pragma: no cover
    RunReport = None  # type: ignore
    raster_quickstats = None  # type: ignore
    mask_fraction = None  # type: ignore


import numpy as np
import pandas as pd
import joblib
import rasterio
import geopandas as gpd
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import new improvement modules
try:
    from validation import validate_pipeline_config, ValidationResult
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    
try:
    from checkpoints import PipelineCheckpoint, CheckpointStage, CheckpointContext
    CHECKPOINTS_AVAILABLE = True
except ImportError:
    CHECKPOINTS_AVAILABLE = False

try:
    from predict_parallel import predict_scene_parallel, ParallelConfig, estimate_parallel_benefit
    PARALLEL_PREDICT_AVAILABLE = True
except ImportError:
    PARALLEL_PREDICT_AVAILABLE = False


def detect_active_config_profile(aoi_bbox: list, config_path: str = "sdb_config.json") -> str:
    """Return the name of the active config profile for this AOI center.

    - If a region bbox matches: return region['name']
    - Else: return 'defaults' if present, otherwise 'default'
    """
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        w, s, e, n = aoi_bbox
        c_lon = (w + e) / 2.0
        c_lat = (s + n) / 2.0
        for region in data.get("regions", []):
            rw, rs, re, rn = region["bbox"]
            if (rw <= c_lon <= re) and (rs <= c_lat <= rn):
                return str(region.get("name", "region"))
        return "defaults" if "defaults" in data else "default"
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# Vertical Datum Transformation (MSL → NAVD88)
# -----------------------------------------------------------------------------

def convert_sdb_msl_to_navd88(
    input_tif: Path,
    output_tif: Path,
    source_vdatum: str = "epsg:4269+5714",  # NAD83 + MSL
    target_vdatum: str = "epsg:4269+5703",  # NAD83 + NAVD88
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    Convert SDB raster from MSL to NAVD88 vertical datum using CUDEM dlim.
    
    SCIENTIFIC NOTE:
    SDB outputs are ELEVATION values relative to MSL (Mean Sea Level = 0).
    - A pixel value of -5.0m means the seabed is at elevation -5.0m MSL
    - This is NOT "depth below instantaneous water surface"
    - MSL is the zero reference due to:
      * Multi-temporal S2 compositing (averages tidal variations)
      * ICESat-2 training data uses EGM2008 orthometric heights ≈ MSL
    
    Since SDB values are already elevations (relative to MSL), we can directly
    apply a vertical datum transformation to convert to NAVD88:
    
        elev_NAVD88 = elev_MSL + (NAVD88 - MSL separation)
    
    Example:
        - Input: -5.0m MSL (seabed 5m below MSL)
        - MSL-NAVD88 separation: +0.3m (MSL is 0.3m above NAVD88 locally)
        - Output: -5.0 + 0.3 = -4.7m NAVD88
    
    Args:
        input_tif: Path to input SDB raster (elevations relative to MSL)
        output_tif: Path to output raster (elevations in NAVD88)
        source_vdatum: Source compound EPSG (default: NAD83+MSL)
        target_vdatum: Target compound EPSG (default: NAD83+NAVD88)
        logger: Optional logger instance
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    _log = logger or logging.getLogger("sdb_main")
    
    dlim_exe = shutil.which("dlim")
    if dlim_exe is None:
        msg = "dlim not found on PATH; cannot perform vertical datum transformation"
        _log.warning(f"[VDATUM] {msg}")
        return False, msg
    
    input_tif = Path(input_tif)
    output_tif = Path(output_tif)
    
    if not input_tif.exists():
        msg = f"Input raster not found: {input_tif}"
        _log.error(f"[VDATUM] {msg}")
        return False, msg
    
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        dlim_exe,
        str(input_tif),
        "-J", source_vdatum,
        "-P", target_vdatum,
        "-O", str(output_tif),
    ]
    
    _log.info(f"[VDATUM] Converting SDB from MSL to NAVD88")
    _log.debug(f"[VDATUM] Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if result.returncode != 0:
            msg = f"dlim failed with code {result.returncode}: {result.stderr[:500]}"
            _log.error(f"[VDATUM] {msg}")
            return False, msg
        
        if not output_tif.exists():
            msg = f"dlim completed but output not found: {output_tif}"
            _log.error(f"[VDATUM] {msg}")
            return False, msg
        
        _log.info(f"[VDATUM] Successfully converted to NAVD88: {output_tif}")
        return True, f"Converted to {target_vdatum}"
        
    except subprocess.TimeoutExpired:
        msg = "dlim timed out after 600 seconds"
        _log.error(f"[VDATUM] {msg}")
        return False, msg
    except Exception as e:
        msg = f"dlim exception: {e}"
        _log.error(f"[VDATUM] {msg}")
        return False, msg


def apply_sdb_metadata(
    raster_path: Path,
    vertical_datum: str = "MSL",
    vertical_datum_epsg: int = 5714,
) -> None:
    """Apply vertical datum metadata tags to an SDB raster."""
    try:
        tags = {
            "VALUE_TYPE": "bed_elevation",
            "UNITS": "m",
            "VERTICAL_DATUM": vertical_datum,
            "VERTICAL_DATUM_EPSG": str(vertical_datum_epsg),
            "SIGN_CONVENTION": "negative_below_datum",
            "NOTE": (
                f"SDB-derived seabed elevation relative to {vertical_datum}. "
                "Negative values = below datum zero. "
                "Derived from Sentinel-2 imagery trained on ICESat-2."
            ),
        }
        with rasterio.open(raster_path, "r+") as dst:
            dst.update_tags(**tags)
    except Exception as e:
        logging.getLogger("sdb_main").warning(f"[VDATUM] Failed to update metadata: {e}")


# -----------------------------------------------------------------------------
# Import Modular Components
# -----------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import atl
    import fusion
    import train
    import predict
    import vis
except (ImportError, IndentationError, SyntaxError) as exc:
    import traceback
    print("CRITICAL ERROR: Could not import one or more SDB modules.")
    print(f"  {type(exc).__name__}: {exc}")
    traceback.print_exc()
    print("\nTip: If this is an IndentationError/SyntaxError in predict.py, replace predict.py with the fixed version from sdb_river_fixed (v0.5.0+).")
    sys.exit(1)
except Exception as exc:
    import traceback
    print("CRITICAL ERROR: Unexpected exception while importing SDB modules.")
    print(f"  {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------

VERSION = "sdb_modular_flat_v2.0.0_with_fixes"

def _load_s2_module(s2_module_path: str):
    """Load s2_optics module from a .py file path (safe for dataclasses / py3.13)."""
    import sys
    import importlib.util
    from pathlib import Path

    p = Path(s2_module_path)
    if not p.exists():
        p = Path(__file__).resolve().parent / s2_module_path
    if not p.exists():
        raise FileNotFoundError(f"Cannot find s2 module: {s2_module_path}")

    # Give it a stable, unique module name based on filename
    mod_name = f"s2_optics_dyn_{p.stem}"

    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {p}")

    mod = importlib.util.module_from_spec(spec)

    # CRITICAL: register in sys.modules *before* executing (needed by dataclasses in py3.13)
    sys.modules[mod_name] = mod

    spec.loader.exec_module(mod)
    return mod


log = logging.getLogger("sdb_main")

# -----------------------------------------------------------------------------
# Waffles Coastline Logic (Integrated)
# -----------------------------------------------------------------------------

# Default waffles resolution (arc-seconds). You can override via WAFFLES_INC_ARCSEC env var.
WAFFLES_INC_ARCSEC = float(os.environ.get("WAFFLES_INC_ARCSEC", "1.0"))

def _run(cmd: str):
    """
    Deprecated wrapper retained for backward compatibility.

    SECURITY: This function will NOT execute via shell=True under any circumstances.
    It attempts to safely parse the command string into argv and then delegates to _run_safe().
    """
    import shlex
    log.warning("[SECURITY] _run() is deprecated. Prefer _run_safe(list_args).")
    try:
        cmd_list = shlex.split(cmd)
    except Exception as e:
        raise RuntimeError(f"[SECURITY] Could not parse command safely (refusing shell=True): {e}")
    return _run_safe(cmd_list)


def _run_safe(cmd_list: list):
    """
    Run a command safely without shell=True (SECURITY FIX).
    
    Args:
        cmd_list: Command as list, e.g., ['python', 'script.py', '--flag=value']
    
    Returns:
        stdout string
    
    Raises:
        RuntimeError: If command fails
    """
    import subprocess
    
    res = subprocess.run(
        cmd_list,
        shell=False,  # SECURITY: Never use shell=True
        capture_output=True,
        text=True,
        check=False
    )
    
    if res.returncode != 0:
        cmd_str = ' '.join(cmd_list)
        raise RuntimeError(
            f"Command failed ({res.returncode}): {cmd_str}\n"
            f"STDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}"
        )
    
    return res.stdout

def _buffer_aoi(aoi, pct: float = 0.05) -> str:
    """Buffer AOI W/E/S/N by a percentage of its width/height."""
    if isinstance(aoi, str):
        parts = aoi.split("/")
        if len(parts) != 4:
            raise ValueError(f"AOI must be 'W/E/S/N', got: {aoi}")
        w, e, s, n = map(float, parts)
    else:
        w, e, s, n = map(float, aoi)
    dx = (e - w) * pct
    dy = (n - s) * pct
    return f"{w - dx:.8f}/{e + dx:.8f}/{s - dy:.8f}/{n + dy:.8f}"

def align_mask_to_target(src_mask: str, target_raster: str, dst_mask: str):
    """Warp src_mask to match target_raster grid (CRS/transform/shape)."""
    import rasterio
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    with rasterio.open(target_raster) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_h, dst_w = ref.height, ref.width
        dst_profile = ref.profile.copy()
        dst_profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            compress="DEFLATE",
            tiled=True,
            nodata=None  # <--- FIXED: 0 is valid Water, NOT NoData
        )

    with rasterio.open(src_mask) as src:
        src_data = src.read(1)

        dst = np.zeros((dst_h, dst_w), dtype=np.uint8)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=None,  # <--- FIXED: Treat source 0 as valid
            dst_nodata=None   # <--- FIXED: Treat destination 0 as valid
        )

    with rasterio.open(dst_mask, "w", **dst_profile) as out:
        out.write(dst, 1)

def generate_coastline_mask(target_raster_path, aoi, cache_masks, out_mask_tif, sdb_mode):
    """
    Generates a coastline mask using Waffles CLI.

    If `target_raster_path` is None: Returns the path to the RAW Waffles output (unaligned).
    If `target_raster_path` is provided: Aligns the Waffles output to the target and returns `out_mask_tif`.
    """
    # 1. Check if final aligned mask already exists and is VALID
    if target_raster_path and out_mask_tif:
        out_mask_tif = Path(out_mask_tif)
        if out_mask_tif.exists() and out_mask_tif.stat().st_size > 0:
            # FIX: Check if the existing mask has the 'nodata=0' bug. If so, delete it.
            try:
                import rasterio
                with rasterio.open(out_mask_tif) as ds:
                    if ds.nodata == 0:
                        log.warning(f"[MASK] Found cached mask with nodata=0 (BUG). Deleting to regenerate: {out_mask_tif}")
                        out_mask_tif.unlink()
                    else:
                        return str(out_mask_tif)
            except Exception:
                pass # Re-generate if check fails

    # 2. Setup Cache and Params
    aoi_buf = _buffer_aoi(aoi, pct=0.05)
    inc_str = f"{WAFFLES_INC_ARCSEC:.9f}s"
    cache_masks = Path(cache_masks)
    cudem_cache_dir = Path(".cudem_cache")
    cudem_cache_dir.mkdir(exist_ok=True)
    cache_masks.mkdir(parents=True, exist_ok=True)

    if sdb_mode == "lakes":
        want_nhd = "False"
        want_lakes = "True"
    elif sdb_mode == "ocean":
        want_nhd = "True"
        want_lakes = "False"
    else:
        want_nhd = "True"
        want_lakes = "True"

    waffles_params = f"want_nhd={want_nhd}:want_lakes={want_lakes}"
    chash = hashlib.sha1(
        f"{aoi_buf}|{WAFFLES_INC_ARCSEC:.9f}|{waffles_params}".encode()
    ).hexdigest()[:12]

    base_prefix = cache_masks / f"waffles_coastline_{chash}"
    base_tif = base_prefix.with_suffix(".tif")

    # 3. Run Waffles if Raw TIF missing
    if (not base_tif.exists()) or base_tif.stat().st_size == 0:
        cmd_list = [
            "waffles",
            "-M",
            f"coastline:{waffles_params}",
            f"-R={aoi_buf}",
            "-E",
            inc_str,
            "-O",
            str(base_prefix),
        ]
        log.info(f"[WAFFLES] Running: {' '.join(cmd_list)}")
        try:
            _run_safe(cmd_list)
        except Exception:
            # Fallback check for glob if name varied slightly
            pass

        if not base_tif.exists():
            # Robust discovery: waffles can append suffixes and/or write into a directory
            patterns = [
                f"waffles_coastline_{chash}*.tif",
                f"{base_prefix.name}*.tif",
                f"{base_prefix.name}*coast*.tif",
                "*coastline*.tif",
            ]
            candidates = []
            seen = set()
            for pat in patterns:
                for p in cache_masks.glob(pat):
                    if p.suffix.lower() != ".tif":
                        continue
                    if p in seen:
                        continue
                    seen.add(p)
                    candidates.append(p)
            # Some waffles versions write into a directory named after the output prefix
            prefix_dir = base_prefix
            if prefix_dir.is_dir():
                for p in prefix_dir.rglob("*.tif"):
                    if p not in seen:
                        seen.add(p)
                        candidates.append(p)
            coasty = [p for p in candidates if "coast" in p.name.lower()]
            if coasty:
                candidates = coasty
            if candidates:
                candidates = sorted(candidates, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
                base_tif = candidates[0]
            else:
                existing = sorted([p.name for p in cache_masks.glob("*.tif")])
                raise RuntimeError(
                    "Waffles did not produce a coastline raster. "
                    f"Expected {base_tif} or a matching *coast*.tif in {cache_masks}. "
                    f"Existing .tif files: {existing}. "
                    "Ensure 'waffles' is in your PATH and that the coastline module ran successfully."
                )

    # 4. Return Logic
    # If no target provided (e.g. pre-S2 download QC), return raw path
    if not target_raster_path:
        return str(base_tif)

    # Otherwise, align to target
    if not out_mask_tif:
        raise ValueError("out_mask_tif must be provided if target_raster_path is set.")

    align_mask_to_target(str(base_tif), target_raster_path, str(out_mask_tif))
    return str(out_mask_tif)

def _setup_file_logging(out_dir: Path):
    """Add a file handler to the logger for this run."""
    log_path = out_dir / "run.log"
    for h in log.handlers[:]:
        if isinstance(h, logging.FileHandler):
            log.removeHandler(h)

    fh = logging.FileHandler(str(log_path))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(fh)

def _write_report(out_dir: Path, status: str, **kwargs):
    """Write a JSON run report."""
    try:
        if RunReport is not None:
            rr = RunReport(out_dir)
            rr.add("run.version", VERSION)
            rr.add("run.status", status)
            rr.merge(kwargs or {})
            rr.write(status=status)
            return
    except Exception:
        pass

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
        "status": status,
        **kwargs
    }
    with open(out_dir / "run_report.json", "w") as f:
        json.dump(report, f, indent=2)


def _rr_add(rr, key: str, value):
    try:
        if rr is not None:
            rr.add(key, value)
    except Exception:
        pass

def _rr_artifact(rr, name: str, path: str):
    try:
        if rr is not None and hasattr(rr, "record_artifact"):
            rr.record_artifact(name, str(path))
    except Exception:
        pass

def _rr_merge_json(rr, key: str, path: Path):
    try:
        if rr is None:
            return
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            with open(p, "r") as f:
                d = json.load(f)
            rr.add_dict(key, d if isinstance(d, dict) else {"value": d})
            _rr_artifact(rr, key.replace(".", "_") + "_json", str(p))
    except Exception:
        pass

def shift_start_back_one_month(start_str: str) -> str:
    """Backoff utility for retry loop."""
    dt = datetime.fromisoformat(start_str)
    if dt.month == 1:
        new_dt = dt.replace(year=dt.year - 1, month=12, day=1)
    else:
        new_dt = dt.replace(month=dt.month - 1, day=1)
    return new_dt.strftime("%Y-%m-%d")

def _save_training_gpkg(out_path: Path, df_all, df_train, df_test):
    """Saves training data artifacts to a multi-layer GeoPackage."""
    log.info(f"[GPKG] Saving training data to {out_path}...")

    def _to_gdf(df):
        if df is None or df.empty: return None
        # Ensure we have coordinates
        if "longitude" not in df.columns or "latitude" not in df.columns:
            return None
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )

    try:
        # Layer 4: All Depths
        gdf_all = _to_gdf(df_all)
        if gdf_all is not None:
            gdf_all.to_file(out_path, layer="all_depths", driver="GPKG")

        # Layer 1: Train Depths
        gdf_train = _to_gdf(df_train)
        if gdf_train is not None:
            gdf_train.to_file(out_path, layer="train_depths", driver="GPKG")

        # Layers 2 & 3: Test Depths (Actual & Predicted)
        gdf_test = _to_gdf(df_test)
        if gdf_test is not None:
            gdf_test.to_file(out_path, layer="test_depths", driver="GPKG")

        log.info("[GPKG] Export complete.")
    except Exception as exc:
        log.warning(f"[GPKG] Failed to save GeoPackage: {exc}")

# -----------------------------------------------------------------------------
# Config Loading Helper
# -----------------------------------------------------------------------------

def load_config_overrides(aoi_bbox: list, config_path: str = "sdb_config.json") -> dict:
    """
    Checks if the AOI center falls into a known region in the config JSON.
    Returns:
        A dict merging 'defaults' + 'region_settings' (if match found).
        Regional settings take precedence.
    """
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r") as f:
            data = json.load(f)

        # 1. Start with global defaults
        config = data.get("defaults", data.get("default", {})).copy()

        # Calculate AOI Center
        w, s, e, n = aoi_bbox
        c_lon = (w + e) / 2.0
        c_lat = (s + n) / 2.0

        # 2. Check specific regions (First Match Wins)
        for region in data.get("regions", []):
            rw, rs, re, rn = region["bbox"]
            if (rw <= c_lon <= re) and (rs <= c_lat <= rn):
                log.info(f"[Config] AOI center ({c_lon:.2f}, {c_lat:.2f}) matches region: {region['name']}")
                log.info(f"[Config] Description: {region['description']}")

                # Merge region settings ON TOP of defaults
                config.update(region["settings"])
                return config

        log.info("[Config] No specific region matched. Using global defaults.")
        return config

    except Exception as e:
        log.warning(f"[Config] Failed to load config: {e}")
        return {}

# -----------------------------------------------------------------------------
# Raster Evaluation & Validation Helpers
# -----------------------------------------------------------------------------

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(mean_squared_error(a[m], b[m]))) if np.any(m) else np.nan

def _r2(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(r2_score(a[m], b[m])) if np.sum(m) >= 2 else np.nan


def _normalize_depth_pair(y_true: np.ndarray, y_pred: np.ndarray):
    """Normalize depths to a consistent positive-down convention for plots/metrics."""
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    m = np.isfinite(yt) & np.isfinite(yp)

    if not np.any(m):
        return yt, yp

    ref_flip = -1.0 if np.nanmedian(yt[m]) < 0 else 1.0
    yt_fixed = yt * ref_flip

    yp_raw_masked = yp[m]
    alignment_flip = -1.0 if np.nanmedian(yp_raw_masked) < 0 else 1.0
    yp_fixed = yp * alignment_flip

    return yt_fixed, yp_fixed

def _calc_metrics(df_test):
    """Returns a dict of basic metrics for a test dataframe."""
    if df_test is None or df_test.empty:
        return {"n":0, "rmse":np.nan, "r2":np.nan, "mae":np.nan}
    y_true = df_test["depth_m"].to_numpy(np.float32)
    y_pred = df_test["depth_pred_m"].to_numpy(np.float32)

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return {"n":0, "rmse":np.nan, "r2":np.nan, "mae":np.nan}

    y_true, y_pred = y_true[m], y_pred[m]
    y_true, y_pred = _normalize_depth_pair(y_true, y_pred)

    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred))
    }

def evaluate_raster_against_points(
    raster_path: str,
    df_points,
    log_dir: Path,
    plot_dir: Path,
    prefix: str = "SDB_Raster_vs_ICESat_Test",
    max_depth: float = None,
) -> dict:
    if df_points is None or df_points.empty:
        log.warning("[EVAL] No points provided for raster evaluation.")
        return {}

    required_cols = {"longitude", "latitude", "depth_m"}
    if not required_cols.issubset(df_points.columns):
        log.warning("[EVAL] Points DataFrame missing required columns for evaluation.")
        return {}

    raster_path = str(raster_path)
    if not os.path.exists(raster_path):
        log.warning(f"[EVAL] Raster not found: {raster_path}")
        return {}

    log.info(f"[EVAL] Evaluating raster '{raster_path}' against {len(df_points)} points.")

    lons = df_points["longitude"].to_numpy(np.float64)
    lats = df_points["latitude"].to_numpy(np.float64)
    z_true = df_points["depth_m"].to_numpy(np.float32)

    with rasterio.open(raster_path) as ds:
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        coords = np.column_stack([xs, ys])

        samples = ds.sample(coords)
        z_rast = np.fromiter((v[0] for v in samples), dtype=np.float32, count=len(coords))

        nodata = ds.nodata
        if nodata is None:
            nodata = -9999.0

    z_pred_raw = z_rast
    m = np.isfinite(z_pred_raw) & np.isfinite(z_true) & (z_pred_raw != nodata)

    if not np.any(m):
        log.warning("[EVAL] No valid overlapping samples between raster and points.")
        return {}

    z_true_m = z_true[m]
    z_pred_m = z_pred_raw[m]

    # FIX: Robust Sign Alignment to prevent massive negative R2
    z_true_m, z_pred_m = _normalize_depth_pair(z_true_m, z_pred_m)

    if z_true_m.size == 0:
        log.warning("[EVAL] No valid overlapping samples between raster and points.")
        return {}

    # Metrics on ALL overlap samples
    rmse_all = _rmse(z_true_m, z_pred_m)
    mae_all = float(mean_absolute_error(z_true_m, z_pred_m))
    bias_all = float(np.nanmean(z_pred_m - z_true_m))
    r2_all = _r2(z_true_m, z_pred_m)
    n_all = int(len(z_true_m))

    log.info(f"[EVAL] Raster vs points (ALL) – RMSE={rmse_all:.2f} m, MAE={mae_all:.2f} m, R²={r2_all:.2f}, N={n_all}")

    out_png = plot_dir / f"{prefix}.png"

    # Optional metrics on limited depth-of-support
    rmse_val, mae_val, bias_val, r2_val, n_val = rmse_all, mae_all, bias_all, r2_all, n_all
    if max_depth is not None:
        md = float(max_depth)
        mlim = (np.abs(z_true_m) <= md)
        if np.any(mlim):
            zt = z_true_m[mlim]
            zp = z_pred_m[mlim]
            rmse_val = _rmse(zt, zp)
            mae_val = float(mean_absolute_error(zt, zp))
            bias_val = float(np.nanmean(zp - zt))
            r2_val = _r2(zt, zp)
            n_val = int(len(zt))
            log.info(f"[EVAL] Raster vs points (|depth|≤{md:g}m) – RMSE={rmse_val:.2f} m, MAE={mae_val:.2f} m, R²={r2_val:.2f}, N={n_val}")
            out_png = plot_dir / f"{prefix}.png"

    train.scatter_plot(
        y_true=z_true_m,
        y_pred=z_pred_m,
        title="Raster vs ICESat-2 Test",
        out_png=out_png,
        max_depth=max_depth,
        x_label="ICESat-2 Depth (m)",
        y_label="SDB Raster Depth (m)"
    )

    metrics = {"all": {"rmse_m": rmse_all, "mae_m": mae_all, "bias_m": bias_all, "r2": r2_all, "n": n_all}, "limited": {"rmse_m": rmse_val, "mae_m": mae_val, "bias_m": bias_val, "r2": r2_val, "n": n_val}, "max_depth_m": (None if max_depth is None else float(max_depth))}
    with open(log_dir / "eval_metrics_raster_vs_icesat_test.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

def _deg_per_meter(lat_deg: float):
    """Approx degrees per meter for (lon, lat) at a given latitude."""
    lat_rad = np.deg2rad(lat_deg)
    dlat = 1.0 / 111_320.0
    dlon = 1.0 / (111_320.0 * max(np.cos(lat_rad), 1e-6))
    return dlon, dlat

def _overlap_or_adjacent_buffer_bbox(a: dict, b: dict, buffer_m: float, touch_tol_deg: float = 1e-6):
    """
    Returns a WESN bbox representing the *overlap region* between AOIs.
    """
    aw, ae, as_, an = a["w"], a["e"], a["s"], a["n"]
    bw, be, bs, bn = b["w"], b["e"], b["s"], b["n"]

    # True intersection
    w = max(aw, bw); e = min(ae, be); s = max(as_, bs); n = min(an, bn)
    if (e - w) > 0 and (n - s) > 0:
        return (w, e, s, n)

    # If adjacent, create a buffered strip
    c_lat = (max(as_, bs) + min(an, bn)) / 2.0
    dlon, dlat = _deg_per_meter(c_lat)

    # East/West adjacency
    if abs(ae - bw) <= touch_tol_deg or abs(be - aw) <= touch_tol_deg:
        boundary = ae if abs(ae - bw) <= touch_tol_deg else aw
        w2 = boundary - buffer_m * dlon
        e2 = boundary + buffer_m * dlon
        s2 = max(as_, bs)
        n2 = min(an, bn)
        if (n2 - s2) > 0:
            return (w2, e2, s2, n2)

    # North/South adjacency
    if abs(an - bs) <= touch_tol_deg or abs(bn - as_) <= touch_tol_deg:
        boundary = an if abs(an - bs) <= touch_tol_deg else as_
        s2 = boundary - buffer_m * dlat
        n2 = boundary + buffer_m * dlat
        w2 = max(aw, bw)
        e2 = min(ae, be)
        if (e2 - w2) > 0:
            return (w2, e2, s2, n2)

    return None

def _sample_raster_lonlat(raster_path: str, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Sample a single-band raster at lon/lat points. Returns float32 array with NaNs for nodata."""
    import rasterio
    from pyproj import Transformer

    with rasterio.open(raster_path) as ds:
        crs = ds.crs
        nod = ds.nodata
        if crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = tfm.transform(lons, lats)
        vals = np.fromiter(
            (v[0] for v in ds.sample(zip(xs, ys), indexes=1)),
            dtype=np.float32,
            count=len(lons),
        )
        if nod is not None and np.isfinite(nod):
            vals[vals == np.float32(nod)] = np.nan
        return vals

def overlap_consistency_diagnostic(
    *,
    current_run_aoi: dict,
    neighbor_run_aoi: dict,
    current_pred_tif: str,
    neighbor_pred_tif: str,
    buffer_m: float = 500.0,
    sample_spacing_m: float = 30.0,
    max_samples: int = 200_000,
    df_ref_points=None,
    ref_depth_col: str = "depth_m",
) -> dict:
    """Compare predictions between adjacent/overlapping runs."""
    bbox = _overlap_or_adjacent_buffer_bbox(current_run_aoi, neighbor_run_aoi, buffer_m=buffer_m)
    if bbox is None:
        return {"status": "skip", "reason": "no overlap or adjacency detected"}

    w, e, s, n = bbox
    c_lat = (s + n) / 2.0
    dlon, dlat = _deg_per_meter(c_lat)

    step_lon = max(sample_spacing_m * dlon, 1e-6)
    step_lat = max(sample_spacing_m * dlat, 1e-6)

    lons = np.arange(w, e, step_lon, dtype=np.float64)
    lats = np.arange(s, n, step_lat, dtype=np.float64)
    if lons.size == 0 or lats.size == 0:
        return {"status": "skip", "reason": "overlap region too small"}

    Lon, Lat = np.meshgrid(lons, lats)
    pts_lon = Lon.ravel()
    pts_lat = Lat.ravel()

    total = pts_lon.size
    if total > max_samples:
        stride = int(np.ceil(total / max_samples))
        pts_lon = pts_lon[::stride]
        pts_lat = pts_lat[::stride]

    cur = _sample_raster_lonlat(current_pred_tif, pts_lon, pts_lat)
    nb = _sample_raster_lonlat(neighbor_pred_tif, pts_lon, pts_lat)

    m = np.isfinite(cur) & np.isfinite(nb)
    if not np.any(m):
        return {"status": "skip", "reason": "no overlapping valid pixels in both rasters"}

    diff = cur[m] - nb[m]
    absdiff = np.abs(diff)

    out = {
        "status": "ok",
        "bbox_wesn": [float(w), float(e), float(s), float(n)],
        "buffer_m": float(buffer_m),
        "sample_spacing_m": float(sample_spacing_m),
        "n_samples_total": int(len(cur)),
        "n_samples_valid": int(np.count_nonzero(m)),
        "median_diff_m": float(np.nanmedian(diff)),
        "mean_diff_m": float(np.nanmean(diff)),
        "p90_abs_diff_m": float(np.nanpercentile(absdiff, 90)),
        "p99_abs_diff_m": float(np.nanpercentile(absdiff, 99)),
    }

    if df_ref_points is not None and len(df_ref_points) > 0 and ref_depth_col in df_ref_points.columns:
        lon_col = "lon" if "lon" in df_ref_points.columns else ("longitude" if "longitude" in df_ref_points.columns else None)
        lat_col = "lat" if "lat" in df_ref_points.columns else ("latitude" if "latitude" in df_ref_points.columns else None)
        if lon_col and lat_col:
            dfr = df_ref_points.copy()
            dfr = dfr[(dfr[lon_col] >= w) & (dfr[lon_col] <= e) & (dfr[lat_col] >= s) & (dfr[lat_col] <= n)]
            if len(dfr) > 20:
                rlons = dfr[lon_col].to_numpy(dtype=np.float64)
                rlats = dfr[lat_col].to_numpy(dtype=np.float64)
                z = dfr[ref_depth_col].to_numpy(dtype=np.float64)
                pc = _sample_raster_lonlat(current_pred_tif, rlons, rlats).astype(np.float64)
                pn = _sample_raster_lonlat(neighbor_pred_tif, rlons, rlats).astype(np.float64)
                mc = np.isfinite(pc) & np.isfinite(z)
                mn = np.isfinite(pn) & np.isfinite(z)
                if np.any(mc):
                    out["rmse_vs_ref_current_m"] = float(np.sqrt(np.mean((pc[mc] - z[mc]) ** 2)))
                    out["n_ref_current"] = int(np.count_nonzero(mc))
                if np.any(mn):
                    out["rmse_vs_ref_neighbor_m"] = float(np.sqrt(np.mean((pn[mn] - z[mn]) ** 2)))
                    out["n_ref_neighbor"] = int(np.count_nonzero(mn))

    return out

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    import rasterio
    args = parse_args()

    try:
        w, e, s, n = [float(x) for x in args.aoi.split("/")]
        bbox_list = [w, s, e, n]
        bbox_wesn = (w, e, s, n)
    except Exception:
        log.error("Invalid AOI format. Use W/E/S/N")
        sys.exit(1)

    # ------------------------------------------------------------------
    # NEW: Validate configuration before processing
    # ------------------------------------------------------------------
    # Build config dict for validation and checkpointing
    config_dict = {
        "aoi": args.aoi,
        "start_date": getattr(args, "start", None),
        "end_date": getattr(args, "end", None),
        "max_depth_sdb": getattr(args, "max_depth_sdb", None),
        "rmse_target_sdb": getattr(args, "rmse_target_sdb", None),
        "depth_bin_m": getattr(args, "depth_bin_m", None),
        "out_dir": getattr(args, "out_dir", None),
        "cache_root": getattr(args, "cache_root", None),
    }
    
    if VALIDATION_AVAILABLE:
        validation_result = validate_pipeline_config(config_dict)
        if not validation_result.is_valid:
            log.error("Configuration validation failed. Exiting.")
            for err in validation_result.errors:
                log.error(f"  - {err}")
            sys.exit(1)
        elif validation_result.warnings:
            for warn in validation_result.warnings:
                log.warning(f"  - {warn}")
    
    # ------------------------------------------------------------------
    # NEW: Initialize checkpoint manager if available
    # ------------------------------------------------------------------
    checkpoint = None
    if CHECKPOINTS_AVAILABLE and not getattr(args, "no_checkpoints", False):
        try:
            out_dir = Path(getattr(args, "out_dir", "output"))
            checkpoint = PipelineCheckpoint(
                output_dir=out_dir,
                config=config_dict,
                enabled=True
            )
            progress = checkpoint.get_progress_summary()
            if progress.get("completed_stages"):
                log.info(f"[CHECKPOINT] Resuming from checkpoint: {progress['completed_stages']}")
        except Exception as e:
            log.warning(f"[CHECKPOINT] Could not initialize: {e}")
            checkpoint = None


    # ------------------------------------------------------------------
    # Resolve max_depth_sdb (supports numeric or 'auto')
    # ------------------------------------------------------------------
    max_depth_arg_raw = getattr(args, "max_depth_sdb", "20.0")
    auto_max_depth = False
    max_depth_sdb_product = 20.0

    try:
        if isinstance(max_depth_arg_raw, str) and max_depth_arg_raw.strip().lower() == "auto":
            auto_max_depth = True
            max_depth_sdb_product = 20.0
        else:
            max_depth_sdb_product = float(max_depth_arg_raw)
    except Exception:
        log.warning(f"[Config] Invalid --max-depth-sdb value '{max_depth_arg_raw}'. Falling back to 20.0 m.")
        max_depth_sdb_product = 20.0

    MAX_DEPTH_TRAIN_FETCH = max(30.0, max_depth_sdb_product)

    # --- UPDATED LOGIC: Allow auto-depth without forced spatial validation ---
    if auto_max_depth and not getattr(args, "validate_spatial", True):
        log.info("[Config] --max-depth-sdb auto requested with --no-validate-spatial. "
                 "Using STRATIFIED RANDOM SPLIT to estimate depth-of-support (interpolation limit).")
        # Do NOT force validate_spatial = True

    # --- AUTO-CONFIGURATION OVERRIDE ---
    config_settings = load_config_overrides(bbox_list)
    active_profile = detect_active_config_profile(bbox_list)
    overrides_applied = list(config_settings.keys()) if isinstance(config_settings, dict) else []

    if config_settings:
        log.info("--- Applying Regional Configuration ---")

        for key, value in config_settings.items():
            if key == "preferred_months":
                setattr(args, "preferred_months", value)
                log.info(f"   [Auto-Config] Set preferred_months: {value}")
                continue

            if not hasattr(args, key):
                # We log this but still set it on args so it appears in the final dump
                # (useful for custom flags like 'gl_red_max' not in argparse)
                setattr(args, key, value)
                log.info(f"   [Auto-Config] Set {key} (custom): {value}")
                continue

            cli_flag = "--" + key.replace("_", "-")
            if key == "min_depth_atl03_val": cli_flag = "--min-depth-atl03"
            elif key == "refraction":
                if not value: cli_flag = "--no-refraction"

            if cli_flag in sys.argv:
                log.info(f"   [Override Skipped] User explicitly set {cli_flag}. Keeping value: {getattr(args, key)}")
            else:
                old_val = getattr(args, key)
                setattr(args, key, value)
                if old_val != value:
                    log.info(f"   [Auto-Config] Set {key}: {old_val} -> {value}")

        log.info("---------------------------------------")

    # --- UPDATED: Comprehensive Grouped Argument Printer ---
    log.info("\n=== Final Effective Arguments ===")

    # Defined Groups matching sdb_config.json + argparse keys
    arg_groups = {
        "Core": [
            "aoi", "start", "end", "out_dir", "sdb_mode", "s2_module"
        ],
        "Caching": [
            "cache_root", "cache_strict", "cache_code_strict"
        ],
        "Sentinel-2 / Cloud": [
            "cloud", "s2_scene_limit", "s2_only", "min_scene_valid_frac",
            "stac_max_items", "stac_page_limit", "cloud_report_only", "preferred_months",
            "edge_weight_power", "edge_weight_min", "temporal_median_k"
        ],
        "Bright Pixel / Turbidity Handling": [
            "mask_bright_pixels",
            "allow_bright_shallow_pixels", "bright_shallow_nir_max",
            "apply_gl_turbidity_reject", "gl_nir_max", "gl_nir_green_ratio_max", "gl_red_max"
        ],
        "Optical / Masks": [
            "land_mask_user", "cw_min", "land_max",
            "land_mask_water_val", "land_mask_invert", "land_mask_type", "land_mask_threshold"
        ],
        "ICESat-2": [
            "icesat", "atl_only", "water_class", "atl03_conf_min", "min_depth_atl03_val",
            "max_depth_atl03", "refraction", "water_temp_c", "debug_atl03_qc",
            "atl24_conf_min", "force_redl_atl24", "write_atl03_tracks_shp",
            "min_bottom_photons", "min_bottom_frac"
        ],
        "Training / Validation": [
            "max_depth_sdb", "min_training_points_for_sdb", "use_stumpf_depth", "seed",
            "validate_spatial", "rmse_target_sdb", "depth_bin_m", "min_samples_per_bin",
            "depth_binning",
            "linf_enabled", "linf_estimate_deepwater", "linf_percentile",
            "linf_deepwater_nir_max", "linf_deepwater_bright_max"
        ],
        "Prediction": [
            "tile", "s2_smooth_kernel", "no_doa", "skip_nad83"
        ]
    }

    printed_keys = set()

    for group_name, keys in arg_groups.items():
        log.info(f"--- {group_name} ---")
        for k in keys:
            if hasattr(args, k):
                val = getattr(args, k)
                log.info(f"  {k}: {val}")
                printed_keys.add(k)
            # Handle config keys that might not be in argparse definition (dynamically added)
            elif k in vars(args):
                 val = getattr(args, k)
                 log.info(f"  {k}: {val}")
                 printed_keys.add(k)

    # Catch-all for anything else not in groups
    all_keys = set(vars(args).keys())
    remaining = sorted(list(all_keys - printed_keys))
    if remaining:
        log.info("--- Other / Uncategorized ---")
        for k in remaining:
            # Skip hidden private attrs
            if not k.startswith("_"):
                log.info(f"  {k}: {getattr(args, k)}")

    log.info("=================================\n")


    # ------------------------------------------------------------
    # Enforce lake-mode defaults & constraints
    # ------------------------------------------------------------
    if getattr(args, "sdb_mode", None) == "lakes":
        log.info("[MODE] Lake mode enabled")
        if getattr(args, "icesat", None) in ("atl24", "all_atl"):
            log.info("[MODE] ATL24 disabled in lake mode (not defined over lakes)")
            args.icesat = "atl03"
        if getattr(args, "cw_min", None) is None:
            args.cw_min = 0.8
            log.info("[MODE] Setting default cw_min=0.8 for lakes")
        if getattr(args, "land_max", None) is None:
            args.land_max = 0.1
            log.info("[MODE] Setting default land_max=0.1 for lakes")

    # Load requested s2_optics implementation
    s2_optics = _load_s2_module(getattr(args, "s2_module", "s2_optics.py"))

    # 1. Setup Organized Output Structure
    if args.out_dir:
        out_root = Path(args.out_dir)
        # Ensure a stable run directory name for logs/reports/titles even when user supplies --out-dir
        dir_name = out_root.name
    else:
        dir_name = f"sdb_{w:.3f}_{e:.3f}_{s:.3f}_{n:.3f}_{args.start}_{args.end}"
        out_root = Path("output") / dir_name
    dir_data = out_root / "data"
    dir_model = out_root / "model"
    dir_rast = out_root / "rasters"
    dir_plot = out_root / "plots"
    dir_logs = out_root / "logs"

    for d in [dir_data, dir_model, dir_rast, dir_plot, dir_logs]:
        d.mkdir(parents=True, exist_ok=True)

    _setup_file_logging(dir_logs)

    # ------------------------------------------------------------------
    # RunReport (structured run_report.json)
    # ------------------------------------------------------------------
    rr = RunReport(out_root) if RunReport is not None else None
    _rr_add(rr, "run.version", VERSION)
    _rr_add(rr, "run.args", vars(args))
    _rr_add(rr, "sdb.max_depth.auto.rmse_target_m", float(getattr(args,'rmse_target_sdb',0.5)))
    _rr_add(rr, "run.aoi", {"w": w, "e": e, "s": s, "n": n})
    _rr_add(rr, "config.active_profile", active_profile if "active_profile" in locals() else "unknown")
    _rr_add(rr, "config.overrides_applied", overrides_applied if "overrides_applied" in locals() else [])

    _rr_artifact(rr, "out_root", str(out_root))
    _rr_artifact(rr, "dir_data", str(dir_data))
    _rr_artifact(rr, "dir_model", str(dir_model))
    _rr_artifact(rr, "dir_rasters", str(dir_rast))
    _rr_artifact(rr, "dir_plots", str(dir_plot))
    _rr_artifact(rr, "dir_logs", str(dir_logs))


    cache_root = Path(getattr(args, "cache_root", "cache"))
    atl_cache = cache_root / "icesat2"
    s2_cache = cache_root / "sentinel2"
    mask_cache = cache_root / "masks"

    for _d in (atl_cache, s2_cache, mask_cache):
        try:
            Path(_d).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    log.info(f"--- Starting SDB Run [{args.sdb_mode}] ---")
    log.info(f"Output Directory: {out_root}")

    # 3. Data Acquisition
    current_start = args.start
    end_date = args.end
    s2_paths = None
    atl_files_map = {"ATL03": [], "ATL24": []}
    found_data = False

    pref_months = getattr(args, "preferred_months", None)

    s2_cache_dir = s2_cache if args.out_dir is None else out_root / "cache_s2"
    atl_cache_dir = atl_cache if args.out_dir is None else out_root / "cache_icesat2"
    mask_cache_dir = mask_cache if args.out_dir is None else out_root / "cache_masks"


    for attempt in range(6):
        log.info(f"\n[Acquisition] Attempt {attempt+1}: {current_start} to {end_date}")

        if (not args.atl_only) and (s2_paths is None):
            try:
                job_name = f"S2_{w:.3f}_{e:.3f}_{s:.3f}_{n:.3f}_{current_start}_{end_date}"
                s2_out = s2_cache / job_name

                coast_mask_raw = None
                if args.land_mask_user:
                    coast_mask_raw = None
                else:
                    try:
                        waffles_cache = Path(os.environ.get("WAFFLES_CACHE_ROOT", str(mask_cache)))
                        coast_mask_raw = generate_coastline_mask(None, args.aoi, waffles_cache, None, args.sdb_mode)
                        log.info(f"[MASK] Using waffles coastline mask for S2 date QC: {coast_mask_raw}")
                    except Exception as exc:
                        log.warning(f"[MASK] Waffles generation failed ({exc}). S2 date QC will be AOI-only.")
                        coast_mask_raw = None


                # Build S2 composite (filter kwargs for compatibility with custom s2 modules)
                glint_vis = [x.strip() for x in re.split(r"[,\s]+", str(getattr(args, "glint_vis_bands", "B02,B03,B04")).strip()) if x.strip()]
                s2_kwargs = dict(
                    out_dir=s2_out,
                    bbox_wesn=bbox_wesn,
                    start_date=current_start,
                    end_date=end_date,
                    max_cloud=args.cloud,
                    scene_limit=args.s2_scene_limit,
                    preferred_months=pref_months,
                    shared_date_mode="strict",
                    stac_max_items=args.stac_max_items,
                    stac_page_limit=args.stac_page_limit,
                    cache_dir=s2_cache_dir,
                    cache_strict=getattr(args, "cache_strict", True),
                    cache_code_strict=getattr(args, "cache_code_strict", False),
                    cache_ignore_code=getattr(args, "cache_ignore_code", True),
                    min_scene_valid_frac=args.min_scene_valid_frac,
                    edge_weight_power=args.edge_weight_power,
                    edge_weight_min=args.edge_weight_min,
                    allow_bright_shallow_pixels=args.allow_bright_shallow_pixels,
                    bright_shallow_nir_max=args.bright_shallow_nir_max,
                    apply_gl_turbidity_reject=args.apply_gl_turbidity_reject,
                    gl_nir_max=args.gl_nir_max,
                    gl_nir_green_ratio_max=args.gl_nir_green_ratio_max,
                    gl_red_max=args.gl_red_max,
                    b02_thresh=(args.mask_bright_pixels if args.mask_bright_pixels is not None else 0.25),
                    coastline_mask_path=coast_mask_raw,
                    coastline_mask_water_value=0,
                    coastline_mask_invert=False,
                    coastline_erode_px=getattr(args, "coastline_erode_px", 3),
                    harmonize=True,
                    harmonize_max_abs_offset=0.03,
                    temporal_median_k=args.temporal_median_k,
                    # Glint correction (optional)
                    glint_correct=getattr(args, "glint_correct", False),
                    glint_nir_band=getattr(args, "glint_nir_band", "B08"),
                    glint_vis_bands=glint_vis if glint_vis else None,
                    glint_nir_min_percentile=float(getattr(args, "glint_nir_min_percentile", 1.0)),
                    glint_deepwater_b02_max=float(getattr(args, "glint_deepwater_b02_max", 0.20)),
                    glint_min_samples=int(getattr(args, "glint_min_samples", 5000)),
                    glint_max_samples=int(getattr(args, "glint_max_samples", 2000000)),
                    glint_clip_min=float(getattr(args, "glint_clip_min", 1e-6)),
                )
                try:
                    sig = inspect.signature(s2_optics.build_weighted_shared_date_composite)
                    allowed = set(sig.parameters.keys())
                    s2_kwargs = {k: v for k, v in s2_kwargs.items() if k in allowed}
                except Exception:
                    pass

                s2_paths = s2_optics.build_weighted_shared_date_composite(**s2_kwargs)


                try:
                    if rr is not None and isinstance(s2_paths, dict):
                        rep = s2_paths.pop("_report", None)
                        if isinstance(rep, dict):
                            rr.add_dict("s2.composite", rep)
                        for k, v in s2_paths.items():
                            if isinstance(v, str) and v:
                                _rr_artifact(rr, f"s2_{k}".lower(), v)
                        _rr_merge_json(rr, "s2.date_qc", s2_out / "S2_DATE_QC.json")
                except Exception:
                    pass
                log.info("[S2] Composite acquired.")

                if getattr(args, 'cloud_report_only', False):
                    log.info('[S2] Cloud report only requested; exiting after STAC scan.')
                    return

                rgb_src = s2_out / "RGB_10m.tif"
                if rgb_src.exists():
                    rgb_dst = dir_rast / "RGB_10m.tif"
                    shutil.copy(rgb_src, rgb_dst)
                else:
                    log.warning(f"[S2] RGB_10m.tif not found in {s2_out}")

            except Exception as exc:
                log.warning(f"[S2] Failed: {exc}")
                s2_paths = None

        if (not args.atl_only) and (s2_paths is not None):
            log.info("[S2] Reusing existing composite from earlier attempt; skipping S2 reacquisition.")

        if not args.s2_only:
            atl_run_name = f"ATL_{w:.3f}_{e:.3f}_{s:.3f}_{n:.3f}_{current_start}_{end_date}"
            atl_run_dir = atl_cache / atl_run_name

            try:
                if args.icesat in ["atl03", "all_atl"]:
                    target_dir_03 = atl_run_dir / "ATL03"
                    f03, _, _ = atl.ensure_icesat_files_harmony_cachefirst(
                        target_dir_03, "ATL03", bbox_list, current_start, end_date, rr=rr
                    )
                    atl_files_map["ATL03"] = f03
                if args.icesat in ["atl24", "all_atl"]:
                    if getattr(args, "sdb_mode", None) == "lakes":
                        log.info("[ATL] Skipping ATL24 in lake mode (not defined over lakes).")
                        atl_files_map["ATL24"] = []
                    else:
                        target_dir_24 = atl_run_dir / "ATL24"
                        f24, _, _ = atl.ensure_icesat_files_harmony_cachefirst(
                            target_dir_24, "ATL24", bbox_list, current_start, end_date,
                            force_redl=args.force_redl_atl24
                        )
                        atl_files_map["ATL24"] = f24

            except Exception as exc:
                log.warning(f"[ATL] Failed: {exc}")

        has_s2 = (s2_paths is not None) or args.atl_only
        has_atl = (len(atl_files_map["ATL03"]) > 0 or len(atl_files_map["ATL24"]) > 0) or args.s2_only

        if has_s2 and has_atl:
            found_data = True
            break
        current_start = shift_start_back_one_month(current_start)

    if not found_data:
        log.error("Could not find S2 and/or ATL data after retries.")
        sys.exit(1)

    if args.s2_only or args.atl_only:
        return

    # 4. Mask Generation
    log.info("\n--- Mask Generation ---")
    land_mask_out = dir_rast / "LAND_MASK_aligned.tif"
    waffles_cache = Path(os.environ.get("WAFFLES_CACHE_ROOT", mask_cache))

    if args.land_mask_user:
        s2_optics.prepare_user_land_mask(args.land_mask_user, s2_paths["B02"], str(land_mask_out))
    else:
        try:
            generate_coastline_mask(s2_paths["B02"], args.aoi, waffles_cache, land_mask_out, args.sdb_mode)
        except Exception as exc:
            log.warning(f"Mask generation failed ({exc}).")

    log.info(f"[MASK] Using LAND mask as hard gate: keep pixels where LAND <= {args.land_max} (0.0 = waffles water-only).")

    # 5. ATL Processing
    log.info("\n--- Processing ICESat-2 Data ---")
    df_atl03 = None
    df_atl24 = None

    land_mask_water_val = int(args.land_mask_water_val) if getattr(args, "land_mask_water_val", None) is not None else 0
    land_mask_invert = bool(getattr(args, "land_mask_invert", False))
    land_mask_type = str(getattr(args, "land_mask_type", "auto"))
    land_mask_threshold = float(getattr(args, "land_mask_threshold", 0.5))

    if rr is not None:
        try:
            rr.add("mask.land_mask_water_val", land_mask_water_val)
            rr.add("mask.land_mask_invert", land_mask_invert)
            rr.add("mask.land_mask_type", land_mask_type)
            rr.add("mask.land_mask_threshold", land_mask_threshold)
        except Exception:
            pass


    if atl_files_map["ATL03"]:
        if args.write_atl03_tracks_shp:
            atl.build_atl03_track_lines(atl_files_map["ATL03"], args.aoi, str(dir_data / "ATL03_tracks.shp"))

        df_atl03 = atl.collect_training_points_from_atl03(
            atl_files_map["ATL03"], lat_res=0.00005, height_res=0.25, aoi_str=args.aoi,
            atl03_conf_min=args.atl03_conf_min, atl03_bottom_percentile=90.0,
            use_refraction=args.refraction, default_temp_c=args.water_temp_c,
            default_wavelength_nm=532.0, min_bottom_photons=args.min_bottom_photons,
            min_bottom_frac=args.min_bottom_frac,
            min_depth_m=args.min_depth_atl03_val,
            max_depth_m=args.max_depth_atl03,
            debug_atl03_qc=args.debug_atl03_qc,
            land_mask_path=str(land_mask_out),
            land_mask_water_val=land_mask_water_val,
            land_mask_invert=land_mask_invert,
            cache_dir=str(atl_cache_dir),
            cache_strict=getattr(args, 'cache_strict', True),
            cache_code_strict=getattr(args, 'cache_code_strict', False),
        rr=rr,
    )

    if atl_files_map["ATL24"]:
        df_atl24 = atl.collect_training_points_from_atl24(
            atl_files_map["ATL24"], aoi_str=args.aoi,
            max_depth_m=MAX_DEPTH_TRAIN_FETCH,
            train_relax_buffer=0.01, limit_train_samples=None, seed=args.seed,
            atl24_conf_min=args.atl24_conf_min,
            land_mask_path=str(land_mask_out),
            land_mask_water_val=land_mask_water_val,
            land_mask_invert=land_mask_invert,
            cache_dir=str(atl_cache_dir),
            cache_strict=getattr(args, 'cache_strict', True),
            cache_code_strict=getattr(args, 'cache_code_strict', False),
        rr=rr,
    )

    # Optional: harmonize ATL horizontal CRS before fusion (e.g., WGS84 -> NAD83).
    # NOTE: This is HORIZONTAL-ONLY. depth_m is NOT transformed because it's relative
    # water depth, not a geodetic height. Transforming depth would corrupt values.
    if getattr(args, "atl_transform_datum", False):
        df_atl03 = atl.transform_xyz_dataframe_crs(df_atl03, src_srs=args.atl_src_srs, dst_srs=args.atl_dst_srs, logger=log)
        df_atl24 = atl.transform_xyz_dataframe_crs(df_atl24, src_srs=args.atl_src_srs, dst_srs=args.atl_dst_srs, logger=log)

    # 6. Fusion
    log.info("\n--- Fusion & QC ---")
    df_xyz = None
    if args.extra_xyz:
        df_xyz = atl.load_extra_xyz_points(args.extra_xyz, args.extra_xyz_crs, args.aoi)
        try:
            n_xyz = 0 if df_xyz is None else len(df_xyz)
            if n_xyz > 0:
                dmin = float(np.nanmin(df_xyz["depth_m"])) if "depth_m" in df_xyz.columns else float("nan")
                dmed = float(np.nanmedian(df_xyz["depth_m"])) if "depth_m" in df_xyz.columns else float("nan")
                dmax = float(np.nanmax(df_xyz["depth_m"])) if "depth_m" in df_xyz.columns else float("nan")
                log.info(f"[XYZ] Loaded extra XYZ points: n={n_xyz} depth(min/med/max)={dmin:.3f}/{dmed:.3f}/{dmax:.3f} m")
            else:
                log.warning("[XYZ] Extra XYZ inputs were provided but produced 0 usable points after parsing/filtering.")
        except Exception:
            log.warning("[XYZ] Loaded extra XYZ points (stats unavailable).", exc_info=True)

    fused_df = fusion.build_fused_training_dataframe(
        atl03_df=df_atl03, atl24_df=df_atl24, xyz_df=df_xyz,
        xyz_max_dist_m=30.0, atl_max_dist_m=20.0,
        atl03_weight=args.atl03_weight_factor, atl24_weight=1.0, xyz_weight=10.0,
        rr=rr,
        # Adaptive sampling parameters (v0.7.0)
        enable_adaptive_sampling=getattr(args, 'enable_adaptive_sampling', False),
        sampling_target_points=getattr(args, 'sampling_target_points', 2000),
        sampling_min_threshold=getattr(args, 'sampling_min_threshold', 3000),
        sampling_max_gap_m=getattr(args, 'sampling_max_gap_m', 100.0),
    )

    # Helpful provenance summary (especially to confirm external XYZ participation)
    try:
        if fused_df is not None and len(fused_df) > 0:
            if "source" in fused_df.columns:
                src_counts = fused_df["source"].value_counts(dropna=False).to_dict()
                log.info(f"[FUSION] Point provenance after fusion: {src_counts}")
            else:
                log.info(f"[FUSION] Fused training points: n={len(fused_df)} (no 'source' column present)")
        else:
            log.warning("[FUSION] Fusion produced 0 points. Check water/land masks, AOI, and input data.")
    except Exception:
        log.warning("[FUSION] Unable to summarize point provenance.", exc_info=True)
    fused_df.to_csv(dir_data / "training_data_fused.csv", index=False)

    # Helpful provenance logging: how many fused points came from each source.
    try:
        if "source" in fused_df.columns:
            src_counts = fused_df["source"].value_counts(dropna=False).to_dict()
            log.info(f"[FUSION] Fused point sources: {src_counts}")
        else:
            log.info("[FUSION] Fused dataframe has no 'source' column; provenance counts unavailable.")
    except Exception:
        log.warning("[FUSION] Could not compute fused source counts.", exc_info=True)

    if fused_df.empty:
        log.error("No valid training points found after Fusion.")
        return

    # 7. Training & Validation
    log.info("\n--- Model Training & Validation ---")
    df_with_s2 = train.sample_s2_bands_at_points(fused_df, s2_paths, str(land_mask_out))

    # After S2 sampling/masking, re-log provenance to confirm XYZ is still present.
    try:
        if "source" in df_with_s2.columns:
            src_counts2 = df_with_s2["source"].value_counts(dropna=False).to_dict()
            log.info(f"[TRAIN][PROVENANCE] After S2 sampling/masking: {src_counts2}")
    except Exception:
        log.warning("[TRAIN][PROVENANCE] Could not compute source counts after S2 sampling.", exc_info=True)

    log.info("[TRAIN] Running Standard Training (Random Split)...")

    linf_est_mode = "deepwater" if args.linf_estimate_deepwater else "none"

    land_mask_type_for_train = str(getattr(args, "land_mask_type", "auto"))
    land_mask_water_val_for_train = getattr(args, "land_mask_water_val", None)
    if land_mask_water_val_for_train is not None:
        land_mask_water_val_for_train = int(land_mask_water_val_for_train)


    # UPDATED: We directly respect 'validate_spatial' flag now.
    # If auto_max_depth is True, train.py handles the calculation regardless of split type.
    rf, stumpf_lr, df_train_final, df_test_final, model_meta = train.train_sdb_model(
        train_df=df_with_s2,
        max_depth_sdb=max_depth_sdb_product,
        seed=args.seed,
        plots_dir=dir_plot, water_class=args.water_class,
        use_stumpf_depth=args.use_stumpf_depth,
        min_training_points_for_sdb=args.min_training_points_for_sdb,
        spatial_split=False,  # Production model uses stratified random split
        cw_min=args.cw_min,
        land_max=args.land_max,
        land_mask_type=land_mask_type_for_train,
        land_mask_water_val=land_mask_water_val_for_train,
        land_mask_invert=land_mask_invert,
        linf_enabled=args.linf_enabled,
        linf_estimate=linf_est_mode,
        linf_deepwater_nir_max=args.linf_deepwater_nir_max,
        linf_deepwater_bright_max=args.linf_deepwater_bright_max,
        linf_percentile=args.linf_percentile,
        rmse_target_sdb=getattr(args,'rmse_target_sdb',1.0),
        raster_paths=s2_paths,
        depth_bin_m=float(getattr(args, 'depth_bin_m', 1.0)),
        depth_binning=str(getattr(args, 'depth_binning', 'quantile')),
        min_samples_per_bin=int(getattr(args, 'min_samples_per_bin', 10)),
    )

    if df_train_final.empty:
        log.error("Training failed (0 samples).")
        return

    src_rand_base = dir_plot / "SDB_Accuracy_Assessment_RandomSplit.png"
    if src_rand_base.exists():
        src_rand_base.rename(dir_plot / "Validation_RandomSplit.png")

    src_rand_v1 = dir_plot / "SDB_Accuracy_Assessment_RandomSplit_V1_AllData.png"
    if src_rand_v1.exists():
        src_rand_v1.rename(dir_plot / "Validation_RandomSplit_V1_AllData.png")


    joblib.dump(rf, dir_model / "rf_model.pkl")
    if stumpf_lr:
        joblib.dump(stumpf_lr, str(dir_model / "stumpf_lr.pkl"))

    _rr_artifact(rr, "rf_model_pkl", str(dir_model / "rf_model.pkl"))
    if stumpf_lr:
        _rr_artifact(rr, "stumpf_lr_pkl", str(dir_model / "stumpf_lr.pkl"))

    if args.no_doa:
        log.info("[Config] Domain of Applicability (DoA) enforcement DISABLED by user.")
        if "training_bounds" in model_meta:
            del model_meta["training_bounds"]

    model_meta["max_depth_sdb"] = max_depth_sdb_product
    model_meta["water_class"] = args.water_class
    with open(dir_model / "model_meta.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    chosen_max_depth = max_depth_sdb_product

    # B. Optional Spatial Validation
    if args.validate_spatial:
        log.info("\n[TRAIN] Running Secondary Spatial Validation (Spatial Split)...")

        _, _, _, df_test_spatial, model_meta_spatial = train.train_sdb_model(
            train_df=df_with_s2,
            max_depth_sdb=max_depth_sdb_product,
            seed=args.seed,
            plots_dir=dir_plot, water_class=args.water_class,
            use_stumpf_depth=args.use_stumpf_depth,
            min_training_points_for_sdb=args.min_training_points_for_sdb,
            spatial_split=True,
            cw_min=args.cw_min,
            land_max=args.land_max,
            land_mask_type=land_mask_type_for_train,
            land_mask_water_val=land_mask_water_val_for_train,
            land_mask_invert=land_mask_invert,
            linf_enabled=args.linf_enabled,
            linf_estimate=linf_est_mode,
            linf_deepwater_nir_max=args.linf_deepwater_nir_max,
            linf_deepwater_bright_max=args.linf_deepwater_bright_max,
            linf_percentile=args.linf_percentile,
            rmse_target_sdb=getattr(args,'rmse_target_sdb',1.0),
            raster_paths=s2_paths,
            depth_bin_m=float(getattr(args, 'depth_bin_m', 1.0)),
        depth_binning=str(getattr(args, 'depth_binning', 'quantile')),
            min_samples_per_bin=int(getattr(args, 'min_samples_per_bin', 10)),
        )

        src_spat_base = dir_plot / "SDB_Accuracy_Assessment_SpatialSplit.png"
        if src_spat_base.exists():
            src_spat_base.rename(dir_plot / "Validation_SpatialSplit.png")

        src_spat_v1 = dir_plot / "SDB_Accuracy_Assessment_SpatialSplit_V1_AllData.png"
        if src_spat_v1.exists():
            src_spat_v1.rename(dir_plot / "Validation_SpatialSplit_V1_AllData.png")

        m_rand = _calc_metrics(df_test_final)
        m_spat = _calc_metrics(df_test_spatial)


        # ------------------------------------------------------------------
        # Auto max-depth selection (depth-of-support)
        # ------------------------------------------------------------------
        chosen_max_depth = max_depth_sdb_product
        src_key = None

        def _load_train_report_auto_depth() -> tuple:
            cand_paths = [
                dir_logs / "train_report.json",
                out_root / "train_report.json",
                dir_model / "train_report.json",
            ]
            for tp in cand_paths:
                try:
                    if tp.exists() and tp.stat().st_size > 0:
                        with open(tp, "r") as f:
                            trj = json.load(f)
                        tr = trj.get("train", {}) if isinstance(trj, dict) else {}
                        v = tr.get("max_depth_sdb_auto_m", None)
                        if v is None:
                            diag = tr.get("max_depth_sdb_auto_diagnostics", None)
                            if isinstance(diag, dict):
                                dm = diag.get("depths_m", {}) or diag.get("depths", {})
                                if isinstance(dm, dict):
                                    v = dm.get("strict", dm.get("relaxed", None))
                        if v is None:
                            return None, None
                        fv = float(v)
                        if fv <= 0:
                            return None, None
                        return fv, f"{tp.name}:train.max_depth_sdb_auto_m"
                except Exception:
                    continue
            return None, None

        if auto_max_depth:
            fv, src = _load_train_report_auto_depth()
            if fv is None:
                # If auto-depth requested but not found (e.g. training failed or strict mode yielded nothing), 
                # fall back to safe default rather than hard crash, but warn loudly.
                log.warning("[AUTO-DEPTH] Estimation failed or returned None. Falling back to default max depth.")
                chosen_max_depth = max_depth_sdb_product
                src_key = "fallback_default"
            else:
                src_key = src
                chosen_max_depth = min(float(fv), float(max_depth_sdb_product))
                log.info(
                    f"[AUTO-DEPTH] Final product cap max_depth_sdb={chosen_max_depth:.2f} m "
                    f"(from {src_key}; rmse_target={float(getattr(args,'rmse_target_sdb',0.5)):.2f} m; hard_cap={float(max_depth_sdb_product):.2f} m)."
                )
        else:
            src_key = "cli"

        _rr_add(rr, "sdb.max_depth.final_m", float(chosen_max_depth))
        _rr_add(rr, "sdb.max_depth.source", str(src_key))

        model_meta["max_depth_sdb"] = chosen_max_depth
        model_meta["max_depth_sdb_final"] = chosen_max_depth
        if auto_max_depth:
            model_meta["max_depth_sdb_source"] = src_key
            model_meta["rmse_target_sdb"] = float(getattr(args,'rmse_target_sdb',0.5))
        else:
            model_meta["max_depth_sdb_source"] = "cli"


        try:
            with open(dir_model / "model_meta.json", "w") as f:
                json.dump(model_meta, f, indent=2)
        except Exception:
            pass


        report_lines = [
            "="*60,
            f"VALIDATION COMPARISON: {dir_name}",
            "="*60,
            f"{'Metric':<15} | {'Random (Prod)':<18} | {'Spatial (Val)':<18}",
            "-" * 60,
            f"{'RMSE (m)':<15} | {m_rand['rmse']:<18.3f} | {m_spat['rmse']:<18.3f}",
            f"{'R²':<15} | {m_rand['r2']:<18.3f} | {m_spat['r2']:<18.3f}",
            f"{'MAE (m)':<15} | {m_rand['mae']:<18.3f} | {m_spat['mae']:<18.3f}",
            f"{'Test Size':<15} | {m_rand['n']:<18} | {m_spat['n']:<18}",
            "="*60
        ]

        if m_spat['rmse'] > m_rand['rmse'] * 1.5:
            report_lines.append("[INSIGHT] Significant drop in Spatial Accuracy detected.")
        else:
            report_lines.append("[INSIGHT] Spatial Accuracy is comparable. Model generalizes well.")

        report_text = "\n".join(report_lines)
        print(report_text)

        with open(dir_logs / "validation_comparison.txt", "w") as f:
            f.write(report_text)

    # --- SAVE GPKG ---
    _save_training_gpkg(dir_data / "icesat_depths.gpkg", df_with_s2, df_train_final, df_test_final)

    # 8. Visual Debugging
    if not df_train_final.empty:
        atl03_subset = df_train_final[df_train_final["source"] == "atl03"]
        atl24_subset = df_train_final[df_train_final["source"] == "atl24"]

        if not atl03_subset.empty and atl_files_map["ATL03"]:
            try:
                vis.generate_atl03_debug_plots(
                    training_df=atl03_subset,
                    atl03_files=atl_files_map["ATL03"],
                    aoi_bbox=bbox_list,
                    plots_dir=dir_plot,
                    conf_min=args.atl03_conf_min,
                    atl24_df=atl24_subset,
                    gpkg_full_path=str(dir_data / "icesat_depths_full.gpkg"),
                    gpkg_slim_path=str(dir_data / "icesat_depths_slim.gpkg"),
                    write_gpkgs=True,
                    water_temp_c=float(args.water_temp_c),
                    wavelength_nm=532.0,
                    depth_mode="n_multiplier"
                )
            except Exception as exc:
                log.warning(f"[VIS] Plot generation failed: {exc}")

    # 9. Prediction
    log.info("\n--- Prediction ---")
    out_tif = dir_rast / "SDB_Prediction_10m.tif"

    # === NEW: Chunked processing integration (v0.6.2) ===
    use_chunked = False
    if args.chunked_prediction == "on":
        use_chunked = True
        log.info("[PREDICT] Chunked processing: ENABLED (forced)")
    elif args.chunked_prediction == "off":
        use_chunked = False
        log.info("[PREDICT] Chunked processing: DISABLED (forced)")
    elif args.chunked_prediction == "auto":
        # Auto-detect based on memory requirement
        try:
            from predict_chunked import estimate_memory_requirement_gb
            
            # Get raster dimensions from first band
            import rasterio
            with rasterio.open(s2_paths[list(s2_paths.keys())[0]]) as src:
                width, height = src.width, src.height
            
            estimated_gb = estimate_memory_requirement_gb(width, height)
            use_chunked = estimated_gb > args.max_memory_gb
            
            log.info(f"[PREDICT] Memory estimation: {estimated_gb:.1f} GB (threshold: {args.max_memory_gb} GB)")
            log.info(f"[PREDICT] Chunked processing: {'ENABLED (auto)' if use_chunked else 'DISABLED (auto)'}")
        except Exception as e:
            log.warning(f"[PREDICT] Failed to estimate memory: {e}. Using standard prediction.")
            use_chunked = False
    
    if use_chunked:
        # Use chunked processing
        try:
            from predict_chunked import predict_scene_chunked
            
            log.info(f"[PREDICT] Using chunked processing: tile_size={args.tile_size}, overlap={args.tile_overlap}")
            
            result = predict_scene_chunked(
                model_dir=dir_model,
                band_paths=s2_paths,
                out_dir=dir_rast,
                land_mask_path=str(land_mask_out),
                max_memory_gb=args.max_memory_gb,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
                sdb_mode=args.sdb_mode,
                enable_doa=(not args.no_doa),
                cw_min=args.cw_min,
                land_max=args.land_max
            )
            
            # Copy output to expected location
            if result['output_raster'].exists():
                shutil.copy2(result['output_raster'], out_tif)
                log.info(f"[PREDICT] Chunked prediction complete: {out_tif}")
            else:
                raise FileNotFoundError(f"Chunked output not found: {result['output_raster']}")
                
        except ImportError:
            log.error("[PREDICT] predict_chunked module not available. Falling back to standard prediction.")
            use_chunked = False
        except Exception as e:
            log.error(f"[PREDICT] Chunked processing failed: {e}. Falling back to standard prediction.")
            use_chunked = False
    
    if not use_chunked:
        # Standard prediction
        predict.predict_scene(
            s2_paths=s2_paths,
            land_mask_path=str(land_mask_out),
            rf_model_path=str(dir_model / "rf_model.pkl"),
            meta_json_path=str(dir_model / "model_meta.json"),
            out_path=str(out_tif),
            stumpf_lr_path=str(dir_model / "stumpf_lr.pkl") if stumpf_lr else None,
            sdb_mode=args.sdb_mode, tile_size=args.tile, s2_smooth_kernel=args.s2_smooth_kernel,
            enable_doa=(not args.no_doa),
            cw_min=args.cw_min, land_max=args.land_max,
            land_mask_type=land_mask_type_for_train,
            land_mask_water_val=land_mask_water_val_for_train,
            land_mask_invert=land_mask_invert,
            land_mask_threshold=land_mask_threshold,
            linf_estimate_deepwater=args.linf_estimate_deepwater,
            linf_deepwater_nir_max=args.linf_deepwater_nir_max,
            linf_deepwater_bright_max=args.linf_deepwater_bright_max,
            linf_percentile=args.linf_percentile,
            # Post-prediction alignment (Palaseanu-Lovejoy et al. 2026)
            align_mode=args.align_mode,
            align_tie_points_gpkg=str(dir_data / "icesat_depths.gpkg") if (dir_data / "icesat_depths.gpkg").exists() else None,
            align_min_points=args.align_min_points,
            align_depth_bins=args.align_depth_bins,
            align_source_priority=args.align_source_priority,
            align_extra_points=args.align_extra_points,
            align_max_abs_residual_m_for_fit=args.align_max_abs_residual_for_fit if args.align_max_abs_residual_for_fit > 0 else None,
        )

    # 9a. Optional conversion to bed elevation in NAVD88
    # This converts SDB depth (relative to ~MSL) to bed elevation in NAVD88
    # for consistency with river bathymetry and coastal DEMs.
    #
    # IMPORTANT SCIENTIFIC NOTE:
    # - SDB output = DEPTH below water surface (negative values = below surface)
    # - Water surface ≈ MSL (due to temporal compositing)
    # - This conversion produces BED ELEVATION in NAVD88 (not depth in NAVD88)
    # - Bed elevation = MSL_height_in_NAVD88 + depth_below_MSL
    #
    out_tif_navd88 = None
    if getattr(args, "convert_sdb_to_navd88", False):
        log.info("\n--- Convert SDB from MSL to NAVD88 ---")
        out_tif_navd88 = dir_rast / "SDB_Prediction_10m_NAVD88.tif"
        
        # Apply MSL metadata to original raster
        apply_sdb_metadata(out_tif, vertical_datum="MSL", vertical_datum_epsg=5714)
        
        # Convert from MSL to NAVD88
        success, msg = convert_sdb_msl_to_navd88(
            input_tif=out_tif,
            output_tif=out_tif_navd88,
            source_vdatum=args.sdb_source_vdatum,
            target_vdatum=args.sdb_target_vdatum,
            logger=log,
        )
        
        if success:
            # Apply NAVD88 metadata to the converted output
            apply_sdb_metadata(out_tif_navd88, vertical_datum="NAVD88", vertical_datum_epsg=5703)
            
            # Also convert uncertainty raster if it exists
            unc_tif = Path(str(out_tif).replace(".tif", "_uncertainty.tif"))
            if unc_tif.exists():
                unc_tif_navd88 = dir_rast / "SDB_Prediction_10m_uncertainty_NAVD88.tif"
                convert_sdb_msl_to_navd88(
                    input_tif=unc_tif,
                    output_tif=unc_tif_navd88,
                    source_vdatum=args.sdb_source_vdatum,
                    target_vdatum=args.sdb_target_vdatum,
                    logger=log,
                )
            
            if rr is not None:
                rr.add("vdatum_conversion.status", "success")
                rr.add("vdatum_conversion.source", args.sdb_source_vdatum)
                rr.add("vdatum_conversion.target", args.sdb_target_vdatum)
                rr.add("vdatum_conversion.output", str(out_tif_navd88))
        else:
            log.warning(f"[VDATUM] Conversion failed: {msg}")
            if rr is not None:
                rr.add("vdatum_conversion.status", "failed")
                rr.add("vdatum_conversion.error", msg)

    # 9b. Optional adjacent-tile overlap consistency diagnostic
    if getattr(args, "overlap_neighbor", None):
        try:
            neighbor_in = Path(str(args.overlap_neighbor))
            neighbor_rr = neighbor_in / "run_report.json" if neighbor_in.is_dir() else neighbor_in
            if not neighbor_rr.exists():
                log.warning(f"[QA][OVERLAP] Neighbor run_report not found: {neighbor_rr}")
            else:
                with open(neighbor_rr, "r") as f:
                    nb = json.load(f)
                nb_wesn = None
                if isinstance(nb.get("aoi", None), dict):
                    nb_wesn = nb["aoi"].get("wesn") or nb["aoi"].get("bbox")
                if nb_wesn is None and isinstance(nb.get("run", None), dict):
                    nb_wesn = nb["run"].get("aoi_wesn")
                if nb_wesn is None:
                    log.warning("[QA][OVERLAP] Could not determine neighbor AOI WESN from run_report.json")
                else:
                    nb_w, nb_e, nb_s, nb_n = [float(x) for x in nb_wesn]
                    nb_pred = None
                    try:
                        nb_pred = nb.get("outputs", {}).get("sdb_prediction", {}).get("path", None)
                    except Exception:
                        nb_pred = None
                    if not nb_pred:
                        base = neighbor_in if neighbor_in.is_dir() else neighbor_in.parent
                        cand = base / "rasters" / "SDB_Prediction_10m.tif"
                        nb_pred = str(cand) if cand.exists() else None
                    if not nb_pred or not Path(nb_pred).exists():
                        log.warning(f"[QA][OVERLAP] Neighbor prediction raster not found: {nb_pred}")
                    else:
                        w, e, s, n = [float(x) for x in args.aoi.split("/")[:4]] if isinstance(args.aoi, str) else aoi_wesn
                        cur_aoi = {"w": float(w), "e": float(e), "s": float(s), "n": float(n)}
                        nb_aoi = {"w": nb_w, "e": nb_e, "s": nb_s, "n": nb_n}
                        res = overlap_consistency_diagnostic(
                            current_run_aoi=cur_aoi,
                            neighbor_run_aoi=nb_aoi,
                            current_pred_tif=str(out_tif),
                            neighbor_pred_tif=str(nb_pred),
                            buffer_m=float(getattr(args, "overlap_buffer_m", 500.0)),
                            sample_spacing_m=float(getattr(args, "overlap_sample_spacing_m", 30.0)),
                            max_samples=int(getattr(args, "overlap_max_samples", 200000)),
                            df_ref_points=df_test_final,
                            ref_depth_col="depth_m",
                        )
                        rr.add_dict("qa.overlap_consistency", res)
                        log.info(f"[QA][OVERLAP] {res.get('status')} n_valid={res.get('n_samples_valid')} median_diff={res.get('median_diff_m')} m p90_abs={res.get('p90_abs_diff_m')} m")
        except Exception as e:
            log.warning(f"[QA][OVERLAP] Failed to compute overlap consistency diagnostic: {e}")

    # --- Post-Process Brightness Masking (Smart) ---
    if args.mask_bright_pixels is not None and s2_paths and "B02" in s2_paths:
        log.info(f"[Post-Process] Brightness masking using B02 thresh={args.mask_bright_pixels} (smart={bool(args.allow_bright_shallow_pixels)})")
    try:
        with rasterio.open(str(out_tif), "r+") as dst:
            sdb_data = dst.read(1)
            nodata = dst.nodata if dst.nodata is not None else -9999.0

            with rasterio.open(s2_paths["B02"]) as src_b2:
                b2_data = src_b2.read(1, out_shape=dst.shape, resampling=rasterio.enums.Resampling.nearest)

            scale_factor = 10000.0 if np.nanmax(b2_data) > 100 else 1.0
            b2_norm = b2_data / scale_factor

            bad_mask = (b2_norm > args.mask_bright_pixels)

            if bool(args.allow_bright_shallow_pixels) and ("B08" in s2_paths) and (s2_paths.get("B08") is not None):
                nir_thr = float(args.bright_shallow_nir_max) if args.bright_shallow_nir_max is not None else 0.03
                log.info(f"[Post-Process] Bright-shallow escape hatch enabled (B08 < {nir_thr})")
                with rasterio.open(s2_paths["B08"]) as src_b8:
                    b8_data = src_b8.read(1, out_shape=dst.shape, resampling=rasterio.enums.Resampling.nearest)
                b8_norm = b8_data / scale_factor

                is_shallow_sand = (b2_norm > args.mask_bright_pixels) & (b8_norm < nir_thr)
                bad_mask = bad_mask & (~is_shallow_sand)

            mask_count = int(np.sum(bad_mask))
            if mask_count > 0:
                sdb_data[bad_mask] = nodata
                dst.write(sdb_data, 1)
                log.info(f"[Post-Process] Masked {mask_count} pixels (cloud/glint).")
            else:
                log.info("[Post-Process] No pixels masked by brightness gate.")
    except Exception as exc:
        log.warning(f"[Post-Process] Brightness masking failed: {exc}")

    # --- NEW: Create Masked RGB (Prediction Only) ---
    rgb_full = dir_rast / "RGB_10m.tif"
    if rgb_full.exists():
        rgb_masked_out = dir_rast / "RGB_10m_predict.tif"
        log.info(f"[Post-Process] Creating Masked RGB: {rgb_masked_out}")
        try:
            with rasterio.open(str(out_tif)) as src_sdb:
                sdb_arr = src_sdb.read(1)
                sdb_nodata = src_sdb.nodata if src_sdb.nodata is not None else -9999.0
                valid_mask = (sdb_arr != sdb_nodata) & np.isfinite(sdb_arr)

            with rasterio.open(rgb_full) as src_rgb:
                prof = src_rgb.profile.copy()
                if prof.get('nodata') is None:
                    prof.update(nodata=0)

                r = src_rgb.read(1)
                g = src_rgb.read(2)
                b = src_rgb.read(3)

                fill_val = prof['nodata']
                if r.shape != valid_mask.shape:
                    log.warning("[Post-Process] RGB and SDB shapes mismatch. skipping RGB mask.")
                else:
                    r[~valid_mask] = fill_val
                    g[~valid_mask] = fill_val
                    b[~valid_mask] = fill_val

                    with rasterio.open(rgb_masked_out, 'w', **prof) as dst_rgb:
                        dst_rgb.write(r, 1)
                        dst_rgb.write(g, 2)
                        dst_rgb.write(b, 3)

        except Exception as exc:
            log.warning(f"[Post-Process] Failed to create masked RGB: {exc}")

    if not args.skip_nad83:
        predict.reproject_to_nad83(str(out_tif), str(dir_rast / "SDB_Prediction_10m_NAD83.tif"))
        unc_src = str(Path(out_tif).with_name(Path(out_tif).stem + "_uncertainty.tif"))
        if os.path.exists(unc_src):
             predict.reproject_to_nad83(unc_src, str(dir_rast / "SDB_Prediction_10m_uncertainty_NAD83.tif"))

    if df_test_final is not None and not df_test_final.empty:
        try:
            if df_train_final is not None and not df_train_final.empty:
                df_points_all = pd.concat([df_train_final, df_test_final], ignore_index=True)
            else:
                df_points_all = df_test_final
        except Exception:
            df_points_all = df_test_final

        evaluate_raster_against_points(str(out_tif), df_points_all, dir_logs, dir_plot, max_depth=chosen_max_depth)


    try:
        if rr is not None:
            _rr_merge_json(rr, "train", out_root / "train_report.json")
            _rr_merge_json(rr, "predict", dir_rast / "predict_report.json")

            if raster_quickstats is not None:
                if out_tif is not None and Path(out_tif).exists():
                    rr.add_dict("outputs.sdb_prediction", raster_quickstats(str(out_tif)))
                unc = Path(str(out_tif)).with_name(Path(str(out_tif)).stem + "_uncertainty.tif")
                if unc.exists():
                    rr.add_dict("outputs.sdb_uncertainty", raster_quickstats(str(unc)))

            _rr_artifact(rr, "rf_model_pkl", str(dir_model / "rf_model.pkl"))
            if (dir_model / "stumpf_lr.pkl").exists():
                _rr_artifact(rr, "stumpf_lr_pkl", str(dir_model / "stumpf_lr.pkl"))

            rr.write(status="ok")
            log.info("\n=== SDB Pipeline Completed Successfully ===")
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(
        "Modular SDB Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core
    p.add_argument("--aoi", required=True, help="W/E/S/N (west/east/south/north)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    p.add_argument("--out-dir", default=None, help="Custom output directory")

    # Caching / reproducibility
    p.add_argument("--cache-root", default="cache",
                   help="Root directory for reusable stage caches (S2/ATL/masks)")
    p.add_argument("--cache-strict", action="store_true", default=True,
                   help="Require exact-match (params+inputs) cache hits; otherwise rebuild.")
    p.add_argument("--no-cache-strict", dest="cache_strict", action="store_false",
                   help="Disable strict cache matching (legacy behavior).")
    p.add_argument("--cache-code-strict", action="store_true", default=False,
                   help="Use content hash (not mtime) for code fingerprinting.")
    p.add_argument("--cache-ignore-code", action="store_true", default=True,
                   help="Exclude code fingerprint from cache keys (default: True for dev workflow).")
    p.add_argument("--no-cache-ignore-code", dest="cache_ignore_code", action="store_false",
                   help="Include code fingerprint in cache keys (invalidates cache when code changes).")
    p.add_argument("--s2-module", default="s2_optics.py",
                   help="Path to s2_optics module .py file to use for composites")

    # ModesE
    p.add_argument("--sdb-mode", default="all_sdb",
                   choices=["lakes", "ocean", "all_sdb"],
                   help="Masking + model mode")

    p.add_argument("--icesat", default="all_atl",
                   choices=["atl03", "atl24", "all_atl"],
                   help="Which ICESat-2 product(s) to fetch")

    # Cloud
    p.add_argument("--cloud", type=int, default=70,
                    help="Maximum cloud percentage")
    p.add_argument("--allow-bright-shallow-pixels", action="store_true",
                   help="Allow bright shallow pixels to pass the Blue (B02) brightness gate when NIR is low.")
    p.add_argument("--bright-shallow-nir-max", type=float, default=0.03,
                   help="NIR reflectance threshold for the bright-shallow escape hatch.")

    p.add_argument("--apply-gl-turbidity-reject", action="store_true",
                   help="Enable Great Lakes turbidity reject (NIR, NIR/Green, Red thresholds).")
    p.add_argument("--gl-nir-max", type=float, default=0.06, help="Max NIR reflectance for turbidity accept.")
    p.add_argument("--gl-nir-green-ratio-max", type=float, default=0.35, help="Max (NIR/Green) ratio for turbidity accept.")
    p.add_argument("--gl-red-max", type=float, default=0.08, help="Max Red reflectance for turbidity accept.")

    # Sun-glint correction (Hedley-style; optional)
    p.add_argument("--glint-correct", action="store_true", default=False,
                   help="Apply Hedley-style sun-glint correction to visible S2 bands (post-composite).")
    p.add_argument("--glint-nir-band", default="B08",
                   help="NIR band used for glint correction regression (default: B08).")
    p.add_argument("--glint-vis-bands", default="B02,B03,B04",
                   help="Comma-separated visible bands to correct (default: B02,B03,B04).")
    p.add_argument("--glint-nir-min-percentile", type=float, default=1.0,
                   help="NIR percentile over stable water used as nir_min (default: 1).")
    p.add_argument("--glint-deepwater-b02-max", type=float, default=0.20,
                   help="Max B02 allowed in stable-water mask for glint fitting (default: 0.20).")
    p.add_argument("--glint-min-samples", type=int, default=5000,
                   help="Minimum stable-water samples required for glint fitting.")
    p.add_argument("--glint-max-samples", type=int, default=2000000,
                   help="Maximum stable-water samples used for glint fitting (random subset).")
    p.add_argument("--glint-clip-min", type=float, default=1e-6,
                   help="Clip corrected reflectance to at least this value.")

    # Sentinel-2 selection / STAC pagination (metadata-first)
    p.add_argument("--s2-scene-limit", type=int, default=10,
                   help="Number of Sentinel-2 scenes to download per tile (selected by lowest cloud %).")
    p.add_argument("--min-scene-valid_frac", type=float, default=0.90,
                   help="Drop any candidate scene whose valid-pixel fraction is below this (artifact rejection).")
    p.add_argument("--edge-weight-power", type=float, default=2.0,
                   help="Edge downweight power (higher = stronger penalty near tile edges).")
    p.add_argument("--edge-weight-min", type=float, default=0.0,
                   help="Minimum edge weight multiplier (0..1).")
    p.add_argument("--temporal-median-k", type=int, default=8,
                   help="If >0, use only the best K dates for the final median composite.")
    p.add_argument("--stac-max-items", type=int, default=5000,
                   help="Maximum number of STAC items to scan (paginated) before filtering/selection.")
    p.add_argument("--stac-page-limit", type=int, default=100,
                   help="Per-page STAC limit (server may cap; pagination is used when available).")
    p.add_argument("--cloud-report-csv", default=None,
                   help="Optional CSV path to write per-item cloud cover metadata (fast; no downloads).")
    p.add_argument("--cloud-report-only", action="store_true",
                   help="Only run STAC cloud report (no downloads/composite). Useful for tuning thresholds.")

    # ATL03
    p.add_argument("--water-class", default="clear_ocean",
                   help="Water class for training filtering (use 'unknown' to disable)")

    p.add_argument("--atl03-conf-min", type=int, default=4, help="Min ATL03 photon confidence")
    p.add_argument("--max-depth-atl03", type=float, default=15.0, help="Max depth cutoff for ATL03")
    p.add_argument("--min-bottom-photons", type=int, default=5, help="Minimum photons to form a bottom segment")

    p.add_argument("--min-depth-atl03", dest="min_depth_atl03_val", type=float, default=0.5,
                   help="Minimum depth (m) to accept from ATL03 (rejects surface noise)")
    p.add_argument("--min-bottom-frac", type=float, default=0.25,
                   help="Min fraction of bottom photons vs column noise")

    # Refraction default = True. Flag turns it OFF.
    p.add_argument("--no-refraction", dest="refraction", action="store_false", default=True,
                   help="Disable refraction correction (Default: Enabled)")

    p.add_argument("--water-temp-c", type=float, default=20.0, help="Water temp (C) for refraction")
    p.add_argument("--write-atl03-tracks-shp", action="store_true", help="Write Tracklines SHP")

    p.add_argument("--no-debug-atl03-qc", dest="debug_atl03_qc", action="store_false", default=True,
                   help="Disable ATL03 photon cloud diagnostic plotting (Default: Enabled)")

    # ATL24
    p.add_argument("--atl24-conf-min", type=float, default=0.8, help="Min confidence for ATL24")
    p.add_argument("--force-redl-atl24", action="store_true", help="Force re-download of ATL24")

    # Fusion
    p.add_argument("--atl03-weight-factor", type=float, default=1.0, help="Weight multiplier for ATL03 points")
    p.add_argument("--extra-xyz", nargs="+", 
                   help="Optional extra XYZ file paths. IMPORTANT: Z values should be depths referenced to MSL "
                        "(approximately). Use dlim with -P epsg:4269+5714 to convert from MLLW to MSL before "
                        "providing to this pipeline. Example: dlim -R=W/E/S/N hydronos -P epsg:4269+5714 > soundings_msl.xyz")
    p.add_argument("--extra-xyz-crs", default="EPSG:4326", 
                   help="Horizontal CRS for extra XYZ data. Vertical datum should be MSL (see --extra-xyz help).")

    # Working / datum harmonization
    p.add_argument("--working-srs", default="auto",
                   help="Working horizontal CRS (usually matches Sentinel-2 UTM). Used for reporting / downstream tools.")
    p.add_argument("--working-vcrs-epsg", type=int, default=5703,
                   help="Working vertical CRS EPSG code (default 5703 = NAVD88 height).")

    # Vertical datum transformation (MSL → NAVD88)
    p.add_argument("--convert-sdb-to-navd88", action="store_true", default=False,
                   help="Convert SDB from MSL to NAVD88 vertical datum using dlim. "
                        "SDB values are elevations relative to MSL (negative = below MSL). "
                        "Output: SDB_Prediction_10m_NAVD88.tif. Requires dlim (CUDEM) on PATH.")
    p.add_argument("--sdb-source-vdatum", default="epsg:4269+5714",
                   help="Source compound EPSG (default: epsg:4269+5714 = NAD83+MSL). "
                        "SDB is referenced to MSL due to temporal compositing of S2/ICESat-2.")
    p.add_argument("--sdb-target-vdatum", default="epsg:4269+5703",
                   help="Target compound EPSG (default: epsg:4269+5703 = NAD83+NAVD88).")

    # ATL horizontal CRS transformation (PROJ/pyproj).
    # NOTE: This is HORIZONTAL-ONLY (lon/lat). depth_m is NOT transformed because
    # it's relative water depth, not a geodetic height.
    p.add_argument("--atl-src-srs", default="EPSG:4326",
                   help="Source horizontal CRS for ATL lon/lat (default EPSG:4326 = WGS84).")
    p.add_argument("--atl-dst-srs", default="EPSG:4269",
                   help="Destination horizontal CRS for ATL lon/lat (default EPSG:4269 = NAD83).")
    p.add_argument("--atl-transform-datum", action="store_true", default=False,
                   help="Transform ATL lon/lat from --atl-src-srs to --atl-dst-srs (horizontal only, depth unchanged).")

    # Training
    p.add_argument("--max-depth-sdb", type=str, default="auto",
                   help="Max SDB depth (m) for training/output. Use a number (e.g., 6) or 'auto' to set per-run from spatial validation.")
    p.add_argument("--rmse-target-sdb", type=float, default=1.0,
                   help="Target spatial RMSE (m) used when --max-depth-sdb auto (depth-of-support). More conservative = smaller value.")
    p.add_argument("--depth-binning", choices=["fixed","quantile"], default="quantile",
                   help="Depth binning mode used for max-depth auto-diagnostics. quantile=equal-count bins; fixed=uniform bins (depth-bin-m).")
    p.add_argument("--depth-bin-m", type=float, default=1.0,
                   help="Depth bin width (m) used for spatial error-vs-depth diagnostics (default: 0.5).")
    p.add_argument("--min-samples-per-bin", type=int, default=10,
                   help="Minimum spatial validation samples per depth bin for auto depth-of-support diagnostics.")
    p.add_argument("--min-training-points-for-sdb", type=int, default=10, help="Min points required to train")
    
    # Physics-based depth limits (Kd estimation)
    p.add_argument("--kd-algorithm", choices=["lee2005", "mueller2000", "morel1988", "ratio_empirical"],
                   default="lee2005", help="Algorithm for Kd(490) estimation from imagery")
    p.add_argument("--kd-confidence-level", choices=["conservative", "moderate", "optimistic"],
                   default="moderate", help="Confidence level for physics-based depth limit (conservative=shallowest)")
    p.add_argument("--disable-kd-depth", action="store_true", default=False,
                   help="Disable physics-based (Kd) depth limit estimation")

    p.add_argument("--no-stumpf", dest="use_stumpf_depth", action="store_false", default=True,
                   help="Disable Log-Ratio physics feature (Default: Enabled)")

    p.add_argument("--s2-smooth-kernel", type=int, default=3, help="Smoothing kernel size for prediction")
    p.add_argument("--tile", type=int, default=1024, help="Inference tile size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # v0.6.1: Memory management and performance options
    p.add_argument("--chunked-prediction", choices=["auto", "on", "off"], default="auto",
                   help="Use chunked processing for large AOIs (auto=decide based on memory)")
    p.add_argument("--max-memory-gb", type=float, default=8.0,
                   help="Maximum memory (GB) for prediction (triggers chunking if exceeded)")
    p.add_argument("--tile-size", type=int, default=2048,
                   help="Tile size (pixels) for chunked processing")
    p.add_argument("--tile-overlap", type=int, default=256,
                   help="Tile overlap (pixels) for seamless mosaicking")
    
    # v0.6.1: ML enhancement options
    p.add_argument("--tune-hyperparameters", action="store_true",
                   help="Enable automated hyperparameter tuning for RF model")
    p.add_argument("--n-tuning-iter", type=int, default=30,
                   help="Number of hyperparameter combinations to try")
    p.add_argument("--use-adaptive-spatial-cv", action="store_true",
                   help="Use adaptive spatial cross-validation with density-based clustering")
    
    # Adaptive Spatial Sampling (NEW in v0.7.0, ENABLED BY DEFAULT in v0.7.1)
    p.add_argument("--disable-adaptive-sampling", dest="enable_adaptive_sampling",
                   action="store_false", default=True,
                   help="Disable adaptive spatial sampling (sampling is ON by default)")
    p.add_argument("--sampling-target-points", type=int, default=2000,
                   help="Target number of training points after adaptive sampling (default: 2000)")
    p.add_argument("--sampling-min-threshold", type=int, default=3000,
                   help="Minimum input points before adaptive sampling is applied (default: 3000)")
    p.add_argument("--sampling-max-gap-m", type=float, default=100.0,
                   help="Maximum spatial gap allowed in meters (default: 100m)")


# Post-prediction residual alignment (ATL24 / ATL03 / XYZ / external reference)
    p.add_argument("--align-mode", default="median", help="none|median|depth|planar (apply residual alignment using tie points)")
    p.add_argument("--align-min-points", type=int, default=150, help="Minimum tie points required to apply alignment")
    p.add_argument("--align-depth-bins", default="auto", help="Depth bin edges for depth-stratified mode, e.g. '0,2,5,10,20,40' or 'auto'")
    p.add_argument("--align-source-priority", default="atl24,atl03,xyz,other",
               help="Comma-separated source priority if tie points include a 'source' column")
    p.add_argument("--align-extra-points", action="append", default=None,
               help="Additional reference point files (CSV/GPKG/GeoJSON) to include. Can be repeated.")
    p.add_argument("--align-max-abs-residual-for-fit", type=float, default=10.0,
               help="Gate outliers when fitting alignment (meters). Use <=0 to disable.")

# CHANGED: Default is now True for Spatial Validation
    p.add_argument(
    "--validate-spatial",
    dest="validate_spatial",
    action="store_true",
    default=True,
    help="Perform extra Spatial Validation split (default: True)."
)
    p.add_argument(
    "--no-validate-spatial",
    dest="validate_spatial",
    action="store_false",
    help="Disable spatial validation."
)

# Masks
    p.add_argument("--land-mask", dest="land_mask_user", default=None,
               help="User-supplied land mask to override waffles")

# Mask gating thresholds (applied consistently in training and prediction)
# If you want "waffles defines water", keep land_max=0.0 (hard water-only constraint).
    p.add_argument("--cw-min", type=float, default=None,
               help="Minimum CLEAR_WATER mask value to accept (binary masks: 1.0 strict, 0.5 permissive)")
    p.add_argument("--land-max", type=float, default=0.5,
               help="Maximum LAND mask value to accept (binary masks: 0.0 = water-only)")

# Land mask semantics (for raster masks where "water" is encoded as a specific value)
    p.add_argument("--land-mask-water-val", type=int, default=None,
               help="Raster value that represents WATER in the land/coast mask. Default: auto (infer from training points when possible).")
    p.add_argument("--land-mask-invert", action="store_true", default=False,
               help="Invert land/coast mask semantics (keep pixels != water-val).")


    p.add_argument("--land-mask-type", type=str, default="auto",
           choices=["auto","water_only","land_binary","land_probability"],
           help="Land mask semantics: auto (infer), water_only (keep == water-val), land_binary (0=water,1=land), land_probability (keep <= threshold).")
    p.add_argument("--land-mask-threshold", type=float, default=0.5,
           help="Threshold used when land-mask-type=land_probability (or auto chooses probability).")

    # L-infinity / Dark-object subtraction handling (prediction-time)
    p.add_argument("--linf-enabled", action="store_true",
                   help="Enable L_inf (Dark Object Subtraction) correction during training.")
    p.add_argument("--linf-estimate-deepwater", action="store_true",
                   help="Estimate L_inf from deep-water pixels in the composite (raster-based).")
    p.add_argument("--linf-deepwater-nir-max", type=float, default=0.03,
                   help="Deepwater selector: max NIR (B08) reflectance (normalized 0..1).")
    p.add_argument("--linf-deepwater-bright-max", type=float, default=0.15,
                   help="Deepwater selector: max visible brightness (mean of B02/B03/B04) (normalized 0..1).")
    p.add_argument("--linf-percentile", type=float, default=1.0,
                   help="Percentile (0-100) used to estimate L_inf from deep-water pixels (e.g., 1.0 for 1st percentile).")
    p.add_argument("--kernel1", type=int, default=9, help="Morphology Kernel 1")
    p.add_argument("--kernel2", type=int, default=31, help="Morphology Kernel 2")
    p.add_argument("--erosion", type=int, default=2, help="Erosion iterations")

    p.add_argument("--mask-bright-pixels", type=float, default=0.25,
                   help="Mask pixels with reflectance > this value (Default: 0.25) to remove bright bridges")

    # NEW: Domain of Applicability
    p.add_argument("--no-doa", action="store_true",
                   help="Disable Domain of Applicability (DoA) enforcement during prediction.")

    # Flow control
    p.add_argument("--s2-only", action="store_true", help="Run only Sentinel-2 acquisition")
    p.add_argument("--atl-only", action="store_true", help="Run only ICESat-2 acquisition")
    p.add_argument("--skip-nad83", action="store_true",
                   help="Skip NAD83 output")

    p.add_argument("--overlap-neighbor", type=str, default=None,
                   help="Path to neighbor run_report.json OR its output directory for adjacent AOI overlap consistency diagnostic.")
    p.add_argument("--overlap-buffer-m", type=float, default=500.0,
                   help="Half-width (meters) of the comparison strip around a shared edge when AOIs are adjacent (default: 500m).")
    p.add_argument("--overlap-sample-spacing-m", type=float, default=30.0,
                   help="Sampling spacing (meters) inside the overlap/buffer region (default: 30m).")
    p.add_argument("--overlap-max-samples", type=int, default=200000,
                   help="Maximum number of raster samples used for overlap comparison (default: 200k).")

    # NEW v0.7.6: Checkpoint and Resume
    p.add_argument("--no-checkpoints", action="store_true",
                   help="Disable checkpoint saving (no resume capability)")
    p.add_argument("--clean-checkpoints", action="store_true",
                   help="Clear existing checkpoints before running")
    
    # NEW v0.7.6: Parallel Processing
    p.add_argument("--parallel-predict", action="store_true",
                   help="Enable parallel tile processing for prediction (2-4x speedup)")
    p.add_argument("--parallel-workers", type=int, default=None,
                   help="Number of parallel workers (default: auto-detect from CPU count)")

    return p.parse_args()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error(f"Pipeline Failed: {exc}")
        traceback.print_exc()
        sys.exit(1)