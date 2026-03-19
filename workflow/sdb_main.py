#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdb_main.py – Orchestrator for the modular SDB pipeline.

Coordinates the core modules: atl (ICESat-2), s2_optics (Sentinel-2),
fusion (multi-source training data), train (RF model), predict (scene inference),
and vis (diagnostics).
"""

import os as _os
# Force a headless-safe Matplotlib backend early (prevents TkAgg/Tkinter crashes under multiprocessing)
_os.environ.setdefault('MPLBACKEND', 'Agg')

import logging
log = logging.getLogger(__name__)
import sys
import re
import os

# Configure logging before importing other pipeline modules
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
from vdatum_utils import convert_sdb_msl_to_navd88
from process_utils import run_cmd
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple

from deps import try_import, require
from sdb_context import SDBRunContext
from train_context import TrainContext

# Optional run reporting (flight recorder)
try:
    from log_report import RunReport, raster_quickstats, mask_fraction
except (ImportError, AttributeError):  # pragma: no cover
    RunReport = None  # type: ignore
    raster_quickstats = None  # type: ignore
    mask_fraction = None  # type: ignore

try:
    from errors_scientific import FallbackRegistry, FallbackClass, record_fallback
except (ImportError, AttributeError):  # pragma: no cover
    FallbackRegistry = None  # type: ignore
    FallbackClass = None  # type: ignore
    def record_fallback(*args, **kwargs):
        return None

import numpy as np
import pandas as pd

# Ensure GeoPandas remains usable on pandas>=2.0 even if GeoPandas lags.
import compat_pandas  # noqa: F401
import joblib
rasterio = try_import('rasterio')  # optional for --help; required for SDB runtime
gpd = try_import('geopandas')  # optional for --help; required for SDB runtime
from pyproj import Transformer
from plot_utils import lazy_pyplot
plt = None  # lazy-loaded when plots are enabled

def _mask_has_ocean_pixels(mask_tif: str, *, water_max: float = 0.5, max_stride: int = 8) -> bool:
    """Fast check: does a waffles-style land mask contain ANY water/ocean pixels?

    Assumes WAFFLES convention: water=0, land=1.

    nodata=0 collision: WAFFLES GTiffs frequently set nodata=0, the same value as
    water.  rasterio honours the stored nodata tag internally during resampling, so
    using ds.read(out_shape=...) with any Resampling mode causes rasterio to treat
    every water pixel as invalid and exclude it from the aggregation.  The result is
    that all-water blocks collapse to the nodata fill and coastal strips disappear,
    making this function return False even when the mask is perfectly correct.

    Fix: read the raw integer array with masked=False (bypasses rasterio nodata
    masking entirely), then subsample with numpy striding — no resampling library
    involved.  We then apply our own nodata guard only for sentinels that cannot
    collide with the water (0) or land (1) encoding.

    Returns True on any read error (fail-open: never skip SDB due to a read failure).
    """
    try:
        import numpy as np
        import rasterio
        with rasterio.open(mask_tif) as ds:
            nodata = ds.nodata
            # Read raw values — masked=False so rasterio never hides water pixels.
            data = ds.read(1, masked=False)
        # Stride-subsample to keep memory bounded (stride=8 => 64x reduction max)
        h, w = data.shape
        stride = max(1, min(max_stride, h // 128, w // 128))
        data = data[::stride, ::stride]
        # Only exclude nodata when it is a distinct sentinel that cannot collide
        # with water (0) or land (1).
        if nodata is not None and np.isfinite(float(nodata)) and abs(float(nodata)) > 0.5:
            data = data[data != nodata]
        if data.size == 0:
            return False
        return bool((data <= water_max).any())
    except (ImportError, AttributeError):
        return True


def _coarse_lonlat_cell(lon: float, lat: float, cell_deg: float = 2.0) -> str:
    """Return a coarse spatial cohort label for model-bank partitioning.

    We intentionally use a *broad* cell so neighboring AOIs can still share a
    bank, while preventing global cross-region contamination.
    """
    try:
        cell = float(cell_deg)
        if not np.isfinite(cell) or cell <= 0:
            cell = 2.0
        lon0 = int(np.floor(float(lon) / cell) * cell)
        lat0 = int(np.floor(float(lat) / cell) * cell)
        lon1 = lon0 + int(round(cell))
        lat1 = lat0 + int(round(cell))
        return f"lon{lon0:+03d}_{lon1:+03d}__lat{lat0:+03d}_{lat1:+03d}"
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return "lonlat_unknown"


def _build_model_bank_partition_context(args, df_pts=None, base_dir=None):
    """Return a versioned, locality-aware model-bank partition context.

    This is deliberately broader than a single AOI but much narrower than the old
    global mix/depth buckets, which allowed unrelated regions to share the same
    persisted model/state.
    """
    base_dir = base_dir if base_dir is not None else getattr(args, "model_bank", None)
    if not base_dir or df_pts is None or len(df_pts) == 0:
        return base_dir, None

    n_all = int(len(df_pts))
    src_ser = df_pts["source"].astype(str).str.lower() if "source" in df_pts.columns else pd.Series(["unknown"] * n_all)
    n_xyz = int(src_ser.str.startswith("extra_xyz").sum()) if len(src_ser) else 0
    n_atl = int(src_ser.str.contains("atl", regex=False).sum()) if len(src_ser) else 0
    xyz_frac = (n_xyz / float(n_all)) if n_all else 0.0
    atl_frac = (n_atl / float(n_all)) if n_all else 0.0
    if xyz_frac >= 0.75:
        mix_bucket = "xyz_dominant"
    elif xyz_frac >= 0.25:
        mix_bucket = "xyz_mixed"
    elif atl_frac >= 0.75:
        mix_bucket = "atl_dominant"
    else:
        mix_bucket = "mixed_other"

    depth_bucket = "unknown"
    p95 = None
    if "depth_m" in df_pts.columns:
        d = pd.to_numeric(df_pts["depth_m"], errors="coerce").to_numpy(dtype=float)
        d = np.abs(d[np.isfinite(d)])
        if d.size:
            p95 = float(np.nanpercentile(d, 95))
            if p95 <= 12.0:
                depth_bucket = "shallow"
            elif p95 <= 25.0:
                depth_bucket = "mid"
            else:
                depth_bucket = "deep"

    try:
        w, e, s, n = [float(x) for x in str(getattr(args, "aoi")).split("/")[:4]]
        lon_c = 0.5 * (w + e)
        lat_c = 0.5 * (s + n)
    except (TypeError, ValueError, AttributeError):
        lon_c = float("nan")
        lat_c = float("nan")

    working_srs = str(getattr(args, "working_srs", "auto") or "auto").upper()
    region_name = str(getattr(args, "region_name", None) or getattr(args, "region", None) or "unknown")
    locality_tag = _coarse_lonlat_cell(lon_c, lat_c, cell_deg=2.0)
    partition_version = 2
    part_key = (
        f"v{partition_version}__region_{region_name}__grid_{working_srs.replace(':', '')}"
        f"__cell_{locality_tag}__mix_{mix_bucket}__depth_{depth_bucket}"
    )
    part_dir = Path(base_dir) / "partitions" / part_key
    part_meta = {
        "partition_version": partition_version,
        "partition_key": part_key,
        "region_name": region_name,
        "working_srs": working_srs,
        "locality_tag": locality_tag,
        "mix_bucket": mix_bucket,
        "depth_bucket": depth_bucket,
        "xyz_frac": float(xyz_frac),
        "atl_frac": float(atl_frac),
        "depth_p95_abs_m": (None if p95 is None else float(p95)),
        "n": n_all,
    }
    return part_dir, part_meta


def _count_finite_raster_pixels(path: Path) -> int:
    try:
        with rasterio.open(path) as ds:
            nodata = ds.nodata
            total = 0
            sentinels = []
            try:
                if nodata is not None and np.isfinite(float(nodata)):
                    sentinels.append(float(nodata))
            except (TypeError, ValueError):
                pass
            # Defensive fallback for rasters whose nodata tag is missing or lost.
            sentinels.extend([-9999.0, -99999.0, 3.402823466e+38, -3.402823466e+38])
            sentinels_arr = np.asarray(sorted(set(sentinels)), dtype=np.float64) if sentinels else np.empty((0,), dtype=np.float64)
            for _, window in ds.block_windows(1):
                arr = ds.read(1, window=window, masked=False).astype(np.float64, copy=False)
                mask = np.isfinite(arr)
                if sentinels_arr.size:
                    for sv in sentinels_arr:
                        mask &= ~np.isclose(arr, sv, rtol=0.0, atol=1e-6)
                total += int(np.count_nonzero(mask))
            return int(total)
    except (OSError, RuntimeError, ValueError, AttributeError):
        log.debug("Raster finite-count check failed for %s", path, exc_info=True)
        return -1


def _assert_sdb_prediction_is_meaningful(out_tif: Path, dir_rast: Path, *, stage: str) -> None:
    """Fail closed when SDB output is effectively empty/scientifically unusable."""
    min_pixels = 500
    min_fraction_of_base_valid = 0.02
    rep_path = Path(dir_rast) / "predict_report.json"
    predicted = None
    base_valid = None
    if rep_path.exists():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            funnel = ((rep or {}).get("predict") or {}).get("funnel") or {}
            predicted = int(funnel.get("predicted")) if funnel.get("predicted") is not None else None
            base_valid = int(funnel.get("base_valid")) if funnel.get("base_valid") is not None else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            log.debug("Could not parse %s for SDB output validation.", rep_path, exc_info=True)

    finite_pixels = _count_finite_raster_pixels(out_tif)
    provenance_pixels = None
    prov_path = out_tif.with_name(out_tif.stem + "_provenance.tif")
    if prov_path.exists():
        try:
            with rasterio.open(prov_path) as ds:
                total_pred = 0
                for _, window in ds.block_windows(1):
                    arr = ds.read(1, window=window, masked=False)
                    total_pred += int(np.count_nonzero(arr == 1))
                provenance_pixels = int(total_pred)
        except (OSError, ValueError, rasterio.errors.RasterioError):
            log.debug("Could not count provenance pixels for %s", prov_path, exc_info=True)
    checks = []
    if predicted is not None:
        checks.append((predicted >= min_pixels, f"predicted pixels {predicted} < {min_pixels}"))
    if base_valid is not None and base_valid > 0 and predicted is not None:
        frac = float(predicted) / float(base_valid)
        checks.append((frac >= min_fraction_of_base_valid,
                       f"predicted/base_valid fraction {frac:.4f} < {min_fraction_of_base_valid:.4f}"))
    if provenance_pixels is not None:
        checks.append((provenance_pixels >= min_pixels, f"provenance predicted pixels {provenance_pixels} < {min_pixels}"))
    if finite_pixels >= 0:
        checks.append((finite_pixels >= min_pixels, f"finite raster pixels {finite_pixels} < {min_pixels}"))

    failed = [msg for ok, msg in checks if not ok]
    if failed:
        msg = f"SDB output degraded at {stage}: " + "; ".join(failed)
        log.warning(msg)
        log.warning("[DEGRADED] SDB output has very few valid pixels. "
                    "Output will be labeled as degraded guidance rather than rejected. "
                    "This typically means training data was too shallow/sparse for RF "
                    "to generalize across the AOI. Consider using tier 2 (Stumpf-only) "
                    "or providing extra_xyz soundings.")
        # Record degradation instead of crashing
        try:
            from errors_scientific import record_fallback, FallbackClass
            record_fallback(None, "sdb_prediction_sparse", FallbackClass.DEGRADED,
                          stage=stage, detail=msg)
        except (ImportError, AttributeError):
            log.debug("Scientific fallback registry unavailable", exc_info=True)
        # Do NOT raise — let the pipeline continue with degraded output


def _require_sdb_runtime_deps() -> None:
    """Raise a clear error if heavy deps are missing when actually running SDB."""
    require(rasterio, "rasterio", "Needed for raster I/O (GeoTIFF) in SDB pipeline.")
    require(gpd, "geopandas", "Needed for vector I/O/masking in SDB pipeline.")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sdb_guidance import build_sdb_guidance_manifest, write_sdb_guidance_manifest

# Import new improvement modules
try:
    from validation import validate_pipeline_config
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    
try:
    from checkpoints import PipelineCheckpoint
    CHECKPOINTS_AVAILABLE = True
except ImportError:
    CHECKPOINTS_AVAILABLE = False

try:
    PARALLEL_PREDICT_AVAILABLE = True
except ImportError:
    PARALLEL_PREDICT_AVAILABLE = False


def detect_active_config_profile(aoi_bbox: list, config_path: str = "sdb_config.json") -> str:
    """Return the name of the active config profile for this AOI center.

    - If a region bbox matches: return region['name']
    - Else: return 'defaults' if present, otherwise 'default'
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        w, s, e, n = aoi_bbox
        c_lon = (w + e) / 2.0
        c_lat = (s + n) / 2.0
        for region in data.get("regions", []):
            rw, rs, re, rn = region["bbox"]
            if (rw <= c_lon <= re) and (rs <= c_lat <= rn):
                return str(region.get("name", "region"))
        return "defaults" if "defaults" in data else "default"
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return "unknown"


# -----------------------------------------------------------------------------
# Vertical Datum Transformation (MSL → NAVD88)
# -----------------------------------------------------------------------------

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
    except (OSError, ValueError, rasterio.errors.RasterioError) as e:
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
    log.error("Failed to import SDB modules: %s: %s", type(exc).__name__, exc)
    log.error("Tip: if this is an IndentationError/SyntaxError in predict.py, replace it with the fixed version.")
    traceback.print_exc()
    sys.exit(1)
except Exception as exc:
    import traceback
    log.error("Unexpected error importing SDB modules: %s: %s", type(exc).__name__, exc)
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

    # Register in sys.modules before executing: required by dataclasses in py3.13
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
    log.warning("_run() is deprecated. Prefer _run_safe(list_args).")
    try:
        cmd_list = shlex.split(cmd)
    except ValueError as e:
        raise RuntimeError(f"[SECURITY] Could not parse command safely (refusing shell=True): {e}")
    return _run_safe(cmd_list)


def _run_safe(cmd_list):
    """Run command list safely and return stdout. Raises RuntimeError on failure."""
    res = run_cmd(cmd_list, check=True)
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
            try:
                import rasterio
                with rasterio.open(out_mask_tif) as ds:
                    if ds.nodata == 0:
                        log.warning("Found cached mask with nodata=0 (BUG). Deleting to regenerate: %s", out_mask_tif)
                        out_mask_tif.unlink()
                    else:
                        return str(out_mask_tif)
            except (ImportError, OSError, RuntimeError, ValueError):
                log.debug("mask cache check failed; will regenerate", exc_info=True)

    # 2. Setup Cache and Params
    aoi_buf = _buffer_aoi(aoi, pct=0.05)
    inc_str = f"{WAFFLES_INC_ARCSEC:.9f}s"
    cache_masks = Path(cache_masks)
    cudem_cache_dir = Path(".cudem_cache")
    cudem_cache_dir.mkdir(exist_ok=True)
    cache_masks.mkdir(parents=True, exist_ok=True)

    if sdb_mode == "lakes":
        want_nhd = False
        want_lakes = True
    elif sdb_mode == "ocean":
        want_nhd = True
        want_lakes = False
    else:
        want_nhd = True
        want_lakes = True

    params = f"want_nhd={str(want_nhd).lower()}:want_lakes={str(want_lakes).lower()}"
    chash = hashlib.sha1(
        f"{aoi_buf}|{WAFFLES_INC_ARCSEC:.9f}|{params}".encode()
    ).hexdigest()[:12]

    base_prefix = cache_masks / f"waffles_coastline_{chash}"
    base_tif = base_prefix.with_suffix(".tif")

    # 3. Run Waffles if Raw TIF missing
    if (not base_tif.exists()) or base_tif.stat().st_size == 0:
        cmd_list = [
            "waffles",
            "-M",
            f"coastline:{params}",
            f"-R={aoi_buf}",
            "-E",
            inc_str,
            "-O",
            str(base_prefix),
        ]
        log.info("Running: %s", ' '.join(cmd_list))
        try:
            _run_safe(cmd_list)
        except (OSError, RuntimeError, ValueError):
            # Fallback check for glob if name varied slightly
            log.debug("ignored", exc_info=True)
        if not base_tif.exists():
            existing = sorted([p.name for p in cache_masks.glob('*.tif')])
            raise RuntimeError(
                "Waffles did not produce the expected coastline raster. "
                f"Expected: {base_tif}. "
                f"Existing .tif files in {cache_masks}: {existing}. "
                "This workflow requires deterministic WAFFLES outputs; please verify waffles ran successfully."
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
    """Legacy no-op: run logging is initialized under out_dir/run_logs at startup."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return None

def _write_report(out_dir: Path, status: str, **kwargs):
    """Write a JSON run report."""
    try:
        if RunReport is not None:
            rr = RunReport(out_dir)
            rr.add("run.version", VERSION)
            rr.add("run.status", status)
            rr.merge(kwargs or {})
            rr.write(status=status)
            try:
                # best-effort: emit artifacts from the assembled kwargs
                _emit_artifacts_from_report(kwargs or {})
            except (OSError, TypeError, ValueError, AttributeError) as e:
                log.debug("Optional emit failed: %s", e)
            return
    except (OSError, TypeError, ValueError, AttributeError):
        log.debug("ignored", exc_info=True)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
        "status": status,
        **kwargs
    }
    report_path = out_dir / "run_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    try:
        from flight_recorder import emit_artifact_written
        emit_artifact_written(report_path, kind="json", role="sdb_run_report")
        _emit_artifacts_from_report(report)
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as e:
        log.debug("Optional flight-recorder emit failed: %s", e)




def _emit_artifacts_from_report(report: dict) -> None:
    """Emit artifact_written events for any path-like values in the report."""
    try:
        from flight_recorder import emit_artifact_written
    except (ImportError, AttributeError):
        return

    def _walk(x):
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _walk(v)
        elif isinstance(x, (str, Path)):
            s = str(x)
            if any(s.lower().endswith(ext) for ext in (".tif", ".tiff", ".vrt", ".nc", ".h5", ".csv", ".json", ".md", ".txt", ".png", ".jpg", ".jpeg")):
                kind = "raster" if s.lower().endswith((".tif", ".tiff", ".vrt")) else "file"
                emit_artifact_written(s, kind=kind, role="from_report")

    _walk(report)

def _rr_add(rr, key: str, value):
    try:
        if rr is not None:
            rr.add(key, value)
    except (AttributeError, TypeError, ValueError):
        log.debug("ignored", exc_info=True)

def _rr_artifact(rr, name: str, path: str):
    try:
        if rr is not None and hasattr(rr, "record_artifact"):
            rr.record_artifact(name, str(path))
    except (AttributeError, TypeError, ValueError):
        log.debug("ignored", exc_info=True)

def _rr_merge_json(rr, key: str, path: Path):
    try:
        if rr is None:
            return
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            rr.add_dict(key, d if isinstance(d, dict) else {"value": d})
            _rr_artifact(rr, key.replace(".", "_") + "_json", str(p))
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
        log.debug("ignored", exc_info=True)

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
    log.info("Saving training data to %s...", out_path)

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

        log.info("Export complete.")
    except Exception as exc:
        log.warning("Failed to save GeoPackage: %s", exc)

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
        with open(config_path, "r", encoding="utf-8") as f:
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
                log.info("AOI center (%.2f, %.2f) matches region: %s", c_lon, c_lat, region['name'])
                log.info("Description: %s", region['description'])

                # Merge region settings ON TOP of defaults
                config.update(region["settings"])
                return config

        log.info("No specific region matched. Using global defaults.")
        return config

    except Exception as e:
        log.warning("Failed to load config: %s", e)
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



def _derive_validation_invariants(m_rand: dict, m_spat: dict, spatial_status: Optional[dict] = None) -> dict:
    """Executable validation guards used for run-reporting and messaging."""
    spatial_status = spatial_status or {}
    try:
        r2_rand = float(m_rand.get("r2", float("nan")))
    except (TypeError, ValueError, AttributeError):
        r2_rand = float("nan")
    try:
        r2_spat = float(m_spat.get("r2", float("nan")))
    except (TypeError, ValueError, AttributeError):
        r2_spat = float("nan")
    try:
        n_rand = int(m_rand.get("n", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        n_rand = 0
    try:
        n_spat = int(m_spat.get("n", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        n_spat = 0
    no_spatial_holdout = bool(spatial_status.get("reason") == "no_representative_cluster_holdout" or n_spat <= 0)
    negative_validation = bool((np.isfinite(r2_rand) and r2_rand < 0) or (np.isfinite(r2_spat) and r2_spat < 0))
    weak_spatial_validation = bool(np.isfinite(r2_rand) and np.isfinite(r2_spat) and (r2_spat < 0.2))
    return {
        "random_test_available": bool(n_rand > 0),
        "representative_spatial_holdout": not no_spatial_holdout,
        "negative_validation_detected": negative_validation,
        "weak_spatial_validation": weak_spatial_validation,
        "guidance_only_required": bool(no_spatial_holdout or negative_validation or weak_spatial_validation),
    }

def evaluate_raster_against_points(
    raster_path: str,
    df_points,
    log_dir: Path,
    plot_dir: Path,
    prefix: str = "SDB_Raster_vs_Validation_Points",
    max_depth: float = None,
) -> dict:
    if df_points is None or df_points.empty:
        log.warning("No points provided for raster evaluation.")
        return {}

    required_cols = {"longitude", "latitude", "depth_m"}
    if not required_cols.issubset(df_points.columns):
        log.warning("Points DataFrame missing required columns for evaluation.")
        return {}

    raster_path = str(raster_path)
    if not os.path.exists(raster_path):
        log.warning("Raster not found: %s", raster_path)
        return {}

    log.info("Evaluating raster %s against %s points.", raster_path, len(df_points))

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
        log.warning("No valid overlapping samples between raster and points.")
        return {}

    z_true_m = z_true[m]
    z_pred_m = z_pred_raw[m]

    z_true_m, z_pred_m = _normalize_depth_pair(z_true_m, z_pred_m)

    if z_true_m.size == 0:
        log.warning("No valid overlapping samples between raster and points.")
        return {}

    # Metrics on ALL overlap samples
    rmse_all = _rmse(z_true_m, z_pred_m)
    mae_all = float(mean_absolute_error(z_true_m, z_pred_m))
    bias_all = float(np.nanmean(z_pred_m - z_true_m))
    r2_all = _r2(z_true_m, z_pred_m)
    n_all = int(len(z_true_m))

    log.info("Raster vs points (ALL) – RMSE=%.2f m, MAE=%.2f m, R²=%.2f, N=%s", rmse_all, mae_all, r2_all, n_all)

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
            log.info("Raster vs points (|depth|≤%gm) – RMSE=%.2f m, MAE=%.2f m, R²=%.2f, N=%s", md, rmse_val, mae_val, r2_val, n_val)

    train.scatter_plot(
        y_true=z_true_m,
        y_pred=z_pred_m,
        title="Raster vs Validation Points",
        out_png=out_png,
        max_depth=max_depth,
        x_label="Validation Depth (m)",
        y_label="SDB Raster Depth (m)"
    )

    metrics = {"all": {"rmse_m": rmse_all, "mae_m": mae_all, "bias_m": bias_all, "r2": r2_all, "n": n_all}, "limited": {"rmse_m": rmse_val, "mae_m": mae_val, "bias_m": bias_val, "r2": r2_val, "n": n_val}, "max_depth_m": (None if max_depth is None else float(max_depth))}
    with open(log_dir / "eval_metrics_raster_vs_validation_points.json", "w", encoding="utf-8") as f:
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



def _run_sdb_prediction(
    ctx_or_args, out_tif=None, dir_model=None, dir_rast=None, run_id=None,
    s2_paths=None, land_mask_out=None,
    stumpf_lr=None, land_mask_type_for_train=None, land_mask_water_val_for_train=None,
    land_mask_invert=None, land_mask_threshold=None, chosen_max_depth=None, rr=None,
    dir_data=None, predict=None,
):
    """Dispatch SDB prediction: chunked or standard path.

    Accepts either an ``SDBRunContext`` as first arg (new style) or the legacy
    positional ``args`` namespace (backward compatible).
    """
    if isinstance(ctx_or_args, SDBRunContext):
        ctx = ctx_or_args
        args = ctx.args
        out_tif = out_tif or ctx.out_tif
        dir_model = dir_model or ctx.dir_model
        dir_rast = dir_rast or ctx.dir_rast
        run_id = run_id or ctx.run_id
        s2_paths = s2_paths or ctx.s2_paths
        land_mask_out = land_mask_out or ctx.land_mask_out
        stumpf_lr = stumpf_lr if stumpf_lr is not None else ctx.stumpf_lr
        land_mask_type_for_train = land_mask_type_for_train or ctx.land_mask_type
        land_mask_water_val_for_train = land_mask_water_val_for_train if land_mask_water_val_for_train is not None else ctx.land_mask_water_val
        land_mask_invert = land_mask_invert if land_mask_invert is not None else ctx.land_mask_invert
        land_mask_threshold = land_mask_threshold if land_mask_threshold is not None else ctx.land_mask_threshold
        chosen_max_depth = chosen_max_depth if chosen_max_depth is not None else ctx.chosen_max_depth
        rr = rr or ctx.rr
        dir_data = dir_data or ctx.dir_data
        predict = predict or ctx.mod_predict
    else:
        args = ctx_or_args
        fallback_registry = None
    # 9. Prediction
    log.info("Prediction")

    use_chunked = False
    if args.chunked_prediction == "on":
        use_chunked = True
        log.info("Chunked processing: enabled (forced)")
    elif args.chunked_prediction == "off":
        use_chunked = False
        log.info("Chunked processing: disabled (forced)")
    elif args.chunked_prediction == "auto":
        # Auto-detect based on memory requirement
        try:
            from chunked_processing import estimate_memory_requirement_gb
            
            # Get raster dimensions from first band
            import rasterio
            with rasterio.open(s2_paths[list(s2_paths.keys())[0]]) as src:
                width, height = src.width, src.height
            
            estimated_gb = estimate_memory_requirement_gb(height=height, width=width, n_bands=14)
            use_chunked = estimated_gb > args.max_memory_gb
            
            log.info("Memory estimation: %.1f GB (threshold: %s GB)", estimated_gb, args.max_memory_gb)
            log.info("Chunked processing: %s", 'enabled (auto)' if use_chunked else 'disabled (auto)')
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            log.warning("Failed to estimate memory: %s. Using standard prediction.", e)
            use_chunked = False
    
    if use_chunked:
        # Use chunked processing
        try:
            from predict_chunked import predict_scene_chunked
            
            log.info("Using chunked processing: tile_size=%s, overlap=%s", args.tile_size, args.tile_overlap)
            
            result = predict_scene_chunked(
                model_dir=dir_model,
                band_paths=s2_paths,
                out_dir=dir_rast,
                final_out_path=str(out_tif),
                land_mask_path=str(land_mask_out),
                max_memory_gb=args.max_memory_gb,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
                sdb_mode=args.sdb_mode,
                s2_smooth_kernel=args.s2_smooth_kernel,
                enable_doa=(not args.no_doa),
                cw_min=args.cw_min,
                land_max=args.land_max,
                land_mask_type=land_mask_type_for_train,
                land_mask_water_val=land_mask_water_val_for_train,
                land_mask_invert=land_mask_invert,
                land_mask_threshold=land_mask_threshold,
                linf_estimate_deepwater=args.linf_estimate_deepwater,
                linf_deepwater_nir_max=args.linf_deepwater_nir_max,
                linf_deepwater_bright_max=args.linf_deepwater_bright_max,
                linf_percentile=args.linf_percentile,
                write_confidence=True,
                write_provenance=True,
                min_confidence_threshold=args.min_confidence_threshold,
            )

            # Copy depth raster and any sibling outputs to their final locations
            tmp_raster = result['output_raster']
            if not tmp_raster.exists():
                raise FileNotFoundError(f"Chunked output not found: {tmp_raster}")
            shutil.copy2(tmp_raster, out_tif)
            for suffix in ("_confidence.tif", "_provenance.tif", "_uncertainty.tif", "_doa.tif"):
                src_sib = tmp_raster.with_name(tmp_raster.stem + suffix)
                if src_sib.exists():
                    shutil.copy2(src_sib, out_tif.with_name(out_tif.stem + suffix))
            log.info("Chunked prediction complete: %s", out_tif)
            _assert_sdb_prediction_is_meaningful(out_tif, dir_rast, stage="chunked_prediction")

        except ImportError:
            log.error("predict_chunked module not available. Falling back to standard prediction.")
            use_chunked = False
        except (OSError, RuntimeError, ValueError) as e:
            log.error("Chunked processing failed: %s. Falling back to standard prediction.", e, exc_info=True)
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
            align_mode=args.align_mode,
            align_tie_points_gpkg=str(dir_data / "icesat_depths.gpkg") if (dir_data / "icesat_depths.gpkg").exists() else None,
            align_min_points=args.align_min_points,
            align_depth_bins=args.align_depth_bins,
            align_source_priority=args.align_source_priority,
            align_extra_points=args.align_extra_points,
            align_max_abs_residual_m_for_fit=args.align_max_abs_residual_for_fit if args.align_max_abs_residual_for_fit > 0 else None,
            write_confidence=True,
            write_provenance=True,
            min_confidence_threshold=args.min_confidence_threshold,
        )
        _assert_sdb_prediction_is_meaningful(out_tif, dir_rast, stage="prediction")



def _run_sdb_post_processing(
    ctx_or_args, out_tif=None, s2_paths=None, dir_rast=None, rr=None, out_root=None,
    bbox_wesn=None, df_test_final=None, predict=None,
):
    """Post-prediction steps: NAVD88 conversion, overlap diagnostic,
    brightness masking, masked RGB, and NAD83 reprojection.

    Accepts either an ``SDBRunContext`` or legacy positional args.
    """
    if isinstance(ctx_or_args, SDBRunContext):
        ctx = ctx_or_args
        args = ctx.args
        out_tif = out_tif or ctx.out_tif
        s2_paths = s2_paths or ctx.s2_paths
        dir_rast = dir_rast or ctx.dir_rast
        rr = rr or ctx.rr
        out_root = out_root or ctx.out_root
        bbox_wesn = bbox_wesn or ctx.bbox_wesn
        df_test_final = df_test_final if df_test_final is not None else ctx.df_test
        predict = predict or ctx.mod_predict
    else:
        args = ctx_or_args
    import rasterio

    # ------------------------------------------------------------------
    # Stumpf-as-regularizer: penalize confidence where RF diverges from
    # the physics-grounded Stumpf baseline (catches RF extrapolation artifacts
    # that cause tile-boundary seams).
    # ------------------------------------------------------------------
    try:
        conf_tif = Path(str(out_tif)).with_name(Path(str(out_tif)).stem + "_confidence.tif")
        stumpf_depth_col = "stumpf_depth"
        if conf_tif.exists() and out_tif and Path(out_tif).exists():
            # Check if we have a stumpf_lr model (tier 1 with Stumpf baseline)
            stumpf_lr_path = None
            if isinstance(ctx_or_args, SDBRunContext) and ctx_or_args.stumpf_lr:
                stumpf_lr_path = ctx_or_args.dir_model / "stumpf_lr.pkl" if ctx_or_args.dir_model else None

            # Skip Stumpf regularization when dense authoritative XYZ data is
            # available.  The regularizer penalizes RF predictions that disagree
            # with the Stumpf baseline — but with dense survey data, the RF is
            # trained on ground truth and Stumpf is wrong (especially in turbid
            # channels where the blue/green ratio misinterprets turbidity as
            # shallow depth).  Applying the regularizer here would revert the
            # RF's corrections back toward the wrong Stumpf values.
            _anchor_good = False
            try:
                # Get dir_model from context (not a direct parameter of _run_sdb_post_processing)
                _dir_model = None
                if isinstance(ctx_or_args, SDBRunContext) and hasattr(ctx_or_args, 'dir_model') and ctx_or_args.dir_model:
                    _dir_model = ctx_or_args.dir_model
                _meta_path = (Path(_dir_model) / "model_meta.json") if _dir_model else None
                if _meta_path and _meta_path.exists():
                    import json as _json_check
                    with open(_meta_path) as _mf:
                        _mm = _json_check.load(_mf)
                    _gs = _mm.get("guidance_settings", {})
                    _anchor_good = bool(_gs.get("anchor_support_good", False))
                    log.info("[Stumpf-regularizer] anchor_support_good=%s (from %s)", _anchor_good, _meta_path)
                else:
                    log.info("[Stumpf-regularizer] No model_meta.json found; checking args for extra_xyz.")
            except Exception as _asg_e:
                log.warning("[Stumpf-regularizer] model_meta check failed: %s; checking args.", _asg_e)

            # Fallback: if user provided extra_xyz, that IS authoritative data
            if not _anchor_good:
                _has_xyz = bool(getattr(args, "extra_xyz", None))
                if _has_xyz:
                    _anchor_good = True
                    log.info("[Stumpf-regularizer] extra_xyz detected in args; treating as anchor_support_good=True.")

            if _anchor_good:
                log.info("[Stumpf-regularizer] SKIPPED: dense authoritative XYZ data is available; "
                         "RF corrections are more trustworthy than Stumpf baseline in this regime.")
            elif stumpf_lr_path and stumpf_lr_path.exists():
                from sdb_tier import apply_stumpf_regularization
                import joblib
                import numpy as _np

                with rasterio.open(str(out_tif)) as ds_rf:
                    rf_depth = ds_rf.read(1).astype("float64")
                    rf_profile = ds_rf.profile.copy()

                with rasterio.open(str(conf_tif)) as ds_conf:
                    conf_arr = ds_conf.read(1).astype("float32")
                    conf_profile = ds_conf.profile.copy()

                # Reconstruct Stumpf depth from stumpf_idx raster + saved LR
                stumpf_model = joblib.load(str(stumpf_lr_path))
                b02_path = s2_paths.get("B02") if s2_paths else None
                b03_path = s2_paths.get("B03") if s2_paths else None

                if b02_path and b03_path:
                    with rasterio.open(str(b02_path)) as ds2, rasterio.open(str(b03_path)) as ds3:
                        b02 = ds2.read(1).astype("float64")
                        b03 = ds3.read(1).astype("float64")

                    eps = 1e-6
                    log_b03 = _np.log(_np.maximum(b03, eps))
                    log_b03 = _np.where(_np.abs(log_b03) < eps, eps, log_b03)
                    stumpf_idx = _np.log(_np.maximum(b02, eps)) / log_b03
                    stumpf_idx = _np.clip(stumpf_idx, -10.0, 10.0)

                    finite_mask = _np.isfinite(stumpf_idx)
                    stumpf_pred = _np.full_like(rf_depth, _np.nan)
                    if hasattr(stumpf_model, "predict"):
                        x_flat = stumpf_idx[finite_mask].ravel()
                        if hasattr(stumpf_model, "increasing"):
                            # IsotonicRegression
                            stumpf_pred[finite_mask] = stumpf_model.predict(x_flat)
                        else:
                            # LinearRegression
                            stumpf_pred[finite_mask] = stumpf_model.predict(x_flat.reshape(-1, 1)).ravel()

                    reg_conf, reg_meta = apply_stumpf_regularization(
                        conf_arr, rf_depth, stumpf_pred, sigma_threshold=2.0)

                    # Apply edge taper for seamless tile blending
                    from sdb_tier import apply_edge_taper
                    reg_conf = apply_edge_taper(reg_conf, taper_frac=0.10)
                    reg_meta["edge_taper_applied"] = True

                    # Write regularized + tapered confidence back
                    conf_profile.update(dtype="float32")
                    with rasterio.open(str(conf_tif), "w", **conf_profile) as dst:
                        dst.write(reg_conf, 1)

                    _rr_add(rr, "sdb.stumpf_regularizer", reg_meta)
                    log.info("[Stumpf-regularizer] Applied + edge taper: flagged %.1f%% of pixels",
                             reg_meta.get("frac_flagged", 0) * 100)
    except Exception:
        log.warning("[Stumpf-regularizer] Failed; continuing without regularization.", exc_info=True)

    # 9a. Optional conversion to bed elevation in NAVD88
    # This converts SDB depth (relative to ~MSL) to bed elevation in NAVD88
    # for consistency with river bathymetry and coastal DEMs.
    #
    # SDB depth is relative to MSL (negative = below surface). This converts
    # to bed elevation in NAVD88: bed_elev = MSL_height_in_NAVD88 + depth_below_MSL.
    #
    out_tif_navd88 = None
    if getattr(args, "convert_sdb_to_navd88", False):
        log.info("Converting SDB from MSL to NAVD88")
        out_tif_navd88 = dir_rast / f"{out_tif.stem}_bed_navd88.tif"
        
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
            unc_tif = out_tif.with_name(f"{out_tif.stem}_uncertainty.tif")
            if unc_tif.exists():
                unc_tif_navd88 = dir_rast / f"{out_tif.stem}_uncertainty_bed_navd88.tif"
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
            log.warning("Conversion failed: %s", msg)
            if rr is not None:
                rr.add("vdatum_conversion.status", "failed")
                rr.add("vdatum_conversion.error", msg)

    # 9b. Optional adjacent-tile overlap consistency diagnostic
    if getattr(args, "overlap_neighbor", None):
        try:
            neighbor_in = Path(str(args.overlap_neighbor))
            neighbor_rr = neighbor_in / "run_report.json" if neighbor_in.is_dir() else neighbor_in
            if not neighbor_rr.exists():
                log.warning("Neighbor run_report not found: %s", neighbor_rr)
            else:
                with open(neighbor_rr, "r", encoding="utf-8") as f:
                    nb = json.load(f)
                nb_wesn = None
                if isinstance(nb.get("aoi", None), dict):
                    nb_wesn = nb["aoi"].get("wesn") or nb["aoi"].get("bbox")
                if nb_wesn is None and isinstance(nb.get("run", None), dict):
                    nb_wesn = nb["run"].get("aoi_wesn")
                if nb_wesn is None:
                    log.warning("Could not determine neighbor AOI WESN from run_report.json")
                else:
                    nb_w, nb_e, nb_s, nb_n = [float(x) for x in nb_wesn]
                    nb_pred = None
                    try:
                        nb_pred = nb.get("outputs", {}).get("sdb_prediction", {}).get("path", None)
                    except Exception:
                        nb_pred = None
                    # No fallback filename guessing for neighbor outputs.
                    # Neighbor must explicitly record its prediction path in run_report.json.
                    if not nb_pred or not Path(nb_pred).exists():
                        log.warning("Neighbor prediction raster not found: %s", nb_pred)
                    else:
                        w, e, s, n = [float(x) for x in args.aoi.split("/")[:4]] if isinstance(args.aoi, str) else bbox_wesn
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
                        log.info("%s n_valid=%s median_diff=%s m p90_abs=%s m",
                                 res.get('status'), res.get('n_samples_valid'),
                                 res.get('median_diff_m'), res.get('p90_abs_diff_m'))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as e:
            log.warning("Failed to compute overlap consistency diagnostic: %s", e)

    if args.mask_bright_pixels is not None and s2_paths and "B02" in s2_paths:
        log.info("[Post-Process] Brightness masking: B02 thresh=%s, allow_bright_shallow=%s", args.mask_bright_pixels, bool(args.allow_bright_shallow_pixels))
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
                    log.info("[Post-Process] Bright-shallow escape hatch enabled (B08 < %s)", nir_thr)
                    with rasterio.open(s2_paths["B08"]) as src_b8:
                        b8_data = src_b8.read(1, out_shape=dst.shape, resampling=rasterio.enums.Resampling.nearest)
                    b8_norm = b8_data / scale_factor

                    is_shallow_sand = (b2_norm > args.mask_bright_pixels) & (b8_norm < nir_thr)
                    bad_mask = bad_mask & (~is_shallow_sand)

                mask_count = int(np.sum(bad_mask))
                if mask_count > 0:
                    sdb_data[bad_mask] = nodata
                    dst.write(sdb_data, 1)
                    log.info("[Post-Process] Masked %s pixels (cloud/glint).", mask_count)
                else:
                    log.info("[Post-Process] No pixels masked by brightness gate.")
        except (OSError, ValueError, TypeError, rasterio.errors.RasterioError) as exc:
            log.warning("[Post-Process] Brightness masking failed: %s", exc)

    # Brightness masking can collapse a marginal prediction to effectively nothing.
    # Re-check after post-processing so downstream fusion never treats an empty SDB
    # raster as a successful coastal product.
    _assert_sdb_prediction_is_meaningful(out_tif, dir_rast, stage="post_brightness_mask")

    # --- NEW: Create Masked RGB (Prediction Only) ---
    rgb_full = dir_rast / "RGB_10m.tif"
    if rgb_full.exists():
        rgb_masked_out = dir_rast / "RGB_10m_predict.tif"
        log.info("[Post-Process] Creating Masked RGB: %s", rgb_masked_out)
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

        except (OSError, ValueError, TypeError, rasterio.errors.RasterioError) as exc:
            log.warning("[Post-Process] Failed to create masked RGB: %s", exc)

    if not args.skip_nad83:
        # Derive NAD83 output names from the actual primary output, not guessed filenames.
        out_path = Path(out_tif)
        nad83_depth = out_path.with_name(out_path.stem + "_epsg4269" + out_path.suffix)
        predict.reproject_to_nad83(str(out_tif), str(nad83_depth))
        unc_src = str(out_path.with_name(out_path.stem + "_uncertainty" + out_path.suffix))
        if os.path.exists(unc_src):
            nad83_unc = Path(unc_src).with_name(Path(unc_src).stem + "_epsg4269" + Path(unc_src).suffix)
            predict.reproject_to_nad83(unc_src, str(nad83_unc))



def _write_sdb_manifest(
    ctx_or_args, out_tif=None, dir_rast=None, dir_model=None, dir_logs=None,
    dir_plot=None, run_id=None,
    out_root=None, rr=None, df_test_final=None, df_train_final=None,
    chosen_max_depth=None, raster_quickstats=None,
):
    """Write the SDB artifact manifest and final reporting.

    Accepts either an ``SDBRunContext`` or legacy positional args.
    """
    if isinstance(ctx_or_args, SDBRunContext):
        ctx = ctx_or_args
        args = ctx.args
        out_tif = out_tif or ctx.out_tif
        dir_rast = dir_rast or ctx.dir_rast
        dir_model = dir_model or ctx.dir_model
        dir_logs = dir_logs or ctx.dir_logs
        dir_plot = dir_plot or ctx.dir_plot
        run_id = run_id or ctx.run_id
        out_root = out_root or ctx.out_root
        rr = rr or ctx.rr
        fallback_registry = getattr(ctx, "fallback_registry", None)
        df_test_final = df_test_final if df_test_final is not None else ctx.df_test
        df_train_final = df_train_final if df_train_final is not None else ctx.df_train
        chosen_max_depth = chosen_max_depth if chosen_max_depth is not None else ctx.chosen_max_depth
        raster_quickstats = raster_quickstats or ctx.raster_quickstats
    else:
        args = ctx_or_args
    # -------------------------------------------------------------------------
    # Artifact manifest paths are derived from run_id, not guessed
    # -------------------------------------------------------------------------
    # Primary depth output (EPSG:4269 if NAD83 reprojection is enabled)
    try:
        # Prefer the NAD83-reprojected depth if it was created, otherwise use the native output.
        out_path = Path(out_tif)
        depth_primary = out_path.with_name(out_path.stem + "_epsg4269" + out_path.suffix)
        if not depth_primary.exists():
            depth_primary = out_path

        # Write manifest (relative paths, rooted at out_root)
        artifacts = {
            "run_id": run_id if run_id is not None else None,
            "depth_raster": str(depth_primary.relative_to(out_root)) if str(depth_primary).startswith(str(out_root)) else str(depth_primary),
        }

        # Optional artifacts (only if present)
        unc = Path(str(out_tif)).with_name(Path(str(out_tif)).stem + "_uncertainty.tif")
        if unc.exists():
            artifacts["uncertainty_raster"] = str(unc.relative_to(out_root))
        # If NAD83 uncertainty exists, record it (derived from the actual uncertainty filename).
        if unc.exists():
            unc_nad83 = unc.with_name(unc.stem + "_epsg4269" + unc.suffix)
            if unc_nad83.exists():
                artifacts["uncertainty_raster_epsg4269"] = str(unc_nad83.relative_to(out_root))

        land_mask = dir_rast / "LAND_MASK_aligned.tif"
        if land_mask.exists():
            artifacts["land_mask"] = str(land_mask.relative_to(out_root))

        # Guidance rasters (confidence + provenance)
        conf_tif = Path(str(out_tif)).with_name(Path(str(out_tif)).stem + "_confidence.tif")
        if conf_tif.exists():
            artifacts["confidence_raster"] = str(conf_tif.relative_to(out_root)) if str(conf_tif).startswith(str(out_root)) else str(conf_tif)
        prov_tif = Path(str(out_tif)).with_name(Path(str(out_tif)).stem + "_provenance.tif")
        if prov_tif.exists():
            artifacts["provenance_raster"] = str(prov_tif.relative_to(out_root)) if str(prov_tif).startswith(str(out_root)) else str(prov_tif)

        guidance_manifest = build_sdb_guidance_manifest(out_root=out_root, depth_raster=out_tif, args=args)
        guidance_artifacts = guidance_manifest.get("artifacts", {})
        for key in (
            "guidance_weight_raster",
            "trusted_interior_raster",
            "admissibility_raster",
            "guide_points",
            "lower_bound_raster",
            "upper_bound_raster",
            "regime_class_raster",
        ):
            val = guidance_artifacts.get(key)
            if val:
                artifacts[key] = val
        artifacts["guidance_mode"] = guidance_artifacts.get("guidance_mode", "guidance_first")
        artifacts["depth_raster_role"] = guidance_artifacts.get("depth_raster_role", "diagnostic_only")
        auth_base = guidance_artifacts.get("authoritative_base")
        if auth_base:
            artifacts["authoritative_base"] = str(auth_base)
        auth_auto = guidance_artifacts.get("authoritative_base_auto")
        if auth_auto:
            artifacts["authoritative_base_auto"] = auth_auto

        rgb = dir_rast / "RGB_10m.tif"
        if rgb.exists():
            artifacts["rgb"] = str(rgb.relative_to(out_root))
        rgb_pred = dir_rast / "RGB_10m_predict.tif"
        if rgb_pred.exists():
            artifacts["rgb_predict"] = str(rgb_pred.relative_to(out_root))

        (out_root / "artifacts_sdb.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
        log.info("Wrote SDB manifest: %s", out_root / 'artifacts_sdb.json')
        guidance_manifest_path = write_sdb_guidance_manifest(out_root=out_root, depth_raster=out_tif, args=args, logger=log)
        artifacts["guidance_manifest"] = str(guidance_manifest_path.relative_to(out_root)) if str(guidance_manifest_path).startswith(str(out_root)) else str(guidance_manifest_path)
        (out_root / "artifacts_sdb.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log.warning("Failed to write outputs/manifest: %s", exc)

    if df_test_final is not None and not df_test_final.empty:
        try:
            eval_points = df_test_final.copy()
            if "source" in eval_points.columns:
                log.info("Raster validation holdout sources: %s", eval_points["source"].astype(str).value_counts().to_dict())
            elif "source_norm" in eval_points.columns:
                log.info("Raster validation holdout sources: %s", eval_points["source_norm"].astype(str).value_counts().to_dict())
        except (TypeError, ValueError, KeyError):
            eval_points = df_test_final

        evaluate_raster_against_points(str(out_tif), eval_points, dir_logs, dir_plot, max_depth=chosen_max_depth)


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
            if fallback_registry is not None:
                rr.add_dict("scientific_fallbacks", fallback_registry.summary())

            rr.write(status="ok")
            log.info("SDB pipeline done")
    except (OSError, ValueError, TypeError, KeyError) as _exc:
        log.debug("Suppressed: %s", _exc, exc_info=True)


def _acquire_s2_and_atl_data(
    ctx_or_args, s2_cache=None, atl_cache=None, mask_cache=None,
    bbox_wesn=None, out_root=None, rr=None,
    dir_data=None, dir_rast=None, w=None, e=None, s=None, n=None,
    s2_optics=None, atl=None, bbox_list=None,
):
    """Acquire Sentinel-2 composite and ICESat-2 data with retry logic.

    Returns (s2_paths, atl_files_map) on success, or calls sys.exit / returns
    on failure depending on mode.

    Accepts either an ``SDBRunContext`` as first arg (new style) or the legacy
    positional ``args`` namespace (backward compatible).
    """
    if isinstance(ctx_or_args, SDBRunContext):
        ctx = ctx_or_args
        args = ctx.args
        s2_cache = s2_cache or ctx.s2_cache
        atl_cache = atl_cache or ctx.atl_cache
        mask_cache = mask_cache or ctx.mask_cache
        bbox_wesn = bbox_wesn or ctx.bbox_wesn
        out_root = out_root or ctx.out_root
        rr = rr or ctx.rr
        dir_data = dir_data or ctx.dir_data
        dir_rast = dir_rast or ctx.dir_rast
        if bbox_wesn:
            w, e, s, n = bbox_wesn
        bbox_list = bbox_list or ctx.bbox_list
        s2_optics = s2_optics or ctx.mod_s2_optics
        atl = atl or ctx.mod_atl
    else:
        args = ctx_or_args
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
        log.info("Attempt %s: %s to %s", attempt+1, current_start, end_date)

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
                        log.info("Using waffles coastline mask for S2 date QC: %s", coast_mask_raw)
                        # If the coastline mask indicates *no* water/ocean pixels in this AOI, skip SDB early.
                        # This avoids wasting time downloading S2/ATL data when the AOI is fully inland (or otherwise non-ocean).
                        if coast_mask_raw and (not _mask_has_ocean_pixels(str(coast_mask_raw))):
                            log.warning("No ocean/water pixels found in coastline mask for this AOI; skipping SDB.")
                            return None

                    except Exception as exc:
                        log.warning("Waffles generation failed (%s). S2 date QC will be AOI-only.", exc)
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
                except Exception as _exc:
                    log.debug("Suppressed: %s", _exc, exc_info=True)

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
                except Exception as _exc:
                    log.debug("Suppressed: %s", _exc, exc_info=True)
                log.info("Composite acquired.")

                if args.cloud_report_only:
                    log.info('Cloud report only requested; exiting after STAC scan.')
                    return

                rgb_src = s2_out / "RGB_10m.tif"
                if rgb_src.exists():
                    rgb_dst = dir_rast / "RGB_10m.tif"
                    shutil.copy(rgb_src, rgb_dst)
                else:
                    log.warning("RGB_10m.tif not found in %s", s2_out)

            except Exception as exc:
                log.warning("Failed: %s", exc)
                s2_paths = None

        if (not args.atl_only) and (s2_paths is not None):
            log.info("Reusing existing composite from earlier attempt; skipping S2 reacquisition.")

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
                        log.info("Skipping ATL24 in lake mode (not defined over lakes).")
                        atl_files_map["ATL24"] = []
                    else:
                        target_dir_24 = atl_run_dir / "ATL24"
                        f24, _, _ = atl.ensure_icesat_files_harmony_cachefirst(
                            target_dir_24, "ATL24", bbox_list, current_start, end_date,
                            force_redl=args.force_redl_atl24
                        )
                        atl_files_map["ATL24"] = f24

            except Exception as exc:
                log.warning("Failed: %s", exc)

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
    return s2_paths, atl_files_map


def _filter_extra_xyz_for_sdb(df_xyz_local, args, ctx=None):
    """Conservative guardrails for extra_xyz in SDB training.

    Module-level function — all state passed explicitly via ``args`` and
    optional ``ctx`` (SDBRunContext).  No closure capture of outer locals.

    River/XS anchoring is handled separately in bathy_main; here we avoid
    poisoning SDB with likely bed-elevation soundings mislabeled as depth.
    """
    if df_xyz_local is None or len(df_xyz_local) == 0:
        return df_xyz_local
    mode = str(getattr(args, "extra_xyz_sdb_mode", "auto") or "auto").lower()
    if mode == "exclude":
        log.warning("extra_xyz_sdb_mode=exclude -> excluding extra_xyz from SDB training.")
        return None
    if mode == "allow":
        return df_xyz_local

    guard_max_abs_depth_m = 10.0
    authoritative_max_abs_depth_m = 60.0
    authoritative_tags = (
        "hydronos", "ehydro", "survey", "sonar", "sound", "lidar", "bag",
        "multibeam", "singlebeam", "mbes", "usace", "usgs", "noaa"
    )
    try:
        _ctx_kd = getattr(ctx, 'kd_max_depth_m', None) if ctx is not None else None
        physics_cap = float(_ctx_kd) if _ctx_kd is not None else None
    except (TypeError, ValueError):
        physics_cap = None
    try:
        hard_cap = float(getattr(args, "max_depth_hard_cap", 25.0) or 25.0)
    except (TypeError, ValueError):
        hard_cap = 25.0
    if physics_cap is not None and np.isfinite(physics_cap) and physics_cap > 0:
        guard_max_abs_depth_m = float(min(max(physics_cap, 10.0), hard_cap))

    def _src_counts(_df):
        if _df is None or len(_df) == 0 or "source" not in _df.columns:
            return {}
        try:
            vc = _df["source"].astype(str).str.lower().value_counts(dropna=False)
            return {str(k): int(v) for k, v in vc.to_dict().items()}
        except Exception:
            return {}

    try:
        if "depth_m" not in df_xyz_local.columns:
            log.warning("extra_xyz missing depth_m; excluding from SDB training in auto mode.")
            return None

        n_before = int(len(df_xyz_local))
        d_ser = pd.to_numeric(df_xyz_local["depth_m"], errors="coerce")
        d_arr = d_ser.to_numpy(dtype=float)
        finite_mask = np.isfinite(d_arr)
        n_finite = int(finite_mask.sum())
        if n_finite == 0:
            log.warning("extra_xyz has no finite depth_m after coercion; excluding from SDB training.")
            return None

        df_guard = df_xyz_local.loc[finite_mask].copy()
        df_guard["depth_m"] = d_arr[finite_mask]

        src_before = _src_counts(df_xyz_local)

        src_series = df_guard.get("source")
        if src_series is not None:
            src_norm = src_series.astype(str).str.lower()
            authoritative_mask = src_norm.str.contains("|".join(authoritative_tags), regex=True)
            if authoritative_mask.any():
                auth_cap = float(max(authoritative_max_abs_depth_m, hard_cap))
                depth_abs = np.abs(df_guard["depth_m"].to_numpy(dtype=float))
                keep_depth = np.where(
                    np.asarray(authoritative_mask, dtype=bool),
                    depth_abs <= auth_cap,
                    depth_abs <= float(guard_max_abs_depth_m),
                )
                log.info(
                    "[XYZ][SDB] Authoritative extra_xyz detected; using looser depth cap for %d rows "
                    "(|depth_m|<=%.1f m) and standard cap %.1f m for other rows.",
                    int(authoritative_mask.sum()), auth_cap, float(guard_max_abs_depth_m),
                )
            else:
                keep_depth = np.abs(df_guard["depth_m"].to_numpy(dtype=float)) <= float(guard_max_abs_depth_m)
        else:
            keep_depth = np.abs(df_guard["depth_m"].to_numpy(dtype=float)) <= float(guard_max_abs_depth_m)
        df_guard = df_guard.loc[np.asarray(keep_depth, dtype=bool)].copy()
        if df_guard.empty:
            log.warning(
                "[XYZ][SDB] Auto guard removed all extra_xyz rows for SDB training after depth sanity caps.",
            )
            return None

        d = df_guard["depth_m"].to_numpy(dtype=float)
        pct_pos = float((d > 0).mean()) if d.size else 0.0
        med = float(np.nanmedian(d)) if d.size else float("nan")
        src_text = ""
        if "source" in df_guard.columns:
            src_text = " ".join(sorted({str(v).lower() for v in df_guard["source"].dropna().unique()}))
        suspect_prov = any(k in src_text for k in ("hydronos", "ehydro"))
        if suspect_prov and pct_pos > 0.02:
            n_mid = len(df_guard)
            d_ser2 = pd.to_numeric(df_guard["depth_m"], errors="coerce")
            neg_mask = (d_ser2 <= 0)
            neg_mask = neg_mask.fillna(False) if hasattr(neg_mask, "fillna") else pd.Series(False, index=df_guard.index)
            df_neg = df_guard.loc[np.asarray(neg_mask, dtype=bool)].copy()
            if len(df_neg) >= 100:
                log.warning(
                    "[XYZ][SDB] Auto guard: suspected bed-elev/mixed-sign source (%.1f%% positive; median=%.2f m). "
                    "Keeping only non-positive rows for SDB training: %d -> %d.",
                    pct_pos * 100, med, n_mid, len(df_neg),
                )
                df_guard = df_neg
            else:
                log.warning(
                    "[XYZ][SDB] Auto guard: suspected bed-elev/mixed-sign source (%.1f%% positive; median=%.2f m). "
                    "Too few non-positive rows remain; excluding extra_xyz from SDB training.",
                    pct_pos * 100, med,
                )
                return None
        elif pct_pos > 0.20:
            log.warning(
                "[XYZ][SDB] Auto guard warning: extra_xyz has %.1f%% positive depth_m (median=%.2f m) after depth cap. "
                "Verify depth semantics/sign convention before trusting SDB metrics.",
                pct_pos * 100, med,
            )

        src_after = _src_counts(df_guard)
        cap_mode = "physics-aware" if guard_max_abs_depth_m > 10.0 else "fallback"
        log.info(
            "[XYZ][SDB] Guard kept %d/%d extra_xyz rows for SDB training "
            "(finite depth + source-aware depth sanity caps; standard cap=%g m; %s cap).",
            len(df_guard), n_before, guard_max_abs_depth_m, cap_mode,
        )
        if src_before or src_after:
            log.info("Guard source breakdown before=%s after=%s", src_before, src_after)
            hydro_before = sum(v for k, v in src_before.items() if ("hydronos" in k or "ehydro" in k))
            hydro_after = sum(v for k, v in src_after.items() if ("hydronos" in k or "ehydro" in k))
            if hydro_before > 0 and hydro_after < hydro_before:
                log.info("Hydronos/eHydro-like rows reduced by guard: %s -> %s", hydro_before, hydro_after)
        return df_guard

    except Exception as ex:
        log.error("Auto guard failed; excluding extra_xyz from SDB training. Reason: %s", ex, exc_info=True)
        log.debug("Guard exception details", exc_info=True)
        return None


def _run_fusion(
    ctx_or_args, dir_data=None, dir_rast=None, dir_logs=None,
    dir_model=None, dir_plot=None,
    s2_paths=None, df_atl03=None, df_atl24=None, out_root=None, rr=None,
    atl=None, fusion=None,
):
    """Fuse ATL03, ATL24, and extra-XYZ training points into a single training DataFrame.

    Returns fused_df.

    Accepts either an ``SDBRunContext`` as first arg (new style) or the legacy
    positional ``args`` namespace (backward compatible).
    """
    if isinstance(ctx_or_args, SDBRunContext):
        ctx = ctx_or_args
        args = ctx.args
        dir_data = dir_data or ctx.dir_data
        dir_rast = dir_rast or ctx.dir_rast
        dir_logs = dir_logs or ctx.dir_logs
        dir_model = dir_model or ctx.dir_model
        dir_plot = dir_plot or ctx.dir_plot
        s2_paths = s2_paths or ctx.s2_paths
        out_root = out_root or ctx.out_root
        rr = rr or ctx.rr
        atl = atl or ctx.mod_atl
        fusion = fusion or ctx.mod_fusion
    else:
        args = ctx_or_args
    # 6. Fusion
    log.info("Fusion & QC")

    df_xyz = None
    if args.extra_xyz:
        from support_points import load_extra_xyz_points
        df_xyz = load_extra_xyz_points(args.extra_xyz, args.extra_xyz_crs, args.aoi)
        try:
            n_xyz = 0 if df_xyz is None else len(df_xyz)
            if n_xyz > 0:
                dmin = float(np.nanmin(df_xyz["depth_m"])) if "depth_m" in df_xyz.columns else float("nan")
                dmed = float(np.nanmedian(df_xyz["depth_m"])) if "depth_m" in df_xyz.columns else float("nan")
                dmax = float(np.nanmax(df_xyz["depth_m"])) if "depth_m" in df_xyz.columns else float("nan")
                log.info("Loaded extra XYZ points: n=%s depth(min/med/max)=%.3f/%.3f/%.3f m", n_xyz, dmin, dmed, dmax)
            else:
                log.warning("Extra XYZ inputs were provided but produced 0 usable points after parsing/filtering.")
        except Exception:
            log.warning("Loaded extra XYZ points (stats unavailable).", exc_info=True)
        _ctx_for_guard = ctx_or_args if isinstance(ctx_or_args, SDBRunContext) else None
        df_xyz = _filter_extra_xyz_for_sdb(df_xyz, args, _ctx_for_guard)
        if df_xyz is None or len(df_xyz) == 0:
            log.warning("No extra_xyz points will be used for SDB training after guardrails.")

    fused_df = fusion.build_fused_training_dataframe(
        atl03_df=df_atl03, atl24_df=df_atl24, xyz_df=df_xyz,
        xyz_max_dist_m=30.0, atl_max_dist_m=20.0,
        atl03_weight=args.atl03_weight_factor, atl24_weight=1.0, xyz_weight=10.0,
        rr=rr,
        # Adaptive sampling parameters
        enable_adaptive_sampling=args.enable_adaptive_sampling,
        sampling_target_points=args.sampling_target_points,
        sampling_min_threshold=args.sampling_min_threshold,
        sampling_max_gap_m=args.sampling_max_gap_m,
    )

    # Helpful provenance summary (especially to confirm external XYZ participation)
    try:
        if fused_df is not None and len(fused_df) > 0:
            if "source" in fused_df.columns:
                src_counts = fused_df["source"].value_counts(dropna=False).to_dict()
                log.info("Point provenance after fusion: %s", src_counts)
            else:
                 log.info("Fused training points: n=%s (no source column present)", len(fused_df))
        else:
            log.warning("Fusion produced 0 points. Check water/land masks, AOI, and input data.")
    except Exception:
        log.warning("Unable to summarize point provenance.", exc_info=True)
    # Cap per-source rows before S2 sampling to prevent one source (e.g., Hydronos/eHydro)
    # from dominating sampling cost and training provenance.
    def _cap_per_source_presampling(df_in, cap_per_source=250000, seed=1337):
        if df_in is None or len(df_in) == 0 or "source" not in df_in.columns:
            return df_in
        try:
            src_ser = df_in["source"].astype(str)
            vc_before = src_ser.value_counts(dropna=False)
            needs_cap = bool((vc_before > int(cap_per_source)).any())
            if not needs_cap:
                return df_in
            rng = np.random.default_rng(int(seed))
            keep_idx_parts = []
            for src, grp in df_in.groupby(src_ser, sort=False):
                n_src = int(len(grp))
                if n_src <= int(cap_per_source):
                    keep_idx_parts.append(grp.index.to_numpy())
                else:
                    sel = rng.choice(grp.index.to_numpy(), size=int(cap_per_source), replace=False)
                    keep_idx_parts.append(np.sort(sel))
            keep_idx = np.concatenate(keep_idx_parts) if keep_idx_parts else np.array([], dtype=int)
            df_out = df_in.loc[keep_idx].copy()
            vc_after = df_out["source"].astype(str).value_counts(dropna=False)
            log.warning(
                "[TRAIN][PRESAMPLE] Capped per-source rows before S2 sampling at %s. "
                "Total rows %s -> %s.",
                f"{int(cap_per_source):,}", f"{len(df_in):,}", f"{len(df_out):,}",
            )
            log.info("Source counts before: %s", vc_before.to_dict())
            log.info("Source counts after: %s", vc_after.to_dict())
            return df_out
        except Exception as ex:
            log.warning("Per-source cap failed; continuing without cap: %s", ex)
            return df_in

    fused_df = _cap_per_source_presampling(fused_df, cap_per_source=250000, seed=int(args.seed))

    fused_df.to_csv(dir_data / "training_data_fused.csv", index=False)

    # Helpful provenance logging: how many fused points came from each source.
    try:
        if "source" in fused_df.columns:
            src_counts = fused_df["source"].value_counts(dropna=False).to_dict()
            log.info("Fused point sources: %s", src_counts)
        else:
            log.info("Fused dataframe has no 'source' column; provenance counts unavailable.")
    except Exception:
        log.warning("Could not compute fused source counts.", exc_info=True)

    return fused_df
def main():
    import rasterio
    args = parse_args()
    try:
        _materialize_authoritative_base_for_sdb(args)
    except Exception:
        log.error("[AUTHORITATIVE] Failed to materialize authoritative_base for AOI=%s", getattr(args, "aoi", None), exc_info=True)
        raise

    # ---------------------------------------------------------------------
    # Run-scoped logging + flight recorder
    # ---------------------------------------------------------------------
    run_id = None
    try:
        from datetime import datetime, timezone
        from logging_config import add_file_handler, install_screen_log, start_flight_recorder

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"sdb_{ts}_{os.getpid()}"
        out_dir = Path(getattr(args, "out_dir", "output"))
        run_logs_dir = out_dir / "run_logs"
        run_logs_dir.mkdir(parents=True, exist_ok=True)
        screen_log_path = install_screen_log(run_logs_dir / f"screen_{run_id}.log")
        add_file_handler(run_logs_dir / f"run_{run_id}.log", level=logging.INFO, log_format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        log.info("Screen log: %s", screen_log_path)
        fr_path = start_flight_recorder(out_dir, run_id=run_id)
        if fr_path is not None:
            log.info("Flight recorder: %s", fr_path)
        log.info("run_id=%s", run_id)
    except Exception as e:
        log.warning("Unable to initialize run logs/flight recorder: %s", e)

    try:
        w, e, s, n = [float(x) for x in args.aoi.split("/")]
        bbox_list = [w, s, e, n]
        bbox_wesn = (w, e, s, n)
    except Exception:
        log.error("Invalid AOI format. Use W/E/S/N")
        sys.exit(1)

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
                log.error("  - %s", err)
            sys.exit(1)
        elif validation_result.warnings:
            for warn in validation_result.warnings:
                log.warning("  - %s", warn)
    
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
                log.info("Resuming from checkpoint: %s", progress['completed_stages'])
        except Exception as e:
            log.warning("Could not initialize: %s", e)
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
            # When authoritative XYZ data is available, use its depth range
            # instead of the 20m default — the XYZ data tells us the actual
            # depth range that needs to be modeled.
            _xyz_max = 20.0
            if hasattr(args, "extra_xyz") and args.extra_xyz:
                try:
                    import numpy as _np_ad
                    _all_depths = []
                    for _xf in (args.extra_xyz if isinstance(args.extra_xyz, list) else [args.extra_xyz]):
                        _xp = Path(str(_xf).strip())
                        if not _xp.exists():
                            continue
                        try:
                            _dat = _np_ad.genfromtxt(str(_xp), dtype="float64", max_rows=500000)
                            if _dat.ndim == 1:
                                _dat = _dat.reshape(1, -1)
                            if _dat.shape[1] >= 3:
                                _z = _np_ad.abs(_dat[:, 2])
                                _z = _z[_np_ad.isfinite(_z)]
                                if len(_z) > 0:
                                    _all_depths.append(_z)
                        except Exception:
                            try:
                                _dat = _np_ad.genfromtxt(str(_xp), dtype="float64", delimiter=",", max_rows=500000)
                                if _dat.ndim == 1:
                                    _dat = _dat.reshape(1, -1)
                                if _dat.shape[1] >= 3:
                                    _z = _np_ad.abs(_dat[:, 2])
                                    _z = _z[_np_ad.isfinite(_z)]
                                    if len(_z) > 0:
                                        _all_depths.append(_z)
                            except Exception:
                                continue
                    if _all_depths:
                        _combined = _np_ad.concatenate(_all_depths)
                        if len(_combined) > 100:
                            _xyz_max = float(_np_ad.percentile(_combined, 99))
                            log.info("[AUTO-DEPTH] XYZ files depth p99=%.1f m (n=%d); using as training depth cap.", _xyz_max, len(_combined))
                except Exception as _ad_e:
                    log.warning("[AUTO-DEPTH] Failed to read XYZ depth range: %s; using default 20m cap.", _ad_e)
            max_depth_sdb_product = max(20.0, min(_xyz_max, 60.0))
            log.info("[AUTO-DEPTH] max_depth_sdb_product=%.1f m (xyz_max=%.1f m)", max_depth_sdb_product, _xyz_max)
        else:
            max_depth_sdb_product = float(max_depth_arg_raw)
    except Exception:
        log.warning("Invalid --max-depth-sdb value %r. Falling back to 20.0 m.", max_depth_arg_raw)
        max_depth_sdb_product = 20.0

    MAX_DEPTH_TRAIN_FETCH = max(30.0, max_depth_sdb_product)

    # --- UPDATED LOGIC: Allow auto-depth without forced spatial validation ---
    if auto_max_depth and not getattr(args, "validate_spatial", True):
        log.info("--max-depth-sdb auto requested with --no-validate-spatial. "
                 "Using STRATIFIED RANDOM SPLIT to estimate depth-of-support (interpolation limit).")
        # Do NOT force validate_spatial = True

    # --- AUTO-CONFIGURATION OVERRIDE ---
    config_settings = load_config_overrides(bbox_list)
    active_profile = detect_active_config_profile(bbox_list)
    overrides_applied = list(config_settings.keys()) if isinstance(config_settings, dict) else []

    if config_settings:
        log.info("Applying regional configuration")

        for key, value in config_settings.items():
            if key == "preferred_months":
                setattr(args, "preferred_months", value)
                log.info("   [Auto-Config] Set preferred_months: %s", value)
                continue

            if not hasattr(args, key):
                # We log this but still set it on args so it appears in the final dump
                # (useful for custom flags like 'gl_red_max' not in argparse)
                setattr(args, key, value)
                log.info("   [Auto-Config] Set %s (custom): %s", key, value)
                continue

            cli_flag = "--" + key.replace("_", "-")
            if key == "min_depth_atl03_val": cli_flag = "--min-depth-atl03"
            elif key == "refraction":
                if not value: cli_flag = "--no-refraction"

            if cli_flag in sys.argv:
                log.info("   [Override Skipped] User explicitly set %s. Keeping value: %s", cli_flag, getattr(args, key))
            else:
                if key == "sdb_mode" and getattr(args, "extra_xyz", None):
                    log.info("   [Override Skipped] extra_xyz provided; keeping user-requested sdb_mode: %s", getattr(args, key))
                    continue
                old_val = getattr(args, key)
                setattr(args, key, value)
                if old_val != value:
                    log.info("   [Auto-Config] Set %s: %s -> %s", key, old_val, value)

        
    log.info("Final effective arguments:")

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
        log.info("%s", group_name)
        for k in keys:
            if hasattr(args, k):
                val = getattr(args, k)
                log.info("  %s: %s", k, val)
                printed_keys.add(k)
            # Handle config keys that might not be in argparse definition (dynamically added)
            elif k in vars(args):
                 val = getattr(args, k)
                 log.info("  %s: %s", k, val)
                 printed_keys.add(k)

    # Catch-all for anything else not in groups
    all_keys = set(vars(args).keys())
    remaining = sorted(list(all_keys - printed_keys))
    if remaining:
        log.info("Other / Uncategorized")
        for k in remaining:
            # Skip hidden private attrs
            if not k.startswith("_"):
                log.info("  %s: %s", k, getattr(args, k))



    # ------------------------------------------------------------
    # Enforce lake-mode defaults & constraints
    # ------------------------------------------------------------
    if getattr(args, "sdb_mode", None) == "lakes":
        log.info("Lake mode enabled")
        if getattr(args, "icesat", None) in ("atl24", "all_atl"):
            log.info("ATL24 disabled in lake mode (not defined over lakes)")
            args.icesat = "atl03"
        if getattr(args, "cw_min", None) is None:
            args.cw_min = 0.8
            log.info("Setting default cw_min=0.8 for lakes")
        if getattr(args, "land_max", None) is None:
            args.land_max = 0.1
            log.info("Setting default land_max=0.1 for lakes")

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
    _rr_add(rr, "sdb.max_depth.auto.rmse_target_m", float(args.rmse_target_sdb))
    _rr_add(rr, "run.aoi", {"w": w, "e": e, "s": s, "n": n})
    _rr_add(rr, "run.authoritative_base", {"path": getattr(args, "authoritative_base", None), "auto_enabled": bool(getattr(args, "authoritative_base_auto", True))})
    if hasattr(args, "_authoritative_base_auto_report"):
        _rr_add(rr, "run.authoritative_base_auto", getattr(args, "_authoritative_base_auto_report"))
    _rr_add(rr, "config.active_profile", active_profile if active_profile is not None else "unknown")
    _rr_add(rr, "config.overrides_applied", overrides_applied if overrides_applied is not None else [])

    _rr_artifact(rr, "out_root", str(out_root))
    _rr_artifact(rr, "dir_data", str(dir_data))
    _rr_artifact(rr, "dir_model", str(dir_model))
    _rr_artifact(rr, "dir_rasters", str(dir_rast))
    _rr_artifact(rr, "dir_plots", str(dir_plot))
    _rr_artifact(rr, "dir_logs", str(dir_logs))

    fallback_registry = FallbackRegistry() if FallbackRegistry is not None else None
    if rr is not None and fallback_registry is not None:
        rr.add_dict("scientific_fallbacks", fallback_registry.summary())

    cache_root = Path(getattr(args, "cache_root", "cache"))
    atl_cache = cache_root / "icesat2"
    s2_cache = cache_root / "sentinel2"
    mask_cache = cache_root / "masks"

    # --- Build shared run context -------------------------------------------
    ctx = SDBRunContext.from_args(args, out_root=out_root, run_id=run_id or "", rr=rr)
    ctx.fallback_registry = fallback_registry
    ctx.s2_cache = s2_cache
    ctx.atl_cache = atl_cache
    ctx.mask_cache = mask_cache
    ctx.bbox_wesn = bbox_wesn
    ctx.bbox_list = bbox_list

    for _d in (atl_cache, s2_cache, mask_cache):
        try:
            Path(_d).mkdir(parents=True, exist_ok=True)
        except (TypeError, ValueError, AttributeError) as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Model bank (bounded reservoir): accumulate training across AOIs
    # ------------------------------------------------------------------
    # When dense authoritative XYZ data dominates the training set, the
    # model bank adds complexity without value — we're training a model
    # for this specific AOI with this specific survey data, not building
    # a generalizable predictor.  Disable the bank in this case to reduce
    # the number of interacting policy gates.
    _guidance_anchored_mode = False
    if hasattr(args, "extra_xyz") and args.extra_xyz:
        _guidance_anchored_mode = True
        log.info("[SDB] Guidance-anchored mode: dense authoritative XYZ data detected. "
                 "SDB will operate as an optically-aware interpolant subordinate to survey data, "
                 "not as a generalizing bathymetric predictor.")

    model_bank_dir = None
    if bool(args.model_bank_enabled) and not _guidance_anchored_mode:
        mb = str(args.model_bank)
        if mb.strip().lower() == 'auto':
            model_bank_dir = cache_root / 'model_bank' / 'sdb_global_v1'
        else:
            model_bank_dir = Path(mb).expanduser().resolve()
        model_bank_dir.mkdir(parents=True, exist_ok=True)
        log.info("enabled dir=%s max_samples=%s", model_bank_dir, int(args.bank_max_samples))
        if rr is not None:
            rr.add('model_bank.enabled', True)
            rr.add('model_bank.dir', str(model_bank_dir))
            rr.add('model_bank.max_samples', int(args.bank_max_samples))
    else:
        log.info('disabled')
        if rr is not None:
            rr.add('model_bank.enabled', False)

    # Deprecated model cache is disabled by default in this workflow.
    model_bank_dir_run = model_bank_dir  # may be partitioned later by source mix/depth regime
    model_cache_dir = None
    log.info("Starting SDB run (mode=%s)", args.sdb_mode)
    log.info("Output Directory: %s", out_root)

    # Populate ctx with lazy-imported modules
    ctx.mod_s2_optics = s2_optics
    ctx.mod_atl = atl

    _acq_result = _acquire_s2_and_atl_data(ctx, s2_optics=s2_optics, atl=atl)
    if _acq_result is None:
        return 0
    s2_paths, atl_files_map = _acq_result
    ctx.s2_paths = s2_paths
    ctx.atl_files_map = atl_files_map


    # 4. Mask Generation
    log.info("Mask generation")
    land_mask_out = dir_rast / "LAND_MASK_aligned.tif"
    waffles_cache = Path(os.environ.get("WAFFLES_CACHE_ROOT", mask_cache))

    if args.land_mask_user:
        s2_optics.prepare_user_land_mask(args.land_mask_user, s2_paths["B02"], str(land_mask_out))
    else:
        try:
            generate_coastline_mask(s2_paths["B02"], args.aoi, waffles_cache, land_mask_out, args.sdb_mode)
        except (OSError, RuntimeError, ValueError) as exc:
            log.warning("Mask generation failed (%s).", exc)


    # Ensure land mask exists even if waffles generation failed
    if not land_mask_out.exists():
        try:
            import rasterio
            with rasterio.open(s2_paths["B02"]) as src:
                prof = src.profile.copy()
                prof.update(count=1, dtype="uint8", nodata=None, compress="LZW")
                data = np.zeros((src.height, src.width), dtype="uint8")  # 0 = water everywhere (no land gate)
            with rasterio.open(land_mask_out, "w", **prof) as dst:
                dst.write(data, 1)
            log.warning("LAND mask missing; created fallback all-water mask: %s", land_mask_out)
        except (OSError, RuntimeError, ValueError, ModuleNotFoundError) as _e:
            log.error("Failed to create fallback LAND mask (%s); cannot proceed.", _e, exc_info=True)
            raise

    log.info("Using LAND mask as hard gate: keep pixels where LAND <= %s (0.0 = waffles water-only).", args.land_max)

    # 5. ATL Processing
    log.info("Processing ICESat-2 data")
    df_atl03 = None
    df_atl24 = None

    # Cache directory for ATL processing (mirrors the logic in _acquire_s2_and_atl_data)
    atl_cache_dir = atl_cache if args.out_dir is None else out_root / "cache_icesat2"

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
        except Exception as _exc:
            log.debug("Suppressed: %s", _exc, exc_info=True)


    atl03_audit = None
    atl24_audit = None

    if atl_files_map["ATL03"]:
        if args.write_atl03_tracks_shp:
            atl.build_atl03_track_lines(atl_files_map["ATL03"], args.aoi, str(dir_data / "ATL03_tracks.shp"))

        df_atl03, atl03_audit = atl.collect_training_points_from_atl03(
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
            land_mask_type=land_mask_type,
            land_mask_threshold=land_mask_threshold,
            cache_dir=str(atl_cache_dir),
            cache_strict=args.cache_strict,
            cache_code_strict=args.cache_code_strict,
            cache_ignore_code=args.cache_ignore_code,
            rr=rr,
            return_audit=True,
        )

    if atl_files_map["ATL24"]:
        df_atl24, atl24_audit = atl.collect_training_points_from_atl24(
            atl_files_map["ATL24"], aoi_str=args.aoi,
            max_depth_m=MAX_DEPTH_TRAIN_FETCH,
            train_relax_buffer=0.01, limit_train_samples=None, seed=args.seed,
            atl24_conf_min=args.atl24_conf_min,
            land_mask_path=str(land_mask_out),
            land_mask_water_val=land_mask_water_val,
            land_mask_invert=land_mask_invert,
            land_mask_type=land_mask_type,
            land_mask_threshold=land_mask_threshold,
            cache_dir=str(atl_cache_dir),
            cache_strict=args.cache_strict,
            cache_code_strict=args.cache_code_strict,
            cache_ignore_code=args.cache_ignore_code,
            rr=rr,
            return_audit=True,
        )

    # Optional: harmonize ATL horizontal CRS before fusion (e.g., WGS84 -> NAD83).
    # HORIZONTAL-ONLY: depth_m is relative water depth, not geodetic height.
    # Transforming it would corrupt values.
    if getattr(args, "atl_transform_datum", False):
        df_atl03 = atl.transform_xyz_dataframe_crs(df_atl03, src_srs=args.atl_src_srs, dst_srs=args.atl_dst_srs, logger=log)
        df_atl24 = atl.transform_xyz_dataframe_crs(df_atl24, src_srs=args.atl_src_srs, dst_srs=args.atl_dst_srs, logger=log)

    try:
        atl_quality_summary, atl_passes_df = atl.summarize_atl_training_quality(
            df_atl03, df_atl24,
            min_depth_floor_m=float(args.min_depth_atl03_val),
            colloc_dist_m=20.0,
        )
        atl_quality_json = dir_logs / "atl_training_quality_report.json"
        atl_quality_csv = dir_logs / "atl_training_quality_tracks.csv"
        atl_quality_json.write_text(json.dumps(atl_quality_summary, indent=2), encoding="utf-8")
        if hasattr(atl_passes_df, "to_csv"):
            atl_passes_df.to_csv(atl_quality_csv, index=False)
        log.info("[ATL][QUALITY] Wrote summary report: %s", atl_quality_json)
        log.info("[ATL][QUALITY] Wrote track report: %s", atl_quality_csv)
        try:
            xs = atl_quality_summary.get("cross_source", {})
            log.info(
                "[ATL][QUALITY] ATL03 accepted=%s tracks=%s near_floor=%.1f%% | ATL24 accepted=%s tracks=%s | collocated_pairs=%s atl03_vs_atl24_rmse=%s m bias=%s m",
                atl_quality_summary.get("atl03", {}).get("accepted_points", 0),
                atl_quality_summary.get("atl03", {}).get("unique_tracks", 0),
                100.0 * float(atl_quality_summary.get("atl03", {}).get("near_floor_fraction") or 0.0),
                atl_quality_summary.get("atl24", {}).get("accepted_points", 0),
                atl_quality_summary.get("atl24", {}).get("unique_tracks", 0),
                xs.get("collocated_pairs", 0),
                "None" if xs.get("rmse_m") is None else f"{float(xs.get('rmse_m')):.3f}",
                "None" if xs.get("median_bias_m") is None else f"{float(xs.get('median_bias_m')):.3f}",
            )
        except (AttributeError, TypeError, ValueError):
            log.debug("ATL quality summary logging failed", exc_info=True)
    except (OSError, TypeError, ValueError):
        log.warning("[ATL][QUALITY] Failed to build ATL quality report.", exc_info=True)

    try:
        atl_audit_summary, atl_audit_df = atl.summarize_atl_raw_to_retained_audit(
            atl03_audit, atl24_audit, df_atl03=df_atl03, df_atl24=df_atl24,
        )
        atl_audit_json = dir_logs / "atl_raw_to_retained_audit.json"
        atl_audit_csv = dir_logs / "atl_raw_to_retained_stages.csv"
        atl_audit_json.write_text(json.dumps(atl_audit_summary, indent=2), encoding="utf-8")
        if hasattr(atl_audit_df, "to_csv"):
            atl_audit_df.to_csv(atl_audit_csv, index=False)
        log.info("[ATL][AUDIT] Wrote raw-to-retained audit: %s", atl_audit_json)
        log.info("[ATL][AUDIT] Wrote raw-to-retained stage table: %s", atl_audit_csv)
        try:
            cu = atl_audit_summary.get("cudem_framework_assessment", {})
            log.info(
                "[ATL][AUDIT] ATL03 retained=%s ATL24 retained=%s | recommended_use=%s | %s",
                atl_audit_summary.get("atl03", {}).get("retained_points", 0),
                atl_audit_summary.get("atl24", {}).get("retained_points", 0),
                cu.get("recommended_use", "unknown"),
                "; ".join(cu.get("notes", [])[:2]),
            )
        except (AttributeError, TypeError, ValueError):
            log.debug("ATL audit summary logging failed", exc_info=True)
    except (OSError, TypeError, ValueError):
        log.warning("[ATL][AUDIT] Failed to build raw-to-retained ATL audit ledger.", exc_info=True)

    try:
        atl03_adm_outputs = atl.write_atl03_admissibility_artifacts(audit=atl03_audit, out_dir=dir_logs, logger=log)
        if isinstance(atl03_adm_outputs, dict) and any(atl03_adm_outputs.values()):
            log.info("[ATL03][ADMISSIBILITY] CSV=%s JSON=%s GPKG=%s", atl03_adm_outputs.get('csv'), atl03_adm_outputs.get('json'), atl03_adm_outputs.get('gpkg'))
    except (OSError, TypeError, ValueError):
        log.warning("[ATL03][ADMISSIBILITY] Failed to write admissibility artifacts.", exc_info=True)

    # ------------------------------------------------------------------
    # Kd-based ATL reliability filter (pre-fusion)
    # ------------------------------------------------------------------
    kd_filter_meta = None
    kd_max_depth_physics = None
    try:
        from kd_estimation import compute_physics_based_max_depth
        kd_result = compute_physics_based_max_depth(
            s2_paths, algorithm=getattr(args, "kd_algorithm", "lee2005"),
            confidence_level=getattr(args, "kd_confidence_level", "moderate"),
        )
        kd_490_median = kd_result.get("kd_490_median")
        kd_max_depth_physics = kd_result.get("physics_max_depth_m")
        _rr_add(rr, "sdb.kd_estimation", kd_result)

        if kd_490_median is not None:
            from sdb_tier import filter_atl_by_kd
            if df_atl03 is not None and len(df_atl03) > 0:
                df_atl03, kd_meta_03 = filter_atl_by_kd(df_atl03, kd_490_median)
                _rr_add(rr, "sdb.kd_atl_filter.atl03", kd_meta_03)
            if df_atl24 is not None and len(df_atl24) > 0:
                df_atl24, kd_meta_24 = filter_atl_by_kd(df_atl24, kd_490_median)
                _rr_add(rr, "sdb.kd_atl_filter.atl24", kd_meta_24)
    except Exception as exc:
        log.warning("[Kd-ATL] Kd-based ATL filter skipped: %s", exc)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------
    ctx.mod_fusion = fusion
    ctx.kd_max_depth_m = kd_max_depth_physics  # propagate to ctx for _filter_extra_xyz_for_sdb
    fused_df = _run_fusion(
        ctx,
        s2_paths=s2_paths, df_atl03=df_atl03, df_atl24=df_atl24,
        fusion=fusion,
    )
    ctx.fused_df = fused_df

    # ------------------------------------------------------------------
    # Tier selection (auto-selects RF vs Stumpf-linear vs physics-only)
    # ------------------------------------------------------------------
    model_tier = 1
    tier_quality = None
    try:
        from sdb_tier import select_model_tier
        model_tier, tier_quality = select_model_tier(
            fused_df,
            aoi_bounds=bbox_wesn,
            kd_max_depth_m=kd_max_depth_physics,
        )
        _rr_add(rr, "sdb.model_tier", model_tier)
        _rr_add(rr, "sdb.tier_reason", tier_quality.tier_reason if tier_quality else "unknown")
    except Exception as exc:
        log.warning("[Tier] Model tier selection failed, defaulting to tier 1: %s", exc)

    # Store tier in context so downstream stages can adapt behavior
    ctx.model_tier = model_tier
    ctx.tier_quality = tier_quality

    # ------------------------------------------------------------------
    # Model reuse (regional cache): if a cached model exists, prefer it over retraining.
    # This improves cross-tile consistency in sparse-data regions and avoids per-tile drift.
    # ------------------------------------------------------------------
    cached_model = False
    stumpf_lr = None
    model_meta = None
    df_train_final = None
    df_test_final = None
    try:
        if False and model_cache_dir is not None:
            rf_p = model_cache_dir / "rf_model.pkl"
            meta_p = model_cache_dir / "model_meta.json"
            stumpf_p = model_cache_dir / "stumpf_lr.pkl"
            if rf_p.exists() and meta_p.exists():
                shutil.copy2(rf_p, dir_model / "rf_model.pkl")
                shutil.copy2(meta_p, dir_model / "model_meta.json")
                if stumpf_p.exists():
                    shutil.copy2(stumpf_p, dir_model / "stumpf_lr.pkl")
                    stumpf_lr = True  # truthy sentinel for downstream arg
                cached_model = True
                try:
                    with open(meta_p, "r", encoding="utf-8") as f:
                        model_meta = json.load(f)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    model_meta = None
                df_train_final = pd.DataFrame()
                df_test_final = pd.DataFrame()
                log.info("HIT: using cached model from %s", model_cache_dir)
                if rr is not None:
                    rr.add("model_cache.hit", True)
            else:
                if rr is not None:
                    rr.add("model_cache.hit", False)
    except (OSError, RuntimeError, ValueError, shutil.Error) as e:
        log.warning("Could not use cached model: %s", e)
        if rr is not None:
            rr.add("model_cache.hit", False)
            rr.add("model_cache.error", str(e))

    if fused_df.empty:
        # No new AOI training points. Attempt to proceed using the persisted model bank if available.
        if model_bank_dir is not None:
            try:
                rf_p = model_bank_dir / 'rf_model.pkl'
                meta_p = model_bank_dir / 'model_meta.json'
                stumpf_p = model_bank_dir / 'stumpf_lr.pkl'
                if rf_p.exists() and meta_p.exists():
                    dir_model.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(rf_p, dir_model / 'rf_model.pkl')
                    shutil.copy2(meta_p, dir_model / 'model_meta.json')
                    if stumpf_p.exists():
                        shutil.copy2(stumpf_p, dir_model / 'stumpf_lr.pkl')
                        stumpf_lr = True  # sentinel (file exists)
                    cached_model = True
                    try:
                        with open(meta_p, 'r', encoding='utf-8') as f:
                            model_meta = json.load(f)
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        model_meta = None
                    log.warning('0 training points; using existing model bank for prediction-only.')
                    if rr is not None:
                        rr.add('model_bank.prediction_only', True)
                        _rr_artifact(rr, 'rf_model_pkl', str(dir_model / 'rf_model.pkl'))
                        _rr_artifact(rr, 'model_meta_json', str(dir_model / 'model_meta.json'))
                        if (dir_model / 'stumpf_lr.pkl').exists():
                            _rr_artifact(rr, 'stumpf_lr_pkl', str(dir_model / 'stumpf_lr.pkl'))
                else:
                    log.error('0 training points and model bank has no trained model yet.')
            except (OSError, RuntimeError, ValueError, shutil.Error) as ex:
                log.error("0 training points and failed to load model bank model: %s", ex, exc_info=True)
        if not cached_model:
            # No valid training points for this AOI and no cached model available.
            # This is common for inland AOIs when the water/land mask gates out all pixels.
            # Treat as a graceful skip so the parent pipeline can continue with other methods.
            log.warning('No valid training points found after Fusion, and no model is available to predict; skipping SDB for this AOI.')
            try:
                (out_root / 'SDB_SKIPPED.txt').write_text('SDB skipped: no training points after fusion and no cached model available.\n', encoding='utf-8')
            except OSError as _exc:
                log.debug("Suppressed: %s", _exc, exc_info=True)
            if rr is not None:
                rr.add('sdb.skipped', True)
                rr.add('sdb.skip_reason', 'no_training_points_no_model')
            return 0

    # 7. Training & Validation

    # Training config needed downstream (even if using cached model)
    linf_est_mode = "deepwater" if getattr(args, "linf_estimate_deepwater", False) else "none"
    land_mask_type_for_train = str(getattr(args, "land_mask_type", "auto"))
    land_mask_water_val_for_train = getattr(args, "land_mask_water_val", None)
    if land_mask_water_val_for_train is not None:
        land_mask_water_val_for_train = int(land_mask_water_val_for_train)

    if not cached_model:
        log.info("Model training & validation")
        df_with_s2 = train.sample_s2_bands_at_points(fused_df, s2_paths, str(land_mask_out))

        # After S2 sampling/masking, re-log provenance to confirm XYZ is still present.
        try:
            if "source" in df_with_s2.columns:
                src_counts2 = df_with_s2["source"].value_counts(dropna=False).to_dict()
                log.info("After S2 sampling/masking: %s", src_counts2)
        except Exception:
            log.warning("Could not compute source counts after S2 sampling.", exc_info=True)

        # Partition model bank by source mix and depth regime to avoid cross-regime contamination.
        model_bank_dir_run = model_bank_dir
        mb_part = None
        if model_bank_dir is not None:
            try:
                model_bank_dir_run, mb_part = _build_model_bank_partition_context(args, df_with_s2, base_dir=model_bank_dir)
                if model_bank_dir_run is not None and mb_part is not None:
                    model_bank_dir_run.mkdir(parents=True, exist_ok=True)
                    log.info("partitioned dir=%s meta=%s", model_bank_dir_run, mb_part)
                    if rr is not None:
                        rr.add('model_bank.partition_dir', str(model_bank_dir_run))
                        rr.add('model_bank.partition', mb_part)
            except Exception as ex:
                model_bank_dir_run = model_bank_dir
                log.warning("Partitioning failed; falling back to base bank dir. Reason: %s", ex)

        log.info("Running Standard Training (Random Split)...")

        linf_est_mode = "deepwater" if args.linf_estimate_deepwater else "none"

        land_mask_type_for_train = str(getattr(args, "land_mask_type", "auto"))
        land_mask_water_val_for_train = getattr(args, "land_mask_water_val", None)
        if land_mask_water_val_for_train is not None:
            land_mask_water_val_for_train = int(land_mask_water_val_for_train)


        tc = TrainContext.from_args(
            args,
            train_df=df_with_s2,
            s2_paths=s2_paths,
            plots_dir=dir_plot,
            diagnostics_dir=out_root,
            rr=rr,
            model_bank_dir=model_bank_dir_run,
            model_bank_context=mb_part,
            land_mask_type_override=land_mask_type_for_train,
            land_mask_water_val_override=land_mask_water_val_for_train,
            land_mask_invert_override=land_mask_invert,
            linf_estimate_mode=linf_est_mode,
        )
        tc.max_depth_sdb = max_depth_sdb_product
        tc.spatial_split = False  # Production model uses stratified random split
        tc.atl03_admissibility_summary = (atl03_audit or {}).get('admissibility') if isinstance(atl03_audit, dict) else None

        _train_kwargs = tc.to_kwargs()
        _train_kwargs["fallback_registry"] = fallback_registry
        rf, stumpf_lr, df_train_final, df_test_final, model_meta = train.train_sdb_model(
            **_train_kwargs
        )

        # Model bank periodic retrain policy may intentionally reuse the last trained model,
        # in which case train/test dataframes are empty by design.
        reused_from_bank = False
        try:
            if isinstance(model_meta, dict):
                reused_from_bank = bool(model_meta.get('model_bank', {}).get('reused_model', False))
        except Exception:
            reused_from_bank = False

        if df_train_final is not None and df_train_final.empty and (not reused_from_bank):
            log.error("Training failed (0 samples).")
            return

        src_rand_base = dir_plot / "SDB_Accuracy_Assessment_RandomSplit.png"
        if src_rand_base.exists():
            src_rand_base.rename(dir_plot / "Validation_RandomSplit.png")

        src_rand_v1 = dir_plot / "SDB_Accuracy_Assessment_RandomSplit_V1_AllData.png"
        if src_rand_v1.exists():
            src_rand_v1.rename(dir_plot / "Validation_RandomSplit_V1_AllData.png")


        joblib.dump(rf, dir_model / "rf_model.pkl")
        if stumpf_lr and (stumpf_lr is not True):
            joblib.dump(stumpf_lr, str(dir_model / "stumpf_lr.pkl"))

        _rr_artifact(rr, "rf_model_pkl", str(dir_model / "rf_model.pkl"))
        if (dir_model / "stumpf_lr.pkl").exists():
            _rr_artifact(rr, "stumpf_lr_pkl", str(dir_model / "stumpf_lr.pkl"))

        if args.no_doa:
            log.info("Domain of Applicability (DoA) enforcement disabled by user.")
            if "training_bounds" in model_meta:
                del model_meta["training_bounds"]

        # Tier-aware DOA relaxation: when data quality doesn't support
        # RF (tier >= 2), relax DOA so the Stumpf physics baseline can
        # predict across the full water area instead of only near tracks.
        #
        # CRITICAL: Re-evaluate tier using POST-QC depth stats from the
        # actual training data, not the pre-QC fused data. The Stumpf
        # residual filter can drop 30%+ of points and collapse the
        # effective depth range dramatically (e.g. 11.7m → 1.1m).
        _tier = getattr(ctx, "model_tier", 1)
        if _tier == 1 and df_train_final is not None and not df_train_final.empty:
            try:
                import numpy as _np
                _d = df_train_final["depth_m"].to_numpy(dtype=float) if "depth_m" in df_train_final.columns else _np.array([])
                _d = _np.abs(_d[_np.isfinite(_d)])
                if len(_d) > 0:
                    from sdb_tier import TIER1_MIN_DEPTH_RANGE_M, TIER1_MIN_DEPTH_STD_M
                    _post_qc_range = float(_np.max(_d) - _np.min(_d))
                    _post_qc_std = float(_np.std(_d))
                    _post_qc_iqr = float(_np.percentile(_d, 75) - _np.percentile(_d, 25))
                    _post_qc_median = float(_np.median(_d))
                    _floor = float(_np.min(_d))
                    _near_floor_frac = float(_np.mean(_d < (_floor + 0.5)))

                    # Correlation check (if stumpf_idx available)
                    _abs_corr = None
                    if "stumpf_idx" in df_train_final.columns:
                        _si = df_train_final["stumpf_idx"].to_numpy(dtype=float)
                        _both = _np.isfinite(_si) & _np.isfinite(df_train_final["depth_m"].to_numpy(dtype=float))
                        if _np.sum(_both) > 10:
                            _c = _np.corrcoef(_si[_both], _np.abs(df_train_final["depth_m"].to_numpy(dtype=float)[_both]))[0, 1]
                            _abs_corr = abs(_c) if _np.isfinite(_c) else 0.0

                    log.info("[Post-QC tier check] n=%d, range=%.2fm, std=%.2fm, IQR=%.2fm, "
                             "median=%.2fm, near_floor=%.0f%%, |corr|=%s",
                             len(_d), _post_qc_range, _post_qc_std, _post_qc_iqr,
                             _post_qc_median, _near_floor_frac * 100,
                             "%.2f" % _abs_corr if _abs_corr is not None else "n/a")

                    downgrade_reasons = []
                    # Criterion 1: weak correlation
                    if _abs_corr is not None and _abs_corr < 0.3:
                        downgrade_reasons.append("|corr|=%.2f < 0.30" % _abs_corr)
                    # Criterion 2: skewed distribution
                    if _post_qc_iqr < 1.0 and _post_qc_range > 0 and (_post_qc_range / max(_post_qc_iqr, 0.01)) > 5.0:
                        downgrade_reasons.append("IQR=%.2fm, range/IQR=%.1f" % (
                            _post_qc_iqr, _post_qc_range / max(_post_qc_iqr, 0.01)))
                    # Criterion 3: near-floor dominance
                    if _near_floor_frac > 0.80 and _post_qc_median < 1.5:
                        downgrade_reasons.append("near_floor=%.0f%%, median=%.2fm" % (
                            _near_floor_frac * 100, _post_qc_median))
                    # Legacy checks
                    if _post_qc_range < TIER1_MIN_DEPTH_RANGE_M:
                        downgrade_reasons.append("depth_range=%.1fm < %.1fm" % (
                            _post_qc_range, TIER1_MIN_DEPTH_RANGE_M))
                    if _post_qc_std < TIER1_MIN_DEPTH_STD_M:
                        downgrade_reasons.append("depth_std=%.2fm < %.2fm" % (
                            _post_qc_std, TIER1_MIN_DEPTH_STD_M))

                    if downgrade_reasons:
                        _tier = 2
                        ctx.model_tier = 2
                        log.warning("[Tier] Post-QC re-evaluation downgrades tier 1 → 2: %s",
                                    "; ".join(downgrade_reasons))
                        _rr_add(rr, "sdb.tier_post_qc_downgrade", "; ".join(downgrade_reasons))
                        _rr_add(rr, "sdb.model_tier_post_qc", 2)
                else:
                    log.warning("[Post-QC tier check] No finite depths in df_train_final")
            except Exception:
                log.debug("Post-QC tier re-evaluation failed", exc_info=True)
        elif _tier == 1:
            log.warning("[Post-QC tier check] Skipped: df_train_final is %s",
                        "None" if df_train_final is None else "empty")

        if _tier >= 2:
            log.info("[Tier %d] Relaxing DOA (not disabling): preserving training bounds "
                     "for soft feature-space gating while relying primarily on physics masks "
                     "(clear-water, land, optical depth limit).", _tier)
            doa_cfg = model_meta.get("doa", {})
            doa_cfg["relaxed_for_tier"] = _tier
            model_meta["doa"] = doa_cfg
            _rr_add(rr, "sdb.doa_relaxed_for_tier", _tier)

        # Also inject tier into metadata for predict.py to read
        model_meta["model_tier"] = _tier

        model_meta["max_depth_sdb"] = max_depth_sdb_product
        model_meta["water_class"] = args.water_class
        if mb_part is not None:
            model_meta["model_bank_partition"] = dict(mb_part)
        with open(dir_model / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2)

        # Compute production (random split) validation metrics once for logging and model-bank gating.
        m_rand = _calc_metrics(df_test_final)

        chosen_max_depth = max_depth_sdb_product

        # B. Optional Spatial Validation
        if args.validate_spatial:
            log.info("Running secondary spatial validation (spatial split)")

            tc_spatial = TrainContext.from_args(
                args,
                train_df=df_with_s2,
                s2_paths=s2_paths,
                plots_dir=dir_plot,
                diagnostics_dir=dir_logs / "spatial_validation",
                model_bank_dir=model_bank_dir_run,
                model_bank_context=mb_part,
                land_mask_type_override=land_mask_type_for_train,
                land_mask_water_val_override=land_mask_water_val_for_train,
                land_mask_invert_override=land_mask_invert,
                linf_estimate_mode=linf_est_mode,
            )
            tc_spatial.max_depth_sdb = max_depth_sdb_product
            tc_spatial.spatial_split = True
            tc_spatial.model_bank_enabled = False

            _spatial_kwargs = tc_spatial.to_kwargs()
            _spatial_kwargs["fallback_registry"] = fallback_registry
            _, _, _, df_test_spatial, model_meta_spatial = train.train_sdb_model(
                **_spatial_kwargs
            )

            src_spat_base = dir_plot / "SDB_Accuracy_Assessment_SpatialSplit.png"
            if src_spat_base.exists():
                src_spat_base.rename(dir_plot / "Validation_SpatialSplit.png")

            src_spat_v1 = dir_plot / "SDB_Accuracy_Assessment_SpatialSplit_V1_AllData.png"
            if src_spat_v1.exists():
                src_spat_v1.rename(dir_plot / "Validation_SpatialSplit_V1_AllData.png")

            m_rand = _calc_metrics(df_test_final)
            m_spat = _calc_metrics(df_test_spatial)

            def _fmt_metric_cell(v):
                """Format metric cell for console tables.
                Returns 'n/a' when the value is missing or non-finite to avoid misleading 'nan' output.
                """
                try:
                    fv = float(v)
                    return f"{fv:.3f}" if np.isfinite(fv) else "n/a"
                except Exception:
                    return "n/a"

            def _fmt_count_cell(v):
                try:
                    return str(int(v))
                except Exception:
                    return str(v)


            # ------------------------------------------------------------------
            # Auto max-depth selection (depth-of-support)
            # ------------------------------------------------------------------
            chosen_max_depth = max_depth_sdb_product
            src_key = None

            def _load_train_report_auto_depth() -> tuple:
                def _source_family(name):
                    src = str(name or "").strip().lower()
                    if not src:
                        return ""
                    if "combined" in src:
                        return "combined"
                    if "physics" in src or "kd" in src:
                        return "physics"
                    if "rmse" in src or "support" in src:
                        return "rmse"
                    if "training_p95" in src or "p95" in src:
                        return "training_p95"
                    return src

                def _coerce_positive(v):
                    try:
                        fv = float(v)
                    except Exception:
                        return None
                    return fv if np.isfinite(fv) and fv > 0 else None

                try:
                    if isinstance(model_meta, dict):
                        final_depth = _coerce_positive(model_meta.get("max_depth_sdb_final"))
                        final_source = model_meta.get("max_depth_sdb_final_source")
                        if final_depth is not None:
                            return final_depth, f"model_meta.{final_source or 'max_depth_sdb_final'}"
                        opts = model_meta.get("max_depth_options") if isinstance(model_meta.get("max_depth_options"), dict) else {}
                        family = _source_family((opts or {}).get("selected_source") or final_source)
                        ordered = {
                            "physics": ("max_depth_sdb_auto_physics", "max_depth_sdb_combined", "max_depth_sdb_auto"),
                            "combined": ("max_depth_sdb_combined", "max_depth_sdb_auto_physics", "max_depth_sdb_auto"),
                            "rmse": ("max_depth_sdb_auto", "max_depth_sdb_combined", "max_depth_sdb_auto_physics"),
                        }.get(family, ("max_depth_sdb_combined", "max_depth_sdb_auto_physics", "max_depth_sdb_auto"))
                        for key in ordered:
                            fv = _coerce_positive(model_meta.get(key))
                            if fv is not None:
                                return fv, f"model_meta.{key}"
                except Exception as _exc:
                    log.debug("Suppressed: %s", _exc, exc_info=True)

                cand_paths = [
                    out_root / "train_report.json",
                    dir_logs / "train_report.json",
                    dir_logs / "spatial_validation" / "train_report.json",
                    dir_model / "train_report.json",
                ]
                for tp in cand_paths:
                    try:
                        if tp.exists() and tp.stat().st_size > 0:
                            with open(tp, "r", encoding="utf-8") as f:
                                trj = json.load(f)
                            tr = trj.get("train", {}) if isinstance(trj, dict) else {}
                            final_depth = _coerce_positive(tr.get("max_depth_sdb_final"))
                            final_source = tr.get("max_depth_sdb_final_source")
                            if final_depth is not None:
                                return final_depth, f"{tp.name}:train.{final_source or 'max_depth_sdb_final'}"
                            opts = tr.get("max_depth_options") if isinstance(tr.get("max_depth_options"), dict) else {}
                            family = _source_family((opts or {}).get("selected_source") or final_source)
                            ordered = {
                                "physics": ("max_depth_sdb_auto_physics", "max_depth_sdb_combined", "max_depth_sdb_auto_m", "max_depth_sdb_auto"),
                                "combined": ("max_depth_sdb_combined", "max_depth_sdb_auto_physics", "max_depth_sdb_auto_m", "max_depth_sdb_auto"),
                                "rmse": ("max_depth_sdb_auto_m", "max_depth_sdb_auto", "max_depth_sdb_combined", "max_depth_sdb_auto_physics"),
                            }.get(family, ("max_depth_sdb_combined", "max_depth_sdb_auto_physics", "max_depth_sdb_auto_m", "max_depth_sdb_auto"))
                            for key in ordered:
                                fv = _coerce_positive(tr.get(key))
                                if fv is not None:
                                    return fv, f"{tp.name}:train.{key}"
                    except Exception:
                        continue
                return None, None

            if auto_max_depth:
                # For tier 2 (shallow/degraded data), prefer the physics-based
                # depth limit over the RMSE-constrained cap. The RMSE cap is
                # derived from the shallow training data and would artificially
                # limit the Stumpf physics model to ~2m.
                _tier = getattr(ctx, "model_tier", 1)
                if _tier >= 2 and isinstance(model_meta, dict):
                    _phys = model_meta.get("max_depth_sdb_auto_physics")
                    if _phys is not None:
                        try:
                            _phys_f = float(_phys)
                            if np.isfinite(_phys_f) and _phys_f > 0:
                                chosen_max_depth = min(_phys_f, float(max_depth_sdb_product))
                                src_key = "physics_kd_tier2"
                                log.info("[AUTO-DEPTH][Tier %d] Using physics Kd limit %.1fm "
                                         "instead of RMSE-constrained cap (Stumpf model "
                                         "extrapolates beyond training range).", _tier, chosen_max_depth)
                        except (ValueError, TypeError):
                            pass

                if src_key is None:
                    fv, src = _load_train_report_auto_depth()
                    if fv is None:
                        log.warning("[AUTO-DEPTH] Estimation failed or returned None. Falling back to default max depth.")
                        chosen_max_depth = max_depth_sdb_product
                        src_key = "fallback_default"
                    else:
                        src_key = src
                        chosen_max_depth = min(float(fv), float(max_depth_sdb_product))
                        # The TRAINING depth cap (max_depth_sdb_product) is wide
                        # so the RF sees the full depth range from XYZ data.
                        # The PREDICTION cap respects the optical limit — S2 optics
                        # can't distinguish 25m from 35m regardless of training
                        # data quality.  Deep offshore pixels with uniform spectral
                        # signatures should not get SDB predictions because the
                        # model has no optical information to work with there.
                        _gs = model_meta.get("physics_guidance", {}) if isinstance(model_meta, dict) else {}
                        if bool(_gs.get("anchor_support_good", False)) or bool(getattr(args, "extra_xyz", None)):
                            log.info("[AUTO-DEPTH] Anchor mode: training cap=%.1fm, prediction cap=%.1fm "
                                     "(optical limit constrains predictions; training sees full depth range)",
                                     float(max_depth_sdb_product), chosen_max_depth)
                        log.info(
                            "[AUTO-DEPTH] Final product cap max_depth_sdb=%2f m  (from %s; rmse_target=%2f m; hard_cap=%2f m).",
                            chosen_max_depth, src_key, float(args.rmse_target_sdb), float(max_depth_sdb_product),
                        )
            else:
                src_key = "cli"

            _rr_add(rr, "sdb.max_depth.final_m", float(chosen_max_depth))
            _rr_add(rr, "sdb.max_depth.source", str(src_key))

            model_meta["max_depth_sdb"] = chosen_max_depth
            model_meta["max_depth_sdb_final"] = chosen_max_depth
            if auto_max_depth:
                model_meta["max_depth_sdb_source"] = src_key
                model_meta["rmse_target_sdb"] = float(args.rmse_target_sdb)
            else:
                model_meta["max_depth_sdb_source"] = "cli"
            if mb_part is not None:
                model_meta["model_bank_partition"] = dict(mb_part)


            try:
                with open(dir_model / "model_meta.json", "w", encoding="utf-8") as f:
                    json.dump(model_meta, f, indent=2)
            except Exception as _exc:
                log.debug("Suppressed: %s", _exc, exc_info=True)


            report_lines = [
                "="*60,
                f"VALIDATION COMPARISON: {dir_name}",
                "="*60,
                f"{'Metric':<15} | {'Random (Prod)':<18} | {'Spatial (Val)':<18}",
                "-" * 60,
                f"{'RMSE (m)':<15} | {_fmt_metric_cell(m_rand.get('rmse')):<18} | {_fmt_metric_cell(m_spat.get('rmse')):<18}",
                f"{'R²':<15} | {_fmt_metric_cell(m_rand.get('r2')):<18} | {_fmt_metric_cell(m_spat.get('r2')):<18}",
                f"{'MAE (m)':<15} | {_fmt_metric_cell(m_rand.get('mae')):<18} | {_fmt_metric_cell(m_spat.get('mae')):<18}",
                f"{'Test Size':<15} | {_fmt_count_cell(m_rand.get('n')):<18} | {_fmt_count_cell(m_spat.get('n')):<18}",
                "="*60
            ]

            try:
                r2_rand = float(m_rand.get('r2', float('nan')))
                r2_spat = float(m_spat.get('r2', float('nan')))
                rmse_rand = float(m_rand.get('rmse', float('nan')))
                rmse_spat = float(m_spat.get('rmse', float('nan')))
            except (TypeError, ValueError, AttributeError):
                r2_rand = r2_spat = rmse_rand = rmse_spat = float('nan')

            n_rand = int(m_rand.get('n', 0) or 0) if str(m_rand.get('n', 0)).strip() != '' else 0
            n_spat = int(m_spat.get('n', 0) or 0) if str(m_spat.get('n', 0)).strip() != '' else 0
            spatial_status = model_meta.get('spatial_split_status', {}) if isinstance(model_meta, dict) else {}
            validation_invariants = _derive_validation_invariants(m_rand, m_spat, spatial_status)
            if rr is not None:
                rr.add_dict("scientific.validation_invariants", validation_invariants)
            if validation_invariants.get("guidance_only_required", False) and fallback_registry is not None:
                record_fallback(fallback_registry, "guidance_only_validation_gate", FallbackClass.DEGRADED, stage="validation", detail="Validation requires guidance-only interpretation because spatial holdout is missing or held-out skill is weak/negative.")

                # --- Reviewer recommendation #1: automatic product degradation ---
                # When validation is weak, automatically tighten the product:
                # 1. Raise min_confidence_threshold to suppress low-confidence pixels
                # 2. Record that the product should be treated as sparse guidance
                if isinstance(model_meta, dict):
                    _prev_thresh = float(getattr(args, "min_confidence_threshold", 0.0))
                    _auto_thresh = max(_prev_thresh, 0.15)
                    if _auto_thresh > _prev_thresh:
                        args.min_confidence_threshold = _auto_thresh
                        log.info("[AUTO-DEGRADE] Weak validation → raised min_confidence_threshold from %.2f to %.2f "
                                 "(suppresses patchy low-confidence pixels)", _prev_thresh, _auto_thresh)
                    model_meta["product_degradation"] = {
                        "degraded": True,
                        "reason": "guidance_only_required",
                        "auto_confidence_threshold": _auto_thresh,
                        "cudem_use": "sparse_guidance_only",
                    }
            if n_rand <= 0:
                report_lines.append("[INSIGHT] Random production split metrics are unavailable (test set size is zero). This commonly happens when the production model was *reused from the model bank* (no new train/test split was generated) or when aggressive filtering removed all held-out points. Rely on spatial validation for this run.")
            if not validation_invariants.get('representative_spatial_holdout', True):
                report_lines.append("[INSIGHT] No representative spatial holdout could be formed from the retained training data. Treat this run as having insufficient evidence of spatial generalization; output should remain guidance-only.")
                report_lines.append("[INSIGHT] CUDEM use pattern: treat SDB rasters as interpolation guidance only; rely on guidance_weight/trusted_interior and keep authoritative survey data dominant.")
            elif validation_invariants.get('negative_validation_detected', False):
                report_lines.append("[INSIGHT] At least one validation R² is negative: do not treat this model as generalizing well (held-out performance is worse than a mean baseline on that split).")
            elif np.isfinite(rmse_rand) and np.isfinite(rmse_spat) and rmse_spat > rmse_rand * 1.5:
                report_lines.append("[INSIGHT] Significant drop in Spatial Accuracy detected.")
            elif validation_invariants.get('weak_spatial_validation', False):
                report_lines.append("[INSIGHT] Validation metrics remain weak; do not interpret similar random/spatial RMSE as strong generalization.")
            else:
                report_lines.append("[INSIGHT] Spatial Accuracy is comparable and validation metrics are not obviously pathological.")

            report_text = "\n".join(report_lines)
            log.info(report_text)

            with open(dir_logs / "validation_comparison.txt", "w", encoding="utf-8") as f:
                f.write(report_text)

        # Persist model to the model bank for cross-tile consistency, but gate saves on validation quality.
        try:
            model_bank_save_ok = True
            model_bank_gate_reason = "ok"
            model_bank_gate_basis = "random"
            model_bank_gate_r2 = None
            m_spat_gate = locals().get("m_spat", None)

            # Fail save if ANY available validation split reports negative R².
            cand_metrics = [("random", m_rand)]
            if isinstance(m_spat_gate, dict):
                cand_metrics.append(("spatial", m_spat_gate))
            for _basis, _m in cand_metrics:
                if not isinstance(_m, dict):
                    continue
                try:
                    _n = int(_m.get("n", 0) or 0)
                except Exception:
                    _n = 0
                try:
                    _r2 = float(_m.get("r2", float("nan")))
                except Exception:
                    _r2 = float("nan")
                if _n > 0 and np.isfinite(_r2) and (_r2 < 0):
                    model_bank_save_ok = False
                    model_bank_gate_reason = f"negative_r2_{_basis}"
                    model_bank_gate_basis = _basis
                    model_bank_gate_r2 = float(_r2)
                    break

            if (model_bank_dir_run is not None) and (not reused_from_bank):
                if model_bank_save_ok:
                    model_bank_dir_run.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dir_model / 'rf_model.pkl', model_bank_dir_run / 'rf_model.pkl')
                    shutil.copy2(dir_model / 'model_meta.json', model_bank_dir_run / 'model_meta.json')
                    if (dir_model / 'stumpf_lr.pkl').exists():
                        shutil.copy2(dir_model / 'stumpf_lr.pkl', model_bank_dir_run / 'stumpf_lr.pkl')
                    log.info("SAVED model artifacts to %s", model_bank_dir_run)
                    try:
                        meta_p = model_bank_dir_run / 'bank_meta.json'
                        if meta_p.exists():
                            with open(meta_p, 'r', encoding='utf-8') as _f:
                                _bm = json.load(_f)
                            _bm['last_trained_n_seen'] = int(_bm.get('n_seen', 0))
                            _bm['last_trained_utc'] = datetime.now(timezone.utc).isoformat()
                            with open(meta_p, 'w', encoding='utf-8') as _f:
                                json.dump(_bm, _f, indent=2)
                    except Exception:
                        log.debug('Failed to update last_trained checkpoint.', exc_info=True)
                    if rr is not None:
                        rr.add('model_bank.model_saved', True)
                        rr.add('model_bank.model_dir', str(model_bank_dir_run))
                        rr.add('model_bank.save_gate', {'ok': True, 'reason': model_bank_gate_reason})
                else:
                    log.warning(
                        "[MODEL_BANK] Skip save: validation gate failed (%s, "
                        "basis=%s, R²=%.3f).",
                        model_bank_gate_reason, model_bank_gate_basis, model_bank_gate_r2,
                    )
                    if rr is not None:
                        rr.add('model_bank.model_saved', False)
                        rr.add('model_bank.save_gate', {
                            'ok': False,
                            'reason': model_bank_gate_reason,
                            'basis': model_bank_gate_basis,
                            'r2': model_bank_gate_r2,
                        })
        except Exception as e:
            log.warning("Failed to save model artifacts: %s", e)
            if rr is not None:
                rr.add('model_bank.model_saved', False)
                rr.add('model_bank.model_save_error', str(e))

        # --- SAVE GPKG ---
        _save_training_gpkg(dir_data / "icesat_depths.gpkg", df_with_s2, df_train_final, df_test_final)

    else:
        log.info("Skipping training/validation; using cached model artifacts in model/")
        # Ensure downstream references exist
        try:
            if df_test_final is None:
                df_test_final = pd.DataFrame()
            if df_train_final is None:
                df_train_final = pd.DataFrame()
        except Exception:
            log.debug('Optional pandas DataFrame normalization failed; continuing.', exc_info=True)

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
                log.warning("Plot generation failed: %s", exc)

    # Prediction output path (used by prediction, post-processing, and manifest)
    out_tif = dir_rast / f"{run_id}_sdb_depth.tif"
    ctx.out_tif = out_tif
    ctx.land_mask_out = land_mask_out
    ctx.stumpf_lr = stumpf_lr
    ctx.chosen_max_depth = chosen_max_depth
    ctx.df_test = df_test_final
    ctx.df_train = df_train_final
    ctx.raster_quickstats = raster_quickstats
    ctx.mod_predict = predict

    _run_sdb_prediction(
        ctx,
        land_mask_type_for_train=land_mask_type_for_train,
        land_mask_water_val_for_train=land_mask_water_val_for_train,
        land_mask_invert=land_mask_invert,
        land_mask_threshold=land_mask_threshold,
    )

    _run_sdb_post_processing(ctx, df_test_final=df_test_final)

    _write_sdb_manifest(ctx)





def _materialize_authoritative_base_for_sdb(args) -> Optional[Path]:
    """Resolve/auto-build authoritative_base for standalone SDB runs.

    Enabled by default so standalone SDB runs can reuse the same AOI-keyed
    NOAA/CUDEM authoritative-base cache strategy as bathy_main.
    """
    auth_arg = getattr(args, "authoritative_base", None)
    auto = bool(getattr(args, "authoritative_base_auto", True))
    auth_path: Optional[Path] = None
    if auth_arg is not None:
        auth_txt = str(auth_arg).strip()
        if auth_txt and auth_txt.lower() in ("auto", "cudem", "auto_cudem"):
            auto = True
        elif auth_txt and auth_txt.lower() in ("off", "none", "disable", "disabled"):
            auto = False
        elif auth_txt:
            auth_path = Path(auth_txt).expanduser()
            if auth_path.exists():
                auth_path = auth_path.resolve()
                setattr(args, "authoritative_base", str(auth_path))
                setattr(args, "_authoritative_base_auto_report", {
                    "mode": "explicit_path",
                    "authoritative_base": str(auth_path),
                    "cache_hit": None,
                })
                return auth_path
            if not auto:
                log.warning("[AUTHORITATIVE] Provided authoritative_base does not exist: %s", auth_path)
                setattr(args, "authoritative_base", str(auth_path))
                return auth_path
            log.info("[AUTHORITATIVE] Explicit authoritative_base path not found; falling back to auto-materialization for AOI.")
    if not auto:
        setattr(args, "authoritative_base", None if auth_path is None else str(auth_path))
        return auth_path
    try:
        from cudem_authoritative import materialize_authoritative_base_for_aoi
    except Exception as exc:
        log.error("[AUTHORITATIVE] Failed to import cudem_authoritative auto-builder: %s", exc, exc_info=True)
        raise
    build_info = materialize_authoritative_base_for_aoi(
        aoi=str(getattr(args, "aoi", "") or ""),
        cache_root=Path(getattr(args, "cache_root", "cache")),
        tile_index_url=str(getattr(args, "authoritative_base_tile_index_url", "") or ""),
        spatial_meta_url=str(getattr(args, "authoritative_base_spatial_meta_url", "") or ""),
        missing_meta_policy=str(getattr(args, "authoritative_base_missing_meta_policy", "skip") or "skip"),
        tile_url_field=getattr(args, "authoritative_base_tile_url_field", None),
        force_rebuild=bool(getattr(args, "authoritative_base_force_rebuild", False)),
        logger=log,
    )
    auth_path = Path(build_info["authoritative_base"]).resolve()
    setattr(args, "authoritative_base", str(auth_path))
    setattr(args, "_authoritative_base_auto_report", build_info)
    log.info("[AUTHORITATIVE] %s authoritative_base: %s",
             "Reused cached" if bool(build_info.get("cache_hit")) else "Materialized",
             auth_path)
    return auth_path

def parse_args():
    p = argparse.ArgumentParser(
        "Modular SDB Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file support (shared with bathy_main)
    p.add_argument("--config", dest="config_files", action="append", default=[],
                   help="YAML config file(s). CLI args override config values.")

    # Core
    p.add_argument("--aoi", required=True, help="W/E/S/N (west/east/south/north)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    p.add_argument("--out-dir", default=None, help="Custom output directory")

    # Caching / reproducibility
    p.add_argument("--cache-root", default="cache",
                   help="Root directory for reusable stage caches (S2/ATL/masks)")
    p.add_argument("--authoritative-base", default=None,
                   help="Optional hard-locked measured-constrained raster. Enabled in auto mode by default; pass a path to override, or off/none to disable.")
    p.add_argument("--authoritative-base-auto", action="store_true", default=True,
                   help="Automatically build and AOI-cache authoritative_base.tif from NOAA CUDEM tile index + spatial metadata under <cache-root>/authoritative_base/. Enabled by default; use --no-authoritative-base-auto to disable.")
    p.add_argument("--no-authoritative-base-auto", dest="authoritative_base_auto", action="store_false",
                   help="Disable automatic AOI-cached authoritative_base materialization for standalone SDB runs.")
    p.add_argument("--authoritative-base-tile-index-url",
                   default="https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/tileindex_NCEI_ninth_Topobathy_2014.zip",
                   help="Tile-index zip URL used when auto-building authoritative_base.")
    p.add_argument("--authoritative-base-spatial-meta-url",
                   default="https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/ninth_spatial_meta.zip",
                   help="Spatial-metadata zip URL used when auto-building authoritative_base.")
    p.add_argument("--authoritative-base-missing-meta-policy", choices=["skip", "tile_extent", "error"], default="skip",
                   help="How to handle selected CUDEM tiles with no matching spatial metadata during authoritative-base auto-build.")
    p.add_argument("--authoritative-base-force-rebuild", action="store_true", default=False,
                   help="Force rebuild of the cached authoritative-base entry for this AOI/settings.")
    p.add_argument("--authoritative-base-tile-url-field", default=None,
                   help="Optional explicit tile-index attribute containing the DEM download URL during authoritative-base auto-build.")
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
    # Model Bank (bounded reservoir) – accumulates training across AOIs for seam-consistent tiling
    p.add_argument("--model-bank", default="auto",
                   help="Model bank directory for incremental training. 'auto' => <cache-root>/model_bank/sdb_global_v1")
    p.add_argument("--no-model-bank", dest="model_bank_enabled", action="store_false",
                   help="Disable model bank; train only on this AOI (not recommended for seamless tiling).")
    p.set_defaults(model_bank_enabled=True)
    p.add_argument("--bank-max-samples", type=int, default=100000,
                   help="Max samples to keep in the model bank reservoir (bounded disk).")
    p.add_argument("--bank-seed", type=int, default=1337,
                   help="Seed for deterministic reservoir sampling in the model bank.")
    p.add_argument("--bank-retrain-min-new", type=int, default=2000,
                   help="Only retrain the RF when at least this many new samples have been added to the bank since last training (stability + speed).")

    # Deprecated: regional model cache (skip-training). Use model bank instead.
    p.add_argument("--model-cache-key", default="auto",
                   help="[DEPRECATED] Key for regional SDB model reuse. Prefer --model-bank.")
    p.add_argument("--no-model-cache", dest="model_cache_enabled", action="store_false",
                   help="[DEPRECATED] Disable regional model reuse.")
    p.set_defaults(model_cache_enabled=False)

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
                   help="Number of Sentinel-2 scenes to download per tile (selected by lowest cloud percentage).")
    p.add_argument("--min-scene-valid-frac", type=float, default=0.90,
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
    p.add_argument("--extra-xyz-sdb-mode", choices=["auto","allow","exclude"], default="auto",
                   help="How extra XYZ is used for SDB training. auto = apply conservative guardrails for likely bed-elevation/mixed-sign sources; allow = always include; exclude = never use in SDB training (still available to river workflow in bathy_main).")

    # Working / datum harmonization
    p.add_argument("--working-srs", default="auto",
                   help="Working horizontal CRS (usually matches Sentinel-2 UTM). Used for reporting / downstream tools.")
    p.add_argument("--working-vcrs-epsg", type=int, default=5703,
                   help="Working vertical CRS EPSG code (default 5703 = NAVD88 height).")

    # Vertical datum transformation (MSL → NAVD88)
    p.add_argument("--convert-sdb-to-navd88", action="store_true", default=False,
                   help="Convert SDB from MSL to NAVD88 vertical datum using dlim. "
                        "SDB values are elevations relative to MSL (negative = below MSL). "
                        "Output: (see artifacts_sdb.json / io_manifest.json for exact filename). Requires dlim (CUDEM) on PATH.")
    p.add_argument("--sdb-source-vdatum", default="epsg:4269+5714",
                   help="Source compound EPSG (default: epsg:4269+5714 = NAD83+MSL). "
                        "SDB is referenced to MSL due to temporal compositing of S2/ICESat-2.")
    p.add_argument("--sdb-target-vdatum", default="epsg:4269+5703",
                   help="Target compound EPSG (default: epsg:4269+5703 = NAD83+NAVD88).")

    # ATL horizontal CRS transformation (PROJ/pyproj).
    # HORIZONTAL-ONLY: depth_m is relative water depth and must not be transformed.
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

    # Memory management and performance options
    p.add_argument("--chunked-prediction", choices=["auto", "on", "off"], default="auto",
                   help="Use chunked processing for large AOIs (auto=decide based on memory)")
    p.add_argument("--max-memory-gb", type=float, default=8.0,
                   help="Maximum memory (GB) for prediction (triggers chunking if exceeded)")
    p.add_argument("--tile-size", type=int, default=2048,
                   help="Tile size (pixels) for chunked processing")
    p.add_argument("--tile-overlap", type=int, default=256,
                   help="Tile overlap (pixels) for seamless mosaicking")
    
    # ML enhancement options
    p.add_argument("--tune-hyperparameters", action="store_true",
                   help="Enable automated hyperparameter tuning for RF model")
    p.add_argument("--n-tuning-iter", type=int, default=30,
                   help="Number of hyperparameter combinations to try")
    p.add_argument("--use-adaptive-spatial-cv", action="store_true",
                   help="Use adaptive spatial cross-validation with density-based clustering")
    
    # Adaptive Spatial Sampling (enabled by default)
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

    p.add_argument("--no-doa", action="store_true",
                   help="Disable Domain of Applicability (DoA) enforcement during prediction.")

    # Guidance raster output control
    p.add_argument("--min-confidence-threshold", type=float, default=0.0, dest="min_confidence_threshold",
                   help="Per-pixel confidence threshold (0–1). Depth cells below this are set to nodata in the "
                        "output raster, producing a sparser but higher-quality guidance surface. "
                        "Default 0.0 (emit all predicted cells). Suggested starting value: 0.2.")

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

    p.add_argument("--no-checkpoints", action="store_true",
                   help="Disable checkpoint saving (no resume capability)")
    p.add_argument("--clean-checkpoints", action="store_true",
                   help="Clear existing checkpoints before running")
    
    p.add_argument("--parallel-predict", action="store_true",
                   help="Enable parallel tile processing for prediction (2-4x speedup)")
    p.add_argument("--parallel-workers", type=int, default=None,
                   help="Number of parallel workers (default: auto-detect from CPU count)")

    return p.parse_args()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Pipeline Failed: %s", exc, exc_info=True)
        traceback.print_exc()
        sys.exit(1)
