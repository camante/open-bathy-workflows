#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bathy_main.py – Unified Coastal + River Bathymetry Pipeline

Orchestrates:
- SDB pipeline (sdb_main.py)
- River cross-section interpolation (river_network.py -> xs_builder.py -> xs_infer_bathy_raster.py)

Key behavior:
- Streams subprocess stdout live (SDB logs are on stdout)
- Captures stderr tails for debugging
- Robustly discovers SDB depth raster output
- Uses correct river_network.py CLI flags (--out-gpkg)
"""


# IMPORTANT: Configure logging FIRST, before importing other pipeline modules
import os as _os
# Force a headless-safe Matplotlib backend early (prevents TkAgg/Tkinter crashes under multiprocessing)
_os.environ.setdefault('MPLBACKEND', 'Agg')

import logging
import sys

# Set up centralized logging before any other imports
try:
    from logging_config import setup_logging
    setup_logging()
except ImportError:
    # Fallback if logging_config.py is not present
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

import argparse
import hashlib
import math
import json
import os
import re
import shlex
import shutil
from vdatum_utils import convert_sdb_msl_to_navd88
from process_utils import run_cmd

# Phase-1 anti-soup refactor: shared helpers
from core.paths import ensure_dir
from core.exec import run_command, run_command_stdout_to_file
from core.fingerprint import hash_key as _hash_key
from core.fingerprint import sha1_text as _stable_hash_str

# Defensive fallback: if core.fingerprint import is unavailable for any reason,
# provide a local stable hash implementation (prevents domain-inference NameError).
if "_stable_hash_str" not in globals():
    def _stable_hash_str(text: str, n: int = 12) -> str:
        try:
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            return h[:n]
        except Exception:
            return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]

from core.cmd import build_cmd as _build_cmd
from core.json_io import write_json
from pipeline.aoi import (
    parse_aoi_bbox as _parse_aoi_bbox,
    bbox_to_aoi_str as _bbox_to_aoi_str,
    aoi_centroid as _aoi_centroid,
    expand_bbox_km as _expand_bbox_km,
    expand_bbox_frac as _expand_bbox_frac,
    utm_epsg_from_lonlat as _utm_epsg_from_lonlat,
    aoi_center_lonlat as _aoi_center_lonlat,
    buffer_aoi as _buffer_aoi,
)

# Phase-2B anti-soup refactor: raster operations
from geo.raster_ops import (
    _clip_raster_to_mask,
    _clip_raster_to_mask_reproject,
    _gaussian_smooth_masked,
    _apply_tile_edge_taper_epsg4269,
    _compute_edge_band_metrics_epsg4269,
    _clip_raster_to_bbox,
    _crop_raster_extent_to_bbox,
    _mask_raster_to_waffles,
    _mask_raster_to_nhdarea,
    apply_depth_metadata,
    apply_elevation_metadata,
    compute_depth_from_bed_and_dem,
    warp_raster_to_srs,
)

# Central constants (versioning, nodata)
import constants
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _apply_final_domain_policy(cfg: "BathyConfig", out_dir: Path, derived_cache_root: Path) -> None:
    """Apply final domain clipping rules:

    Invariants:
      1) If WAFFLES masks were computed for this run, *all deliverable rasters* must be nodata
         outside the selected WAFFLES water mask (water=0, land=1).
      2) River deliverables must additionally be nodata outside the river channel mask (inside=1).

    Policy:
      - If both SDB + river are on: clip combined + SDB + river deliverables to WAFFLES coastline WITH NHD.
      - If only SDB is on: clip combined + SDB deliverables to WAFFLES ocean-only (fallback to WITH NHD).
      - If only river is on: clip river deliverables to WAFFLES WITH NHD (fallback to ocean-only), and also
        clip to river channel mask.

    This runs at the very end as a last-resort safety net. It should *never* be a substitute for correct
    upstream masking; if required masks are missing, we fail closed.
    """
    methods = [m.strip().lower() for m in (getattr(cfg, "methods", []) or [])]
    sdb_on = "sdb" in methods
    river_on = "river" in methods

    # Collect deliverable rasters explicitly recorded in run reports (no filename assumptions).
    def _collect_recorded_rasters(_out_dir: Path) -> List[Path]:
        candidates: List[Path] = []
        for rep_name in ("unified_bathy_report.json", "bathy_report.json"): 
            rp = _out_dir / rep_name
            if not rp.exists():
                continue
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:
                continue

            def _walk(obj: Any) -> None:
                if isinstance(obj, dict):
                    for v in obj.values():
                        _walk(v)
                elif isinstance(obj, list):
                    for v in obj:
                        _walk(v)
                elif isinstance(obj, str):
                    s = obj.strip()
                    if not s:
                        return
                    if s.lower().endswith((".tif", ".tiff")):
                        p = Path(s)
                        if not p.is_absolute():
                            p = _out_dir / p
                        if p.exists():
                            candidates.append(p.resolve())

            # Prefer explicit outputs blocks if present, but fall back to walking the full report.
            if isinstance(data, dict) and isinstance(data.get("outputs"), dict):
                _walk(data.get("outputs"))
            else:
                _walk(data)

        # De-duplicate while preserving order
        seen: set[str] = set()
        out: List[Path] = []
        for p in candidates:
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            out.append(p)
        return out

    deliverable_rasters = _collect_recorded_rasters(out_dir)
    river_rasters = [p for p in deliverable_rasters if ("river" in p.parts)]
    def _clip(path: Path, mask: Path, *, inside_value: int, nodata: float) -> None:
        if path.exists() and mask.exists():
            _clip_raster_to_mask_reproject(path, mask, inside_value=inside_value, invert=False, nodata=nodata)

    # Prefer run-scoped WAFFLES masks captured in cfg by domain inference (avoid stale discovery).
    wm_ocean = getattr(cfg, "waffles_ocean_mask", None)
    wm_nhd = getattr(cfg, "waffles_with_nhd_mask", None)
    wm_ocean = Path(wm_ocean) if wm_ocean else None
    wm_nhd = Path(wm_nhd) if wm_nhd else None

    # No guessing: WAFFLES masks must be provided by domain inference (shared cache staged into derived_cache).
    if wm_ocean is not None and (not wm_ocean.exists()):
        wm_ocean = None
    if wm_nhd is not None and (not wm_nhd.exists()):
        wm_nhd = None
        wm_nhd = None

    if sdb_on and river_on:
        wm = wm_nhd
        if not (wm and wm.exists()):
            raise RuntimeError("Final domain policy requires WAFFLES with-NHD mask, but it was not found.")
        for pth in deliverable_rasters:
            _clip(pth, wm, inside_value=0, nodata=float(getattr(cfg, "final_nodata", -9999.0)))
    elif sdb_on and (not river_on):
        wm = wm_ocean if (wm_ocean and wm_ocean.exists()) else wm_nhd
        if not (wm and wm.exists()):
            raise RuntimeError("Final domain policy requires a WAFFLES water mask (ocean-only or with-NHD), but none were found.")
        for pth in deliverable_rasters:
            _clip(pth, wm, inside_value=0, nodata=float(getattr(cfg, "final_nodata", -9999.0)))
    elif river_on and (not sdb_on):
        wm = wm_nhd if (wm_nhd and wm_nhd.exists()) else wm_ocean
        if not (wm and wm.exists()):
            raise RuntimeError("Final domain policy requires a WAFFLES water mask (with-NHD preferred), but none were found.")
        for pth in deliverable_rasters:
            _clip(pth, wm, inside_value=0, nodata=float(getattr(cfg, "final_nodata", -9999.0)))

        # Additionally restrict river deliverables to the river channel mask (inside=1).
        ch = getattr(cfg, "river_channel_mask", None)
        ch = Path(ch) if ch else None
        if ch is not None and ch.exists():
            for pth in river_rasters:
                _clip(pth, ch, inside_value=1, nodata=float(getattr(cfg, "final_nodata", -9999.0)))
    else:
        for src in to_handle:
            try:
                src.unlink(missing_ok=True)
            except Exception:
                logging.getLogger(__name__).debug("Failed to delete %s", src, exc_info=True)

    # Remove empty directories (deepest-first), leaving out_dir and debug_dir
    for d in sorted([p for p in out_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        if d == out_dir:
            continue
        if d == debug_dir:
            continue
        if debug_dir.exists() and debug_dir in d.parents:
            continue
        try:
            next(d.iterdir())
        except StopIteration:
            try:
                d.rmdir()
            except Exception:
                pass




def _ensure_output_contract(out_dir: Path, cache_root: Path) -> None:
    """Deprecated: output contracts should be derived from explicit manifests.

    Older versions created symlinks/copies to canonical filenames (e.g.
    *_final_epsg4269.tif). That violates the "no filename guessing" invariant.
    The pipeline now relies on io_manifest.json and report-recorded outputs.

    This function is intentionally a no-op and kept only for backwards
    compatibility with older call sites.
    """
    _ = out_dir
    _ = cache_root
    return


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

log = logging.getLogger("bathy_main")

def _fingerprint_path(p: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Cheap fingerprint for cache invalidation (no content hashing)."""
    if p is None:
        return None
    try:
        st = p.stat()
        return {"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except FileNotFoundError:
        return {"path": str(p), "missing": True}


def _fingerprint_script(name: str) -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    fp = _fingerprint_path(here / name)
    return fp or {"path": str(here / name), "missing": True}


def _river_cache_key_and_manifest(cfg: "BathyConfig") -> tuple[str, Dict[str, Any]]:
    """Stable cache key for river interpolation products.

    NOTE: River products are highly sensitive to priors and domain masks. Keep this manifest
    reasonably complete so cached outputs are not silently reused under different settings.
    """
    soundings: List[Path] = []
    if cfg.river_soundings:
        for part in str(cfg.river_soundings).split(","):
            part = part.strip()
            if part:
                soundings.append(Path(part))

    manifest = {
        "aoi": cfg.aoi,
        "river_method": getattr(cfg, "river_method", "xs"),
        "river_dem": _fingerprint_path(cfg.river_dem),
        "soundings": [_fingerprint_path(p) for p in soundings],
        "river_soundings_mode": getattr(cfg, "river_soundings_mode", "auto"),
        "river_soundings_max_dist_m": float(getattr(cfg, "river_soundings_max_dist_m", 1500.0)),
        "river_soundings_min_r": float(getattr(cfg, "river_soundings_min_r", 0.25)),
        "river_soundings_enforce": bool(getattr(cfg, "river_soundings_enforce", True)),
        "snap_m": cfg.snap_m,

        "river_authoritative_bed": _fingerprint_path(getattr(cfg, "river_authoritative_bed", None)),
        "river_authoritative_bed_max_dist_m": float(getattr(cfg, "river_authoritative_bed_max_dist_m", 2000.0)),
        "river_residual_blend_sigma_m": float(getattr(cfg, "river_residual_blend_sigma_m", 120.0)),
        "river_nodata": float(getattr(cfg, "river_nodata", -9999.0)),
        "mask_river_to_waffles": bool(getattr(cfg, "mask_river_to_waffles", True)),
        "river_use_nhdarea": bool(getattr(cfg, "river_use_nhdarea", True)),
        "river_nhdarea_layer": getattr(cfg, "river_nhdarea_layer", "nhdarea_clip"),

        # Legacy XS parameters (only used when river_method == "xs")
        "xs_spacing_m": cfg.xs_spacing_m,
        "xs_length_m": cfg.xs_length_m,
        "river_continuous": cfg.river_continuous,
        "river_continuous_buffer_m": cfg.river_continuous_buffer_m,
        "river_continuous_k": cfg.river_continuous_k,
        "river_idw_power": cfg.river_idw_power,
        "river_aniso_along_scale_m": cfg.river_aniso_along_scale_m,
        "river_aniso_cross_scale_m": cfg.river_aniso_cross_scale_m,
        "river_thalweg_weight": cfg.river_thalweg_weight,
        "river_thalweg_only": getattr(cfg, "river_thalweg_only", False),
        "river_thalweg_densify_factor": getattr(cfg, "river_thalweg_densify_factor", 0.5),
        "river_thalweg_densify_step_m": getattr(cfg, "river_thalweg_densify_step_m", None),
        "river_overlap_reducer": cfg.river_overlap_reducer,

        # XS generation controls (geometry stability / artifact reduction)
        "xs_smoothing_window_m": getattr(cfg, "xs_smoothing_window_m", 0.0),
        "xs_trim_overlaps": bool(getattr(cfg, "xs_trim_overlaps", True)),
        "xs_global_deconflict": bool(getattr(cfg, "xs_global_deconflict", True)),
        "xs_deconflict_tol_m": float(getattr(cfg, "xs_deconflict_tol_m", 2.0)),
        "xs_skip_junctions": bool(getattr(cfg, "xs_skip_junctions", True)),
        "xs_junction_snap_m": float(getattr(cfg, "xs_junction_snap_m", 30.0)),
        "xs_junction_buffer_m": float(getattr(cfg, "xs_junction_buffer_m", 120.0)),
        "xs_densify_step_m": float(getattr(cfg, "xs_densify_step_m", 20.0)),

        # Skeleton parameters (only used when river_method == "skeleton")
        "river_channel_buffer_m": getattr(cfg, "river_channel_buffer_m", 400.0),
        "river_max_channel_width_m": getattr(cfg, "river_max_channel_width_m", 600.0),
        "river_mainstem_min_order": getattr(cfg, "river_mainstem_min_order", 5),
        "river_max_mainstem_width_m": getattr(cfg, "river_max_mainstem_width_m", 2500.0),
        "river_shape_exp": getattr(cfg, "river_shape_exp", 0.5),
        "river_dmax_min_m": getattr(cfg, "river_dmax_min_m", 0.5),
        "river_dmax_max_m": getattr(cfg, "river_dmax_max_m", 30.0),

        # Priors (shared)
        "river_prior_mode": cfg.river_prior_mode,
        "river_mv_a0": cfg.river_mv_a0,
        "river_mv_bw": cfg.river_mv_bw,
        "river_mv_ba": cfg.river_mv_ba,
        "river_mv_bs": cfg.river_mv_bs,
        "river_mv_eps_a": cfg.river_mv_eps_a,
        "river_mv_eps_s": cfg.river_mv_eps_s,

        # USGS anchors / gage-derived priors
        "river_usgs_sites": getattr(cfg, "river_usgs_sites", None),
        "river_usgs_start": getattr(cfg, "river_usgs_start", None),
        "river_usgs_end": getattr(cfg, "river_usgs_end", None),
        "river_usgs_cache_dir": _fingerprint_path(getattr(cfg, "river_usgs_cache_dir", None)),
        "river_usgs_max_dist_m": float(getattr(cfg, "river_usgs_max_dist_m", 5000.0)),
        "river_usgs_mean_to_dmax": getattr(cfg, "river_usgs_mean_to_dmax", "auto"),
        "river_usgs_a_stat": getattr(cfg, "river_usgs_a_stat", "median"),
        "river_usgs_q_quantile_lo": float(getattr(cfg, "river_usgs_q_quantile_lo", 0.20)),
        "river_usgs_q_quantile_hi": float(getattr(cfg, "river_usgs_q_quantile_hi", 0.80)),
        "river_usgs_a_cv_warn": float(getattr(cfg, "river_usgs_a_cv_warn", 0.50)),
        "river_usgs_width_ratio_max": float(getattr(cfg, "river_usgs_width_ratio_max", 3.0)),
        "river_usgs_width_ratio_blend": bool(getattr(cfg, "river_usgs_width_ratio_blend", True)),
        "river_gage_snap_max_dist_m": float(getattr(cfg, "river_gage_snap_max_dist_m", 1000.0)),

        # Width-stage CSV anchors
        "river_width_stage_csv": getattr(cfg, "river_width_stage_csv", None),
        "river_width_stage_max_dist_m": float(getattr(cfg, "river_width_stage_max_dist_m", 5000.0)),
        "river_width_stage_min_n": int(getattr(cfg, "river_width_stage_min_n", 6)),
        "river_width_stage_min_r2": float(getattr(cfg, "river_width_stage_min_r2", 0.25)),
        "river_width_stage_max_weight": float(getattr(cfg, "river_width_stage_max_weight", 0.8)),

        # Slope/WSE profile controls
        "river_slope_proxy_window": int(getattr(cfg, "river_slope_proxy_window", 9)),
        "river_slope_min": float(getattr(cfg, "river_slope_min", 1e-5)),
        "river_slope_max": float(getattr(cfg, "river_slope_max", 0.05)),
        "river_slope_proxy_min_n": int(getattr(cfg, "river_slope_proxy_min_n", 7)),
        "river_wse_profile_enabled": bool(getattr(cfg, "river_wse_profile_enabled", True)),
        "river_wse_profile_window": int(getattr(cfg, "river_wse_profile_window", 9)),
        "river_wse_profile_min_n": int(getattr(cfg, "river_wse_profile_min_n", 7)),
        "river_wse_profile_monotonic": bool(getattr(cfg, "river_wse_profile_monotonic", True)),

        # Skeleton scientific controls
        "river_skeleton_wse_mode": str(getattr(cfg, "river_skeleton_wse_mode", "bank")),
        "river_skeleton_wse_smooth_sigma_m": float(getattr(cfg, "river_skeleton_wse_smooth_sigma_m", 0.0)),
        "river_skeleton_wse_profile_step_m": float(getattr(cfg, "river_skeleton_wse_profile_step_m", 20.0)),
        "river_skeleton_wse_profile_resample_m": float(getattr(cfg, "river_skeleton_wse_profile_resample_m", 20.0)),
        "river_skeleton_wse_profile_smooth_sigma_m": float(getattr(cfg, "river_skeleton_wse_profile_smooth_sigma_m", 200.0)),
        "river_skeleton_wse_profile_max_slope": float(getattr(cfg, "river_skeleton_wse_profile_max_slope", 0.005)),
        "river_skeleton_wse_profile_min_samples": int(getattr(cfg, "river_skeleton_wse_profile_min_samples", 10)),
        "river_skeleton_wse_profile_max_query_dist_m": float(getattr(cfg, "river_skeleton_wse_profile_max_query_dist_m", 250.0)),
        "river_skeleton_junction_mode": str(getattr(cfg, "river_skeleton_junction_mode", "smooth")),
        "river_skeleton_junction_buffer_m": float(getattr(cfg, "river_skeleton_junction_buffer_m", 120.0)),
        "river_skeleton_junction_degree_min": int(getattr(cfg, "river_skeleton_junction_degree_min", 3)),
        "river_skeleton_junction_smooth_sigma_m": float(getattr(cfg, "river_skeleton_junction_smooth_sigma_m", 80.0)),
        "river_skeleton_junction_max_width_m": float(getattr(cfg, "river_skeleton_junction_max_width_m", 300.0)),
        "river_skeleton_asymmetry_mode": str(getattr(cfg, "river_skeleton_asymmetry_mode", "none")),
        "river_skeleton_asymmetry_strength": float(getattr(cfg, "river_skeleton_asymmetry_strength", 0.25)),
        "river_skeleton_asymmetry_curv_ref": float(getattr(cfg, "river_skeleton_asymmetry_curv_ref", 0.002)),
        "river_skeleton_asymmetry_max_shift": float(getattr(cfg, "river_skeleton_asymmetry_max_shift", 0.20)),
        "river_skeleton_asymmetry_min_width_m": float(getattr(cfg, "river_skeleton_asymmetry_min_width_m", 10.0)),
        "river_skeleton_asymmetry_min_curv": float(getattr(cfg, "river_skeleton_asymmetry_min_curv", 0.0005)),
        "river_skeleton_asymmetry_densify_step_m": float(getattr(cfg, "river_skeleton_asymmetry_densify_step_m", 20.0)),

        "tnm_enable": bool(cfg.tnm_enable),
        "tnm_dataset": cfg.tnm_dataset,
        "scripts": {
            "river_network.py": _fingerprint_script("river_network.py"),
            "xs_builder.py": _fingerprint_script("xs_builder.py"),
            "xs_infer_bathy_raster.py": _fingerprint_script("xs_infer_bathy_raster.py"),
            "river_domain_mask.py": _fingerprint_script("river_domain_mask.py"),
            "river_skeleton_bathy.py": _fingerprint_script("river_skeleton_bathy.py"),
        },
            "version": "river_cache_v8",
    }
    s = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return key, manifest


# -----------------------------------------------------------------------------
# Config

# -----------------------------------------------------------------------------
# IO manifest helpers (explicit paths only; no filename guessing)
# -----------------------------------------------------------------------------

def _is_probably_path(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    # Reject whitespace-containing strings (e.g., full command lines); this manifest is paths only.
    if any(ch.isspace() for ch in s):
        return False
    # Avoid URLs
    if "://" in s:
        return False
    # Heuristic: has a path separator or looks like a filename with extension
    if "/" in s or "\\" in s:
        return True
    if re.search(r"\.[A-Za-z0-9]{2,6}$", s):
        return True
    return False


def _collect_paths_from_obj(obj: Any, out: list[str]) -> None:
    """Collect explicit path-like strings from nested dict/list structures."""
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _collect_paths_from_obj(v, out)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _collect_paths_from_obj(v, out)
        elif isinstance(obj, str):
            if _is_probably_path(obj):
                out.append(obj)
    except Exception:
        return


def _parse_paths_from_command(cmd: str) -> dict[str, list[str]]:
    """
    Parse subprocess command strings and extract explicit file paths from known flags.
    Returns {"inputs": [...], "outputs": [...]}.
    """
    ins: list[str] = []
    outs: list[str] = []
    if not isinstance(cmd, str) or not cmd.strip():
        return {"inputs": ins, "outputs": outs}
    try:
        toks = shlex.split(cmd)
    except Exception:
        toks = cmd.split()

    # Flags where the next token is a path
    out_flags = {
        "-O", "--out", "--out-dir", "--out_gpkg", "--out-gpkg", "--out_tif", "--out-tif",
        "--out-xyz", "--out-csv", "--out-json", "--out-md", "--out-mask", "--out-raster",
        "--output", "--output-dir", "--output-path",
        "--dem-out", "--mask-out",
    }
    in_flags = {
        "-i", "--in", "--input", "--input-path",
        "--dem", "--template", "--template-raster", "--mask", "--mask-raster",
        "--ocean-mask", "--water-mask", "--river-mask",
        "--xs", "--xs-gpkg", "--network", "--network-gpkg",
        "--soundings", "--xyz", "--extra-xyz",
    }

    i = 0
    while i < len(toks):
        t = toks[i]
        if t in out_flags and i + 1 < len(toks):
            p = toks[i + 1]
            if _is_probably_path(p):
                outs.append(p)
            i += 2
            continue
        if t in in_flags and i + 1 < len(toks):
            p = toks[i + 1]
            if _is_probably_path(p):
                ins.append(p)
            i += 2
            continue
        # Common pattern: --flag=/path
        if t.startswith("--") and "=" in t:
            flag, val = t.split("=", 1)
            if _is_probably_path(val):
                if flag in out_flags:
                    outs.append(val)
                elif flag in in_flags:
                    ins.append(val)
                else:
                    # Unknown role: treat as input (conservative)
                    ins.append(val)
        i += 1

    return {"inputs": ins, "outputs": outs}


def build_io_manifest(report: dict[str, Any]) -> dict[str, Any]:
    """
    Build an explicit IO manifest from the *actual* run report and executed commands.
    This contains only paths observed in the report/commands. No canonical naming assumptions.
    """
    manifest: dict[str, Any] = {
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": report.get("run_id"),
        "out_dir": report.get("out_dir"),
        "inputs": [],
        "outputs": [],
    }

    # 1) Collect explicit paths from report structure (bathy_report / unified report already store them)
    collected: list[str] = []
    _collect_paths_from_obj(report, collected)

    # 2) Collect inputs/outputs from step command lines
    steps_obj = None
    try:
        steps_obj = (report.get("river", {}) or {}).get("steps")
    except Exception:
        steps_obj = None

    step_list = []
    if isinstance(steps_obj, dict):
        step_list = list(steps_obj.values())
    elif isinstance(steps_obj, list):
        step_list = steps_obj

    for step in step_list:
        cmd = step.get("command") if isinstance(step, dict) else None
        parsed = _parse_paths_from_command(cmd or "")
        collected.extend(parsed.get("inputs", []))
        collected.extend(parsed.get("outputs", []))

    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for p in collected:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    # Split into inputs/outputs using explicit report fields where possible
    # - Anything listed under report["outputs"] or nested "*outputs" dicts -> outputs
    out_paths: set[str] = set()
    def _collect_outputs(obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "outputs" and isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, str) and _is_probably_path(vv):
                            out_paths.add(vv)
                _collect_outputs(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _collect_outputs(v)
    _collect_outputs(report)

    # Also treat explicit step outputs as outputs
    for step in step_list:
        cmd = step.get("command") if isinstance(step, dict) else ""
        parsed = _parse_paths_from_command(cmd or "")
        for p in parsed.get("outputs", []):
            out_paths.add(p)

    inputs = []
    outputs = []
    for p in uniq:
        if p in out_paths:
            outputs.append(p)
        else:
            inputs.append(p)

    manifest["inputs"] = inputs
    manifest["outputs"] = outputs
    return manifest


def write_io_manifest(out_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    """Write io_manifest.json and io_manifest.md into out_dir."""
    io = build_io_manifest(report)
    p_json = out_dir / "io_manifest.json"
    p_md = out_dir / "io_manifest.md"

    # JSON
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(io, f, indent=2, sort_keys=True)

    # MD
    lines = []
    lines.append("# IO manifest")
    lines.append("")
    lines.append(f"- run_id: `{io.get('run_id')}`")
    lines.append(f"- out_dir: `{io.get('out_dir')}`")
    lines.append("")
    lines.append("## Inputs (explicit paths observed)")
    for p in io.get("inputs", []):
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Outputs (explicit paths observed)")
    for p in io.get("outputs", []):
        lines.append(f"- `{p}`")
    lines.append("")
    p_md.write_text("\n".join(lines), encoding="utf-8")

    return p_json, p_md


def _emit_artifacts_from_report(report: dict[str, Any]) -> None:
    """
    Best-effort: emit artifact_written events for paths explicitly recorded in the report.
    This is used only for run summaries/flight recorder; it never guesses filenames.
    """
    try:
        from flight_recorder import emit_artifact_written
    except Exception:
        return

    # Emit outputs from report["outputs"] dicts only
    def _emit_outputs(obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "outputs" and isinstance(v, dict):
                    for name, path in v.items():
                        if isinstance(path, str) and _is_probably_path(path):
                            emit_artifact_written(Path(path), kind="file", role=str(name))
                _emit_outputs(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _emit_outputs(v)
    _emit_outputs(report)

# -----------------------------------------------------------------------------

@dataclass
class BathyConfig:
    aoi: str
    start_date: str
    end_date: str
    # Operational tiling policy:
    # - Run processing on an expanded AOI to reduce edge effects.
    # - Clip final outputs back to the tile AOI.
    aoi_tile: Optional[str] = None  # original tile AOI string (W/E/S/N)
    tile_bbox: Optional[Tuple[float, float, float, float]] = None  # (W,S,E,N) in EPSG:4269
    tile_buffer_km: float = 15.0
    tile_edge_taper_enabled: bool = True
    tile_edge_taper_km: float = 2.0
    tile_edge_smooth_sigma_km: float = 10.0
    tile_edge_metrics_band_km: float = 2.0

    # SDB cross-tile consistency policy (DEFAULT):
    # Use a bounded "model bank" (reservoir sample) across AOIs and periodically retrain
    # the RF only when enough new samples have accumulated.
    # This drives cross-tile consistency without storing unbounded training data.
    sdb_model_bank_enabled: bool = True
    sdb_model_bank: str = "auto"
    sdb_bank_max_samples: int = 100000
    sdb_bank_seed: int = 1337
    sdb_bank_retrain_min_new: int = 2000

    # Deprecated regional model cache (kept for backwards compatibility)
    sdb_model_cache_enabled: bool = False
    sdb_model_cache_key: str = "auto"


    out_dir: Path = Path("output/unified")
    # Output retention policy
    # Default (save_intermediates=False): keep only final deliverables + run metadata/logs.
    # If enabled: move all non-deliverable artifacts into out_dir/<intermediates_dirname>/...
    save_intermediates: bool = False
    intermediates_dirname: str = "debug"
    methods: List[str] = field(default_factory=lambda: ["sdb", "river"])
    priority: str = "sdb"  # "sdb" or "river"

    # SDB args passed through
    cloud: int = 70
    icesat: str = "all_atl"
    sdb_mode: str = "all_sdb"
    cache_root: Path = Path("cache")
    align_mode: str = "median"

    # Sun-glint correction (Hedley-style) applied to Sentinel-2 composites (SDB only)
    glint_correct: bool = False
    glint_nir_band: str = "B08"
    glint_vis_bands: str = "B02,B03,B04"
    glint_nir_min_percentile: float = 1.0
    glint_deepwater_b02_max: float = 0.20
    glint_min_samples: int = 5000
    glint_max_samples: int = 2000000
    glint_clip_min: float = 1e-6


    # CRS policy
    # working_srs: CRS used internally for meter-based operations (thinning, river DEM warps).
    # If not provided, we attempt to read it from the Sentinel-2 RGB_10m.tif CRS; otherwise
    # we compute a WGS84 UTM zone from the AOI center.
    working_srs: str = "auto"
    working_vcrs_epsg: int = 5703  # NAVD88 height (EPSG:5703)

    # River constraint guardrail (fusion gating)
    require_river_constraints: str = "none"

    # Final output CRS for rasters (default NAD83 geographic (horizontal-only) height)
    final_out_srs: str = "EPSG:4269"

    # River DEM auto-download (TNM 1/3 arc-sec) when --river-dem is not provided.
    river_dem_auto: bool = True
    river_dem_source: str = "tnm:datasets=3"
    river_dem_res_m: float = 10.0
    extra_xyz_crs: str = "EPSG:4326"

    # External bathymetry soundings
    # - extra_xyz: user-provided files (normalized later into cfg.river_soundings)
    # - extra_xyz_cudem: requested CUDEM dlim providers (e.g., hydronos, ehydro)
    extra_xyz: List[str] = field(default_factory=list)
    extra_xyz_cudem: List[str] = field(default_factory=list)

    # River args
    river_dem: Optional[Path] = None
    river_soundings: Optional[str] = None
    # If river soundings are provided, they can refine the skeleton Dmax prior and optionally be enforced.
    river_soundings_mode: str = "auto"            # auto | depth_pos | depth_neg
    river_soundings_max_dist_m: float = 1500.0    # max distance for soundings to influence skeleton prior
    river_soundings_min_r: float = 0.25           # min r when inverting depth->Dmax
    river_soundings_enforce: bool = True          # enforce observed depths at sounding pixels
    # Optional authoritative bed elevation raster blending (NAVD88, etc.)
    river_authoritative_bed: Optional[Path] = None
    river_authoritative_bed_max_dist_m: float = 2000.0   # max distance for authoritative residual influence (m)
    river_residual_blend_sigma_m: float = 120.0          # Gaussian sigma for residual blending (m); 0 disables
    # River bathymetry method:
    # - "skeleton": raster distance-transform "channel skeleton" method (no cross-sections). Recommended for sinuous/tidal channels.
    # - "xs": legacy vector cross-section method (xs_builder.py + xs_infer_bathy_raster.py)
    river_method: str = "hybrid"

    # Skeleton (distance-transform) method parameters
    # These control *where* river bathy is applied (river vs. ocean) and the within-channel depth profile.
    river_channel_buffer_m: float = 400.0           # buffer around NHD flowlines to define candidate river corridor
    river_max_channel_width_m: float = 600.0        # max channel width allowed in corridor (prevents filling open bays)
    river_mainstem_min_order: int = 5               # stream order threshold for allowing larger widths (if available)
    river_max_mainstem_width_m: float = 2500.0      # max width allowed for mainstem corridor (m)
    river_shape_exp: float = 0.5                    # depth profile exponent (0.5 ~ U-shaped; 1.0 ~ V-shaped)
    river_dmax_min_m: float = 0.5                   # clamp Dmax prior (m)
    river_dmax_max_m: float = 30.0                  # clamp Dmax prior (m)
    # Optional: longitudinal bed profile constraints (skeleton method)
    river_bed_profile_max_slope: float = 0.0       # max |dz/ds| along flow (m/m); 0 disables
    river_bed_profile_max_curv: float = 0.0        # max |d2z/ds2| along flow (1/m); 0 disables
    river_bed_profile_step_m: float = 25.0         # sampling step (m) for profile constraints
    river_bed_profile_strength: float = 0.6        # blend strength (0..1)
    river_bed_profile_power: float = 2.0           # distance-decay power when spreading correction
    river_save_skeleton_debug: bool = False         # write debug rasters (r, d_bank, d_center, dmax, wse)

    # Skeleton WSE proxy controls (scientific correctness guardrails)
    river_skeleton_wse_mode: str = "bank"          # 'bank' (recommended) or 'skeleton' (legacy)
    river_skeleton_wse_smooth_sigma_m: float = 0.0 # optional smoothing of WSE field inside channel
    # Confluence/junction handling (degree>=3 graph nodes). Helps suppress artifacts near confluences.
    river_skeleton_junction_mode: str = "smooth"   # smooth | mask | none
    river_skeleton_junction_buffer_m: float = 120.0
    river_skeleton_junction_degree_min: int = 3
    river_skeleton_junction_smooth_sigma_m: float = 80.0
    river_skeleton_junction_max_width_m: float = 300.0
    # Optional: constrain river bathymetry domain using polygonal channel features (NHDArea)
    river_use_nhdarea: bool = True
    river_nhdarea_layer: str = "nhdarea_clip"
    river_ocean_keep_dist_m: float = 0.0  # allow ocean-connected water near flowlines (tidal mouths)
    # River depth inference priors / anchors (passed through to xs_infer_bathy_raster.py)
    river_prior_mode: str = "powerlaw"  # powerlaw | multivariate
    river_mv_a0: float = 0.18
    river_mv_bw: float = 0.50
    river_mv_ba: float = 0.0
    river_mv_bs: float = -0.10
    river_mv_eps_a: float = 1.0
    river_mv_eps_s: float = 1e-4

    river_usgs_sites: Optional[str] = None           # comma-separated site numbers
    river_usgs_start: Optional[str] = None           # YYYY-MM-DD
    river_usgs_end: Optional[str] = None             # YYYY-MM-DD
    river_usgs_cache_dir: Optional[Path] = None
    river_usgs_max_dist_m: float = 5000.0
    river_usgs_mean_to_dmax: str = "auto"
    river_usgs_a_stat: str = "median"                # median | p90 | mean
    river_usgs_q_quantile_lo: float = 0.20
    river_usgs_q_quantile_hi: float = 0.80
    river_usgs_a_cv_warn: float = 0.50
    river_usgs_width_ratio_max: float = 3.0
    river_usgs_width_ratio_blend: bool = True
    river_gage_snap_max_dist_m: float = 1000.0


    river_width_stage_csv: Optional[str] = None      # comma-separated CSV paths
    river_width_stage_max_dist_m: float = 5000.0
    river_width_stage_min_n: int = 6
    river_width_stage_min_r2: float = 0.25
    river_width_stage_max_weight: float = 0.8
    river_slope_proxy_window: int = 9
    river_slope_min: float = 1e-5
    river_slope_max: float = 0.05
    river_slope_proxy_min_n: int = 7

    # Longitudinal WSE profile fit (preferred slope proxy when reach slope attribute is missing)
    river_wse_profile_enabled: bool = True
    river_wse_profile_window: int = 9
    river_wse_profile_min_n: int = 7
    river_wse_profile_monotonic: bool = True

    # Optional: pass through to river_network
    # Hydrography acquisition strategy for river network & polygons
    # - "arcgis": ArcGIS REST only (fast, avoids TNM catalog crawl)
    # - "arcgis_tnm": ArcGIS first, then TNM fallback if ArcGIS fails
    # - "tnm": TNM preferred (will still fall back to ArcGIS as a guardrail)
    river_hydrography_source: str = "arcgis"

    tnm_enable: bool = False
    tnm_dataset: str = "NHDPlusHR"
    snap_m: float = 30.0

    # XS params
    xs_spacing_m: float = 200.0
    xs_length_m: float = 300.0
    xs_smoothing_window_m: float = 0.0   # 0 = auto (use xs_spacing_m)
    xs_trim_overlaps: bool = True        # local/adjacent trimming
    xs_global_deconflict: bool = True    # drop XS that intersect non-adjacent XS within a reach
    xs_deconflict_tol_m: float = 2.0     # treat endpoint "touches" within this tolerance as non-harmful
    xs_skip_junctions: bool = True       # avoid XS too close to confluences/junction nodes
    xs_junction_snap_m: float = 30.0     # snapping scale for junction detection (m)
    xs_junction_buffer_m: float = 75.0  # do not place XS within this distance of junctions (m)
    xs_densify_step_m: float = 20.0      # densify centerlines to this vertex spacing before tangents (m)

    # River patch rasterization / interpolation options (passed to xs_infer_bathy_raster.py)
    # NOTE: xs_infer defaults can be expensive for large networks; expose here for control.
    river_continuous: str = "walid_aniso"  # median | walid | aidw | aniso | walid_aniso
    river_continuous_buffer_m: Optional[float] = None
    river_continuous_k: int = 12
    river_idw_power: float = 2.0
    river_aniso_along_scale_m: float = 500.0
    river_aniso_cross_scale_m: float = 30.0
    river_thalweg_weight: float = 6.0

    river_thalweg_only: bool = False
    river_thalweg_densify_factor: float = 0.5
    river_thalweg_densify_step_m: Optional[float] = None
    river_overlap_reducer: str = "median"  # min | median
    river_nodata: float = -9999.0

    # Fusion controls
    fusion_strategy: str = "spatial_taper"  # weighted_overlap | spatial_taper | priority | blend
    fusion_primary_weight: float = 0.70
    fusion_secondary_weight: float = 0.30
    fusion_taper_m: float = 75.0  # meters for spatial taper inside river corridor

    # Output masking
    mask_river_to_waffles: bool = True  # apply waffles coastline mask to river depth + bed outputs when available

    # Intelligent gap-filling (Tier 1–2): prior + residual interpolation
    gapfill_enabled: bool = False
    gapfill_hq: Optional[List[str]] = None  # list of HQ point files (x,y,depth)
    gapfill_water_mask: Optional[Path] = None  # optional explicit water mask (1=water)
    gapfill_method: str = "rbf"  # rbf | gp | idw
    gapfill_river_smooth_sigma_m: float = 500.0  # along-channel smoothing sigma
    gapfill_prior_sigma_raster: Optional[Path] = None  # optional prior uncertainty raster
    gapfill_bank_elev_raster: Optional[Path] = None  # optional bank elevation for constraints
    gapfill_output_cudem_xyz: bool = False  # output CUDEM-compatible XYZ with uncertainty

    # Run health / contracts
    strict: bool = False  # if True, run contract tests and fail fast on invalid outputs


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _choose_waffles_mask_for_river(cfg: "BathyConfig", report: Dict[str, Any]) -> Optional[Path]:
    """Choose the *least destructive* WAFFLES mask for clipping river outputs.

    Invariant: never wipe valid river pixels just because an *ocean-only* WAFFLES
    mask happens to be the newest under cache_root/masks.

    Preference order (explicit only, no guessing):
      1) Run-scoped domain inference mask (with NHD), if available.
      2) River report's recorded waffles water mask (from river_domain_mask step).

    Returns None if nothing is available.
    """
    # 1) Run-scoped (preferred)
    for attr in ("waffles_with_nhd_mask", "water_domain_mask_for_final"):
        try:
            p = getattr(cfg, attr, None)
            if p:
                pp = Path(p)
                if pp.exists() and pp.stat().st_size > 0:
                    return pp
        except Exception:
            pass

    # 2) Report hint
    try:
        wm = report.get("river", {}).get("outputs", {}).get("waffles_water_mask", None)
        if wm:
            pp = Path(wm)
            if pp.exists() and pp.stat().st_size > 0:
                return pp
    except Exception:
        pass

    return None

def _count_mask_water_pixels(mask_tif: Path, aoi_bounds_wgs84: Tuple[float, float, float, float], *, water_max: float = 0.5) -> Optional[int]:
    """Count water pixels (mask <= water_max) inside AOI bounds.

    Returns None if mask cannot be read.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer

        xmin, xmax, ymin, ymax = aoi_bounds_wgs84
        with rasterio.open(mask_tif) as ds:
            if ds.crs and ds.crs.to_epsg() not in (None, 4326):
                tx = Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True)
                xmin2, ymin2 = tx.transform(xmin, ymin)
                xmax2, ymax2 = tx.transform(xmax, ymax)
                xmin, xmax = min(xmin2, xmax2), max(xmin2, xmax2)
                ymin, ymax = min(ymin2, ymax2), max(ymin2, ymax2)

            bxmin, bymin, bxmax, bymax = ds.bounds
            ixmin, ixmax = max(xmin, bxmin), min(xmax, bxmax)
            iymin, iymax = max(ymin, bymin), min(ymax, bymax)
            if ixmin >= ixmax or iymin >= iymax:
                return 0

            win = from_bounds(ixmin, iymin, ixmax, iymax, transform=ds.transform)
            arr = ds.read(1, window=win, masked=True)
            if arr.size == 0:
                return 0
            water = (arr <= water_max)
            if hasattr(water, 'filled'):
                water = water.filled(False)
            return int(np.count_nonzero(water))
    except Exception:
        return None





def _stage_cached_waffles_mask(src: Path, dst: Path, log: Optional[logging.Logger]=None) -> Path:
    """Stage a cached WAFFLES mask into a run-scoped directory without regenerating it.

    This creates a symlink when possible (fast, deterministic). Falls back to copying.
    The returned path is the staged destination (which may already exist).
    """
    ensure_dir(dst.parent)
    try:
        if dst.exists():
            return dst
        # Prefer symlink to avoid copies and ensure exact reuse.
        try:
            os.symlink(src, dst)
            return dst
        except Exception:
            shutil.copy2(src, dst)
            return dst
    except Exception as e:
        if log:
            log.warning(f"[WAFFLES] Failed to stage cached mask {src} -> {dst}: {e}")
        # Best effort: return source so callers can still proceed.
        return src

def _ensure_waffles_coastline_mask(
    cache_masks: Path,
    aoi: str,
    inc_arcsec: float = 1.0,
    want_nhd: bool = True,
    want_lakes: bool = False,
    prefix: str = "waffles_coastline",
    out_tif: Optional[Path] = None,
    force: bool = False,
    log: bool = True,
) -> Path:
    """Ensure a WAFFLES coastline land/water mask exists.

    This is a *derived* product. By default we allow reuse if it already exists,
    but callers that want to avoid stale derived outputs should pass force=True
    and/or provide a run-scoped out_tif path.
    """
    ensure_dir(cache_masks)
    logger = logging.getLogger("bathy_main")

    params = dict(
        aoi=str(aoi),
        inc_arcsec=float(inc_arcsec),
        want_nhd=bool(want_nhd),
        want_lakes=bool(want_lakes),
        prefix=str(prefix),
    )
    chash = _stable_hash_str(json.dumps(params, sort_keys=True, default=str))

    if out_tif is not None:
        out_tif_path = Path(out_tif)
        out_prefix = out_tif_path.with_suffix("")
    else:
        out_prefix = cache_masks / f"{prefix}_{chash}"
        out_tif_path = out_prefix.with_suffix(".tif")

    # Derived products must not be reused unless explicitly allowed.
    if force:
        for fp in [out_tif_path, out_tif_path.with_suffix(out_tif_path.suffix + ".aux.xml")]:
            try:
                if fp.exists():
                    fp.unlink()
            except Exception:
                pass

    if (not force) and out_tif_path.exists() and out_tif_path.stat().st_size > 0:
        if log:
            logger.info(f"[WAFFLES] Cache hit: {out_tif_path}")
        return out_tif_path

    # Build WAFFLES command deterministically.
    # NOTE: WAFFLES CLI does not accept long-form flags like --want-nhd; module parameters must be
    # passed via the -M module string (e.g., coastline:want_nhd=true:want_lakes=false).
    module = (
        f"coastline:want_nhd={'true' if want_nhd else 'false'}:"
        f"want_lakes={'true' if want_lakes else 'false'}"
    )
    cmd = [
        "waffles",
        "-R",
        str(aoi),
        "-E",
        f"{inc_arcsec}s",
        "-M",
        module,
        "-O",
        str(out_prefix),
        "-F",
        "GTiff",
    ]

    if log:
        logger.info("[WAFFLES] Command: %s", " ".join(cmd))

    # Use the shared subprocess helper (deterministic, no shell). Capture stdout/stderr without
    # streaming by default to keep WAFFLES output from flooding the main logs.
    rc, out, err = run_command(cmd, stream_stdout=False, stream_stderr=False)
    if rc != 0:
        _tail_err = (err or "").strip()[:500]
        _tail_out = (out or "").strip()[:500]
        raise RuntimeError(
            f"waffles coastline failed (rc={rc}): stderr={_tail_err} stdout={_tail_out}"
        )

    if out_tif_path.exists() and out_tif_path.stat().st_size > 0:
        return out_tif_path

    # No guessing: require the exact expected output path.
    # If WAFFLES produced something unexpected, fail loudly with a directory listing.
    existing = sorted([p.name for p in cache_masks.glob("*.tif")])
    raise RuntimeError(
        f"WAFFLES did not produce expected mask: {out_tif_path}. Existing in {cache_masks}: {existing[:50]}"
    )
def _waffles_water_fraction(mask_tif: Path, max_samples: int = 200000) -> float:
    """
    Compute fraction of water pixels in a WAFFLES coastline mask.

    Background / gotcha:
      - WAFFLES coastline masks are typically binary (water vs land), but the GTiff can
        carry a NODATA value.
      - If NODATA is set to 0 (or a driver interprets 0 as nodata), reading with
        rasterio.read(masked=True) will *mask out* valid water pixels (value==0),
        causing false "no water" results and skipping SDB/river.

    Implementation:
      - Read *unmasked* data (masked=False), and use the GDAL validity mask
        (read_masks) when available.
      - Treat WAFFLES convention water=0, land=1 (the value-based convention),
        but do not allow NODATA masking to erase water pixels.

    Returns 0.0 if raster missing or has no valid pixels.
    """
    try:
        import rasterio
        import numpy as np
        from rasterio.windows import Window

        if mask_tif is None or (not Path(mask_tif).exists()):
            return 0.0

        with rasterio.open(str(mask_tif)) as ds:
            h, w = ds.height, ds.width

            # Deterministic grid sampling of small windows
            step_y = max(1, int(round(h / 12)))
            step_x = max(1, int(round(w / 12)))
            total = 0
            water = 0

            for y0 in range(0, h, step_y):
                for x0 in range(0, w, step_x):
                    win = Window(col_off=x0, row_off=y0, width=min(256, w - x0), height=min(256, h - y0))

                    # IMPORTANT: masked=False to avoid nodata==0 erasing water pixels.
                    a = ds.read(1, window=win, masked=False)

                    if a is None:
                        continue

                    # Prefer dataset validity mask; if unavailable/empty, treat all as valid.
                    try:
                        vm = ds.read_masks(1, window=win)
                        m = (vm > 0)
                    except Exception:
                        m = np.ones_like(a, dtype=bool)

                    if m is None or (not np.any(m)):
                        # Some files don't carry a mask; fall back to all-valid.
                        m = np.ones_like(a, dtype=bool)

                    total += int(np.sum(m))
                    water += int(np.sum((a == 0) & m))

                    if total >= int(max_samples):
                        break
                if total >= int(max_samples):
                    break

            if total <= 0:
                return 0.0
            return float(water) / float(total)
    except Exception:
        logging.getLogger(__name__).debug("WAFFLES water fraction check failed.", exc_info=True)
        return 0.0


def _determine_effective_methods_from_waffles(cfg: "BathyConfig", report: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Determine which methods to actually run based on WAFFLES masks.

    Policy:
      - Default requested methods are sdb,river,fuse (but user may request subsets).
      - Always build two WAFFLES masks (when any water-driven method requested):
          * ocean-only: want_nhd=false, want_lakes=false
          * with-nhd:  want_nhd=true,  want_lakes=false
      - Compute water fractions from these masks.
      - If ocean-only has ~no water -> skip SDB.
      - If with-nhd has ~no water -> skip river.
      - Fuse runs if requested; it becomes pass-through if only one source exists.

    River is treated as the authoritative vertical reference in coastal overlap handling
    (fusion handles offset calibration + taper when configured).
    """
    requested = [m.strip().lower() for m in (getattr(cfg, "methods", []) or []) if m.strip()]
    if not requested:
        requested = ["sdb", "river", "fuse"]

    req_set = set(requested)
    want_sdb = "sdb" in req_set
    want_river = "river" in req_set
    want_fuse = "fuse" in req_set or (want_sdb and want_river)

    # Only do WAFFLES inference if any water-based methods requested.
    if not (want_sdb or want_river or want_fuse):
        return requested, {"requested": requested, "effective": requested, "waffles": {}}

    cache_masks_shared = Path(getattr(cfg, "cache_root", ".")).resolve() / "masks"
    ensure_dir(cache_masks_shared)

    # Reuse WAFFLES coastline masks across runs for the same AOI + resolution + module params.
    # These masks should not depend on time range, extra_xyz, or run_id.
    ocean_mask_cache = _ensure_waffles_coastline_mask(
        cache_masks=cache_masks_shared,
        aoi=str(getattr(cfg, "aoi", "")),
        inc_arcsec=float(getattr(cfg, "waffles_inc_arcsec", 1.0) or 1.0),
        want_nhd=False,
        want_lakes=False,
        prefix="waffles_coastline_ocean_only",
        force=False,
    )
    nhd_mask_cache = _ensure_waffles_coastline_mask(
        cache_masks=cache_masks_shared,
        aoi=str(getattr(cfg, "aoi", "")),
        inc_arcsec=float(getattr(cfg, "waffles_inc_arcsec", 1.0) or 1.0),
        want_nhd=True,
        want_lakes=False,
        prefix="waffles_coastline_with_nhd",
        force=False,
    )

    # Stage into run-scoped derived_cache for easy inspection (symlink/copy), but do not regenerate.
    cache_masks_run = Path(cfg.derived_cache_root) / "masks"
    ocean_mask = _stage_cached_waffles_mask(ocean_mask_cache, cache_masks_run / "waffles_coastline_ocean_only.tif", log=logging.getLogger("bathy_main"))
    nhd_mask = _stage_cached_waffles_mask(nhd_mask_cache, cache_masks_run / "waffles_coastline_with_nhd.tif", log=logging.getLogger("bathy_main"))
    ocean_frac = _waffles_water_fraction(ocean_mask) if ocean_mask else 0.0
    nhd_frac = _waffles_water_fraction(nhd_mask) if nhd_mask else 0.0

    # Threshold policy: keep as a named knob so you can tune without code surgery.
    # Default is conservative: treat tiny slivers as "no meaningful water domain".
    try:
        min_frac = float(getattr(cfg, "waffles_min_water_fraction", 0.001) or 0.001)
    except Exception:
        min_frac = 0.001

    run_sdb = bool(want_sdb and (ocean_frac >= min_frac))
    run_river = bool(want_river and (nhd_frac >= min_frac))

    effective: List[str] = []
    skipped: Dict[str, str] = {}

    if want_sdb:
        if run_sdb:
            effective.append("sdb")
        else:
            skipped["sdb"] = "no_ocean_water_detected_by_waffles"
    if want_river:
        if run_river:
            effective.append("river")
        else:
            skipped["river"] = "no_nhd_water_detected_by_waffles"
    if want_fuse:
        # Fuse is only meaningful if we have at least one upstream method effective.
        if effective:
            effective.append("fuse")
        else:
            skipped["fuse"] = "no_upstream_sources"

    # Stash masks for downstream usage (fusion + final clipping)
    try:
        cfg.waffles_ocean_mask = Path(ocean_mask) if ocean_mask else None
        cfg.waffles_with_nhd_mask = Path(nhd_mask) if nhd_mask else None
        cfg.ocean_domain_mask_for_fusion = Path(ocean_mask) if ocean_mask else None
        cfg.water_domain_mask_for_final = Path(nhd_mask) if nhd_mask else None
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    meta = {
        "requested": requested,
        "effective": effective,
        "skipped": skipped,
        "waffles": {
            "ocean_only_mask": str(ocean_mask) if ocean_mask else None,
            "with_nhd_mask": str(nhd_mask) if nhd_mask else None,
            "ocean_only_mask_cache": str(ocean_mask_cache) if "ocean_mask_cache" in locals() else None,
            "with_nhd_mask_cache": str(nhd_mask_cache) if "nhd_mask_cache" in locals() else None,
            "ocean_water_fraction": ocean_frac,
            "with_nhd_water_fraction": nhd_frac,
            "min_water_fraction": min_frac,
        },
    }
    report.setdefault("domain_inference", {}).update(meta)
    return effective, meta

def detect_working_srs(cfg: 'BathyConfig') -> str:
    """Determine the working CRS.

    Preference order:
      1) User-specified cfg.working_srs (anything other than 'auto')
      2) CRS of an existing Sentinel-2 RGB_10m.tif in the cache
      3) WGS84 UTM zone based on AOI center

    Returns a CRS string usable by GDAL/PROJ (e.g., 'EPSG:32616').
    """
    if cfg.working_srs and str(cfg.working_srs).lower() != "auto":
        return str(cfg.working_srs)

    # Try to detect from cached Sentinel-2 products
    try:
        import rasterio
        s2_root = Path(cfg.cache_root) / "sentinel2"
        if s2_root.exists():
            candidates = sorted(s2_root.glob("S2_*/*RGB_10m.tif")) + sorted(s2_root.glob("S2_*/RGB_10m.tif"))
            for p in candidates:
                try:
                    with rasterio.open(p) as ds:
                        if ds.crs:
                            return ds.crs.to_string()
                except Exception:
                    continue
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Fallback: AOI center UTM zone (WGS84)
    lon, lat = _aoi_center_lonlat(cfg.aoi)
    return _utm_epsg_from_lonlat(lon, lat)

def _compound_srs(horizontal_srs: str, v_epsg: int) -> str:
    # Keep whatever CRS string the user provided (EPSG:xxxx or WKT), but append +EPSG:xxxx for vertical.
    # PROJ/GDAL accept 'EPSG:XXXX+YYYY' for compound CRS in many contexts.
    h = str(horizontal_srs)
    v = f"{int(v_epsg)}"
    if "+" in h:
        # If already compound, leave as-is.
        return h
    if h.lower().startswith("epsg:"):
        return f"{h}+{v}"
    return f"{h}+{v}"



def ensure_river_dem_auto(cfg: 'BathyConfig', report: Dict[str, Any]) -> Optional[Path]:
    """Auto-download/build a river DEM using CUDEM `fetches` (TNM 1/3 arc-sec) when cfg.river_dem is not provided.

    Steps (best-effort):
      1) `fetches -R=<AOI> tnm:datasets=3` into <cache_root>/river_dem/tnm/
      2) If multiple versions of a tile exist (historical), keep the newest by date in filename.
      3) Build a clipped, reprojected DEM in the *working* CRS (cfg.working_srs) at cfg.river_dem_res_m.

    Returns a path to a GeoTIFF, or None on failure.
    """
    if cfg.river_dem and Path(cfg.river_dem).exists():
        return Path(cfg.river_dem)

    if not cfg.river_dem_auto:
        report.setdefault("river", {})["status"] = "skipped"
        report["river"]["reason"] = "river_dem_missing_and_auto_disabled"
        return None

    fetches_exe = shutil.which("fetches")
    if fetches_exe is None:
        report.setdefault("river", {})["status"] = "skipped"
        report["river"]["reason"] = "fetches_not_found"
        log.warning("[RIVER][DEM] fetches not found on PATH; cannot auto-download TNM DEM.")
        return None

    # Working CRS (projected, meter units preferred)
    working_srs = detect_working_srs(cfg)
    cfg.working_srs = working_srs

    dem_cache = ensure_dir(Path(cfg.cache_root) / "river_dem")
    tnm_dir = ensure_dir(dem_cache / "tnm")
    manifest = {
        "aoi": cfg.aoi,
        "source": cfg.river_dem_source,
        "working_srs": working_srs,
        "res_m": cfg.river_dem_res_m,
        "tiles_dir": str(tnm_dir),
    }
    report.setdefault("river", {})["dem_auto"] = manifest

    # Download into tnm_dir (fetches writes into subdir in CWD, so we run with cwd=tnm_dir parent)
    try:
        cmd = [fetches_exe, f'-R={cfg.aoi}', cfg.river_dem_source]
        log.info("[RIVER][DEM] Auto-download: %s", " ".join(cmd))
        proc = run_cmd(cmd, cwd=str(dem_cache))
        report["river"]["dem_auto"]["fetches_rc"] = proc.returncode
        report["river"]["dem_auto"]["fetches_stderr_tail"] = (proc.stderr or "")[-4000:]
        if proc.returncode != 0:
            log.warning("[RIVER][DEM] fetches failed (rc=%s). stderr tail: %s", proc.returncode, report["river"]["dem_auto"]["fetches_stderr_tail"])
    except Exception as e:
        log.warning("[RIVER][DEM] fetches exception: %s", e)

    # fetches for tnm typically creates a 'tnm' subdir in cwd; support both.
    tnm_search_dirs = [tnm_dir, dem_cache / "tnm"]
    tifs = []
    for d in tnm_search_dirs:
        if d.exists():
            tifs.extend(sorted(d.glob("*.tif")))
    if not tifs:
        log.warning("[RIVER][DEM] No TNM GeoTIFFs found after fetches.")
        report["river"]["dem_auto"]["status"] = "failed_no_tifs"
        return None

    # De-duplicate by tile id with newest date (USGS_13_n41w075_YYYYMMDD.tif)
    def tile_key(p: Path) -> str:
        m = re.search(r"(n\d{2}w\d{3})", p.name.lower())
        return m.group(1) if m else p.stem.lower()

    def date_key(p: Path) -> int:
        m = re.search(r"(\d{8})", p.name)
        return int(m.group(1)) if m else 0

    best = {}
    for p in tifs:
        k = tile_key(p)
        if k not in best or date_key(p) > date_key(best[k]):
            best[k] = p
    keep = sorted(best.values())
    report["river"]["dem_auto"]["tiles_kept"] = [str(p) for p in keep]

    # Build VRT then warp to working CRS + clip to AOI
    out_dem = dem_cache / f"river_dem_tnm_{_hash_key(cfg.aoi, working_srs, cfg.river_dem_res_m)}.tif"
    if out_dem.exists() and out_dem.stat().st_size > 0:
        log.info("[RIVER][DEM] Using cached river DEM: %s", out_dem)
        return out_dem

    gdalbuildvrt = shutil.which("gdalbuildvrt")
    gdalwarp = shutil.which("gdalwarp")
    if gdalbuildvrt is None or gdalwarp is None:
        log.warning("[RIVER][DEM] gdalbuildvrt/gdalwarp not found; cannot mosaic/warp TNM DEM.")
        report["river"]["dem_auto"]["status"] = "failed_no_gdal"
        return None

    vrt = dem_cache / "tnm_mosaic.vrt"
    try:
        cmd_vrt = [gdalbuildvrt, "-overwrite", str(vrt)] + [str(p) for p in keep]
        log.info("[RIVER][DEM] Build VRT: %s", " ".join(cmd_vrt[:6]) + (" ..." if len(cmd_vrt) > 6 else ""))
        run_cmd(cmd_vrt, check=True)
    except Exception as e:
        log.warning("[RIVER][DEM] gdalbuildvrt failed: %s", e)
        report["river"]["dem_auto"]["status"] = "failed_vrt"
        return None

    # Compute AOI bounds in working CRS for clipping
    try:
        from pyproj import Transformer
        w, e, s, n = [float(x) for x in cfg.aoi.split("/")]
        t = Transformer.from_crs("EPSG:4326", working_srs, always_xy=True)
        xs, ys = t.transform([w, e, w, e], [s, s, n, n])
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
    except Exception:
        minx = miny = maxx = maxy = None

    cmd_warp = [gdalwarp, "-overwrite", "-t_srs", working_srs, "-r", "bilinear",
                "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES"]
    # Force a sensible meter grid
    if cfg.river_dem_res_m and cfg.river_dem_res_m > 0:
        cmd_warp += ["-tr", str(cfg.river_dem_res_m), str(cfg.river_dem_res_m), "-tap"]
    if minx is not None:
        cmd_warp += ["-te", str(minx), str(miny), str(maxx), str(maxy)]
    cmd_warp += [str(vrt), str(out_dem)]

    try:
        log.info("[RIVER][DEM] Warp/clip: %s", " ".join(cmd_warp[:10]) + (" ..." if len(cmd_warp) > 10 else ""))
        run_cmd(cmd_warp, check=True)
        if out_dem.exists() and out_dem.stat().st_size > 0:
            report["river"]["dem_auto"]["status"] = "success"
            return out_dem
    except Exception as e:
        log.warning("[RIVER][DEM] gdalwarp failed: %s", e)
        report["river"]["dem_auto"]["status"] = "failed_warp"

    return None



# -----------------------------------------------------------------------------
# CUDEM dlim auto-soundings helpers
# -----------------------------------------------------------------------------

_CUDEM_XYZ_SOURCE_ALIASES = {
    # NOAA/NOS Hydrographic surveys via CUDEM provider name
    "nos": "hydronos",
    "hydronos": "hydronos",
    # USACE eHydro
    "usace": "ehydro",
    "ehydro": "ehydro",
}


def _normalize_cudem_source_name(src: str) -> str:
    s = (src or "").strip().lower()
    if not s:
        return s
    return _CUDEM_XYZ_SOURCE_ALIASES.get(s, s)


def _reproject_xyz_file(
    xyz_path: Path,
    src_crs: str,
    dst_crs: str,
    cache_dir: Path,
) -> Path:
    """Reproject an XYZ file from src_crs to dst_crs.
    
    Returns path to reprojected file (cached).
    Only reprojects X,Y coordinates; Z (depth) is unchanged.
    """
    from pyproj import Transformer
    import numpy as np
    
    # Normalize CRS strings
    src_crs_str = str(src_crs).upper()
    dst_crs_str = str(dst_crs).upper()
    
    # If same CRS, return original
    # Strip vertical component for comparison (e.g., EPSG:4269+5703 -> EPSG:4269)
    src_h = src_crs_str.split("+")[0] if "+" in src_crs_str else src_crs_str
    dst_h = dst_crs_str.split("+")[0] if "+" in dst_crs_str else dst_crs_str
    if src_h == dst_h:
        return xyz_path
    
    # Build cache key and output path
    key = _hash_key(str(xyz_path), src_crs_str, dst_crs_str)
    out_path = cache_dir / f"{xyz_path.stem}_{dst_h.replace(':', '')}_{key}.xyz"
    
    # Cache hit
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    
    # Read XYZ (space or comma delimited, 3+ columns: x, y, z, ...)
    data = []
    with open(xyz_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    extra = parts[3:] if len(parts) > 3 else []
                    data.append((x, y, z, extra))
                except ValueError:
                    continue
    
    if not data:
        raise ValueError(f"No valid XYZ data in {xyz_path}")
    
    # Transform coordinates (horizontal only)
    xs = np.array([d[0] for d in data])
    ys = np.array([d[1] for d in data])
    zs = np.array([d[2] for d in data])
    extras = [d[3] for d in data]
    
    # Use horizontal CRS for transformation (strip vertical)
    transformer = Transformer.from_crs(src_h, dst_h, always_xy=True)
    xs_out, ys_out = transformer.transform(xs, ys)
    
    # Write output
    tmp_path = out_path.with_suffix(".xyz.tmp")
    with open(tmp_path, "w") as f:
        for i in range(len(xs_out)):
            extra_str = " ".join(extras[i]) if extras[i] else ""
            if extra_str:
                f.write(f"{xs_out[i]:.6f} {ys_out[i]:.6f} {zs[i]:.4f} {extra_str}\n")
            else:
                f.write(f"{xs_out[i]:.6f} {ys_out[i]:.6f} {zs[i]:.4f}\n")
    
    tmp_path.replace(out_path)
    return out_path


def fetch_cudem_soundings_via_dlim(
    *,
    aoi: str,
    sources: List[str],
    cache_root: Path,
    out_crs: str,
    thin_res_m: Optional[float] = 10.0,
    filter_spec: Optional[str] = None,
    force: bool = False,
    prefix: str = "[XYZ][DLIM] ",
) -> Tuple[List[Path], Dict[str, Any]]:
    """Fetch external soundings using CUDEM `dlim` providers.

    Writes one .xyz per source into: <cache_root>/xyz/
    Returns (paths, report_dict). Never raises (best-effort).
    """
    report: Dict[str, Any] = {
        "requested_sources": list(sources),
        "out_crs": out_crs,
        "thin_res_m": thin_res_m,
        "filter_spec": filter_spec,
        "outputs": [],
        "status": "skipped",
    }
    # Fatal post-processing issues that should cause a non-zero exit even if a
    # "final" raster exists (e.g., requested reprojection outputs missing).
    fatal_errors: List[str] = []

    dlim_exe = shutil.which("dlim")
    if dlim_exe is None:
        report["status"] = "skipped"
        report["reason"] = "dlim_not_found"
        log.warning("%sdlim not found on PATH; skipping --extra-xyz-cudem", prefix)
        return [], report

    # WARNING: If output CRS is geographic, block_thin:res is interpreted in *degrees*.
    # For convenience, when the user passes thin_res_m (meters) and did not provide an
    # explicit filter_spec, we convert meters -> degrees using an AOI-center latitude approximation.
    # NOTE: dlim's block_thin takes a single scalar "res" in coordinate units, so we choose a degree
    # value that is conservative in longitude (accounts for cos(lat)).
    out_crs_lower = str(out_crs).lower()
    is_geographic = any(x in out_crs_lower for x in ["4326", "4269", "4267"])
    thin_res_m_eff = thin_res_m
    if is_geographic and thin_res_m and thin_res_m > 0 and (filter_spec is None or str(filter_spec).strip() == ""):
        try:
            w, e, s, n = [float(x) for x in str(aoi).split("/")]
            lat0 = 0.5 * (s + n)
        except Exception:
            lat0 = 0.0
        import math
        coslat = max(0.2, abs(math.cos(math.radians(lat0))))
        meters_per_deg_lon = 111320.0 * coslat
        thin_res_m_eff = float(thin_res_m) / meters_per_deg_lon
        log.warning(
            "%sGeographic output CRS (%s): interpreting --extra-xyz-cudem-thin-res-m=%s m as ~%.8f degrees for dlim block_thin (lat0=%.4f).",
            prefix, out_crs, thin_res_m, thin_res_m_eff, lat0
        )
    elif is_geographic and thin_res_m and thin_res_m > 0:
        log.info(
            "%sNote: Output CRS is geographic (%s). Your explicit filter_spec will be passed to dlim unchanged.",
            prefix, out_crs
        )

    xyz_cache = ensure_dir(Path(cache_root) / "xyz")
    out_paths: List[Path] = []

    for src_raw in sources:
        src = _normalize_cudem_source_name(src_raw)
        if not src:
            continue

        key = _hash_key(
            "dlim",
            src,
            aoi,
            out_crs,
            filter_spec or (f"block_thin:res={thin_res_m_eff}" if thin_res_m_eff else "no_filter"),
        )
        out_xyz = xyz_cache / f"{src}_{key}.xyz"

        # Build dlim command
        # Basic: dlim -R=W/E/S/N <source>
        # With projection: dlim -R=W/E/S/N <source> -P epsg:XXXX
        # With filter: dlim -R=W/E/S/N <source> -F block_thin:res=10
        cmd = [dlim_exe, f'-R={aoi}', src]
        
        # Add projection if specified (dlim defaults to epsg:4326 output)
        if out_crs:
            # dlim -P expects lowercase 'epsg:' format
            crs_str = str(out_crs).lower()
            if not crs_str.startswith("epsg:"):
                crs_str = f"epsg:{crs_str}" if crs_str.isdigit() else crs_str
            cmd += ["-P", crs_str]
        
        # Optional thinning/filtering
        this_filter = filter_spec or (f"block_thin:res={thin_res_m_eff}" if thin_res_m_eff and thin_res_m_eff > 0 else None)
        if this_filter:
            cmd += ["-F", this_filter]
        
        cmd_str = " ".join(str(c) for c in cmd)

        # Cache hit
        if out_xyz.exists() and out_xyz.stat().st_size > 0 and not force:
            log.info("%sCache hit for %s: %s", prefix, src, str(out_xyz))
            out_paths.append(out_xyz)
            report["outputs"].append({
                "source": src,
                "path": str(out_xyz),
                "status": "cached",
                "command": cmd_str,
            })
            continue

        # (Re)download
        log.info("%sFetching %s via dlim -> %s", prefix, src, str(out_xyz))
        log.info("%sCommand: %s", prefix, cmd_str)
        stderr_tail = ""
        rc = 999
        try:
            # Write dlim stdout to an atomic temp file alongside the target.
            rc, stderr_tail = run_command_stdout_to_file(cmd, out_xyz, prefix=prefix, tail_chars=4000)
            tmp_path = out_xyz.with_suffix(out_xyz.suffix + ".tmp")

            # Basic sanity: non-empty file
            file_size = tmp_path.stat().st_size if tmp_path.exists() else 0
            if rc != 0 or file_size == 0:
                if tmp_path.exists():
                    tmp_path.unlink()
                if rc == 0 and file_size == 0:
                    log.warning("%sNo data returned for %s (file is empty, rc=0).", prefix, src)
                    log.warning("%sThis could mean: (1) no data in AOI, (2) filter too aggressive, or (3) projection issue.", prefix)
                    if stderr_tail:
                        log.warning("%sdlim stderr: %s", prefix, stderr_tail.strip()[-500:])
                    status = "no_data"
                else:
                    log.warning("%sFailed fetch for %s (rc=%s).", prefix, src, rc)
                    if stderr_tail:
                        log.warning("%sdlim stderr: %s", prefix, stderr_tail.strip()[-500:])
                    status = "failed"
                report["outputs"].append({
                    "source": src,
                    "path": str(out_xyz),
                    "status": status,
                    "returncode": rc,
                    "command": cmd_str,
                    "stderr_tail": stderr_tail,
                })
                continue

            # Atomic move into cache
            tmp_path.replace(out_xyz)

            out_paths.append(out_xyz)
            report["outputs"].append({
                "source": src,
                "path": str(out_xyz),
                "status": "downloaded",
                "returncode": rc,
                "command": cmd_str,
            })

        except Exception as e:
            try:
                if out_xyz.exists() and out_xyz.stat().st_size == 0:
                    out_xyz.unlink()
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
            log.warning("%sException fetching %s via dlim: %s", prefix, src, str(e))
            report["outputs"].append({
                "source": src,
                "path": str(out_xyz),
                "status": "failed",
                "returncode": rc,
                "command": cmd_str,
                "stderr_tail": stderr_tail,
                "exception": str(e),
            })

    if out_paths:
        report["status"] = "success"
    elif report["outputs"]:
        report["status"] = "failed"
    else:
        report["status"] = "skipped"

    return out_paths, report


def _append_river_soundings_args(cmd, cfg, *, include_calib_args=True, include_mode_args=False):
    """Append soundings (extra XYZ) args for river inference scripts.

    cfg.river_soundings is a comma-separated list of files, already in working CRS.
    - xs_infer_bathy_raster.py supports calibration args (--calib-*) and --soundings-crs
    - river_skeleton_bathy.py supports soundings mode/weighting args (--soundings-mode, etc.)
    """
    try:
        snd = getattr(cfg, 'river_soundings', None)
        if not snd:
            return
        snd_list = [s.strip() for s in str(snd).split(',') if s.strip()]
        if not snd_list:
            return

        # Common soundings file args
        # xs_infer_bathy_raster.py defines --soundings as action='append' (one argument per flag),
        # while river_skeleton_bathy.py defines --soundings with nargs='*' (many after one flag).
        # Use the CLI shape that matches the target script to avoid stray positional args.
        if include_calib_args and not include_mode_args:
            for _snd in snd_list:
                cmd.append(f"--soundings={_snd}")
        else:
            cmd.extend(['--soundings'] + snd_list)

        # XY are already in working CRS (reprojected earlier when needed).
        work_srs = str(getattr(cfg, 'working_srs', '') or '').strip()
        if work_srs:
            cmd.append(f"--soundings-crs={work_srs}")

        if include_calib_args:
            try:
                d = float(getattr(cfg, 'river_soundings_calib_max_dist_m', 0.0) or 0.0)
                if d > 0:
                    cmd.append(f"--calib-max-dist-m={d}")
            except Exception:
                pass
            try:
                st = str(getattr(cfg, 'river_soundings_calib_stat', '') or '').strip()
                if st:
                    cmd.append(f"--calib-stat={st}")
            except Exception:
                pass

        if include_mode_args:
            # Skeleton-side sounding assimilation controls
            try:
                sm = str(getattr(cfg, 'river_soundings_mode', 'auto') or 'auto').strip()
                if sm:
                    cmd.append(f"--soundings-mode={sm}")
            except Exception:
                pass
            try:
                sp = float(getattr(cfg, 'river_soundings_cell_percentile', 25.0) or 0.0)
                if sp > 0:
                    cmd.append(f"--soundings-cell-percentile={sp}")
            except Exception:
                pass
            try:
                md = float(getattr(cfg, 'river_soundings_max_dist_m', 150.0) or 0.0)
                if md > 0:
                    cmd.append(f"--soundings-max-dist-m={md}")
            except Exception:
                pass
            try:
                mr = float(getattr(cfg, 'river_soundings_min_r', 0.15) or 0.0)
                if mr > 0:
                    cmd.append(f"--soundings-min-r={mr}")
            except Exception:
                pass
            if bool(getattr(cfg, 'river_no_soundings_enforce', False)):
                cmd.append('--no-soundings-enforce')
    except Exception:
        logging.getLogger(__name__).debug('Failed to append river soundings args; continuing.', exc_info=True)


def _score_sdb_candidate(tif: Path) -> float:
    """
    Heuristic scoring to pick the best SDB depth raster.
    Higher = better.
    """
    name = tif.name.lower()

    # Exclude obvious non-products
    bad_tokens = ["rgb", "mask", "land", "clear", "qa", "scl", "doa", "weights", "uncert", "error", "diff"]
    if any(tok in name for tok in bad_tokens):
        return -1.0

    # Prefer typical product tokens
    good = 0.0
    if "sdb" in name:
        good += 5.0
    if "rf" in name:
        good += 3.0
    if "depth" in name or "bathy" in name:
        good += 2.0
    if "10m" in name:
        good += 1.0
    if "aligned" in name or "final" in name or "product" in name:
        good += 1.0

    # Prefer bigger/newer files (often the actual raster product)
    try:
        size = tif.stat().st_size
        mtime = tif.stat().st_mtime
        good += min(size / 1e8, 5.0)   # cap size influence
        good += min(mtime / 1e10, 5.0) # small nudge for recency
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    return good


def _is_depth_raster_like(path: Path, expect_negative: bool = True) -> tuple[bool, str]:
    """Quick guardrail to prevent mistaking optical imagery/masks for bathymetry.

    Returns (ok, reason). This is intentionally lightweight and uses a small decimated read.
    """
    try:
        import rasterio
        import numpy as np
        from rasterio.enums import Resampling

        if not path.exists():
            return False, "missing"
        with rasterio.open(path) as ds:
            if int(ds.count) != 1:
                return False, f"band_count={ds.count}"
            # Prefer float depth products; integer-only is usually masks/reflectance
            if str(ds.dtypes[0]).startswith(("uint", "int")):
                # Allow int only if values look like small signed depths (rare)
                pass

            h, w = int(ds.height), int(ds.width)
            out_h = min(256, h)
            out_w = min(256, w)
            arr = ds.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest,
            ).astype("float32")

            nod = ds.nodata
            m = np.isfinite(arr)
            if nod is not None:
                m &= (arr != float(nod))
            if not np.any(m):
                # Could be a valid depth raster with no predictions (all nodata)
                return True, "all_nodata"

            vals = arr[m]
            p1, p50, p99 = np.percentile(vals, [1, 50, 99])

            # Depth magnitudes should not be enormous. If they are, this is likely reflectance/mask encoded.
            if np.abs(p99) > 500.0:
                return False, f"p99_abs_too_large={float(np.abs(p99)):.2f}"

            # Optical reflectance products are typically non-negative; depths should be mostly negative-down
            if expect_negative:
                frac_neg = float(np.mean(vals < 0))
                if frac_neg < 0.01:
                    # If everything is small non-negative (0..1 or 0..10000), it's almost certainly imagery/mask
                    if float(p1) >= -1e-6:
                        return False, f"too_few_negative(frac_neg={frac_neg:.3f}, p1={float(p1):.3f}, p99={float(p99):.3f})"

            return True, f"ok(p50={float(p50):.3f}, p99={float(p99):.3f})"
    except Exception as e:
        return False, f"exception:{e}"

def find_sdb_depth_raster(sdb_dir: Path) -> Optional[Path]:
    """Locate the SDB depth product raster under *sdb_dir* using explicit manifests only.

    No filename guessing and no directory scanning is allowed.

    Resolution order:
      1) <sdb_dir>/artifacts_sdb.json : explicit artifact manifest written by sdb_main.py

    Returns None if the artifact is not present (e.g., SDB was skipped for this AOI).
    """
    if not sdb_dir.exists():
        return None

    manifest = sdb_dir / "artifacts_sdb.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            rel = data.get("depth_raster", None)
            if isinstance(rel, str) and rel.strip():
                p = (sdb_dir / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
                if p.exists():
                    return p
        except Exception:
            logging.getLogger(__name__).debug("Failed to read SDB artifact manifest.", exc_info=True)

    return None
def find_sdb_land_mask(sdb_dir: Path) -> Optional[Path]:
    """Locate the aligned land mask produced by sdb_main.py using explicit artifacts.

    This function is intentionally *no-guess*: it only uses the SDB artifact manifest
    (<sdb_dir>/artifacts_sdb.json). If the land mask is not recorded there, returns None.
    """
    if not sdb_dir.exists():
        return None
    manifest = sdb_dir / "artifacts_sdb.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        rel = data.get("land_mask", None)
        if isinstance(rel, str) and rel.strip():
            p = (sdb_dir / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
            if p.exists():
                return p
    except Exception:
        logging.getLogger(__name__).debug("Failed to read SDB artifact manifest for land_mask.", exc_info=True)
    return None

def find_sdb_wse_navd88_raster(sdb_dir: Path) -> Optional[Path]:
    """Locate an SDB water-surface elevation raster (NAVD88) using explicit artifacts only.

    No-guess policy: only consult <sdb_dir>/artifacts_sdb.json. If not present, return None.
    """
    if not sdb_dir.exists():
        return None
    manifest = sdb_dir / "artifacts_sdb.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        rel = data.get("wse_navd88", None) or data.get("water_surface_navd88", None)
        if isinstance(rel, str) and rel.strip():
            p = (sdb_dir / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
            if p.exists():
                return p
    except Exception:
        logging.getLogger(__name__).debug("Failed to read SDB artifact manifest for WSE.", exc_info=True)
    return None


def run_sdb(cfg: BathyConfig, report: Dict[str, Any]) -> Optional[Path]:
    log.info("=" * 60)
    log.info("RUNNING SDB PIPELINE (Coastal/Nearshore)")
    log.info("=" * 60)

    sdb_dir = ensure_dir(cfg.out_dir / "sdb")

    # Early skip: if waffles coastline mask indicates no ocean pixels in this AOI, skip SDB entirely.
    # This avoids unnecessary S2/ATL acquisition for inland tiles.
    waffles_mask = (
        None
        or None
    )
    if waffles_mask and waffles_mask.exists():
        n_water = _count_mask_water_pixels(waffles_mask, cfg.aoi)
        if n_water == 0:
            logging.info(f"[SDB] Early-skip: waffles coastline mask has 0 ocean/water pixels in AOI; skipping SDB. mask={waffles_mask}")
            report.setdefault('sdb', {})['skipped_reason'] = 'no_ocean_pixels'
            return None
    script_dir = Path(__file__).parent

    cmd = [
        sys.executable, "sdb_main.py",
        f"--aoi={cfg.aoi}",
        f"--start={cfg.start_date}",
        f"--end={cfg.end_date}",
        f"--out-dir={sdb_dir}",
        f"--cloud={cfg.cloud}",
        f"--icesat={cfg.icesat}",
        f"--sdb-mode={cfg.sdb_mode}",
        f"--cache-root={cfg.cache_root}",
        f"--align-mode={cfg.align_mode}",
        f"--working-srs={cfg.working_srs}",
        f"--working-vcrs-epsg={cfg.working_vcrs_epsg}",
        # NOTE: --atl-transform-datum is NOT passed by default.
        # ATL depths are relative (not geodetic heights), so vertical transforms would corrupt them.
        # The horizontal transform (WGS84 -> NAD83) is tiny and usually unnecessary.
    ]

    # Bounded model bank (reservoir) + periodic retrain (DEFAULT)
    try:
        if bool(getattr(cfg, 'sdb_model_bank_enabled', True)):
            cmd.append(f"--model-bank={getattr(cfg, 'sdb_model_bank', 'auto')}")
            cmd.append(f"--bank-max-samples={int(getattr(cfg, 'sdb_bank_max_samples', 100000))}")
            cmd.append(f"--bank-seed={int(getattr(cfg, 'sdb_bank_seed', 1337))}")
            cmd.append(f"--bank-retrain-min-new={int(getattr(cfg, 'sdb_bank_retrain_min_new', 2000))}")
        else:
            cmd.append("--no-model-bank")
    except Exception:
        log.debug('Unexpected exception suppressed (model bank policy).', exc_info=True)

    # Deprecated regional model cache passthrough (kept for compatibility; disabled by default)
    try:
        if bool(getattr(cfg, "sdb_model_cache_enabled", False)):
            cmd.append(f"--model-cache-key={getattr(cfg, 'sdb_model_cache_key', 'auto')}")
        else:
            cmd.append("--no-model-cache")
    except Exception:
        log.debug('Unexpected exception suppressed (model cache policy).', exc_info=True)

    # Forward S2 sun-glint correction flags into SDB, if supported.
    if getattr(cfg, "glint_correct", False):
        sdb_main_path = (Path(__file__).parent / "sdb_main.py")
        supports_glint = False
        try:
            if sdb_main_path.exists():
                txt = sdb_main_path.read_text(encoding="utf-8", errors="ignore")
                supports_glint = ("--glint-correct" in txt) or ("glint_correct" in txt)
        except Exception:
            supports_glint = False

        if not supports_glint:
            log.warning("[SDB][GLINT] --glint-correct requested, but sdb_main.py does not appear to support glint flags; skipping glint passthrough.")
        else:
            cmd.append("--glint-correct")
            cmd.append(f"--glint-nir-band={cfg.glint_nir_band}")
            cmd.append(f"--glint-vis-bands={cfg.glint_vis_bands}")
            cmd.append(f"--glint-nir-min-percentile={cfg.glint_nir_min_percentile}")
            cmd.append(f"--glint-deepwater-b02-max={cfg.glint_deepwater_b02_max}")
            cmd.append(f"--glint-min-samples={cfg.glint_min_samples}")
            cmd.append(f"--glint-max-samples={cfg.glint_max_samples}")
            cmd.append(f"--glint-clip-min={cfg.glint_clip_min}")

    # Forward extra XYZ bathymetry into SDB training/fusion if provided.
    # cfg.river_soundings is a comma-separated list of files produced by --extra-xyz normalization.
    if cfg.river_soundings:
        xyz_list = [p.strip() for p in str(cfg.river_soundings).split(",") if p.strip()]
        if xyz_list:
            cmd += ["--extra-xyz"] + xyz_list
            cmd.append(f"--extra-xyz-crs={cfg.extra_xyz_crs}")
    
    # Forward adaptive sampling parameters (v0.7.0+, enabled by default v0.7.1)
    enable_sampling = getattr(cfg, 'enable_adaptive_sampling', True)
    if not enable_sampling:
        # User explicitly disabled it
        cmd.append("--disable-adaptive-sampling")
    
    # Always pass the tuning parameters (even if disabled, sdb_main will ignore them)
    cmd.append(f"--sampling-target-points={getattr(cfg, 'sampling_target_points', 2000)}")
    cmd.append(f"--sampling-min-threshold={getattr(cfg, 'sampling_min_threshold', 3000)}")
    cmd.append(f"--sampling-max-gap-m={getattr(cfg, 'sampling_max_gap_m', 100.0)}")

    log.info(f"[SDB] Command: {cmd}")
    logs_dir = ensure_dir(cfg.out_dir / "logs")
    rc, out, err = run_command(
        cmd,
        cwd=script_dir,
        prefix="[SDB] ",
        stdout_log_path=logs_dir / "sdb.stdout.log",
        stderr_log_path=logs_dir / "sdb.stderr.log",
    )

    report["sdb"] = {
        "status": "success" if rc == 0 else "failed",
        "returncode": rc,
        "command": cmd,
        "stdout_tail": out,
        "stderr_tail": err,
    }

    if rc != 0:
        log.error(f"[SDB] Failed with code {rc}")
        return None

    depth = find_sdb_depth_raster(sdb_dir)
    if depth is None:
        log.warning("[SDB] Completed but could not find a depth raster in SDB output tree.")
        return None

    log.info(f"[SDB] Depth raster found: {depth}")
    try:
        apply_depth_metadata(Path(depth))
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    return Path(depth).resolve() if not isinstance(depth, Path) else depth.resolve()


# -----------------------------------------------------------------------------
# River
# -----------------------------------------------------------------------------

def run_river(cfg: BathyConfig, report: Dict[str, Any]) -> Optional[Path]:
    log.info("=" * 60)
    # Ensure report structure exists (avoid KeyError if caller created only partial report dict)
    if report is not None:
        report.setdefault("river", {})
        report["river"].setdefault("steps", {})
    log.info("RUNNING RIVER PIPELINE (method=%s)", getattr(cfg, "river_method", "xs"))
    log.info("=" * 60)

    if cfg.river_dem is None:
        auto_dem = ensure_river_dem_auto(cfg, report)
        if auto_dem is None:
            log.error("[RIVER] Missing --river-dem (and auto-build failed)")
            report["river"] = {"status": "failed", "reason": "missing river_dem"}
            return None
        cfg.river_dem = auto_dem

    river_dir = ensure_dir(cfg.out_dir / "river")
    script_dir = Path(__file__).parent

    # NOTE: We allow reuse of downloaded/raw inputs via cache_root, but we do NOT reuse derived
    # river products across runs. All river derived outputs are run-scoped under derived_cache_root.
    cache_dir = Path(cfg.derived_cache_root) / "river"
    ensure_dir(cache_dir)
    work_dir = ensure_dir(cache_dir / "work")
    raw_hydro_cache = ensure_dir(Path(cfg.cache_root) / "hydrography")

    # Cached (run-scoped) river bed raster path used by downstream steps
    cached_bed_tif = cache_dir / "river_bed_elev_cached.tif"

    # Cached (run-scoped) river depth raster (negative-down, relative to DEM terrain surface)
    cached_depth_tif = cache_dir / "river_depth_cached.tif"

    # Run-scoped manifest (for debugging / provenance only). Not used for cache hits.
    manifest = cache_dir / "_manifest.json"
    payload = {
        "run_id": getattr(cfg, "run_id", None),
        "aoi_tile": cfg.aoi_tile,
        # In this codebase, cfg.aoi is the *processing* AOI (may be expanded beyond the tile).
        # cfg.aoi_tile is the *tile* AOI (final clipping target).
        "aoi_data": cfg.aoi,
        "start": cfg.start_date,
        "end": cfg.end_date,
        "working_srs": str(cfg.working_srs),
        "river_method": cfg.river_method,
        "river_dem_source": cfg.river_dem_source,
        "extra_xyz_cudem": cfg.extra_xyz_cudem,
        "timestamp": "run_scoped",
    }
    try:
        manifest.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass
    # Step 1: river network
    log.info("[RIVER] Step 1: Extracting river network...")
    network_gpkg = work_dir / "river_network.gpkg"

    # River network provenance lock (deterministic key; prevents silent dataset drift)
    prov_dir = Path(cfg.cache_root) / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    # Canonicalize AOI string for provenance key (stable float formatting)
    aoi_key = str(cfg.aoi)
    try:
        _parts = aoi_key.split('/')
        if len(_parts) >= 4:
            _vals = [float(_parts[0]), float(_parts[1]), float(_parts[2]), float(_parts[3])]
            aoi_key = '/'.join([f"{v:.6f}" for v in _vals])
    except Exception:
        pass
    prov_key = {
        "aoi": aoi_key,
        "hydrography_source": str(cfg.river_hydrography_source),
        "tnm_dataset": str(cfg.tnm_dataset),
        "tnm_enable": bool(cfg.tnm_enable),
        "snap_m": float(cfg.snap_m),
    }
    prov_hash = hashlib.sha1(json.dumps(prov_key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    river_network_lock = prov_dir / f"river_network_lock_{prov_hash}.json"


    # SECURITY FIX: Use list-based command construction (no shell injection)
    cmd = [
        sys.executable, "river_network.py",
        f"--aoi={cfg.aoi}",
        f"--cache-dir={raw_hydro_cache}",
        f"--out-gpkg={network_gpkg}",
        f"--provenance-lock={river_network_lock}",
        f"--hydrography-source={cfg.river_hydrography_source}",
        f"--tnm-dataset={cfg.tnm_dataset}",
        f"--snap-m={cfg.snap_m}",
    ]
    if cfg.tnm_enable:
        cmd.append("--tnm-enable")

    rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
    cmd_str = " ".join(str(c) for c in cmd)  # For logging only
    report["river"]["steps"]["network"] = {
        "status": "success" if rc == 0 else "failed",
        "returncode": rc,
        "command": cmd_str,
        "stdout_tail": out,
        "stderr_tail": err,
    }
    if rc != 0 or not network_gpkg.exists():
        log.error("[RIVER] Failed to extract river network.")
        report["river"]["status"] = "failed"
        return None

    log.info(f"[RIVER] Network extracted: {network_gpkg}")


    river_method = str(getattr(cfg, "river_method", "hybrid")).lower().strip()
    

    # Hybrid default: XS only on mainstem (order>=threshold + largest component), skeleton elsewhere.
    # Rationale: XS is most defensible on mainstem and most failure-prone at dense tributary junctions.
    # This preserves continuity along the mainstem while avoiding tributary overlap artifacts.

    def _build_domain_masks(_work_dir: Path, *, strict: bool = False):
        """Build channel/open-water/mainstem masks (DEM-aligned).

        If strict=True, missing channel mask is treated as fatal.
        """
        channel_mask_tif = _work_dir / "river_channel_mask.tif"
        open_water_mask_tif = _work_dir / "open_water_mask.tif"
        mainstem_mask_tif = _work_dir / "mainstem_mask.tif"

        # Reuse WAFFLES coastline masks across runs for the same AOI + resolution + module params.
        cache_masks_shared = Path(getattr(cfg, "cache_root", ".")).resolve() / "masks"
        ensure_dir(cache_masks_shared)
        cache_masks_run = Path(cfg.derived_cache_root) / "masks"
        ensure_dir(cache_masks_run)
        aoi_buf = str(cfg.aoi)
        inc_arcsec = float(getattr(cfg, "waffles_inc_arcsec", 1.0) or 1.0)

        ocean_mask = None
        with_nhd_mask = None
        try:
            ocean_cache = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks_shared,
                aoi=aoi_buf,
                inc_arcsec=inc_arcsec,
                want_nhd=False,
                want_lakes=False,
                prefix="waffles_coastline_ocean_only",
                log=log,
                force=False,
            )
            ocean_mask = _stage_cached_waffles_mask(ocean_cache, cache_masks_run / "waffles_coastline_ocean_only.tif", log=log)
        except Exception as e:
            log.warning(f"[WAFFLES] Ocean-only mask unavailable; ocean bleed protection degraded: {e}")
            ocean_mask = None

        try:
            nhd_cache = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks_shared,
                aoi=aoi_buf,
                inc_arcsec=inc_arcsec,
                want_nhd=True,
                want_lakes=False,
                prefix="waffles_coastline_with_nhd",
                log=log,
                force=False,
            )
            with_nhd_mask = _stage_cached_waffles_mask(nhd_cache, cache_masks_run / "waffles_coastline_with_nhd.tif", log=log)
        except Exception as e:
            log.warning(f"[WAFFLES] With-NHD mask unavailable (TNM flaky?): {e}. Proceeding with corridor+ArcGIS NHD flowlines + ocean-only mask.")
            with_nhd_mask = None


        if strict:
            # River domain building relies on a WAFFLES water mask to prevent ocean/land bleed
            # and to make downstream clipping deterministic. Fail closed if we can't get one.
            if (with_nhd_mask is None) or (not Path(with_nhd_mask).exists()):
                raise RuntimeError("Strict river domain build requested but WAFFLES with-NHD water mask is unavailable. Fix WAFFLES generation first.")
        cmd = [
            sys.executable, "river_domain_mask.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-channel-mask={channel_mask_tif}",
            f"--out-open-water-mask={open_water_mask_tif}",
            f"--out-mainstem-mask={mainstem_mask_tif}",
            f"--channel-buffer-m={getattr(cfg, 'river_channel_buffer_m', 400.0)}",
            f"--max-channel-width-m={getattr(cfg, 'river_max_channel_width_m', 600.0)}",
            f"--mainstem-min-order={getattr(cfg, 'river_mainstem_min_order', 5)}",
            f"--max-mainstem-width-m={getattr(cfg, 'river_max_mainstem_width_m', 2500.0)}",
        ]

        chan_src = str(getattr(cfg, 'river_channel_source', 'auto') or 'auto').strip().lower()
        if chan_src not in ('auto', 'nhdarea', 'corridor'):
            log.warning(f"[RIVER] Unknown river_channel_source='{chan_src}', defaulting to 'auto'.")
            chan_src = 'auto'
        cmd.append(f"--channel-source={chan_src}")

        nhd_allow = getattr(cfg, 'river_nhdarea_allow_ftype', None)
        if nhd_allow is None:
            nhd_allow = "460"
        cmd.append(f"--nhdarea-allow-ftype={nhd_allow}")
        nhd_allow_fcode = getattr(cfg, 'river_nhdarea_allow_fcode', None)
        if nhd_allow_fcode:
            cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")

        if (chan_src in ('auto', 'nhdarea')) and bool(getattr(cfg, "river_use_nhdarea", True)):
            cmd.append(f"--nhdarea-gpkg={network_gpkg}")
            cmd.append(f"--nhdarea-layer={getattr(cfg, 'river_nhdarea_layer', 'nhdarea_clip')}")

        if ocean_mask and Path(ocean_mask).exists():
            cmd.append(f"--ocean-mask={ocean_mask}")
        oke = float(getattr(cfg, 'river_ocean_keep_dist_m', 0.0) or 0.0)
        if (oke > 0.0):
            cmd.append(f"--ocean-keep-dist-m={oke}")

        if with_nhd_mask and Path(with_nhd_mask).exists():
            cmd.append(f"--water-mask={with_nhd_mask}")

        # Persist the waffles mask used (for downstream final clipping/QA).
        try:
            wm = None
            if with_nhd_mask and Path(with_nhd_mask).exists():
                wm = with_nhd_mask
            elif ocean_mask and Path(ocean_mask).exists():
                wm = ocean_mask
            report.setdefault("river", {}).setdefault("outputs", {})["waffles_water_mask"] = str(wm) if wm else None
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["domain_mask"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or (not channel_mask_tif.exists()):
            msg = "[RIVER] Failed to build channel mask"
            if strict:
                log.error("%s (strict mode).", msg)
                raise RuntimeError("River domain mask generation failed in strict mode; aborting.")
            else:
                log.warning("%s; continuing without hard domain constraint.", msg)
        else:
            # Validate that the channel mask is actually usable and overlaps the template raster.
            try:
                import rasterio
                import numpy as np
                from rasterio.windows import Window

                def _count_inside_pixels(mask_path: Path, inside_val: int = 1) -> int:
                    with rasterio.open(mask_path) as ds:
                        nod = ds.nodata
                        total = 0
                        for _, w in ds.block_windows(1):
                            a = ds.read(1, window=w)
                            if nod is not None:
                                a = a[a != nod]
                            total += int(np.count_nonzero(a == inside_val))
                        return total

                with rasterio.open(cfg.river_dem) as tmpl, rasterio.open(channel_mask_tif) as cm:
                    if tmpl.crs and cm.crs and (tmpl.crs != cm.crs):
                        raise RuntimeError(f"river_channel_mask CRS mismatch vs template DEM: {cm.crs} != {tmpl.crs}")
                    # basic overlap sanity
                    tb = tmpl.bounds
                    mb = cm.bounds
                    if (mb.right <= tb.left) or (mb.left >= tb.right) or (mb.top <= tb.bottom) or (mb.bottom >= tb.top):
                        raise RuntimeError("river_channel_mask does not overlap template DEM extent (likely reprojection/extent bug).")

                n_inside = _count_inside_pixels(channel_mask_tif, inside_val=1)
                if n_inside <= 0:
                    raise RuntimeError("river_channel_mask has zero inside pixels; river outputs would be all nodata. Fix corridor/NHDArea inputs or AOI.")
                report.setdefault("river", {}).setdefault("masking", {})["channel_mask_inside_pixels"] = int(n_inside)
            except Exception as e:
                if strict:
                    log.error("[RIVER] Channel mask validation failed (strict): %s", e)
                    raise
                log.warning("[RIVER] Channel mask validation warning (non-strict): %s", e)

            try:
                cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        return channel_mask_tif, open_water_mask_tif, mainstem_mask_tif


    def _combine_mainstem_xs_and_skeleton(
        bed_xs: Path,
        bed_skel: Path,
        mainstem_mask: Path,
        out_bed: Path,
        template: Path,
        nodata: float,
    ) -> None:
        """Hybrid combine: use XS where available inside mainstem mask; otherwise skeleton.

        This guarantees a continuous mainstem bed even if XS has gaps (fallback to skeleton).
        Output is aligned to template grid.
        """
        import numpy as np
        import rasterio
        from rasterio.warp import reproject, Resampling

        def _read_align(src_path: Path, band: int = 1, resamp=Resampling.nearest):
            with rasterio.open(template) as tmpl:
                prof = tmpl.profile.copy()
                prof.update(dtype="float32", nodata=nodata, count=1, compress="deflate")
                arr = np.full((tmpl.height, tmpl.width), nodata, dtype=np.float32)
                with rasterio.open(src_path) as src:
                    reproject(
                        source=rasterio.band(src, band),
                        destination=arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tmpl.transform,
                        dst_crs=tmpl.crs,
                        resampling=resamp,
                        src_nodata=src.nodata,
                        dst_nodata=nodata,
                    )
            return arr, prof

        xs_a, prof = _read_align(bed_xs, resamp=Resampling.nearest)
        sk_a, _ = _read_align(bed_skel, resamp=Resampling.nearest)
        ms_a, _ = _read_align(mainstem_mask, resamp=Resampling.nearest)
        ms = ms_a > 0.5

        out = sk_a.copy()
        # Guard against NaNs propagating into the combined raster.
        xs_ok = np.isfinite(xs_a) & (xs_a != nodata)
        take = ms & xs_ok
        out[take] = xs_a[take]

        out_bed.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_bed, "w", **prof) as dst:
            dst.write(out.astype(np.float32), 1)


    if river_method == "hybrid":
        log.info("[RIVER] Using HYBRID method: XS(mainstem) + skeleton(elsewhere)")

        # Step 2: build domain masks (includes mainstem corridor)
        log.info("[RIVER] Step 2: Building river channel/mainstem masks (hybrid)...")
        channel_mask_tif, open_water_mask_tif, mainstem_mask_tif = _build_domain_masks(work_dir, strict=True)

        # Hybrid requires a valid channel mask (skeleton method depends on it). Fail closed if it is missing.
        if channel_mask_tif is None or (not Path(channel_mask_tif).exists()):
            log.error("[RIVER] Hybrid requires a valid channel mask, but it was not created: %s", str(channel_mask_tif))
            report["river"]["status"] = "failed"
            return None

        # If the mainstem mask is empty/missing, we can still run XS builder (largest component),
        # but XS will not be injected into the final hybrid raster; warn loudly.
        if mainstem_mask_tif is None or (not Path(mainstem_mask_tif).exists()):
            log.warning("[RIVER] Mainstem mask not created; hybrid will effectively behave like skeleton-only (XS not injected).")

        # Step 3a: XS on mainstem only
        log.info("[RIVER] Step 3a: Generating cross-sections (mainstem only, conservative defaults)...")
        xs_gpkg = work_dir / "cross_sections_mainstem.gpkg"

        cmd = [
            sys.executable, "xs_builder.py",
            f"--river-gpkg={network_gpkg}",
            f"--dem={cfg.river_dem}",
            f"--out-gpkg={xs_gpkg}",
            f"--spacing-m={cfg.xs_spacing_m}",
            f"--half-width-m={cfg.xs_length_m / 2.0}",
            f"--smoothing-window-m={getattr(cfg, 'xs_smoothing_window_m', 0.0)}",
            f"--deconflict-tol-m={getattr(cfg, 'xs_deconflict_tol_m', 2.0)}",
            f"--junction-snap-m={getattr(cfg, 'xs_junction_snap_m', 30.0)}",
            f"--junction-buffer-m={getattr(cfg, 'xs_junction_buffer_m', 75.0)}",
            f"--densify-step-m={getattr(cfg, 'xs_densify_step_m', 20.0)}",
            f"--min-stream-order={int(getattr(cfg, 'river_mainstem_min_order', 5))}",
            "--keep-top-components=1",
        ]
        if not bool(getattr(cfg, "xs_trim_overlaps", True)):
            cmd.append("--no-trim-overlaps")
        if not bool(getattr(cfg, "xs_global_deconflict", True)):
            cmd.append("--no-global-deconflict")
        if not bool(getattr(cfg, "xs_skip_junctions", True)):
            cmd.append("--no-skip-junctions")

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["xs_builder_mainstem"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not xs_gpkg.exists():
            log.error("[RIVER] Failed to build mainstem cross-sections for hybrid method.")
            report["river"]["status"] = "failed"
            return None

        log.info(f"[RIVER] Mainstem cross-sections generated: {xs_gpkg}")

        log.info("[RIVER] Step 3b: Inferring mainstem bathymetry from XS...")
        bed_xs_tif = work_dir / "river_bed_elev_xs_mainstem.tif"
        bathy_gpkg = work_dir / "river_bathy_xs_mainstem.gpkg"
        xs_mainstem_meta_json = work_dir / "xs_mainstem_constraints_meta.json"
        xs_mainstem_acct_json = work_dir / "xs_mainstem_constraints_accounting.json"

        cmd = [
            sys.executable, "xs_infer_bathy_raster.py",
            f"--xs-gpkg={xs_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-gpkg={bathy_gpkg}",
            f"--out-bathy-raster={bed_xs_tif}",
            f"--out-meta-json={xs_mainstem_meta_json}",
            f"--out-accounting-json={xs_mainstem_acct_json}",
            f"--continuous={getattr(cfg, 'river_continuous', 'walid_aniso')}",
            f"--continuous-k={getattr(cfg, 'river_continuous_k', 12)}",
            f"--idw-power={getattr(cfg, 'river_idw_power', 2.0)}",
            f"--aniso-along-scale-m={getattr(cfg, 'river_aniso_along_scale_m', 500.0)}",
            f"--aniso-cross-scale-m={getattr(cfg, 'river_aniso_cross_scale_m', 30.0)}",
            f"--thalweg-weight={getattr(cfg, 'river_thalweg_weight', 6.0)}",
            f"--nodata={getattr(cfg, 'river_nodata', -9999.0)}",
            f"--overlap-reducer={getattr(cfg, 'river_overlap_reducer', 'min')}",
        ]
        if channel_mask_tif is not None and Path(channel_mask_tif).exists():
            cmd.append(f"--channel-mask-raster={channel_mask_tif}")
            cmd.append("--channel-mask-inside-value=1")
        cmd.append(f"--river-gpkg={network_gpkg}")
        cmd.append(f"--prior-mode={cfg.river_prior_mode}")
        cmd.append(f"--mv-a0={cfg.river_mv_a0}")
        cmd.append(f"--mv-bw={cfg.river_mv_bw}")
        cmd.append(f"--mv-ba={cfg.river_mv_ba}")
        cmd.append(f"--mv-bs={cfg.river_mv_bs}")
        cmd.append(f"--mv-eps-a={cfg.river_mv_eps_a}")
        cmd.append(f"--mv-eps-s={cfg.river_mv_eps_s}")
        cmd.append(f"--slope-proxy-window={cfg.river_slope_proxy_window}")
        cmd.append(f"--slope-min={cfg.river_slope_min}")
        cmd.append(f"--slope-max={cfg.river_slope_max}")
        cmd.append(f"--slope-proxy-min-n={cfg.river_slope_proxy_min_n}")
        if bool(getattr(cfg, "river_wse_profile_enabled", True)):
            cmd.append("--wse-profile-enabled")
        else:
            cmd.append("--no-wse-profile")
        cmd.append(f"--wse-profile-window={getattr(cfg, 'river_wse_profile_window', cfg.river_slope_proxy_window)}")
        cmd.append(f"--wse-profile-min-n={getattr(cfg, 'river_wse_profile_min_n', cfg.river_slope_proxy_min_n)}")
        if bool(getattr(cfg, "river_wse_profile_monotonic", True)):
            cmd.append("--wse-profile-monotonic")
        else:
            cmd.append("--no-wse-profile-monotonic")

        # Pass authoritative point soundings (e.g., extra_xyz subsets) into XS inference anchoring when available.
        if (not getattr(cfg, "river_soundings", None)) and getattr(cfg, "extra_xyz_files", None):
            try:
                _fallback_soundings = [str(p) for p in (cfg.extra_xyz_files or []) if p]
                if _fallback_soundings:
                    cfg.river_soundings = _fallback_soundings
                    log.info("[RIVER] XS inference: using extra_xyz fallback as soundings anchors (n=%d)", len(_fallback_soundings))
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # If soundings are available, generate a single cached subset ONCE and reuse it for BOTH
        # XS + skeleton so they see the exact same points (seam consistency + reproducibility).
        # Default output is parquet (fast/small).
        soundings_subset_path = None
        try:
            if getattr(cfg, "river_soundings", None):
                soundings_subset_path = work_dir / "river_soundings_subset.parquet"
                soundings_subset_meta = work_dir / "river_soundings_subset.meta.json"

                # Build a lightweight signature so we don't accidentally reuse a stale subset when
                # the soundings inputs or sampling config changed between reruns.
                def _soundings_signature(cfg_obj) -> str:
                    import hashlib
                    import json
                    inp = getattr(cfg_obj, "river_soundings", None)
                    if inp is None:
                        files = []
                    elif isinstance(inp, (list, tuple)):
                        files = [str(p) for p in inp]
                    else:
                        files = [str(inp)]

                    info = []
                    for fp in files:
                        try:
                            p = Path(fp)
                            if p.exists() and p.is_file():
                                st = p.stat()
                                info.append({"path": fp, "size": int(st.st_size), "mtime": float(st.st_mtime)})
                            else:
                                info.append({"path": fp, "size": None, "mtime": None})
                        except Exception:
                            info.append({"path": fp, "size": None, "mtime": None})

                    payload = {
                        "files": sorted(info, key=lambda d: d.get("path") or ""),
                        "soundings_max_points": int(getattr(cfg_obj, "soundings_max_points", 0) or 0),
                        "soundings_sample_seed": int(getattr(cfg_obj, "soundings_sample_seed", 0) or 0),
                    }
                    b = json.dumps(payload, sort_keys=True).encode("utf-8")
                    return hashlib.sha1(b).hexdigest()

                sig_now = _soundings_signature(cfg)

                # If subset exists but meta is missing or doesn't match, rebuild.
                if soundings_subset_path.exists():
                    try:
                        import json as _json
                        if not soundings_subset_meta.exists():
                            log.info("[RIVER] Cached soundings subset exists but meta missing; rebuilding for safety.")
                            soundings_subset_path.unlink(missing_ok=True)
                        else:
                            meta = _json.loads(soundings_subset_meta.read_text(encoding="utf-8"))
                            if meta.get("signature") != sig_now:
                                log.info("[RIVER] Cached soundings subset signature mismatch; rebuilding.")
                                soundings_subset_path.unlink(missing_ok=True)
                    except Exception:
                        logging.getLogger(__name__).debug("Subset meta check failed; rebuilding subset to be safe.", exc_info=True)
                        try:
                            soundings_subset_path.unlink(missing_ok=True)
                        except Exception:
                            pass

                # Create the subset if missing.
                if not soundings_subset_path.exists():
                    _tmp_out_gpkg = work_dir / "_tmp_soundings_subset_only.gpkg"
                    _tmp_out_tif = work_dir / "_tmp_soundings_subset_only.tif"
                    _tmp_meta_json = work_dir / "_tmp_soundings_subset_only_constraints_meta.json"
                    _tmp_acct_json = work_dir / "_tmp_soundings_subset_only_constraints_accounting.json"
                    cmd_subset = [
                        sys.executable, "xs_infer_bathy_raster.py",
                        f"--xs-gpkg={xs_gpkg}",
                        f"--template-raster={cfg.river_dem}",
                        f"--out-gpkg={_tmp_out_gpkg}",
                        f"--out-bathy-raster={_tmp_out_tif}",
                        f"--out-meta-json={_tmp_meta_json}",
                        f"--out-accounting-json={_tmp_acct_json}",
                        f"--continuous={getattr(cfg, 'river_continuous', 'walid_aniso')}",
                        f"--nodata={getattr(cfg, 'river_nodata', -9999.0)}",
                        f"--write-soundings-subset={soundings_subset_path}",
                        "--only-write-soundings-subset",
                    ]
                    _append_river_soundings_args(cmd_subset, cfg, include_calib_args=True, include_mode_args=False)
                    rc_s, out_s, err_s = run_command(cmd_subset, cwd=script_dir, prefix="[RIVER] ")
                    report["river"]["steps"]["soundings_subset"] = {
                        "status": "success" if rc_s == 0 else "failed",
                        "returncode": rc_s,
                        "command": " ".join(str(c) for c in cmd_subset),
                        "stdout_tail": out_s,
                        "stderr_tail": err_s,
                    }
                    if rc_s != 0 or not soundings_subset_path.exists():
                        log.warning("[RIVER] Failed to create cached soundings subset (rc=%s); continuing with raw soundings.", rc_s)
                        soundings_subset_path = None

                    # Write a tiny meta sidecar for reproducibility / stale-cache protection.
                    if soundings_subset_path is not None and soundings_subset_path.exists():
                        try:
                            import json as _json
                            import pandas as _pd
                            meta = {
                                "signature": sig_now,
                                "subset_path": str(soundings_subset_path),
                                "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "soundings_max_points": int(getattr(cfg, "soundings_max_points", 0) or 0),
                                "soundings_sample_seed": int(getattr(cfg, "soundings_sample_seed", 0) or 0),
                            }
                            # Record per-source counts if available.
                            try:
                                if soundings_subset_path.suffix.lower() == ".parquet":
                                    df = _pd.read_parquet(soundings_subset_path, columns=["_src_file"])
                                    if "_src_file" in df.columns:
                                        vc = df["_src_file"].astype(str).value_counts()
                                        meta["by_src"] = {Path(str(k)).stem if str(k) not in ["", "nan", "None"] else "unknown": int(v) for k, v in vc.items()}
                                        meta["n_subset"] = int(len(df))
                            except Exception:
                                pass
                            soundings_subset_meta.write_text(_json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
                        except Exception:
                            logging.getLogger(__name__).debug("Failed to write subset meta; continuing.", exc_info=True)

                # If we have a valid subset, point cfg.river_soundings at it so ALL downstream steps share it.
                if soundings_subset_path is not None and soundings_subset_path.exists():
                    cfg.river_soundings = str(soundings_subset_path)
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        _append_river_soundings_args(cmd, cfg, include_calib_args=True, include_mode_args=False)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["infer_xs_mainstem"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not bed_xs_tif.exists():
            if rc != 0 and bathy_gpkg.exists() and not bed_xs_tif.exists():
                log.warning("[RIVER] XS inference wrote GPKG but no raster (rc=%s); falling back to skeleton-only. stderr_tail=%s", rc, (err or '').strip()[-500:])
            elif rc != 0:
                log.warning("[RIVER] XS mainstem inference failed (rc=%s); hybrid will fall back to skeleton-only. stderr_tail=%s", rc, (err or '').strip()[-500:])
            else:
                log.warning("[RIVER] XS mainstem inference completed but raster missing; hybrid will fall back to skeleton-only.")
            bed_xs_tif = None

        # Capture explicit constraint metadata produced by xs_infer_bathy_raster.py (if present).
        try:
            import json as _json
            meta_path = Path(_tmp_meta_json) if "_tmp_meta_json" in locals() else Path(str(_tmp_out_tif) + ".meta.json")
            if meta_path.exists():
                meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                report.setdefault("river", {}).setdefault("constraints", {})["xs_mainstem"] = meta.get("constraints", {})
                report.setdefault("river", {}).setdefault("outputs", {})["xs_mainstem_constraint_meta"] = str(meta_path)

            acct_path = Path(_tmp_acct_json) if "_tmp_acct_json" in locals() else None
            if acct_path is not None and acct_path.exists():
                acct = _json.loads(acct_path.read_text(encoding="utf-8"))
                report.setdefault("river", {}).setdefault("constraint_accounting", {})["soundings_subset_step"] = acct
                report.setdefault("river", {}).setdefault("outputs", {})["soundings_subset_constraint_accounting"] = str(acct_path)
        except Exception:
            logging.getLogger(__name__).debug("Failed to read XS constraint meta; continuing.", exc_info=True)

        # Note: cfg.river_soundings may already point at the cached subset (preferred).

        # Capture explicit constraint meta + accounting produced by xs_infer_bathy_raster.py (no guessing).
        try:
            import json as _json
            if xs_mainstem_meta_json.exists():
                meta = _json.loads(xs_mainstem_meta_json.read_text(encoding="utf-8"))
                report.setdefault("river", {}).setdefault("constraints", {})["xs_mainstem"] = meta.get("constraints", {})
                report.setdefault("river", {}).setdefault("outputs", {})["xs_mainstem_constraint_meta"] = str(xs_mainstem_meta_json)
            if xs_mainstem_acct_json.exists():
                acct = _json.loads(xs_mainstem_acct_json.read_text(encoding="utf-8"))
                report.setdefault("river", {}).setdefault("constraint_accounting", {})["xs_mainstem"] = acct
                report.setdefault("river", {}).setdefault("outputs", {})["xs_mainstem_constraint_accounting"] = str(xs_mainstem_acct_json)
        except Exception:
            logging.getLogger(__name__).debug("Failed to read XS mainstem constraint meta/accounting; continuing.", exc_info=True)

        # Step 3c: Skeleton for full river network
        log.info("[RIVER] Step 3c: Inferring bathymetry from skeleton (full network)...")
        bed_skel_tif = work_dir / "river_bed_elev_skeleton_full.tif"

        cmd = [
            sys.executable, "river_skeleton_bathy.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--dem={cfg.river_dem}",
            f"--channel-mask={channel_mask_tif}",
            f"--out-bed={bed_skel_tif}",
            f"--dmax-min-m={getattr(cfg, 'river_dmax_min_m', 1.0)}",
            f"--dmax-max-m={getattr(cfg, 'river_dmax_max_m', 20.0)}",
            f"--shape-exp={getattr(cfg, 'river_shape_exp', 0.33)}",
            f"--bed-profile-max-slope={getattr(cfg, 'river_bed_profile_max_slope', 0.015)}",
            f"--bed-profile-max-curv={getattr(cfg, 'river_bed_profile_max_curv', 0.0005)}",
            f"--bed-profile-step-m={getattr(cfg, 'river_bed_profile_step_m', 50.0)}",
            f"--bed-profile-strength={getattr(cfg, 'river_bed_profile_strength', 0.65)}",
            f"--bed-profile-power={getattr(cfg, 'river_bed_profile_power', 2.0)}",
            f"--prior-mode={cfg.river_prior_mode}",
            f"--mv-a0={cfg.river_mv_a0}",
            f"--mv-bw={cfg.river_mv_bw}",
            f"--mv-ba={cfg.river_mv_ba}",
            f"--mv-bs={cfg.river_mv_bs}",
            f"--mv-eps-a={cfg.river_mv_eps_a}",
            f"--mv-eps-s={cfg.river_mv_eps_s}",
            f"--residual-blend-sigma-m={float(getattr(cfg, 'river_residual_blend_sigma_m', 120.0))}",
            f"--authoritative-bed-max-dist-m={float(getattr(cfg, 'river_authoritative_bed_max_dist_m', 2000.0))}",
            f"--wse-mode={getattr(cfg, 'river_skeleton_wse_mode', 'bank_profile')}",
            f"--wse-smooth-sigma-m={getattr(cfg, 'river_skeleton_wse_smooth_sigma_m', 250.0)}",
        ]

        # Optional authoritative bed raster blending
        if getattr(cfg, 'river_authoritative_bed', None):
            try:
                ab = Path(getattr(cfg, 'river_authoritative_bed'))
                if ab.exists():
                    cmd.append(f"--authoritative-bed-raster={ab}")
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # Junction / confluence handling
        jmode = str(getattr(cfg, "river_skeleton_junction_mode", "smooth")).strip().lower()
        if jmode and jmode != "none":
            cmd.append(f"--junction-mode={jmode}")
            cmd.append(f"--junction-buffer-m={float(getattr(cfg, 'river_skeleton_junction_buffer_m', 120.0))}")
            cmd.append(f"--junction-degree-min={int(getattr(cfg, 'river_skeleton_junction_degree_min', 3))}")
            jsig = float(getattr(cfg, "river_skeleton_junction_smooth_sigma_m", 80.0) or 0.0)
            if jsig > 0.0:
                cmd.append(f"--junction-smooth-sigma-m={jsig}")
            cmd.append(f"--junction-max-width-m={float(getattr(cfg, 'river_skeleton_junction_max_width_m', 300.0))}")

        # Curvature-driven asymmetry
        amode = str(getattr(cfg, "river_skeleton_asymmetry_mode", "none")).strip().lower()
        if amode and amode != "none":
            cmd.append(f"--asymmetry-mode={amode}")
            cmd.append(f"--asymmetry-strength={float(getattr(cfg, 'river_skeleton_asymmetry_strength', 0.25))}")
            cmd.append(f"--asymmetry-curv-ref={float(getattr(cfg, 'river_skeleton_asymmetry_curv_ref', 0.002))}")
            cmd.append(f"--asymmetry-max-shift={float(getattr(cfg, 'river_skeleton_asymmetry_max_shift', 0.20))}")
            cmd.append(f"--asymmetry-min-width-m={float(getattr(cfg, 'river_skeleton_asymmetry_min_width_m', 10.0))}")
            cmd.append(f"--asymmetry-min-curv={float(getattr(cfg, 'river_skeleton_asymmetry_min_curv', 0.0005))}")
            cmd.append(f"--asymmetry-densify-step-m={float(getattr(cfg, 'river_skeleton_asymmetry_densify_step_m', 20.0))}")
        _append_river_soundings_args(cmd, cfg, include_calib_args=False, include_mode_args=True)


        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["skeleton_full"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not Path(bed_skel_tif).exists():
            log.error("[RIVER] Skeleton bathymetry failed (hybrid).")
            report["river"]["status"] = "failed"
            return None

        # Step 3d: Combine
        log.info("[RIVER] Step 3d: Combining XS(mainstem) + skeleton(full) into cached bed raster...")
        bed_tif = cached_bed_tif
        nod = float(getattr(cfg, 'river_nodata', -9999.0))
        if bed_xs_tif is not None and Path(mainstem_mask_tif).exists():
            _combine_mainstem_xs_and_skeleton(Path(bed_xs_tif), Path(bed_skel_tif), Path(mainstem_mask_tif), Path(bed_tif), Path(cfg.river_dem), nod)
        else:
            import shutil
            shutil.copy2(str(bed_skel_tif), str(bed_tif))

        report.setdefault('river', {}).setdefault('outputs', {})['bed_elev_xs_mainstem'] = str(bed_xs_tif) if bed_xs_tif is not None else None
        report.setdefault('river', {}).setdefault('outputs', {})['bed_elev_skeleton_full'] = str(bed_skel_tif)
        report.setdefault('river', {}).setdefault('outputs', {})['mainstem_mask'] = str(mainstem_mask_tif) if Path(mainstem_mask_tif).exists() else None

    if river_method == "hybrid":
        # HYBRID handled above (bed_tif already built)
        pass
    elif river_method == "skeleton":
        # Step 2: Build a river channel mask (river vs. open water) from NHD flowlines + waffles water mask.
        log.info("[RIVER] Step 2: Building river channel mask (skeleton method)...")
        channel_mask_tif = work_dir / "river_channel_mask.tif"
        open_water_mask_tif = work_dir / "open_water_mask.tif"

        # Waffles-derived masks:
        #   1) ocean-only (want_nhd=False) always attempted first to prevent ocean bleed (resilient if TNM is flaky).
        #   2) with-NHD (want_nhd=True) attempted second; if it fails we still proceed using corridor+ArcGIS flowlines.
        # Reuse WAFFLES coastline masks across runs for the same AOI + resolution + module params.
        cache_masks_shared = Path(getattr(cfg, "cache_root", ".")).resolve() / "masks"
        ensure_dir(cache_masks_shared)
        cache_masks_run = Path(cfg.derived_cache_root) / "masks"
        ensure_dir(cache_masks_run)
        aoi_buf = str(cfg.aoi)
        inc_arcsec = float(getattr(cfg, "waffles_inc_arcsec", 1.0) or 1.0)

        ocean_mask = None
        with_nhd_mask = None
        try:
            ocean_cache = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks_shared,
                aoi=aoi_buf,
                inc_arcsec=inc_arcsec,
                want_nhd=False,
                want_lakes=False,
                prefix="waffles_coastline_ocean_only",
                log=log,
                force=False,
            )
            ocean_mask = _stage_cached_waffles_mask(ocean_cache, cache_masks_run / "waffles_coastline_ocean_only.tif", log=log)
        except Exception as e:
            log.warning(f"[WAFFLES] Ocean-only mask unavailable; ocean bleed protection degraded: {e}")
            ocean_mask = None

        try:
            nhd_cache = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks_shared,
                aoi=aoi_buf,
                inc_arcsec=inc_arcsec,
                want_nhd=True,
                want_lakes=False,
                prefix="waffles_coastline_with_nhd",
                log=log,
                force=False,
            )
            with_nhd_mask = _stage_cached_waffles_mask(nhd_cache, cache_masks_run / "waffles_coastline_with_nhd.tif", log=log)
        except Exception as e:
            log.warning(f"[WAFFLES] With-NHD mask unavailable (TNM flaky?): {e}. Proceeding with corridor+ArcGIS NHD flowlines + ocean-only mask.")
            with_nhd_mask = None

        if strict:
            # River domain building relies on a WAFFLES water mask to prevent ocean/land bleed
            # and to make downstream clipping deterministic. Fail closed if we can't get one.
            if (with_nhd_mask is None) or (not Path(with_nhd_mask).exists()):
                raise RuntimeError("Strict river domain build requested but WAFFLES with-NHD water mask is unavailable. Fix WAFFLES generation first.")
        cmd = [
            sys.executable, "river_domain_mask.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-channel-mask={channel_mask_tif}",
            f"--out-open-water-mask={open_water_mask_tif}",
            f"--channel-buffer-m={getattr(cfg, 'river_channel_buffer_m', 400.0)}",
            f"--max-channel-width-m={getattr(cfg, 'river_max_channel_width_m', 600.0)}",
            f"--mainstem-min-order={getattr(cfg, 'river_mainstem_min_order', 5)}",
            f"--max-mainstem-width-m={getattr(cfg, 'river_max_mainstem_width_m', 2500.0)}",
        ]

        # Channel domain source policy
        # - auto: prefer NHDArea river polygons when usable, else fall back to corridor
        # - nhdarea: require river polygons (exclude lakes)
        # - corridor: buffered flowline corridor only
        chan_src = str(getattr(cfg, 'river_channel_source', 'auto') or 'auto').strip().lower()
        if chan_src not in ('auto', 'nhdarea', 'corridor'):
            log.warning(f"[RIVER] Unknown river_channel_source='{chan_src}', defaulting to 'auto'.")
            chan_src = 'auto'
        cmd.append(f"--channel-source={chan_src}")

        # NHDArea filtering: keep Stream/River polygons only (exclude lakes/reservoirs).
        # Default is conservative: FType=460 (Stream/River). Users can override via config.
        nhd_allow = getattr(cfg, 'river_nhdarea_allow_ftype', None)
        if nhd_allow is None:
            nhd_allow = "460"
        cmd.append(f"--nhdarea-allow-ftype={nhd_allow}")
        nhd_allow_fcode = getattr(cfg, 'river_nhdarea_allow_fcode', None)
        if nhd_allow_fcode:
            cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")

        # Optional: NHDArea constraint (if river_network.py wrote polygons into the gpkg).
        # NOTE: river_domain_mask filters NHDArea to river/stream polygons (excluding lakes).
        # Only pass NHDArea inputs when using auto/nhdarea mode.
        if (chan_src in ('auto', 'nhdarea')) and bool(getattr(cfg, "river_use_nhdarea", True)):
            cmd.append(f"--nhdarea-gpkg={network_gpkg}")
            cmd.append(f"--nhdarea-layer={getattr(cfg, 'river_nhdarea_layer', 'nhdarea_clip')}")
        if ocean_mask and Path(ocean_mask).exists():
            cmd.append(f"--ocean-mask={ocean_mask}")
        oke = float(getattr(cfg, 'river_ocean_keep_dist_m', 0.0) or 0.0)
        if (oke > 0.0):
            cmd.append(f"--ocean-keep-dist-m={oke}")

        if with_nhd_mask and Path(with_nhd_mask).exists():
            cmd.append(f"--water-mask={with_nhd_mask}")
        if getattr(cfg, "river_save_skeleton_debug", False):
            cmd.append("--write-debug")

        # For reporting: prefer the more inclusive water mask (with_nhd) if it exists,
        # otherwise fall back to the ocean-only mask (still useful to prevent ocean bleed).
        wm = None
        if with_nhd_mask and Path(with_nhd_mask).exists():
            wm = with_nhd_mask
        elif ocean_mask and Path(ocean_mask).exists():
            wm = ocean_mask

        # Persist the waffles mask used (for downstream final clipping/QA).
        try:
            report.setdefault("river", {}).setdefault("outputs", {})["waffles_water_mask"] = str(wm) if wm else None
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["domain_mask"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
            "waffles_mask": str(wm) if wm else None,
        }
        if rc != 0 or (not channel_mask_tif.exists()):
            log.error("[RIVER] Failed to build river channel mask.")
            report["river"]["status"] = "failed"
            return None

        log.info(f"[RIVER] Channel mask built: {channel_mask_tif}")

        try:
            report.setdefault("river", {}).setdefault("outputs", {})["river_channel_mask"] = str(channel_mask_tif)
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


        # Stash river domain/channel mask for fusion: inside this mask, river should override SDB to avoid tile seams
        try:
            cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


        # Step 3: Skeleton bathymetry (distance-transform, no cross-sections)
        log.info("[RIVER] Step 3: Inferring bathymetry (channel skeleton)...")
        bed_tif = cached_bed_tif

        cmd = [
            sys.executable, "river_skeleton_bathy.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--dem={cfg.river_dem}",
            f"--channel-mask={channel_mask_tif}",
            f"--out-bed={bed_tif}",
            f"--shape-exp={getattr(cfg, 'river_shape_exp', 0.5)}",
            f"--dmax-min-m={getattr(cfg, 'river_dmax_min_m', 0.5)}",
            f"--dmax-max-m={getattr(cfg, 'river_dmax_max_m', 30.0)}",
            f"--bed-profile-max-slope={float(getattr(cfg,'river_bed_profile_max_slope',0.0) or 0.0)}",
            f"--bed-profile-max-curv={float(getattr(cfg,'river_bed_profile_max_curv',0.0) or 0.0)}",
            f"--bed-profile-step-m={float(getattr(cfg,'river_bed_profile_step_m',25.0) or 25.0)}",
            f"--bed-profile-strength={float(getattr(cfg,'river_bed_profile_strength',0.6) or 0.6)}",
            f"--bed-profile-power={float(getattr(cfg,'river_bed_profile_power',2.0) or 2.0)}",
            f"--prior-mode={cfg.river_prior_mode}",
            f"--mv-a0={cfg.river_mv_a0}",
            f"--mv-bw={cfg.river_mv_bw}",
            f"--mv-ba={cfg.river_mv_ba}",
            f"--mv-bs={cfg.river_mv_bs}",
            f"--mv-eps-a={cfg.river_mv_eps_a}",
            f"--mv-eps-s={cfg.river_mv_eps_s}",
            f"--residual-blend-sigma-m={float(getattr(cfg, 'river_residual_blend_sigma_m', 120.0))}",
            f"--authoritative-bed-max-dist-m={float(getattr(cfg, 'river_authoritative_bed_max_dist_m', 2000.0))}",
        ]

        # WSE proxy controls (bank-derived WSE is usually more robust than in-channel DEM sampling)
        cmd.append(f"--wse-mode={getattr(cfg, 'river_skeleton_wse_mode', 'bank')}")
        wse_sig = float(getattr(cfg, 'river_skeleton_wse_smooth_sigma_m', 0.0) or 0.0)
        if wse_sig > 0.0:
            cmd.append(f"--wse-smooth-sigma-m={wse_sig}")
        if str(getattr(cfg, "river_skeleton_wse_mode", "bank")).strip().lower() == "bank_profile":
            cmd.append(f"--wse-profile-step-m={float(getattr(cfg, 'river_skeleton_wse_profile_step_m', 20.0) or 20.0)}")
            cmd.append(f"--wse-profile-resample-m={float(getattr(cfg, 'river_skeleton_wse_profile_resample_m', 20.0) or 20.0)}")
            cmd.append(f"--wse-profile-smooth-sigma-m={float(getattr(cfg, 'river_skeleton_wse_profile_smooth_sigma_m', 200.0) or 0.0)}")
            cmd.append(f"--wse-profile-max-slope={float(getattr(cfg, 'river_skeleton_wse_profile_max_slope', 0.005) or 0.0)}")
            cmd.append(f"--wse-profile-min-samples={int(getattr(cfg, 'river_skeleton_wse_profile_min_samples', 10) or 10)}")
            cmd.append(f"--wse-profile-max-query-dist-m={float(getattr(cfg, 'river_skeleton_wse_profile_max_query_dist_m', 250.0) or 0.0)}")
            # Optional: SWOT RiverSP anchoring (vector WSE observations)
            if getattr(cfg, 'river_swot_riversp', None):
                for fp in getattr(cfg, 'river_swot_riversp'):
                    cmd.append(f"--swot-riversp={fp}")
                if getattr(cfg, 'river_swot_wse_field', None):
                    cmd.append(f"--swot-wse-field={getattr(cfg, 'river_swot_wse_field')}")
                if getattr(cfg, 'river_swot_qual_field', None):
                    cmd.append(f"--swot-qual-field={getattr(cfg, 'river_swot_qual_field')}")
                cmd.append(f"--swot-max-dist-m={float(getattr(cfg, 'river_swot_max_dist_m', 300.0) or 0.0)}")
                cmd.append(f"--swot-min-samples={int(getattr(cfg, 'river_swot_min_samples', 5) or 0)}")
                cmd.append(f"--swot-correct-sigma-m={float(getattr(cfg, 'river_swot_correct_sigma_m', 2000.0) or 0.0)}")
                cmd.append(f"--swot-weight={float(getattr(cfg, 'river_swot_weight', 1.0) or 0.0)}")
                cmd.append(f"--swot-max-correction-m={float(getattr(cfg, 'river_swot_max_correction_m', 5.0) or 0.0)}")
                cmd.append(f"--swot-wse-offset-m={float(getattr(cfg, 'river_swot_wse_offset_m', 0.0) or 0.0)}")

                cmd.append(f"--swot-offset-mode={str(getattr(cfg, 'river_swot_offset_mode', 'median_mad') or 'median_mad')}")
                cmd.append(f"--swot-offset-min-samples={int(getattr(cfg, 'river_swot_offset_min_samples', 25) or 0)}")
                cmd.append(f"--swot-offset-mad-z={float(getattr(cfg, 'river_swot_offset_mad_z', 3.5) or 3.5)}")
                cmd.append(f"--swot-offset-max-abs-m={float(getattr(cfg, 'river_swot_offset_max_abs_m', 10.0) or 0.0)}")

        # Junction / confluence handling
        jmode = str(getattr(cfg, "river_skeleton_junction_mode", "smooth")).strip().lower()
        if jmode and jmode != "none":
            cmd.append(f"--junction-mode={jmode}")
            cmd.append(f"--junction-buffer-m={float(getattr(cfg, 'river_skeleton_junction_buffer_m', 120.0))}")
            cmd.append(f"--junction-degree-min={int(getattr(cfg, 'river_skeleton_junction_degree_min', 3))}")
            jsig = float(getattr(cfg, "river_skeleton_junction_smooth_sigma_m", 80.0) or 0.0)
            if jsig > 0.0:
                cmd.append(f"--junction-smooth-sigma-m={jsig}")
            cmd.append(f"--junction-max-width-m={float(getattr(cfg, 'river_skeleton_junction_max_width_m', 300.0))}")

        # Curvature-driven asymmetry (outer-bank deeper in bends)
        amode = str(getattr(cfg, "river_skeleton_asymmetry_mode", "none")).strip().lower()
        if amode and amode != "none":
            cmd.append(f"--asymmetry-mode={amode}")
            cmd.append(f"--asymmetry-strength={float(getattr(cfg, 'river_skeleton_asymmetry_strength', 0.25))}")
            cmd.append(f"--asymmetry-curv-ref={float(getattr(cfg, 'river_skeleton_asymmetry_curv_ref', 0.002))}")
            cmd.append(f"--asymmetry-max-shift={float(getattr(cfg, 'river_skeleton_asymmetry_max_shift', 0.20))}")
            cmd.append(f"--asymmetry-min-width-m={float(getattr(cfg, 'river_skeleton_asymmetry_min_width_m', 10.0))}")
            cmd.append(f"--asymmetry-min-curv={float(getattr(cfg, 'river_skeleton_asymmetry_min_curv', 0.0005))}")
            cmd.append(f"--asymmetry-densify-step-m={float(getattr(cfg, 'river_skeleton_asymmetry_densify_step_m', 20.0))}")
        if getattr(cfg, "river_save_skeleton_debug", False):
            cmd.append(f"--debug-dir={work_dir / 'skeleton_debug'}")

        # Optional: use external soundings (extra XYZ) to refine the skeleton prior and enforce depth anchors
        if getattr(cfg, 'river_soundings', None):
            snd_list = [s.strip() for s in str(cfg.river_soundings).split(',') if s.strip()]
            if snd_list:
                if getattr(cfg, 'extra_xyz_crs', None):
                    cmd.append(f"--soundings-crs={cfg.extra_xyz_crs}")
                cmd.append(f"--soundings-mode={getattr(cfg, 'river_soundings_mode', 'auto')}")
                cmd.append(f"--soundings-max-dist-m={float(getattr(cfg, 'river_soundings_max_dist_m', 1500.0))}")
                cmd.append(f"--soundings-min-r={float(getattr(cfg, 'river_soundings_min_r', 0.25))}")
                if not bool(getattr(cfg, 'river_soundings_enforce', True)):
                    cmd.append('--no-soundings-enforce')
                cmd.extend(['--soundings'] + snd_list)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["skeleton"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not Path(bed_tif).exists():
            log.error("[RIVER] Skeleton bathymetry failed.")
            report["river"]["status"] = "failed"
            return None

        log.info(f"[RIVER] Skeleton bathymetry raster: {bed_tif}")

        # Optional: constrain river outputs to NHDArea polygons (best-effort).
        # NOTE: This must happen AFTER the bed raster exists.
        if bool(getattr(cfg, "river_use_nhdarea", True)):
            try:
                applied = _mask_raster_to_nhdarea(
                    bed_tif,
                    network_gpkg,
                    nhd_layer=getattr(cfg, "river_nhdarea_layer", "nhdarea_clip"),
                    nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                )
                report.setdefault("river", {}).setdefault("masking", {})["nhdarea_bed_masked"] = bool(applied)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # Final safety: ensure river bed raster is nodata outside the river channel mask.
        try:
            if Path(channel_mask_tif).exists():
                ok = _clip_raster_to_mask(
                    Path(bed_tif),
                    Path(channel_mask_tif),
                    inside_value=1,
                    invert=False,
                    nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                )
                report.setdefault("river", {}).setdefault("masking", {})["channel_mask_clip"] = bool(ok)
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    else:
        # Step 2: Generating cross-sections...
        log.info("[RIVER] Step 2: Generating cross-sections...")
        xs_gpkg = work_dir / "cross_sections.gpkg"

        # SECURITY FIX: Use list-based command construction
        cmd = [
            sys.executable, "xs_builder.py",
            f"--river-gpkg={network_gpkg}",
            f"--dem={cfg.river_dem}",
            f"--out-gpkg={xs_gpkg}",
            f"--spacing-m={cfg.xs_spacing_m}",
            f"--half-width-m={cfg.xs_length_m / 2.0}",
            f"--smoothing-window-m={getattr(cfg, 'xs_smoothing_window_m', 0.0)}",
            f"--deconflict-tol-m={getattr(cfg, 'xs_deconflict_tol_m', 2.0)}",
            f"--junction-snap-m={getattr(cfg, 'xs_junction_snap_m', 30.0)}",
            f"--junction-buffer-m={getattr(cfg, 'xs_junction_buffer_m', 120.0)}",
            f"--densify-step-m={getattr(cfg, 'xs_densify_step_m', 20.0)}",
        ]
        if not bool(getattr(cfg, "xs_trim_overlaps", True)):
            cmd.append("--no-trim-overlaps")
        if not bool(getattr(cfg, "xs_global_deconflict", True)):
            cmd.append("--no-global-deconflict")
        if not bool(getattr(cfg, "xs_skip_junctions", True)):
            cmd.append("--no-skip-junctions")


        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["xs_builder"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not xs_gpkg.exists():
            log.error("[RIVER] Failed to build cross-sections.")
            report["river"]["status"] = "failed"
            return None

        log.info(f"[RIVER] Cross-sections generated: {xs_gpkg}")
        # Step 2b: Build river channel domain mask (so final river raster is river-only)
        channel_mask_tif = work_dir / "river_channel_mask.tif"
        open_water_mask_tif = work_dir / "open_water_mask.tif"

        # Reuse the same domain-mask logic used by the skeleton method for consistency.
        # This ensures river outputs cannot bleed into ocean/lakes when using XS method.
        # Reuse WAFFLES coastline masks across runs for the same AOI + resolution + module params.
        cache_masks_shared = Path(getattr(cfg, "cache_root", ".")).resolve() / "masks"
        ensure_dir(cache_masks_shared)
        cache_masks_run = Path(cfg.derived_cache_root) / "masks"
        ensure_dir(cache_masks_run)
        aoi_buf = str(cfg.aoi)
        inc_arcsec = float(getattr(cfg, "waffles_inc_arcsec", 1.0) or 1.0)

        ocean_mask = None
        with_nhd_mask = None
        try:
            ocean_cache = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks_shared,
                aoi=aoi_buf,
                inc_arcsec=inc_arcsec,
                want_nhd=False,
                want_lakes=False,
                prefix="waffles_coastline_ocean_only",
                log=log,
                force=False,
            )
            ocean_mask = _stage_cached_waffles_mask(ocean_cache, cache_masks_run / "waffles_coastline_ocean_only.tif", log=log)
        except Exception as e:
            log.warning(f"[WAFFLES] Ocean-only mask unavailable; ocean bleed protection degraded: {e}")
            ocean_mask = None

        try:
            nhd_cache = _ensure_waffles_coastline_mask(
                cache_masks=cache_masks_shared,
                aoi=aoi_buf,
                inc_arcsec=inc_arcsec,
                want_nhd=True,
                want_lakes=False,
                prefix="waffles_coastline_with_nhd",
                log=log,
                force=False,
            )
            with_nhd_mask = _stage_cached_waffles_mask(nhd_cache, cache_masks_run / "waffles_coastline_with_nhd.tif", log=log)
        except Exception as e:
            log.warning(f"[WAFFLES] With-NHD mask unavailable (TNM flaky?): {e}. Proceeding with corridor+ArcGIS NHD flowlines + ocean-only mask.")
            with_nhd_mask = None

        if strict:
            # River domain building relies on a WAFFLES water mask to prevent ocean/land bleed
            # and to make downstream clipping deterministic. Fail closed if we can't get one.
            if (with_nhd_mask is None) or (not Path(with_nhd_mask).exists()):
                raise RuntimeError("Strict river domain build requested but WAFFLES with-NHD water mask is unavailable. Fix WAFFLES generation first.")
        cmd = [
            sys.executable, "river_domain_mask.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-channel-mask={channel_mask_tif}",
            f"--out-open-water-mask={open_water_mask_tif}",
            f"--channel-buffer-m={getattr(cfg, 'river_channel_buffer_m', 400.0)}",
            f"--max-channel-width-m={getattr(cfg, 'river_max_channel_width_m', 600.0)}",
            f"--mainstem-min-order={getattr(cfg, 'river_mainstem_min_order', 5)}",
            f"--max-mainstem-width-m={getattr(cfg, 'river_max_mainstem_width_m', 2500.0)}",
        ]

        chan_src = str(getattr(cfg, 'river_channel_source', 'auto') or 'auto').strip().lower()
        if chan_src not in ('auto', 'nhdarea', 'corridor'):
            log.warning(f"[RIVER] Unknown river_channel_source='{chan_src}', defaulting to 'auto'.")
            chan_src = 'auto'
        cmd.append(f"--channel-source={chan_src}")

        nhd_allow = getattr(cfg, 'river_nhdarea_allow_ftype', None)
        if nhd_allow is None:
            nhd_allow = "460"
        cmd.append(f"--nhdarea-allow-ftype={nhd_allow}")
        nhd_allow_fcode = getattr(cfg, 'river_nhdarea_allow_fcode', None)
        if nhd_allow_fcode:
            cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")

        if (chan_src in ('auto', 'nhdarea')) and bool(getattr(cfg, "river_use_nhdarea", True)):
            cmd.append(f"--nhdarea-gpkg={network_gpkg}")
            cmd.append(f"--nhdarea-layer={getattr(cfg, 'river_nhdarea_layer', 'nhdarea_clip')}")

        if ocean_mask and Path(ocean_mask).exists():
            cmd.append(f"--ocean-mask={ocean_mask}")
        oke = float(getattr(cfg, 'river_ocean_keep_dist_m', 0.0) or 0.0)
        if (oke > 0.0):
            cmd.append(f"--ocean-keep-dist-m={oke}")

        if with_nhd_mask and Path(with_nhd_mask).exists():
            cmd.append(f"--water-mask={with_nhd_mask}")

        # Persist the waffles mask used (for downstream final clipping/QA).
        try:
            wm = None
            if with_nhd_mask and Path(with_nhd_mask).exists():
                wm = with_nhd_mask
            elif ocean_mask and Path(ocean_mask).exists():
                wm = ocean_mask
            report.setdefault("river", {}).setdefault("outputs", {})["waffles_water_mask"] = str(wm) if wm else None
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["domain_mask"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or (not channel_mask_tif.exists()):
            log.warning("[RIVER] Failed to build channel mask for XS method; continuing without hard domain constraint.")
        else:
            try:
                cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


        # Step 3: infer bathymetry (raster + optional GPKG)
        log.info("[RIVER] Step 3: Inferring bathymetry...")
        bed_tif = cached_bed_tif
        bathy_gpkg = work_dir / "river_bathy.gpkg"
        xs_meta_json = work_dir / "xs_constraints_meta.json"
        xs_acct_json = work_dir / "xs_constraints_accounting.json"

        # SECURITY FIX: Use list-based command construction
        cmd = [
            sys.executable, "xs_infer_bathy_raster.py",
            f"--xs-gpkg={xs_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-gpkg={bathy_gpkg}",
            f"--out-bathy-raster={bed_tif}",
            f"--out-meta-json={xs_meta_json}",
            f"--out-accounting-json={xs_acct_json}",
            f"--continuous={getattr(cfg, 'river_continuous', 'walid_aniso')}",
            f"--continuous-k={getattr(cfg, 'river_continuous_k', 12)}",
            f"--idw-power={getattr(cfg, 'river_idw_power', 2.0)}",
            f"--aniso-along-scale-m={getattr(cfg, 'river_aniso_along_scale_m', 500.0)}",
            f"--aniso-cross-scale-m={getattr(cfg, 'river_aniso_cross_scale_m', 30.0)}",
            f"--thalweg-weight={getattr(cfg, 'river_thalweg_weight', 6.0)}",
            f"--nodata={getattr(cfg, 'river_nodata', -9999.0)}",
            f"--overlap-reducer={getattr(cfg, 'river_overlap_reducer', 'min')}",
        ]
        # Hard constrain interpolation/output to the river domain mask when available.
        if 'channel_mask_tif' in locals() and channel_mask_tif is not None and Path(channel_mask_tif).exists():
            cmd.append(f"--channel-mask-raster={channel_mask_tif}")
            cmd.append("--channel-mask-inside-value=1")
        cmd.append(f"--river-gpkg={network_gpkg}")
        cmd.append(f"--prior-mode={cfg.river_prior_mode}")
        cmd.append(f"--mv-a0={cfg.river_mv_a0}")
        cmd.append(f"--mv-bw={cfg.river_mv_bw}")
        cmd.append(f"--mv-ba={cfg.river_mv_ba}")
        cmd.append(f"--mv-bs={cfg.river_mv_bs}")
        cmd.append(f"--mv-eps-a={cfg.river_mv_eps_a}")
        cmd.append(f"--mv-eps-s={cfg.river_mv_eps_s}")
        cmd.append(f"--slope-proxy-window={cfg.river_slope_proxy_window}")
        cmd.append(f"--slope-min={cfg.river_slope_min}")
        cmd.append(f"--slope-max={cfg.river_slope_max}")
        cmd.append(f"--slope-proxy-min-n={cfg.river_slope_proxy_min_n}")

        # Longitudinal WSE profile fit (preferred slope proxy)
        if bool(getattr(cfg, "river_wse_profile_enabled", True)):
            cmd.append("--wse-profile-enabled")
        else:
            cmd.append("--no-wse-profile")
        cmd.append(f"--wse-profile-window={getattr(cfg, 'river_wse_profile_window', cfg.river_slope_proxy_window)}")
        cmd.append(f"--wse-profile-min-n={getattr(cfg, 'river_wse_profile_min_n', cfg.river_slope_proxy_min_n)}")
        if bool(getattr(cfg, "river_wse_profile_monotonic", True)):
            cmd.append("--wse-profile-monotonic")
        else:
            cmd.append("--no-wse-profile-monotonic")

        # Option A: USGS discharge *measurement* anchors
        if cfg.river_usgs_sites:
            cmd.append(f"--usgs-sites={cfg.river_usgs_sites}")
            if cfg.river_usgs_start:
                cmd.append(f"--usgs-start={cfg.river_usgs_start}")
            if cfg.river_usgs_end:
                cmd.append(f"--usgs-end={cfg.river_usgs_end}")
            if cfg.river_usgs_cache_dir:
                cmd.append(f"--usgs-cache-dir={cfg.river_usgs_cache_dir}")
            cmd.append(f"--usgs-max-dist-m={cfg.river_usgs_max_dist_m}")
            cmd.append(f"--usgs-mean-to-dmax={cfg.river_usgs_mean_to_dmax}")
            cmd.append(f"--usgs-a-stat={cfg.river_usgs_a_stat}")
            cmd.append(f"--usgs-q-quantile-lo={cfg.river_usgs_q_quantile_lo}")
            cmd.append(f"--usgs-q-quantile-hi={cfg.river_usgs_q_quantile_hi}")
            cmd.append(f"--usgs-a-cv-warn={cfg.river_usgs_a_cv_warn}")
            cmd.append(f"--usgs-width-ratio-max={cfg.river_usgs_width_ratio_max}")
            cmd.append(f"--gage-snap-max-dist-m={cfg.river_gage_snap_max_dist_m}")
            if cfg.river_usgs_width_ratio_blend:
                cmd.append("--usgs-width-ratio-blend")
            else:
                cmd.append("--no-usgs-width-ratio-blend")

        # Optional: width-stage inversion anchors
        if cfg.river_width_stage_csv:
            cmd.append(f"--width-stage-csv={cfg.river_width_stage_csv}")
            cmd.append(f"--width-stage-max-dist-m={cfg.river_width_stage_max_dist_m}")
            cmd.append(f"--width-stage-min-n={cfg.river_width_stage_min_n}")
            cmd.append(f"--width-stage-min-r2={cfg.river_width_stage_min_r2}")
            cmd.append(f"--width-stage-max-weight={cfg.river_width_stage_max_weight}")


        # Optional: Manning inversion prior (blended)
        if getattr(cfg, "river_manning_mode", "off") != "off":
            cmd.append(f"--manning-mode={cfg.river_manning_mode}")
            if cfg.river_manning_q_cms is not None:
                cmd.append(f"--manning-q-cms={cfg.river_manning_q_cms}")
            if cfg.river_manning_q_field:
                cmd.append(f"--manning-q-field={cfg.river_manning_q_field}")
            cmd.append(f"--manning-n={cfg.river_manning_n}")
            cmd.append(f"--manning-region={getattr(cfg, 'river_manning_region', 'default')}")
            cmd.append(f"--manning-min-confidence={getattr(cfg, 'river_manning_min_confidence', 0.30)}")
            cmd.append(f"--manning-max-weight={cfg.river_manning_max_weight}")
            cmd.append(f"--manning-backwater-slope-thresh={cfg.river_manning_backwater_slope_thresh}")
            if cfg.river_manning_dist_to_mouth_field:
                cmd.append(f"--manning-dist-to-mouth-field={cfg.river_manning_dist_to_mouth_field}")
            cmd.append(f"--manning-dist-to-mouth-km-max={cfg.river_manning_dist_to_mouth_km_max}")



        # Optional: Regional hydraulic geometry curve prior
        if getattr(cfg, "river_regional_curve_enabled", False):
            cmd.append("--regional-curve-enabled")
            cmd.append(f"--regional-curve-region={cfg.river_regional_curve_region}")
            if cfg.river_regional_curve_c is not None:
                cmd.append(f"--regional-curve-c={cfg.river_regional_curve_c}")
            if cfg.river_regional_curve_f is not None:
                cmd.append(f"--regional-curve-f={cfg.river_regional_curve_f}")
            cmd.append(f"--regional-curve-da-units={cfg.river_regional_curve_da_units}")
            cmd.append(f"--regional-curve-depth-units={cfg.river_regional_curve_depth_units}")
            cmd.append(f"--regional-curve-depth-type={cfg.river_regional_curve_depth_type}")
            cmd.append(f"--regional-curve-to-dmax={cfg.river_regional_curve_to_dmax}")
            cmd.append(f"--regional-curve-to-dmax-factor={cfg.river_regional_curve_to_dmax_factor}")
            cmd.append(f"--regional-curve-unc-pct={cfg.river_regional_curve_unc_pct}")
            cmd.append(f"--regional-curve-max-weight={cfg.river_regional_curve_max_weight}")
            cmd.append(f"--regional-curve-min-da-km2={cfg.river_regional_curve_min_da_km2}")
        if getattr(cfg, "river_continuous_buffer_m", None) is not None:
            cmd.append(f"--continuous-buffer-m={cfg.river_continuous_buffer_m}")
        _append_river_soundings_args(cmd, cfg, include_calib_args=False, include_mode_args=True)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["infer_raster"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
        }
        if rc != 0 or not bed_tif.exists():
            # Include useful diagnostics inline (command + stderr/stdout tail), so users don't have to open the JSON report.
            log.error("[RIVER] Failed to infer river bed patch raster (rc=%s, exists=%s).", rc, bed_tif.exists())
            log.error("[RIVER] Command: %s", cmd_str)
            if err:
                log.error("[RIVER] stderr_tail:\n%s", err[-4000:])
            if out:
                log.error("[RIVER] stdout_tail:\n%s", out[-4000:])
            report["river"]["status"] = "failed"
            return None

        
    report["river"]["status"] = "success"
    log.info(f"[RIVER] Success: {bed_tif}")

    # Capture explicit XS constraint meta + accounting (no guessing).
    try:
        import json as _json
        if xs_meta_json.exists():
            meta = _json.loads(xs_meta_json.read_text(encoding="utf-8"))
            report.setdefault("river", {}).setdefault("constraints", {})["xs"] = meta.get("constraints", {})
            report.setdefault("river", {}).setdefault("outputs", {})["xs_constraint_meta"] = str(xs_meta_json)
        if xs_acct_json.exists():
            acct = _json.loads(xs_acct_json.read_text(encoding="utf-8"))
            report.setdefault("river", {}).setdefault("constraint_accounting", {})["xs"] = acct
            report.setdefault("river", {}).setdefault("outputs", {})["xs_constraint_accounting"] = str(xs_acct_json)
    except Exception:
        logging.getLogger(__name__).debug("Failed to read XS constraint meta/accounting; continuing.", exc_info=True)

    # Materialize into run output folder for convenience
    # Compute depth relative to DEM terrain surface (negative down): depth = bed_elev - dem
    try:
        compute_depth_from_bed_and_dem(bed_tif, Path(cfg.river_dem), cached_depth_tif, depth_sign="negative_down")
    except Exception as e:
        log.error("[RIVER] Failed to compute depth from bed elevation and DEM: %s", e)
        report["river"]["status"] = "failed"
        return None

    # Final hard guarantee: river outputs (bed + depth) must be nodata outside the river channel domain.
    try:
        _maskp = None
        if getattr(cfg, "river_domain_mask_for_fusion", None):
            mp = Path(getattr(cfg, "river_domain_mask_for_fusion"))
            if mp.exists():
                _maskp = mp
        if _maskp is None and 'channel_mask_tif' in locals():
            try:
                mp2 = Path(channel_mask_tif)
                if mp2.exists():
                    _maskp = mp2
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
        if _maskp is not None:
            nval = float(getattr(cfg, "river_nodata", -9999.0))
            ok_bed = _clip_raster_to_mask(Path(bed_tif), _maskp, inside_value=1, invert=False, nodata=nval)
            ok_dep = _clip_raster_to_mask(Path(cached_depth_tif), _maskp, inside_value=1, invert=False, nodata=nval)
            report.setdefault("river", {}).setdefault("masking", {})["channel_mask_clip_bed"] = bool(ok_bed)
            report.setdefault("river", {}).setdefault("masking", {})["channel_mask_clip_depth"] = bool(ok_dep)
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


    # Optional: mask cached river outputs to waffles coastline (if available)
    try:
        if cfg.mask_river_to_waffles:
            wm = _choose_waffles_mask_for_river(cfg, report)
            if wm and wm.exists():
                # Safety check: avoid wiping the river raster if the chosen mask has no water
                # in the AOI window (common if we accidentally select an ocean-only mask).
                # If the check fails, skip WAFFLES masking rather than producing an all-nodata product.
                aoi_bounds = _parse_aoi_bbox(str(cfg.aoi))
                water_px = _count_mask_water_pixels(wm, aoi_bounds, water_max=0.5)
                if water_px is not None and water_px <= 0:
                    log.warning("[RIVER] Skipping WAFFLES mask (no water pixels in AOI): %s", wm)
                else:
                    md_ok = _mask_raster_to_waffles(cached_depth_tif, wm, nodata=-9999.0)
                    mb_ok = _mask_raster_to_waffles(cached_bed_tif, wm, nodata=-9999.0)
                    report.setdefault("river", {}).setdefault("masking", {})["waffles_mask"] = str(wm)
                    report.setdefault("river", {}).setdefault("masking", {})["masked_depth"] = bool(md_ok)
                    report.setdefault("river", {}).setdefault("masking", {})["masked_bed"] = bool(mb_ok)
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Strict safety: river deliverables must contain at least one valid pixel after masking.
    try:
        import rasterio
        import numpy as np

        def _has_any_valid(rp: Path, nodata_val: float) -> bool:
            if not rp.exists():
                return False
            with rasterio.open(rp) as ds:
                nod = ds.nodata
                if nod is None:
                    nod = nodata_val
                for _, w in ds.block_windows(1):
                    a = ds.read(1, window=w)
                    m = np.isfinite(a) & (a != nod)
                    if np.any(m):
                        return True
            return False

        nd = float(getattr(cfg, "river_nodata", -9999.0))
        if not _has_any_valid(cached_depth_tif, nd):
            raise RuntimeError("River depth raster has no valid pixels after masking; outputs would be all nodata.")
        if not _has_any_valid(cached_bed_tif, nd):
            raise RuntimeError("River bed elevation raster has no valid pixels after masking; outputs would be all nodata.")
    except Exception as e:
        # Fail closed: producing all-nodata deliverables is worse than aborting.
        log.error("[RIVER][FATAL] %s", e)
        report["river"]["status"] = "failed"
        raise

    out_depth = river_dir / "river_depth_terrain_patch.tif"
    out_bed = river_dir / "river_bottom_navd88_patch.tif"
    for src, dst in [(cached_depth_tif, out_depth), (bed_tif, out_bed)]:
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)
        except Exception:
            shutil.copy(src, dst)

    try:
        apply_depth_metadata(out_depth, depth_reference="terrain_surface")
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    try:
        apply_elevation_metadata(out_bed, vertical_datum="NAVD88")
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    report.setdefault("river", {}).setdefault("outputs", {}).update(
        {
            "depth_terrain": str(out_depth),
            "bottom_elevation": str(out_bed),
        }
    )

    # ------------------------------------------------------------------
    # Constraint summary (explicit; no filename guessing)
    # ------------------------------------------------------------------
    # Prefer the xs_infer constraint sidecar if available.
    try:
        cs = {
            "level": "UNKNOWN",
            "soundings_used": False,
            "drainage_area_used": False,
            "slope_used": False,
            "slope_source": "unknown",
            "wse_source": "unknown",
            "requirement": str(getattr(cfg, "require_river_constraints", "none")),
            "meets_requirement": None,
            "unmet_reasons": [],
        }
        xs_meta = report.get("river", {}).get("constraints", {}).get("xs_mainstem", {})
        if not (isinstance(xs_meta, dict) and xs_meta):
            xs_meta = report.get("river", {}).get("constraints", {}).get("xs", {})
        if isinstance(xs_meta, dict) and xs_meta:
            cs["level"] = str(xs_meta.get("level", "UNKNOWN"))
            cs["soundings_used"] = bool(xs_meta.get("soundings_used", False))
            cs["drainage_area_used"] = bool(xs_meta.get("drainage_area_used", False))
            cs["slope_used"] = bool(xs_meta.get("slope_used", False))
            cs["slope_source"] = str(xs_meta.get("slope_source", "unknown"))
            cs["wse_source"] = str(xs_meta.get("wse_source", "unknown"))

        req = str(getattr(cfg, "require_river_constraints", "none") or "none").lower().strip()
        unmet = []
        if req != "none":
            if "soundings" in req and not cs["soundings_used"]:
                unmet.append("soundings_required")
            if "slope" in req and not cs["slope_used"]:
                unmet.append("slope_required")
            if "da" in req and not cs["drainage_area_used"]:
                unmet.append("drainage_area_required")
            cs["unmet_reasons"] = unmet
            cs["meets_requirement"] = (len(unmet) == 0)

        report.setdefault("river", {})["constraints_summary"] = cs
    except Exception:
        logging.getLogger(__name__).debug("Failed to write constraint summary; continuing.", exc_info=True)

    # Persist cache manifest after successful products
    try:
        manifest_path.write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    return out_depth
# -----------------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------------

def fuse(cfg: BathyConfig, sdb_raster: Optional[Path], river_raster: Optional[Path], report: Dict[str, Any]) -> Optional[Path]:
    """
    Build the final combined bathymetry surface.

    Behavior:
    - If only one source exists, that raster becomes the final surface.
    - If both exist, use bathy_fusion.fuse_bathymetry() with strategy="weighted_overlap"
      and C1 weights (0.70 priority / 0.30 secondary) in overlap pixels, plus gap-filling.

    Notes:
    - This function writes outputs under out_dir/combined/.
    - It records fusion decisions and errors into report["fusion"].
    """
    combined_dir = ensure_dir(cfg.out_dir / "combined")
    out_depth = combined_dir / "bathy_combined_depth.tif"
    out_prov = combined_dir / "bathy_combined_provenance.tif"

    # No sources
    if sdb_raster is None and river_raster is None:
        report["fusion"] = {"status": "failed", "reason": "no sources"}
        return None

    # Only one source (fast path)
    if sdb_raster is None and river_raster is not None:
        shutil.copy2(str(river_raster), str(out_depth))
        report["fusion"] = {"status": "success", "mode": "river_only", "outputs": {"depth": str(out_depth)}}
        return out_depth

    if river_raster is None and sdb_raster is not None:
        shutil.copy2(str(sdb_raster), str(out_depth))
        report["fusion"] = {"status": "success", "mode": "sdb_only", "outputs": {"depth": str(out_depth)}}
        return out_depth

    # Both sources available → weighted overlap fusion (C1)
    pri = (cfg.priority or "sdb").strip().lower()
    if pri not in ("sdb", "river"):
        pri = "sdb"
    other = "river" if pri == "sdb" else "sdb"

    # Template grid: use the priority raster if possible
    template = Path(sdb_raster) if pri == "sdb" else Path(river_raster)


    # ------------------------------------------------------------------
    # Sanitize inputs for fusion:
    # SDB can contain large regions of literal 0.0 on land/no-prediction.
    # That blocks "gap filling" with river because fusion treats 0 as valid.
    # Use the SDB land mask (land=1, water=0) to force those cells to nodata.
    # ------------------------------------------------------------------
    sdb_fuse_path = Path(sdb_raster)
    river_fuse_path = Path(river_raster)

    try:
        lm = find_sdb_land_mask(Path(cfg.out_dir) / "sdb")
        if lm and Path(lm).exists():
            import rasterio
            import numpy as np
            from rasterio.warp import reproject, Resampling

            sanitized = combined_dir / "sdb_for_fusion_sanitized.tif"
            with rasterio.open(str(sdb_fuse_path)) as ds:
                prof = ds.profile.copy()
                prof.update(dtype="float32", nodata=-9999.0, compress="deflate", count=1)
                sdb_arr = ds.read(1).astype("float32")
                sdb_nodata = ds.nodata

                # warp land mask onto SDB grid
                land = np.zeros((ds.height, ds.width), dtype="uint8")
                with rasterio.open(str(lm)) as lm_ds:
                    reproject(
                        source=rasterio.band(lm_ds, 1),
                        destination=land,
                        src_transform=lm_ds.transform,
                        src_crs=lm_ds.crs,
                        dst_transform=ds.transform,
                        dst_crs=ds.crs,
                        resampling=Resampling.nearest,
                        src_nodata=lm_ds.nodata,
                        dst_nodata=0,
                    )

                invalid = (land == 1)
                # existing nodata also invalid
                if sdb_nodata is not None:
                    invalid |= (sdb_arr == sdb_nodata)
                invalid |= ~np.isfinite(sdb_arr)

                # Critical: treat 0.0 on land as invalid (typical mask artifact)
                invalid |= ((sdb_arr == 0.0) & (land == 1))

                sdb_arr2 = sdb_arr.copy()
                sdb_arr2[invalid] = prof["nodata"]

                with rasterio.open(str(sanitized), "w", **prof) as dst:
                    dst.write(sdb_arr2.astype("float32"), 1)

            sdb_fuse_path = sanitized
            report.setdefault("fusion", {}).setdefault("inputs_sanitized", {})["sdb_landmask_applied"] = str(lm)
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    # Heuristic: if the SDB raster is overwhelmingly literal 0.0 (common failure mode),
    # treat 0.0 as nodata for fusion so river can gap-fill.
    try:
        from osgeo import gdal
        import numpy as np
        ds0 = gdal.Open(str(sdb_fuse_path))
        if ds0 is not None:
            b0 = ds0.GetRasterBand(1)
            a0 = b0.ReadAsArray()
            nd0 = b0.GetNoDataValue()
            m = np.isfinite(a0)
            if nd0 is not None:
                m &= (a0 != nd0)
            if np.any(m):
                frac0 = float(np.mean(a0[m] == 0.0))
                # Only trigger when nearly all valid pixels are exactly 0
                if frac0 >= 0.95:
                    sanitized0 = combined_dir / "sdb_for_fusion_zeros_sanitized.tif"
                    a1 = a0.astype(np.float32)
                    nodata = -9999.0
                    a1[~np.isfinite(a1)] = nodata
                    if nd0 is not None:
                        a1[a1 == nd0] = nodata
                    a1[a1 == 0.0] = nodata
                    drv = gdal.GetDriverByName("GTiff")
                    out_ds = drv.Create(
                        str(sanitized0),
                        ds0.RasterXSize,
                        ds0.RasterYSize,
                        1,
                        gdal.GDT_Float32,
                        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
                    )
                    out_ds.SetGeoTransform(ds0.GetGeoTransform())
                    out_ds.SetProjection(ds0.GetProjection())
                    ob = out_ds.GetRasterBand(1)
                    ob.SetNoDataValue(nodata)
                    ob.WriteArray(a1)
                    ob.FlushCache()
                    out_ds.FlushCache()
                    out_ds = None
                    sdb_fuse_path = sanitized0
                    report.setdefault("fusion", {}).setdefault("inputs_sanitized", {})["sdb_zero_sanitized"] = {
                        "path": str(sanitized0),
                        "frac0": frac0,
                        "threshold": 0.95,
                    }
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


    def _simple_union_overlay(sdb_path: Path, river_path: Path, out_path: Path, template_path: Path, pri: str) -> None:
        """Guaranteed combine: fill gaps from the secondary raster into the primary on the template grid.
        This is used as a robustness fallback when weighted fusion fails or yields no usable overlap.
        """
        import rasterio
        import numpy as np
        from rasterio.warp import reproject, Resampling

        nodata = -9999.0

        def _read_align(src_path: Path):
            with rasterio.open(template_path) as tmpl:
                prof = tmpl.profile.copy()
                prof.update(dtype="float32", nodata=nodata, count=1, compress="deflate")
                arr = np.full((tmpl.height, tmpl.width), nodata, dtype=np.float32)
                with rasterio.open(src_path) as src:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tmpl.transform,
                        dst_crs=tmpl.crs,
                        resampling=Resampling.nearest,
                        src_nodata=src.nodata,
                        dst_nodata=nodata,
                    )
            return arr, prof

        sdb_a, prof = _read_align(Path(sdb_path))
        riv_a, _ = _read_align(Path(river_path))

        # Enforce strict domain separation:
        #   - Inside river_domain_mask (mask==1): DO NOT allow SDB to contribute.
        #   - Outside river_domain_mask: DO NOT allow river to contribute (safety).
        # This prevents a common failure where the "combined" output looks like S2 imagery inside rivers.
        try:
            if river_domain_mask_for_fusion is not None and Path(river_domain_mask_for_fusion).exists():
                dm_a, _ = _read_align(Path(river_domain_mask_for_fusion))
                dm = (dm_a > 0.5)
                sdb_a = sdb_a.copy()
                riv_a = riv_a.copy()
                sdb_a[dm] = nodata
                riv_a[~dm] = nodata
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        if pri == "river":
            primary, secondary = riv_a, sdb_a
        else:
            primary, secondary = sdb_a, riv_a

        out = primary.copy()
        take = (out == nodata) & (secondary != nodata)
        out[take] = secondary[take]

        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(out.astype(np.float32), 1)


    def _gdal_union_overlay(sdb_path: Path, river_path: Path, out_path: Path, template_path: Path, pri: str) -> None:
        """GDAL-based union overlay fallback (no rasterio dependency).
        Aligns both rasters onto the template grid, then fills primary nodata with secondary values.
        """
        try:
            from osgeo import gdal
            import numpy as np
        except Exception as ee:
            raise RuntimeError(f"GDAL/NumPy not available for fallback fusion: {ee}")

        nodata = -9999.0

        tmpl = gdal.Open(str(template_path))
        if tmpl is None:
            raise RuntimeError(f"Failed to open template raster: {template_path}")

        gt = tmpl.GetGeoTransform()
        proj = tmpl.GetProjection()
        w = tmpl.RasterXSize
        h = tmpl.RasterYSize
        xres = gt[1]
        yres = abs(gt[5])
        xmin = gt[0]
        ymax = gt[3]
        xmax = xmin + xres * w
        ymin = ymax - yres * h

        warp_opts = gdal.WarpOptions(
            format="MEM",
            outputBounds=(xmin, ymin, xmax, ymax),
            xRes=xres,
            yRes=yres,
            dstSRS=proj if proj else None,
            resampleAlg="near",
            dstNodata=nodata,
            targetAlignedPixels=True,
            multithread=True,
        )

        def _warp_to_mem(p: Path):
            ds = gdal.Warp("", str(p), options=warp_opts)
            if ds is None:
                raise RuntimeError(f"gdal.Warp failed for {p}")
            return ds

        sdb_ds = _warp_to_mem(Path(sdb_path))
        riv_ds = _warp_to_mem(Path(river_path))

        sdb_a = sdb_ds.ReadAsArray().astype(np.float32)
        riv_a = riv_ds.ReadAsArray().astype(np.float32)

        # Optional domain enforcement (same policy as main fusion):
        # suppress SDB in river domain; suppress river outside river domain.
        try:
            _dm = river_domain_mask_for_fusion
            if _dm is not None and Path(_dm).exists():
                dm_ds = _warp_to_mem(Path(_dm))
                dm_a = dm_ds.ReadAsArray()
                dm = np.asarray(dm_a) > 0.5
                sdb_a = sdb_a.copy()
                riv_a = riv_a.copy()
                sdb_a[dm] = nodata
                riv_a[~dm] = nodata
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # ensure finite
        sdb_a[~np.isfinite(sdb_a)] = nodata
        riv_a[~np.isfinite(riv_a)] = nodata

        if pri == "river":
            primary, secondary = riv_a, sdb_a
        else:
            primary, secondary = sdb_a, riv_a

        out = primary.copy()
        take = (out == nodata) & (secondary != nodata)
        out[take] = secondary[take]

        drv = gdal.GetDriverByName("GTiff")
        ds_out = drv.Create(
            str(out_path),
            w,
            h,
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
        )
        if ds_out is None:
            raise RuntimeError(f"Failed to create output raster: {out_path}")

        ds_out.SetGeoTransform(gt)
        if proj:
            ds_out.SetProjection(proj)
        band = ds_out.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.WriteArray(out)
        band.FlushCache()
        ds_out.FlushCache()
        ds_out = None
    try:
        from bathy_fusion import fuse_bathymetry, FusionConfig


        # ------------------------------------------------------------------
        # Domain enforcement (critical):
        #   - River bathymetry must be the ONLY contributor inside the river/channel domain.
        #   - SDB must not overwrite river inside that domain (common "looks like satellite" failure mode).
        # We satisfy this by passing a 1/0 river_domain_mask into bathy_fusion and enabling
        # river_overrides_sdb_in_domain, which suppresses SDB where mask==1.
        # ------------------------------------------------------------------
        river_domain_mask_for_fusion: Optional[Path] = None
        try:
            ro = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
            cand = ro.get("river_channel_mask") or ro.get("channel_mask") or ro.get("river_domain_mask")
            if cand:
                river_domain_mask_for_fusion = Path(str(cand))
        except Exception:
            river_domain_mask_for_fusion = None

        if river_domain_mask_for_fusion is None or (not river_domain_mask_for_fusion.exists()):
            try:
                ch = getattr(cfg, 'river_channel_mask', None)
                river_domain_mask_for_fusion = Path(str(ch)) if ch else None
            except Exception:
                river_domain_mask_for_fusion = None

        if river_domain_mask_for_fusion is not None and river_domain_mask_for_fusion.exists():
            report.setdefault("fusion", {}).setdefault("domain", {})["river_domain_mask"] = str(river_domain_mask_for_fusion)
            report["fusion"]["domain"]["river_overrides_sdb_in_domain"] = True
        else:
            report.setdefault("fusion", {}).setdefault("domain", {})["river_domain_mask"] = None
            report["fusion"]["domain"]["river_overrides_sdb_in_domain"] = False
        cfg_fuse = FusionConfig(
            sdb_raster=Path(sdb_fuse_path) if sdb_fuse_path else None,
            river_raster=Path(river_fuse_path) if river_fuse_path else None,
            measured_raster=None,
            dem_raster=None,
            river_domain_mask=river_domain_mask_for_fusion,
            ocean_domain_mask=getattr(cfg, 'ocean_domain_mask_for_fusion', None),
            river_overrides_sdb_in_domain=bool((river_domain_mask_for_fusion is not None and river_domain_mask_for_fusion.exists()) and (str(cfg.fusion_strategy).strip().lower() != 'seam_blend')),
            out_dir=combined_dir,
            strategy=cfg.fusion_strategy,
            priority_order=[pri, other],
            primary_weight=float(cfg.fusion_primary_weight),
            secondary_weight=float(cfg.fusion_secondary_weight),
            nodata=-9999.0,
            template_raster=template,
            taper_m=float(cfg.fusion_taper_m),
        )

        res = fuse_bathymetry(cfg_fuse)

        if getattr(res, "status", None) != "success" or not getattr(res, "combined_raster", None):
            raise RuntimeError(getattr(res, "error", None) or "fusion did not return a valid combined raster")

        # bathy_fusion writes fixed filenames; copy to our pipeline-stable names
        try:
            shutil.copy2(str(res.combined_raster), str(out_depth))
        except Exception:
            # If copy fails, fall back to the produced path
            out_depth = Path(res.combined_raster)

        if getattr(res, "provenance_raster", None):
            try:
                shutil.copy2(str(res.provenance_raster), str(out_prov))
            except Exception:
                out_prov = Path(res.provenance_raster)

        report["fusion"] = {
            "status": "success",
            "mode": "weighted_overlap",
            "priority": pri,
            "weights": {"primary": float(cfg.fusion_primary_weight), "secondary": float(cfg.fusion_secondary_weight)},
            "outputs": {"depth": str(out_depth), "provenance": str(out_prov) if out_prov else None},
            "bathy_fusion_outputs": {
                "combined": str(getattr(res, "combined_raster", "")),
                "provenance": str(getattr(res, "provenance_raster", "")),
                "uncertainty": str(getattr(res, "uncertainty_raster", "")) if getattr(res, "uncertainty_raster", None) else None,
            },
        }


        # Sanity: ensure the combined raster actually incorporated some river pixels (common failure mode
        # when river was misaligned or fully nodata after reprojection). If not, do a guaranteed union overlay.
        try:
            import rasterio, numpy as np
            has_river = False
            with rasterio.open(str(out_prov if out_prov else out_depth)) as ds:
                arr = ds.read(1)
                # If provenance exists and contains river/blended, we consider river present.
                # Otherwise, fall back to checking if river contributes to combined where sdb is nodata.
            if out_prov and Path(out_prov).exists():
                with rasterio.open(str(out_prov)) as dp:
                    p = dp.read(1)
                    has_river = bool(np.any((p == 2) | (p == 6)))  # PROV_RIVER=2, PROV_BLENDED=6
            if not has_river:
                try:
                    _simple_union_overlay(Path(sdb_fuse_path), Path(river_fuse_path), Path(out_depth), Path(template), pri)
                except Exception:
                    _gdal_union_overlay(Path(sdb_fuse_path), Path(river_fuse_path), Path(out_depth), Path(template), pri)
                report["fusion"]["note"] = "River contribution missing after fusion; applied union overlay fallback."
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)



        # ------------------------------------------------------------------
        # Enforce authoritative XYZ constraints in the final fused raster.
        # This is a hard "burn-in": at pixels containing authoritative soundings,
        # the output bed elevation must equal the sounding bed elevation exactly.
        # ------------------------------------------------------------------
        try:
            import rasterio
            import numpy as np
            from rasterio.transform import rowcol

            xyz_str = str(getattr(cfg, 'river_soundings', '') or '').strip()
            xyz_paths = [p for p in [s.strip() for s in xyz_str.split(',')] if p]
            if xyz_paths and Path(out_depth).exists():
                with rasterio.open(str(out_depth), 'r+') as ds:
                    arr = ds.read(1).astype('float32')
                    nodata = ds.nodata
                    if nodata is None:
                        nodata = -9999.0
                    # Collect per-pixel samples (median if multiple points hit same pixel)
                    h, w = ds.height, ds.width
                    pix_idx = []
                    zvals = []
                    for xp in xyz_paths:
                        fp = Path(xp)
                        if not fp.exists():
                            continue
                        try:
                            dat = np.genfromtxt(str(fp), dtype='float64', delimiter=None)
                            if dat.ndim == 1:
                                dat = dat.reshape(1, -1)
                        except Exception:
                            # try comma-delimited
                            try:
                                dat = np.genfromtxt(str(fp), dtype='float64', delimiter=',')
                                if dat.ndim == 1:
                                    dat = dat.reshape(1, -1)
                            except Exception:
                                continue
                        if dat.size == 0:
                            continue
                        # take first 3 columns
                        if dat.shape[1] < 3:
                            continue
                        x = dat[:, 0]
                        y = dat[:, 1]
                        z = dat[:, 2]
                        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                        if not np.any(m):
                            continue
                        x = x[m]; y = y[m]; z = z[m]
                        rr, cc = rowcol(ds.transform, x, y)
                        rr = np.asarray(rr); cc = np.asarray(cc)
                        mm = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                        if not np.any(mm):
                            continue
                        rr = rr[mm]; cc = cc[mm]; z = z[mm]
                        idx = rr * w + cc
                        pix_idx.append(idx)
                        zvals.append(z)

                    if pix_idx:
                        idx = np.concatenate(pix_idx)
                        zv = np.concatenate(zvals).astype('float32')
                        # median per pixel
                        order = np.argsort(idx)
                        idx = idx[order]
                        zv = zv[order]
                        uniq, start = np.unique(idx, return_index=True)
                        # compute median by splitting (fast enough for typical sizes)
                        med = np.empty_like(uniq, dtype='float32')
                        for i, s0 in enumerate(start):
                            s1 = start[i+1] if i+1 < len(start) else len(idx)
                            med[i] = np.median(zv[s0:s1])
                        rr = (uniq // w).astype('int64')
                        cc = (uniq % w).astype('int64')
                        arr[rr, cc] = med
                        ds.write(arr.astype('float32'), 1)
                report.setdefault('fusion', {}).setdefault('constraints', {})['xyz_burned_into_final'] = True
                report['fusion']['constraints']['xyz_files'] = [str(Path(x).name) for x in xyz_paths]
        except Exception as _burn_e:
            report.setdefault('fusion', {}).setdefault('constraints', {})['xyz_burned_into_final'] = False
            report['fusion']['constraints']['xyz_burn_error'] = str(_burn_e)

        return Path(out_depth)

    except Exception as e:
        # Fall back to priority-copy, but keep the error visible in the report
        err = f"weighted_overlap fusion failed; falling back to priority copy: {e}"
        report["fusion"] = {
            "status": "degraded",
            "mode": "priority_fallback",
            "weighted_overlap_failed": True,
            "fallback_used": True,
            "priority": pri,
            "error": err,
            "outputs": {"depth": str(out_depth)},
        }
        log.warning(err)

        # Robust fallback: union-overlay combine (fill priority nodata with secondary). If that fails, do priority copy.
        try:
            _gdal_union_overlay(Path(sdb_fuse_path), Path(river_fuse_path), Path(out_depth), Path(template), pri)
            report["fusion"]["mode"] = "union_overlay_fallback"
            report["fusion"]["note"] = "Fusion failed; used GDAL union overlay fallback."
            return out_depth
        except Exception as ee2:
            report["fusion"]["note"] = f"Union overlay fallback also failed; priority copy used. union_error={ee2}"

        # Priority copy as last resort
        if pri == "river":
            shutil.copy2(str(river_raster), str(out_depth))
        else:
            shutil.copy2(str(sdb_raster), str(out_depth))
        return out_depth


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Coastal + River Bathymetry Pipeline", conflict_handler="resolve")
    p.add_argument("--aoi", required=True, help="AOI W/E/S/N")
    p.add_argument("--start", dest="start_date", required=True)
    p.add_argument("--end", dest="end_date", required=True)
    # Tile seam stability policy (operational mode)
    # We download/query *all* datasets using a single buffered AOI (aoi_data),
    # then clip outputs/metrics back to the user tile AOI (aoi_tile).
    p.add_argument(
        "--buff",
        type=float,
        default=0.0,
        help=(
            "Single data acquisition buffer as fractional expansion of the user AOI bbox. "
            "Example: --buff 0.10 expands width/height by 10% (5% each side). Use 0 for no buffer."
        ),
    )

    # Deprecated legacy buffer (enforced/ignored; use --buff)
    p.add_argument(
        "--tile-buffer-km",
        type=float,
        default=0.0,
        help="DEPRECATED (use --buff). Legacy km buffer; kept for backwards compatibility.",
    )

    p.add_argument("--tile-edge-taper-km", type=float, default=2.0,
                   help="Within this distance (km) of the tile edges, blend toward a low-frequency smooth field to improve seam consistency (default 2 km). Set to 0 to disable.")
    p.add_argument("--tile-edge-smooth-sigma-km", type=float, default=10.0,
                   help="Gaussian smooth sigma (km) for the low-frequency edge taper reference (default 10 km).")
    p.add_argument("--tile-edge-metrics-band-km", type=float, default=2.0,
                   help="Edge band width (km) used for seam diagnostics in run summaries (default 2 km).")
    p.add_argument("--no-tile-edge-taper", dest="tile_edge_taper_enabled", action="store_false",
                   help="Disable tile edge tapering (use raw clipped outputs).")
    p.set_defaults(tile_edge_taper_enabled=True)

    # Adjacent-tile seam comparison (explicit inputs; no guessing)
    p.add_argument(
        "--seam-compare-with-io",
        action="append",
        default=[],
        help=(
            "Path to a neighboring tile's io_manifest.json to compute seam metrics against. "
            "May be specified multiple times. Explicit-only (no auto-discovery)."
        ),
    )
    p.add_argument(
        "--seam-strip-px",
        type=int,
        default=3,
        help="Strip width in pixels sampled on each side of the seam (default 3).",
    )
    p.add_argument(
        "--seam-compare-with-io-list",
        type=str,
        default=None,
        help=(
            "Path to a text file listing neighboring io_manifest.json paths (one per line; '#' comments allowed) "
            "to compute seam metrics against in batch. Explicit-only (no auto-discovery)."
        ),
    )

    # SDB cross-tile consistency (DEFAULT): bounded model bank reservoir + periodic retrain
    p.add_argument("--sdb-model-bank", default="auto",
                   help="Model bank directory for incremental SDB training. 'auto' => <cache-root>/model_bank/sdb_global_v1")
    p.add_argument("--no-sdb-model-bank", dest="sdb_model_bank_enabled", action="store_false",
                   help="Disable model bank; train only on this AOI (not recommended for seamless tiling).")
    p.set_defaults(sdb_model_bank_enabled=True)
    p.add_argument("--sdb-bank-max-samples", type=int, default=100000,
                   help="Max samples to keep in the model bank reservoir (bounded disk).")
    p.add_argument("--sdb-bank-seed", type=int, default=1337,
                   help="Seed for deterministic reservoir sampling in the model bank.")
    p.add_argument("--sdb-bank-retrain-min-new", type=int, default=2000,
                   help="Only retrain the RF when at least this many new samples have been added to the bank since last training.")

    # Deprecated: regional model cache (kept for backwards compatibility; OFF by default)
    p.add_argument("--sdb-model-cache-key", default="auto",
                   help="[DEPRECATED] Key for regional SDB model reuse (prefer model bank).")
    p.add_argument("--sdb-model-cache", dest="sdb_model_cache_enabled", action="store_true",
                   help="[DEPRECATED] Enable regional SDB model reuse (prefer model bank).")
    p.add_argument("--no-sdb-model-cache", dest="sdb_model_cache_enabled", action="store_false",
                   help="[DEPRECATED] Disable regional SDB model reuse.")
    p.set_defaults(sdb_model_cache_enabled=False)


    p.add_argument("--out-dir", default="output/unified")
    p.add_argument("--save-intermediates", action="store_true", default=False,
                   help="Preserve intermediate artifacts (rasters/caches) by moving them under <out-dir>/<intermediates-dirname>/... (default: off; intermediates are deleted).")
    p.add_argument("--intermediates-dirname", default="debug",
                   help="Subfolder name under <out-dir> to store intermediates when --save-intermediates is enabled (default: debug).")
    p.add_argument("--methods", default="sdb,river,fuse")
    p.add_argument("--priority", default="sdb", choices=["sdb", "river"])

    # SDB passthrough
    p.add_argument("--cloud", type=int, default=70)
    p.add_argument("--icesat", default="all_atl")
    p.add_argument("--sdb-mode", default="all_sdb")
    p.add_argument("--cache-root", default="cache")
    p.add_argument("--align-mode", default="median")

    # S2 sun-glint correction (Hedley-style). Only used when methods include 'sdb'.
    p.add_argument("--glint-correct", action="store_true", default=False,
                   help="Enable Hedley-style sun-glint correction on Sentinel-2 composite reflectance (SDB only).")
    p.add_argument("--glint-nir-band", default="B08",
                   help="NIR band used as glint proxy (default B08).")
    p.add_argument("--glint-vis-bands", default="B02,B03,B04",
                   help="Comma-separated visible bands to correct (default B02,B03,B04).")
    p.add_argument("--glint-nir-min-percentile", type=float, default=1.0,
                   help="Percentile for baseline NIR (R_nir,min) over stable-water mask (default 1.0).")
    p.add_argument("--glint-deepwater-b02-max", type=float, default=0.20,
                   help="Max B02 reflectance allowed in deep/stable water mask (default 0.20).")
    p.add_argument("--glint-min-samples", type=int, default=5000,
                   help="Minimum number of stable-water pixels required to fit betas (default 5000).")
    p.add_argument("--glint-max-samples", type=int, default=2000000,
                   help="Max number of samples used to fit betas (default 2000000).")
    p.add_argument("--glint-clip-min", type=float, default=1e-6,
                   help="Clip minimum for corrected reflectance to avoid downstream log/ratio issues (default 1e-6).")


    # River
    p.add_argument("--river-dem", default=None)

    # River constraint guardrails
    p.add_argument(
        "--require-river-constraints",
        default="none",
        choices=[
            "none",
            "slope",
            "slope+da",
            "soundings",
            "soundings+slope",
            "soundings+slope+da",
        ],
        help=(
            "Guardrail: require specific constraints for river bathy before fusion. "
            "If unmet, river outputs may still be written but fusion will be skipped and the run report will be flagged. "
            "Choices: none, slope, slope+da, soundings, soundings+slope, soundings+slope+da. Default: none."
        ),
    )

    p.add_argument("--working-srs", default="auto",
                   help="Working horizontal CRS for meter-based operations (thinning, river DEM warps). "
                        "Use 'auto' to detect from cached Sentinel-2 RGB_10m.tif or fall back to AOI-center UTM (WGS84).")
    p.add_argument("--working-vcrs-epsg", type=int, default=5703,
                   help="Working vertical CRS EPSG code (default 5703 = NAVD88 height).")
    p.add_argument("--final-out-srs", default="EPSG:4269",
                   help="Final output raster horizontal CRS (default EPSG:4269 = NAD83 geographic). Depth values are relative to water surface; vertical datum is stored as metadata tags, not CRS.")

    # Vertical datum transformation (MSL → NAVD88)
    p.add_argument("--convert-sdb-to-navd88", action="store_true", default=False,
                   help="Convert SDB from MSL to NAVD88 vertical datum using dlim. "
                        "Output path is recorded in the run reports/manifests. Requires dlim (CUDEM) on PATH.")
    p.add_argument("--sdb-source-vdatum", default="epsg:4269+5714",
                   help="Source compound EPSG (default: epsg:4269+5714 = NAD83+MSL).")
    p.add_argument("--sdb-target-vdatum", default="epsg:4269+5703",
                   help="Target compound EPSG (default: epsg:4269+5703 = NAD83+NAVD88).")

    p.add_argument("--river-dem-auto", action="store_true", default=True,
                   help="If --river-dem is not provided, auto-download TNM 1/3 arc-sec DEM via `fetches` and build a clipped DEM.")
    p.add_argument("--no-river-dem-auto", dest="river_dem_auto", action="store_false",
                   help="Disable auto river DEM download/build; river method requires --river-dem.")
    p.add_argument("--river-dem-source", default="tnm:datasets=3",
                   help="CUDEM fetches source string for river DEM auto-download (default tnm:datasets=3).")
    p.add_argument("--river-dem-res-m", type=float, default=10.0,
                   help="Target resolution (meters) for the auto-built river DEM in working CRS (default 10).")
    # Unified extra bathymetry soundings input (preferred user-facing name).
    # You can repeat this flag, pass a comma-separated list, or pass a directory of files.
    p.add_argument(
        "--extra-xyz",
        dest="extra_xyz",
        action="append",
        default=[],
        help="Extra bathymetry soundings (file, comma-list, or directory). Supports .xyz/.csv/.txt/.dat/.gpkg/.shp/.geojson. Can be repeated."
    )


    p.add_argument("--extra-xyz-crs", default="EPSG:4326", help="CRS for raw extra XYZ files (default EPSG:4326).")
    p.add_argument(
        "--extra-xyz-cudem",
        dest="extra_xyz_cudem",
        action="append",
        default=[],
        help=("Auto-download external soundings with CUDEM dlim into cache (e.g., hydronos, ehydro). "
              "Provide a comma-separated list and/or repeat the flag. Example: --extra-xyz-cudem hydronos,ehydro"),
    )
    p.add_argument(
        "--extra-xyz-cudem-crs",
        default="epsg:4269+5714",
        help=("Compound CRS string passed to dlim -P for --extra-xyz-cudem outputs. "
              "Default: epsg:4269+5714 = NAD83 + MSL height. This ensures downloaded soundings "
              "(which are typically in MLLW) are converted to MSL to match ICESat-2 training data. "
              "Use --convert-sdb-to-navd88 to convert final SDB output from MSL to NAVD88."),
    )

    p.add_argument(
        "--extra-xyz-cudem-thin-res-m",
        type=float,
        default=10.0,
        help=("If set, apply CUDEM dlim filtering/thinning to auto-downloaded XYZ (default: 10 meters). "
              "Implemented as -F block_thin:res=<value>. Set to 0 to disable."),
    )
    p.add_argument(
        "--extra-xyz-cudem-filter",
        default=None,
        help=("Explicit CUDEM dlim -F filter string for auto-downloaded XYZ (overrides --extra-xyz-cudem-thin-res-m). "
              "Example: block_thin:res=10"),
    )
    p.add_argument(
        "--extra-xyz-cudem-force",
        action="store_true",
        help="Force re-download of --extra-xyz-cudem soundings even if cached file exists.",
    )

    # Backwards-compatibility alias (hidden from --help)
    p.add_argument(
        "--river-soundings",
        dest="extra_xyz",
        action="append",
        help=argparse.SUPPRESS
    )
    
    # Adaptive Spatial Sampling (v0.7.0+, ENABLED BY DEFAULT in v0.7.1)
    p.add_argument("--disable-adaptive-sampling", dest="enable_adaptive_sampling",
                   action="store_false", default=True,
                   help="Disable adaptive spatial sampling (sampling is ON by default)")
    p.add_argument("--sampling-target-points", type=int, default=2000,
                   help="Target number of training points after adaptive sampling (default: 2000)")
    p.add_argument("--sampling-min-threshold", type=int, default=3000,
                   help="Minimum input points before adaptive sampling is applied (default: 3000)")
    p.add_argument("--sampling-max-gap-m", type=float, default=100.0,
                   help="Maximum spatial gap allowed in meters (default: 100m)")
    
    # River network parameters
    p.add_argument("--river-hydrography-source", default="arcgis", choices=["arcgis","arcgis_tnm","tnm"],
                   help="Hydrography acquisition for river network: arcgis (default) queries ArcGIS REST only; arcgis_tnm uses TNM as a fallback; tnm prefers TNM (and may still fall back to ArcGIS).")
    p.add_argument("--no-tnm", action="store_true", help="Disable TNM download (if using local flowlines)")
    p.add_argument("--tnm-dataset", default="NHDPlusHR")
    p.add_argument("--snap-m", type=float, default=30.0)


    # River bathymetry method selection
    p.add_argument("--river-method", choices=["hybrid","skeleton", "xs"], default="hybrid",
               help=("River bathy method. 'hybrid' (default) runs XS only on the mainstem and uses the skeleton method elsewhere, then combines them so mainstem depths are continuous. 'skeleton' uses a raster distance-transform channel skeleton (recommended for dense tributaries/meanders/tidal channels). 'xs' uses cross-sections everywhere (more artifact-prone at junctions)."))
    # Skeleton (distance-transform) parameters: domain mask (river vs ocean) + channel profile
    p.add_argument("--river-channel-buffer-m", type=float, default=400.0,
               help="Buffer around NHD flowlines used to define candidate river corridor (meters).")
    p.add_argument("--river-max-channel-width-m", type=float, default=600.0,
               help="Maximum channel width allowed inside the corridor (meters). Prevents filling open bays/ocean.")
    p.add_argument("--river-mainstem-min-order", type=int, default=5,
               help="Stream order threshold for allowing larger widths (if stream order attribute exists).")
    p.add_argument("--river-max-mainstem-width-m", type=float, default=2500.0,
               help="Maximum channel width allowed for mainstem corridor pixels (meters).")

    # Optional: constrain river predictions to NHDArea polygons when available

    # River channel domain policy
    # - auto: prefer NHDArea river/stream polygons when available; otherwise fall back to buffered flowline corridor
    # - nhdarea: require NHDArea river/stream polygons (excludes lakes)
    # - corridor: buffered flowline corridor only
    p.add_argument("--river-channel-source", dest="river_channel_source",
               choices=["auto", "nhdarea", "corridor"], default="auto",
               help="Channel mask source policy for skeleton method: auto|nhdarea|corridor (default: auto).")

    # Connectivity filter for channel domain (recommended on): keeps only water components connected to the river network seed.
    p.add_argument("--river-connectivity-filter", dest="river_connectivity_filter",
               action="store_true", default=True,
               help="Enable connectivity filtering when building the river channel mask (default: on).")
    p.add_argument("--no-river-connectivity-filter", dest="river_connectivity_filter",
               action="store_false",
               help="Disable connectivity filtering when building the river channel mask.")

    # NHDArea filtering: by default keep Stream/River polygons only (FType=460). This excludes lakes/ponds.
    p.add_argument("--river-nhdarea-allow-ftype", dest="river_nhdarea_allow_ftype", default="460",
               help="Comma-separated list of allowed NHDArea FType codes to treat as river/stream area (default: 460 Stream/River).")
    p.add_argument("--river-nhdarea-allow-fcode", dest="river_nhdarea_allow_fcode", default=None,
               help="Optional comma-separated list of allowed NHDArea FCode values (used only if FType is not present).")

    p.add_argument("--no-river-nhdarea", dest="river_use_nhdarea", action="store_false", default=True,
               help="Disable NHDArea polygon constraint for river domain (if available).")
    p.add_argument("--river-nhdarea-layer", default="nhdarea_clip",
               help="Layer name in river_network.gpkg containing NHDArea polygons (default nhdarea_clip).")
    p.add_argument("--river-ocean-keep-dist-m", dest="river_ocean_keep_dist_m", type=float, default=0.0,
               help="Allow ocean-connected water within this distance (m) of flowlines when building the river channel mask. Useful for tidal river mouths/estuaries where the mainstem is classified as ocean water. 0 disables.")
    p.add_argument("--river-shape-exp", type=float, default=0.5,
               help="Depth profile exponent for skeleton method (0.5 ~ U-shape; 1.0 ~ V-shape).")
    p.add_argument("--river-dmax-min-m", type=float, default=0.5, help="Clamp minimum Dmax prior (meters).")
    p.add_argument("--river-dmax-max-m", type=float, default=30.0, help="Clamp maximum Dmax prior (meters).")
    # Optional: longitudinal bed profile constraints (skeleton method)
    p.add_argument("--river-bed-profile-max-slope", dest="river_bed_profile_max_slope", type=float, default=0.0,
               help="Max absolute bed slope |dz/ds| along the river skeleton (m/m). 0 disables.")
    p.add_argument("--river-bed-profile-max-curv", dest="river_bed_profile_max_curv", type=float, default=0.0,
               help="Max absolute bed curvature |d2z/ds2| along the river skeleton (1/m). 0 disables.")
    p.add_argument("--river-bed-profile-step-m", dest="river_bed_profile_step_m", type=float, default=25.0,
               help="Sampling step (m) along flowlines when building the bed profile constraint (default: 25).")
    p.add_argument("--river-bed-profile-strength", dest="river_bed_profile_strength", type=float, default=0.6,
               help="Blend strength (0..1) for applying bed profile constraints (default: 0.6).")
    p.add_argument("--river-bed-profile-power", dest="river_bed_profile_power", type=float, default=2.0,
               help="Distance-decay power for spreading bed profile corrections from the skeleton (default: 2.0).")
    p.add_argument("--river-save-skeleton-debug", action="store_true", default=False,
               help="Write skeleton debug rasters (r, d_bank, d_center, dmax, wse).")
    p.add_argument("--river-skeleton-wse-mode", dest="river_skeleton_wse_mode", default="bank",
               choices=["bank", "skeleton", "bank_profile"],
               help="Skeleton WSE proxy mode: 'bank' uses bank-adjacent DEM samples (recommended); 'skeleton' uses in-channel centerline sampling (legacy).")
    p.add_argument("--river-skeleton-wse-smooth-sigma-m", dest="river_skeleton_wse_smooth_sigma_m", type=float, default=0.0,
               help="Optional Gaussian smoothing sigma (m) for the skeleton WSE proxy field inside the channel.")
# Longitudinal WSE profile controls (river-skeleton-wse-mode=bank_profile)
    p.add_argument("--river-skeleton-wse-profile-step-m", dest="river_skeleton_wse_profile_step_m", type=float, default=20.0,
               help="Densification step (m) for sampling along flowlines when building a longitudinal WSE profile.")
    p.add_argument("--river-skeleton-wse-profile-resample-m", dest="river_skeleton_wse_profile_resample_m", type=float, default=20.0,
               help="Resample step (m) for 1D WSE profile smoothing along flow distance.")
    p.add_argument("--river-skeleton-wse-profile-smooth-sigma-m", dest="river_skeleton_wse_profile_smooth_sigma_m", type=float, default=200.0,
               help="Gaussian smoothing sigma (m) applied to the 1D WSE profile along flow distance.")
    p.add_argument("--river-skeleton-wse-profile-max-slope", dest="river_skeleton_wse_profile_max_slope", type=float, default=0.005,
               help="Optional maximum absolute slope (m/m) enforced along the 1D WSE profile (0 to disable).")
    p.add_argument("--river-skeleton-wse-profile-min-samples", dest="river_skeleton_wse_profile_min_samples", type=int, default=10,
               help="Minimum number of valid samples along flowlines to build a profile; otherwise falls back to wse-mode=bank.")
    p.add_argument("--river-skeleton-wse-profile-max-query-dist-m", dest="river_skeleton_wse_profile_max_query_dist_m", type=float, default=250.0,
               help="Maximum XY distance (m) for assigning profile samples to centerline pixels (0 to disable). Helps avoid cross-reach snapping.")


    # Optional: SWOT RiverSP anchoring for river-skeleton-wse-mode=bank_profile
    p.add_argument("--river-swot-riversp", dest="river_swot_riversp", nargs="+", default=None,
               help="One or more SWOT RiverSP vector files (reach or node product; e.g., .shp/.gpkg/.geojson) with WSE. Used only when river-skeleton-wse-mode=bank_profile.")

    # Auto-fetch (optional): if --river-swot-riversp is omitted, attempt to fetch RiverSP via Earthdata/PO.DAAC.
    p.add_argument("--river-swot-auto", dest="river_swot_auto", action="store_true", default=True,
               help="Enable auto-fetch of SWOT RiverSP when --river-swot-riversp is not provided (default: enabled). Requires Earthdata login (~/.netrc or EARTHDATA_USERNAME/EARTHDATA_PASSWORD).")
    p.add_argument("--no-river-swot-auto", dest="river_swot_auto", action="store_false",
               help="Disable auto-fetch of SWOT RiverSP (only use SWOT if --river-swot-riversp is provided).")
    p.add_argument("--river-swot-cache-root", dest="river_swot_cache_root", default=None,
               help="Cache root for auto-fetched SWOT RiverSP (default: <cache-root>/swot).")
    p.add_argument("--river-swot-product", dest="river_swot_product", choices=["reach","node"], default="reach",
               help="RiverSP product to search when auto-fetching: reach (default) or node.")
    p.add_argument("--river-swot-shortname", dest="river_swot_shortname", default=None,
               help="Optional PO.DAAC short_name to use for auto-fetch (advanced). If omitted, a best-effort search is performed.")

    p.add_argument("--river-swot-wse-field", dest="river_swot_wse_field", default=None,
               help="Column name for SWOT WSE in the RiverSP file(s). If omitted, common candidates will be searched.")
    p.add_argument("--river-swot-qual-field", dest="river_swot_qual_field", default=None,
               help="Optional column name for a SWOT quality flag; if provided, values >0 are treated as bad and filtered out.")
    p.add_argument("--river-swot-max-dist-m", dest="river_swot_max_dist_m", type=float, default=300.0,
               help="Max distance (m) from a flowline sample point to accept a SWOT WSE observation.")
    p.add_argument("--river-swot-min-samples", dest="river_swot_min_samples", type=int, default=5,
               help="Minimum number of SWOT samples on a flowline required to apply anchoring corrections.")
    p.add_argument("--river-swot-correct-sigma-m", dest="river_swot_correct_sigma_m", type=float, default=2000.0,
               help="Smoothing scale (m) for along-channel SWOT correction (Gaussian sigma along distance).")
    p.add_argument("--river-swot-weight", dest="river_swot_weight", type=float, default=1.0,
               help="Weight (0..1) to apply SWOT correction to the bank-derived WSE profile.")
    p.add_argument("--river-swot-max-correction-m", dest="river_swot_max_correction_m", type=float, default=5.0,
               help="Clamp the along-channel correction magnitude (m) applied from SWOT (helps avoid datum/offset mistakes).")
    p.add_argument("--river-swot-wse-offset-m", dest="river_swot_wse_offset_m", type=float, default=0.0,
               help="Constant offset (m) added to SWOT WSE before use. Use to reconcile vertical datums until full datum transform is implemented.")
    p.add_argument("--river-skeleton-junction-mode", dest="river_skeleton_junction_mode", default="smooth",
               choices=["smooth","mask","none"],
               help="How to handle confluence/junction zones (degree>=3 graph nodes): smooth (default), mask, or none.")
    p.add_argument("--river-skeleton-junction-buffer-m", dest="river_skeleton_junction_buffer_m", type=float, default=120.0,
               help="Buffer radius (m) around junction nodes used for smoothing/masking.")
    p.add_argument("--river-skeleton-junction-degree-min", dest="river_skeleton_junction_degree_min", type=int, default=3,
               help="Minimum graph node degree to be treated as a junction.")
    p.add_argument("--river-skeleton-junction-smooth-sigma-m", dest="river_skeleton_junction_smooth_sigma_m", type=float, default=80.0,
               help="Gaussian smoothing sigma (m) used in junction zones when mode=smooth.")
    p.add_argument("--river-skeleton-junction-max-width-m", dest="river_skeleton_junction_max_width_m", type=float, default=300.0,
               help="Limit junction smoothing/masking to pixels with estimated channel width <= this (m). Helps avoid over-smoothing wide confluences.")
    # Curvature-driven asymmetry (outer-bank deeper in bends)
    p.add_argument("--river-skeleton-asymmetry-mode", dest="river_skeleton_asymmetry_mode",
               choices=["none","curvature"], default="none",
               help="Optional curvature-driven asymmetry: biases depth toward the outer bank using signed centerline curvature.")
    p.add_argument("--river-skeleton-asymmetry-strength", dest="river_skeleton_asymmetry_strength", type=float, default=0.25,
               help="Strength of curvature-driven r-shift (dimensionless; typical 0.1–0.4).")
    p.add_argument("--river-skeleton-asymmetry-curv-ref", dest="river_skeleton_asymmetry_curv_ref", type=float, default=0.002,
               help="Reference curvature (1/m) for scaling (e.g., 0.002 ~ 500 m radius).")
    p.add_argument("--river-skeleton-asymmetry-max-shift", dest="river_skeleton_asymmetry_max_shift", type=float, default=0.20,
               help="Max absolute shift applied to r (clamped).")
    p.add_argument("--river-skeleton-asymmetry-min-width-m", dest="river_skeleton_asymmetry_min_width_m", type=float, default=10.0,
               help="Only apply asymmetry where estimated channel width >= this (m).")
    p.add_argument("--river-skeleton-asymmetry-min-curv", dest="river_skeleton_asymmetry_min_curv", type=float, default=0.0005,
               help="Only apply asymmetry where |curvature| >= this (1/m).")
    p.add_argument("--river-skeleton-asymmetry-densify-step-m", dest="river_skeleton_asymmetry_densify_step_m", type=float, default=20.0,
               help="Vertex spacing (m) used when estimating curvature/tangent from flowlines.")

    p.add_argument("--river-swot-offset-mode", dest="river_swot_offset_mode",
                   choices=["none", "median_mad"], default="median_mad",
                   help=("Vertical datum reconciliation mode for SWOT WSE vs bank_profile WSE. "
                         "'median_mad' estimates a robust constant offset from overlapping samples and subtracts it from SWOT WSE. "
                         "'none' disables auto offset estimation (use --river-swot-wse-offset-m instead)."))
    p.add_argument("--river-swot-offset-min-samples", dest="river_swot_offset_min_samples", type=int, default=25,
                   help="Minimum number of SWOT samples required to estimate an auto vertical offset.")
    p.add_argument("--river-swot-offset-mad-z", dest="river_swot_offset_mad_z", type=float, default=3.5,
                   help="MAD-based outlier rejection threshold (in robust-sigma units) for auto vertical offset estimation.")
    p.add_argument("--river-swot-offset-max-abs-m", dest="river_swot_offset_max_abs_m", type=float, default=10.0,
                   help="Clamp absolute value of the auto-estimated vertical offset (m). If exceeded, it is clamped and a warning is logged.")


    p.add_argument("--river-soundings-mode", choices=["auto","depth_pos","depth_neg","bed_elev"], default="auto",
                   help="How to interpret extra XYZ Z values for river skeleton: auto/depth_pos/depth_neg (depths) or bed_elev (bed elevations, same vertical datum as DEM).")
    p.add_argument("--river-soundings-max-dist-m", type=float, default=10000.0,
                   help="Max distance (m) for extra XYZ to influence the skeleton Dmax prior.")
    p.add_argument("--river-soundings-min-r", type=float, default=0.25,
                   help="Minimum r used when converting sounding depth -> implied Dmax (stabilizes near banks).")
    p.add_argument("--no-river-soundings-enforce", dest="river_soundings_enforce", action="store_false", default=True,
               help="Disable enforcing observed soundings at their grid cells (default enforces).")
    p.add_argument("--river-authoritative-bed", default=None,
                   help="Optional authoritative river bed elevation raster to blend/enforce inside the river mask (same vertical datum as DEM).")
    p.add_argument("--river-authoritative-bed-max-dist-m", type=float, default=2000.0,
                   help="Max distance (m) from authoritative bed pixels to influence residual blending.")
    p.add_argument("--river-residual-blend-sigma-m", type=float, default=600.0,
                   help="Gaussian sigma (m) for residual blending smoothing. 0 disables smoothing (nearest-only).")
    p.add_argument("--xs-spacing-m", type=float, default=200.0)
    p.add_argument("--xs-length-m", type=float, default=300.0)

    p.add_argument("--xs-smoothing-window-m", type=float, default=0.0,
                   help="Smoothing window (m) for XS orientation tangents. 0=auto (uses xs-spacing-m).")
    p.add_argument("--xs-deconflict-tol-m", type=float, default=2.0,
                   help="Endpoint tolerance (m) when identifying intersecting cross-sections.")
    p.add_argument("--xs-junction-snap-m", type=float, default=30.0,
                   help="Snapping scale (m) for junction detection from reach endpoints.")
    p.add_argument("--xs-junction-buffer-m", type=float, default=75.0,
                   help="Skip XS within this distance (m) of junction nodes.")
    p.add_argument("--xs-densify-step-m", type=float, default=20.0,
                   help="Densify centerlines to this vertex spacing (m) before computing tangents.")

    p.add_argument("--no-xs-trim-overlaps", dest="xs_trim_overlaps", action="store_false",
                   help="Disable local overlap trimming between adjacent XS.")
    p.set_defaults(xs_trim_overlaps=True)

    p.add_argument("--no-xs-global-deconflict", dest="xs_global_deconflict", action="store_false",
                   help="Disable dropping XS that intersect non-adjacent XS within a reach.")
    p.set_defaults(xs_global_deconflict=True)

    p.add_argument("--no-xs-skip-junctions", dest="xs_skip_junctions", action="store_false",
                   help="Do not skip XS near confluences/junctions.")
    p.set_defaults(xs_skip_junctions=True)

    p.add_argument("--river-thalweg-only", action="store_true",
                   help="Use a thalweg-only control spine (1 control point per cross-section) for river interpolation; also builds corridor from buffered thalweg spine to avoid cross-channel ribbing.")
    p.add_argument("--river-thalweg-densify-factor", type=float, default=0.5,
                   help="Densify thalweg spine vertices to this fraction of the river template raster pixel size (default 0.5 => >=2 vertices per pixel).")
    p.add_argument("--river-thalweg-densify-step-m", type=float, default=None,
                   help="Explicit thalweg spine densify step in meters. Overrides --river-thalweg-densify-factor if provided.")

    # River patch rasterization / interpolation options (forwarded to xs_infer_bathy_raster.py)
    p.add_argument("--river-continuous", choices=["median", "walid", "aidw", "aniso", "walid_aniso"], default="walid_aniso",
                   help="How to produce the river bed patch raster surface from inferred bathy points (see xs_infer_bathy_raster.py --continuous).")
    p.add_argument("--river-continuous-buffer-m", type=float, default=None,
                   help="Buffer (meters) around points used to define interpolation corridor. Default scales with pixel size.")
    p.add_argument("--river-continuous-k", type=int, default=12, help="K nearest points used for IDW/AIDW/anisotropic modes.")
    p.add_argument("--river-idw-power", type=float, default=2.0, help="IDW power for --river-continuous modes.")
    p.add_argument("--river-aniso-along-scale-m", type=float, default=500.0, help="Anisotropic IDW along-channel scale (meters).")
    p.add_argument("--river-aniso-cross-scale-m", type=float, default=30.0, help="Anisotropic IDW cross-channel scale (meters).")
    p.add_argument("--river-thalweg-weight", type=float, default=6.0, help="Extra influence for thalweg control points in walid modes.")
    p.add_argument("--river-overlap-reducer", choices=["min", "median"], default="min",
                   help="When multiple points fall in the same output pixel, how to collapse them. 'min' keeps the deeper bed.")
    p.add_argument("--river-nodata", type=float, default=-9999.0, help="Nodata value for river float rasters.")
    # River depth inference priors / anchors (passed through to xs_infer_bathy_raster.py)
    p.add_argument("--river-prior-mode", choices=["powerlaw", "multivariate"], default="powerlaw",
                   help=("River Dmax prior mode. 'powerlaw' uses Dmax=a*W^b. "
                         "'multivariate' uses W plus reach attributes (drainage area, slope). "
                         "For regional curves DA→Dbkf (Dbkf=c*DA^f), use multivariate with mv_bw=0, mv_ba=f, mv_a0=c."))
    p.add_argument("--river-mv-a0", type=float, default=0.18)
    p.add_argument("--river-mv-bw", type=float, default=0.50)
    p.add_argument("--river-mv-ba", type=float, default=0.0)
    p.add_argument("--river-mv-bs", type=float, default=-0.10)
    p.add_argument("--river-mv-eps-a", type=float, default=1.0)
    p.add_argument("--river-mv-eps-s", type=float, default=1e-4)
    # Slope proxy stabilization (used when slope attribute is missing)
    p.add_argument("--river-slope-proxy-window", type=int, default=9, help="Rolling median window (XS count) for WSE smoothing before slope differencing.")
    p.add_argument("--river-slope-min", type=float, default=1e-5, help="Minimum plausible slope (m/m) for slope proxy.")
    p.add_argument("--river-slope-max", type=float, default=0.05, help="Maximum plausible slope (m/m) for slope proxy.")
    p.add_argument("--river-slope-proxy-min-n", type=int, default=7, help="Minimum XS per river_id required to compute slope proxy.")

    # Longitudinal WSE profile fitting (preferred slope proxy when reach slope attribute is missing)
    p.add_argument("--river-wse-profile-enabled", dest="river_wse_profile_enabled", action="store_true", help="Enable longitudinal WSE profile fitting (default).")
    p.add_argument("--no-river-wse-profile", dest="river_wse_profile_enabled", action="store_false", help="Disable WSE profile fitting; use legacy slope proxy only.")
    p.set_defaults(river_wse_profile_enabled=True)
    p.add_argument("--river-wse-profile-window", type=int, default=9, help="Rolling window (XS count) for WSE profile smoothing.")
    p.add_argument("--river-wse-profile-min-n", type=int, default=7, help="Minimum XS per river_id required to fit a WSE profile.")
    p.add_argument("--river-wse-profile-monotonic", dest="river_wse_profile_monotonic", action="store_true", help="Enforce monotonic WSE along stationing (default).")
    p.add_argument("--no-river-wse-profile-monotonic", dest="river_wse_profile_monotonic", action="store_false", help="Disable monotonic constraint.")
    p.set_defaults(river_wse_profile_monotonic=True)

    # Slope proxy stabilization (used when slope attribute is missing)

    # Option A: USGS discharge *measurement* anchors
    p.add_argument("--river-usgs-sites", default=None,
                   help="Comma-separated USGS site numbers to use as anchors (e.g., 01646500,01651000).")
    p.add_argument("--river-usgs-start", default=None, help="Start date YYYY-MM-DD for USGS discharge measurements.")
    p.add_argument("--river-usgs-end", default=None, help="End date YYYY-MM-DD for USGS discharge measurements.")
    p.add_argument("--river-usgs-cache-dir", default=None, help="Optional cache directory for NWIS responses.")
    p.add_argument("--river-usgs-max-dist-m", type=float, default=5000.0,
                   help="Max distance (m) from gage to XS center for applying USGS anchor.")
    p.add_argument("--river-usgs-mean-to-dmax", default="auto",
                   help="Convert mean depth (area/width) to Dmax for anchor fitting. Use a number (e.g., 1.3) or 'auto' to use 2/(1+bottom_width_frac).")
    p.add_argument("--river-usgs-a-stat", choices=["median", "p90", "mean"], default="median",
                   help="How to aggregate a_site from multiple discharge measurements.")
    p.add_argument("--river-usgs-q-quantile-lo", type=float, default=0.20, help="Lower discharge quantile for filtering USGS measurements (0-1).")
    p.add_argument("--river-usgs-q-quantile-hi", type=float, default=0.80, help="Upper discharge quantile for filtering USGS measurements (0-1).")
    p.add_argument("--river-usgs-a-cv-warn", type=float, default=0.50, help="Warn/soften USGS anchor when CV of fitted a-values exceeds this.")
    p.add_argument("--river-usgs-width-ratio-max", type=float, default=3.0, help="Threshold for (bank width / median measured wet width) to down-weight/skip USGS anchor.")
    p.add_argument("--no-river-usgs-width-ratio-blend", dest="river_usgs_width_ratio_blend", action="store_false", help="Disable blending; skip USGS anchor when width ratio is large.")
    p.set_defaults(river_usgs_width_ratio_blend=True)
    p.add_argument("--river-gage-snap-max-dist-m", type=float, default=1000.0, help="Max distance (m) to snap gages to the river network.")

    # Optional: width-stage inversion anchors
    p.add_argument("--river-width-stage-csv", default=None,
                   help="Comma-separated width-stage CSV paths (station_id,date,width_m,stage_m optional).")
    p.add_argument("--river-width-stage-max-dist-m", type=float, default=5000.0)
    p.add_argument("--river-width-stage-min-n", type=int, default=6)
    p.add_argument("--river-width-stage-min-r2", type=float, default=0.25, help="Minimum R^2 for width–stage fit.")
    p.add_argument("--river-width-stage-max-weight", type=float, default=0.8, help="Maximum blend weight for width–stage anchor.")

    # Optional: Manning inversion prior (secondary, blended; requires Q + W + S)
    p.add_argument("--river-manning-enabled", action="store_true", help="Alias: enable Manning prior (sets --river-manning-mode=q2_regional if mode is off).")
    p.add_argument("--river-manning-mode", choices=["off","constant","from_field","q2_regional"], default="off",
                   help=("Add a Manning-based depth prior and blend it into the river Dmax estimate. "
                         "Mode off=disabled; constant=use a single Q for all reaches; from_field=read Q from a reach attribute field in river gpkg."))
    p.add_argument("--river-manning-q-cms", type=float, default=None,
                   help="Discharge Q (m^3/s) to use when --river-manning-mode=constant.")
    p.add_argument("--river-manning-q-field", default=None,
                   help="Reach attribute field name containing discharge Q (m^3/s) when --river-manning-mode=from_field.")
    p.add_argument("--river-manning-n", type=float, default=0.035, help="Manning n roughness (typical 0.03-0.08).")
    p.add_argument("--river-manning-region", default="auto",
                   help="Region key for --river-manning-mode=q2_regional (uses drainage area). Prefer state/region published regressions.")
    p.add_argument("--river-manning-region-auto-map", default=None,
                   help="Optional JSON mapping US state abbreviations -> manning region keys (used when --river-manning-region=auto).")
    p.add_argument("--river-manning-min-confidence", type=float, default=0.30,
                   help="Minimum confidence required to apply Manning prior (0-1).")
    p.add_argument("--river-manning-max-weight", type=float, default=0.60, help="Maximum blend weight for Manning prior (0-1).")
    p.add_argument("--river-manning-backwater-slope-thresh", type=float, default=1e-4,
                   help="Disable/zero Manning prior when slope is below this (backwater/tidal risk).")
    p.add_argument("--river-manning-dist-to-mouth-field", default=None,
                   help="Optional reach attribute field containing distance-to-mouth (km). If provided, can disable Manning prior near mouth.")
    p.add_argument("--river-manning-dist-to-mouth-km-max", type=float, default=10.0,
                   help="If distance-to-mouth field is provided, disable Manning prior when distance <= this (km).")

    # Optional: Regional hydraulic geometry curves (DA -> bankfull depth) soft prior
    p.add_argument("--river-regional-curve-enabled", action="store_true",
                   help="Enable regional curve prior (DA -> bankfull depth) blended into the Dmax prior.")
    p.add_argument("--river-regional-curve-region", default="default",
                   help="Region key for built-in (illustrative) coefficients. Prefer providing --river-regional-curve-c and --river-regional-curve-f from published regional curves.")
    p.add_argument("--river-regional-curve-c", type=float, default=None,
                   help="Coefficient c in D_bkf = c * DA^f. Units depend on --river-regional-curve-da-units and --river-regional-curve-depth-units.")
    p.add_argument("--river-regional-curve-f", type=float, default=None,
                   help="Exponent f in D_bkf = c * DA^f.")
    p.add_argument("--river-regional-curve-da-units", choices=["km2","mi2"], default="km2",
                   help="Drainage area units expected by the curve coefficients.")
    p.add_argument("--river-regional-curve-depth-units", choices=["m","ft"], default="m",
                   help="Depth units of the curve coefficients.")
    p.add_argument("--river-regional-curve-depth-type", choices=["mean","max"], default="mean",
                   help="Whether curve depth represents mean or max depth at bankfull.")
    p.add_argument("--river-regional-curve-to-dmax", choices=["auto","factor"], default="auto",
                   help="Conversion from bankfull depth to Dmax. 'auto' uses trapezoid mean->Dmax conversion; 'factor' uses --river-regional-curve-to-dmax-factor.")
    p.add_argument("--river-regional-curve-to-dmax-factor", type=float, default=1.25,
                   help="Factor to convert bankfull mean depth to Dmax when --river-regional-curve-to-dmax=factor.")
    p.add_argument("--river-regional-curve-unc-pct", type=float, default=40.0,
                   help="Uncertainty of the regional curve (percent, used to down-weight the prior).")
    p.add_argument("--river-regional-curve-max-weight", type=float, default=0.60,
                   help="Maximum blend weight for regional curve prior (0-1).")
    p.add_argument("--river-regional-curve-min-da-km2", type=float, default=1.0,
                   help="Minimum drainage area (km^2) to apply the regional curve prior.")

    
    # Fusion controls
    p.add_argument("--fusion-strategy", default="seam_blend",
                   choices=["weighted_overlap", "spatial_taper", "priority", "blend", "seam_blend"],
                   help="How to fuse SDB + river surfaces when both exist.")
    p.add_argument("--fusion-primary-weight", type=float, default=0.70,
                   help="Primary weight in overlap pixels (only used by weighted_overlap/spatial_taper).")
    p.add_argument("--fusion-secondary-weight", type=float, default=0.30,
                   help="Secondary weight in overlap pixels (only used by weighted_overlap/spatial_taper).")
    p.add_argument("--fusion-taper-m", type=float, default=75.0,
                   help="Taper distance (meters) for spatial_taper inside the river corridor.")

    # Output masking
    p.add_argument("--no-mask-river-to-waffles", action="store_true", default=False,
                   help="Disable masking river depth+bed outputs to the waffles coastline mask (if available).")

    # Intelligent gap filling (Tier 1–2)
    p.add_argument("--gapfill-enabled", action="store_true", default=False,
                   help="Apply intelligent gap-fill correction to the fused bathymetry raster using residual interpolation against high-quality points.")
    p.add_argument("--gapfill-hq", nargs='*', default=None,
                   help="One or more high-quality point files (CSV/TXT/GPKG/Shp) with x/y/z (or lon/lat/depth) columns. If omitted, uses --extra-xyz (if provided).")
    p.add_argument("--gapfill-water-mask", default=None,
                   help="Optional explicit water mask raster (1=water) to constrain gapfill. If omitted, uses waffles mask when available, otherwise finite prior pixels.")
    p.add_argument("--gapfill-method", default="rbf", choices=["rbf", "gp", "idw"],
                   help="Interpolation method for residuals: rbf (thin-plate spline), gp (Gaussian Process), idw (inverse distance).")
    p.add_argument("--gapfill-river-smooth-sigma", type=float, default=500.0,
                   help="Along-channel smoothing sigma (meters) for river corridors.")
    p.add_argument("--gapfill-prior-sigma", default=None,
                   help="Optional prior uncertainty raster for better uncertainty propagation.")
    p.add_argument("--gapfill-bank-elev", default=None,
                   help="Optional bank elevation raster for enforcing bed < bank constraint.")
    p.add_argument("--gapfill-cudem-xyz", action="store_true", default=False,
                   help="Also output CUDEM-compatible XYZ file with uncertainty column.")

    # Run health gates
    p.add_argument("--strict", action="store_true", default=False,
                   help="Fail the run if contract tests or output sanity checks fail.")

    return p.parse_args()


def _apply_output_retention_policy(cfg, log, report, final_path=None, final_for_user_path=None):
    """Enforce output retention policy.

    Default (cfg.save_intermediates=False):
      - Keep only final deliverables + run metadata/logs in cfg.out_dir.
      - Delete everything else under cfg.out_dir.

    If cfg.save_intermediates=True:
      - Move everything else under cfg.out_dir/<cfg.intermediates_dirname>/... and keep deliverables in place.

    This function operates ONLY on cfg.out_dir (never touches cfg.cache_root).
    """
    import shutil
    from pathlib import Path
    import fnmatch
    import uuid

    out_dir = Path(cfg.out_dir)
    if not out_dir.exists():
        return

    # --- Build explicit keep list from run reports (NO filename guessing) ---
    keep_abs = set()  # absolute paths

    def _collect_output_paths(report_obj):
        out_paths = set()
        def _walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "outputs" and isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, str) and _is_probably_path(vv):
                                out_paths.add(vv)
                    _walk(v)
            elif isinstance(o, list):
                for it in o:
                    _walk(it)
        _walk(report_obj)
        return out_paths

    # Prefer unified report if present, else fall back to bathy_report
    report_paths = [out_dir / "unified_bathy_report.json", out_dir / "bathy_report.json"]
    for rp in report_paths:
        if rp.exists() and rp.is_file():
            try:
                robj = json.loads(rp.read_text(encoding="utf-8"))
                for s in _collect_output_paths(robj):
                    try:
                        pp = Path(s)
                        keep_abs.add(pp if pp.is_absolute() else (out_dir / pp).resolve())
                    except Exception:
                        continue
            except Exception:
                continue

    # Also keep io_manifest outputs if present (explicit list)
    io_json = out_dir / "io_manifest.json"
    if io_json.exists() and io_json.is_file():
        try:
            io = json.loads(io_json.read_text(encoding="utf-8"))
            for s in (io.get("outputs") or []):
                if isinstance(s, str) and _is_probably_path(s):
                    pp = Path(s)
                    keep_abs.add(pp if pp.is_absolute() else (out_dir / pp).resolve())
        except Exception:
            pass

    # Always keep run logs + summaries/reports
    keep_dirs = {"run_logs"}
    keep_globs = [
        "run_summary*.json",
        "run_summary*.md",
        "unified_bathy_report*.json",
        "unified_bathy_report*.md",
        "bathy_report*.json",
        "bathy_report*.md",
        "metrics_summary*.csv",
    ]

    # Only enforce policy if we produced a final deliverable
    final_ok = False

    # Prefer explicit outputs recorded in reports/manifest
    if keep_abs:
        for pth in list(keep_abs):
            try:
                if Path(pth).exists():
                    final_ok = True
                    break
            except Exception:
                continue

    # Fall back to final paths passed in
    if (not final_ok) and final_for_user_path:
        final_ok = Path(final_for_user_path).exists()
    if (not final_ok) and final_path:
        final_ok = Path(final_path).exists()

    if not final_ok:
        log.info("[OUTPUT] Retention policy skipped (no final deliverable found).")
        return

    # Stage kept files into a temp folder to allow deterministic cleanup of everything else.
    tmp = out_dir / f"keep_tmp_{uuid.uuid4().hex[:10]}"
    tmp.mkdir(parents=True, exist_ok=True)

    try:

        def _copy_rel(relpath: str):
            src = out_dir / relpath
            if src.exists() and src.is_file():
                dst = tmp / relpath
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))

        # Copy explicitly recorded output files (relative within out_dir when possible)
        for ap in sorted(keep_abs, key=lambda x: str(x)):
            try:
                ap = Path(ap)
                if (not ap.exists()) or (not ap.is_file()):
                    continue
                try:
                    rel = ap.resolve().relative_to(out_dir.resolve())
                except Exception:
                    # If output is outside out_dir, copy into tmp/_external/ for inspection
                    rel = Path("_external") / ap.name
                dst = tmp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(ap), str(dst))
            except Exception as e:
                log.warning('[OUTPUT] Failed to stage output %s (%s). Continuing.', str(ap), e)

        # Copy report-ish files in out_dir root
        for p in out_dir.iterdir():
            if p.is_file():
                for g in keep_globs:
                    if fnmatch.fnmatch(p.name, g):
                        dst = tmp / p.name
                        shutil.copy2(str(p), str(dst))
                        break

        # Copy run_logs directory (as-is)
        run_logs = out_dir / "run_logs"
        if run_logs.exists() and run_logs.is_dir():
            dst = tmp / "run_logs"
            shutil.copytree(str(run_logs), str(dst), dirs_exist_ok=True)

        # If saving intermediates, move everything except tmp and intermediates dir under intermediates.
        intermediates_dir = out_dir / str(getattr(cfg, "intermediates_dirname", "debug") or "debug")

        if getattr(cfg, "save_intermediates", False):
            intermediates_dir.mkdir(parents=True, exist_ok=True)
            for child in list(out_dir.iterdir()):
                if child.name == tmp.name:
                    continue
                if child.resolve() == intermediates_dir.resolve():
                    continue
                # Move everything into intermediates dir
                dest = intermediates_dir / child.name
                if dest.exists():
                    if dest.is_dir():
                        try:
                            shutil.rmtree(str(dest))
                        except Exception as e:
                            log.warning('[OUTPUT] Failed to remove existing dest dir %s (%s). Continuing.', str(dest), e)
                    else:
                        try:
                            dest.unlink()
                        except Exception as e:
                            log.warning('[OUTPUT] Failed to remove existing dest file %s (%s). Continuing.', str(dest), e)
                try:
                    shutil.move(str(child), str(dest))
                except Exception as e:
                    log.warning('[OUTPUT] Failed to move %s -> %s (%s). Continuing.', str(child), str(dest), e)
            log.info("[OUTPUT] Saved intermediates under: %s", str(intermediates_dir))
        else:
            # Delete everything except tmp
            for child in list(out_dir.iterdir()):
                if child.name == tmp.name:
                    continue
                if child.is_dir():
                    try:
                        shutil.rmtree(str(child))
                    except Exception as e:
                        log.warning('[OUTPUT] Failed to delete dir %s (%s). Continuing.', str(child), e)
                else:
                    try:
                        child.unlink(missing_ok=True)
                    except Exception as e:
                        log.warning('[OUTPUT] Failed to delete file %s (%s). Continuing.', str(child), e)
            log.info("[OUTPUT] Deleted intermediates (kept deliverables + run metadata only).")

        # Restore kept content from tmp
        for src in tmp.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(tmp)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))

        # Remove tmp
        shutil.rmtree(str(tmp), ignore_errors=True)
    finally:
        # Always attempt to remove temp staging folder (crash-safe retention)
        if tmp.exists():
            shutil.rmtree(str(tmp), ignore_errors=True)


def main() -> int:
    args = parse_args()
    # ---------------------------------------------------------------------
    # Unified data acquisition buffer (single AOI buffer knob):
    # - User supplies --buff as a fractional expansion of the provided AOI bbox.
    # - We download / query ALL datasets using the buffered AOI.
    # - We keep the original AOI as the tile extent for final clipping/metrics.
    # ---------------------------------------------------------------------
    tile_bbox = _parse_aoi_bbox(getattr(args, "aoi", None))
    if not tile_bbox:
        raise SystemExit("Invalid --aoi. Expected bbox 'W/E/S/N' (lon_min/lon_max/lat_min/lat_max).")

    args.aoi_tile = getattr(args, "aoi", None)  # user-requested tile AOI
    args.tile_bbox = tile_bbox

    # Legacy: tile-buffer-km is deprecated; enforce single buffer knob.
    try:
        legacy_buf_km = float(getattr(args, "tile_buffer_km", 0.0) or 0.0)
    except Exception:
        legacy_buf_km = 0.0
    if legacy_buf_km not in (0.0, -0.0):
        log.warning("[AOI] --tile-buffer-km is deprecated and ignored. Use --buff instead. (legacy=%s)", legacy_buf_km)

    try:
        buff_frac = float(getattr(args, "buff", 0.0) or 0.0)
    except Exception:
        buff_frac = 0.0
    if buff_frac < 0:
        raise SystemExit("--buff must be >= 0")

    if buff_frac > 0:
        expanded = _expand_bbox_frac(tile_bbox, buff_frac)
        args.aoi = _bbox_to_aoi_str(expanded)
        log.info("[AOI] Data buffer enabled: --buff=%s -> aoi_data=%s (aoi_tile=%s)", buff_frac, args.aoi, args.aoi_tile)
    else:
        log.info("[AOI] No data buffer: --buff=0 -> aoi_data=aoi_tile=%s", args.aoi_tile)

    # ---------------------------------------------------------------------
    # Hydraulic modeling context buffer (river seam stability)
    # ---------------------------------------------------------------------
    # Rule:
    #   - River modeling MUST run on the buffered AOI (aoi_data == args.aoi)
    #   - Final deliverables MUST be clipped back to the original tile AOI (args.aoi_tile)
    # This stabilizes longitudinal fits and boundary conditions across adjacent tiles.
    args.aoi_hydro = getattr(args, "aoi", None)

    # ---------------------------------------------------------------------
    # Run-scoped logging + flight recorder
    # ---------------------------------------------------------------------
    run_id = None
    try:
        from datetime import datetime, timezone
        from logging_config import add_file_handler, start_flight_recorder

        # Deterministic-enough run id for log grouping
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"bathy_{ts}_{os.getpid()}"

        out_dir = Path(getattr(args, 'out_dir', 'output'))
        (out_dir / "run_logs").mkdir(parents=True, exist_ok=True)

        # Per-run text log
        add_file_handler(out_dir / "run_logs" / f"run_{run_id}.log", level=logging.INFO)

        # Structured flight recorder JSONL
        fr_path = start_flight_recorder(out_dir, run_id=run_id)
        if fr_path is not None:
            log.info(f"[RUN] Flight recorder: {fr_path}")
        log.info(f"[RUN] run_id={run_id}")
    except Exception as e:
        log.debug(f"[RUN] Unable to initialize run logs/flight recorder: {e}")
    # Auto-select published regression/curve coefficients from sdb_config.json based on AOI centroid.
    # This sets:
    #   - --river-manning-region (for Q2 DA->Q2 regressions used by Manning inversion)
    #   - --river-regional-curve-c/f (for DA->bankfull depth curves)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(here, 'sdb_config.json')
        cfg_json = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_json = json.load(f)
        # Parse AOI as bbox: w/e/s/n
        lon_c = lat_c = None
        try:
            if isinstance(getattr(args, 'aoi', None), str) and '/' in args.aoi:
                w,e,s,n = [float(x) for x in args.aoi.strip().strip('"').split('/')[:4]]
                lon_c = (w + e) / 2.0
                lat_c = (s + n) / 2.0
        except Exception:
            lon_c = lat_c = None

        if lon_c is not None and lat_c is not None and cfg_json:
            from region_resolver import resolve_q2_regression, resolve_bankfull_curve

            # Manning Q2 regression region key
            if str(getattr(args, 'river_manning_region', 'default')).strip().lower() == 'auto':
                q2_curve, msg = resolve_q2_regression(cfg_json, lon_c, lat_c)
                if q2_curve is not None and q2_curve.name:
                    args.river_manning_region = q2_curve.name
                    log.info(f"[RIVER][MANNING] Auto region: {msg}")
                else:
                    # fallback to older coarse mapping
                    from manning_inversion import infer_manning_region_from_aoi
                    region_key, st = infer_manning_region_from_aoi(getattr(args, 'aoi', None), state_region_map_json=getattr(args, 'river_manning_region_auto_map', None), default_region='default')
                    args.river_manning_region = region_key
                    log.warning(f"[RIVER][MANNING] Published regression not found; using coarse region={region_key} (state={st})")

            # Bankfull depth regional curve coefficients
            if getattr(args, 'river_regional_curve_enabled', False) and (getattr(args, 'river_regional_curve_c', None) is None or getattr(args, 'river_regional_curve_f', None) is None):
                bcurve, msg = resolve_bankfull_curve(cfg_json, lon_c, lat_c)
                if bcurve is not None:
                    args.river_regional_curve_region = bcurve.name
                    args.river_regional_curve_c = float(bcurve.c)
                    args.river_regional_curve_f = float(bcurve.f)
                    # units in registry are recorded; keep user-selected units args as-is, but warn if mismatch
                    log.info(f"[RIVER][REGIONAL] Auto coefficients: {msg}")
                else:
                    log.warning(f"[RIVER][REGIONAL] Auto coefficients unavailable: {msg}")
    except Exception as e:
        log.debug(f"[RIVER] Auto coefficient resolution failed: {e}")


    # Optional: auto-fetch SWOT RiverSP if user didn't provide a local path.
    try:
        if (str(getattr(args, "river_skeleton_wse_mode", "bank")).strip().lower() == "bank_profile"
                and getattr(args, "river_swot_riversp", None) is None
                and bool(getattr(args, "river_swot_auto", True))):
            # Parse AOI bbox: w/e/s/n
            w = e = s = n = None
            try:
                if isinstance(getattr(args, "aoi", None), str) and "/" in args.aoi:
                    w, e, s, n = [float(x) for x in args.aoi.strip().strip('"').split("/")[:4]]
            except Exception:
                w = e = s = n = None
    
            if None not in (w, e, s, n):
                swot_cache_root = getattr(args, "river_swot_cache_root", None)
                if not swot_cache_root:
                    # keep RiverSP under cache_root/swot by default
                    swot_cache_root = os.path.join(str(getattr(args, "cache_root", "cache")), "swot")
                from swot_riversp_fetch import fetch_riversp
                res = fetch_riversp(
                    bbox_wesn=(float(w), float(e), float(s), float(n)),
                    start_date=str(getattr(args, "start_date", "")),
                    end_date=str(getattr(args, "end_date", "")),
                    cache_root=str(swot_cache_root),
                    product=str(getattr(args, "river_swot_product", "reach") or "reach"),
                    short_name=getattr(args, "river_swot_shortname", None),
                    logger=log,
                )
                if res.files:
                    args.river_swot_riversp = res.files
                    log.info(f"[SWOT] Auto-fetched RiverSP ({len(res.files)} file(s)) -> {res.cache_dir}")
                else:
                    log.info(f"[SWOT] RiverSP auto-fetch not used: {res.message}")
            else:
                log.debug("[SWOT] AOI bbox parse failed; skipping RiverSP auto-fetch")
    except Exception as _e:
        log.debug(f"[SWOT] RiverSP auto-fetch failed: {_e}")
    # Backwards-compat alias: --river-manning-enabled
    if getattr(args, "river_manning_enabled", False) and str(getattr(args, "river_manning_mode", "off")) == "off":
        args.river_manning_mode = "q2_regional"

    
    # ------------------------------------------------------------------
    # CUDEM / waffles cache
    # ------------------------------------------------------------------
    # Many CUDEM tools (e.g., waffles) use CUDEM_CACHE independent of this
    # pipeline's --cache-root. To keep runs reproducible and avoid failures
    # when the default cache directory doesn't exist, we scope CUDEM_CACHE
    # under --cache-root unless the user explicitly sets CUDEM_CACHE.
    try:
        _cc = os.environ.get("CUDEM_CACHE", "").strip()
        if not _cc:
            cudem_cache_dir = Path(args.cache_root) / "cudem_cache"
            ensure_dir(cudem_cache_dir)
            # Some CUDEM tools expect provider subdirs (e.g., "tnm") to exist.
            # Pre-create common ones to avoid spurious ENOENT warnings.
            try:
                ensure_dir(cudem_cache_dir / "tnm")
            except Exception:
                pass
            os.environ["CUDEM_CACHE"] = str(cudem_cache_dir)
            log.info("[CUDEM_CACHE] Using pipeline-scoped CUDEM cache: %s", cudem_cache_dir)
        else:
            ensure_dir(Path(_cc))
            try:
                ensure_dir(Path(_cc) / "tnm")
            except Exception:
                pass
            log.info("[CUDEM_CACHE] Using existing CUDEM_CACHE: %s", _cc)
    except Exception as e:
        log.warning("[CUDEM_CACHE] Unable to prepare CUDEM cache directory: %s", e)
    cfg = BathyConfig(
            aoi=args.aoi,
            start_date=args.start_date,
            end_date=args.end_date,
            aoi_tile=getattr(args, "aoi_tile", None),
            tile_bbox=getattr(args, "tile_bbox", None),
            tile_buffer_km=float(getattr(args, "tile_buffer_km", 0.0) or 0.0),
            tile_edge_taper_enabled=bool(getattr(args, "tile_edge_taper_enabled", True)),
            tile_edge_taper_km=float(getattr(args, "tile_edge_taper_km", 0.0) or 0.0),
            tile_edge_smooth_sigma_km=float(getattr(args, "tile_edge_smooth_sigma_km", 0.0) or 0.0),
            tile_edge_metrics_band_km=float(getattr(args, "tile_edge_metrics_band_km", 0.0) or 0.0),
            sdb_model_bank_enabled=bool(getattr(args, "sdb_model_bank_enabled", True)),
            sdb_model_bank=str(getattr(args, "sdb_model_bank", "auto")),
            sdb_bank_max_samples=int(getattr(args, "sdb_bank_max_samples", 100000)),
            sdb_bank_seed=int(getattr(args, "sdb_bank_seed", 1337)),
            sdb_bank_retrain_min_new=int(getattr(args, "sdb_bank_retrain_min_new", 2000)),
            sdb_model_cache_enabled=bool(getattr(args, "sdb_model_cache_enabled", False)),
            sdb_model_cache_key=str(getattr(args, "sdb_model_cache_key", "auto")),
            out_dir=Path(args.out_dir).resolve(),
            save_intermediates=bool(getattr(args, "save_intermediates", False)),
            intermediates_dirname=str(getattr(args, "intermediates_dirname", "debug")),
            methods=[m.strip().lower() for m in args.methods.split(",") if m.strip()],
            priority=args.priority,
    
            cloud=args.cloud,
            icesat=args.icesat,
            sdb_mode=args.sdb_mode,
            cache_root=Path(args.cache_root).resolve(),
            align_mode=args.align_mode,
            glint_correct=args.glint_correct,
            glint_nir_band=args.glint_nir_band,
            glint_vis_bands=args.glint_vis_bands,
            glint_nir_min_percentile=args.glint_nir_min_percentile,
            glint_deepwater_b02_max=args.glint_deepwater_b02_max,
            glint_min_samples=args.glint_min_samples,
            glint_max_samples=args.glint_max_samples,
            glint_clip_min=args.glint_clip_min,
            working_srs=args.working_srs,
            working_vcrs_epsg=args.working_vcrs_epsg,
            final_out_srs=args.final_out_srs,
            require_river_constraints=str(getattr(args, "require_river_constraints", "none")),
            river_dem_auto=args.river_dem_auto,
            river_dem_source=args.river_dem_source,
            river_dem_res_m=args.river_dem_res_m,
            extra_xyz_crs=args.extra_xyz_crs,
            extra_xyz=list(getattr(args, "extra_xyz", []) or []),
            extra_xyz_cudem=list(getattr(args, "extra_xyz_cudem", []) or []),
    
            river_dem=Path(args.river_dem).resolve() if args.river_dem else None,
            river_soundings=None,
            river_prior_mode=args.river_prior_mode,
            river_mv_a0=args.river_mv_a0,
            river_mv_bw=args.river_mv_bw,
            river_mv_ba=args.river_mv_ba,
            river_mv_bs=args.river_mv_bs,
            river_mv_eps_a=args.river_mv_eps_a,
            river_mv_eps_s=args.river_mv_eps_s,
    
            river_slope_proxy_window=args.river_slope_proxy_window,
            river_slope_min=args.river_slope_min,
            river_slope_max=args.river_slope_max,
            river_slope_proxy_min_n=args.river_slope_proxy_min_n,
    
            river_wse_profile_enabled=bool(getattr(args, "river_wse_profile_enabled", True)),
            river_wse_profile_window=int(getattr(args, "river_wse_profile_window", args.river_slope_proxy_window)),
            river_wse_profile_min_n=int(getattr(args, "river_wse_profile_min_n", args.river_slope_proxy_min_n)),
            river_wse_profile_monotonic=bool(getattr(args, "river_wse_profile_monotonic", True)),
    
            river_usgs_sites=args.river_usgs_sites,
            river_usgs_start=args.river_usgs_start,
            river_usgs_end=args.river_usgs_end,
            river_usgs_cache_dir=Path(args.river_usgs_cache_dir) if args.river_usgs_cache_dir else None,
            river_usgs_max_dist_m=args.river_usgs_max_dist_m,
            river_usgs_mean_to_dmax=args.river_usgs_mean_to_dmax,
            river_usgs_a_stat=args.river_usgs_a_stat,
            river_usgs_q_quantile_lo=args.river_usgs_q_quantile_lo,
            river_usgs_q_quantile_hi=args.river_usgs_q_quantile_hi,
            river_usgs_a_cv_warn=args.river_usgs_a_cv_warn,
            river_usgs_width_ratio_max=args.river_usgs_width_ratio_max,
            river_usgs_width_ratio_blend=args.river_usgs_width_ratio_blend,
            river_gage_snap_max_dist_m=args.river_gage_snap_max_dist_m,
    
            river_width_stage_csv=args.river_width_stage_csv,
            river_width_stage_max_dist_m=args.river_width_stage_max_dist_m,
            river_width_stage_min_n=args.river_width_stage_min_n,
            river_width_stage_min_r2=args.river_width_stage_min_r2,
            river_width_stage_max_weight=args.river_width_stage_max_weight,
    
            river_hydrography_source=args.river_hydrography_source,
            tnm_enable=((args.river_hydrography_source in ('arcgis_tnm','tnm')) and (not args.no_tnm)),
            tnm_dataset=args.tnm_dataset,
            snap_m=args.snap_m,
    
    
            river_method=args.river_method,
            river_channel_buffer_m=args.river_channel_buffer_m,
            river_max_channel_width_m=args.river_max_channel_width_m,
            river_mainstem_min_order=args.river_mainstem_min_order,
            river_max_mainstem_width_m=args.river_max_mainstem_width_m,
            river_use_nhdarea=bool(getattr(args, "river_use_nhdarea", True)),
            river_nhdarea_layer=getattr(args, "river_nhdarea_layer", "nhdarea_clip"),
            river_shape_exp=args.river_shape_exp,
            river_dmax_min_m=args.river_dmax_min_m,
            river_dmax_max_m=args.river_dmax_max_m,
            river_bed_profile_max_slope=float(getattr(args,'river_bed_profile_max_slope',0.0) or 0.0),
            river_bed_profile_max_curv=float(getattr(args,'river_bed_profile_max_curv',0.0) or 0.0),
            river_bed_profile_step_m=float(getattr(args,'river_bed_profile_step_m',25.0) or 25.0),
            river_bed_profile_strength=float(getattr(args,'river_bed_profile_strength',0.6) or 0.6),
            river_bed_profile_power=float(getattr(args,'river_bed_profile_power',2.0) or 2.0),
            river_save_skeleton_debug=bool(args.river_save_skeleton_debug),
            river_skeleton_wse_mode=getattr(args, "river_skeleton_wse_mode", "bank"),
            river_skeleton_wse_smooth_sigma_m=float(getattr(args, "river_skeleton_wse_smooth_sigma_m", 0.0)),
            river_skeleton_junction_mode=str(getattr(args, "river_skeleton_junction_mode", "smooth")),
            river_skeleton_junction_buffer_m=float(getattr(args, "river_skeleton_junction_buffer_m", 120.0)),
            river_skeleton_junction_degree_min=int(getattr(args, "river_skeleton_junction_degree_min", 3)),
            river_skeleton_junction_smooth_sigma_m=float(getattr(args, "river_skeleton_junction_smooth_sigma_m", 80.0)),
            river_soundings_mode=args.river_soundings_mode,
            river_soundings_max_dist_m=args.river_soundings_max_dist_m,
            river_soundings_min_r=args.river_soundings_min_r,
            river_soundings_enforce=bool(getattr(args, 'river_soundings_enforce', True)),
    
            xs_spacing_m=args.xs_spacing_m,
            xs_length_m=args.xs_length_m,
            xs_smoothing_window_m=args.xs_smoothing_window_m,
            xs_trim_overlaps=bool(getattr(args, 'xs_trim_overlaps', True)),
            xs_global_deconflict=bool(getattr(args, 'xs_global_deconflict', True)),
            xs_deconflict_tol_m=args.xs_deconflict_tol_m,
            xs_skip_junctions=bool(getattr(args, 'xs_skip_junctions', True)),
            xs_junction_snap_m=args.xs_junction_snap_m,
            xs_junction_buffer_m=args.xs_junction_buffer_m,
            xs_densify_step_m=args.xs_densify_step_m,
            river_continuous=args.river_continuous,
            river_continuous_buffer_m=args.river_continuous_buffer_m,
            river_continuous_k=args.river_continuous_k,
            river_idw_power=args.river_idw_power,
            river_aniso_along_scale_m=args.river_aniso_along_scale_m,
            river_aniso_cross_scale_m=args.river_aniso_cross_scale_m,
            river_thalweg_weight=args.river_thalweg_weight,
            river_thalweg_only=bool(getattr(args,'river_thalweg_only', False)),
            river_thalweg_densify_factor=float(getattr(args,'river_thalweg_densify_factor', 0.5)),
            river_thalweg_densify_step_m=(None if getattr(args,'river_thalweg_densify_step_m', None) is None else float(getattr(args,'river_thalweg_densify_step_m'))),
            river_overlap_reducer=args.river_overlap_reducer,
            river_nodata=args.river_nodata,
            fusion_strategy=args.fusion_strategy,
            fusion_primary_weight=args.fusion_primary_weight,
            fusion_secondary_weight=args.fusion_secondary_weight,
            fusion_taper_m=args.fusion_taper_m,
            mask_river_to_waffles=(not args.no_mask_river_to_waffles),
    
            gapfill_enabled=bool(getattr(args, "gapfill_enabled", False)),
            gapfill_hq=list(getattr(args, "gapfill_hq", None)) if getattr(args, "gapfill_hq", None) else None,
            gapfill_water_mask=Path(getattr(args, "gapfill_water_mask", "")) if getattr(args, "gapfill_water_mask", None) else None,
            gapfill_method=getattr(args, "gapfill_method", "rbf"),
            gapfill_river_smooth_sigma_m=float(getattr(args, "gapfill_river_smooth_sigma", 500.0)),
            gapfill_prior_sigma_raster=Path(getattr(args, "gapfill_prior_sigma", "")) if getattr(args, "gapfill_prior_sigma", None) else None,
            gapfill_bank_elev_raster=Path(getattr(args, "gapfill_bank_elev", "")) if getattr(args, "gapfill_bank_elev", None) else None,
            gapfill_output_cudem_xyz=bool(getattr(args, "gapfill_cudem_xyz", False)),
            strict=args.strict,
    
        )
    # Resolve working CRS (used for river DEM and for meter-based thinning).
    cfg.working_srs = detect_working_srs(cfg)

    ensure_dir(cfg.out_dir)

    # Run-scoped derived-cache root: we can safely keep downloaded/raw inputs cached,
    # but derived products must be regenerated per run to avoid stale/invalid outputs.
    cfg.run_id = run_id
    cfg.derived_cache_root = cfg.out_dir / "derived_cache" / run_id
    ensure_dir(cfg.derived_cache_root)

    log.info("=" * 70)
    log.info("UNIFIED BATHYMETRY PIPELINE")
    log.info("=" * 70)
    log.info(f"AOI: {cfg.aoi}")
    log.info(f"Date range: {cfg.start_date} to {cfg.end_date}")
    log.info(f"Methods: {cfg.methods}")
    log.info(f"Priority: {cfg.priority}")
    log.info(f"Output: {cfg.out_dir}")
    log.info("=" * 70)

    # ---------------------------------------------------------------------
    # WAFFLES-driven domain inference: decide which methods are applicable
    # ---------------------------------------------------------------------
    domain_inference: Dict[str, Any] = {}
    try:
        cfg.methods_requested = list(getattr(cfg, "methods", []) or [])
        effective_methods, _meta = _determine_effective_methods_from_waffles(cfg, domain_inference)
        cfg.methods_effective = list(effective_methods)
        cfg.methods = list(effective_methods)

        # One-line operational log for scanability
        skipped = (_meta or {}).get("skipped", {}) if isinstance(_meta, dict) else {}
        waffles_meta = (_meta or {}).get("waffles", {}) if isinstance(_meta, dict) else {}
        if skipped:
            # Include key numeric diagnostics in the one-line log so "no water" decisions are auditable.
            log.info("[DOMAIN] WAFFLES inference: requested=%s -> effective=%s (skipped=%s; ocean_frac=%.6f; nhd_frac=%.6f; min_frac=%.6f)",
                     ",".join(cfg.methods_requested), ",".join(cfg.methods_effective),
                     ",".join([f"{k}:{v}" for k, v in skipped.items()]),
                     float(waffles_meta.get("ocean_water_fraction", 0.0) or 0.0),
                     float(waffles_meta.get("with_nhd_water_fraction", 0.0) or 0.0),
                     float(waffles_meta.get("min_water_fraction", 0.0) or 0.0))
        else:
            log.info("[DOMAIN] WAFFLES inference: requested=%s -> effective=%s",
                     ",".join(cfg.methods_requested), ",".join(cfg.methods_effective))
    except Exception as e:
        # HARD-FAIL POLICY: WAFFLES domain inference is required for deterministic method gating
        # and for enforcing final domain clipping. If it fails, abort with non-zero exit so the
        # user is forced to fix masks/data rather than producing misleading empty rasters.
        log.error("[DOMAIN][FATAL] WAFFLES domain inference failed; aborting run (exit=2).", exc_info=True)
        log.error("[DOMAIN][FATAL] Fix WAFFLES/mask generation issues, then re-run. Common causes: missing WAFFLES deps/data, GDAL/PROJ errors, or stale/corrupt mask cache under derived_cache_root.")
        raise SystemExit(2)

    # ---------------------------------------------------------------------
    # External soundings (extra XYZ): only fetch/normalize if any water method
    # will actually run (WAFFLES already decided effective methods above).
    # ---------------------------------------------------------------------
    need_xyz = ("sdb" in cfg.methods) or ("river" in cfg.methods)

    if not need_xyz:
        if getattr(args, "extra_xyz_cudem", None) or getattr(args, "extra_xyz", None):
            log.info("[XYZ] Skipping extra XYZ fetch/normalization: no effective water methods (methods=%s).",
                     ",".join(cfg.methods_effective or cfg.methods_requested or cfg.methods))
    else:
        # Optional: auto-download external soundings via CUDEM `dlim` providers.
        # (Only after WAFFLES inference, so inland/no-water AOIs avoid unnecessary downloads.)
        if getattr(args, "extra_xyz_cudem", None):
            requested = []
            for item in args.extra_xyz_cudem:
                if item is None:
                    continue
                requested.extend([p.strip() for p in str(item).split(",") if p.strip()])

            if requested:
                # Determine output CRS for dlim.
                # For *depth* soundings we prefer a projected working CRS so thinning happens in meters.
                # Keep this horizontal-only; Z is preserved as depth.
                user_crs = str(getattr(args, "extra_xyz_cudem_crs", "")).strip().lower()
                if user_crs and user_crs not in ("", "auto"):
                    dlim_crs = user_crs
                else:
                    try:
                        dlim_crs = str(cfg.working_srs)
                    except Exception:
                        dlim_crs = "epsg:4269"

                auto_xyz, auto_rep = fetch_cudem_soundings_via_dlim(
                    aoi=cfg.aoi,
                    sources=requested,
                    cache_root=Path(cfg.cache_root),
                    out_crs=dlim_crs,
                    thin_res_m=float(getattr(args, "extra_xyz_cudem_thin_res_m", 10.0))
                        if float(getattr(args, "extra_xyz_cudem_thin_res_m", 10.0)) > 0 else None,
                    filter_spec=getattr(args, "extra_xyz_cudem_filter", None),
                    force=bool(getattr(args, "extra_xyz_cudem_force", False)),
                )

                # Attach to run report (report is initialized later; stash on args for now)
                setattr(args, "_xyz_auto_report", auto_rep)

                # Make these participate in the normal --extra-xyz normalization flow
                if auto_xyz:
                    working_crs = str(cfg.working_srs)
                    reprojected_xyz = []
                    for xyz_path in auto_xyz:
                        try:
                            reproj_path = _reproject_xyz_file(
                                xyz_path,
                                # Source CRS is the dlim output CRS (horizontal-only). Z is preserved.
                                src_crs=dlim_crs,
                                dst_crs=working_crs,
                                cache_dir=Path(cfg.cache_root) / "xyz",
                            )
                            reprojected_xyz.append(reproj_path)
                            log.info("[XYZ][DLIM] Reprojected %s -> %s (%s)",
                                     xyz_path.name, reproj_path.name, working_crs)
                        except Exception as e:
                            log.warning("[XYZ][DLIM] Failed to reproject %s: %s. Using original.", xyz_path.name, e)
                            reprojected_xyz.append(xyz_path)

                    if not getattr(args, "extra_xyz", None):
                        args.extra_xyz = []
                    args.extra_xyz.extend([str(p) for p in reprojected_xyz])

                    # Set the CRS to working CRS since we reprojected
                    cfg.extra_xyz_crs = working_crs

        # Normalize extra XYZ inputs (files or directories). Prefer --extra-xyz; --river-soundings is a hidden alias.
        # Supports repeats, comma-separated lists, and directories.
        if getattr(args, "extra_xyz", None):
            xyz_files = []
            for item in args.extra_xyz:
                if item is None:
                    continue
                parts = [p.strip() for p in str(item).split(",") if p.strip()]
                for part in parts:
                    pth = Path(part)
                    if pth.is_dir():
                        for ext in ("*.xyz", "*.csv", "*.txt", "*.dat", "*.gpkg", "*.shp", "*.geojson", "*.json"):
                            xyz_files.extend(sorted(pth.glob(ext)))
                    else:
                        xyz_files.append(pth)

            # de-dup while preserving order
            seen = set()
            xyz_files2 = []
            for pth in xyz_files:
                sp = str(pth)
                if sp not in seen:
                    seen.add(sp)
                    xyz_files2.append(pth)

            if xyz_files2:
                cfg.river_soundings = ",".join(str(p) for p in xyz_files2)
                log.info("[XYZ] Using %d external bathymetry file(s):", len(xyz_files2))
                for pth in xyz_files2:
                    log.info("   - %s", str(pth))

# If glint correction was requested but SDB isn't being run, accept the flag but ignore it.
    if getattr(cfg, "glint_correct", False) and ("sdb" not in cfg.methods):
        log.info("[GLINT] --glint-correct set, but methods does not include 'sdb'; ignoring glint options for this run.")
        cfg.glint_correct = False

    # Fatal post-processing issues that should cause a non-zero exit even if a
    # "final" raster exists (e.g., requested reprojection outputs missing).
    # NOTE: Must be defined in main() scope; referenced by the final warp/emit block.
    fatal_errors: List[str] = []

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pipeline_version": getattr(constants, "PIPELINE_VERSION", "unknown"),
        "config": {
            "aoi": cfg.aoi,
            "aoi_tile": getattr(cfg, "aoi_tile", None),
            "tile_bbox": list(getattr(cfg, "tile_bbox", None)) if getattr(cfg, "tile_bbox", None) else None,
            "tile_buffer_km": float(getattr(cfg, "tile_buffer_km", 0.0) or 0.0),
            "buff_frac": float(getattr(args, "buff", 0.0) or 0.0),
            "tile_edge_taper_enabled": bool(getattr(cfg, "tile_edge_taper_enabled", True)),
            "tile_edge_taper_km": float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0),
            "tile_edge_smooth_sigma_km": float(getattr(cfg, "tile_edge_smooth_sigma_km", 0.0) or 0.0),
            "sdb_model_bank_enabled": bool(getattr(cfg, "sdb_model_bank_enabled", True)),
            "sdb_model_bank": str(getattr(cfg, "sdb_model_bank", "auto")),
            "sdb_bank_max_samples": int(getattr(cfg, "sdb_bank_max_samples", 100000)),
            "sdb_bank_seed": int(getattr(cfg, "sdb_bank_seed", 1337)),
            "sdb_bank_retrain_min_new": int(getattr(cfg, "sdb_bank_retrain_min_new", 2000)),
            "sdb_model_cache_enabled": bool(getattr(cfg, "sdb_model_cache_enabled", False)),
            "sdb_model_cache_key": str(getattr(cfg, "sdb_model_cache_key", "auto")),
            "start": cfg.start_date,
            "end": cfg.end_date,
            "methods": cfg.methods,
            "priority": cfg.priority,
            "require_river_constraints": str(getattr(cfg, "require_river_constraints", "none")),
            "out_dir": str(cfg.out_dir),
            "river_dem": str(cfg.river_dem) if cfg.river_dem else None,
            "working_srs": str(cfg.working_srs),
            "final_out_srs": str(cfg.final_out_srs),
            "depth_value_type": "depth",
            "depth_units": "m",
            "depth_sign": "negative_down",
            "depth_reference": "water_surface",
        }
    }
    # Merge WAFFLES domain inference into the run report (if available)
    try:
        if isinstance(locals().get('domain_inference'), dict) and domain_inference:
            report.update(domain_inference)
    except Exception:
        pass

    # Include any auto-fetched CUDEM soundings in the run report
    if hasattr(args, "_xyz_auto_report"):
        report["xyz_auto"] = getattr(args, "_xyz_auto_report")

    sdb_raster = None
    river_raster = None

    if "sdb" in cfg.methods:
        sdb_raster = run_sdb(cfg, report)

    if "river" in cfg.methods:
        river_raster = run_river(cfg, report)

    # ------------------------------------------------------------------
    # Guardrail: do not fuse PRIOR-ONLY (or otherwise under-constrained) river
    # unless it meets the requested constraint requirement.
    # ------------------------------------------------------------------
    river_for_fuse = river_raster
    river_excluded = None
    try:
        req = str(getattr(cfg, "require_river_constraints", "none") or "none").lower().strip()
        cs = report.get("river", {}).get("constraints_summary", {})
        if req != "none" and isinstance(cs, dict) and (cs.get("meets_requirement") is False):
            river_for_fuse = None
            river_excluded = {
                "reason": "river_constraints_unmet",
                "requirement": req,
                "unmet_reasons": list(cs.get("unmet_reasons", [])),
                "level": cs.get("level", "UNKNOWN"),
            }
            log.warning(
                "[RIVER][CONSTRAINTS] Excluding river from fusion (requirement=%s unmet=%s level=%s).",
                req,
                ",".join(river_excluded.get("unmet_reasons") or []),
                str(river_excluded.get("level")),
            )
    except Exception:
        logging.getLogger(__name__).debug("Constraint guardrail check failed; continuing.", exc_info=True)

    # If fusion would have no sources after guardrails, skip it explicitly.
    if sdb_raster is None and river_for_fuse is None:
        report["fusion"] = {
            "status": "skipped",
            "reason": "no_fusable_sources",
            "note": "Both sources missing or river excluded by constraint guardrail.",
            "river_excluded": river_excluded,
        }
        final = None
    else:
        final = fuse(cfg, sdb_raster, river_for_fuse, report)
        if river_excluded is not None:
            report.setdefault("fusion", {}).setdefault("notes", [])
            report["fusion"]["notes"].append(river_excluded)

    # Optional intelligent gap filling (Tier 1–2): prior + residual interpolation
    if cfg.gapfill_enabled and final:
        try:
            from gapfill_intelligent import gapfill_depth_raster, GapfillConfig

            # HQ points: prefer explicit --gapfill-hq; otherwise fall back to --extra-xyz (if present).
            hq_files: List[str] = []
            if cfg.gapfill_hq:
                hq_files = [str(p) for p in cfg.gapfill_hq if p]
            else:
                # args.extra_xyz may not exist in older CLI variants
                ex = getattr(args, "extra_xyz", None)
                if ex:
                    if isinstance(ex, (list, tuple)):
                        hq_files = [str(p) for p in ex if p]
                    else:
                        hq_files = [str(ex)]

            if not hq_files:
                log.warning("[GAPFILL] Enabled but no HQ points provided (--gapfill-hq or --extra-xyz). Skipping.")
            else:
                combined_dir = ensure_dir(cfg.out_dir / "combined")
                out_gap = combined_dir / "bathy_combined_depth_gapfill.tif"
                out_sig = combined_dir / "bathy_combined_depth_gapfill_sigma.tif"
                out_prov = combined_dir / "bathy_combined_depth_gapfill_provenance.tif"

                # Water mask preference order:
                #   1) explicit --gapfill-water-mask
                #   2) waffles coastline mask (most recent in cache)
                #   3) implicit: finite pixels in prior
                wm: Optional[Path] = None
                if cfg.gapfill_water_mask and Path(cfg.gapfill_water_mask).exists():
                    wm = Path(cfg.gapfill_water_mask)
                else:
                    wm = _find_latest_waffles_mask(cfg.cache_root)

                # River aids (optional): xs gpkg and river corridor mask from river output.
                xs_gpkg = cfg.out_dir / "river" / "river_xs_params.gpkg"
                if not xs_gpkg.exists():
                    xs_gpkg = None
                river_mask = None
                if river_raster and Path(river_raster).exists():
                    river_mask = Path(river_raster)

                # Build GapfillConfig from pipeline config
                gcfg = GapfillConfig(
                    interpolation_method=cfg.gapfill_method,
                    rbf_function="thin_plate",
                    prior_sigma_default=1.0,
                    rbf_min_points=25,
                    component_min_points=10,
                    river_smooth_sigma_m=cfg.gapfill_river_smooth_sigma_m,
                    residual_sigma_floor=0.25,
                    distance_sigma_scale_m=1500.0,
                    output_cudem_xyz=cfg.gapfill_output_cudem_xyz,
                )
                
                # Call the gap-fill function
                gapfill_stats = gapfill_depth_raster(
                    prior_raster=Path(final),
                    hq_point_files=hq_files,
                    out_raster=out_gap,
                    out_sigma=out_sig,
                    out_provenance=out_prov,
                    cfg=gcfg,
                    prior_sigma_raster=cfg.gapfill_prior_sigma_raster,
                    water_mask_raster=wm,
                    river_mask_raster=river_mask,
                    bank_elev_raster=cfg.gapfill_bank_elev_raster,
                    xs_params_gpkg=xs_gpkg,
                    logger=log,
                )

                report.setdefault("gapfill", {})["status"] = "success"
                report["gapfill"]["stats"] = gapfill_stats
                report["gapfill"]["outputs"] = {
                    "depth": str(out_gap),
                    "sigma": str(out_sig),
                    "provenance": str(out_prov),
                }
                if cfg.gapfill_output_cudem_xyz and "cudem_xyz" in gapfill_stats:
                    report["gapfill"]["outputs"]["cudem_xyz"] = gapfill_stats["cudem_xyz"]
                final = out_gap
        except Exception as e:
            report.setdefault("gapfill", {})["status"] = "failed"
            report["gapfill"]["error"] = str(e)
            log.warning("[GAPFILL] Failed: %s", e)

    # Prefer reporting the horizontal-only final outputs when available
    final_for_user = final

    # Default: also emit reprojected rasters in final_out_srs (horizontal-only; depth metadata tags included).
    try:
        dst_srs = cfg.final_out_srs
        # Derive a deterministic tag from the destination SRS for filenames (no canonical guessing).
        try:
            m = re.search(r"(\d{4,6})", str(dst_srs))
            dst_tag = f"epsg{m.group(1)}" if m else "dst"
        except Exception:
            dst_tag = "dst"
        # Determine horizontal EPSG code for seam-stability logic (handles compound CRS like epsg:4269+5714).
        dst_horiz_epsg = None
        try:
            m2 = re.search(r"(\d{4,6})", str(dst_srs))
            dst_horiz_epsg = int(m2.group(1)) if m2 else None
        except Exception:
            dst_horiz_epsg = None
        dst_is_4269 = (dst_horiz_epsg == 4269)

        if final:
            final_p = Path(final)
            combined_dir = ensure_dir(cfg.out_dir / "combined")
            out_name = f"{final_p.stem}_{dst_tag}.tif"
            warped = warp_raster_to_srs(final_p, combined_dir / out_name, dst_srs)
            if warped:
                report.setdefault("outputs", {})["combined_warped_context"] = str(warped)

                # Tile deliverable: always clip back to the original tile AOI when destination is EPSG:4269.
                bbox_tile = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                deliverable = Path(warped)
                try:
                    if bbox_tile and dst_is_4269:
                        clipped = combined_dir / f"{deliverable.stem}_tile.tif"
                        clipped = _clip_raster_to_bbox(deliverable, bbox_tile, clipped, nodata=float(getattr(cfg, 'final_nodata', -9999.0)))
                        if clipped and Path(clipped).exists():
                            deliverable = Path(clipped)
                            report.setdefault("outputs", {})["combined_warped"] = str(deliverable)
                        else:
                            report.setdefault("outputs", {})["combined_warped"] = str(deliverable)
                    else:
                        report.setdefault("outputs", {})["combined_warped"] = str(deliverable)
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                    report.setdefault("outputs", {})["combined_warped"] = str(deliverable)

                final_for_user = str(deliverable)

            else:
                # Treat requested final reprojection failures as fatal: users depend on stable
                # EPSG:4269 outputs for tile-to-tile comparisons and downstream tooling.
                msg = f"Final warp failed: {final_p.name} -> {out_name} ({dst_srs})"
                fatal_errors.append(msg)
                report.setdefault("outputs", {})["combined_warped"] = None
                log.error("[WARP] %s", msg)

            if warped:
                # Final domain clipping is handled by _apply_final_domain_policy() after pipeline completion.
                # Seam-stability edge taper (operational tiling) on the *tile deliverable*.
                try:
                    bbox = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                    deliver = Path(report.get("outputs", {}).get("combined_warped") or warped)
                    if bbox and dst_is_4269 and deliver.exists():
                        try:
                            if bool(getattr(cfg, "tile_edge_taper_enabled", True)) and float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0) > 0:
                                m = _compute_edge_band_metrics_epsg4269(
                                    deliver,
                                    bbox,
                                    band_km=float(getattr(cfg, "tile_edge_metrics_band_km", 2.0) or 2.0),
                                    smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                    nodata=float(getattr(cfg, "final_nodata", -9999.0)),
                                )
                                report.setdefault("seams", {})["combined_edge_metrics"] = m
                                tapered = _apply_tile_edge_taper_epsg4269(
                                    deliver,
                                    bbox,
                                    taper_km=float(getattr(cfg, "tile_edge_taper_km", 2.0) or 2.0),
                                    smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                    nodata=float(getattr(cfg, "final_nodata", -9999.0)),
                                )
                                if tapered and Path(tapered).exists():
                                    shutil.move(str(tapered), str(deliver))
                                    report.setdefault("outputs", {})["combined_edge_tapered"] = str(deliver)
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)


        if sdb_raster:
            sdb_p = Path(sdb_raster)
            sdb_dir = ensure_dir(cfg.out_dir / "sdb")
            sdb_out_name = f"{sdb_p.stem}_{dst_tag}.tif"
            warped = warp_raster_to_srs(sdb_p, sdb_dir / sdb_out_name, dst_srs)
            if warped:
                report.setdefault("outputs", {})["sdb_warped_context"] = str(warped)
                deliverable = Path(warped)
                try:
                    bbox_tile = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                    if bbox_tile and dst_is_4269:
                        clipped = sdb_dir / f"{deliverable.stem}_tile.tif"
                        clipped = _clip_raster_to_bbox(deliverable, bbox_tile, clipped, nodata=float(getattr(cfg, 'final_nodata', -9999.0)))
                        if clipped and Path(clipped).exists():
                            deliverable = Path(clipped)
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                report.setdefault("outputs", {})["sdb_warped"] = str(deliverable)
            else:
                msg = f"SDB warp failed: {sdb_p.name} -> {sdb_out_name} ({dst_srs})"
                fatal_errors.append(msg)
                report.setdefault("outputs", {})["sdb_warped"] = None
                log.error("[WARP] %s", msg)

        if river_raster:
            r_p = Path(river_raster)
            river_dir = ensure_dir(cfg.out_dir / "river")
            river_out_name = f"{r_p.stem}_{dst_tag}.tif"
            warped = warp_raster_to_srs(r_p, river_dir / river_out_name, dst_srs)
            if warped:
                report.setdefault("outputs", {})["river_warped_context"] = str(warped)

                # Tile deliverable: clip back to original tile AOI when destination is EPSG:4269.
                deliverable = Path(warped)
                try:
                    bbox_tile = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                    if bbox_tile and dst_is_4269:
                        clipped = river_dir / f"{deliverable.stem}_tile.tif"
                        clipped = _clip_raster_to_bbox(deliverable, bbox_tile, clipped, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                        if clipped and Path(clipped).exists():
                            deliverable = Path(clipped)
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                report.setdefault("outputs", {})["river_warped"] = str(deliverable)

                # Final safety clip: keep river outputs inside waffles water mask (water=0, land=1).
                try:
                    wm = None
                    ro = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
                    wm = ro.get("waffles_water_mask")
                    if wm and Path(wm).exists():
                        _clip_raster_to_mask_reproject(Path(deliverable), Path(wm), inside_value=0, invert=False, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                # Optional edge seam taper on the tile deliverable
                try:
                    bbox = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                    if bbox and dst_is_4269 and deliverable.exists():
                        if bool(getattr(cfg, "tile_edge_taper_enabled", True)) and float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0) > 0:
                            m = _compute_edge_band_metrics_epsg4269(
                                deliverable,
                                bbox,
                                band_km=float(getattr(cfg, "tile_edge_metrics_band_km", 2.0) or 2.0),
                                smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                            )
                            report.setdefault("seams", {})["river_edge_metrics"] = m
                            tapered = _apply_tile_edge_taper_epsg4269(
                                deliverable,
                                bbox,
                                taper_km=float(getattr(cfg, "tile_edge_taper_km", 2.0) or 2.0),
                                smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                            )
                            if tapered and Path(tapered).exists():
                                shutil.move(str(tapered), str(deliverable))
                                report.setdefault("outputs", {})["river_edge_tapered"] = str(deliverable)
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
            else:
                msg = f"River warp failed: {r_p.name} -> {river_out_name} ({dst_srs})"
                fatal_errors.append(msg)
                report.setdefault("outputs", {})["river_warped"] = None
                log.error("[WARP] %s", msg)
                # (No post-processing; warp failed)

            # Also warp river bottom elevation (orthometric heights, typically NAVD88) if available.
            try:
                river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
                bed_src = river_outputs.get("bottom_elevation")
                if bed_src and Path(bed_src).exists():
                    bed_p = Path(bed_src)
                    bed_out_name = f"{bed_p.stem}_{dst_tag}.tif"
                    bed_warp = warp_raster_to_srs(
                        bed_p,
                        river_dir / bed_out_name,
                        dst_srs,
                        write_depth_metadata=False,
                    )
                    if bed_warp:
                        report.setdefault("outputs", {})["river_bottom_warped_context"] = str(bed_warp)

                        deliverable = Path(bed_warp)
                        # Tile deliverable: clip back to original tile AOI when destination is EPSG:4269.
                        try:
                            bbox_tile = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                            if bbox_tile and dst_is_4269:
                                clipped = river_dir / f"{deliverable.stem}_tile.tif"
                                clipped = _clip_raster_to_bbox(deliverable, bbox_tile, clipped, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                                if clipped and Path(clipped).exists():
                                    deliverable = Path(clipped)
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                        report.setdefault("outputs", {})["river_bottom_warped"] = str(deliverable)
                        try:
                            apply_elevation_metadata(deliverable, vertical_datum="NAVD88")
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                        # Final safety clip: keep river bottom inside waffles water mask (water=0, land=1).
                        try:
                            wm = None
                            ro = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
                            wm = ro.get("waffles_water_mask")
                            if wm and Path(wm).exists():
                                _clip_raster_to_mask_reproject(deliverable, Path(wm), inside_value=0, invert=False, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                        # Optional edge seam taper on tile deliverable
                        try:
                            bbox = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                            if bbox and dst_is_4269 and deliverable.exists():
                                if bool(getattr(cfg, "tile_edge_taper_enabled", True)) and float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0) > 0:
                                    m = _compute_edge_band_metrics_epsg4269(
                                        deliverable,
                                        bbox,
                                        band_km=float(getattr(cfg, "tile_edge_metrics_band_km", 2.0) or 2.0),
                                        smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                        nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                                    )
                                    report.setdefault("seams", {})["river_bottom_edge_metrics"] = m
                                    tapered = _apply_tile_edge_taper_epsg4269(
                                        deliverable,
                                        bbox,
                                        taper_km=float(getattr(cfg, "tile_edge_taper_km", 2.0) or 2.0),
                                        smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                                        nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                                    )
                                    if tapered and Path(tapered).exists():
                                        shutil.move(str(tapered), str(deliverable))
                                        report.setdefault("outputs", {})["river_bottom_edge_tapered"] = str(deliverable)
                        except Exception:
                            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
            except Exception:
                logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        # Optional: combined bottom elevation (NAVD88) for river + SDB where possible.
        # River provides bottom directly; SDB requires a WSE(NAVD88) raster to convert depth->bottom.
        try:
            import rasterio
            import numpy as np

            out_combined_bottom = ensure_dir(cfg.out_dir / "combined") / f"bathy_bottom_navd88_{dst_tag}.tif"

            river_bottom = report.get("outputs", {}).get("river_bottom_warped")
            sdb_depth = report.get("outputs", {}).get("sdb_warped")

            # Prefer a WSE raster if the SDB output folder contains one.
            sdb_wse = None
            try:
                sdb_out_dir = Path(cfg.out_dir) / "sdb"
                sdb_wse = find_sdb_wse_navd88_raster(sdb_out_dir)
            except Exception:
                sdb_wse = None

            if river_bottom and Path(river_bottom).exists():
                with rasterio.open(river_bottom) as rb:
                    prof = rb.profile.copy()
                    prof.update(dtype="float32", count=1, nodata=float(getattr(cfg, 'river_nodata', -9999.0)), compress="deflate")
                    out_arr = rb.read(1).astype("float32")
                    nod = float(prof.get("nodata", -9999.0))

                # If we can convert SDB depth to bottom, fill remaining nodata outside river.
                if sdb_depth and Path(sdb_depth).exists() and sdb_wse and Path(sdb_wse).exists():
                    with rasterio.open(sdb_depth) as sd, rasterio.open(sdb_wse) as sw:
                        sd_transform = sd.transform
                        sd_crs = sd.crs
                        # Reproject WSE to match depth grid if needed
                        wse = np.zeros((sd.height, sd.width), dtype=np.float32)
                        from rasterio.warp import reproject, Resampling
                        reproject(
                            source=sw.read(1),
                            destination=wse,
                            src_transform=sw.transform,
                            src_crs=sw.crs,
                            dst_transform=sd.transform,
                            dst_crs=sd.crs,
                            resampling=Resampling.bilinear,
                        )
                        depth = sd.read(1).astype("float32")
                        dnod = float(sd.nodata) if sd.nodata is not None else -9999.0
                        # Depth is negative-down in this pipeline; convert to positive-down magnitude
                        depth_mag = np.where(np.isfinite(depth) & (depth != dnod), np.abs(depth), np.nan)
                        bottom_sdb = wse - depth_mag

                    # Warp SDB bottom to match river_bottom grid
                    with rasterio.open(river_bottom) as rb:
                        bottom_sdb_on_rb = np.full((rb.height, rb.width), np.nan, dtype=np.float32)
                        reproject(
                            source=bottom_sdb,
                            destination=bottom_sdb_on_rb,
                            src_transform=sd_transform,
                            src_crs=sd_crs,
                            dst_transform=rb.transform,
                            dst_crs=rb.crs,
                            resampling=Resampling.bilinear,
                        )

                    fill = (out_arr == nod) & np.isfinite(bottom_sdb_on_rb)
                    if np.any(fill):
                        out_arr[fill] = bottom_sdb_on_rb[fill]

                with rasterio.open(out_combined_bottom, "w", **prof) as dst:
                    dst.write(out_arr.astype("float32"), 1)

                try:
                    apply_elevation_metadata(out_combined_bottom, vertical_datum="NAVD88")
                except Exception:
                    logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

                # Optional crop/taper to tile AOI for combined bottom output
                try:
                  bbox = _parse_aoi_bbox(getattr(cfg, "aoi_tile", None) or getattr(cfg, "aoi", None))
                  if bbox and dst_is_4269:
                      _crop_raster_extent_to_bbox(out_combined_bottom, bbox, nodata=float(getattr(cfg, 'river_nodata', -9999.0)))
                      if bool(getattr(cfg, "tile_edge_taper_enabled", True)) and float(getattr(cfg, "tile_edge_taper_km", 0.0) or 0.0) > 0:
                          m = _compute_edge_band_metrics_epsg4269(
                              out_combined_bottom,
                              bbox,
                              band_km=float(getattr(cfg, "tile_edge_metrics_band_km", 2.0) or 2.0),
                              smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                              nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                          )
                          report.setdefault("seams", {})["combined_bottom_edge_metrics"] = m
                          tapered = _apply_tile_edge_taper_epsg4269(
                              out_combined_bottom,
                              bbox,
                              taper_km=float(getattr(cfg, "tile_edge_taper_km", 2.0) or 2.0),
                              smooth_sigma_km=float(getattr(cfg, "tile_edge_smooth_sigma_km", 10.0) or 10.0),
                              nodata=float(getattr(cfg, "river_nodata", -9999.0)),
                          )
                          if tapered and Path(tapered).exists():
                              shutil.move(str(tapered), str(out_combined_bottom))
                              report.setdefault("outputs", {})["combined_bottom_edge_tapered"] = str(out_combined_bottom)
                except Exception:
                  logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
                report.setdefault("outputs", {})["combined_bottom_navd88"] = str(out_combined_bottom)

        except Exception:
            # Optional output only; do not fail the run if conversion inputs are absent.
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)
    except Exception as e:
        log.warning("[WARP] Could not create final reprojected rasters: %s", e)


    report_path = cfg.out_dir / "bathy_report.json"
    write_json(report_path, report)
    # Explicit IO manifest (paths observed in report + executed commands; no guessing)
    try:
        io_json, io_md = write_io_manifest(cfg.out_dir, report)
    except Exception as e:
        log.debug("Optional IO manifest write failed: %s", e)
        io_json, io_md = None, None
    try:
        from flight_recorder import emit_artifact_written
        emit_artifact_written(report_path, kind="json", role="bathy_report")
        if io_json is not None:
            emit_artifact_written(io_json, kind="json", role="io_manifest")
        if io_md is not None:
            emit_artifact_written(io_md, kind="md", role="io_manifest")
        # Also emit all artifacts we know about from the report (so summaries can be derived from the flight recorder)
        _emit_artifacts_from_report(report)
    except Exception as e:
        logging.getLogger(__name__).debug("Optional flight-recorder emit failed: %s", e)
    log.info(f"Report written: {report_path}")

    # Adjacent-tile seam comparisons (explicit neighbor io_manifest.json paths; no auto-discovery)
    try:
        seam_ios = list(getattr(args, "seam_compare_with_io", []) or [])
        seam_io_list_path = getattr(args, "seam_compare_with_io_list", None)
        if seam_io_list_path:
            p = Path(str(seam_io_list_path))
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    seam_ios.append(s)
            else:
                raise FileNotFoundError(f"--seam-compare-with-io-list not found: {p}")
        if seam_ios:
            from seam_metrics import compute_seam_metrics, load_primary_raster_from_io_manifest
            import json as _json

            this_raster = None
            if final_for_user:
                this_raster = Path(str(final_for_user))
            elif final:
                this_raster = Path(str(final))
            if this_raster is None or (not this_raster.exists()):
                raise RuntimeError("Seam compare requested but this run has no final raster output.")

            seam_results = []
            for nio in seam_ios:
                nio_p = Path(str(nio))
                # Explicit-only: require the manifest file to exist
                if not nio_p.exists():
                    seam_results.append({
                        "status": "error",
                        "neighbor_io_manifest": str(nio_p),
                        "error": "Neighbor io_manifest.json not found",
                    })
                    continue
                try:
                    neighbor_raster = load_primary_raster_from_io_manifest(nio_p)
                    m = compute_seam_metrics(this_raster, neighbor_raster, strip_px=int(getattr(args, "seam_strip_px", 3)))
                    m["neighbor_io_manifest"] = str(nio_p)
                    seam_results.append(m)
                except Exception as e:
                    seam_results.append({
                        "status": "error",
                        "neighbor_io_manifest": str(nio_p),
                        "error": str(e),
                    })

            out_seam_json = Path(cfg.out_dir) / "seam_comparisons.json"
            out_seam_json.write_text(_json.dumps({
                "this_raster": str(this_raster),
                "strip_px": int(getattr(args, "seam_strip_px", 3)),
                "comparisons": seam_results,
            }, indent=2), encoding="utf-8")

            report.setdefault("seams", {})["adjacent_tile_comparisons"] = seam_results
            report.setdefault("outputs", {})["seam_comparisons_json"] = str(out_seam_json)
            # Update report on disk to include seam results
            try:
                write_json(report_path, report)
            except Exception:
                pass
            log.info(f"Seam comparisons written: {out_seam_json}")
    except Exception:
        logging.getLogger(__name__).debug("Optional seam comparison step failed; continuing.", exc_info=True)
    
    # NEW v0.7.1: Create unified bathymetry report
    try:
        from river_diagnostics import create_unified_bathy_report
        
        sdb_out = cfg.out_dir / "sdb" if "sdb" in cfg.methods else None
        river_out = cfg.out_dir / "river" if "river" in cfg.methods else None
        
        unified_path = create_unified_bathy_report(
            output_dir=cfg.out_dir,
            sdb_output=sdb_out,
            river_output=river_out,
            methods=cfg.methods,
            priority=cfg.priority
        )
        log.info(f"Unified report written: {unified_path}")
    except ImportError:
        log.warning("river_diagnostics module not available - unified report not created")
    except Exception as e:
        log.warning(f"Could not create unified report: {e}")

    # Human-friendly, scan-friendly console summary
    try:
        from run_summary import print_human_run_summary

        summary_stats = {
            "command": " ".join(sys.argv),
            "aoi": cfg.aoi,
            "time_window": {"start": cfg.start_date, "end": cfg.end_date},
            "methods": list(cfg.methods),
            "priority": cfg.priority,
            "outputs": report.get("outputs", {}),
        }
        print_human_run_summary(summary_stats, log_fn=log.info)
        report["human_summary"] = summary_stats
    except Exception as e:
        log.debug(f"Human summary skipped: {e}")


    # Write detailed run summaries (technical/scientific/human) using the in-memory report
    try:
        from run_summary import write_run_summary_files
        from flight_recorder import current_run_id, current_flight_path

        rid = current_run_id()
        frp = current_flight_path()
        write_run_summary_files(cfg.out_dir, run_id=rid, stats=report, fr_path=(frp or None))
        log.info(f"Run summaries written under: {Path(cfg.out_dir) / 'run_logs'}")
    except Exception as e:
        log.debug(f"Run summary file write skipped: {e}")



    # Strict mode: run contract tests + basic output sanity checks
    if cfg.strict:
        try:
            from contract_tests import run_contract_tests_cli

            rc = run_contract_tests_cli(cfg.out_dir)
            if rc != 0:
                raise RuntimeError(f"Contract tests failed (exit={rc})")
        except Exception as e:
            log.error("[STRICT] %s", e)
            raise

        # Sanity check: final raster should exist and contain at least some finite pixels.
        try:
            import rasterio
            import numpy as np

            final_path = None
            if final_for_user and Path(final_for_user).exists():
                final_path = Path(final_for_user)
            elif final and Path(final).exists():
                final_path = Path(final)

            if final_path is None or not final_path.exists():
                raise RuntimeError("Final depth raster missing")

            with rasterio.open(final_path) as ds:
                arr = ds.read(1, masked=True)
                finite = np.isfinite(arr.filled(np.nan))
                n_finite = int(np.sum(finite))
                if n_finite < 100:
                    raise RuntimeError(f"Final raster has too few valid pixels (n_valid={n_finite})")
                report.setdefault("sanity", {})["final_valid_pixels"] = n_finite
        except Exception as e:
            log.error("[STRICT][SANITY] %s", e)
            raise



    # Output contract: ensure standard filenames exist (metrics/debug rely on these)
    try:
        _ensure_output_contract(
            Path(getattr(args, "out_dir")),
            Path(getattr(args, "cache_root", Path(getattr(args, "out_dir")) / "cache")),
        )
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Final domain policy: clip final products based on enabled methods
    try:
        _apply_final_domain_policy(cfg, cfg.out_dir, Path(cfg.derived_cache_root))
    except Exception:
        logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

    # Output retention policy: keep only deliverables by default.
    # Only run this step when the pipeline produced at least one final output and there were no fatal errors.
    try:
        if (not fatal_errors) and (final_for_user or final):
            _organize_outputs(cfg)
    except Exception:
        # Do not turn a successful scientific run into a failure just because cleanup/move failed.
        logging.getLogger(__name__).debug("Output organization failed; continuing.", exc_info=True)

    # If any requested final reprojection outputs failed, treat the run as failed.
    if fatal_errors:
        report["status"] = "failed"
        report["fatal_errors"] = list(fatal_errors)
        for m in fatal_errors:
            log.error("[FATAL] %s", m)
    # Output retention policy (final outputs only by default; intermediates optional)
    if (not fatal_errors) and final:
        try:
            _apply_output_retention_policy(cfg, log, report, final_path=final, final_for_user_path=final_for_user)
        except Exception as e:
            log.error("[OUTPUT][WARN] Retention policy failed (leaving outputs as-is): %s", str(e))

    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 70)
    log.info(f"SDB: {report.get('sdb', {}).get('status', 'skipped')}")
    log.info(f"River: {report.get('river', {}).get('status', 'skipped')}")
    log.info(f"Fusion: {report.get('fusion', {}).get('status', 'skipped')}")
    log.info(f"Final output: {final_for_user if final_for_user else (final if final else 'None')}")
    log.info("=" * 70)

    if fatal_errors:
        return 2
    return 0 if final else 2


if __name__ == "__main__":
    raise SystemExit(main())
