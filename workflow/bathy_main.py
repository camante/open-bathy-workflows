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


# Configure logging before importing other pipeline modules
import os as _os
# Force a headless-safe Matplotlib backend early (prevents TkAgg/Tkinter crashes under multiprocessing)
_os.environ.setdefault('MPLBACKEND', 'Agg')

import logging
import sys
from hybrid_merge import merge_hybrid_river_bed
from river_hybrid_stage import run_hybrid_river_stage
from river_structured_stage import run_structured_river_stage
from river_v1_pipeline import run_river_v1_stage
from river_v2_context import RiverV2Context, resolve_river_v2_bank_guidance_inputs
from river_v2_pipeline import run_river_v2_pipeline
from river_v2_contract import RIVER_V2_METHODS
from constants import PIPELINE_VERSION


def _build_simple_river_status_from_v2(*, report: dict, v2_result, v2_context) -> dict:
    status = simple_river_stage_status_placeholder()
    # upstream/base stages
    auth_base = (((report.get("authoritative_base") or {}).get("outputs") or {}).get("authoritative_base")
                 or ((report.get("config") or {}).get("authoritative_base")))
    if auth_base and Path(auth_base).exists():
        status = mark_stage_implemented(status, stage_id="authoritative_base", output_artifact=str(auth_base))
    river_domain = (((report.get("guidance_domains") or {}).get("outputs") or {}).get("river_guidance_domain_mask")
                    or ((report.get("outputs") or {}).get("river_active_domain_mask")))
    if river_domain and Path(river_domain).exists():
        status = mark_stage_implemented(status, stage_id="river_guidance_domain", output_artifact=str(river_domain))

    stage_results = getattr(v2_result, 'stage_results', {}) or {}
    mapping = {
        'river_centerline': 'river_centerline',
        'centerline_wse_proxy': 'centerline_wse_proxy',
        'centerline_authoritative_bed': 'centerline_authoritative_bed',
        'centerline_observed_offset': 'centerline_observed_offset',
        'centerline_offset_modeled': 'centerline_offset_modeled',
        'river_centerline_bed_backbone': 'river_centerline_bed_backbone',
        'river_primary_surface': 'river_primary_surface',
        'river_primary_surface_authoritative_applied': 'river_primary_surface_authoritative_locked',
    }
    for v2_stage_id, simple_stage_id in mapping.items():
        result = stage_results.get(v2_stage_id)
        if result is not None:
            status = mark_stage_implemented(
                status,
                stage_id=simple_stage_id,
                output_artifact=str(result.output_artifact),
                record_count=getattr(result, 'record_count', None),
                receipt_path=str(getattr(result, 'receipt_path', '')) or None,
                warnings=list(getattr(result, 'warnings', []) or []),
            )
    failed = getattr(v2_result, 'failed_stage', None)
    if failed in mapping:
        status = mark_stage_failed(status, stage_id=mapping[failed], error=str(getattr(v2_result, 'error', failed)))
    conditioned = Path(v2_context.cfg.out_dir) / 'combined' / 'conditioned_final_dem_internal.tif'
    if conditioned.exists():
        status = mark_stage_implemented(status, stage_id='conditioned_final_dem_internal', output_artifact=str(conditioned))
    final_dem = Path(v2_context.cfg.out_dir) / 'combined' / 'DEM_enhanced.tif'
    if final_dem.exists():
        status = mark_stage_implemented(status, stage_id='DEM_enhanced', output_artifact=str(final_dem))
    return status


def _enforce_v2_final_route_selection(*, river_outputs: dict, active_locked_surface: str) -> dict:
    return _set_river_v2_final_guidance_surface(
        river_outputs=river_outputs,
        locked_surface=active_locked_surface,
    )




def _canonical_river_v2_output_map(v2_outputs: dict) -> dict[str, str]:
    mapping = {
        "river_v2_summary": "river_v2_summary",
        "river_v2_stage_products_overview": "river_v2_stage_products_overview",
        "river_v2_primary_surface": "river_v2_primary_surface",
        "river_v2_primary_surface_authoritative_applied": "river_v2_primary_surface_authoritative_applied",
        "river_v2_centerline_points": "river_v2_centerline_points",
        "river_v2_wse_points": "river_v2_wse_points",
        "river_v2_authoritative_bed_points": "river_v2_authoritative_bed_points",
        "river_v2_observed_offset_points": "river_v2_observed_offset_points",
        "river_v2_modeled_offset_points": "river_v2_modeled_offset_points",
        "river_v2_backbone_points": "river_v2_backbone_points",
        "river_v2_backbone_dense_points": "river_v2_backbone_dense_points",
    }
    outputs = {}
    for src_key, dst_key in mapping.items():
        value = v2_outputs.get(src_key)
        if value:
            outputs[dst_key] = str(value)
    return outputs

def _register_river_v2_outputs(*, outputs_root: dict, v2_outputs: dict) -> dict:
    outputs_root.update(_canonical_river_v2_output_map(v2_outputs))
    return outputs_root


def _resolve_river_v2_authoritative_support_policy(*, cfg, authoritative_outputs: dict) -> dict:
    mode = str(getattr(cfg, "authoritative_dem_mode", None) or "mixed_requires_metadata").strip().lower()
    if mode not in {"measured_only", "mixed_requires_metadata"}:
        mode = "mixed_requires_metadata"
    authoritative_base = authoritative_outputs.get("authoritative_base")
    support_coverage = authoritative_outputs.get("authoritative_support_coverage")
    support_geometry_count = authoritative_outputs.get("authoritative_support_geometry_count")
    try:
        support_geometry_count = int(support_geometry_count) if support_geometry_count is not None else None
    except (TypeError, ValueError):
        support_geometry_count = None
    if mode == "measured_only":
        artifact = authoritative_base or authoritative_outputs.get("aligned_authoritative_base")
        return {
            "authoritative_dem_mode": mode,
            "trusted_support_mode": "all_finite_cells_trusted",
            "trusted_support_artifact_path": str(artifact) if artifact else None,
            "authoritative_support_policy_warning": None,
            "support_policy_source": "explicit_measured_only_mode",
        }
    if support_coverage and (support_geometry_count is None or support_geometry_count > 0):
        return {
            "authoritative_dem_mode": mode,
            "trusted_support_mode": "metadata_proven_only",
            "trusted_support_artifact_path": str(support_coverage),
            "authoritative_support_policy_warning": None,
            "support_policy_source": "metadata_support_artifact",
        }
    return {
        "authoritative_dem_mode": mode,
        "trusted_support_mode": "low_support_no_trusted_support",
        "trusted_support_artifact_path": None,
        "authoritative_support_policy_warning": (
            "authoritative_bed_metadata_unavailable_low_support_mode"
            if support_geometry_count is None
            else "authoritative_bed_support_coverage_empty_low_support_mode"
        ),
        "support_policy_source": "metadata_unavailable" if support_geometry_count is None else "metadata_support_empty",
    }


def _set_river_v2_final_guidance_surface(*, river_outputs: dict, locked_surface: str) -> dict:
    outputs = dict(river_outputs or {})
    locked = str(locked_surface)
    outputs["primary_river_guidance_surface"] = locked
    outputs["river_primary_surface_authoritative_applied"] = locked
    conflicting_keys = []
    for key in (
        "bottom_elevation_path",
        "depth_terrain_path",
        "merged_bed_tif",
        "river_bottom_navd88_patch",
        "river_channel_surface",
        "river_raster",
    ):
        value = outputs.get(key)
        if not value:
            continue
        if str(value) != locked:
            conflicting_keys.append(key)
    if conflicting_keys:
        raise RuntimeError(
            "River v2 final-route contract violated: legacy river outputs remain eligible for routing: " + ", ".join(conflicting_keys)
        )
    return outputs


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
import subprocess
import time
import numpy as np
from vdatum_utils import convert_sdb_msl_to_navd88
from process_utils import run_cmd
from raster_contract import validate_gdal_output, cached_raster_semantics_valid

# Phase-1 anti-soup refactor: shared helpers
from core.paths import ensure_dir
from core.exec import run_command, run_command_stdout_to_file
from core.fingerprint import hash_key as _hash_key
from core.fingerprint import sha1_text as _stable_hash_str

from core.cmd import build_cmd as _build_cmd
from core.json_io import write_json
from channel_template_invariant import enforce_channel_template_invariant
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
    sanitize_raster_values,
    raster_crs_matches,
    warp_raster_to_srs,
)

# Central constants (versioning, nodata)
import constants
from dataclasses import dataclass, field
from method_activation import determine_method_activation_truth
from river_execution_plan import determine_river_execution_plan
from final_dem_policy import default_final_dem_policy
from final_dem_contract import build_final_dem_contract_summary
from simple_river_stage_contract import simple_river_stage_status_placeholder, mark_stage_implemented, mark_stage_failed
from simple_river_bundle_b import run_simple_river_bundle_b
from sdb_execution_plan import determine_sdb_execution_plan
from river_component_contract import resolve_centerline_component_expectation
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from xs_contracts import validate_soundings_subset_parquet, validate_xs_artifacts


from authoritative_conditioning import (
    build_source_aware_candidate_arrays as _ac_build_source_aware_candidate_arrays,
    compute_support_distance_density_guidance as _ac_compute_support_distance_density_guidance,
    compute_river_anchor_support_fields as _ac_compute_river_anchor_support_fields,
    compute_coastal_sdb_support_confidence as _ac_compute_coastal_sdb_support_confidence,
    support_weighted_condition_arrays as _ac_support_weighted_condition_arrays,
)
from authoritative_cli import build_authoritative_passthrough_args as _build_authoritative_passthrough_args
from authoritative_guidance import build_projected_authoritative_raster as _build_projected_authoritative_raster
from authoritative_guidance import prepare_authoritative_sdb_training_points as _prepare_authoritative_sdb_training_points
from authoritative_guidance import prepare_authoritative_river_soundings_points as _prepare_authoritative_river_soundings_points
from authoritative_guidance import build_sdb_authoritative_support_products as _build_sdb_authoritative_support_products
from authoritative_support import (
    prepare_river_support_from_authoritative as _prepare_river_support_from_authoritative,
    prepare_sdb_guidance_from_authoritative as _prepare_sdb_guidance_from_authoritative,
)
from precedence_audit import summarize_precedence_audit as _summarize_precedence_audit, write_precedence_audit as _write_precedence_audit
from final_route_inputs_stage import collect_final_route_inputs
from guidance_assembly_stage import assemble_guidance_inputs
from deterministic_terrain_stage import run_deterministic_terrain_stage
from final_route_outputs_stage import write_final_route_outputs
from authoritative_cli import record_authoritative_child_passthrough as _record_authoritative_child_passthrough_impl
from authoritative_materialization import resolve_authoritative_base as _resolve_authoritative_base
from river_cli import build_river_skeleton_command as _build_river_skeleton_command
from sdb_cli import build_sdb_command as _build_sdb_command, augment_sdb_command as _augment_sdb_command
from final_run_reporting import build_final_run_state, write_final_run_reporting_bundle as _write_final_run_reporting_bundle
from final_run_stage import execute_final_run_stage as _execute_final_run_stage
from fusion_helpers import copy_fusion_outputs as _copy_fusion_outputs_impl, resolve_river_domain_mask_for_fusion as _resolve_river_domain_mask_for_fusion_impl
from support_classes import SUPPORT_CLASS_CODE_TO_NAME
from nodata_utils import sanitize_array, valid_mask, nodata_mask, resolve_reproject_nodata_value
from sign_semantics import raster_value_semantics, should_expect_negative_depth
from provenance_schema import PROVENANCE_CLASS_CODE_TO_NAME
from final_dem_policy import build_final_dem_policy_dict
from source_guidance_contract import build_river_source_contract, build_sdb_source_contract, summarize_regime_counts
from canonical_river_scaffold import (
    RiverAoiDomains,
    get_river_aoi_domains,
    scaffold_cache_hit,
    scaffold_product_paths,
    scaffold_products_complete,
    persist_scaffold_products,
    cached_scaffold_paths,
    scaffold_domain_metadata,
    write_scaffold_manifest,
)
from trusted_interior import (
    build_authoritative_anchor_support,
    build_soft_guidance_domain,
    build_river_admissibility,
    build_trusted_export_region,
    build_river_trusted_interior,
    restrict_river_admissibility,
    summarize_trusted_export_region,
)
from sdb_results import finalize_sdb_run as _finalize_sdb_run_impl
from sdb_domain_postprocess import load_sdb_artifacts_into_report as _load_sdb_artifacts_into_report
from guidance_domains import ensure_guidance_domains as _ensure_guidance_domains
from guidance_domains import _build_template_aligned_river_water_mask
from river_guidance import (
    apply_guidance_controls_with_reporting as _apply_guidance_controls_with_reporting,
    write_guidance_artifacts_with_reporting as _write_guidance_artifacts_with_reporting,
)
from run_initialization import initialize_run_args as _initialize_run_args, resolve_authoritative_base_args as _resolve_authoritative_base_args
from fusion_postprocess import ensure_river_contribution as _ensure_river_contribution, burn_authoritative_xyz_into_final as _burn_authoritative_xyz_into_final, apply_residual_correction_with_reporting as _apply_residual_correction_with_reporting
from river_masking import build_hydraulic_estuary_hint_mask as _rm_build_hydraulic_estuary_hint_mask
from river_masking import choose_waffles_mask_for_river as _rm_choose_waffles_mask_for_river
from river_masking import clip_channel_mask_for_estuary as _rm_clip_channel_mask_for_estuary
from river_masking import count_mask_water_pixels as _rm_count_mask_water_pixels
from river_masking import determine_effective_methods_from_waffles as _rm_determine_effective_methods_from_waffles
from river_masking import find_latest_waffles_mask as _rm_find_latest_waffles_mask
from river_masking import stage_cached_waffles_mask as _rm_stage_cached_waffles_mask
from river_masking import waffles_water_fraction as _rm_waffles_water_fraction
from river_domain_policy import evaluate_river_domain_summary as _evaluate_river_domain_summary
from river_domain_policy import load_river_domain_summary as _load_river_domain_summary
from final_reporting import write_authoritative_cache_receipt as _fr_write_authoritative_cache_receipt
from final_reporting import write_comparison_package as _fr_write_comparison_package
from final_reporting import write_comparison_summary as _fr_write_comparison_summary
from final_reporting import write_explicit_final_outputs_manifest as _fr_write_explicit_final_outputs_manifest
from final_reporting import write_final_dem_selection_receipt as _fr_write_final_dem_selection_receipt
from final_reporting import write_river_stability_summary as _fr_write_river_stability_summary
from final_reporting import write_validation_invariance_summary as _fr_write_validation_invariance_summary
from benchmark_workflow_stage import run_workflow_benchmark as _run_workflow_benchmark
from final_postrun_contract import build_final_postrun_context
from postrun_benchmark_stage import run_postrun_benchmark_stage
from postrun_regression_stage import run_postrun_regression_stage
from provenance_reporting import write_support_provenance_summary as _pr_write_support_provenance_summary
from method_guidance_provenance import apply_parallel_method_guidance_summary
from final_support_audit import write_final_support_regime_audit as _fsra_write_final_support_regime_audit
from io_artifacts import build_io_manifest as _io_build_io_manifest
from io_artifacts import write_guidance_manifest as _io_write_guidance_manifest
from io_artifacts import write_io_manifest as _io_write_io_manifest
from traceability_manifest import write_traceability_manifest as _write_traceability_manifest
from workflow_actual_trace import write_workflow_actual_trace as _write_workflow_actual_trace
from io_artifacts import emit_artifacts_from_report as _io_emit_artifacts_from_report



def _normalize_multi_path_value(value: Any) -> List[str]:
    """Normalize a multi-path config value into a clean list of file paths.

    Accepts comma-separated strings, lists/tuples/sets, Path-like objects, and
    tolerates accidental Python-list string reprs like "['/tmp/a.csv']".
    """
    parts: List[str] = []
    if value is None:
        return parts

    def _append_one(item: Any) -> None:
        if item is None:
            return
        s = str(item).strip()
        if not s:
            return
        for piece in s.split(","):
            piece = piece.strip()
            if not piece:
                continue
            piece = piece.strip().strip("[]").strip("\"'").strip()
            if piece:
                parts.append(piece)

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_one(item)
    else:
        _append_one(value)

    return parts


def _raster_contract_stats(path: Path, *, min_valid: float | None = None, max_valid: float | None = None) -> Dict[str, Any]:
    """Collect a compact, deterministic raster spec/summary for receipts and contract checks."""
    import rasterio
    path = Path(path)
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nd = ds.nodata
        invalid = nodata_mask(arr, nd)
        finite = np.isfinite(arr) & (~invalid)
        if min_valid is not None:
            finite &= arr >= float(min_valid)
        if max_valid is not None:
            finite &= arr <= float(max_valid)
        stats: Dict[str, Any] = {
            'path': str(path),
            'epsg': ds.crs.to_epsg() if ds.crs else None,
            'crs': str(ds.crs) if ds.crs else None,
            'nodata': None if nd is None else float(nd),
            'dtype': str(ds.dtypes[0]),
            'width': int(ds.width),
            'height': int(ds.height),
            'finite_cells': int(np.count_nonzero(finite)),
            'invalid_cells': int(np.count_nonzero(~finite)),
        }
        if np.count_nonzero(finite):
            vals = arr[finite].astype(float)
            stats.update({
                'min': float(np.nanmin(vals)),
                'max': float(np.nanmax(vals)),
                'p01': float(np.nanpercentile(vals, 1.0)),
                'p99': float(np.nanpercentile(vals, 99.0)),
            })
        else:
            stats.update({'min': None, 'max': None, 'p01': None, 'p99': None})
    return stats


def _write_river_dem_receipt(
    *,
    receipt_path: Path,
    source_class: str,
    cache_key: str,
    projected_raster: Path,
    authoritative_base: Optional[Path] = None,
    fallback_raster: Optional[Path] = None,
    source_error: Optional[str] = None,
) -> Dict[str, Any]:
    receipt: Dict[str, Any] = {
        'source_class': source_class,
        'cache_key': cache_key,
        'authoritative_base': str(authoritative_base) if authoritative_base else None,
        'fallback_raster': str(fallback_raster) if fallback_raster else None,
        'source_error': source_error,
    }
    if authoritative_base and Path(authoritative_base).exists():
        receipt['authoritative_base_stats'] = _raster_contract_stats(Path(authoritative_base), min_valid=-1000.0, max_valid=1000.0)
    if fallback_raster and Path(fallback_raster).exists():
        receipt['fallback_raster_stats'] = _raster_contract_stats(Path(fallback_raster), min_valid=-1000.0, max_valid=1000.0)
    receipt['projected_river_dem_stats'] = _raster_contract_stats(Path(projected_raster), min_valid=-1000.0, max_valid=1000.0)
    write_json(Path(receipt_path), receipt)
    return receipt


def _load_river_authoritative_support_points(cfg: "BathyConfig"):
    """Load river guidance support points with explicit semantics.

    River authoritative support exported from the authoritative raster represents
    absolute elevation anchors (NAVD88 in this workflow), not negative-down
    bathymetric depths. Prefer direct CSV handling for those exports so values
    are preserved exactly; fall back to the generic XYZ loader only when the
    export cannot be read directly.
    """
    try:
        import pandas as pd
        from pyproj import Transformer
        from support_points import load_extra_xyz_points
    except ImportError:
        return None

    files = []
    used_river_soundings = False
    used_authoritative_river_support = False
    try:
        rs = getattr(cfg, "river_soundings", None)
        if rs:
            files.extend(_normalize_multi_path_value(rs))
            used_river_soundings = bool(files)
        if not files:
            ars = getattr(cfg, "river_authoritative_soundings", None)
            if ars:
                files.extend(_normalize_multi_path_value(ars))
                used_authoritative_river_support = bool(files)
        if not files:
            xyz = getattr(cfg, "extra_xyz_files", None)
            if xyz:
                files.extend(_normalize_multi_path_value(xyz))
    except (AttributeError, TypeError, ValueError):
        return None
    if not files:
        return None

    def _direct_load_authoritative_support(csv_files):
        rows = []
        src_crs = str(
            getattr(cfg, "working_srs", None)
            or getattr(cfg, "river_soundings_crs", None)
            or getattr(cfg, "extra_xyz_crs", None)
            or "EPSG:32619"
        )
        tx = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        for f in csv_files:
            path = Path(f)
            if not path.exists() or path.suffix.lower() not in {".csv", ".txt"}:
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                log.debug("[RIVER][GUIDANCE] Failed reading authoritative river support CSV %s", path, exc_info=True)
                continue
            if df.empty:
                continue
            cols = {str(c).lower(): c for c in df.columns}
            xcol = cols.get("x") or cols.get("easting") or cols.get("lon") or cols.get("longitude")
            ycol = cols.get("y") or cols.get("northing") or cols.get("lat") or cols.get("latitude")
            zcol = cols.get("depth_m") or cols.get("z") or cols.get("elevation_m") or cols.get("elev_m")
            if not (xcol and ycol and zcol):
                continue
            x = pd.to_numeric(df[xcol], errors="coerce")
            y = pd.to_numeric(df[ycol], errors="coerce")
            z = pd.to_numeric(df[zcol], errors="coerce")
            keep = x.notna() & y.notna() & z.notna()
            if not keep.any():
                continue
            lon, lat = tx.transform(x[keep].to_numpy(dtype=float), y[keep].to_numpy(dtype=float))
            source_vals = df.loc[keep, "source"].astype(str).to_numpy() if "source" in df.columns else np.full(int(keep.sum()), str(path.name), dtype=object)
            block = pd.DataFrame({
                "lon": lon,
                "lat": lat,
                "depth_m": z[keep].to_numpy(dtype=float),
                "source": source_vals,
                "value_semantics": "absolute_elevation",
            })
            for extra in (
                "authoritative_role",
                "role_confidence",
                "distance_to_bank_m",
                "component_half_width_est_m",
                "normalized_channel_position",
                "inside_channel_mask",
                "inside_river_guidance_domain",
                "inside_mainstem_mask",
                "inside_estuary_clip",
            ):
                if extra in df.columns:
                    vals = df.loc[keep, extra].to_numpy()
                    if extra in {"role_confidence", "distance_to_bank_m", "component_half_width_est_m", "normalized_channel_position"}:
                        block[extra] = pd.to_numeric(vals, errors="coerce")
                    elif extra in {"inside_channel_mask", "inside_river_guidance_domain", "inside_mainstem_mask", "inside_estuary_clip"}:
                        coerced = pd.to_numeric(pd.Series(vals), errors="coerce").fillna(0).astype(int).to_numpy()
                        block[extra] = coerced
                    else:
                        block[extra] = vals
            rows.append(block)
        if not rows:
            return None
        return pd.concat(rows, ignore_index=True).reset_index(drop=True)

    # Gold-standard handling for authoritative-base-derived river support:
    # treat exported values as absolute elevation anchors in the working CRS,
    # transform coordinates only, and preserve their value semantics.
    if used_authoritative_river_support and not used_river_soundings:
        direct = _direct_load_authoritative_support(files)
        if direct is not None and not getattr(direct, "empty", True):
            try:
                src_counts = direct["source"].astype(str).value_counts().to_dict() if "source" in direct.columns else {}
                log.info("[RIVER][GUIDANCE] Loaded %d authoritative support points for river guidance (sources=%s)", len(direct), src_counts)
            except (OSError, RuntimeError, ValueError) as exc:
                log.debug("Output contract step ignored: %s", exc, exc_info=True)
            return direct

    if used_river_soundings or used_authoritative_river_support:
        crs = str(
            getattr(cfg, "river_soundings_crs", None)
            or getattr(cfg, "working_srs", None)
            or getattr(cfg, "extra_xyz_crs", None)
            or "EPSG:4326"
        )
    else:
        crs = str(
            getattr(cfg, "extra_xyz_crs", None)
            or getattr(cfg, "working_srs", None)
            or "EPSG:4326"
        )
    try:
        df = load_extra_xyz_points(files, crs=crs, aoi_str=str(cfg.aoi))
    except (OSError, ValueError, RuntimeError):
        return None
    if df is None or getattr(df, "empty", True):
        return None
    rename_map = {}
    if "longitude" in df.columns and "lon" not in df.columns:
        rename_map["longitude"] = "lon"
    if "latitude" in df.columns and "lat" not in df.columns:
        rename_map["latitude"] = "lat"
    if rename_map:
        df = df.rename(columns=rename_map)
    keep = [c for c in ("lon", "lat", "depth_m", "source", "authoritative_role", "role_confidence", "distance_to_bank_m", "component_half_width_est_m", "normalized_channel_position", "inside_channel_mask", "inside_river_guidance_domain", "inside_mainstem_mask", "inside_estuary_clip") if c in df.columns]
    if not {"lon", "lat", "depth_m"}.issubset(set(keep)):
        return None
    out = df[keep].copy()
    if "depth_m" in out.columns:
        out["depth_m"] = pd.to_numeric(out["depth_m"], errors="coerce")
        out = out[out["depth_m"].notna()]
    if used_authoritative_river_support and not used_river_soundings:
        out["value_semantics"] = "absolute_elevation"
    out = out.reset_index(drop=True)
    try:
        src_counts = out["source"].astype(str).value_counts().to_dict() if "source" in out.columns else {}
        log.info("[RIVER][GUIDANCE] Loaded %d authoritative support points for river guidance (sources=%s)", len(out), src_counts)
    except (OSError, RuntimeError, ValueError) as exc:
        log.debug("Output contract step ignored: %s", exc, exc_info=True)
    return out




def _apply_authoritative_bed_to_centerline_points(
    centerline_points,
    support_points,
    *,
    target_crs,
    max_distance_m: float,
    logger: Optional[logging.Logger] = None,
):
    """Prefer authoritative in-channel bed support for centerline z where nearby.

    The centerline guidance already prefers sampling the authoritative baseline DEM,
    but that DEM can still include interpolated values. This helper upgrades
    centerline point elevations to nearest authoritative in-channel bed support
    values when those support points are available nearby in plan view.
    """
    import pandas as pd
    import numpy as np

    if centerline_points is None or getattr(centerline_points, 'empty', True):
        return centerline_points, {
            'eligible_support_points': 0,
            'matched_centerline_points': 0,
            'max_distance_m': float(max_distance_m),
            'status': 'centerline_points_missing',
        }
    if support_points is None or getattr(support_points, 'empty', True):
        out = centerline_points.copy()
        out['centerline_authoritative_support_applied'] = False
        out['centerline_authoritative_support_distance_m'] = np.nan
        if 'centerline_sample_source' not in out.columns:
            out['centerline_sample_source'] = np.where(np.isfinite(pd.to_numeric(out.get('centerline_z_m'), errors='coerce')), 'authoritative_baseline_dem', 'missing')
        return out, {
            'eligible_support_points': 0,
            'matched_centerline_points': 0,
            'max_distance_m': float(max_distance_m),
            'status': 'support_points_missing',
        }

    out = centerline_points.copy()
    out['centerline_authoritative_support_applied'] = False
    out['centerline_authoritative_support_distance_m'] = np.nan
    if 'centerline_sample_source' not in out.columns:
        out['centerline_sample_source'] = np.where(np.isfinite(pd.to_numeric(out.get('centerline_z_m'), errors='coerce')), 'authoritative_baseline_dem', 'missing')

    try:
        from pyproj import Transformer
        from scipy.spatial import cKDTree
    except ImportError:
        if logger is not None:
            logger.debug('[RIVER][GUIDANCE] centerline authoritative-support upgrade unavailable (missing dependency).', exc_info=True)
        return out, {
            'eligible_support_points': 0,
            'matched_centerline_points': 0,
            'max_distance_m': float(max_distance_m),
            'status': 'dependency_missing',
        }

    support = support_points.copy()
    if not {'lon', 'lat', 'depth_m'}.issubset(set(support.columns)):
        return out, {
            'eligible_support_points': 0,
            'matched_centerline_points': 0,
            'max_distance_m': float(max_distance_m),
            'status': 'support_schema_missing',
        }

    semantics = support['value_semantics'].astype(str) if 'value_semantics' in support.columns else pd.Series([''] * len(support), index=support.index, dtype=object)
    z = pd.to_numeric(support['depth_m'], errors='coerce')
    keep = np.isfinite(pd.to_numeric(support['lon'], errors='coerce')) & np.isfinite(pd.to_numeric(support['lat'], errors='coerce')) & np.isfinite(z)
    if 'value_semantics' in support.columns:
        keep &= semantics.eq('absolute_elevation')
    if 'authoritative_role' in support.columns:
        roles = support['authoritative_role'].astype(str)
        keep &= roles.isin({'authoritative_bed_inner', 'authoritative_bed_core'})
    if 'inside_channel_mask' in support.columns:
        keep &= pd.to_numeric(support['inside_channel_mask'], errors='coerce').fillna(0.0) > 0.5
    if 'inside_river_guidance_domain' in support.columns:
        keep &= pd.to_numeric(support['inside_river_guidance_domain'], errors='coerce').fillna(0.0) > 0.5
    if 'inside_estuary_clip' in support.columns:
        keep &= pd.to_numeric(support['inside_estuary_clip'], errors='coerce').fillna(0.0) < 0.5
    support = support.loc[np.asarray(keep, dtype=bool)].copy()
    if support.empty:
        return out, {
            'eligible_support_points': 0,
            'matched_centerline_points': 0,
            'max_distance_m': float(max_distance_m),
            'status': 'no_eligible_support_points',
        }

    tx = Transformer.from_crs('EPSG:4326', target_crs, always_xy=True)
    xs, ys = tx.transform(
        pd.to_numeric(support['lon'], errors='coerce').to_numpy(dtype=float),
        pd.to_numeric(support['lat'], errors='coerce').to_numpy(dtype=float),
    )
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    keep_xy = np.isfinite(xs) & np.isfinite(ys)
    support = support.loc[keep_xy].copy()
    if support.empty:
        return out, {
            'eligible_support_points': 0,
            'matched_centerline_points': 0,
            'max_distance_m': float(max_distance_m),
            'status': 'projected_support_points_empty',
        }
    support['_x'] = xs[keep_xy]
    support['_y'] = ys[keep_xy]

    query_xy = np.column_stack([out.geometry.x.to_numpy(dtype=float), out.geometry.y.to_numpy(dtype=float)])
    tree = cKDTree(support[['_x', '_y']].to_numpy(dtype=float))
    dist, idx = tree.query(query_xy, k=1, distance_upper_bound=float(max_distance_m))
    idx = np.asarray(idx, dtype=int)
    dist = np.asarray(dist, dtype=float)
    matched = np.isfinite(dist) & (idx >= 0) & (idx < len(support))
    if np.any(matched):
        nearest = support.iloc[idx[matched]].reset_index(drop=True)
        out.loc[matched, 'centerline_z_m'] = pd.to_numeric(nearest['depth_m'], errors='coerce').to_numpy(dtype=float)
        out.loc[matched, 'centerline_authoritative_support_applied'] = True
        out.loc[matched, 'centerline_authoritative_support_distance_m'] = dist[matched]
        out.loc[matched, 'centerline_sample_source'] = 'authoritative_support_point'
        if 'authoritative_role' in nearest.columns:
            out.loc[matched, 'centerline_authoritative_support_role'] = nearest['authoritative_role'].astype(str).to_numpy()
        if 'role_confidence' in nearest.columns:
            out.loc[matched, 'centerline_authoritative_support_confidence'] = pd.to_numeric(nearest['role_confidence'], errors='coerce').to_numpy(dtype=float)
    return out, {
        'eligible_support_points': int(len(support)),
        'matched_centerline_points': int(np.count_nonzero(matched)),
        'max_distance_m': float(max_distance_m),
        'status': 'ok',
    }


def _recover_river_depth_from_support(
    *,
    cfg: "BathyConfig",
    depth_tif: Path,
    bed_tif: Path,
    channel_mask_tif: Optional[Path],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Best-effort recovery for degenerate river depth rasters.

    Builds a channel-constrained negative-down depth surface using the existing
    bed/DEM relationship where finite, then reinforces it with authoritative
    river support points. When enough support exists, the remaining channel gap
    is filled by nearest support inside the channel only.
    """
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.features import rasterize
    from river_bank_guidance import (
        compute_bank_distance_influence,
        compute_xs_bank_guidance_surfaces,
        compute_graph_informed_bank_context_surfaces,
        summarize_bank_qc,
    )
    from river_structured_scaffold import (
        select_retained_river_features,
        build_dense_bank_points,
        build_centerline_points,
        build_xs_support_points,
        build_centerline_core_influence,
        rasterize_point_seed_surface,
        rasterize_point_chainage_surface,
        _nearest_surface_from_points,
        _clip_gdf_to_allowed_mask,
        normalize_structured_flow_component_ids,
        validate_centerline_station_component_contract,
    )
    from river_masking import apply_estuary_first_channel_domain

    result = {"recovered": False, "reason": None, "valid_pixels": 0, "support_seed_pixels": 0}
    try:
        from pyproj import Transformer
    except ImportError:
        result["reason"] = "pyproj_unavailable"
        return result

    support_points = _load_river_authoritative_support_points(cfg)
    has_support_points = support_points is not None and not getattr(support_points, "empty", True)

    try:
        import scipy.ndimage as ndi
    except ImportError:
        ndi = None

    try:
        with rasterio.open(depth_tif) as depth_ds, rasterio.open(bed_tif) as bed_ds, rasterio.open(Path(cfg.river_dem)) as dem_ds:
            if (depth_ds.width, depth_ds.height) != (bed_ds.width, bed_ds.height) or depth_ds.transform != bed_ds.transform:
                result["reason"] = "bed_depth_grid_mismatch"
                return result
            nodata = float(depth_ds.nodata if depth_ds.nodata is not None else getattr(cfg, "river_nodata", -9999.0))
            depth = sanitize_array(depth_ds.read(1), depth_ds.nodata, dtype="float32")
            bed = sanitize_array(bed_ds.read(1), bed_ds.nodata, dtype="float32")
            dem = sanitize_array(dem_ds.read(1, out_shape=(depth_ds.height, depth_ds.width), resampling=rasterio.enums.Resampling.bilinear), dem_ds.nodata, dtype="float32")
            ch = np.ones_like(depth, dtype=bool)
            if channel_mask_tif is not None and Path(channel_mask_tif).exists():
                with rasterio.open(channel_mask_tif) as cm_ds:
                    cm = cm_ds.read(1, out_shape=(depth_ds.height, depth_ds.width), resampling=rasterio.enums.Resampling.nearest)
                    ch = cm > 0

            base = np.full(depth.shape, np.nan, dtype="float32")
            base[ch & np.isfinite(bed) & np.isfinite(dem)] = bed[ch & np.isfinite(bed) & np.isfinite(dem)] - dem[ch & np.isfinite(bed) & np.isfinite(dem)]
            base_valid = ch & np.isfinite(base)
            if int(np.count_nonzero(base_valid)) > 0:
                try:
                    base_p01 = float(np.nanpercentile(base[base_valid], 1.0))
                    base_p99 = float(np.nanpercentile(base[base_valid], 99.0))
                    base_span = float(base_p99 - base_p01)
                    base_max = float(np.nanmax(base[base_valid]))
                    base_frac_neg = float(np.mean(base[base_valid] < 0.0))
                except (TypeError, ValueError, RuntimeError):
                    base_span = np.nan
                    base_max = np.nan
                    base_frac_neg = np.nan
                if (
                    (np.isfinite(base_span) and base_span < 0.05)
                    or (
                        np.isfinite(base_frac_neg)
                        and np.isfinite(base_max)
                        and base_frac_neg < 0.01
                        and base_max <= 1.0
                    )
                ):
                    logger.warning(
                        "[RIVER] Depth base derived from bed-dem is degenerate inside channel "
                        "(p01=%.3f p99=%.3f frac_neg=%.3f max=%.3f); ignoring it and rebuilding "
                        "depth from authoritative support.",
                        float(base_p01) if np.isfinite(base_p01) else float('nan'),
                        float(base_p99) if np.isfinite(base_p99) else float('nan'),
                        float(base_frac_neg) if np.isfinite(base_frac_neg) else float('nan'),
                        float(base_max) if np.isfinite(base_max) else float('nan'),
                    )
                    base[:] = np.nan
    
            pt = support_points.copy() if has_support_points else None
            if pt is not None and "depth_m" not in pt.columns:
                pt = None
            xcol, ycol = "lon", "lat"
            transformer = None
            try:
                if depth_ds.crs is not None and str(depth_ds.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                    transformer = Transformer.from_crs("EPSG:4326", depth_ds.crs, always_xy=True)
            except (OSError, RuntimeError, ValueError, TypeError):
                log.debug("bathy_main: suppressed exception", exc_info=True)
                transformer = None
            coords = []
            if pt is not None:
                for r in pt.itertuples():
                    try:
                        x = float(getattr(r, xcol))
                        y = float(getattr(r, ycol))
                        z = float(getattr(r, "depth_m"))
                        semantics = str(getattr(r, "value_semantics", "depth") or "depth").strip().lower()
                    except (TypeError, ValueError, AttributeError):
                        continue
                    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                        continue
                    if transformer is not None:
                        x, y = transformer.transform(x, y)
                    try:
                        row, col = depth_ds.index(x, y)
                    except (TypeError, ValueError, RuntimeError):
                        continue
                    if row < 0 or col < 0 or row >= depth_ds.height or col >= depth_ds.width:
                        continue
                    if semantics == "absolute_elevation":
                        dem_here = dem[row, col]
                        if not np.isfinite(dem_here):
                            continue
                        z = z - float(dem_here)
                    if z > 0:
                        z = -abs(z)
                    coords.append(({"type": "Point", "coordinates": (x, y)}, float(z)))
            seeds = rasterize(coords, out_shape=depth.shape, transform=depth_ds.transform, fill=np.nan, dtype="float32") if coords else np.full(depth.shape, np.nan, dtype="float32")
            seeds[~ch] = np.nan
            result["support_seed_pixels"] = int(np.count_nonzero(np.isfinite(seeds)))

            bank_proxy = np.full(depth.shape, np.nan, dtype="float32")
            if ndi is not None and np.any(ch):
                edge = ch & (~ndi.binary_erosion(ch, structure=np.ones((3, 3), dtype=bool), border_value=0))
                exterior_edge = (~ch) & ndi.binary_dilation(ch, structure=np.ones((3, 3), dtype=bool), iterations=1)
                bank_seed = np.full(depth.shape, np.nan, dtype="float32")
                bank_seed[exterior_edge & np.isfinite(dem)] = dem[exterior_edge & np.isfinite(dem)]
                if not np.any(np.isfinite(bank_seed)):
                    bank_seed[edge & np.isfinite(dem)] = dem[edge & np.isfinite(dem)]
                if np.any(np.isfinite(bank_seed)) and np.any(np.isfinite(bed) & ch):
                    valid_bank = np.isfinite(bank_seed)
                    bank_idx = ndi.distance_transform_edt(~valid_bank, return_distances=False, return_indices=True)
                    bank_surface = bank_seed[tuple(bank_idx)]
                    bank_proxy[ch & np.isfinite(bed) & np.isfinite(bank_surface)] = bed[ch & np.isfinite(bed) & np.isfinite(bank_surface)] - bank_surface[ch & np.isfinite(bed) & np.isfinite(bank_surface)]
                    bank_proxy[np.isfinite(bank_proxy) & (bank_proxy > 0)] = -np.abs(bank_proxy[np.isfinite(bank_proxy) & (bank_proxy > 0)])

            recovered = np.where(np.isfinite(base), base, seeds).astype("float32")
            if not np.any(np.isfinite(recovered) & ch) or (np.isfinite(recovered[ch]).any() and np.nanpercentile(recovered[ch & np.isfinite(recovered)], 99.0) - np.nanpercentile(recovered[ch & np.isfinite(recovered)], 1.0) < 0.05):
                recovered = np.where(np.isfinite(seeds), seeds, bank_proxy).astype("float32")
                result["reason"] = "bank_proxy_depth" if np.any(np.isfinite(bank_proxy)) else result.get("reason")
            else:
                recovered = np.where(np.isfinite(recovered), recovered, bank_proxy).astype("float32")
            recovered[~ch] = np.nan
            if ndi is not None and np.any(np.isfinite(recovered) & ch):
                valid = np.isfinite(recovered) & ch
                idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
                recovered = recovered[tuple(idx)]
                recovered[~ch] = np.nan
            recovered[np.isfinite(recovered) & (recovered > 0)] = -np.abs(recovered[np.isfinite(recovered) & (recovered > 0)])
            valid = ch & np.isfinite(recovered)
            if int(np.count_nonzero(valid)) <= 0:
                result["reason"] = "recovered_empty"
                return result
            span = float(np.nanpercentile(recovered[valid], 99.0) - np.nanpercentile(recovered[valid], 1.0)) if np.count_nonzero(valid) > 1 else 0.0
            if span < 0.05:
                result["reason"] = "recovered_near_constant"
                return result
    
            profile = depth_ds.profile.copy()
            profile.update(dtype="float32", nodata=nodata, compress="DEFLATE", predictor=2)
            out = np.where(valid, recovered, nodata).astype("float32")
            with rasterio.open(depth_tif, "w", **profile) as dst:
                dst.write(out, 1)
            result.update({"recovered": True, "valid_pixels": int(np.count_nonzero(valid)), "reason": "support_constrained_depth"})
            logger.warning("[RIVER] Recovered invalid river depth raster using channel-constrained recovery (valid=%d, seed_pixels=%d).", result["valid_pixels"], result["support_seed_pixels"])
            return result
    except Exception as exc:
        logger.debug("[RIVER] Support-based depth recovery failed: %s", exc, exc_info=True)
        result["reason"] = f"recovery_exception:{exc}"
        return result


def _guess_first_existing_field(columns, candidates):
    """Return the first candidate field present in ``columns``.

    Matching is case-sensitive first, then falls back to case-insensitive
    comparison so we can tolerate common NHD/source capitalization variants.
    """
    cols = list(columns)
    colset = set(cols)
    for cand in candidates:
        if cand in colset:
            return cand
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        got = lower_map.get(str(cand).lower())
        if got is not None:
            return str(got)
    return None


def _clip_channel_mask_for_estuary(
    channel_mask_tif: "Path",
    cfg: "BathyConfig",
    *,
    ocean_mask_path: "Optional[Path]",
    report: "Dict[str, Any]",
) -> "Tuple[int, Optional[Path]]":
    return _rm_clip_channel_mask_for_estuary(channel_mask_tif, cfg, ocean_mask_path=ocean_mask_path, report=report, logger=log)

def _build_hydraulic_estuary_hint_mask(*, cfg: "BathyConfig", channel, transform, crs, px_size_m: float):
    return _rm_build_hydraulic_estuary_hint_mask(cfg=cfg, channel=channel, transform=transform, crs=crs, px_size_m=px_size_m, logger=log)

def _write_river_guidance_artifacts(
    cfg: "BathyConfig",
    *,
    bed_tif: "Path",
    depth_tif: "Path",
    channel_mask_tif: "Optional[Path]",
    river_dir: "Path",
    report: "Dict[str, Any]",
) -> "Dict[str, Optional[str]]":
    """Export guidance-first river artifacts.

    The inferred river bed remains available as an internal helper surface, but the main exported
    intelligence is a set of guidance rasters/points describing where the product should influence
    downstream interpolation.

    Also produces an estuary_transition mask where the river channel domain
    overlaps or borders the ocean domain — a reduced-trust handoff zone where
    neither SDB nor river should dominate.
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from river_bank_guidance import compute_bank_distance_influence, compute_xs_bank_guidance_surfaces, compute_graph_informed_bank_context_surfaces
    from river_structured_scaffold import (
        select_retained_river_features,
        build_dense_bank_points,
        build_centerline_points,
        build_xs_support_points,
        build_centerline_core_influence,
        rasterize_point_seed_surface,
        rasterize_point_chainage_surface,
        _nearest_surface_from_points,
        _clip_gdf_to_allowed_mask,
        normalize_structured_flow_component_ids,
        validate_centerline_station_component_contract,
    )
    from river_masking import apply_estuary_first_channel_domain

    runtime_ctx = _river_guidance_runtime_context(Path(river_dir))
    work_dir = runtime_ctx["work_dir"]
    channel_template_dir = runtime_ctx["channel_template_dir"]

    network_gpkg_candidates: list[Path] = []
    _append_unique_path(network_gpkg_candidates, work_dir / "river_network.gpkg")
    derived_cache_root = getattr(cfg, "derived_cache_root", None)
    if derived_cache_root:
        _append_unique_path(network_gpkg_candidates, Path(derived_cache_root) / "river" / "work" / "river_network.gpkg")
    try:
        river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
        _append_unique_path(network_gpkg_candidates, Path(river_outputs.get("network_gpkg")) if river_outputs.get("network_gpkg") else None)
        _append_unique_path(network_gpkg_candidates, Path(river_outputs.get("network")) if river_outputs.get("network") else None)
    except (TypeError, ValueError, OSError):
        log.debug("[RIVER][GUIDANCE] Failed to read network path candidates from report.", exc_info=True)

    def _resolve_structured_network_gpkg() -> Path:
        existing = next((cand for cand in network_gpkg_candidates if cand.exists()), None)
        return existing if existing is not None else network_gpkg_candidates[0]

    artifacts = {
        "guidance_weight": None,
        "trusted_interior": None,
        "admissibility": None,
        "guide_points": None,
        "authoritative_support": None,
        "authoritative_support_depth": None,
        "corridor_mask": None,
        "bank_edge_mask": None,
        "bank_distance": None,
        "bank_influence": None,
        "bank_elevation_xs": None,
        "bank_pair_weight": None,
        "bank_points": None,
        "centerline_points": None,
        "xs_support_points": None,
        "centerline_elevation": None,
        "centerline_influence": None,
        "centerline_stationing": None,
        "xs_support_elevation": None,
        "xs_support_weight": None,
        "retained_network": None,
        "scaffold_domains": None,
        "river_dem_projected": None,
        "river_dem_valid_mask": None,
        "bank_elevation_sample_mask": None,
        "centerline_elevation_sample_mask": None,
        "longitudinal_profile_source": None,
        "dem_stage_receipt": None,
    }
    with rasterio.open(depth_tif) as ds:
        depth = sanitize_array(ds.read(1), ds.nodata if ds.nodata is not None else float(getattr(cfg, "river_nodata", -9999.0)), dtype="float32")
        nodata = ds.nodata if ds.nodata is not None else float(getattr(cfg, "river_nodata", -9999.0))
        valid = np.isfinite(depth)
        corridor = np.ones((ds.height, ds.width), dtype=bool)
        if channel_mask_tif is not None and Path(channel_mask_tif).exists():
            with rasterio.open(channel_mask_tif) as ms:
                cm = np.zeros((ds.height, ds.width), dtype="uint8")
                from rasterio.warp import reproject, Resampling
                reproject(
                    source=rasterio.band(ms, 1), destination=cm,
                    src_transform=ms.transform, src_crs=ms.crs,
                    dst_transform=ds.transform, dst_crs=ds.crs,
                    resampling=Resampling.nearest,
                    src_nodata=ms.nodata, dst_nodata=0,
                )
                corridor = (cm > 0)
        else:
            corridor = valid.copy()
        channel = corridor.copy()
        support_points = _load_river_authoritative_support_points(cfg)
        support = np.zeros((ds.height, ds.width), dtype="uint8")
        support_depth = np.full((ds.height, ds.width), np.nan, dtype="float32")
        support_used = 0
        support_depth_used = 0
        support_receipt = {
            "input_points": 0,
            "projected_points": 0,
            "support_pixels": 0,
            "support_depth_pixels": 0,
            "value_semantics": {},
            "raster_crs": str(ds.crs) if ds.crs else None,
            "used_depth_semantics": False,
        }
        river_dem_projected = np.full(depth.shape, np.nan, dtype="float32")
        river_dem_valid_mask = np.zeros(depth.shape, dtype="uint8")
        bank_elevation_sample_mask = np.zeros(depth.shape, dtype="uint8")
        centerline_elevation_sample_mask = np.zeros(depth.shape, dtype="uint8")
        longitudinal_profile_source = np.zeros(depth.shape, dtype="uint8")
        dem_stage_receipt = {
            "river_dem": str(getattr(cfg, "river_dem", "") or ""),
            "river_dem_valid_pixels": 0,
            "bank_sample_pixels": 0,
            "centerline_sample_pixels": 0,
            "xs_support_sample_pixels": 0,
            "longitudinal_profile_source_pixels": {},
        }
        try:
            with rasterio.open(Path(cfg.river_dem)) as dem_src:
                if (dem_src.height, dem_src.width) == (ds.height, ds.width) and dem_src.transform == ds.transform and dem_src.crs == ds.crs:
                    river_dem_projected = sanitize_array(dem_src.read(1), dem_src.nodata, dtype="float32")
                else:
                    from rasterio.warp import reproject, Resampling
                    reproject(
                        source=rasterio.band(dem_src, 1), destination=river_dem_projected,
                        src_transform=dem_src.transform, src_crs=dem_src.crs,
                        dst_transform=ds.transform, dst_crs=ds.crs,
                        resampling=Resampling.bilinear,
                        src_nodata=dem_src.nodata, dst_nodata=float(nodata),
                    )
                    river_dem_projected = sanitize_array(river_dem_projected, float(nodata), dtype="float32")
                river_dem_valid_mask = np.isfinite(river_dem_projected).astype("uint8")
                dem_stage_receipt["river_dem_valid_pixels"] = int(river_dem_valid_mask.sum())
        except (OSError, RuntimeError, ValueError, TypeError):
            log.debug("[RIVER][GUIDANCE] Failed to build projected river DEM diagnostic raster.", exc_info=True)
        px_x = abs(float(ds.transform.a))
        px_y = abs(float(ds.transform.e))
        px_m = max(min(px_x, px_y), 1e-6)
        edge_buffer_px = max(int(round(float(getattr(cfg, "river_trusted_halo_m", 60.0) or 0.0) / px_m)), 0)
        if support_points is not None and not support_points.empty and {"lon", "lat"}.issubset(set(support_points.columns)):
            from pyproj import Transformer
            pts = support_points.copy()
            pts = pts[np.isfinite(pts["lon"]) & np.isfinite(pts["lat"])]
            support_receipt["input_points"] = int(len(pts))
            if not pts.empty:
                if "value_semantics" in pts.columns:
                    support_receipt["value_semantics"] = {str(k): int(v) for k, v in pts["value_semantics"].astype(str).value_counts().to_dict().items()}
                tx = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                xs, ys = tx.transform(pts["lon"].to_numpy(dtype=float), pts["lat"].to_numpy(dtype=float))
                xs = np.asarray(xs, dtype=float)
                ys = np.asarray(ys, dtype=float)
                keep_xy = np.isfinite(xs) & np.isfinite(ys)
                pts = pts.loc[keep_xy].copy()
                pts["_x"] = xs[keep_xy]
                pts["_y"] = ys[keep_xy]
                support_receipt["projected_points"] = int(len(pts))
                xy = pts[["_x", "_y"]].to_numpy(dtype=float, copy=False)
                keep_shape = np.isfinite(xy).all(axis=1)
                shapes = [
                    ({"type": "Point", "coordinates": (float(x), float(y))}, 1)
                    for x, y in xy[keep_shape]
                ]
                if shapes:
                    support = rasterize(shapes, out_shape=(ds.height, ds.width), transform=ds.transform, fill=0, dtype="uint8")
                    support = ((support > 0) & corridor).astype("uint8")
                    support_used = int(support.sum())
                    support_receipt["support_pixels"] = support_used
                depth_pts = pts.iloc[0:0].copy()
                if "depth_m" in pts.columns:
                    depth_pts = pts[np.isfinite(pd.to_numeric(pts["depth_m"], errors="coerce"))].copy()
                    if "value_semantics" in depth_pts.columns:
                        depth_pts = depth_pts[depth_pts["value_semantics"].astype(str) == "depth"]
                if not depth_pts.empty:
                    depth_xy = depth_pts[["_x", "_y", "depth_m"]].copy()
                    depth_xy["depth_m"] = pd.to_numeric(depth_xy["depth_m"], errors="coerce")
                    depth_arr = depth_xy.to_numpy(dtype=float, copy=False)
                    keep_depth_shape = np.isfinite(depth_arr).all(axis=1)
                    depth_shapes = [
                        ({"type": "Point", "coordinates": (float(x), float(y))}, float(depth_m))
                        for x, y, depth_m in depth_arr[keep_depth_shape]
                    ]
                    if depth_shapes:
                        support_depth = rasterize(
                            depth_shapes,
                            out_shape=(ds.height, ds.width),
                            transform=ds.transform,
                            fill=np.nan,
                            dtype="float32",
                        )
                        support_depth[(support <= 0) | (~corridor)] = np.nan
                        support_depth_used = int(np.count_nonzero(np.isfinite(support_depth)))
                        support_receipt["support_depth_pixels"] = support_depth_used
                        support_receipt["used_depth_semantics"] = True
                elif support_receipt["value_semantics"].get("absolute_elevation", 0) > 0:
                    log.info("[RIVER][GUIDANCE] Authoritative support points are absolute elevation anchors; support_depth raster intentionally left empty.")
                if support_used == 0:
                    log.warning("[RIVER][GUIDANCE] Authoritative support points projected to raster CRS but rasterized to zero corridor pixels; inspect river_authoritative_support_receipt.json.")
        # Start with channel-wide guidance weight and then taper it away from
        # authoritative anchor support. The trusted export / admissibility masks
        # are derived later from the shared trusted_interior contract helpers.
        guidance_weight = channel.astype("float32")
        if support_used > 0:
            try:
                from scipy.ndimage import distance_transform_edt
                ramp_m = float(min(max(3.0 * px_m, 30.0), 250.0))
                dist_px = distance_transform_edt(1 - support)
                dist_m = dist_px * px_m
                guidance_weight = np.clip(dist_m / ramp_m, 0.0, 1.0).astype("float32")
                guidance_weight *= channel.astype("float32")
                guidance_weight[support > 0] = 0.0
            except (ImportError, RuntimeError, ValueError):
                guidance_weight = channel.astype("float32")
                guidance_weight[support > 0] = 0.0

        prof_u8 = ds.profile.copy(); prof_u8.update(dtype="uint8", nodata=0, compress="deflate", count=1)
        prof_f32 = ds.profile.copy(); prof_f32.update(dtype="float32", nodata=0.0, compress="deflate", count=1)
        prof_depth = ds.profile.copy(); prof_depth.update(dtype="float32", nodata=float(nodata), compress="deflate", count=1)
        prof_diag = ds.profile.copy(); prof_diag.update(dtype="float32", nodata=float("nan"), compress="deflate", count=1)

    gw_path = river_dir / "river_guidance_weight.tif"
    ti_path = river_dir / "river_trusted_interior.tif"
    ad_path = river_dir / "river_admissibility.tif"
    sg_path = river_dir / "river_soft_guidance_domain.tif"
    sp_path = river_dir / "river_authoritative_support.tif"
    sd_path = river_dir / "river_authoritative_support_depth.tif"
    cm_path = river_dir / "river_corridor_mask.tif"
    be_path = river_dir / "river_bank_edge_mask.tif"
    bd_path = river_dir / "river_bank_distance_m.tif"
    bi_path = river_dir / "river_bank_influence.tif"
    bx_path = river_dir / "river_bank_elevation_xs.tif"
    bpw_path = river_dir / "river_bank_pair_weight.tif"
    bcw_path = river_dir / "river_bank_continuity_weight.tif"
    bgc_path = river_dir / "river_bank_graph_confidence.tif"
    bcd_path = river_dir / "river_bank_confluence_damping.tif"
    besd_path = river_dir / "river_bank_estuary_side_decay.tif"
    bpts_path = river_dir / "river_bank_points.gpkg"
    bqc_points_path = river_dir / "river_xs_bank_qc_points.gpkg"
    bqc_summary_path = river_dir / "river_xs_bank_qc_summary.csv"
    cpts_path = river_dir / "river_centerline_points.gpkg"
    cpts_contract_path = river_dir / "river_centerline_station_contract.json"
    xsp_path = river_dir / "river_xs_support_points.gpkg"
    ce_path = river_dir / "river_centerline_elevation.tif"
    ci_path = river_dir / "river_centerline_influence.tif"
    cs_path = river_dir / "river_centerline_stationing_m.tif"
    xse_path = river_dir / "river_xs_support_elevation.tif"
    xsw_path = river_dir / "river_xs_support_weight.tif"
    retained_network_path = river_dir / "river_retained_network.gpkg"
    scaffold_domains_path = river_dir / "river_scaffold_domains.json"
    dem_proj_path = river_dir / "11_river_dem_projected.tif"
    dem_valid_path = river_dir / "12_river_dem_valid_mask.tif"
    bank_sample_mask_path = river_dir / "13_bank_elevation_sample_mask.tif"
    centerline_sample_mask_path = river_dir / "14_centerline_elevation_sample_mask.tif"
    long_source_path = river_dir / "15_longitudinal_profile_source.tif"
    dem_stage_receipt_path = river_dir / "river_dem_derived_stage_receipt.json"

    # ---------------------------------------------------------------
    # Estuary transition mask
    #
    # Identifies the reduced-trust handoff zone where the river channel
    # transitions away from a clean fluvial regime.  In v10 this was a
    # purely spatial buffer from the ocean mask.  Here we keep that guard
    # but add conservative hydraulic hints from the river network:
    #
    #   - near-mouth reaches (distance-to-mouth field)
    #   - low-slope / backwater-prone reaches (Manning guard analogue)
    #
    # This keeps the transition mask tied to actual regime-change signals,
    # not just geometric proximity to the ocean boundary.
    # ---------------------------------------------------------------
    estuary_transition = np.zeros_like(channel, dtype="uint8")
    estuary_px_count = 0
    ocean_proximity_mask = np.zeros_like(channel, dtype="uint8")
    hydraulic_hint_mask = np.zeros_like(channel, dtype="uint8")
    hydraulic_hint_meta = {
        "enabled": False,
        "flagged_reaches": 0,
        "near_mouth_reaches": 0,
        "backwater_slope_reaches": 0,
        "reach_buffer_m": 0.0,
        "reason": "not_attempted",
    }
    try:
        # Prefer the retained-channel estuary transition mask written by the estuary clip stage.
        # Fall back to the clip mask only for backward compatibility with older artifacts.
        estuary_transition_ref = (
            artifacts.get("estuary_transition")
            or report.get("river", {}).get("outputs", {}).get("estuary_transition")
            or str(Path(cfg.derived_cache_root) / "river" / "work" / "estuary_transition_mask.tif")
        )
        estuary_clip_ref = (
            artifacts.get("estuary_clip_mask")
            or report.get("river", {}).get("outputs", {}).get("estuary_clip_mask")
            or str(Path(cfg.derived_cache_root) / "river" / "work" / "estuary_clip_mask.tif")
        )
        estuary_debug_receipt_ref = (
            report.get("river", {}).get("outputs", {}).get("estuary_debug_receipt")
            or str(Path(cfg.derived_cache_root) / "river" / "work" / "estuary_debug_receipt.json")
        )

        estuary_transition_path = Path(estuary_transition_ref)
        estuary_clip_path = Path(estuary_clip_ref)
        estuary_debug_receipt_path = Path(estuary_debug_receipt_ref)
        artifacts["estuary_clip_mask"] = str(estuary_clip_path)

        def _load_u8_aligned(mask_path: Path) -> np.ndarray:
            with rasterio.open(mask_path) as src, rasterio.open(depth_tif) as ds_ref:
                arr = src.read(1).astype("uint8")
                if arr.shape != (ds_ref.height, ds_ref.width):
                    from rasterio.warp import reproject, Resampling as _Rs
                    aligned = np.zeros((ds_ref.height, ds_ref.width), dtype="uint8")
                    reproject(
                        source=rasterio.band(src, 1), destination=aligned,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=ds_ref.transform, dst_crs=ds_ref.crs,
                        resampling=_Rs.nearest,
                    )
                    arr = aligned
                return arr

        if estuary_transition_path.exists():
            est_data = _load_u8_aligned(estuary_transition_path)
            estuary_transition = (est_data > 0).astype("uint8")
            estuary_px_count = int(estuary_transition.sum())
            artifacts["estuary_transition"] = str(estuary_transition_path)
            log.info(
                "[ESTUARY] Transition mask loaded from estuary_transition_mask: %d pixels",
                estuary_px_count,
            )
        elif estuary_clip_path.exists():
            ec_data = _load_u8_aligned(estuary_clip_path)
            estuary_transition = (ec_data > 0).astype("uint8")
            estuary_px_count = int(estuary_transition.sum())
            log.info(
                "[ESTUARY] Transition mask fallback loaded from estuary_clip_mask: %d pixels",
                estuary_px_count,
            )
        else:
            log.info("[ESTUARY] No estuary transition artifact found; transition mask will be empty.")

        if estuary_debug_receipt_path.exists():
            try:
                est_dbg = json.loads(estuary_debug_receipt_path.read_text(encoding="utf-8"))
                signals = est_dbg.get("signals", {}) if isinstance(est_dbg, dict) else {}
                debug_paths = est_dbg.get("paths", {}) if isinstance(est_dbg, dict) else {}
                debug_dir = Path(debug_paths.get("debug_dir", "")) if debug_paths.get("debug_dir") else None
                ocean_dbg_path = debug_dir / "near_mouth_corridor.tif" if debug_dir is not None else None
                hydraulic_dbg_path = debug_dir / "hydraulic_hint.tif" if debug_dir is not None else None
                if ocean_dbg_path is not None and ocean_dbg_path.exists():
                    ocean_proximity_mask = (_load_u8_aligned(ocean_dbg_path) > 0).astype("uint8")
                if hydraulic_dbg_path is not None and hydraulic_dbg_path.exists():
                    hydraulic_hint_mask = (_load_u8_aligned(hydraulic_dbg_path) > 0).astype("uint8")
                if isinstance(signals.get("low_slope_extension"), dict):
                    hydraulic_hint_meta = dict(signals["low_slope_extension"])
                    hydraulic_hint_meta["enabled"] = True
                log.info(
                    "[ESTUARY] Debug receipt loaded: transition=%d ocean_proximity=%d hydraulic_hint=%d",
                    int(estuary_transition.sum()), int(ocean_proximity_mask.sum()), int(hydraulic_hint_mask.sum()),
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                log.debug("[ESTUARY] Failed to load estuary debug receipt.", exc_info=True)
    except (OSError, ValueError, TypeError, rasterio.errors.RasterioError):
        log.debug("[ESTUARY] Transition mask generation failed; continuing without it.", exc_info=True)

    et_path = river_dir / "estuary_transition_mask.tif"
    with rasterio.open(et_path, "w", **prof_u8) as dst:
        dst.write(estuary_transition.astype("uint8"), 1)
    artifacts["estuary_transition"] = str(et_path)

    contract = build_river_source_contract(
        channel=channel.astype("uint8"),
        valid_depth=valid,
        guidance_weight=guidance_weight,
        authoritative_support=support.astype("uint8"),
        estuary_transition=estuary_transition.astype("uint8"),
        edge_buffer_px=edge_buffer_px,
    )
    anchor_support = contract["authoritative_anchor_support"]
    trusted_interior = contract["trusted_interior"]
    soft_guidance_domain = contract["soft_guidance_domain"]
    admissibility = contract["admissibility"]
    regime_class = contract["regime"]
    guidance_weight = contract["guidance_weight"]

    river_bank_edge, river_bank_distance_m, river_bank_influence = compute_bank_distance_influence(
        channel.astype(bool),
        pixel_size_m=px_m,
        full_influence_m=0.0,
        zero_influence_m=max(float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0) * 0.18, 60.0),
    )
    river_bank_elevation_xs = np.full(depth.shape, np.nan, dtype="float32")
    river_bank_pair_weight = np.zeros(depth.shape, dtype="float32")
    river_bank_continuity_weight = np.zeros(depth.shape, dtype="float32")
    river_bank_graph_confidence = np.zeros(depth.shape, dtype="float32")
    river_bank_confluence_damping = np.ones(depth.shape, dtype="float32")
    river_bank_estuary_side_decay = np.ones(depth.shape, dtype="float32")
    xs_bank_points_gdf = None
    xs_candidates: list[Path] = []
    if derived_cache_root:
        _append_unique_path(xs_candidates, Path(derived_cache_root) / "river" / "work" / "cross_sections_mainstem.gpkg")
    _append_unique_path(xs_candidates, work_dir / "cross_sections.gpkg")
    _append_unique_path(xs_candidates, river_dir / "river_xs_params.gpkg")

    def _ensure_structured_xs_source() -> Optional[Path]:
        existing = next((xp for xp in xs_candidates if xp.exists()), None)
        if existing is not None:
            return existing
        xs_spacing_requested = float(getattr(cfg, "xs_spacing_m", 0.0) or 0.0)
        if xs_spacing_requested <= 0.0:
            return None
        network_gpkg = _resolve_structured_network_gpkg()
        if not network_gpkg.exists():
            raise RuntimeError(
                f"Structured XS generation requested but river network is missing: {network_gpkg}"
            )
        xs_out = work_dir / "cross_sections.gpkg"
        xs_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            from xs_builder import XSConfig, build_xs_for_river, _read_layer
        except ImportError as exc:
            raise RuntimeError("Structured XS generation unavailable: failed to import xs_builder.") from exc
        try:
            rivers = _read_layer(network_gpkg, "rivers_clip")
            edges = _read_layer(network_gpkg, "graph_edges")
            bank_domain_gdf = None
            for layer_name in ("nhdarea_clip", "nhdarea_aoi"):
                try:
                    bank_domain_gdf = _read_layer(network_gpkg, layer_name)
                    if bank_domain_gdf is not None and not bank_domain_gdf.empty:
                        log.info("[RIVER][GUIDANCE] Using %s as corridor-aware bank domain for structured XS generation.", layer_name)
                        break
                except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError):
                    bank_domain_gdf = None
                    log.debug("[RIVER][GUIDANCE] Optional XS bank-domain layer %s unavailable.", layer_name, exc_info=True)
            xs_cfg = XSConfig(
                spacing_m=xs_spacing_requested,
                half_width_m=float(getattr(cfg, "xs_length_m", 300.0) or 300.0) / 2.0,
                smoothing_window_m=float(getattr(cfg, "xs_smoothing_window_m", 0.0) or 0.0),
                trim_overlaps=bool(getattr(cfg, "xs_trim_overlaps", True)),
                global_deconflict=bool(getattr(cfg, "xs_global_deconflict", True)),
                deconflict_tol_m=float(getattr(cfg, "xs_deconflict_tol_m", 2.0) or 2.0),
                skip_junctions=bool(getattr(cfg, "xs_skip_junctions", True)),
                junction_snap_m=float(getattr(cfg, "xs_junction_snap_m", 30.0) or 30.0),
                junction_buffer_m=float(getattr(cfg, "xs_junction_buffer_m", 75.0) or 75.0),
                densify_step_m=float(getattr(cfg, "xs_densify_step_m", 20.0) or 20.0),
            )
            log.info(
                "[RIVER][GUIDANCE] Generating structured-flow XS source: spacing=%.2fm half_width=%.2fm -> %s",
                float(xs_cfg.spacing_m),
                float(xs_cfg.half_width_m),
                xs_out,
            )
            build_xs_for_river(
                rivers_clip=rivers,
                edges=edges,
                cfg=xs_cfg,
                dem_path=Path(cfg.river_dem),
                topo_path=None,
                out_gpkg=xs_out,
                out_csv=None,
                enable_component_prune=True,
                keep_top_components=int(getattr(cfg, "river_scaffold_keep_top_components", 6) or 6),
                min_stream_order=int(getattr(cfg, "river_scaffold_min_stream_order", 3) or 3),
                min_length_km=float(getattr(cfg, "river_scaffold_min_length_km", 0.25) or 0.25),
                ftype_allow=[460, 558],
                include_artificial_path=False,
                bank_domain_gdf=bank_domain_gdf,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
            raise RuntimeError(f"Structured XS generation failed for {network_gpkg}: {exc}") from exc
        if not xs_out.exists():
            raise RuntimeError(
                f"Structured XS generation completed without producing expected artifact: {xs_out}"
            )
        return xs_out

    xs_source = _ensure_structured_xs_source()
    require_explicit_bank_guidance = bool(str(getattr(cfg, "river_method", "") or "").lower() == "v1")
    if xs_source is None and require_explicit_bank_guidance:
        raise RuntimeError("Canonical v1 river workflow requires structured XS-derived bank guidance, but no XS source was available.")
    if xs_source is not None:
        try:
            try:
                from river_bank_guidance import build_persistent_bank_network_points, summarize_bank_qc
                xs_bank_points_gdf = build_persistent_bank_network_points(xs_source, target_crs=ds.crs)
                if xs_bank_points_gdf is not None and not xs_bank_points_gdf.empty:
                    try:
                        if bqc_points_path.exists():
                            bqc_points_path.unlink()
                        xs_bank_points_gdf.to_file(bqc_points_path, driver="GPKG")
                        bank_qc_summary = summarize_bank_qc(xs_bank_points_gdf)
                        bank_qc_summary.to_csv(bqc_summary_path, index=False)
                        log.info(
                            "[RIVER][GUIDANCE] XS bank QC: points=%d replaced=%d high_suspect=%d strong=%d clamped=%d replaced_env=%d rejected=%d",
                            int(len(xs_bank_points_gdf)),
                            int(bank_qc_summary.iloc[0].get("bank_replaced_count", 0)),
                            int(bank_qc_summary.iloc[0].get("high_bank_suspect_count", 0)),
                            int(bank_qc_summary.iloc[0].get("strong_contamination_count", 0)),
                            int(bank_qc_summary.iloc[0].get("bank_clamp_count", 0)),
                            int(bank_qc_summary.iloc[0].get("bank_replace_with_local_envelope_count", 0)),
                            int(bank_qc_summary.iloc[0].get("bank_reject_count", 0)),
                        )
                    except (OSError, RuntimeError, ValueError, TypeError):
                        log.debug("[RIVER][GUIDANCE] Failed to write XS bank QC artifacts.", exc_info=True)
            except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError):
                xs_bank_points_gdf = None
            xs_bank = compute_xs_bank_guidance_surfaces(
                corridor_mask=channel.astype(bool),
                transform=ds.transform,
                auth=np.full(depth.shape, np.nan, dtype="float32"),
                xs_gpkg=xs_source,
                raster_crs=ds.crs,
                max_bank_distance_m=max(float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0) * 0.35, 120.0),
                bank_points_gdf=xs_bank_points_gdf,
            )
            river_bank_elevation_xs = xs_bank["bank_elevation"].astype("float32")
            river_bank_pair_weight = xs_bank["bank_pair_weight"].astype("float32")
            river_bank_continuity_weight = xs_bank.get("bank_continuity_weight", np.zeros(depth.shape, dtype="float32")).astype("float32")
            bank_ctx = compute_graph_informed_bank_context_surfaces(
                corridor_mask=channel.astype(bool),
                transform=ds.transform,
                bank_points_gdf=xs_bank_points_gdf,
                estuary_transition=estuary_transition.astype(bool),
                estuary_decay_distance_m=max(float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0) * 0.22, 120.0),
            )
            river_bank_graph_confidence = bank_ctx.get("bank_graph_confidence", np.zeros(depth.shape, dtype="float32")).astype("float32")
            river_bank_confluence_damping = bank_ctx.get("bank_confluence_damping", np.ones(depth.shape, dtype="float32")).astype("float32")
            river_bank_estuary_side_decay = bank_ctx.get("bank_estuary_side_decay", np.ones(depth.shape, dtype="float32")).astype("float32")
            river_bank_influence = np.clip(
                river_bank_influence
                * (0.40 + (0.20 * np.clip(river_bank_pair_weight, 0.0, 1.0)) + (0.20 * np.clip(river_bank_continuity_weight, 0.0, 1.0)) + (0.20 * np.clip(river_bank_graph_confidence, 0.0, 1.0)))
                * np.clip(river_bank_confluence_damping, 0.0, 1.0)
                * np.clip(river_bank_estuary_side_decay, 0.0, 1.0),
                0.0,
                1.0,
            ).astype("float32")
        except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            if require_explicit_bank_guidance:
                raise RuntimeError("Canonical v1 river workflow requires explicit XS-derived bank guidance, but bank guidance generation failed.") from exc
            log.debug("[RIVER] XS bank guidance build failed; continuing with corridor-edge bank guidance only.", exc_info=True)

    if require_explicit_bank_guidance:
        finite_bank_pixels = int(np.count_nonzero(np.isfinite(river_bank_elevation_xs)))
        active_bank_influence_pixels = int(np.count_nonzero(np.clip(np.nan_to_num(river_bank_influence, nan=0.0), 0.0, 1.0) > 0.05))
        if finite_bank_pixels <= 0:
            raise RuntimeError(
                "Canonical v1 river workflow requires explicit bank elevations, but river_bank_elevation_xs contains no finite pixels."
            )
        if active_bank_influence_pixels <= 0:
            raise RuntimeError(
                "Canonical v1 river workflow requires explicit bank influence, but river_bank_influence contains no active pixels."
            )
        log.info(
            "[RIVER][GUIDANCE] Explicit bank guidance active from authoritative DEM edge sampling: finite_bank_pixels=%d active_bank_influence_pixels=%d xs_source=%s",
            finite_bank_pixels,
            active_bank_influence_pixels,
            xs_source,
        )

    # Structured scaffold products derived from the retained river polygon/network.
    structured_bank_points = None
    centerline_points = None
    xs_support_points = None
    river_centerline_elevation = np.full(depth.shape, np.nan, dtype="float32")
    river_centerline_influence = np.zeros(depth.shape, dtype="float32")
    river_centerline_stationing = np.full(depth.shape, np.nan, dtype="float32")
    river_xs_support_elevation = np.full(depth.shape, np.nan, dtype="float32")
    river_xs_support_weight = np.zeros(depth.shape, dtype="float32")
    retained_network_meta = {"enabled": False, "selection_mode": "not_attempted"}

    def _estimate_native_pixel_spacing_m(ds_obj) -> float:
        try:
            px_x = abs(float(ds_obj.transform.a))
            px_y = abs(float(ds_obj.transform.e))
            if px_x <= 0.0 and px_y <= 0.0:
                return 3.0
            if getattr(ds_obj.crs, "is_geographic", False):
                lat0 = 0.5 * (float(ds_obj.bounds.bottom) + float(ds_obj.bounds.top))
                lat_rad = np.deg2rad(lat0)
                m_per_deg_lat = 111132.92 - (559.82 * np.cos(2.0 * lat_rad)) + (1.175 * np.cos(4.0 * lat_rad))
                m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3.0 * lat_rad)
                sx = px_x * abs(m_per_deg_lon)
                sy = px_y * abs(m_per_deg_lat)
            else:
                sx = px_x
                sy = px_y
            vals = [v for v in (sx, sy) if np.isfinite(v) and v > 0.0]
            return float(min(vals)) if vals else 3.0
        except (TypeError, ValueError, RuntimeError, AttributeError):
            log.debug("[RIVER] pixel-size-m calculation failed; defaulting to 3.0m", exc_info=True)
            return 3.0

    def _resolve_xs_authoritative_sample_raster() -> Optional[Path]:
        candidates = [
            artifacts.get("aligned_authoritative_base"),
            report.get("authoritative_base", {}).get("outputs", {}).get("aligned_authoritative_base") if isinstance(report.get("authoritative_base"), dict) else None,
            report.get("authoritative_base_auto", {}).get("outputs", {}).get("aligned_authoritative_base") if isinstance(report.get("authoritative_base_auto"), dict) else None,
            getattr(cfg, "authoritative_base", None),
        ]
        for cand in candidates:
            try:
                if cand is None:
                    continue
                path = Path(cand)
                if path.exists():
                    return path
            except (TypeError, ValueError, OSError):
                continue
        return None

    network_gpkg = _resolve_structured_network_gpkg()
    if network_gpkg.exists():
        retained = select_retained_river_features(
            network_gpkg,
            target_crs=ds.crs,
            min_stream_order=int(getattr(cfg, "river_scaffold_min_stream_order", 3) or 3),
            min_length_km=float(getattr(cfg, "river_scaffold_min_length_km", 0.25) or 0.25),
            keep_top_components=int(getattr(cfg, "river_scaffold_keep_top_components", 6) or 6),
            nhdarea_layer=str(getattr(cfg, "river_nhdarea_layer", "nhdarea_clip") or "nhdarea_clip"),
        )
        retained_network_meta = dict(retained.metadata)
        retained_network_meta["enabled"] = True
        retained_network_meta["estuary_mask_applied"] = True
        scaffold_allowed_mask = apply_estuary_first_channel_domain(channel.astype("uint8"), estuary_transition.astype("uint8")).astype(bool)
        t_clip0 = time.perf_counter()
        retained_flows_clipped = _clip_gdf_to_allowed_mask(retained.flows, scaffold_allowed_mask, ds.transform)
        retained_polygons_clipped = _clip_gdf_to_allowed_mask(retained.polygons, scaffold_allowed_mask, ds.transform)
        retained_flows_clipped, retained_component_meta = normalize_structured_flow_component_ids(retained_flows_clipped, logger=log)
        t_clip1 = time.perf_counter()
        retained_network_meta["retained_flow_count_after_estuary_clip"] = int(len(retained_flows_clipped)) if retained_flows_clipped is not None else 0
        retained_network_meta["retained_polygon_count_after_estuary_clip"] = int(len(retained_polygons_clipped)) if retained_polygons_clipped is not None else 0
        retained_component_expectation = resolve_centerline_component_expectation(retained_component_meta)
        retained_network_meta["structured_component_id_source"] = str(retained_component_expectation.get("component_id_source", "unknown"))
        retained_network_meta["structured_component_count_before"] = int(retained_component_expectation.get("component_count_before", 0))
        retained_network_meta["structured_component_count_after"] = int(retained_component_expectation.get("component_count_after", 0))
        log.info("[RIVER][GUIDANCE] Estuary-trimmed retained network: flows=%d polygons=%d elapsed=%.2fs",
                 retained_network_meta["retained_flow_count_after_estuary_clip"],
                 retained_network_meta["retained_polygon_count_after_estuary_clip"],
                 t_clip1 - t_clip0)
        auto_spacing_m = _estimate_native_pixel_spacing_m(ds)
        retained_flow_count = int(retained_network_meta.get("retained_flow_count_after_estuary_clip", 0) or 0)
        retained_component_count = int(retained_component_meta.get("component_count_after", 0) or 0)
        try:
            total_retained_length_m = float(np.nansum([float(getattr(g, "length", 0.0) or 0.0) for g in retained_flows_clipped.geometry])) if retained_flows_clipped is not None and not retained_flows_clipped.empty else 0.0
        except Exception:
            log.debug("[RIVER] Failed to compute total retained length; defaulting to 0.", exc_info=True)
            total_retained_length_m = 0.0
        auto_spacing_scale = 1.0
        if retained_flow_count >= 100 or retained_component_count >= 24 or total_retained_length_m >= 250000.0:
            auto_spacing_scale = 3.0
        elif retained_flow_count >= 50 or retained_component_count >= 12 or total_retained_length_m >= 100000.0:
            auto_spacing_scale = 2.0
        elif retained_flow_count >= 20 or retained_component_count >= 8 or total_retained_length_m >= 50000.0:
            auto_spacing_scale = 1.5
        adaptive_spacing_m = float(max(auto_spacing_m * auto_spacing_scale, auto_spacing_m))
        bank_spacing_m = float(getattr(cfg, "river_bank_sample_spacing_m", None) or adaptive_spacing_m)
        centerline_spacing_m = float(getattr(cfg, "river_centerline_sample_spacing_m", None) or adaptive_spacing_m)
        xs_spacing_m = float(getattr(cfg, "river_xs_support_spacing_m", None) or adaptive_spacing_m)
        xs_authoritative_sample_raster = _resolve_xs_authoritative_sample_raster()
        retained_network_meta["auto_sample_spacing_m"] = float(auto_spacing_m)
        retained_network_meta["adaptive_sample_spacing_scale"] = float(auto_spacing_scale)
        retained_network_meta["adaptive_sample_spacing_m"] = float(adaptive_spacing_m)
        retained_network_meta["retained_flow_total_length_m"] = float(total_retained_length_m)
        retained_network_meta["bank_sample_spacing_m"] = float(bank_spacing_m)
        retained_network_meta["centerline_sample_spacing_m"] = float(centerline_spacing_m)
        retained_network_meta["xs_support_spacing_m"] = float(xs_spacing_m)
        retained_network_meta["xs_support_authoritative_sample_raster"] = str(xs_authoritative_sample_raster) if xs_authoritative_sample_raster else None
        retained_network_meta["xs_support_allow_inferred_fallback"] = False
        t_bank0 = time.perf_counter()
        guidance_sample_raster = Path(getattr(cfg, "river_dem", "") or getattr(cfg, "authoritative_base", "") or bed_tif)
        if not guidance_sample_raster.exists():
            guidance_sample_raster = Path(bed_tif)
        retained_network_meta["guidance_sample_raster"] = str(guidance_sample_raster)
        retained_network_meta["guidance_sample_raster_role"] = "authoritative_dem_preferred"
        structured_bank_points = build_dense_bank_points(
            retained_polygons_clipped,
            spacing_m=bank_spacing_m,
            raster_path=guidance_sample_raster,
            normal_search_max_m=float(getattr(cfg, "river_bank_normal_search_max_m", 8.0) or 8.0),
            normal_search_step_m=float(getattr(cfg, "river_bank_normal_search_step_m", 1.0) or 1.0),
            xs_bank_points=xs_bank_points_gdf,
            allowed_mask=scaffold_allowed_mask,
            transform=ds.transform,
        )
        t_bank1 = time.perf_counter()
        log.info("[RIVER][GUIDANCE] Bank points built: n=%d spacing=%.2fm elapsed=%.2fs",
                 0 if structured_bank_points is None else int(len(structured_bank_points)),
                 float(bank_spacing_m),
                 t_bank1 - t_bank0)
        if structured_bank_points is not None and not structured_bank_points.empty:
            try:
                bank_shapes = [(geom, 1) for geom in structured_bank_points.geometry if geom is not None and not geom.is_empty]
                if bank_shapes:
                    bank_elevation_sample_mask = rasterize(bank_shapes, out_shape=depth.shape, transform=ds.transform, fill=0, dtype="uint8")
                    bank_elevation_sample_mask = ((bank_elevation_sample_mask > 0) & channel.astype(bool)).astype("uint8")
                    dem_stage_receipt["bank_sample_pixels"] = int(bank_elevation_sample_mask.sum())
            except (OSError, RuntimeError, ValueError, TypeError):
                log.debug("[RIVER][GUIDANCE] Failed to rasterize bank elevation sample mask.", exc_info=True)
        t_center0 = time.perf_counter()
        log.info("[RIVER][GUIDANCE] Building centerline points from retained flows...")
        centerline_points = build_centerline_points(
            retained_flows_clipped,
            retained_polygons_clipped,
            spacing_m=centerline_spacing_m,
            raster_path=guidance_sample_raster,
            allowed_mask=scaffold_allowed_mask,
            transform=ds.transform,
            logger=log,
        )
        centerline_authoritative_snap_receipt = {
            "status": "not_attempted",
            "eligible_support_points": 0,
            "matched_centerline_points": 0,
            "max_distance_m": 0.0,
        }
        if centerline_points is not None and not centerline_points.empty:
            centerline_authoritative_snap_m = float(
                getattr(cfg, 'river_centerline_authoritative_snap_m', None)
                or min(max(centerline_spacing_m * 0.50, 6.0), 20.0)
            )
            centerline_points, centerline_authoritative_snap_receipt = _apply_authoritative_bed_to_centerline_points(
                centerline_points,
                support_points,
                target_crs=ds.crs,
                max_distance_m=centerline_authoritative_snap_m,
                logger=log,
            )
            log.info(
                "[RIVER][GUIDANCE] Centerline authoritative support upgrade: matched=%d eligible_support=%d max_distance=%.2fm status=%s",
                int(centerline_authoritative_snap_receipt.get('matched_centerline_points', 0)),
                int(centerline_authoritative_snap_receipt.get('eligible_support_points', 0)),
                float(centerline_authoritative_snap_receipt.get('max_distance_m', 0.0)),
                str(centerline_authoritative_snap_receipt.get('status', 'unknown')),
            )
            retained_network_meta['centerline_authoritative_snap_receipt'] = centerline_authoritative_snap_receipt
        t_center1 = time.perf_counter()
        log.info("[RIVER][GUIDANCE] Centerline points built: n=%d spacing=%.2fm elapsed=%.2fs",
                 0 if centerline_points is None else int(len(centerline_points)),
                 float(centerline_spacing_m),
                 t_center1 - t_center0)
        centerline_station_contract = getattr(centerline_points, "attrs", {}).get("station_contract", {}) if centerline_points is not None else {}
        centerline_component_expectation = resolve_centerline_component_expectation(
            retained_component_meta,
            centerline_station_contract,
        )
        centerline_component_contract = validate_centerline_station_component_contract(
            centerline_points,
            expected_component_count=int(centerline_component_expectation.get("expected_component_count", 0)),
            expected_component_source=str(centerline_component_expectation.get("component_id_source", "unknown")),
            logger=log,
            context="retained_centerline_points",
        )
        centerline_dup_fraction = float(centerline_station_contract.get("duplicate_component_station_fraction", 0.0) or 0.0)
        centerline_dup_warn_threshold = float(getattr(cfg, "river_centerline_duplicate_warn_fraction", 0.05) or 0.05)
        centerline_dup_fail_threshold = getattr(cfg, "river_centerline_duplicate_fail_fraction", None)
        if centerline_dup_fail_threshold is not None:
            centerline_dup_fail_threshold = float(centerline_dup_fail_threshold)
        retained_network_meta["centerline_duplicate_component_station_rows_collapsed"] = int(centerline_station_contract.get("duplicate_component_station_rows_collapsed", 0))
        retained_network_meta["centerline_duplicate_component_station_fraction"] = float(centerline_dup_fraction)
        retained_network_meta["centerline_unique_component_station_count"] = int(centerline_station_contract.get("unique_component_station_count", 0))
        retained_network_meta["centerline_component_count"] = int(centerline_component_contract.get("component_count", 0))
        retained_network_meta["centerline_component_id_source"] = str(centerline_component_expectation.get("component_id_source", "unknown"))
        retained_network_meta["centerline_component_count_before"] = int(centerline_component_expectation.get("component_count_before", 0))
        retained_network_meta["centerline_component_count_after"] = int(centerline_component_expectation.get("component_count_after", 0))
        retained_network_meta["centerline_expected_component_count"] = int(centerline_component_expectation.get("expected_component_count", 0))
        retained_network_meta["centerline_station_contract_path"] = str(cpts_contract_path)
        if centerline_dup_fraction >= centerline_dup_warn_threshold:
            log.warning(
                "[RIVER][GUIDANCE] Centerline duplicate component/station collapse remains elevated: fraction=%.3f threshold=%.3f worst_component=%s",
                centerline_dup_fraction,
                centerline_dup_warn_threshold,
                centerline_station_contract.get("worst_duplicate_component_id"),
            )
        if centerline_dup_fail_threshold is not None and centerline_dup_fraction >= centerline_dup_fail_threshold:
            raise RuntimeError(
                f"centerline_duplicate_station_fraction_exceeds_threshold:{centerline_dup_fraction:.6f}>={centerline_dup_fail_threshold:.6f}"
            )
        if bool(getattr(cfg, "river_disable_xs_influence", False)):
            retained_network_meta["xs_support_contract"] = {
                "status": "disabled_by_option",
                "source_path": str(xs_source) if xs_source is not None else None,
                "spacing_m": float(xs_spacing_m),
                "kept_final_count": 0,
                "authoritative_sample_count": 0,
                "fallback_sample_count": 0,
                "missing_sample_count": 0,
                "allow_inferred_fallback": False,
                "disabled_by_option": True,
            }
            log.info("[RIVER][GUIDANCE] XS support generation disabled by option; using longitudinal/centerline guidance only")
        elif xs_source is not None:
            t_xs0 = time.perf_counter()
            log.info("[RIVER][GUIDANCE] Building XS support points from retained cross-sections...")
            xs_support_points = build_xs_support_points(
                xs_source,
                retained_polygons_clipped,
                spacing_fraction=float(getattr(cfg, "river_xs_support_fraction", 0.25) or 0.25),
                spacing_m=xs_spacing_m,
                preferred_raster_path=xs_authoritative_sample_raster,
                raster_path=guidance_sample_raster,
                allow_inferred_fallback=False,
                allowed_mask=scaffold_allowed_mask,
                transform=ds.transform,
                logger=log,
            )
            t_xs1 = time.perf_counter()
            log.info("[RIVER][GUIDANCE] XS support points built: n=%d spacing=%.2fm elapsed=%.2fs",
                     0 if xs_support_points is None else int(len(xs_support_points)),
                     float(xs_spacing_m),
                     t_xs1 - t_xs0)
            xs_support_contract = dict(getattr(xs_support_points, "attrs", {}).get("xs_support_contract", {})) if xs_support_points is not None else {}
            if xs_support_contract:
                xs_support_contract["source_path"] = str(xs_source)
                xs_support_contract["spacing_m"] = float(xs_spacing_m)
                retained_network_meta["xs_support_contract"] = xs_support_contract
            if xs_support_points is not None and not xs_support_points.empty and "xs_sample_source" in xs_support_points.columns:
                xs_source_counts = xs_support_points["xs_sample_source"].fillna("missing").astype(str).value_counts().to_dict()
                retained_network_meta["xs_support_sample_source_counts"] = {str(k): int(v) for k, v in xs_source_counts.items()}
                log.info(
                    "[RIVER][GUIDANCE] XS support sample sources: authoritative=%d fallback_cached_bed=%d missing=%d",
                    int(xs_source_counts.get("authoritative", 0)),
                    int(xs_source_counts.get("fallback_cached_bed", 0)),
                    int(xs_source_counts.get("missing", 0)),
                )
        else:
            retained_network_meta["xs_support_contract"] = {
                "status": "xs_source_missing_or_not_requested",
                "source_path": None,
                "spacing_m": float(xs_spacing_m),
                "kept_final_count": 0,
                "authoritative_sample_count": 0,
                "fallback_sample_count": 0,
                "missing_sample_count": 0,
                "allow_inferred_fallback": False,
            }
        retained_network_meta["structured_bank_point_count"] = int(len(structured_bank_points)) if structured_bank_points is not None else 0
        retained_network_meta["centerline_point_count"] = int(len(centerline_points)) if centerline_points is not None else 0
        retained_network_meta["xs_support_point_count"] = int(len(xs_support_points)) if xs_support_points is not None else 0
        if structured_bank_points is not None and not structured_bank_points.empty:
            if xs_bank_points_gdf is not None and not xs_bank_points_gdf.empty:
                # Keep dense polygon-bank points as the primary bank-boundary scaffold but retain
                # side-tagged XS bank points for continuity/side-aware bank fields.
                bank_points_out = structured_bank_points.copy()
                bank_points_out["artifact_role"] = "bank_boundary_control"
            else:
                bank_points_out = structured_bank_points.copy()
            if not bank_points_out.empty:
                if bpts_path.exists():
                    bpts_path.unlink()
                bank_points_out.to_file(bpts_path, driver="GPKG")
        if centerline_points is not None and not centerline_points.empty:
            if cpts_path.exists():
                cpts_path.unlink()
            centerline_points.to_file(cpts_path, driver="GPKG")
            centerline_station_contract = dict(centerline_station_contract)
            centerline_station_contract.setdefault("expected_component_source", str(centerline_component_expectation.get("component_id_source", "unknown")))
            centerline_station_contract.setdefault("expected_component_count", int(centerline_component_expectation.get("expected_component_count", 0)))
            with open(cpts_contract_path, "w", encoding="utf-8") as f:
                json.dump(centerline_station_contract, f, indent=2, sort_keys=True)
            try:
                bundle_b_stage_status = report.setdefault("river", {}).get("simple_stage_status")
                if not isinstance(bundle_b_stage_status, dict) or not bundle_b_stage_status:
                    bundle_b_stage_status = simple_river_stage_status_placeholder()
                bundle_b_result = run_simple_river_bundle_b(
                    river_context={
                        "centerline_points_gdf": centerline_points,
                        "existing_centerline_path": str(cpts_path),
                        "bank_wse_profile_summary_path": str(river_dir / "river_bank_wse_proxy_profile_summary.csv"),
                        "bank_wse_edge_guidance_path": str(river_dir / "river_bank_wse_edge_guidance.tif"),
                        "authoritative_base_path": str(getattr(cfg, "authoritative_base", "") or ""),
                        "input_artifacts": [str(retained_network_path)] if retained_network_path.exists() else [],
                    },
                    out_dir=str(river_dir),
                    stage_status=bundle_b_stage_status,
                )
                report.setdefault("river", {})["simple_stage_status"] = bundle_b_result.get("simple_river_stage_status", bundle_b_stage_status)
                report["river"]["simple_river_stage_outputs"] = dict(bundle_b_result.get("simple_river_stage_outputs", {}))
                plan_dict = report["river"].get("execution_plan")
                if isinstance(plan_dict, dict):
                    plan_dict["simple_stage_status"] = report["river"]["simple_stage_status"]
                    plan_dict["simple_river_stage_outputs"] = dict(report["river"].get("simple_river_stage_outputs", {}))
            except Exception:
                log.exception("[RIVER][BUNDLE_B] Failed to materialize simple river centerline stage")
                river_block = report.setdefault("river", {})
                current_status = bundle_b_stage_status if isinstance(bundle_b_stage_status, dict) else simple_river_stage_status_placeholder()
                river_block.setdefault("simple_stage_status", current_status)
                failed_stage = "river_centerline"
                error_text = "unknown bundle B failure"
                bundle_b_result_local = locals().get("bundle_b_result")
                if isinstance(bundle_b_result_local, dict):
                    failed_stage = str(bundle_b_result_local.get("failed_stage") or bundle_b_result_local.get("stage_id") or failed_stage)
                    error_text = str(bundle_b_result_local.get("error") or bundle_b_result_local.get("warnings") or error_text)
                    river_block["simple_stage_status"] = bundle_b_result_local.get("simple_river_stage_status", river_block["simple_stage_status"])
                    river_block["simple_river_stage_outputs"] = dict(bundle_b_result_local.get("simple_river_stage_outputs", {}))
                river_block["simple_river_stage_failure"] = {"failed_stage": failed_stage, "error": error_text}
            t_center_surf0 = time.perf_counter()
            centerline_transition_m = max(float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0) * 0.45, 120.0)
            centerline_influence_scale = float(getattr(cfg, "river_centerline_influence_scale", 1.0) or 1.0)
            river_centerline_elevation = rasterize_point_seed_surface(
                shape=depth.shape,
                transform=ds.transform,
                domain_mask=channel.astype(bool),
                points_gdf=centerline_points,
                value_field="centerline_z_m",
            )
            river_centerline_influence = build_centerline_core_influence(
                shape=depth.shape,
                transform=ds.transform,
                domain_mask=channel.astype(bool),
                points_gdf=centerline_points,
                bank_distance_m=river_bank_distance_m,
                max_distance_m=centerline_transition_m,
                influence_scale=centerline_influence_scale,
            ).astype("float32")
            if "station_m" in centerline_points.columns:
                river_centerline_stationing = rasterize_point_chainage_surface(
                    shape=depth.shape,
                    transform=ds.transform,
                    domain_mask=channel.astype(bool),
                    points_gdf=centerline_points,
                    station_field="station_m",
                )
            centerline_domain_mask = channel.astype(bool)
            centerline_domain_pixels = int(np.count_nonzero(centerline_domain_mask))
            centerline_positive = np.asarray(river_centerline_influence > 0.0, dtype=bool)
            centerline_gt05 = np.asarray(river_centerline_influence >= 0.05, dtype=bool)
            centerline_gt25 = np.asarray(river_centerline_influence >= 0.25, dtype=bool)
            centerline_gt50 = np.asarray(river_centerline_influence >= 0.50, dtype=bool)
            log.info(
                "[RIVER][GUIDANCE] Centerline seed/influence rasterized: points=%d seed_pixels=%d influence_pixels=%d domain_pixels=%d frac_gt05=%.3f frac_gt25=%.3f frac_gt50=%.3f scale=%.3f elapsed=%.2fs",
                int(len(centerline_points)),
                int(np.count_nonzero(np.isfinite(river_centerline_elevation))),
                int(np.count_nonzero(centerline_positive)),
                centerline_domain_pixels,
                (float(np.count_nonzero(centerline_gt05 & centerline_domain_mask)) / float(max(centerline_domain_pixels, 1))),
                (float(np.count_nonzero(centerline_gt25 & centerline_domain_mask)) / float(max(centerline_domain_pixels, 1))),
                (float(np.count_nonzero(centerline_gt50 & centerline_domain_mask)) / float(max(centerline_domain_pixels, 1))),
                float(centerline_influence_scale),
                time.perf_counter() - t_center_surf0,
            )
            try:
                center_shapes = [(geom, 1) for geom in centerline_points.geometry if geom is not None and not geom.is_empty]
                if center_shapes:
                    centerline_elevation_sample_mask = rasterize(center_shapes, out_shape=depth.shape, transform=ds.transform, fill=0, dtype="uint8")
                    centerline_elevation_sample_mask = ((centerline_elevation_sample_mask > 0) & channel.astype(bool)).astype("uint8")
                    dem_stage_receipt["centerline_sample_pixels"] = int(centerline_elevation_sample_mask.sum())
            except (OSError, RuntimeError, ValueError, TypeError):
                log.debug("[RIVER][GUIDANCE] Failed to rasterize centerline elevation sample mask.", exc_info=True)
        if xs_support_points is not None and not xs_support_points.empty:
            if xsp_path.exists():
                xsp_path.unlink()
            xs_support_points.to_file(xsp_path, driver="GPKG")
            t_xs_surf0 = time.perf_counter()
            river_xs_support_elevation, river_xs_support_weight = _nearest_surface_from_points(
                shape=depth.shape,
                transform=ds.transform,
                domain_mask=channel.astype(bool),
                points_gdf=xs_support_points,
                value_field="xs_z_m",
                max_distance_m=max(float(getattr(cfg, "river_scaffold_transition_m", 800.0) or 800.0) * 0.30, 80.0),
            )
            log.info(
                "[RIVER][GUIDANCE] XS support surface rasterized: points=%d domain_pixels=%d elapsed=%.2fs",
                int(len(xs_support_points)),
                int(np.count_nonzero(channel)),
                time.perf_counter() - t_xs_surf0,
            )
            dem_stage_receipt["xs_support_sample_pixels"] = int(np.count_nonzero(np.isfinite(river_xs_support_elevation)))
        # Replace generic corridor cloud with a structured scaffold union restricted to retained polygons.
        structured_guide = []
        for gdf in (structured_bank_points, centerline_points, xs_support_points):
            if gdf is not None and not gdf.empty:
                structured_guide.append(gdf)
        _template_artifacts = _resolve_channel_template_artifacts(cfg, work_dir=work_dir, river_dir=river_dir)
        if bool(getattr(cfg, "river_channel_template_enabled", False)):
            expected_template_dir = Path(channel_template_dir)
            resolved_template_dir = Path(_template_artifacts.get("bundle_dir", expected_template_dir))
            if resolved_template_dir != expected_template_dir:
                raise RuntimeError(
                    f"Channel-template artifacts resolved from unexpected directory {resolved_template_dir}; expected {expected_template_dir}."
                )
        _template_dense_points_path = _template_artifacts["dense_points"]
        _template_runtime_summary_path = _template_artifacts["summary"]
        _template_runtime = None
        if cfg.river_channel_template_enabled:
            _template_runtime_summary_path, _template_runtime = _load_channel_template_runtime_summary(
                cfg,
                work_dir=work_dir,
                river_dir=river_dir,
            )
            _template_dense_points_path = _validate_channel_template_runtime_summary(
                _template_runtime_summary_path,
                _template_runtime,
            )
        _template_dense_points_gdf = None
        if bool(getattr(cfg, "river_channel_template_enabled", False)) and _template_dense_points_path.exists():
            try:
                import geopandas as gpd
                _template_dense_points_gdf = gpd.read_file(_template_dense_points_path)
                if _template_dense_points_gdf is None or _template_dense_points_gdf.empty:
                    raise RuntimeError(f"Channel-template dense guide points are empty: {_template_dense_points_path}")
                if "z_bed_pred_m" in _template_dense_points_gdf.columns and "xs_z_m" not in _template_dense_points_gdf.columns:
                    _template_dense_points_gdf["xs_z_m"] = pd.to_numeric(_template_dense_points_gdf["z_bed_pred_m"], errors="coerce")
                _template_dense_points_gdf["feature_role"] = "channel_template_dense_points"
                structured_guide.append(_template_dense_points_gdf)
                log.info("[RIVER][GUIDANCE] Included channel-template dense guide points: n=%d", int(len(_template_dense_points_gdf)))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load required channel-template dense guide points: {_template_dense_points_path} ({exc})"
                ) from exc
        if structured_guide:
            guide_gdf = _concat_structured_guide_layers(structured_guide, expected_crs=structured_guide[0].crs)
            gp_path = river_dir / 'river_guide_points.gpkg'
            if gp_path.exists():
                gp_path.unlink()
            t_guide0 = time.perf_counter()
            guide_gdf.to_file(gp_path, driver='GPKG')
            artifacts['guide_points'] = str(gp_path)
            log.info("[RIVER][GUIDANCE] Structured guide points written: n=%d elapsed=%.2fs", int(len(guide_gdf)), time.perf_counter() - t_guide0)
        artifacts.update(
            _channel_template_output_entries(
                cfg,
                dense_points_path=_template_dense_points_path,
                runtime_summary_path=_template_runtime_summary_path,
            )
        )
        import geopandas as gpd
        retained_layers = []
        if retained_flows_clipped is not None and not retained_flows_clipped.empty:
            rf = retained_flows_clipped.copy(); rf["feature_role"] = "retained_flowline"; retained_layers.append(rf)
        if retained_polygons_clipped is not None and not retained_polygons_clipped.empty:
            rp = retained_polygons_clipped.copy(); rp["feature_role"] = "retained_polygon"; retained_layers.append(rp)
        if retained_layers:
                if retained_network_path.exists():
                    retained_network_path.unlink()
                for i, layer in enumerate(retained_layers):
                    lname = "retained_flowlines" if i == 0 else "retained_polygons"
                    layer.to_file(retained_network_path, layer=lname, driver="GPKG")

    rc_path = river_dir / "river_regime_class.tif"
    longitudinal_profile_source[np.isfinite(river_bank_elevation_xs) & channel.astype(bool)] = 1
    longitudinal_profile_source[np.isfinite(river_centerline_elevation) & channel.astype(bool)] = 2
    longitudinal_profile_source[np.isfinite(river_xs_support_elevation) & channel.astype(bool)] = 3
    dem_stage_receipt["longitudinal_profile_source_pixels"] = {
        str(int(k)): int(v) for k, v in zip(*np.unique(longitudinal_profile_source[longitudinal_profile_source > 0], return_counts=True))
    }

    with rasterio.open(gw_path, "w", **prof_f32) as dst:
        dst.write(guidance_weight.astype("float32"), 1)
    with rasterio.open(ti_path, "w", **prof_u8) as dst:
        dst.write(trusted_interior.astype("uint8"), 1)
    with rasterio.open(ad_path, "w", **prof_u8) as dst:
        dst.write(admissibility.astype("uint8"), 1)
    with rasterio.open(sg_path, "w", **prof_u8) as dst:
        dst.write(soft_guidance_domain.astype("uint8"), 1)
    with rasterio.open(sp_path, "w", **prof_u8) as dst:
        dst.write(anchor_support.astype("uint8"), 1)
    with rasterio.open(sd_path, "w", **prof_depth) as dst:
        dst.write(np.where(np.isfinite(support_depth), support_depth, float(nodata)).astype("float32"), 1)
    with rasterio.open(dem_proj_path, "w", **prof_diag) as dst:
        dst.write(np.where(np.isfinite(river_dem_projected), river_dem_projected, np.nan).astype("float32"), 1)
    with rasterio.open(dem_valid_path, "w", **prof_u8) as dst:
        dst.write(river_dem_valid_mask.astype("uint8"), 1)
    with rasterio.open(bank_sample_mask_path, "w", **prof_u8) as dst:
        dst.write(bank_elevation_sample_mask.astype("uint8"), 1)
    with rasterio.open(centerline_sample_mask_path, "w", **prof_u8) as dst:
        dst.write(centerline_elevation_sample_mask.astype("uint8"), 1)
    with rasterio.open(long_source_path, "w", **prof_u8) as dst:
        dst.write(longitudinal_profile_source.astype("uint8"), 1)
    write_json(dem_stage_receipt_path, dem_stage_receipt)
    with rasterio.open(cm_path, "w", **prof_u8) as dst:
        dst.write(corridor.astype("uint8"), 1)
    with rasterio.open(be_path, "w", **prof_u8) as dst:
        dst.write(river_bank_edge.astype("uint8"), 1)
    with rasterio.open(bd_path, "w", **prof_f32) as dst:
        dst.write(np.where(np.isfinite(river_bank_distance_m), river_bank_distance_m, 0.0).astype("float32"), 1)
    with rasterio.open(bi_path, "w", **prof_f32) as dst:
        dst.write(river_bank_influence.astype("float32"), 1)
    with rasterio.open(bx_path, "w", **prof_diag) as dst:
        dst.write(np.where(np.isfinite(river_bank_elevation_xs), river_bank_elevation_xs, float(nodata)).astype("float32"), 1)
    with rasterio.open(bpw_path, "w", **prof_f32) as dst:
        dst.write(np.clip(river_bank_pair_weight, 0.0, 1.0).astype("float32"), 1)
    with rasterio.open(bcw_path, "w", **prof_f32) as dst:
        dst.write(np.clip(river_bank_continuity_weight, 0.0, 1.0).astype("float32"), 1)
    with rasterio.open(bgc_path, "w", **prof_f32) as dst:
        dst.write(np.clip(river_bank_graph_confidence, 0.0, 1.0).astype("float32"), 1)
    with rasterio.open(bcd_path, "w", **prof_f32) as dst:
        dst.write(np.clip(river_bank_confluence_damping, 0.0, 1.0).astype("float32"), 1)
    with rasterio.open(besd_path, "w", **prof_f32) as dst:
        dst.write(np.clip(river_bank_estuary_side_decay, 0.0, 1.0).astype("float32"), 1)
    with rasterio.open(ce_path, "w", **prof_diag) as dst:
        dst.write(np.where(np.isfinite(river_centerline_elevation), river_centerline_elevation, float(nodata)).astype("float32"), 1)
    with rasterio.open(ci_path, "w", **prof_f32) as dst:
        dst.write(np.clip(river_centerline_influence, 0.0, 1.0).astype("float32"), 1)
    if np.any(np.isfinite(river_centerline_stationing)):
        with rasterio.open(cs_path, "w", **prof_diag) as dst:
            dst.write(np.where(np.isfinite(river_centerline_stationing), river_centerline_stationing, float(nodata)).astype("float32"), 1)
    with rasterio.open(xse_path, "w", **prof_diag) as dst:
        dst.write(np.where(np.isfinite(river_xs_support_elevation), river_xs_support_elevation, float(nodata)).astype("float32"), 1)
    with rasterio.open(xsw_path, "w", **prof_f32) as dst:
        dst.write(np.clip(river_xs_support_weight, 0.0, 1.0).astype("float32"), 1)
    with rasterio.open(rc_path, "w", **prof_u8) as dst:
        dst.write(regime_class.astype("uint8"), 1)

    scaffold_domains = get_river_aoi_domains(
        str(cfg.aoi),
        float(getattr(cfg, "river_network_halo_km", 0.0) or 0.0),
        float(getattr(cfg, "river_trusted_halo_m", 60.0) or 0.0),
    )
    scaffold_domains = scaffold_domains.as_dict()
    scaffold_domains["trusted_halo_px"] = int(edge_buffer_px)
    trusted_summary = summarize_trusted_export_region(
        channel=channel.astype("uint8"),
        trusted_export_region=trusted_interior.astype("uint8"),
        estuary_transition=estuary_transition.astype("uint8"),
        edge_buffer_px=int(edge_buffer_px),
    )
    trusted_summary_path = river_dir / "river_trusted_interior_summary.json"
    write_json(trusted_summary_path, trusted_summary)
    support_receipt_path = river_dir / "river_authoritative_support_receipt.json"
    write_json(support_receipt_path, support_receipt)

    upstream_mask_exports = {}
    for key, src in (
        ("upstream_waffles_with_nhd_water_mask", getattr(cfg, "waffles_with_nhd_mask", None)),
        ("upstream_river_channel_mask", channel_mask_tif),
        ("upstream_river_guidance_domain_mask", getattr(cfg, "river_guidance_domain_mask", None)),
        ("upstream_open_water_mask", Path(channel_mask_tif).parent / "open_water_mask.tif" if channel_mask_tif is not None else None),
        ("upstream_mainstem_mask", Path(channel_mask_tif).parent / "mainstem_mask.tif" if channel_mask_tif is not None else None),
    ):
        try:
            src_path = Path(src) if src is not None else None
            if src_path is not None and src_path.exists():
                dst_path = river_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                upstream_mask_exports[key] = str(dst_path)
            else:
                upstream_mask_exports[key] = None
        except (OSError, RuntimeError, ValueError):
            log.debug("[RIVER][GUIDANCE] Failed to export upstream mask %s", key, exc_info=True)
            upstream_mask_exports[key] = None

    write_scaffold_manifest(
        scaffold_domains_path,
        domains=get_river_aoi_domains(
            str(cfg.aoi),
            float(getattr(cfg, "river_network_halo_km", 0.0) or 0.0),
            float(getattr(cfg, "river_trusted_halo_m", 60.0) or 60.0),
        ),
        extra=scaffold_domains,
    )

    artifacts.update({
        "guidance_weight": str(gw_path),
        "trusted_interior": str(ti_path),
        "admissibility": str(ad_path),
        "soft_guidance_domain": str(sg_path),
        "authoritative_support": str(sp_path),
        "authoritative_support_depth": str(sd_path),
        "corridor_mask": str(cm_path),
        "bank_edge_mask": str(be_path),
        "bank_distance": str(bd_path),
        "bank_influence": str(bi_path),
        "bank_elevation_xs": str(bx_path),
        "bank_pair_weight": str(bpw_path),
        "bank_continuity_weight": str(bcw_path),
        "bank_graph_confidence": str(bgc_path),
        "bank_confluence_damping": str(bcd_path),
        "bank_estuary_side_decay": str(besd_path),
        "bank_points": str(bpts_path) if bpts_path.exists() else None,
        "xs_bank_qc_points": str(bqc_points_path) if bqc_points_path.exists() else None,
        "xs_bank_qc_summary": str(bqc_summary_path) if bqc_summary_path.exists() else None,
        "centerline_points": str(cpts_path) if cpts_path.exists() else None,
        "centerline_station_contract": str(cpts_contract_path) if cpts_contract_path.exists() else None,
        "xs_support_points": str(xsp_path) if xsp_path.exists() else None,
        **_channel_template_output_entries(
            cfg,
            dense_points_path=_template_dense_points_path,
            runtime_summary_path=_template_runtime_summary_path,
        ),
        "centerline_elevation": str(ce_path),
        "centerline_influence": str(ci_path),
        "centerline_stationing": str(cs_path) if cs_path.exists() else None,
        "xs_support_elevation": str(xse_path),
        "xs_support_weight": str(xsw_path),
        "retained_network": str(retained_network_path) if retained_network_path.exists() else None,
        "regime_class": str(rc_path),
        "scaffold_domains": str(scaffold_domains_path),
        "trusted_interior_summary": str(trusted_summary_path),
        "authoritative_support_receipt": str(support_receipt_path),
        "estuary_clip_mask": str(estuary_clip_path) if estuary_clip_path.exists() else None,
    })
    artifacts.update(upstream_mask_exports)

    # river_guide_points.gpkg is now emitted from the structured scaffold union above.

    support_source_counts = {}
    try:
        if support_points is not None and not support_points.empty and 'source' in support_points.columns:
            support_source_counts = {
                str(k): int(v)
                for k, v in support_points['source'].astype(str).value_counts().to_dict().items()
            }
    except (TypeError, ValueError, KeyError, AttributeError):
        support_source_counts = {}

    report.setdefault('river', {}).setdefault('guidance', {}).update({
        'guidance_only': True,
        'non_authoritative': True,
        'authoritative_support_pixels': int(support_used),
        'authoritative_support_source_counts': support_source_counts,
        'trusted_interior_definition': 'halo-solve export interior: channel pixels inset from solve-domain edges and excluding estuary transition',
        'soft_guidance_definition': 'broader trusted-export guidance domain prior to removing authoritative anchor pixels',
        'authoritative_support_depth_definition': 'rasterized trusted support depth used for exact overwrite where available',
        'guidance_weight_definition': '0 outside trusted export interior and at authoritative anchors; ramps upward within the trusted interior away from anchors',
        'admissibility_definition': '1 inside trusted export interior where valid river guidance exists and no authoritative anchor is present',
        'low_confidence_continuous_fill_definition': 'continuous backstop fill used only where no stronger support class was available after authoritative and guidance-conditioned filling',
        'estuary_transition_definition': '1 inside retained river channel upstream of the width-ratio estuary clip, used as the reduced-trust handoff zone to coastal SDB.',
        'estuary_transition_pixels': estuary_px_count,
        'estuary_transition_buffer_m': float(getattr(cfg, 'estuary_transition_m', 500.0) or 500.0),
        'estuary_transition_ocean_pixels': int(ocean_proximity_mask.sum()),
        'estuary_transition_hydraulic_pixels': int(hydraulic_hint_mask.sum()),
        'estuary_transition_hydraulic': hydraulic_hint_meta,
        'bed_surface_is_internal_helper': True,
        'regime_summary': contract.get('regime_summary', {}),
        'zone_summary': {k: int(np.sum(v > 0)) for k, v in contract.get('zones', {}).items()},
        'corridor_mask': str(cm_path),
        'retained_network': str(retained_network_path) if retained_network_path.exists() else None,
        'retained_network_summary': retained_network_meta,
        'guide_points_definition': 'Structured scaffold union restricted to retained WAFFLES/NHD river polygons: dense bank boundary points, longitudinal centerline control points, and selected cross-stream support nodes. Generic buffered-corridor random sampling is not used.',
        'bank_points_definition': 'Dense bank-boundary control points sampled along retained WAFFLES/NHD polygon banks from the authoritative baseline DEM.',
        'centerline_points_definition': 'Dense longitudinal control points sampled along retained mainstem/key-tributary flowlines inside the retained river polygon.',
        'simple_river_centerline_stage_definition': 'Canonical simple-river centerline stage built from the retained centerline points with normalized schema, receipt, and explicit stage status.',
        'xs_support_points_definition': 'Selected cross-stream support nodes inside the retained river polygon; where available they are rebuilt from XS bathymetry re-verticalized onto the solved longitudinal backbone so cross-channel shape is preserved while absolute bed placement follows the longitudinal profile.',
        'channel_template_dense_points_definition': 'Dense interior channel guide points exported from the XS channel-template path using predicted bed elevations and confidence weights; merged into river guidance when channel template is enabled.',
        'bank_edge_mask_definition': 'interior corridor-edge pixels derived from the WAFFLES/NHD river corridor boundary',
        'bank_distance_definition': 'distance from each corridor pixel to the nearest interior bank edge, used to taper bank-boundary influence inward',
        'bank_influence_definition': 'soft bank-boundary tendency derived from corridor-edge proximity; strongest near banks and decays toward the channel interior',
        'scaffold_domains': scaffold_domains,
        'trusted_interior_summary': trusted_summary,
    })
    report.setdefault('outputs', {})['river_trusted_interior'] = str(ti_path)
    report.setdefault('outputs', {})['river_scaffold_domains'] = str(scaffold_domains_path)
    report.setdefault('outputs', {})['river_trusted_interior_summary'] = str(trusted_summary_path)
    report.setdefault('outputs', {})['river_dem_projected'] = str(dem_proj_path)
    report.setdefault('outputs', {})['river_dem_valid_mask'] = str(dem_valid_path)
    river_report_obj = report.get('river', {}) if isinstance(report.get('river', {}), dict) else {}
    simple_stage_outputs = river_report_obj.get('simple_river_stage_outputs', {}) if isinstance(river_report_obj.get('simple_river_stage_outputs', {}), dict) else {}
    centerline_block = simple_stage_outputs.get('centerline') if isinstance(simple_stage_outputs.get('centerline'), dict) else None
    if centerline_block and centerline_block.get('output_artifact'):
        report.setdefault('outputs', {})['simple_river_centerline_points'] = str(centerline_block.get('output_artifact'))
    if centerline_block and centerline_block.get('receipt_path'):
        report.setdefault('outputs', {})['simple_river_centerline_receipt'] = str(centerline_block.get('receipt_path'))
    wse_block = simple_stage_outputs.get('wse_proxy') if isinstance(simple_stage_outputs.get('wse_proxy'), dict) else None
    if wse_block and wse_block.get('output_artifact'):
        report.setdefault('outputs', {})['simple_river_centerline_wse_proxy_points'] = str(wse_block.get('output_artifact'))
    if wse_block and wse_block.get('receipt_path'):
        report.setdefault('outputs', {})['simple_river_centerline_wse_proxy_receipt'] = str(wse_block.get('receipt_path'))
    auth_bed_block = simple_stage_outputs.get('authoritative_bed') if isinstance(simple_stage_outputs.get('authoritative_bed'), dict) else None
    if auth_bed_block and auth_bed_block.get('output_artifact'):
        report.setdefault('outputs', {})['simple_river_centerline_authoritative_bed_points'] = str(auth_bed_block.get('output_artifact'))
    if auth_bed_block and auth_bed_block.get('receipt_path'):
        report.setdefault('outputs', {})['simple_river_centerline_authoritative_bed_receipt'] = str(auth_bed_block.get('receipt_path'))
    observed_offset_block = simple_stage_outputs.get('observed_offset') if isinstance(simple_stage_outputs.get('observed_offset'), dict) else None
    if observed_offset_block and observed_offset_block.get('output_artifact'):
        report.setdefault('outputs', {})['simple_river_centerline_observed_offset_points'] = str(observed_offset_block.get('output_artifact'))
    if observed_offset_block and observed_offset_block.get('receipt_path'):
        report.setdefault('outputs', {})['simple_river_centerline_observed_offset_receipt'] = str(observed_offset_block.get('receipt_path'))
    report.setdefault('outputs', {})['river_bank_elevation_sample_mask'] = str(bank_sample_mask_path)
    report.setdefault('outputs', {})['river_centerline_elevation_sample_mask'] = str(centerline_sample_mask_path)
    report.setdefault('outputs', {})['river_longitudinal_profile_source'] = str(long_source_path)
    report.setdefault('outputs', {})['river_dem_derived_stage_receipt'] = str(dem_stage_receipt_path)
    if estuary_clip_path.exists():
        report.setdefault('outputs', {})['estuary_clip_mask'] = str(estuary_clip_path)
    return artifacts


def _apply_residual_correction(
    raster_path: "Path",
    xyz_df: "pd.DataFrame",
    *,
    sigma_m: float = 500.0,
    max_correction_m: float = 10.0,
    min_points: int = 10,
    label: str = "residual_correction",
) -> "Dict[str, Any]":
    """Apply smooth residual correction to a raster using authoritative XYZ data.

    Computes residual = predicted - observed at each XYZ location, then
    interpolates the error field using normalized-convolution Gaussian
    smoothing and subtracts it from the raster.

    The correction decays naturally with distance from support because the
    Gaussian convolution spreads the correction signal; far from any
    measurement point the correction converges to zero.

    Args:
        raster_path: Path to the raster to correct (modified in-place).
        xyz_df: DataFrame with columns [lon, lat, depth_m] in geographic CRS.
        sigma_m: Gaussian smoothing sigma in meters.
        max_correction_m: Cap the maximum correction magnitude.
        min_points: Minimum number of valid residual points to apply correction.
        label: Label for logging and report.

    Returns:
        Dict with correction statistics.
    """
    import numpy as np
    import rasterio
    from scipy.ndimage import gaussian_filter

    raster_path = Path(raster_path)
    stats = {"applied": False, "label": label, "n_points": 0, "n_valid": 0}
    if not raster_path.exists():
        stats["reason"] = "raster_not_found"
        return stats
    if xyz_df is None or len(xyz_df) < min_points:
        stats["reason"] = "insufficient_xyz_points"
        return stats

    with rasterio.open(raster_path, "r+") as ds:
        arr = sanitize_array(ds.read(1), ds.nodata if ds.nodata is not None else -9999.0, dtype="float32")
        nodata = float(ds.nodata) if ds.nodata is not None else -9999.0
        transform = ds.transform
        px_size_m = max(abs(float(transform.a)), abs(float(transform.e)), 1e-6)

        # Determine if CRS is geographic (degrees) — need to convert sigma
        is_geographic = ds.crs is not None and ds.crs.is_geographic
        if is_geographic:
            # Approximate: 1 degree ≈ 111km at equator, less at higher latitudes
            mid_lat = abs(float(transform.f + transform.e * ds.height / 2))
            deg_per_m = 1.0 / (111_000.0 * max(np.cos(np.radians(mid_lat)), 0.1))
            sigma_px = sigma_m * deg_per_m / abs(float(transform.a))
        else:
            sigma_px = sigma_m / px_size_m

        # Cap sigma to prevent enormous kernels
        sigma_px = min(sigma_px, min(ds.height, ds.width) / 4.0)

        # Sample raster at XYZ locations
        lons = np.asarray(xyz_df["lon"] if "lon" in xyz_df.columns else xyz_df["longitude"], dtype=float)
        lats = np.asarray(xyz_df["lat"] if "lat" in xyz_df.columns else xyz_df["latitude"], dtype=float)
        depths = np.asarray(xyz_df["depth_m"], dtype=float)

        rows, cols = rasterio.transform.rowcol(transform, lons, lats)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)

        # Filter to valid in-bounds pixels
        h, w = arr.shape
        valid = (
            (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            & np.isfinite(depths) & np.isfinite(lons) & np.isfinite(lats)
        )
        rows, cols, depths = rows[valid], cols[valid], depths[valid]

        # Get predicted values at XYZ locations
        predicted = arr[rows, cols]
        pred_valid = valid_mask(predicted, nodata)
        rows, cols, depths, predicted = rows[pred_valid], cols[pred_valid], depths[pred_valid], predicted[pred_valid]

        stats["n_points"] = int(len(rows))
        if len(rows) < min_points:
            stats["reason"] = "insufficient_valid_residuals"
            return stats

        # Compute residuals: positive = predicted too deep, negative = predicted too shallow
        residuals = predicted - depths

        # Clip extreme residuals (likely datum mismatches or bad soundings)
        residuals = np.clip(residuals, -max_correction_m, max_correction_m)

        # Build sparse residual field and weight field
        resid_field = np.zeros_like(arr, dtype="float64")
        weight_field = np.zeros_like(arr, dtype="float64")
        # Use np.add.at for accumulation at duplicate pixel locations
        np.add.at(resid_field, (rows, cols), residuals)
        np.add.at(weight_field, (rows, cols), 1.0)

        # Normalized convolution: smooth both numerator and denominator
        resid_smooth = gaussian_filter(resid_field, sigma=sigma_px, mode="constant", cval=0.0)
        weight_smooth = gaussian_filter(weight_field, sigma=sigma_px, mode="constant", cval=0.0)

        # Avoid division by zero — where weight is negligible, correction is zero
        correction = np.divide(resid_smooth, weight_smooth, out=np.zeros_like(resid_smooth, dtype="float64"), where=weight_smooth > 1e-10).astype("float32")

        # Cap correction magnitude
        correction = np.clip(correction, -max_correction_m, max_correction_m)

        # Apply correction: subtract residual (if predicted was too deep, reduce depth)
        valid_arr = valid_mask(arr, nodata)
        arr_corrected = arr.copy()
        arr_corrected[valid_arr] -= correction[valid_arr]

        # Write back
        ds.write(arr_corrected.astype("float32"), 1)

    stats.update({
        "applied": True,
        "n_valid": int(len(rows)),
        "sigma_m": float(sigma_m),
        "max_correction_m": float(max_correction_m),
        "residual_median": float(np.median(residuals)),
        "residual_p95": float(np.percentile(np.abs(residuals), 95)),
        "residual_max_abs": float(np.max(np.abs(residuals))),
        "correction_field_p50": float(np.median(np.abs(correction[valid_arr]))),
        "correction_field_p95": float(np.percentile(np.abs(correction[valid_arr]), 95)),
    })
    log.info(
        "[%s] Applied to %s: %d points, residual median=%.2f m, p95=%.2f m, correction p50=%.2f m",
        label, raster_path.name, len(rows),
        stats["residual_median"], stats["residual_p95"], stats["correction_field_p50"],
    )
    return stats


def _apply_river_guidance_to_fused_output(
    combined_path: "Path",
    *,
    river_path: "Optional[Path]",
    sdb_path: "Optional[Path]",
    provenance_path: "Optional[Path]",
    river_outputs: "Dict[str, Any]",
    report: "Dict[str, Any]",
    estuary_max_weight: float = 0.25,
) -> None:
    """Apply guidance-only river controls to the fused raster.

    Domain-aware fusion policy:
      - authoritative support / trusted interior => exact overwrite (unchanged)
      - fluvial_core (admissible & NOT estuary_transition) => full river guidance weight
      - estuary_transition => river guidance weight capped to estuary_max_weight
      - outside admissible domain => river has no influence

    The estuary transition zone is a reduced-trust handoff where neither
    river nor SDB should dominate.  Capping river weight there prevents
    the raw ``(1-w)*sdb + w*river`` blend from creating seams where the
    river model's physical assumptions break down.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling

    def _align_to_template(src_path: Optional[Path], tmpl) -> Optional[np.ndarray]:
        if src_path is None:
            return None
        sp = Path(src_path)
        if not sp.exists():
            return None
        with rasterio.open(sp) as src:
            dtype = src.dtypes[0]
            if dtype.startswith('uint') or dtype.startswith('int'):
                out = np.zeros((tmpl.height, tmpl.width), dtype=np.float32)
                dst_nodata = 0.0
                rs = Resampling.nearest
            else:
                out = np.full((tmpl.height, tmpl.width), np.nan, dtype=np.float32)
                dst_nodata = np.nan
                rs = Resampling.nearest
            reproject(
                source=rasterio.band(src, 1),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tmpl.transform,
                dst_crs=tmpl.crs,
                resampling=rs,
                src_nodata=src.nodata,
                dst_nodata=dst_nodata,
            )
            return out

    with rasterio.open(combined_path, 'r+') as ds:
        arr = sanitize_array(ds.read(1), ds.nodata if ds.nodata is not None else -9999.0, dtype='float32')
        nodata = ds.nodata if ds.nodata is not None else -9999.0
        river = _align_to_template(river_path, ds)
        sdb = _align_to_template(sdb_path, ds)
        gw = _align_to_template(Path(river_outputs.get('guidance_weight')) if river_outputs.get('guidance_weight') else None, ds)
        ti = _align_to_template(Path(river_outputs.get('trusted_interior')) if river_outputs.get('trusted_interior') else None, ds)
        adm = _align_to_template(Path(river_outputs.get('admissibility')) if river_outputs.get('admissibility') else None, ds)
        sup = _align_to_template(Path(river_outputs.get('authoritative_support')) if river_outputs.get('authoritative_support') else None, ds)
        sup_depth = _align_to_template(Path(river_outputs.get('authoritative_support_depth')) if river_outputs.get('authoritative_support_depth') else None, ds)
        et = _align_to_template(Path(river_outputs.get('estuary_transition')) if river_outputs.get('estuary_transition') else None, ds)
        # Estuary clip mask: pixels removed from river channel because fluvial
        # physics break down there.  SDB should be the sole guidance source in
        # these zones (the river module produced no values here).
        estuary_clip = _align_to_template(Path(river_outputs.get('estuary_clip_mask')) if river_outputs.get('estuary_clip_mask') else None, ds)

        if river is None or gw is None or adm is None:
            report.setdefault('fusion', {}).setdefault('guidance_controls', {})['river_guidance_applied'] = False
            report['fusion']['guidance_controls']['reason'] = 'missing_required_river_guidance_artifacts'
            return

        valid_riv = valid_mask(river, nodata)
        valid_sdb = valid_mask(sdb, nodata) if sdb is not None else np.zeros_like(valid_riv, dtype=bool)
        domain = (adm > 0.5)
        is_estuary = (et > 0.5) if et is not None else np.zeros_like(domain, dtype=bool)
        ti_mask = (ti > 0.5) if ti is not None else np.zeros_like(domain, dtype=bool)
        sup_mask = (sup > 0.5) if sup is not None else np.zeros_like(domain, dtype=bool)
        exact = domain & (ti_mask | sup_mask)

        # --- Domain-aware weight capping ---
        # In estuary_transition, cap river guidance weight to estuary_max_weight.
        # In fluvial_core (domain & ~estuary), use full guidance weight.
        w = np.clip(np.nan_to_num(gw, nan=0.0), 0.0, 1.0).astype('float32')
        w_capped = w.copy()
        estuary_cap = float(np.clip(estuary_max_weight, 0.0, 1.0))
        w_capped[is_estuary] = np.minimum(w[is_estuary], estuary_cap)

        blend = domain & (~exact) & valid_riv
        sup_depth_valid = valid_mask(sup_depth, nodata) if sup_depth is not None else np.zeros_like(domain, dtype=bool)
        exact_support = exact & sup_depth_valid
        exact_river = exact & (~exact_support) & valid_riv
        exact_written = int(np.sum(exact_support | exact_river))

        # Blend zones: estuary uses capped weight, fluvial_core uses full weight
        both = blend & valid_sdb
        river_only = blend & (~valid_sdb)

        # Sub-classify blend zones
        both_estuary = both & is_estuary
        both_fluvial = both & (~is_estuary)
        river_only_estuary = river_only & is_estuary
        river_only_fluvial = river_only & (~is_estuary)

        arr_out = arr.copy()

        # Exact overwrites (unchanged — authoritative always wins)
        if np.any(exact_support):
            arr_out[exact_support] = sup_depth[exact_support]
        if np.any(exact_river):
            arr_out[exact_river] = river[exact_river]

        # Fluvial core: full-strength river guidance blend
        if np.any(both_fluvial):
            wf = w[both_fluvial]
            arr_out[both_fluvial] = ((1.0 - wf) * sdb[both_fluvial]) + (wf * river[both_fluvial])
        if np.any(river_only_fluvial):
            arr_out[river_only_fluvial] = river[river_only_fluvial]

        # Estuary transition: capped-weight blend (reduced trust)
        if np.any(both_estuary):
            we = w_capped[both_estuary]
            arr_out[both_estuary] = ((1.0 - we) * sdb[both_estuary]) + (we * river[both_estuary])
        estuary_no_base_fill_skipped = 0
        if np.any(river_only_estuary):
            # In the estuary transition, river guidance is reduced-trust.
            # If there is already a fused/base value, blend toward river with the capped weight.
            # If there is no existing base value, do not inject pure river structure into the
            # handoff zone; leave the cell unchanged and record the skip in the report.
            we_ro = w_capped[river_only_estuary]
            existing = arr[river_only_estuary]
            has_existing = valid_mask(existing, nodata)
            riv_vals = river[river_only_estuary]
            updated = np.where(
                has_existing,
                ((1.0 - we_ro) * existing) + (we_ro * riv_vals),
                existing,
            )
            arr_out[river_only_estuary] = updated
            estuary_no_base_fill_skipped = int(np.sum(~has_existing))

        # --- Estuary clip zone: SDB is sole guidance source ---
        # Where the estuary clip mask is active, the river module produced
        # no values (the channel mask was zeroed there).  If SDB has valid
        # data in those pixels, write it directly — this is the estuarine
        # zone where SDB optics may still work even though river physics
        # broke down.  Authoritative exact-overwrites are preserved.
        estuary_sdb_fill_n = 0
        is_clipped_estuary = (estuary_clip > 0.5) if estuary_clip is not None else np.zeros_like(domain, dtype=bool)
        if np.any(is_clipped_estuary) and sdb is not None:
            sdb_fill = is_clipped_estuary & valid_sdb & (~exact_support)
            if np.any(sdb_fill):
                arr_out[sdb_fill] = sdb[sdb_fill]
                estuary_sdb_fill_n = int(sdb_fill.sum())
                log.info(
                    "[FUSION] SDB fills %d pixels in estuary-clip zone (river excluded, SDB admissible).",
                    estuary_sdb_fill_n,
                )

        ds.write(arr_out.astype('float32'), 1)

    # --- Provenance update ---
    if provenance_path is not None and Path(provenance_path).exists():
        try:
            from constants import Provenance
            with rasterio.open(provenance_path, 'r+') as dp:
                prov = dp.read(1)
                if np.any(exact_support):
                    prov[exact_support] = Provenance.MEASURED
                if np.any(exact_river):
                    prov[exact_river] = Provenance.RIVER
                if np.any(both_fluvial):
                    prov[both_fluvial] = Provenance.BLENDED
                if np.any(river_only_fluvial):
                    prov[river_only_fluvial] = Provenance.RIVER
                # Estuary transition gets its own provenance code
                if np.any(both_estuary):
                    prov[both_estuary] = Provenance.ESTUARY_TRANSITION
                if np.any(river_only_estuary):
                    prov[river_only_estuary] = Provenance.ESTUARY_TRANSITION
                # Estuary clip zone filled by SDB
                if estuary_sdb_fill_n > 0:
                    prov[is_clipped_estuary & valid_sdb & (~exact_support)] = Provenance.SDB
                dp.write(prov, 1)
        except Exception:
            log.debug('ignored', exc_info=True)

    n_estuary_blend = int(np.sum(both_estuary)) + int(np.sum(river_only_estuary))
    report.setdefault('fusion', {}).setdefault('guidance_controls', {}).update({
        'river_guidance_applied': True,
        'authoritative_exact_overwrite_pixels': exact_written,
        'river_guidance_blend_fluvial_pixels': int(np.sum(both_fluvial)),
        'river_guidance_river_only_fluvial_pixels': int(np.sum(river_only_fluvial)),
        'estuary_transition_blend_pixels': n_estuary_blend,
        'estuary_transition_no_base_fill_skipped_pixels': int(estuary_no_base_fill_skipped),
        'estuary_clip_sdb_fill_pixels': int(estuary_sdb_fill_n),
        'estuary_max_weight': estuary_cap,
        'policy': (
            'trusted support exact overwrite using authoritative depth where available; '
            'full-strength river guidance only in fluvial_core; '
            'capped river guidance (max_w=%.2f) in estuary_transition; '
            'SDB-only outside admissible river domain' % estuary_cap
        ),
    })
def _is_final_dem_user_raster(path: Path, out_dir: Path) -> bool:
    try:
        return path.resolve() == (Path(out_dir) / "combined" / "DEM_enhanced.tif").resolve()
    except Exception:
        return str(path).endswith(str(Path("combined") / "DEM_enhanced.tif"))


def _is_debug_final_route_raster(path: Path, out_dir: Path) -> bool:
    try:
        resolved = path.resolve()
        return (Path(out_dir) / "combined" / "debug_final_route").resolve() in resolved.parents
    except Exception:
        return "combined/debug_final_route/" in str(path).replace('\\', '/')


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
    methods = [m.strip().lower() for m in (cfg.methods or [])]
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
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
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
    def _is_full_aoi_deliverable(path: Path) -> bool:
        stem = path.stem.lower()
        return stem.endswith("_all") or stem.endswith("_all_hillshade")
    deliverable_rasters = [p for p in deliverable_rasters if not _is_full_aoi_deliverable(p)]
    protected_rasters = []
    filtered_deliverables = []
    for p in deliverable_rasters:
        if _is_final_dem_user_raster(p, out_dir) or _is_debug_final_route_raster(p, out_dir):
            protected_rasters.append(p)
            continue
        filtered_deliverables.append(p)
    deliverable_rasters = filtered_deliverables
    if protected_rasters:
        try:
            log.info("[FINAL_DOMAIN] Protected post-final artifacts skipped from clipping: %s", ", ".join(str(p) for p in protected_rasters))
        except Exception:
            pass
    river_rasters = [p for p in deliverable_rasters if ("river" in p.parts)]
    combined_rasters = [p for p in deliverable_rasters if ("combined" in p.parts)]
    def _clip(path: Path, mask: Path, *, inside_value: int, nodata: float) -> None:
        if path.exists() and mask.exists():
            _clip_raster_to_mask_reproject(path, mask, inside_value=inside_value, invert=False, nodata=nodata)

    # Prefer run-scoped WAFFLES masks captured in cfg by domain inference (avoid stale discovery).
    wm_ocean = cfg.waffles_ocean_mask
    wm_nhd = cfg.waffles_with_nhd_mask
    wm_ocean = Path(wm_ocean) if wm_ocean else None
    wm_nhd = Path(wm_nhd) if wm_nhd else None

    # No guessing: WAFFLES masks must be provided by domain inference (shared cache staged into derived_cache).
    if wm_ocean is not None and (not wm_ocean.exists()):
        wm_ocean = None
    if wm_nhd is not None and (not wm_nhd.exists()):
        wm_nhd = None

    if sdb_on and river_on:
        wm = wm_nhd
        if not (wm and wm.exists()):
            raise RuntimeError("Final domain policy requires WAFFLES with-NHD mask, but it was not found.")
        for pth in deliverable_rasters:
            _clip(pth, wm, inside_value=0, nodata=cfg.final_nodata)
    elif sdb_on and (not river_on):
        wm = wm_ocean if (wm_ocean and wm_ocean.exists()) else wm_nhd
        if not (wm and wm.exists()):
            raise RuntimeError("Final domain policy requires a WAFFLES water mask (ocean-only or with-NHD), but none were found.")
        for pth in deliverable_rasters:
            _clip(pth, wm, inside_value=0, nodata=cfg.final_nodata)
    elif river_on and (not sdb_on):
        wm = wm_nhd if (wm_nhd and wm_nhd.exists()) else wm_ocean
        if not (wm and wm.exists()):
            raise RuntimeError("Final domain policy requires a WAFFLES water mask (with-NHD preferred), but none were found.")
        for pth in deliverable_rasters:
            _clip(pth, wm, inside_value=0, nodata=cfg.final_nodata)

        # Additionally restrict river deliverables to the river channel mask (inside=1).
        ch = cfg.river_channel_mask
        ch = Path(ch) if ch else None
        if ch is not None and ch.exists():
            for pth in river_rasters:
                _clip(pth, ch, inside_value=1, nodata=cfg.final_nodata)
            for pth in combined_rasters:
                if pth.stem.startswith("bathy_combined_depth") or pth.stem.startswith("bathy_cudem_enhanced_navd88") or pth.stem.startswith("cudem_baseline_interpolation"):
                    _clip(pth, ch, inside_value=1, nodata=cfg.final_nodata)
    # else: neither sdb nor river — no deliverables to clip; nothing to do.

    # Remove empty intermediate directories (deepest-first), preserving out_dir and the
    # intermediates/debug dir so that debug artefacts are not silently pruned.
    debug_dir = out_dir / str(cfg.intermediates_dirname or "debug")
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
            except OSError:
                log.debug("ignored", exc_info=True)




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


def _confirm_exists(path: str | Path) -> bool:
    """Return True if a path exists on disk (files or directories)."""
    try:
        return Path(path).exists()
    except (OSError, TypeError, ValueError):
        return False


def _log_confirmed_output(label: str, path: str | Path) -> None:
    """Log an output path only if it truly exists."""
    p = str(path)
    if _confirm_exists(p):
        log.info("• %s: %s", label, p)
    else:
        log.warning("[OUTPUT] Expected output missing (%s): %s", label, p)

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
        soundings = [Path(part) for part in _normalize_multi_path_value(cfg.river_soundings)]

    manifest = {
        "aoi": cfg.aoi,
        "river_method": cfg.river_method,
        "river_dem": _fingerprint_path(cfg.river_dem),
        "soundings": [_fingerprint_path(p) for p in soundings],
        "river_soundings_mode": cfg.river_soundings_mode,
        "river_soundings_max_dist_m": cfg.river_soundings_max_dist_m,
        "river_soundings_min_r": cfg.river_soundings_min_r,
        "river_soundings_enforce": cfg.river_soundings_enforce,
        "snap_m": cfg.snap_m,
        "river_da_raster": _fingerprint_path(cfg.river_da_raster),
        "river_da_raster_band": int(cfg.river_da_raster_band or 1),
        "river_da_raster_units": str(cfg.river_da_raster_units or "km2"),

        "river_authoritative_bed": _fingerprint_path(cfg.river_authoritative_bed),
        "river_authoritative_bed_max_dist_m": cfg.river_authoritative_bed_max_dist_m,
        "river_residual_blend_sigma_m": cfg.river_residual_blend_sigma_m,
        "river_nodata": cfg.river_nodata,
        "mask_river_to_waffles": cfg.mask_river_to_waffles,
        "river_use_nhdarea": cfg.river_use_nhdarea,
        "river_nhdarea_layer": cfg.river_nhdarea_layer,

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
        "river_thalweg_only": cfg.river_thalweg_only,
        "river_thalweg_densify_factor": cfg.river_thalweg_densify_factor,
        "river_thalweg_densify_step_m": cfg.river_thalweg_densify_step_m,
        "river_overlap_reducer": cfg.river_overlap_reducer,

        # XS generation controls (geometry stability / artifact reduction)
        "xs_smoothing_window_m": cfg.xs_smoothing_window_m,
        "xs_trim_overlaps": cfg.xs_trim_overlaps,
        "xs_global_deconflict": cfg.xs_global_deconflict,
        "xs_deconflict_tol_m": cfg.xs_deconflict_tol_m,
        "xs_skip_junctions": cfg.xs_skip_junctions,
        "xs_junction_snap_m": cfg.xs_junction_snap_m,
        "xs_junction_buffer_m": cfg.xs_junction_buffer_m,
        "xs_densify_step_m": cfg.xs_densify_step_m,

        # Skeleton parameters (only used when river_method == "skeleton")
        "river_channel_buffer_m": cfg.river_channel_buffer_m,
        "river_max_channel_width_m": cfg.river_max_channel_width_m,
        "river_mainstem_method": cfg.river_mainstem_method,
        "river_mainstem_solve_layer": cfg.river_mainstem_solve_layer,
        "river_mainstem_min_order": cfg.river_mainstem_min_order,
        "river_max_mainstem_width_m": cfg.river_max_mainstem_width_m,
        "river_guidance_bank_margin_m": cfg.river_guidance_bank_margin_m,
        "river_shape_exp": cfg.river_shape_exp,
        "river_dmax_min_m": cfg.river_dmax_min_m,
        "river_dmax_max_m": cfg.river_dmax_max_m,

        # Priors (shared)
        "river_prior_mode": cfg.river_prior_mode,
        "river_mv_a0": cfg.river_mv_a0,
        "river_mv_bw": cfg.river_mv_bw,
        "river_mv_ba": cfg.river_mv_ba,
        "river_mv_bs": cfg.river_mv_bs,
        "river_mv_eps_a": cfg.river_mv_eps_a,
        "river_mv_eps_s": cfg.river_mv_eps_s,

        # USGS anchors / gage-derived priors
        "river_usgs_sites": cfg.river_usgs_sites,
        "river_usgs_start": cfg.river_usgs_start,
        "river_usgs_end": cfg.river_usgs_end,
        "river_usgs_cache_dir": _fingerprint_path(cfg.river_usgs_cache_dir),
        "river_usgs_max_dist_m": cfg.river_usgs_max_dist_m,
        "river_usgs_mean_to_dmax": cfg.river_usgs_mean_to_dmax,
        "river_usgs_a_stat": cfg.river_usgs_a_stat,
        "river_usgs_q_quantile_lo": cfg.river_usgs_q_quantile_lo,
        "river_usgs_q_quantile_hi": cfg.river_usgs_q_quantile_hi,
        "river_usgs_a_cv_warn": cfg.river_usgs_a_cv_warn,
        "river_usgs_width_ratio_max": cfg.river_usgs_width_ratio_max,
        "river_usgs_width_ratio_blend": cfg.river_usgs_width_ratio_blend,
        "river_gage_snap_max_dist_m": cfg.river_gage_snap_max_dist_m,

        # Width-stage CSV anchors
        "river_width_stage_csv": cfg.river_width_stage_csv,
        "river_width_stage_max_dist_m": cfg.river_width_stage_max_dist_m,
        "river_width_stage_min_n": cfg.river_width_stage_min_n,
        "river_width_stage_min_r2": cfg.river_width_stage_min_r2,
        "river_width_stage_max_weight": cfg.river_width_stage_max_weight,

        # Slope/WSE profile controls
        "river_slope_proxy_window": cfg.river_slope_proxy_window,
        "river_slope_min": cfg.river_slope_min,
        "river_slope_max": cfg.river_slope_max,
        "river_slope_proxy_min_n": cfg.river_slope_proxy_min_n,
        "river_wse_profile_enabled": cfg.river_wse_profile_enabled,
        "river_wse_profile_window": cfg.river_wse_profile_window,
        "river_wse_profile_min_n": cfg.river_wse_profile_min_n,
        "river_wse_profile_monotonic": cfg.river_wse_profile_monotonic,

        # Skeleton scientific controls
        "river_skeleton_wse_mode": cfg.river_skeleton_wse_mode,
        "river_skeleton_wse_smooth_sigma_m": cfg.river_skeleton_wse_smooth_sigma_m,
        "river_skeleton_wse_profile_step_m": cfg.river_skeleton_wse_profile_step_m,
        "river_skeleton_wse_profile_resample_m": cfg.river_skeleton_wse_profile_resample_m,
        "river_skeleton_wse_profile_smooth_sigma_m": cfg.river_skeleton_wse_profile_smooth_sigma_m,
        "river_skeleton_wse_profile_max_slope": cfg.river_skeleton_wse_profile_max_slope,
        "river_skeleton_wse_profile_min_samples": cfg.river_skeleton_wse_profile_min_samples,
        "river_skeleton_wse_profile_max_query_dist_m": cfg.river_skeleton_wse_profile_max_query_dist_m,
        "river_skeleton_junction_mode": cfg.river_skeleton_junction_mode,
        "river_skeleton_junction_buffer_m": cfg.river_skeleton_junction_buffer_m,
        "river_skeleton_junction_degree_min": cfg.river_skeleton_junction_degree_min,
        "river_skeleton_junction_smooth_sigma_m": cfg.river_skeleton_junction_smooth_sigma_m,
        "river_skeleton_junction_max_width_m": cfg.river_skeleton_junction_max_width_m,
        "river_skeleton_asymmetry_mode": cfg.river_skeleton_asymmetry_mode,
        "river_skeleton_asymmetry_strength": cfg.river_skeleton_asymmetry_strength,
        "river_skeleton_asymmetry_curv_ref": cfg.river_skeleton_asymmetry_curv_ref,
        "river_skeleton_asymmetry_max_shift": cfg.river_skeleton_asymmetry_max_shift,
        "river_skeleton_asymmetry_min_width_m": cfg.river_skeleton_asymmetry_min_width_m,
        "river_skeleton_asymmetry_min_curv": cfg.river_skeleton_asymmetry_min_curv,
        "river_skeleton_asymmetry_densify_step_m": cfg.river_skeleton_asymmetry_densify_step_m,

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
    except (RecursionError, TypeError, ValueError):
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
    except ValueError:
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
    return _io_build_io_manifest(report)

def _ensure_bundle_a_route_contract_state(report: "Dict[str, Any]") -> Dict[str, Any]:
    """Populate shared route/policy/reporting metadata needed by Bundle A patch 3."""
    policy = default_final_dem_policy()
    river_report = report.setdefault("river", {}) if isinstance(report, dict) else {}
    execution_plan = river_report.get("execution_plan", {}) if isinstance(river_report.get("execution_plan", {}), dict) else {}
    v2_route_contract = river_report.get("v2_route_contract") if isinstance(river_report.get("v2_route_contract"), dict) else {}
    v2_requested_or_active = bool(v2_route_contract.get("runtime_enforced")) or str(river_report.get("execution_mode") or "").startswith("river_v2_")
    current_route_mode = str(
        (policy.target_route_mode if v2_requested_or_active else None)
        or execution_plan.get("route_mode")
        or river_report.get("route_mode")
        or (report.get("final_dem_route", {}) if isinstance(report.get("final_dem_route", {}), dict) else {}).get("route_mode")
        or policy.current_route_mode
    )
    target_route_mode = str(
        execution_plan.get("target_contract_mode")
        or river_report.get("target_contract_mode")
        or policy.target_route_mode
    )
    stage_status = execution_plan.get("simple_stage_status")
    if not isinstance(stage_status, dict) or not stage_status:
        stage_status = river_report.get("simple_stage_status") if isinstance(river_report.get("simple_stage_status"), dict) else simple_river_stage_status_placeholder()
    legacy_transitional_artifacts = execution_plan.get("legacy_transitional_components")
    if not isinstance(legacy_transitional_artifacts, list):
        legacy_transitional_artifacts = river_report.get("legacy_transitional_components") if isinstance(river_report.get("legacy_transitional_components"), list) else ["legacy_structured_river_guidance"]

    report.setdefault("final_dem_route", {})["route_mode"] = current_route_mode
    report["final_dem_route"]["target_route_mode"] = target_route_mode
    report.setdefault("workflow_execution_state", {})["current_route_mode"] = current_route_mode
    report["workflow_execution_state"]["target_route_mode"] = target_route_mode
    river_report["route_mode"] = current_route_mode
    river_report["target_contract_mode"] = target_route_mode
    river_report["simple_stage_status"] = stage_status
    river_report["legacy_transitional_components"] = legacy_transitional_artifacts

    report.setdefault("final_dem_policy", {}).update({
        "final_dem_filename": policy.final_dem_filename,
        "internal_final_dem_filename": policy.internal_final_dem_filename,
        "authoritative_hard_lock_required": bool(policy.authoritative_hard_lock_required),
        "write_final_dem_once": bool(policy.write_final_dem_once),
        "verify_only_postwrite": bool(policy.verify_only_postwrite),
        "current_route_mode": current_route_mode,
        "target_route_mode": target_route_mode,
    })
    report.setdefault("final_dem_contract", {}).update(
        build_final_dem_contract_summary(
            report,
            policy=policy,
            stage_status=stage_status,
            legacy_transitional_artifacts=list(legacy_transitional_artifacts),
        )
    )
    return {
        "current_route_mode": current_route_mode,
        "target_route_mode": target_route_mode,
        "simple_river_stage_status": stage_status,
        "legacy_transitional_artifacts": list(legacy_transitional_artifacts),
    }

def _write_guidance_manifest(cfg: "BathyConfig", report: "Dict[str, Any]") -> Path:
    from constants import Provenance, PIPELINE_VERSION
    _ensure_bundle_a_route_contract_state(report)
    out_path = _io_write_guidance_manifest(cfg, report, provenance_enum=Provenance, pipeline_version=PIPELINE_VERSION)
    log.info("Guidance manifest written: %s", out_path)
    return out_path

def write_io_manifest(out_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    return _io_write_io_manifest(out_dir, report)

def _emit_artifacts_from_report(report: dict[str, Any]) -> None:
    _io_emit_artifacts_from_report(report)

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
    make_figs: bool = False
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
    river_dem_res_m: float = 0.0
    extra_xyz_crs: str = "EPSG:4326"
    sdb_authoritative_extra_xyz: Optional[Path] = None
    sdb_authoritative_support_mask: Optional[Path] = None
    sdb_authoritative_support_values: Optional[Path] = None
    sdb_authoritative_support_points: Optional[Path] = None
    sdb_authoritative_support_contract: Optional[Path] = None
    river_authoritative_soundings: Optional[Path] = None
    river_withheld_support_csv: Optional[Path] = None
    river_withheld_support_receipt: Optional[Path] = None
    river_contract_mode: str = "canonical_v322"

    # WAFFLES domain-detection gate.
    # Minimum water fraction required for SDB (ocean mask) or river (NHD mask) to be enabled.
    # Set to 0.0 to bypass the gate entirely and always run requested methods.
    # Useful when waffles returns all-land for a valid inland river AOI.
    waffles_min_water_fraction: float = 0.001

    # Force regeneration of cached WAFFLES coastline masks (deletes stale cache entries).
    # Use when a prior run cached an all-land mask that is incorrect.
    force_waffles_masks: bool = False

    # ── Outputs / nodata ────────────────────────────────────────────────────
    final_nodata: float = -9999.0

    # Validation / invariance framework
    validation_truth: Optional[Path] = None
    validation_case_specs: List[str] = field(default_factory=list)
    validation_case_manifest: Optional[Path] = None
    validation_guidance_baseline_case: str = "baseline_cudem_interpolation"
    validation_guidance_target_case: str = "selected_final"
    validation_require_guidance_non_degradation: bool = False
    validation_guidance_rmse_tolerance: float = 0.0

    # ── WAFFLES mask configuration ───────────────────────────────────────────
    waffles_inc_arcsec: float = 1.0          # mask resolution (1.0 ≈ 30 m)
    # Runtime paths — set during pipeline, not from CLI:
    waffles_ocean_mask: Optional[Path] = None
    waffles_with_nhd_mask: Optional[Path] = None

    # ── Domain masks (set during pipeline) ──────────────────────────────────
    ocean_domain_mask_for_fusion: Optional[Path] = None
    river_domain_mask_for_fusion: Optional[Path] = None
    river_channel_mask: Optional[Path] = None

    # ── Run identity ─────────────────────────────────────────────────────────
    run_id: Optional[str] = None

    # ── River channel / network source ──────────────────────────────────────
    river_channel_source: str = "auto"   # 'auto' | 'nhd' | 'nhdarea' | 'waffles'
    river_nhdarea_allow_ftype: Optional[str] = None
    river_nhdarea_allow_fcode: Optional[str] = None
    river_domain_min_water_corridor_frac: float = 0.02
    river_domain_min_channel_corridor_frac: float = 0.001
    river_domain_min_channel_pixels: int = 1
    river_domain_hard_fail: bool = False

    # ── Soundings calibration ────────────────────────────────────────────────
    river_soundings_calib_max_dist_m: float = 0.0   # 0 = disabled
    river_soundings_calib_stat: str = ""
    river_soundings_cell_percentile: float = 25.0
    river_soundings_crs: str = ""
    river_no_soundings_enforce: bool = False

    # ── Regional hydraulic geometry curves ──────────────────────────────────
    river_regional_curve_enabled: bool = False

    # ── Channel template system (learned XS shape from measured sections) ────
    river_channel_template_enabled: bool = False
    river_channel_template_min_xs: int = 3
    river_channel_template_fit_min_xs: int = 5
    river_channel_template_n_bins: int = 50
    river_channel_template_min_depth_m: float = 0.3
    river_channel_template_distance_sigma_m: float = 2000.0
    river_channel_template_estuary_buffer_m: float = 500.0
    river_channel_template_junction_buffer_m: float = 120.0
    river_channel_template_width_depth_ratio_max: Optional[float] = None
    river_channel_template_loo_max_rmse_norm: float = 0.25
    river_channel_template_loo_max_dmax_error_m: float = 1.5

    # ── SWOT RiverSP integration ─────────────────────────────────────────────
    river_swot_riversp: Optional[str] = None
    river_swot_wse_field: Optional[str] = None
    river_swot_qual_field: Optional[str] = None
    river_swot_max_dist_m: float = 300.0
    river_swot_weight: float = 1.0
    river_swot_wse_offset_m: float = 0.0
    river_swot_min_samples: int = 5
    river_swot_correct_sigma_m: float = 2000.0
    river_swot_max_correction_m: float = 5.0
    river_swot_offset_mode: str = "median_mad"
    river_swot_offset_min_samples: int = 25
    river_swot_offset_max_abs_m: float = 10.0
    river_swot_offset_mad_z: float = 3.5

    # ── Skeleton asymmetry (curvature-driven thalweg shift) ──────────────────
    river_skeleton_asymmetry_mode: str = "none"       # 'none'|'curvature'|'curvature_smooth'
    river_skeleton_asymmetry_strength: float = 0.25
    river_skeleton_asymmetry_min_width_m: float = 10.0
    river_skeleton_asymmetry_max_shift: float = 0.20
    river_skeleton_asymmetry_min_curv: float = 0.0005
    river_skeleton_asymmetry_curv_ref: float = 0.002
    river_skeleton_asymmetry_densify_step_m: float = 20.0

    # ── Skeleton WSE profile ─────────────────────────────────────────────────
    river_skeleton_wse_profile_step_m: float = 20.0
    river_skeleton_wse_profile_resample_m: float = 20.0
    river_skeleton_wse_profile_smooth_sigma_m: float = 200.0
    river_skeleton_wse_profile_min_samples: int = 10
    river_skeleton_wse_profile_max_slope: float = 0.005
    river_skeleton_wse_profile_max_query_dist_m: float = 250.0

    # ── Spatial sampling ─────────────────────────────────────────────────────
    sampling_target_points: int = 2000
    sampling_min_threshold: int = 3000
    sampling_max_gap_m: float = 100.0
    enable_adaptive_sampling: bool = True

    # ── Soundings subsetting ─────────────────────────────────────────────────
    soundings_max_points: int = 0    # 0 = unlimited
    soundings_sample_seed: int = 0

    # ── Deprecated alias kept for backwards compatibility ────────────────────
    extra_xyz_files: Optional[List[str]] = None

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
    river_method: str = "structured"

    # Skeleton (distance-transform) method parameters
    # These control *where* river bathy is applied (river vs. ocean) and the within-channel depth profile.
    river_channel_buffer_m: float = 400.0           # buffer around NHD flowlines to define candidate river corridor
    river_max_channel_width_m: float = 600.0        # max channel width allowed in corridor (prevents filling open bays)
    river_mainstem_method: str = "dominant_trunk"  # mainstem identification policy for corridor widening
    river_mainstem_solve_layer: str = "auto"        # preferred network layer for dominant-trunk solve
    river_mainstem_min_order: int = 5               # stream order threshold for allowing larger widths (if available)
    river_max_mainstem_width_m: float = 2500.0      # max width allowed for mainstem corridor (m)
    river_guidance_bank_margin_m: float = 3.0      # exclude near-bank pixels from the river guidance domain review/fusion mask
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

    # Estuary transition zone — reduced-trust handoff between river and SDB
    estuary_transition_m: float = 500.0     # buffer (m) from ocean domain into river channel that defines the transition
    estuary_max_weight: float = 0.25        # max river guidance weight allowed in estuary transition (0..1)
    estuary_width_ratio_thresh: float = 3.0  # local width / median width ratio that triggers estuary classification
    estuary_connect_dist_m: float = 200.0   # max distance (m) from ocean to keep estuary detections (filters inland false positives)

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

    # XS-only physics stabilization
    river_enable_1d_energy_solver: bool = False
    river_energy_allow_dem_proxy_wse: bool = False

    # Discharge-driven priors (Manning inversion; passed through to xs_infer_bathy_raster.py)
    river_manning_mode: str = "off"              # off | constant | from_field | q2_regional
    river_manning_q_cms: Optional[float] = None  # discharge (m^3/s) when mode=constant
    river_manning_q_field: Optional[str] = None  # reach field with Q (m^3/s) when mode=from_field
    river_manning_n: float = 0.035
    river_manning_region: str = "auto"
    river_manning_min_confidence: float = 0.30
    river_manning_max_weight: float = 0.60
    river_manning_backwater_slope_thresh: float = 1e-4
    river_manning_dist_to_mouth_field: Optional[str] = None
    river_manning_dist_to_mouth_km_max: float = 10.0

    # Optional: pass through to river_network
    # Hydrography acquisition strategy for river network & polygons
    # - "arcgis": ArcGIS REST only (fast, avoids TNM catalog crawl)
    # - "arcgis_tnm": ArcGIS first, then TNM fallback if ArcGIS fails
    # - "tnm": TNM preferred (will still fall back to ArcGIS as a guardrail)
    river_hydrography_source: str = "arcgis"


    # Optional drainage-area raster fallback (sampled in river_network.py when NHDPlus DA is unavailable)
    river_da_raster: Optional[Path] = None
    river_da_raster_band: int = 1
    river_da_raster_units: str = "km2"  # km2 | m2
    
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
    # xs_infer defaults can be expensive for large networks; expose here for control.
    river_continuous: str = "walid_aniso"  # median | walid | aidw | aniso | walid_aniso
    river_continuous_buffer_m: Optional[float] = None
    river_continuous_k: int = 12
    river_idw_power: float = 2.0
    river_aniso_along_scale_m: float = 500.0
    river_aniso_cross_scale_m: float = 30.0
    river_thalweg_weight: float = 6.0
    river_xs_profile_shape: str = "parabolic"  # parabolic | linear_trapezoid | cosine_trapezoid

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
    authoritative_base: Optional[Path] = None  # hard-locked measured-constrained raster; only its nodata gaps are eligible for fill
    authoritative_base_auto: bool = True  # auto-materialize authoritative base from NOAA CUDEM tile index + spatial metadata by default
    authoritative_base_tile_index_url: str = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/tileindex_NCEI_ninth_Topobathy_2014.zip"
    authoritative_base_spatial_meta_url: str = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/ninth_spatial_meta.zip"
    authoritative_base_missing_meta_policy: str = "skip"
    authoritative_base_force_rebuild: bool = False
    authoritative_base_tile_url_field: Optional[str] = None
    authoritative_support_decay_m: float = 300.0  # e-folding distance controlling how quickly inferred guidance can dominate away from hard control
    authoritative_support_density_radius_m: float = 250.0  # neighborhood radius used to estimate local authoritative support density
    coastal_sdb_support_transition_m: float = 600.0  # distance scale controlling when stable optical SDB support can dominate in estuary/nearshore gaps
    river_anchor_density_radius_m: float = 200.0  # neighborhood radius used to estimate local river anchor density
    river_scaffold_transition_m: float = 800.0  # distance scale controlling when river guidance becomes channel-structure-guidance dominant away from anchors
    river_bank_sample_spacing_m: Optional[float] = None  # default: authoritative DEM pixel spacing in meters
    river_centerline_sample_spacing_m: Optional[float] = None  # default: authoritative DEM pixel spacing in meters
    river_centerline_influence_scale: float = 1.0  # multiplier on local centerline-core half-width; >1 broadens, <1 narrows, very large values saturate near full half-width
    river_xs_support_spacing_m: Optional[float] = None  # default: authoritative DEM pixel spacing in meters
    river_disable_xs_influence: bool = False  # if True, suppress XS-derived river influence and rely on longitudinal/centerline/backbone guidance only
    river_bank_normal_search_max_m: float = 8.0
    river_bank_normal_search_step_m: float = 1.0
    river_network_halo_km: float = 2.0  # halo used to build a more stable river scaffold domain than the clipped export AOI
    river_trusted_halo_m: float = 60.0  # interior buffer used to suppress edge-sensitive river guidance near solve-domain boundaries
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

def _authoritative_passthrough_args(cfg: "BathyConfig", *, for_river: bool = False) -> List[str]:
    """Thin wrapper around authoritative_cli helper for child-process passthrough."""
    return _build_authoritative_passthrough_args(cfg, for_river=for_river, logger=log)






def _concat_structured_guide_layers(layers, *, expected_crs, value_columns: tuple[str, ...] = ("xs_z_m", "bank_z_m", "centerline_z_m")):
    import geopandas as gpd
    from pyproj import CRS

    def _crs_equivalent(lhs, rhs) -> bool:
        lhs_crs = CRS.from_user_input(lhs)
        rhs_crs = CRS.from_user_input(rhs)
        try:
            if lhs_crs.is_exact_same(rhs_crs):
                return True
        except Exception:
            log.debug("CRS is_exact_same comparison failed; trying .equals()", exc_info=True)
        try:
            if lhs_crs.equals(rhs_crs):
                return True
        except Exception:
            log.debug("CRS .equals() comparison failed; falling back to EPSG/string", exc_info=True)
        lhs_epsg = lhs_crs.to_epsg()
        rhs_epsg = rhs_crs.to_epsg()
        if lhs_epsg is not None and rhs_epsg is not None:
            return lhs_epsg == rhs_epsg
        return str(lhs_crs) == str(rhs_crs)

    cleaned_layers = []
    expected_crs_norm = CRS.from_user_input(expected_crs) if expected_crs is not None else None
    for idx, layer in enumerate(layers):
        if layer is None or getattr(layer, "empty", True):
            continue
        layer_crs = getattr(layer, "crs", None)
        if expected_crs_norm is not None:
            if layer_crs is None:
                raise RuntimeError(
                    f"Structured river guidance layer {idx} is missing CRS; expected {str(expected_crs_norm)}."
                )
            layer_crs_norm = CRS.from_user_input(layer_crs)
            if not _crs_equivalent(expected_crs_norm, layer_crs_norm):
                raise RuntimeError(
                    f"Structured river guidance layer {idx} has CRS {str(layer_crs_norm)}, expected {str(expected_crs_norm)}."
                )
        layer_clean = layer.copy()
        if "geometry" not in layer_clean.columns:
            raise RuntimeError(f"Structured river guidance layer {idx} is missing geometry.")
        geom = layer_clean.geometry
        valid_geom = geom.notna() & (~geom.is_empty)
        layer_clean = layer_clean.loc[valid_geom].copy()
        if layer_clean.empty:
            continue
        present_value_columns = [col for col in value_columns if col in layer_clean.columns]
        if present_value_columns:
            finite_any = False
            for col in present_value_columns:
                vals = pd.to_numeric(layer_clean[col], errors="coerce")
                layer_clean[col] = vals
                finite_any = finite_any or bool(np.any(np.isfinite(vals.to_numpy(dtype=float, copy=False))))
            if not finite_any:
                raise RuntimeError(
                    f"Structured river guidance layer {idx} has no finite value columns among {present_value_columns}."
                )
        if expected_crs_norm is not None:
            layer_clean = layer_clean.set_crs(expected_crs_norm, allow_override=True)
        cleaned_layers.append(layer_clean)
    if not cleaned_layers:
        raise RuntimeError("Structured river guidance layers were all empty after validation.")
    plain_frames = []
    for layer_clean in cleaned_layers:
        layer_df = pd.DataFrame(layer_clean.drop(columns="geometry")).copy()
        layer_df["geometry"] = list(layer_clean.geometry)
        plain_frames.append(layer_df)
    merged = pd.concat(plain_frames, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=expected_crs_norm if expected_crs_norm is not None else expected_crs)

def _channel_template_cache_key(cfg) -> str:
    import hashlib
    key_fields = {
        "aoi": str(getattr(cfg, "aoi", "")),
        "river_dem": str(getattr(cfg, "river_dem", "")),
        "river_method": str(getattr(cfg, "river_method", "")),
        "xs_spacing_m": float(getattr(cfg, "xs_spacing_m", 0.0) or 0.0),
        "river_channel_template_min_xs": int(getattr(cfg, "river_channel_template_min_xs", 0) or 0),
        "river_channel_template_fit_min_xs": int(getattr(cfg, "river_channel_template_fit_min_xs", 0) or 0),
        "river_channel_template_n_bins": int(getattr(cfg, "river_channel_template_n_bins", 0) or 0),
        "river_channel_template_min_depth_m": float(getattr(cfg, "river_channel_template_min_depth_m", 0.0) or 0.0),
        "river_channel_template_distance_sigma_m": float(getattr(cfg, "river_channel_template_distance_sigma_m", 0.0) or 0.0),
        "river_channel_template_estuary_buffer_m": float(getattr(cfg, "river_channel_template_estuary_buffer_m", 0.0) or 0.0),
        "river_channel_template_junction_buffer_m": float(getattr(cfg, "river_channel_template_junction_buffer_m", 0.0) or 0.0),
        "river_channel_template_width_depth_ratio_max": (
            float(getattr(cfg, "river_channel_template_width_depth_ratio_max"))
            if getattr(cfg, "river_channel_template_width_depth_ratio_max", None) is not None
            else None
        ),
        "river_channel_template_loo_max_rmse_norm": float(getattr(cfg, "river_channel_template_loo_max_rmse_norm", 0.0) or 0.0),
        "river_channel_template_loo_max_dmax_error_m": float(getattr(cfg, "river_channel_template_loo_max_dmax_error_m", 0.0) or 0.0),
        "river_shape_exp": float(getattr(cfg, "river_shape_exp", 0.0) or 0.0),
        "river_mv_a0": float(getattr(cfg, "river_mv_a0", 0.0) or 0.0),
        "river_mv_bw": float(getattr(cfg, "river_mv_bw", 0.0) or 0.0),
    }
    key_material = json.dumps(key_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key_material.encode()).hexdigest()[:12]


def _channel_template_runtime_dir(river_dir: Path) -> Path:
    return Path(river_dir) / "channel_template"


def _river_guidance_runtime_context(river_dir: Path) -> dict[str, Path]:
    river_dir = Path(river_dir)
    work_dir = river_dir / "work"
    return {
        "river_dir": river_dir,
        "work_dir": work_dir,
        "channel_template_dir": _channel_template_runtime_dir(river_dir),
    }


def _channel_template_primary_dir(river_dir: Path | None, work_dir: Path) -> Path:
    if river_dir is not None:
        return _channel_template_runtime_dir(Path(river_dir))
    return Path(work_dir) / "channel_template"


def _append_unique_path(candidates: list[Path], cand: Path | None) -> None:
    if cand is None:
        return
    cand = Path(cand)
    if cand not in candidates:
        candidates.append(cand)


def _channel_template_candidate_dirs(cfg, work_dir: Path, river_dir: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    primary_dir = _channel_template_primary_dir(river_dir, work_dir)
    _append_unique_path(candidates, primary_dir)
    if river_dir is not None:
        _append_unique_path(candidates, Path(river_dir) / "work" / "channel_template")
    _append_unique_path(candidates, Path(work_dir) / "channel_template")
    _append_unique_path(candidates, Path(work_dir) / "work" / "channel_template")
    derived_cache_root = getattr(cfg, "derived_cache_root", None)
    if derived_cache_root:
        dcr = Path(derived_cache_root)
        _append_unique_path(candidates, dcr / "river" / "channel_template")
        _append_unique_path(candidates, dcr / "river" / "work" / "channel_template")
    return candidates


def _load_json_file(path: Path, *, label: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {label}: {path} ({exc})") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read {label}: {path} ({exc})") from exc


def _load_channel_template_runtime_summary(cfg, work_dir: Path, river_dir: Path | None = None) -> tuple[Path, dict]:
    artifacts = _resolve_channel_template_artifacts(cfg, work_dir=work_dir, river_dir=river_dir)
    summary_path = artifacts["summary"]
    summary = _load_json_file(summary_path, label="channel-template runtime summary")
    return summary_path, summary


def _validate_channel_template_runtime_summary(summary_path: Path, summary: dict) -> Path:
    dense_written = int(summary.get("dense_points_written", 0) or 0)
    if not bool(summary.get("built")):
        raise RuntimeError(
            f"Channel template enabled but XS stage did not build a valid template. See {summary_path}"
        )
    if dense_written <= 0:
        raise RuntimeError(
            f"Channel template enabled but dense guide-point export wrote 0 points. See {summary_path}"
        )
    dense_path_raw = summary.get("dense_points_path")
    if not dense_path_raw:
        raise RuntimeError(
            f"Channel template enabled and runtime summary reported dense points, but dense_points_path was empty. See {summary_path}"
        )
    dense_path = Path(str(dense_path_raw))
    if not dense_path.exists():
        raise RuntimeError(
            f"Channel template enabled and runtime summary reported dense points, but file does not exist: {dense_path}"
        )
    return dense_path


def _load_channel_template_json(cfg, work_dir: Path, river_dir: Path | None = None) -> tuple[Path, dict]:
    artifacts = _resolve_channel_template_artifacts(cfg, work_dir=work_dir, river_dir=river_dir)
    template_path = artifacts["template_json"]
    template = _load_json_file(template_path, label="channel-template JSON")
    return template_path, template


def _enforce_template_disable_runtime_contract(cfg, rc: int, out: str | None, err: str | None) -> tuple[int, str | None]:
    """Ensure an explicit template-disable request is honored by the XS child runtime."""
    if bool(getattr(cfg, "river_channel_template_enabled", False)):
        return rc, err
    combined = "\n".join(part for part in (out, err) if part)
    if "Channel template enabled" in combined:
        msg = "Explicit template disable request was violated: XS child reported channel template enabled."
        err = f"{err}\n{msg}" if err else msg
        return (rc if rc != 0 else 97), err
    if rc == 0 and "Channel template state: disabled by explicit user request" not in combined:
        msg = "Explicit template disable request was not acknowledged by XS child runtime."
        err = f"{err}\n{msg}" if err else msg
        return 98, err
    return rc, err

def _append_channel_template_args(cmd: list[str], cfg, river_dir: Path) -> Path | None:
    if not bool(getattr(cfg, "river_channel_template_enabled", False)):
        cmd.append("--no-channel-template")
        log.info("[RIVER] Forwarding explicit channel-template disable to XS child.")
        return None
    out_dir = _channel_template_runtime_dir(Path(river_dir))
    cmd.append("--channel-template-enabled")
    cmd.append(f"--channel-template-out-dir={out_dir}")
    cmd.append(f"--channel-template-min-xs={cfg.river_channel_template_min_xs}")
    cmd.append(f"--channel-template-fit-min-xs={cfg.river_channel_template_fit_min_xs}")
    cmd.append(f"--channel-template-n-bins={cfg.river_channel_template_n_bins}")
    cmd.append(f"--channel-template-min-depth-m={cfg.river_channel_template_min_depth_m}")
    cmd.append(f"--channel-template-distance-sigma-m={cfg.river_channel_template_distance_sigma_m}")
    cmd.append(f"--channel-template-estuary-buffer-m={cfg.river_channel_template_estuary_buffer_m}")
    cmd.append(f"--channel-template-junction-buffer-m={cfg.river_channel_template_junction_buffer_m}")
    ratio_max = getattr(cfg, "river_channel_template_width_depth_ratio_max", None)
    if ratio_max is not None:
        cmd.append(f"--channel-template-width-depth-ratio-max={ratio_max}")
    cmd.append(f"--channel-template-loo-max-rmse-norm={cfg.river_channel_template_loo_max_rmse_norm}")
    cmd.append(f"--channel-template-loo-max-dmax-error-m={cfg.river_channel_template_loo_max_dmax_error_m}")
    return out_dir

def _resolve_channel_template_artifacts(cfg, work_dir: Path, river_dir: Path | None = None) -> dict[str, Path]:
    candidate_dirs = _channel_template_candidate_dirs(cfg, work_dir=work_dir, river_dir=river_dir)
    selected_dir: Path | None = None
    fallback_dir: Path | None = None
    for template_dir in candidate_dirs:
        summary_cand = template_dir / "channel_template_runtime_summary.json"
        template_cand = template_dir / "channel_template.json"
        diagnostics_cand = template_dir / "channel_template_profile_diagnostics.csv"
        filter_cand = template_dir / "channel_template_filter.json"
        if summary_cand.exists():
            selected_dir = template_dir
            break
        if fallback_dir is None and (
            template_cand.exists() or diagnostics_cand.exists() or filter_cand.exists()
        ):
            fallback_dir = template_dir
    primary_dir = _channel_template_primary_dir(river_dir, work_dir)
    bundle_dir = selected_dir or fallback_dir or primary_dir
    return {
        "dir": bundle_dir,
        "summary": bundle_dir / "channel_template_runtime_summary.json",
        "dense_points": bundle_dir / "channel_template_dense_points.gpkg",
        "template_json": bundle_dir / "channel_template.json",
        "diagnostics": bundle_dir / "channel_template_profile_diagnostics.csv",
        "filter": bundle_dir / "channel_template_filter.json",
    }


def _channel_template_output_entries(cfg, *, dense_points_path: Path | None, runtime_summary_path: Path | None) -> dict[str, str | None]:
    if not bool(getattr(cfg, "river_channel_template_enabled", False)):
        return {
            "channel_template_dense_points": None,
            "channel_template_runtime_summary": None,
        }
    return {
        "channel_template_dense_points": str(dense_points_path) if dense_points_path is not None and Path(dense_points_path).exists() else None,
        "channel_template_runtime_summary": str(runtime_summary_path) if runtime_summary_path is not None and Path(runtime_summary_path).exists() else None,
    }
def _record_authoritative_child_passthrough(report: Dict[str, Any], stage: str, cmd: List[str]) -> None:
    """Thin wrapper around authoritative_cli helper for child-process receipts."""
    _record_authoritative_child_passthrough_impl(report, stage, cmd)


def _write_final_reporting_bundle(cfg: "BathyConfig", report: Dict[str, Any], *, final_native: Optional[Path], final_for_user: Optional[Path | str], final_provenance: Optional[Path | str]) -> Path:
    """Build the final report bundle from canonical execution state."""
    _ensure_bundle_a_route_contract_state(report)
    apply_parallel_method_guidance_summary(report=report, out_root=cfg.out_dir)
    state = build_final_run_state(
        cfg=cfg,
        report=report,
        final_native=final_native,
        final_for_user=final_for_user,
        final_provenance=final_provenance,
    )
    report_path = _write_final_run_reporting_bundle(
        state,
        write_authoritative_cache_receipt=_write_authoritative_cache_receipt,
        write_guidance_manifest=_write_guidance_manifest,
        write_explicit_final_outputs_manifest=_write_explicit_final_outputs_manifest,
        write_support_provenance_summary=_write_support_provenance_summary,
        write_final_support_regime_audit=_write_final_support_regime_audit,
        write_final_dem_selection_receipt=_write_final_dem_selection_receipt,
        write_comparison_package=_write_comparison_package,
        logger=log,
    )
    _fr_write_validation_invariance_summary(
        cfg,
        report,
        final_native=final_native,
        final_for_user=final_for_user,
        final_provenance=final_provenance,
        logger=log,
        enforce_hard_fail=False,
    )
    return report_path


def _river_domain_artifact_paths(work_dir: Path) -> Dict[str, Path]:
    return {
        "policy_json": Path(work_dir) / "river_domain_policy.json",
        "effective_water_mask": Path(work_dir) / "river_effective_water_mask.tif",
        "corridor_mask": Path(work_dir) / "river_corridor_mask_debug.tif",
        "nhdarea_mask": Path(work_dir) / "river_nhdarea_mask_debug.tif",
    }


def _append_river_domain_policy_args(cmd: List[str], work_dir: Path) -> Dict[str, Path]:
    paths = _river_domain_artifact_paths(work_dir)
    cmd.extend([
        f"--out-policy-json={paths['policy_json']}",
        f"--out-effective-water-mask={paths['effective_water_mask']}",
        f"--out-corridor-mask={paths['corridor_mask']}",
        f"--out-nhdarea-mask={paths['nhdarea_mask']}",
    ])
    return paths


def _record_river_domain_policy(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    *,
    work_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log_local = logger or log
    paths = _river_domain_artifact_paths(work_dir)
    summary = _load_river_domain_summary(paths["policy_json"])
    validation = _evaluate_river_domain_summary(
        summary,
        min_effective_water_corridor_overlap_frac=float(getattr(cfg, "river_domain_min_water_corridor_frac", 0.02) or 0.02),
        min_channel_corridor_overlap_frac=float(getattr(cfg, "river_domain_min_channel_corridor_frac", 0.001) or 0.001),
        min_channel_pixels=int(getattr(cfg, "river_domain_min_channel_pixels", 1) or 1),
    ) if summary else {"ok": None, "reason": "missing river_domain_policy summary"}
    river = report.setdefault("river", {})
    river["domain_policy"] = summary
    river["domain_validation"] = validation
    outputs = river.setdefault("outputs", {})
    outputs["river_domain_policy_json"] = str(paths["policy_json"]) if paths["policy_json"].exists() else None
    outputs["river_effective_water_mask"] = str(paths["effective_water_mask"]) if paths["effective_water_mask"].exists() else None
    outputs["river_corridor_mask_debug"] = str(paths["corridor_mask"]) if paths["corridor_mask"].exists() else None
    outputs["river_nhdarea_mask_debug"] = str(paths["nhdarea_mask"]) if paths["nhdarea_mask"].exists() else None
    if validation.get("ok") is False:
        msg = "; ".join(
            f"{f.get('check')}={f.get('value', f.get('effective', 'fail'))}"
            for f in validation.get("failures", [])
        )
        log_local.warning("[RIVER] River-domain validation failed: %s", msg or validation)
        if bool(getattr(cfg, "river_domain_hard_fail", False)) or bool(getattr(cfg, "strict", False)):
            raise RuntimeError(f"River-domain validation failed: {msg or validation}")
    return validation


def _materialize_authoritative_base_if_requested(cfg: "BathyConfig", args: argparse.Namespace) -> Optional[Path]:
    """Thin wrapper around authoritative_materialization coordinator."""
    return _resolve_authoritative_base(cfg, args, logger=log)


def _choose_waffles_mask_for_river(cfg: "BathyConfig", report: Dict[str, Any]) -> Optional[Path]:
    return _rm_choose_waffles_mask_for_river(cfg, report, logger=log)

def _count_mask_water_pixels(mask_tif: Path, aoi_bounds_wgs84: Tuple[float, float, float, float], *, water_max: float = 0.5) -> Optional[int]:
    return _rm_count_mask_water_pixels(mask_tif, aoi_bounds_wgs84, water_max=water_max)


def _count_mask_value_pixels(mask_tif: Path, *, value: int) -> Optional[int]:
    from activation_truth_utils import count_mask_value_pixels

    pixels = count_mask_value_pixels(mask_tif, value=value)
    if pixels is None:
        log.debug("Failed to count mask value pixels for %s value=%s", mask_tif, value)
    return pixels


def _stage_cached_waffles_mask(src: Path, dst: Path, log: Optional[logging.Logger]=None) -> Path:
    return _rm_stage_cached_waffles_mask(src, dst, logger=log)


def _build_canonical_with_nhd_mask(
    ocean_mask_tif: Path,
    river_gpkg: Path,
    out_tif: Path,
    log: Optional[logging.Logger] = None,
) -> Path:
    """Build the canonical full water-domain mask used by the river workflow.

    Semantics are fixed and explicit:
    - water = 0
    - land = 1

    The result is not a raw WAFFLES product. It is the deterministic union of:
    - ocean/coastal water from waffles_coastline_ocean_only
    - inland river/water geometry from the extracted river network GeoPackage

    This avoids relying on ambiguous/stale `want_nhd=true` WAFFLES outputs.
    """
    logger = log or logging.getLogger("bathy_main")
    import json
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from scipy import ndimage as ndi
    import fiona
    import geopandas as gpd

    ocean_mask_tif = Path(ocean_mask_tif)
    river_gpkg = Path(river_gpkg)
    out_tif = Path(out_tif)
    if not ocean_mask_tif.exists():
        raise RuntimeError(f"Ocean-only WAFFLES mask missing: {ocean_mask_tif}")
    if not river_gpkg.exists():
        raise RuntimeError(f"River network GeoPackage missing: {river_gpkg}")

    with rasterio.open(ocean_mask_tif) as ds:
        ocean = ds.read(1)
        profile = ds.profile.copy()
        transform = ds.transform
        crs = ds.crs
        shape = (ds.height, ds.width)
    ocean_water = (ocean == 0)

    available_layers = set(fiona.listlayers(str(river_gpkg)))
    # Full-water masks must be built from true polygonal water-area layers only.
    # River linework is scaffold geometry, not water area, and rasterizing it into
    # the water mask pollutes river/open-water/estuary domain construction.
    candidate_layers = ["nhdarea_clip"]
    shapes = []
    nonempty_layers = []
    half_cell = 0.5 * max(abs(float(transform.a)), abs(float(transform.e)))
    for layer in candidate_layers:
        if layer not in available_layers:
            continue
        gdf = gpd.read_file(river_gpkg, layer=layer)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise RuntimeError(f"River network layer {layer} has no CRS: {river_gpkg}")
        if str(gdf.crs) != str(crs):
            gdf = gdf.to_crs(crs)
        gdf = gdf[gdf.geometry.notnull()]
        if gdf.empty:
            continue
        poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        gdf = gdf.loc[poly_mask].copy()
        if gdf.empty:
            continue
        # Close vector-to-raster coastal seams deterministically by buffering the
        # inland-water polygons outward by half a cell before rasterization.
        # This is narrowly targeted at the canonical ocean/inland handoff and is
        # safer than globally dilating the ocean mask.
        if np.isfinite(half_cell) and half_cell > 0.0:
            keep_index = []
            buffered_geoms = []
            for idx, geom in zip(gdf.index, gdf.geometry):
                if geom is None or geom.is_empty:
                    continue
                try:
                    bg = geom.buffer(float(half_cell), cap_style=1, join_style=2)
                except Exception:
                    log.debug("bathy_main: suppressed exception", exc_info=True)
                    bg = geom
                if bg is not None and not bg.is_empty:
                    keep_index.append(idx)
                    buffered_geoms.append(bg)
            gdf = gdf.loc[keep_index].copy()
            gdf["geometry"] = buffered_geoms
            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
        if gdf.empty:
            continue
        nonempty_layers.append(layer)
        shapes.extend((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    if not shapes:
        raise RuntimeError(
            "No usable polygonal inland-water layer found in "
            f"{river_gpkg}. Required layer: nhdarea_clip; available layers: {sorted(available_layers)}"
        )

    inland = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    # If rasterization still leaves a one-cell moat/diagonal-only coastal seam,
    # close it in a tightly constrained way by adding only bridge cells that are
    # orthogonally adjacent to both ocean and inland water.
    rook = ndi.generate_binary_structure(2, 1)
    queen = ndi.generate_binary_structure(2, 2)
    ocean_edge = ndi.binary_dilation(ocean_water, structure=rook) & ~ocean_water
    inland_edge = ndi.binary_dilation(inland, structure=rook) & ~inland
    bridge_cells = ocean_edge & inland_edge
    direct_touch_pixels = int(np.count_nonzero(ndi.binary_dilation(ocean_water, structure=rook) & inland))
    diagonal_touch_pixels = int(np.count_nonzero(ndi.binary_dilation(ocean_water, structure=queen) & inland))
    bridge_pixel_count = 0
    if direct_touch_pixels == 0 and diagonal_touch_pixels > 0:
        inland = inland | bridge_cells
        bridge_pixel_count = int(np.count_nonzero(bridge_cells))
        direct_touch_pixels = int(np.count_nonzero(ndi.binary_dilation(ocean_water, structure=rook) & inland))

    full_water = ocean_water | inland
    added_pixels = int(np.count_nonzero(full_water & ~ocean_water))
    if added_pixels == 0:
        raise RuntimeError(
            f"Canonical with-NHD mask would add zero inland-water pixels beyond ocean-only mask. Layers used: {nonempty_layers}"
        )
    out = np.where(full_water, 0, 1).astype("uint8")
    profile.update(driver="GTiff", dtype="uint8", count=1, compress="DEFLATE", nodata=None)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    if int(profile.get("width", 0)) >= 16 and int(profile.get("height", 0)) >= 16:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    else:
        profile.update(tiled=False)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out, 1)

    try:
        diagnostics = {
            "ocean_only_mask": str(ocean_mask_tif),
            "river_gpkg": str(river_gpkg),
            "layers_used": nonempty_layers,
            "half_cell_buffer": float(half_cell),
            "added_inland_pixels": int(added_pixels),
            "direct_touch_pixels": int(direct_touch_pixels),
            "diagonal_touch_pixels": int(diagonal_touch_pixels),
            "bridge_pixel_count": int(bridge_pixel_count),
        }
        diag_json = out_tif.with_name(out_tif.stem + "_diagnostics.json")
        diag_json.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    except Exception:
        logger.debug("[WAFFLES] Failed to write diagnostics JSON", exc_info=True)

    logger.info(
        "[WAFFLES] Built canonical with-NHD mask from ocean-only + river network: %s (added_inland_pixels=%d direct_touch=%d diagonal_touch=%d bridge_pixels=%d layers=%s)",
        out_tif,
        added_pixels,
        direct_touch_pixels,
        diagonal_touch_pixels,
        bridge_pixel_count,
        nonempty_layers,
    )
    return out_tif


def _find_latest_waffles_mask(cache_root: Path) -> Optional[Path]:
    return _rm_find_latest_waffles_mask(cache_root)

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
            except OSError:
                log.debug("ignored", exc_info=True)

    if (not force) and out_tif_path.exists() and out_tif_path.stat().st_size > 0:
        if log:
            logger.info("[WAFFLES] Cache hit: %s", out_tif_path)
        return out_tif_path

    # Build WAFFLES command deterministically.
    # WAFFLES CLI uses -M module strings, not long-form flags (e.g., coastline:want_nhd=true).
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
def _waffles_water_fraction(mask_tif: Path, max_stride: int = 8) -> float:
    return _rm_waffles_water_fraction(mask_tif, max_stride=max_stride, logger=log)

def _determine_effective_methods_from_waffles(cfg: "BathyConfig", report: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    return _rm_determine_effective_methods_from_waffles(cfg, report, logger=logging.getLogger("bathy_main"))


def _determine_effective_methods_from_domains(cfg: "BathyConfig", report: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    effective, meta, truths = determine_method_activation_truth(cfg, report)
    log = logging.getLogger("bathy_main")
    activation_mismatch = meta.get("activation_mismatch") or {}
    if activation_mismatch:
        log.warning("[DOMAIN] Shared-domain activation summary disagreed with validated execution masks; using validated mask counts instead. mismatches=%s", activation_mismatch)
    report.setdefault("guidance_domains", {}).setdefault("activation", meta)
    report["method_activation_truth"] = {name: truth.to_dict() for name, truth in truths.items()}
    return effective, meta


def _validate_shared_domains_for_run(cfg: "BathyConfig", report: Dict[str, Any]) -> Dict[str, Any]:
    validation_path = getattr(cfg, "domain_review_validation", None)
    if validation_path is None:
        validation_path = getattr(cfg, "domain_validation_json", None)
    if validation_path is None:
        raise RuntimeError("shared_domain_validation_missing")
    validation_file = Path(validation_path)
    if not validation_file.exists():
        raise RuntimeError(f"shared_domain_validation_missing: path={validation_file}")
    try:
        with validation_file.open("r", encoding="utf-8") as f:
            validation = json.load(f) or {}
    except Exception as exc:
        raise RuntimeError(f"shared_domain_validation_read_failed: path={validation_file} error={exc}") from exc
    if not isinstance(validation, dict):
        raise RuntimeError(f"shared_domain_validation_invalid_type: path={validation_file} type={type(validation).__name__}")
    ok = bool(validation.get("ok", False))
    report.setdefault("guidance_domains", {})["validation"] = {
        "path": str(validation_path) if validation_path else None,
        "ok": bool(ok),
        "checks": validation.get("checks", []) if isinstance(validation, dict) else [],
    }
    if not ok:
        raise RuntimeError(
            f"shared_domain_validation_failed: validation_path={validation_path} "
            f"checks_failed={[c.get('name') for c in (validation.get('checks', []) if isinstance(validation, dict) else []) if not c.get('ok', False)]}"
        )
    return validation

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
                except (OSError, RuntimeError, ValueError, TypeError):
                    continue
    except (OSError, RuntimeError, ValueError) as exc:
        log.debug("Final domain policy step ignored: %s", exc, exc_info=True)

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






def _prepare_authoritative_guidance_inputs(cfg: "BathyConfig", report: Dict[str, Any]) -> None:
    _prepare_sdb_guidance_from_authoritative(
        cfg=cfg,
        report=report,
        ensure_dir_fn=ensure_dir,
        hash_key_fn=_hash_key,
        prepare_points_fn=_prepare_authoritative_sdb_training_points,
        build_sdb_authoritative_support_products_fn=_build_sdb_authoritative_support_products,
        logger=log,
    )
    outputs = report.setdefault("outputs", {})
    if getattr(cfg, "sdb_authoritative_support_mask", None):
        outputs["sdb_authoritative_support_mask"] = str(cfg.sdb_authoritative_support_mask)
    if getattr(cfg, "sdb_authoritative_support_values", None):
        outputs["sdb_authoritative_support_values"] = str(cfg.sdb_authoritative_support_values)
    if getattr(cfg, "sdb_authoritative_support_points", None):
        outputs["sdb_authoritative_support_points"] = str(cfg.sdb_authoritative_support_points)
    if getattr(cfg, "sdb_authoritative_support_contract", None):
        outputs["sdb_authoritative_support_contract"] = str(cfg.sdb_authoritative_support_contract)


def _prepare_authoritative_river_soundings(cfg: "BathyConfig", report: Dict[str, Any]) -> None:
    _prepare_river_support_from_authoritative(
        cfg=cfg,
        report=report,
        ensure_dir_fn=ensure_dir,
        hash_key_fn=_hash_key,
        prepare_points_fn=_prepare_authoritative_river_soundings_points,
        logger=log,
    )
    outputs = report.setdefault("outputs", {})
    if getattr(cfg, "river_authoritative_soundings", None):
        outputs["river_authoritative_soundings"] = str(cfg.river_authoritative_soundings)
    if getattr(cfg, "river_authoritative_support_contract", None):
        outputs["river_authoritative_support_contract"] = str(cfg.river_authoritative_support_contract)

def _build_tnm_river_dem(cfg: 'BathyConfig', cache_root: Path, report: Dict[str, Any]) -> Optional[Path]:
    """Build the TNM-based river DEM in working CRS for the current AOI."""
    fetches_exe = shutil.which("fetches")
    if fetches_exe is None:
        raise RuntimeError('fetches_not_found')
    cache_root = ensure_dir(cache_root)
    source_dir = ensure_dir(cache_root / "source")
    cmd = [fetches_exe, f'-R={cfg.aoi}', 'tnm:datasets=3']
    rc, out, err = run_command(cmd, cwd=source_dir, stream_stdout=False, stream_stderr=False)
    if rc != 0:
        raise RuntimeError(f'TNM fetch failed (rc={rc}): {(err or out or "").strip()[:500]}')
    tif_candidates = sorted(source_dir.rglob('*.tif'))
    tif_candidates = [p for p in tif_candidates if p.is_file() and p.stat().st_size > 0]
    if not tif_candidates:
        report.setdefault('river', {}).setdefault('dem_auto', {}).update({
            'tnm_source_dir': str(source_dir),
            'tnm_selected_tiles': [],
            'tnm_status': 'no_tiles_returned',
        })
        return None
    latest_by_stem = {}
    for p in tif_candidates:
        latest_by_stem[p.name] = p
    selected = list(latest_by_stem.values())
    vrt = cache_root / f"tnm_river_dem_{_hash_key(cfg.aoi, cfg.working_srs, cfg.river_dem_res_m)}.vrt"
    out_dem = cache_root / f"tnm_river_dem_{_hash_key(cfg.aoi, cfg.working_srs, cfg.river_dem_res_m)}.tif"
    vrt_cmd = ['gdalbuildvrt', str(vrt)] + [str(p) for p in selected]
    rc, out, err = run_command(vrt_cmd, stream_stdout=False, stream_stderr=False)
    if rc != 0 or not vrt.exists():
        raise RuntimeError(f'gdalbuildvrt failed for TNM river DEM (rc={rc}): {(err or out or "").strip()[:500]}')
    warp_cmd = [
        'gdalwarp', '-overwrite', '-t_srs', str(cfg.working_srs), '-r', 'bilinear',
        '-tr', str(float(cfg.river_dem_res_m)), str(float(cfg.river_dem_res_m)),
        '-te_srs', 'EPSG:4326', '-te', *[str(v) for v in str(cfg.aoi).split('/')],
        '-of', 'GTiff', '-co', 'COMPRESS=DEFLATE', '-co', 'TILED=YES',
        str(vrt), str(out_dem),
    ]
    rc, out, err = run_command(warp_cmd, stream_stdout=False, stream_stderr=False)
    if rc != 0 or not out_dem.exists():
        raise RuntimeError(f'gdalwarp failed for TNM river DEM (rc={rc}): {(err or out or "").strip()[:500]}')
    validate_gdal_output(out_dem, operation='river_dem_tnm_build', expected_crs=cfg.working_srs, expected_nodata=-9999.0, source_path=vrt, min_allowed=-1000.0, max_allowed=1000.0)
    report.setdefault('river', {}).setdefault('dem_auto', {}).update({
        'tnm_source_dir': str(source_dir),
        'tnm_selected_tiles': [str(p) for p in selected],
        'tnm_vrt': str(vrt),
    })
    return out_dem



def _estimate_raster_pixel_size_m_for_dst_crs(raster_path: Path | str, dst_crs: str) -> float:
    """Estimate the source raster pixel size in meters expressed in the destination CRS.

    Uses the center pixel and its immediate neighbors so the derived size reflects the
    authoritative grid rather than a separate hardcoded default.
    """
    import rasterio
    from pyproj import CRS, Transformer

    rp = Path(raster_path)
    with rasterio.open(rp) as ds:
        if ds.crs is None:
            raise RuntimeError(f"Raster has no CRS for pixel-size derivation: {rp}")
        t = ds.transform
        cx = max(0.0, (float(ds.width) - 1.0) / 2.0)
        cy = max(0.0, (float(ds.height) - 1.0) / 2.0)
        x0, y0 = t * (cx, cy)
        x1, y1 = t * (cx + 1.0, cy)
        x2, y2 = t * (cx, cy + 1.0)
        if CRS.from_user_input(ds.crs) == CRS.from_user_input(dst_crs):
            dx = float(np.hypot(x1 - x0, y1 - y0))
            dy = float(np.hypot(x2 - x0, y2 - y0))
        else:
            transformer = Transformer.from_crs(ds.crs, dst_crs, always_xy=True)
            tx0, ty0 = transformer.transform(x0, y0)
            tx1, ty1 = transformer.transform(x1, y1)
            tx2, ty2 = transformer.transform(x2, y2)
            dx = float(np.hypot(tx1 - tx0, ty1 - ty0))
            dy = float(np.hypot(tx2 - tx0, ty2 - ty0))
    vals = [v for v in (dx, dy) if np.isfinite(v) and v > 0]
    if not vals:
        raise RuntimeError(f"Could not derive positive pixel size for {rp} in {dst_crs}")
    return float(sum(vals) / len(vals))


def _resolve_river_dem_resolution_m(cfg: 'BathyConfig', working_srs: str, report: Optional[Dict[str, Any]] = None) -> float:
    requested = float(getattr(cfg, 'river_dem_res_m', 0.0) or 0.0)
    source = 'requested'
    resolved = requested
    if not (resolved > 0):
        auth_src = getattr(cfg, 'authoritative_base', None)
        if auth_src and Path(auth_src).exists():
            resolved = _estimate_raster_pixel_size_m_for_dst_crs(Path(auth_src), working_srs)
            source = 'authoritative_base_grid'
            log.info('[RIVER][DEM] Derived river DEM resolution %.6f m from authoritative_base grid: %s', resolved, auth_src)
        else:
            resolved = 10.0
            source = 'legacy_fallback_10m'
            log.warning('[RIVER][DEM] No authoritative_base available for river DEM grid derivation; falling back to %.3f m.', resolved)
    if not (np.isfinite(resolved) and resolved > 0):
        raise RuntimeError(f'Invalid resolved river DEM resolution: requested={requested!r} resolved={resolved!r}')
    cfg.river_dem_res_m = float(resolved)
    if report is not None:
        report.setdefault('river', {}).setdefault('dem_auto', {}).update({
            'river_dem_res_m_requested': requested,
            'river_dem_res_m_resolved': float(resolved),
            'river_dem_res_m_source': source,
        })
    return float(resolved)

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

    auth_src = getattr(cfg, 'authoritative_base', None)
    if auth_src and Path(auth_src).exists():
        working_srs = detect_working_srs(cfg)
        cfg.working_srs = working_srs
        river_dem_res_m = _resolve_river_dem_resolution_m(cfg, working_srs, report)
        dem_cache = ensure_dir(Path(cfg.cache_root) / "river_dem")
        baseline_auth = _resolve_baseline_cudem_interpolation(cfg, report)
        baseline_auth_str = str(baseline_auth) if baseline_auth is not None and Path(baseline_auth).exists() else None
        cache_key = _hash_key(cfg.aoi, working_srs, river_dem_res_m, str(auth_src), baseline_auth_str or '', 'authoritative_only_v4_with_baseline_gapfill')
        out_dem = dem_cache / f"river_dem_authoritative_{cache_key}.tif"
        receipt_path = dem_cache / f"river_dem_authoritative_{cache_key}.receipt.json"
        report.setdefault('river', {}).setdefault('dem_auto', {})
        if out_dem.exists() and out_dem.stat().st_size > 0:
            ok, reason, _ = cached_raster_semantics_valid(
                out_dem,
                expected_crs=working_srs,
                expected_nodata=-9999.0,
                expected_dtype='float32',
                min_allowed=-1000.0,
                max_allowed=10000.0,
            )
            if ok:
                log.info("[RIVER][DEM] Using cached authoritative-only river DEM: %s", out_dem)
                receipt = _write_river_dem_receipt(
                    receipt_path=receipt_path,
                    source_class='authoritative_only',
                    cache_key=cache_key,
                    projected_raster=out_dem,
                    authoritative_base=Path(auth_src),
                )
                report['river']['dem_auto'].update({
                    'status': 'success',
                    'source': 'authoritative_base',
                    'authoritative_base': str(auth_src),
                    'baseline_cudem_interpolation': baseline_auth_str,
                    'projected_river_dem': str(out_dem),
            'resolved_res_m': float(river_dem_res_m),
                    'receipt': str(receipt_path),
                    'source_contract': 'authoritative_plus_baseline_bank_gapfill' if baseline_auth_str else 'authoritative_only',
                    'tnm_gapfill_used': False,
                    'tnm_gapfill_raster': None,
                    'projected_stats': receipt.get('projected_river_dem_stats', {}),
                })
                return out_dem
            log.warning('[RIVER][DEM] Cached authoritative-only river DEM failed semantic validation; rebuilding. %s', reason)
            try:
                out_dem.unlink()
            except Exception:
                log.debug('[RIVER][DEM] Failed to remove invalid cached river DEM before rebuild', exc_info=True)
        try:
            info = _build_projected_authoritative_raster(
                Path(auth_src),
                out_dem,
                aoi=str(cfg.aoi),
                dst_crs=working_srs,
                res_m=float(river_dem_res_m),
                fallback_raster=(Path(baseline_auth_str) if baseline_auth_str else None),
                logger=log,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            report['river']['dem_auto'].update({
                'status': 'failed',
                'source': 'authoritative_base',
                'authoritative_base': str(auth_src),
                'baseline_cudem_interpolation': baseline_auth_str,
                'source_contract': 'authoritative_plus_baseline_bank_gapfill' if baseline_auth_str else 'authoritative_only',
                'authoritative_projection_error': str(exc),
                'tnm_gapfill_used': False,
                'tnm_gapfill_raster': None,
            })
            raise RuntimeError(
                f"Authoritative base exists at {auth_src} but river DEM projection failed under the authoritative-only contract: {exc}"
            ) from exc
        receipt = _write_river_dem_receipt(
            receipt_path=receipt_path,
            source_class='authoritative_only',
            cache_key=cache_key,
            projected_raster=out_dem,
            authoritative_base=Path(auth_src),
        )
        report['river']['dem_auto'].update({
            'status': 'success',
            'source': 'authoritative_base',
            'authoritative_base': str(auth_src),
            'baseline_cudem_interpolation': baseline_auth_str,
            'projected_river_dem': str(out_dem),
            'resolved_res_m': float(river_dem_res_m),
            'finite_cells': int(info.get('finite_cells', 0)),
            'nodata_cells': int(info.get('nodata_cells', 0)),
            'fallback_filled_cells': int(info.get('fallback_filled_cells', 0)),
            'receipt': str(receipt_path),
            'source_contract': 'authoritative_plus_baseline_bank_gapfill' if baseline_auth_str else 'authoritative_only',
            'tnm_gapfill_used': False,
            'tnm_gapfill_raster': None,
            'projected_stats': receipt.get('projected_river_dem_stats', {}),
        })
        if baseline_auth_str:
            log.info('[RIVER][DEM] Using authoritative river DEM/template with baseline CUDEM bank gapfill: %s', out_dem)
        else:
            log.info('[RIVER][DEM] Using authoritative-only river DEM/template built from authoritative_base: %s', out_dem)
        return out_dem

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
    river_dem_res_m = _resolve_river_dem_resolution_m(cfg, working_srs, report)

    dem_cache = ensure_dir(Path(cfg.cache_root) / "river_dem")
    tnm_dir = ensure_dir(dem_cache / "tnm")
    manifest = {
        "aoi": cfg.aoi,
        "source": cfg.river_dem_source,
        "working_srs": working_srs,
        "res_m": river_dem_res_m,
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
    except (OSError, RuntimeError, ValueError, TypeError) as e:
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
    cache_key = _hash_key(cfg.aoi, working_srs, river_dem_res_m, 'sanitized_v4_authoritative_grid')
    out_dem = dem_cache / f"river_dem_tnm_{cache_key}.tif"
    receipt_path = dem_cache / f"river_dem_tnm_{cache_key}.receipt.json"
    if out_dem.exists() and out_dem.stat().st_size > 0:
        log.info("[RIVER][DEM] Using cached river DEM: %s", out_dem)
        receipt = _write_river_dem_receipt(receipt_path=receipt_path, source_class='tnm_fallback', cache_key=cache_key, projected_raster=out_dem)
        report['river']['dem_auto'].update({'status': 'success', 'receipt': str(receipt_path), 'source_contract': 'tnm_fallback', 'projected_stats': receipt.get('projected_river_dem_stats', {})})
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
    except (OSError, RuntimeError, ValueError) as e:
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
    except (ImportError, RuntimeError, ValueError, TypeError):
        minx = miny = maxx = maxy = None

    cmd_warp = [gdalwarp, "-overwrite", "-t_srs", working_srs, "-r", "bilinear",
                "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES"]
    # Force a sensible meter grid
    if river_dem_res_m and river_dem_res_m > 0:
        cmd_warp += ["-tr", str(river_dem_res_m), str(river_dem_res_m), "-tap"]
    if minx is not None:
        cmd_warp += ["-te", str(minx), str(miny), str(maxx), str(maxy)]
    cmd_warp += [str(vrt), str(out_dem)]

    try:
        log.info("[RIVER][DEM] Warp/clip: %s", " ".join(cmd_warp[:10]) + (" ..." if len(cmd_warp) > 10 else ""))
        run_cmd(cmd_warp, check=True)
        if out_dem.exists() and out_dem.stat().st_size > 0:
            validate_gdal_output(out_dem, operation='ensure_river_dem_auto_tnm_warp', expected_crs=working_srs, expected_nodata=-9999.0, source_path=vrt, min_allowed=-1000.0, max_allowed=1000.0)
            receipt = _write_river_dem_receipt(receipt_path=receipt_path, source_class='tnm_fallback', cache_key=cache_key, projected_raster=out_dem)
            report["river"]["dem_auto"].update({"status": "success", "receipt": str(receipt_path), "source_contract": "tnm_fallback", "projected_stats": receipt.get('projected_river_dem_stats', {})})
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
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
    *,
    log_prefix: str = "[XYZ][DLIM] ",
) -> Path:
    """Reproject an XYZ file from src_crs to dst_crs.
    
    Returns path to reprojected file (cached).
    Only reprojects X,Y coordinates; Z (depth) is unchanged.

    IMPORTANT: Some providers may emit geographic XYZ in (lat, lon, z) order.
    If aoi_bounds is provided (west,east,south,north in degrees) and src_crs is
    geographic, we run a deterministic bounds check to detect swapped axis order
    and swap XY prior to reprojection when strongly indicated.
    """
    from pyproj import Transformer, CRS
    import numpy as np
    import json as _json

    log = logging.getLogger(__name__)
    
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
    with open(xyz_path, "r", encoding="utf-8") as f:
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

    # --- Diagnostics + axis-order detection for geographic sources ---
    receipt: Dict[str, Any] = {
        "src_crs": src_h,
        "dst_crs": dst_h,
        "n": int(len(xs)),
        "aoi_bounds": list(aoi_bounds) if aoi_bounds else None,
    }
    try:
        receipt["src_xy_minmax"] = {
            "x_min": float(np.nanmin(xs)), "x_max": float(np.nanmax(xs)),
            "y_min": float(np.nanmin(ys)), "y_max": float(np.nanmax(ys)),
        }
    except (ValueError, TypeError, RuntimeError) as exc:
        log.debug("Ignored source XY min/max receipt update: %s", exc, exc_info=True)

    def _in_aoi_ratio(xv: np.ndarray, yv: np.ndarray) -> Optional[float]:
        if not aoi_bounds:
            return None
        w, e, s, n = aoi_bounds
        m = np.isfinite(xv) & np.isfinite(yv)
        if int(np.count_nonzero(m)) == 0:
            return None
        xv2, yv2 = xv[m], yv[m]
        inside = (xv2 >= w) & (xv2 <= e) & (yv2 >= s) & (yv2 <= n)
        return float(np.count_nonzero(inside)) / float(len(xv2))

    # Only attempt axis-order detection if src looks geographic and AOI bounds are provided.
    # Heuristic: degrees-like ranges.
    try:
        src_is_geo = bool(CRS.from_user_input(src_h).is_geographic)
    except (ValueError, TypeError) as exc:
        log.debug("Unable to determine whether source CRS is geographic: %s", exc)
        src_is_geo = False

    swapped = False
    if src_is_geo and aoi_bounds is not None:
        r_xy = _in_aoi_ratio(xs, ys)
        r_yx = _in_aoi_ratio(ys, xs)
        receipt["geo_aoi_ratio_xy"] = r_xy
        receipt["geo_aoi_ratio_yx"] = r_yx

        # Swap only when strongly indicated to avoid accidental flips.
        if (r_xy is not None) and (r_yx is not None):
            if (r_yx >= 0.95) and (r_xy <= 0.05):
                xs, ys = ys, xs
                swapped = True
                log.warning(
                    "%sDetected geographic XYZ axis order likely (lat,lon); swapping XY before reprojection for %s",
                    log_prefix,
                    str(xyz_path),
                )
    receipt["geo_axis_swap_applied"] = bool(swapped)
    
    # Use horizontal CRS for transformation (strip vertical)
    transformer = Transformer.from_crs(src_h, dst_h, always_xy=True)
    xs_out, ys_out = transformer.transform(xs, ys)

    try:
        receipt["dst_xy_minmax"] = {
            "x_min": float(np.nanmin(xs_out)), "x_max": float(np.nanmax(xs_out)),
            "y_min": float(np.nanmin(ys_out)), "y_max": float(np.nanmax(ys_out)),
        }
    except (TypeError, ValueError, RuntimeError):
        log.debug("ignored", exc_info=True)
    
    # Write output
    tmp_path = out_path.with_suffix(".xyz.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for i in range(len(xs_out)):
            extra_str = " ".join(extras[i]) if extras[i] else ""
            if extra_str:
                f.write(f"{xs_out[i]:.6f} {ys_out[i]:.6f} {zs[i]:.4f} {extra_str}\n")
            else:
                f.write(f"{xs_out[i]:.6f} {ys_out[i]:.6f} {zs[i]:.4f}\n")
    
    tmp_path.replace(out_path)

    # Write a small reprojection receipt (best-effort).
    try:
        receipt_path = out_path.with_suffix(".reproject_receipt.json")
        receipt_path.write_text(_json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        log.debug("ignored", exc_info=True)
    return out_path


def _parse_aoi_bounds_deg(aoi: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse AOI string 'w/e/s/n' into ``(W, E, S, N)`` float bounds (degrees).

    Delegates to :func:`pipeline.aoi.parse_aoi_wesn`.
    """
    from pipeline.aoi import parse_aoi_wesn
    return parse_aoi_wesn(aoi)


def fetch_cudem_soundings_via_dlim(
    *,
    aoi: str,
    sources: List[str],
    cache_root: Path,
    out_crs: str,
    source_vdatum: Optional[str] = None,
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
        "source_vdatum": source_vdatum,
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
    # dlim block_thin takes degrees; choose a value conservative in longitude (cos(lat)).
    out_crs_lower = str(out_crs).lower()
    is_geographic = any(x in out_crs_lower for x in ["4326", "4269", "4267"])
    thin_res_m_eff = thin_res_m
    if is_geographic and thin_res_m and thin_res_m > 0 and (filter_spec is None or str(filter_spec).strip() == ""):
        try:
            w, e, s, n = [float(x) for x in str(aoi).split("/")]
            lat0 = 0.5 * (s + n)
        except (TypeError, ValueError):
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
            source_vdatum or "source_unspecified",
            filter_spec or (f"block_thin:res={thin_res_m_eff}" if thin_res_m_eff else "no_filter"),
        )
        out_xyz = xyz_cache / f"{src}_{key}.xyz"

        # Build dlim command
        # Basic: dlim -R=W/E/S/N <source>
        # With projection: dlim -R=W/E/S/N <source> -P epsg:XXXX
        # With filter: dlim -R=W/E/S/N <source> -F block_thin:res=10
        cmd = [dlim_exe, f'-R={aoi}', src]
        
        # Add explicit vertical source when provided. This is useful when the
        # provider's bathymetry should be treated as a known source datum (for
        # example NAVD88) and converted to the requested target compound CRS
        # (for example an MTL-based training datum) during fetch.
        if source_vdatum:
            src_vdatum = str(source_vdatum).lower()
            if not src_vdatum.startswith("epsg:"):
                src_vdatum = f"epsg:{src_vdatum}" if src_vdatum.isdigit() else src_vdatum
            cmd += ["-J", src_vdatum]

        # Add projection / target compound CRS if specified (dlim defaults to epsg:4326 output)
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

        except (OSError, RuntimeError, TypeError, ValueError) as e:
            try:
                if out_xyz.exists() and out_xyz.stat().st_size == 0:
                    out_xyz.unlink()
            except OSError:
                log.debug("ignored", exc_info=True)
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
        snd = cfg.river_soundings
        if not snd:
            return
        snd_list = _normalize_multi_path_value(snd)
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

        # Soundings XY must be in the river template CRS (cfg.river_dem), not the output CRS.
        # Using cfg.working_srs here can silently mis-project them outside the channel mask.
        # Prefer explicitly tracked soundings CRS if available; otherwise fall back to river DEM CRS
        snd_srs = str(cfg.river_soundings_crs or '').strip()
        if not snd_srs:
            try:
                import rasterio
                rd = cfg.river_dem
                if rd:
                    with rasterio.open(str(rd)) as ds:
                        if ds.crs:
                            snd_srs = str(ds.crs)
            except Exception:
                log.debug("_append_river_soundings_args: suppressed exception", exc_info=True)
                snd_srs = ''
        if snd_srs:
            cmd.append(f"--soundings-crs={snd_srs}")

        if include_calib_args:
            d = float(cfg.river_soundings_calib_max_dist_m or 0.0)
            if d > 0:
                cmd.append(f"--calib-max-dist-m={d}")
            st = str(cfg.river_soundings_calib_stat or '').strip()
            if st:
                cmd.append(f"--calib-stat={st}")

        if include_mode_args:
            # Skeleton-side sounding assimilation controls
            sm = str(cfg.river_soundings_mode or 'auto').strip()
            if sm:
                cmd.append(f"--soundings-mode={sm}")
            sp = float(cfg.river_soundings_cell_percentile or 0.0)
            if sp > 0:
                cmd.append(f"--soundings-cell-percentile={sp}")
            md = float(cfg.river_soundings_max_dist_m or 0.0)
            if md > 0:
                cmd.append(f"--soundings-max-dist-m={md}")
            mr = float(cfg.river_soundings_min_r or 0.0)
            if mr > 0:
                cmd.append(f"--soundings-min-r={mr}")
            if cfg.river_no_soundings_enforce:
                cmd.append('--no-soundings-enforce')
    except Exception:
        log.debug('Failed to append river soundings args; continuing.', exc_info=True)


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
        log.debug("ignored", exc_info=True)

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
            tags = ds.tags() or {}
            declared_semantics = raster_value_semantics(tags)
            expect_negative = should_expect_negative_depth(tags, default=expect_negative)
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

            if declared_semantics == 'absolute_elevation':
                return True, f"absolute_elevation(p50={float(p50):.3f}, p99={float(p99):.3f})"

            # Optical reflectance products are typically non-negative; depths should be mostly negative-down
            if expect_negative:
                frac_neg = float(np.mean(vals < 0))
                if frac_neg < 0.01:
                    # If everything is small non-negative (0..1 or 0..10000), it's almost certainly imagery/mask
                    if float(p1) >= -1e-6:
                        return False, f"too_few_negative(frac_neg={frac_neg:.3f}, p1={float(p1):.3f}, p99={float(p99):.3f})"

            return True, f"ok(p50={float(p50):.3f}, p99={float(p99):.3f})"
    except Exception as e:
        log.debug("bathy_main: suppressed exception", exc_info=True)
        return False, f"exception:{e}"

def _find_sdb_depth_raster_local(sdb_dir: Path) -> Optional[Path]:
    sdb_dir = Path(sdb_dir)
    if not sdb_dir.exists():
        return None
    manifest = sdb_dir / "artifacts_sdb.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    rel = data.get("sdb_guidance_active") or data.get("depth_raster")
    if not isinstance(rel, str) or not rel.strip():
        return None
    p = (sdb_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
    return p if p.exists() else None


def find_sdb_depth_raster(sdb_dir: Path) -> Optional[Path]:
    """Use the canonical resolver when available, otherwise fall back to the local manifest-only contract."""
    try:
        import process_utils
        canonical = getattr(process_utils, "find_sdb_depth_raster", None)
        if callable(canonical):
            return canonical(sdb_dir)
    except ImportError:
        pass
    return _find_sdb_depth_raster_local(sdb_dir)
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
        log.debug("Failed to read SDB artifact manifest for land_mask.", exc_info=True)
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
        log.debug("Failed to read SDB artifact manifest for WSE.", exc_info=True)
    return None


def _validate_soundings_subset(path: Path | None) -> Path | None:
    """Validate the soundings subset parquet strictly and return the path."""
    if path is None:
        return None
    info = validate_soundings_subset_parquet(path)
    log.info("[RIVER][SOUNDINGS] Subset parquet validated: n=%d rows depth_col=%s.", info["rows"], info["depth_col"])
    return path


def run_sdb(cfg: BathyConfig, report: Dict[str, Any]) -> Optional[Path]:
    log.info("SDB pipeline (coastal/nearshore)")

    plan = determine_sdb_execution_plan(
        cfg=cfg,
        report=report,
        ensure_dir_fn=ensure_dir,
        parse_aoi_bbox_fn=_parse_aoi_bbox,
        count_mask_water_pixels_fn=_count_mask_water_pixels,
        logger=log,
    )
    if not plan.should_run:
        return None

    sdb_dir = plan.sdb_dir
    script_dir = plan.script_dir

    cmd = _build_sdb_command(
        cfg,
        out_dir=str(sdb_dir),
        authoritative_passthrough_args=_authoritative_passthrough_args(cfg, for_river=False),
    )
    _record_authoritative_child_passthrough(report, "sdb", cmd)

    cmd = _augment_sdb_command(
        cfg,
        cmd,
        sdb_main_path=str(Path(__file__).parent / "sdb_main.py"),
        logger=log,
    )

    log.info("[SDB] Command: %s", cmd)
    logs_dir = plan.logs_dir
    rc, out, err = run_command(
        cmd,
        cwd=script_dir,
        prefix="[SDB] ",
        stdout_log_path=logs_dir / "sdb.stdout.log",
        stderr_log_path=logs_dir / "sdb.stderr.log",
    )

    sdb_report = report.setdefault("sdb", {})
    sdb_report.update({
        "status": "success" if rc == 0 else "failed",
        "returncode": rc,
        "command": cmd,
        "stdout_tail": out,
        "stderr_tail": err,
    })

    if rc != 0:
        log.error("[SDB] Failed with code %d", rc)
        return None

    depth = _finalize_sdb_run(Path(sdb_dir), report)
    try:
        _load_sdb_artifacts_into_report(sdb_dir=Path(sdb_dir), report=report)
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        log.debug("[SDB] Failed loading artifacts_sdb.json into report.", exc_info=True)
    return depth



def _ensure_river_network_artifact(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    *,
    script_dir: Path,
    cache_dir: Path,
    work_dir: Path,
    raw_hydro_cache: Path,
) -> Optional[Path]:
    """Ensure the canonical river network GeoPackage exists for this run.

    This is the earliest river-side artifact needed by both the river workflow and the
    shared guidance-domain planner. It intentionally owns only the network/scaffold step
    so downstream stages can build their domains before either SDB or river guidance runs.
    """
    report.setdefault("river", {})
    report["river"].setdefault("steps", {})

    network_gpkg = work_dir / "river_network.gpkg"
    prov_dir = Path(cfg.cache_root) / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    aoi_key = str(cfg.aoi)
    try:
        _parts = aoi_key.split('/')
        if len(_parts) >= 4:
            _vals = [float(_parts[0]), float(_parts[1]), float(_parts[2]), float(_parts[3])]
            aoi_key = '/'.join([f"{v:.6f}" for v in _vals])
    except (TypeError, ValueError):
        log.debug("ignored", exc_info=True)
    prov_key = {
        "aoi": aoi_key,
        "hydrography_source": str(cfg.river_hydrography_source),
        "tnm_dataset": str(cfg.tnm_dataset),
        "tnm_enable": bool(cfg.tnm_enable),
        "snap_m": float(cfg.snap_m),
        "river_da_raster": _fingerprint_path(cfg.river_da_raster),
        "river_da_raster_band": int(cfg.river_da_raster_band or 1),
        "river_da_raster_units": str(cfg.river_da_raster_units or "km2"),
        "river_network_halo_km": float(getattr(cfg, "river_network_halo_km", 0.0) or 0.0),
    }
    prov_hash = hashlib.sha1(json.dumps(prov_key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    river_network_lock = prov_dir / f"river_network_lock_{prov_hash}.json"

    scaffold_domains = get_river_aoi_domains(
        str(cfg.aoi),
        float(getattr(cfg, "river_network_halo_km", 0.0) or 0.0),
        float(getattr(cfg, "river_trusted_halo_m", 60.0) or 0.0),
    )
    report["river"].setdefault("scaffold_domains", scaffold_domains.as_dict())

    scaffold_cache = cached_scaffold_paths(
        cache_root=cfg.cache_root,
        domains=scaffold_domains,
        hydrography_source=str(cfg.river_hydrography_source),
        tnm_dataset=str(cfg.tnm_dataset),
        tnm_enable=bool(cfg.tnm_enable),
        snap_m=float(cfg.snap_m),
        da_raster_fingerprint=_fingerprint_path(cfg.river_da_raster),
        da_raster_band=int(cfg.river_da_raster_band or 1),
        da_raster_units=str(cfg.river_da_raster_units or "km2"),
    )
    scaffold_products = scaffold_product_paths(
        cache_root=cfg.cache_root,
        domains=scaffold_domains,
        hydrography_source=str(cfg.river_hydrography_source),
        tnm_dataset=str(cfg.tnm_dataset),
        tnm_enable=bool(cfg.tnm_enable),
        snap_m=float(cfg.snap_m),
        da_raster_fingerprint=_fingerprint_path(cfg.river_da_raster),
        da_raster_band=int(cfg.river_da_raster_band or 1),
        da_raster_units=str(cfg.river_da_raster_units or "km2"),
    )
    scaffold_cache_dir = Path(scaffold_cache["cache_dir"])
    scaffold_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_network_gpkg = Path(scaffold_cache["network_gpkg"])
    scaffold_manifest_path = Path(scaffold_cache["manifest"])
    network_manifest_path = Path(scaffold_cache["network_manifest"])
    hydrologic_domain_path = Path(scaffold_cache["hydrologic_solve_domain"])
    exported_network_manifest = work_dir / "river_network_manifest.json"
    exported_hydrologic_domain = work_dir / "hydrologic_solve_domain.json"
    report["river"]["scaffold_cache_dir"] = str(scaffold_cache_dir)

    cache_hit = scaffold_cache_hit(manifest_path=scaffold_manifest_path, expected_network_gpkg=cached_network_gpkg)
    cmd = None
    out = err = ""
    rc = 0
    if cache_hit:
        if network_gpkg.exists():
            network_gpkg.unlink()
        shutil.copy2(cached_network_gpkg, network_gpkg)
        if network_manifest_path.exists():
            if exported_network_manifest.exists():
                exported_network_manifest.unlink()
            shutil.copy2(network_manifest_path, exported_network_manifest)
        if hydrologic_domain_path.exists():
            if exported_hydrologic_domain.exists():
                exported_hydrologic_domain.unlink()
            shutil.copy2(hydrologic_domain_path, exported_hydrologic_domain)
        report["river"]["steps"]["network"] = {
            "status": "success",
            "returncode": 0,
            "command": "cache_hit",
            "stdout_tail": "",
            "stderr_tail": "",
            "cache_hit": True,
            "cached_network_gpkg": str(cached_network_gpkg),
        }
        log.info("[RIVER] Reusing cached canonical scaffold network: %s", cached_network_gpkg)
    else:
        cmd = [
            sys.executable, "river_network.py",
            f"--aoi={scaffold_domains.scaffold_aoi}",
            f"--export-aoi={scaffold_domains.export_aoi}",
            f"--solve-aoi={scaffold_domains.solve_aoi}",
            f"--scaffold-aoi={scaffold_domains.scaffold_aoi}",
            f"--solve-halo-km={float(getattr(cfg, 'river_network_halo_km', 0.0) or 0.0)}",
            f"--trusted-halo-m={float(getattr(cfg, 'river_trusted_halo_m', 60.0) or 0.0)}",
            f"--solve-domain-role=halo-expanded river solve domain used to reduce AOI-edge sensitivity",
            f"--export-domain-role=delivered tile/domain for downstream conditioning",
            f"--scaffold-domain-role=canonical scaffold domain used for stable river topology and guidance generation",
            f"--solve-domain-rationale=Dominant trunk is solved on the broader scaffold domain before clipping to the export tile so mainstem selection is more stable across AOIs and adjacent tiles.",
            f"--cache-dir={raw_hydro_cache}",
            f"--out-gpkg={cached_network_gpkg}",
            f"--manifest-json={network_manifest_path}",
            f"--provenance-lock={river_network_lock}",
            f"--hydrography-source={cfg.river_hydrography_source}",
            f"--tnm-dataset={cfg.tnm_dataset}",
            f"--snap-m={cfg.snap_m}",
        ]
        if cfg.tnm_enable:
            cmd.append("--tnm-enable")
        if cfg.river_da_raster:
            cmd.append(f"--da-raster={Path(cfg.river_da_raster)}")
            cmd.append(f"--da-raster-band={int(cfg.river_da_raster_band or 1)}")
            cmd.append(f"--da-raster-units={str(cfg.river_da_raster_units or 'km2')}")
        log.info("[RIVER][NETWORK] Command: %s", " ".join(str(c) for c in cmd))
        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["network"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
            "cache_hit": False,
            "cached_network_gpkg": str(cached_network_gpkg),
        }
        if rc == 0 and cached_network_gpkg.exists():
            if network_gpkg.exists():
                network_gpkg.unlink()
            shutil.copy2(cached_network_gpkg, network_gpkg)
            if network_manifest_path.exists():
                if exported_network_manifest.exists():
                    exported_network_manifest.unlink()
                shutil.copy2(network_manifest_path, exported_network_manifest)
            if hydrologic_domain_path.exists():
                if exported_hydrologic_domain.exists():
                    exported_hydrologic_domain.unlink()
                shutil.copy2(hydrologic_domain_path, exported_hydrologic_domain)

    scaffold_product_meta = {}
    scaffold_products_status = "missing"
    if ((cmd is None and cache_hit) or (rc == 0 and cached_network_gpkg.exists())) and cached_network_gpkg.exists():
        try:
            if scaffold_products_complete(scaffold_products):
                scaffold_products_status = "cache_hit"
                scaffold_product_meta = dict(scaffold_products)
                for _key in ("stationing_json", "summary_json"):
                    _path = Path(scaffold_products[_key])
                    if _path.exists():
                        try:
                            scaffold_product_meta[_key.replace("_json", "")] = json.loads(_path.read_text(encoding="utf-8"))
                        except (OSError, ValueError, TypeError):
                            pass
            else:
                scaffold_product_meta = persist_scaffold_products(network_gpkg=cached_network_gpkg, product_paths=scaffold_products)
                scaffold_products_status = "created" if scaffold_product_meta else "missing"
        except (ImportError, OSError, ValueError, TypeError, RuntimeError):
            log.debug("[RIVER] Failed to persist scaffold-derived products", exc_info=True)
            scaffold_product_meta = {}
            scaffold_products_status = "failed"

    try:
        write_scaffold_manifest(
            scaffold_manifest_path,
            domains=scaffold_domains,
            network_gpkg=str(cached_network_gpkg) if cached_network_gpkg.exists() else None,
            provenance_lock=str(river_network_lock),
            extra={
                "network_status": "success" if ((cmd is None and cache_hit) or (rc == 0 and cached_network_gpkg.exists())) else "failed",
                "cache_dir": str(scaffold_cache_dir),
                "cache_hit": bool(cache_hit),
                "export_network_gpkg": str(network_gpkg) if network_gpkg.exists() else None,
                "network_manifest": str(network_manifest_path) if network_manifest_path.exists() else None,
                "export_network_manifest": str(exported_network_manifest) if exported_network_manifest.exists() else None,
                "hydrologic_solve_domain": str(hydrologic_domain_path) if hydrologic_domain_path.exists() else None,
                "export_hydrologic_solve_domain": str(exported_hydrologic_domain) if exported_hydrologic_domain.exists() else None,
                "scaffold_products_status": scaffold_products_status,
                "scaffold_products": scaffold_product_meta,
            },
        )
        report["river"]["scaffold_manifest"] = str(scaffold_manifest_path)
        if network_manifest_path.exists():
            report["river"]["network_manifest"] = str(network_manifest_path)
            try:
                report["river"]["network_manifest_payload"] = json.loads(network_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                pass
        if hydrologic_domain_path.exists():
            report["river"]["hydrologic_solve_domain"] = str(hydrologic_domain_path)
            try:
                report["river"]["hydrologic_solve_domain_payload"] = json.loads(hydrologic_domain_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                pass
        if scaffold_product_meta:
            report["river"]["scaffold_products"] = scaffold_product_meta
    except (OSError, TypeError, ValueError):
        log.debug("[RIVER] Failed to write scaffold manifest", exc_info=True)

    if rc != 0 or not network_gpkg.exists():
        log.error("[RIVER] Failed to extract river network.")
        report["river"]["status"] = "failed"
        return None

    log.info("[RIVER] Network ready: %s", network_gpkg)
    return network_gpkg


def _prepare_guidance_domains_for_run(cfg: "BathyConfig", report: Dict[str, Any]) -> None:
    """Build the shared SDB and river guidance domains before either child stage runs."""
    methods = {str(m).strip().lower() for m in (cfg.methods or [])}
    if not ({"sdb", "river"} & methods):
        return
    report.setdefault("river", {})
    report.setdefault("guidance_domains", {})

    if cfg.river_dem is None:
        auto_dem = ensure_river_dem_auto(cfg, report)
        if auto_dem is None:
            raise RuntimeError("Missing --river-dem and auto-build failed before guidance-domain planning.")
        cfg.river_dem = auto_dem

    script_dir = Path(__file__).parent
    cache_dir = Path(cfg.derived_cache_root) / "river"
    work_dir = ensure_dir(cache_dir / "work")
    raw_hydro_cache = ensure_dir(Path(cfg.cache_root) / "hydrography")
    network_gpkg = _ensure_river_network_artifact(
        cfg,
        report,
        script_dir=script_dir,
        cache_dir=cache_dir,
        work_dir=work_dir,
        raw_hydro_cache=raw_hydro_cache,
    )
    if network_gpkg is None:
        if "river" in methods:
            raise RuntimeError("River network unavailable; cannot plan SDB/river guidance domains.")
        log.info("[DOMAIN] River network unavailable for this AOI; SDB will use its standard coastline domain.")
        return

    try:
        domains = _ensure_guidance_domains(
            cfg,
            river_dem=Path(cfg.river_dem),
            river_gpkg=Path(network_gpkg),
            derived_cache_root=Path(cfg.derived_cache_root),
            review_root=Path(cfg.out_dir),
            report=report,
            logger=log,
        )
    except Exception:
        if "river" in methods:
            raise
        log.debug("[DOMAIN] Shared guidance domains unavailable; continuing with standard SDB coastline domain.", exc_info=True)
        return
    cfg.waffles_ocean_mask = Path(domains.ocean_mask)
    cfg.waffles_with_nhd_mask = Path(domains.with_nhd_water_mask)
    cfg.water_domain_mask_for_final = Path(domains.with_nhd_water_mask)

    # Canonical shared-domain masks used by both river and SDB architecture.
    cfg.base_water_mask = Path(domains.with_nhd_water_mask)
    cfg.ocean_connectivity_mask = Path(domains.ocean_connectivity_mask)
    cfg.river_candidate_domain_mask = Path(domains.river_channel_mask)
    cfg.estuary_handoff_mask = Path(domains.estuary_clip_mask)
    cfg.river_active_domain_mask = Path(domains.river_guidance_domain_mask)
    cfg.sdb_candidate_domain_mask = Path(domains.review_dir) / "sdb_guidance_domain_mask.tif"
    cfg.domain_review_dir = Path(domains.review_dir).parent / "domains"

    # Use the review/shared-domain SDB mask as the canonical execution mask.
    # The derived-cache copy is kept for provenance, but run_sdb must trust the
    # same mask path that shared-domain activation used.
    cfg.sdb_guidance_domain_mask = cfg.sdb_candidate_domain_mask
    cfg.validated_sdb_guidance_domain_mask = cfg.sdb_guidance_domain_mask
    try:
        _aoi_bbox = tuple(float(v) for v in str(cfg.aoi).split("/")) if getattr(cfg, "aoi", None) else None
        cfg.validated_sdb_guidance_domain_pixels = _count_mask_water_pixels(cfg.validated_sdb_guidance_domain_mask, _aoi_bbox) if _aoi_bbox else None
    except Exception:
        log.debug("[DOMAIN] Failed to count SDB guidance domain pixels; defaulting to None.", exc_info=True)
        cfg.validated_sdb_guidance_domain_pixels = None
    cfg.river_channel_mask = Path(domains.river_channel_mask)
    cfg.river_guidance_domain_mask = Path(domains.river_guidance_domain_mask)
    cfg.river_domain_mask_for_fusion = Path(domains.river_guidance_domain_mask)
    cfg.estuary_clip_mask = Path(domains.estuary_clip_mask)
    cfg.estuary_transition_mask = Path(domains.estuary_transition_mask)
    cfg.guidance_domain_manifest = Path(domains.manifest_json)
    cfg.guidance_domain_review_manifest = Path(domains.review_dir) / "guidance_domains_manifest.json"
    cfg.domain_review_manifest = cfg.domain_review_dir / "domain_manifest.json"
    cfg.domain_review_summary = cfg.domain_review_dir / "domain_summary.json"
    cfg.domain_review_validation = cfg.domain_review_dir / "domain_validation.json"
    cfg.shared_domain_stage_ready = True

    outputs = report.setdefault("outputs", {})
    outputs["domain_manifest"] = str(cfg.domain_review_manifest)
    outputs["domain_summary"] = str(cfg.domain_review_summary)
    outputs["domain_validation"] = str(cfg.domain_review_validation)
    outputs["base_water_mask"] = str(cfg.base_water_mask)
    outputs["ocean_connectivity_mask"] = str(cfg.ocean_connectivity_mask)
    outputs["river_candidate_domain_mask"] = str(cfg.river_candidate_domain_mask)
    outputs["estuary_handoff_mask"] = str(cfg.estuary_handoff_mask)
    outputs["river_active_domain_mask"] = str(cfg.river_active_domain_mask)
    outputs["sdb_candidate_domain_mask"] = str(cfg.sdb_candidate_domain_mask)

    domain_stage = report.setdefault("shared_domain_stage", {})
    domain_stage.update({
        "ready": True,
        "review_dir": str(cfg.domain_review_dir),
        "base_water_mask": str(cfg.base_water_mask),
        "ocean_connectivity_mask": str(cfg.ocean_connectivity_mask),
        "river_candidate_domain_mask": str(cfg.river_candidate_domain_mask),
        "estuary_handoff_mask": str(cfg.estuary_handoff_mask),
        "river_active_domain_mask": str(cfg.river_active_domain_mask),
        "sdb_candidate_domain_mask": str(cfg.sdb_candidate_domain_mask),
        "summary_json": str(cfg.domain_review_summary),
        "validation_json": str(cfg.domain_review_validation),
    })
    try:
        with Path(cfg.domain_review_summary).open("r", encoding="utf-8") as _f:
            domain_stage["summary"] = json.load(_f)
    except Exception:
        log.debug("[DOMAIN] Unable to inline domain summary into report", exc_info=True)

    log.info("[DOMAIN] Pre-inference guidance domains ready: review_dir=%s", domains.review_dir)
    log.info("[DOMAIN] Review domain manifest: %s", cfg.domain_review_manifest)
    log.info("[DOMAIN] Review domain summary: %s", cfg.domain_review_summary)
    log.info("[DOMAIN] Review domain validation: %s", cfg.domain_review_validation)
    log.info("[DOMAIN] Review river domain: %s", domains.review_dir / "river_guidance_domain_mask.tif")
    log.info("[DOMAIN] Review SDB domain: %s", domains.review_dir / "sdb_guidance_domain_mask.tif")
    log.info("[DOMAIN] Review manifest: %s", domains.review_dir / "guidance_domains_manifest.json")


# -----------------------------------------------------------------------------
# River
# -----------------------------------------------------------------------------


def _gap_fill_bed_with_skeleton_prior(
    bed_tif: Path,
    dem_tif: Path,
    channel_mask_tif: Optional[Path],
    a: float = 0.18,
    b: float = 0.50,
    shape_exp: float = 0.5,
    dmin: float = 0.5,
    dmax: float = 15.0,
    nodata: float = -9999.0,
    logger: Optional[logging.Logger] = None,
    confidence_tif: Optional[Path] = None,
    confidence_sigma_m: float = 500.0,
) -> int:
    """Fill channel-mask pixels where XS inference left nodata.

    Uses a lightweight skeleton depth model:
      1. Bank-adjacent DEM samples → WSE (nearest-neighbor fill inward)
      2. Distance-to-bank → width proxy → Dmax via a * W^b
      3. Normalized bank distance r → depth = Dmax * r^shape_exp
      4. bed = WSE - depth

    Only fills pixels where the bed raster is nodata AND the channel mask is 1.

    If confidence_tif is provided, writes a companion [0,1] confidence raster:
      - XS-measured pixels → 1.0
      - Gap-filled pixels → Gaussian decay from nearest measured pixel
      - Outside channel → nodata

    Returns the number of pixels filled.
    """
    import rasterio
    from scipy.ndimage import distance_transform_edt, binary_dilation

    _log = logger or logging.getLogger(__name__)

    if channel_mask_tif is None or not Path(channel_mask_tif).exists():
        return 0
    if not Path(bed_tif).exists() or not Path(dem_tif).exists():
        return 0

    with rasterio.open(bed_tif) as bed_ds:
        bed = bed_ds.read(1).astype("float32")
        bed_profile = bed_ds.profile.copy()
        bed_nd = bed_ds.nodata
    with rasterio.open(dem_tif) as dem_ds:
        dem = dem_ds.read(1).astype("float32")
        dem_nd = dem_ds.nodata
    with rasterio.open(channel_mask_tif) as cm_ds:
        ch = cm_ds.read(1)

    channel = (ch == 1)
    bed_valid = np.isfinite(bed) & (bed != bed_nd if bed_nd is not None else np.ones_like(bed, dtype=bool))
    dem_valid = np.isfinite(dem) & (dem != dem_nd if dem_nd is not None else np.ones_like(dem, dtype=bool))

    # Gap pixels: inside channel but missing bed values
    gaps = channel & (~bed_valid)
    n_gaps = int(gaps.sum())
    if n_gaps == 0:
        return 0

    _log.info("[GAP-FILL] Channel pixels: %d, existing bed: %d, gaps: %d (%.1f%%)",
              int(channel.sum()), int((channel & bed_valid).sum()), n_gaps,
              100.0 * n_gaps / max(1, int(channel.sum())))

    # Pixel size (meters, approximate)
    tf = bed_profile.get("transform")
    if tf is None:
        return 0
    pix = float((abs(tf.a) + abs(tf.e)) / 2.0)

    # Distance to bank (EDT from channel boundary)
    d_bank = distance_transform_edt(channel, sampling=pix).astype("float32")

    # WSE from bank DEM samples
    non_channel = ~channel
    touches_bank = binary_dilation(non_channel, structure=np.ones((3, 3), dtype=bool))
    bank_pixels = channel & touches_bank & dem_valid
    if int(bank_pixels.sum()) < 5:
        _log.warning("[GAP-FILL] Insufficient bank DEM samples (%d); skipping.", int(bank_pixels.sum()))
        return 0

    wse_seed = np.full(channel.shape, np.nan, dtype="float32")
    wse_seed[bank_pixels] = dem[bank_pixels]

    # Fill WSE inward via nearest-neighbor
    inv_wse = ~np.isfinite(wse_seed)
    _, (iy, ix) = distance_transform_edt(inv_wse, return_indices=True)
    wse = wse_seed[iy, ix].astype("float32")
    wse = np.where(channel, wse, np.nan).astype("float32")

    # Width proxy and normalized position
    width = (2.0 * np.maximum(d_bank, 0.0)).astype("float32")

    # Build skeleton: rasterize centerline as the max-distance pixel in each column-slice
    # Simplified: use the pixel with maximum d_bank as the centerline proxy
    d_bank_ch = np.where(channel, d_bank, 0.0)
    inv_center = np.ones(channel.shape, dtype=np.uint8)
    # Find local maxima of d_bank along rows (simplified centerline)
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(d_bank_ch, size=3)
    center_mask = channel & (d_bank_ch == local_max) & (d_bank_ch > pix)
    if center_mask.sum() < 5:
        center_mask = channel & (d_bank_ch > np.percentile(d_bank_ch[channel], 90))

    # Distance to center
    inv_c = np.ones(channel.shape, dtype=np.uint8)
    inv_c[center_mask] = 0
    d_center = distance_transform_edt(inv_c, sampling=pix).astype("float32")

    # Normalized bank distance r: 0=bank, 1=center
    d_bank_eff = np.clip(d_bank - pix, 0.0, None).astype("float32")
    denom = (d_bank_eff + d_center + 1e-6).astype("float32")
    r = np.clip(d_bank_eff / denom, 0.0, 1.0).astype("float32")

    # Dmax from width→depth power law
    dmax_arr = np.clip(a * np.power(np.maximum(width, 1e-3), b), dmin, dmax).astype("float32")

    # Depth and bed
    depth_fill = (dmax_arr * np.power(r, shape_exp)).astype("float32")
    bed_fill = (wse - depth_fill).astype("float32")

    # Only fill gap pixels with valid results
    fill_mask = gaps & np.isfinite(bed_fill) & np.isfinite(wse) & (depth_fill > 0)
    n_filled = int(fill_mask.sum())

    if n_filled > 0:
        bed[fill_mask] = bed_fill[fill_mask]
        bed_profile.update(dtype="float32")
        with rasterio.open(bed_tif, "w", **bed_profile) as dst:
            dst.write(bed.astype("float32"), 1)

        _log.info("[GAP-FILL] Filled %d of %d gaps (%.1f%%). Remaining gaps: %d",
                  n_filled, n_gaps, 100.0 * n_filled / max(1, n_gaps), n_gaps - n_filled)

    # Write companion confidence raster if requested
    if confidence_tif is not None:
        try:
            from river_channel_template import compute_gap_fill_confidence
            conf = compute_gap_fill_confidence(
                bed_valid_before=bed_valid,
                channel=channel,
                sigma_m=confidence_sigma_m,
                pixel_size_m=pix,
            )
            conf_profile = bed_profile.copy()
            conf_profile.update(dtype="float32", nodata=-9999.0, compress="deflate")
            conf_out = np.where(np.isfinite(conf), conf, -9999.0).astype("float32")
            Path(confidence_tif).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(confidence_tif, "w", **conf_profile) as dst:
                dst.write(conf_out, 1)
            _log.info("[GAP-FILL] Confidence raster: %s", confidence_tif)
        except Exception:
            _log.debug("[GAP-FILL] Confidence raster write failed", exc_info=True)

    return n_filled


def _run_river_skeleton(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    network_gpkg: Path,
    work_dir: Path,
    cached_bed_tif: Path,
    script_dir: Path,
    out_dir: Path,
) -> Optional[tuple]:
    """Run the skeleton river bathymetry method.

    Builds a river channel mask from NHD flowlines + WAFFLES water mask,
    then computes distance-transform based bathymetry (no cross-sections).

    Returns (bed_tif, channel_mask_tif) on success, None on failure.
    """
    # Step 2: Build a river channel mask (river vs. open water) from NHD flowlines + waffles water mask.
    log.info("[RIVER] Step 2: Building river channel mask (skeleton method)...")
    channel_mask_tif = work_dir / "river_channel_mask.tif"
    open_water_mask_tif = work_dir / "open_water_mask.tif"

    # Waffles-derived masks:
    #   1) ocean-only (want_nhd=False) always attempted first to prevent ocean bleed (resilient if TNM is flaky).
    #   2) with-NHD (want_nhd=True) attempted second; if it fails we still proceed using corridor+ArcGIS flowlines.
    # Reuse WAFFLES coastline masks across runs for the same AOI + resolution + module params.
    cache_masks_shared = cfg.cache_root.resolve() / "masks"
    ensure_dir(cache_masks_shared)
    cache_masks_run = Path(cfg.derived_cache_root) / "masks"
    ensure_dir(cache_masks_run)
    aoi_buf = str(cfg.aoi)
    inc_arcsec = float(cfg.waffles_inc_arcsec or 1.0)

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
        log.warning("[WAFFLES] Ocean-only mask unavailable; ocean bleed protection degraded: %s", e)
        ocean_mask = None

    try:
        if ocean_mask is None:
            raise RuntimeError("Cannot build canonical with-NHD mask without ocean-only mask")
        with_nhd_mask = _build_canonical_with_nhd_mask(
            Path(ocean_mask),
            Path(network_gpkg),
            cache_masks_run / "waffles_coastline_with_nhd.tif",
            log=log,
        )
    except Exception as e:
        log.warning("[WAFFLES] Canonical with-NHD mask unavailable: %s", e)
        with_nhd_mask = None

    if cfg.strict:
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
        f"--channel-buffer-m={cfg.river_channel_buffer_m}",
        f"--max-channel-width-m={cfg.river_max_channel_width_m}",
        f"--mainstem-method={cfg.river_mainstem_method}",
        f"--mainstem-solve-layer={cfg.river_mainstem_solve_layer}",
        f"--mainstem-min-order={cfg.river_mainstem_min_order}",
        f"--max-mainstem-width-m={cfg.river_max_mainstem_width_m}",
    ]

    # Channel domain source policy
    # - auto: prefer NHDArea river polygons when usable, else fall back to corridor
    # - nhdarea: require river polygons (exclude lakes)
    # - corridor: buffered flowline corridor only
    chan_src = str(cfg.river_channel_source or 'auto').strip().lower()
    if chan_src not in ('auto', 'nhdarea', 'corridor'):
        log.warning("[RIVER] Unknown river_channel_source=%r, defaulting to 'auto'.", chan_src)
        chan_src = 'auto'
    cmd.append(f"--channel-source={chan_src}")

    # NHDArea filtering: keep Stream/River polygons only (exclude lakes/reservoirs).
    # Default is conservative: FType=460 (Stream/River). Users can override via config.
    nhd_allow = cfg.river_nhdarea_allow_ftype
    if nhd_allow is None:
        nhd_allow = "460"
    cmd.append(f"--nhdarea-allow-ftype={nhd_allow}")
    nhd_allow_fcode = cfg.river_nhdarea_allow_fcode
    if nhd_allow_fcode:
        cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")

    # Optional: NHDArea constraint (if river_network.py wrote polygons into the gpkg).
    # river_domain_mask filters NHDArea to river/stream polygons (not lakes).
    # Only pass NHDArea inputs in auto/nhdarea mode.
    if (chan_src in ('auto', 'nhdarea')) and cfg.river_use_nhdarea:
        cmd.append(f"--nhdarea-gpkg={network_gpkg}")
        cmd.append(f"--nhdarea-layer={cfg.river_nhdarea_layer}")
    if ocean_mask and Path(ocean_mask).exists():
        cmd.append(f"--ocean-mask={ocean_mask}")
    oke = float(cfg.river_ocean_keep_dist_m or 0.0)
    if (oke > 0.0):
        cmd.append(f"--ocean-keep-dist-m={oke}")

    if with_nhd_mask and Path(with_nhd_mask).exists():
        cmd.append(f"--water-mask={with_nhd_mask}")
        cmd.append("--water-mask-role=waffles_with_nhd")
    if cfg.river_save_skeleton_debug:
        cmd.append("--write-debug")
    _append_river_domain_policy_args(cmd, work_dir)

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
        report.setdefault("river", {}).setdefault("outputs", {})["waffles_ocean_mask"] = str(ocean_mask) if ocean_mask and Path(ocean_mask).exists() else None
    except Exception:
        log.debug("ignored", exc_info=True)

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

    log.info("[RIVER] Channel mask built: %s", channel_mask_tif)

    report.setdefault("river", {}).setdefault("outputs", {})["river_channel_mask"] = str(channel_mask_tif)
    _record_river_domain_policy(cfg, report, work_dir=work_dir, logger=log)

    # Stash river domain/channel mask for fusion: inside this mask, river should override SDB to avoid tile seams
    cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)

    # Step 2b: Clip estuary pixels from the channel mask.
    # The river module must not produce values where channelized-flow assumptions
    # break down (tidal, backwater, wide estuarine geometry).  Auto-detect the
    # estuary boundary from ocean proximity + hydraulic indicators and zero those
    # pixels in the channel mask before skeleton/XS runs.
    try:
        _n_est, _est_mask_path = _clip_channel_mask_for_estuary(
            channel_mask_tif,
            cfg,
            ocean_mask_path=ocean_mask,
            report=report,
        )
        if _est_mask_path is not None:
            report.setdefault("river", {}).setdefault("outputs", {})["estuary_clip_mask"] = str(_est_mask_path)
            # Also clip the mainstem mask — otherwise the XS builder generates
            # cross-sections in the estuary zone where the channel mask has been
            # removed but the mainstem mask still extends.
            try:
                mainstem_mask_tif = Path(channel_mask_tif).parent / "mainstem_mask.tif"
                if mainstem_mask_tif.exists() and _est_mask_path is not None:
                    import rasterio as _rio_est
                    with _rio_est.open(_est_mask_path) as ec:
                        est_arr = ec.read(1)
                    with _rio_est.open(mainstem_mask_tif) as ms:
                        ms_arr = ms.read(1)
                        ms_prof = ms.profile.copy()
                    n_ms_before = int((ms_arr > 0).sum())
                    ms_arr[est_arr > 0] = 0
                    n_ms_after = int((ms_arr > 0).sum())
                    if n_ms_before > n_ms_after:
                        ms_prof.update(dtype="uint8", nodata=0)
                        with _rio_est.open(mainstem_mask_tif, "w", **ms_prof) as dst:
                            dst.write(ms_arr.astype("uint8"), 1)
                        log.info("[ESTUARY-CLIP] Also clipped mainstem mask: %d → %d pixels (%d removed)",
                                 n_ms_before, n_ms_after, n_ms_before - n_ms_after)
            except (FileNotFoundError, OSError, ValueError, RuntimeError):
                log.debug("[ESTUARY-CLIP] Mainstem mask clipping failed", exc_info=True)
    except (FileNotFoundError, OSError, ValueError, RuntimeError):
        log.warning("[RIVER] Estuary clipping failed; continuing with full channel mask.", exc_info=True)


    # Step 3: Skeleton bathymetry (distance-transform, no cross-sections)
    log.info("[RIVER] Step 3: Inferring bathymetry (channel skeleton)...")
    bed_tif = cached_bed_tif

    # Resolve channel template JSON if available from a prior XS/hybrid run or cache
    _skel_template_json = None
    if cfg.river_channel_template_enabled:
        _template_artifacts = _resolve_channel_template_artifacts(cfg, work_dir=work_dir, river_dir=river_dir)
        if _template_artifacts["template_json"].exists():
            _skel_template_json = _template_artifacts["template_json"]
        if _skel_template_json is None:
            try:
                _tpl_cache_p = cfg.cache_root / "channel_template" / _channel_template_cache_key(cfg) / "channel_template.json"
                if _tpl_cache_p.exists():
                    _skel_template_json = _tpl_cache_p
                    log.info("[RIVER] Loaded channel template from shared cache: %s", _tpl_cache_p)
            except Exception:
                log.debug("[RIVER] Shared template cache lookup failed", exc_info=True)

    cmd = _build_river_skeleton_command(
        cfg,
        network_gpkg=network_gpkg,
        template_raster=cfg.river_dem,
        dem=cfg.river_dem,
        channel_mask=channel_mask_tif,
        out_bed=bed_tif,
        authoritative_passthrough_args=_authoritative_passthrough_args(cfg, for_river=True),
        debug_dir=(work_dir / 'skeleton_debug') if cfg.river_save_skeleton_debug else None,
        channel_template_json=_skel_template_json,
    )
    _record_authoritative_child_passthrough(report, "river", cmd)
    # Optional: use external soundings (extra XYZ) to refine the skeleton prior and enforce depth anchors
    _append_river_soundings_args(cmd, cfg, include_calib_args=False, include_mode_args=True)

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

    log.info("[RIVER] Skeleton bathymetry raster: %s", bed_tif)

    # If present, copy the soundings/channel diagnostic receipt into the user-facing
    # output folder so it is easy to find without digging in derived_cache.
    # Only record paths confirmed to exist on disk.
    try:
        receipt_src = Path(bed_tif).with_name("soundings_channel_receipt.json")
        if receipt_src.exists():
            receipt_dst = Path(out_dir) / "river" / receipt_src.name
            receipt_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(receipt_src, receipt_dst)
            if receipt_dst.exists():
                log.info("[RIVER] Soundings/channel receipt copied to: %s", receipt_dst)
            else:
                log.warning(
                    "[RIVER] Expected receipt was not created after copy: %s",
                    receipt_dst,
                )
        else:
            log.warning("[RIVER] Soundings/channel receipt not found: %s", receipt_src)
    except Exception as e:
        log.warning("[RIVER] Failed to copy soundings/channel receipt: %s", e, exc_info=True)

    # Optional: constrain river outputs to NHDArea polygons (best-effort).
    # Must happen after bed raster exists.
    if cfg.river_use_nhdarea:
        try:
            applied = _mask_raster_to_nhdarea(
                bed_tif,
                network_gpkg,
                nhd_layer=cfg.river_nhdarea_layer,
                nodata=cfg.river_nodata,
            )
            report.setdefault("river", {}).setdefault("masking", {})["nhdarea_bed_masked"] = bool(applied)
        except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError):
            log.debug("ignored", exc_info=True)

    # Final safety: ensure river bed raster is nodata outside the river channel mask.
    try:
        if Path(channel_mask_tif).exists():
            ok = _clip_raster_to_mask(
                Path(bed_tif),
                Path(channel_mask_tif),
                inside_value=1,
                invert=False,
                nodata=cfg.river_nodata,
            )
            report.setdefault("river", {}).setdefault("masking", {})["channel_mask_clip"] = bool(ok)
    except Exception:
        log.debug("ignored", exc_info=True)

    return bed_tif, channel_mask_tif


def _run_river_xs(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    network_gpkg: Path,
    work_dir: Path,
    cached_bed_tif: Path,
    script_dir: Path,
    out_dir: Path,
) -> Optional[tuple]:
    """Run the XS-only river bathymetry method.

    Generates cross-sections, builds a channel domain mask, then infers
    bathymetry via xs_infer_bathy_raster.

    Returns (bed_tif, channel_mask_tif, xs_meta_json, xs_acct_json) on
    success, None on failure.
    """
    xs_meta_json = None
    xs_acct_json = None
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
        f"--smoothing-window-m={cfg.xs_smoothing_window_m}",
        f"--deconflict-tol-m={cfg.xs_deconflict_tol_m}",
        f"--junction-snap-m={cfg.xs_junction_snap_m}",
        f"--junction-buffer-m={cfg.xs_junction_buffer_m}",
        f"--densify-step-m={cfg.xs_densify_step_m}",
    ]
    if not cfg.xs_trim_overlaps:
        cmd.append("--no-trim-overlaps")
    if not cfg.xs_global_deconflict:
        cmd.append("--no-global-deconflict")
    if not cfg.xs_skip_junctions:
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

    log.info("[RIVER] Cross-sections generated: %s", xs_gpkg)
    # Step 2b: Build river channel domain mask (so final river raster is river-only)
    channel_mask_tif = work_dir / "river_channel_mask.tif"
    open_water_mask_tif = work_dir / "open_water_mask.tif"

    # Reuse the same domain-mask logic used by the skeleton method for consistency.
    # Prefer the precomputed shared guidance-domain artifacts that were already planned
    # on the river template grid. Recomputing here with the raw shared-grid WAFFLES mask
    # is what caused the later XS-only river_domain_mask call to fail the water∩corridor
    # validation even though the pre-inference guidance-domain planning succeeded.
    precomputed_channel_mask = None
    precomputed_open_water_mask = None
    precomputed_mainstem_mask = None
    precomputed_policy_json = None
    precomputed_effective_water_mask = None
    precomputed_corridor_mask = None
    precomputed_nhdarea_mask = None
    try:
        if getattr(cfg, "river_channel_mask", None):
            pp = Path(cfg.river_channel_mask)
            if pp.exists():
                precomputed_channel_mask = pp
        if getattr(cfg, "guidance_domain_review_manifest", None):
            manifest_path = Path(cfg.guidance_domain_review_manifest)
            if manifest_path.exists():
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_outputs = manifest_payload.get("outputs", {}) if isinstance(manifest_payload, dict) else {}
                for key, attr_name in (
                    ("open_water_mask", "precomputed_open_water_mask"),
                    ("mainstem_mask", "precomputed_mainstem_mask"),
                    ("river_domain_policy_json", "precomputed_policy_json"),
                ):
                    value = manifest_outputs.get(key)
                    if value:
                        cand = Path(str(value))
                        if not cand.is_absolute():
                            cand = (manifest_path.parent / cand).resolve()
                        if cand.exists():
                            if attr_name == "precomputed_open_water_mask":
                                precomputed_open_water_mask = cand
                            elif attr_name == "precomputed_mainstem_mask":
                                precomputed_mainstem_mask = cand
                            elif attr_name == "precomputed_policy_json":
                                precomputed_policy_json = cand
        river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
        for key, attr_name in (
            ("river_effective_water_mask", "precomputed_effective_water_mask"),
            ("river_corridor_mask_debug", "precomputed_corridor_mask"),
            ("river_nhdarea_mask_debug", "precomputed_nhdarea_mask"),
        ):
            value = river_outputs.get(key)
            if value:
                cand = Path(str(value))
                if cand.exists():
                    if attr_name == "precomputed_effective_water_mask":
                        precomputed_effective_water_mask = cand
                    elif attr_name == "precomputed_corridor_mask":
                        precomputed_corridor_mask = cand
                    elif attr_name == "precomputed_nhdarea_mask":
                        precomputed_nhdarea_mask = cand
    except Exception:
        log.debug("[RIVER] Failed to inspect precomputed guidance-domain artifacts", exc_info=True)

    ocean_mask = None
    with_nhd_mask = None
    template_water_support_mask = None

    if precomputed_channel_mask is not None:
        _link_or_copy_file(Path(precomputed_channel_mask), channel_mask_tif)
        if precomputed_open_water_mask is not None:
            _link_or_copy_file(Path(precomputed_open_water_mask), open_water_mask_tif)
        if precomputed_mainstem_mask is not None:
            _link_or_copy_file(Path(precomputed_mainstem_mask), work_dir / "mainstem_mask.tif")
        if precomputed_policy_json is not None:
            _link_or_copy_file(Path(precomputed_policy_json), work_dir / "river_domain_policy.json")
        if precomputed_effective_water_mask is not None:
            _link_or_copy_file(Path(precomputed_effective_water_mask), work_dir / "river_effective_water_mask.tif")
        if precomputed_corridor_mask is not None:
            _link_or_copy_file(Path(precomputed_corridor_mask), work_dir / "river_corridor_mask_debug.tif")
        if precomputed_nhdarea_mask is not None:
            _link_or_copy_file(Path(precomputed_nhdarea_mask), work_dir / "river_nhdarea_mask_debug.tif")
        cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)
        report.setdefault("river", {}).setdefault("steps", {})["domain_mask"] = {
            "status": "reused_precomputed",
            "command": None,
            "source": "shared_guidance_domains",
            "river_channel_mask": str(precomputed_channel_mask),
            "open_water_mask": str(precomputed_open_water_mask) if precomputed_open_water_mask else None,
            "mainstem_mask": str(precomputed_mainstem_mask) if precomputed_mainstem_mask else None,
            "river_domain_policy_json": str(precomputed_policy_json) if precomputed_policy_json else None,
        }
        log.info("[RIVER] Reused precomputed shared guidance-domain masks for XS method: channel=%s", precomputed_channel_mask)
    else:
        # Reuse WAFFLES coastline masks across runs for the same AOI + resolution + module params.
        cache_masks_shared = cfg.cache_root.resolve() / "masks"
        ensure_dir(cache_masks_shared)
        cache_masks_run = Path(cfg.derived_cache_root) / "masks"
        ensure_dir(cache_masks_run)
        aoi_buf = str(cfg.aoi)
        inc_arcsec = float(cfg.waffles_inc_arcsec or 1.0)

        ocean_mask = None
        with_nhd_mask = None
        template_water_support_mask = None
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
            log.warning("[WAFFLES] Ocean-only mask unavailable; ocean bleed protection degraded: %s", e)
            ocean_mask = None

        try:
            if ocean_mask is None:
                raise RuntimeError("Cannot build canonical with-NHD mask without ocean-only mask")
            with_nhd_mask = _build_canonical_with_nhd_mask(
                Path(ocean_mask),
                Path(network_gpkg),
                cache_masks_run / "waffles_coastline_with_nhd.tif",
                log=log,
            )
            template_water_support_mask = _build_template_aligned_river_water_mask(
                Path(with_nhd_mask),
                Path(network_gpkg),
                Path(cfg.river_dem),
                cache_masks_run / "river_template_water_support_mask.tif",
                logger=log,
            )
        except Exception as e:
            log.warning("[WAFFLES] Canonical/template-aligned water support mask unavailable: %s", e)
            with_nhd_mask = None
            template_water_support_mask = None

        if cfg.strict:
            if (template_water_support_mask is None) or (not Path(template_water_support_mask).exists()):
                raise RuntimeError("Strict river domain build requested but template-aligned river water support mask is unavailable. Fix WAFFLES/NHDArea generation first.")

        cmd = [
            sys.executable, "river_domain_mask.py",
            f"--river-gpkg={network_gpkg}",
            f"--template-raster={cfg.river_dem}",
            f"--out-channel-mask={channel_mask_tif}",
            f"--out-open-water-mask={open_water_mask_tif}",
            f"--channel-buffer-m={cfg.river_channel_buffer_m}",
            f"--max-channel-width-m={cfg.river_max_channel_width_m}",
            f"--mainstem-method={cfg.river_mainstem_method}",
            f"--mainstem-solve-layer={cfg.river_mainstem_solve_layer}",
            f"--mainstem-min-order={cfg.river_mainstem_min_order}",
            f"--max-mainstem-width-m={cfg.river_max_mainstem_width_m}",
        ]

        chan_src = str(cfg.river_channel_source or 'auto').strip().lower()
        if chan_src not in ('auto', 'nhdarea', 'corridor'):
            log.warning("[RIVER] Unknown river_channel_source=%r, defaulting to 'auto'.", chan_src)
            chan_src = 'auto'
        cmd.append(f"--channel-source={chan_src}")

        nhd_allow = cfg.river_nhdarea_allow_ftype
        if nhd_allow is None:
            nhd_allow = "460"
        cmd.append(f"--nhdarea-allow-ftype={nhd_allow}")
        nhd_allow_fcode = cfg.river_nhdarea_allow_fcode
        if nhd_allow_fcode:
            cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")

        if (chan_src in ('auto', 'nhdarea')) and cfg.river_use_nhdarea:
            cmd.append(f"--nhdarea-gpkg={network_gpkg}")
            cmd.append(f"--nhdarea-layer={cfg.river_nhdarea_layer}")

        if ocean_mask and Path(ocean_mask).exists():
            cmd.append(f"--ocean-mask={ocean_mask}")
        oke = float(cfg.river_ocean_keep_dist_m or 0.0)
        if oke > 0.0:
            cmd.append(f"--ocean-keep-dist-m={oke}")

        if template_water_support_mask and Path(template_water_support_mask).exists():
            cmd.append(f"--water-mask={template_water_support_mask}")
            cmd.append("--water-mask-role=river_support")
        elif with_nhd_mask and Path(with_nhd_mask).exists():
            cmd.append(f"--water-mask={with_nhd_mask}")
            cmd.append("--water-mask-role=waffles_with_nhd")

        try:
            wm = None
            if template_water_support_mask and Path(template_water_support_mask).exists():
                wm = template_water_support_mask
            elif with_nhd_mask and Path(with_nhd_mask).exists():
                wm = with_nhd_mask
            elif ocean_mask and Path(ocean_mask).exists():
                wm = ocean_mask
            report.setdefault("river", {}).setdefault("outputs", {})["waffles_water_mask"] = str(wm) if wm else None
            report.setdefault("river", {}).setdefault("outputs", {})["waffles_ocean_mask"] = str(ocean_mask) if ocean_mask and Path(ocean_mask).exists() else None
        except Exception:
            log.debug("ignored", exc_info=True)

        rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
        cmd_str = " ".join(str(c) for c in cmd)
        report["river"]["steps"]["domain_mask"] = {
            "status": "success" if rc == 0 else "failed",
            "returncode": rc,
            "command": cmd_str,
            "stdout_tail": out,
            "stderr_tail": err,
            "water_mask_used": str(template_water_support_mask) if template_water_support_mask else (str(with_nhd_mask) if with_nhd_mask else None),
            "water_mask_role": "river_support" if template_water_support_mask else ("waffles_with_nhd" if with_nhd_mask else None),
        }
        if rc != 0 or (not channel_mask_tif.exists()):
            log.warning("[RIVER] Failed to build channel mask for XS method; continuing without hard domain constraint.")
        else:
            try:
                cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)
            except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError):
                log.debug("ignored", exc_info=True)

    # Step 2b: Clip estuary pixels from channel mask (same as skeleton path)
    if channel_mask_tif.exists():
        try:
            _n_est, _est_mask_path = _clip_channel_mask_for_estuary(
                channel_mask_tif,
                cfg,
                ocean_mask_path=ocean_mask,
                report=report,
            )
            if _est_mask_path is not None:
                report.setdefault("river", {}).setdefault("outputs", {})["estuary_clip_mask"] = str(_est_mask_path)
        except (OSError, ValueError, RuntimeError, ImportError) as e:
            report.setdefault("river", {}).setdefault("masking", {})["estuary_clip_warning"] = str(e)
            log.warning("[RIVER] Estuary clipping failed; continuing with full channel mask: %s", e)


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
        f"--continuous={cfg.river_continuous}",
        f"--continuous-k={cfg.river_continuous_k}",
        f"--idw-power={cfg.river_idw_power}",
        f"--aniso-along-scale-m={cfg.river_aniso_along_scale_m}",
        f"--aniso-cross-scale-m={cfg.river_aniso_cross_scale_m}",
        f"--thalweg-weight={cfg.river_thalweg_weight}",
        f"--nodata={cfg.river_nodata}",
        f"--overlap-reducer={cfg.river_overlap_reducer}",
    ]
    # Hard constrain interpolation/output to the river domain mask when available.
    if channel_mask_tif is not None and Path(channel_mask_tif).exists():
        cmd.append(f"--channel-mask-raster={channel_mask_tif}")
        cmd.append("--channel-mask-inside-value=1")
    _estuary_clip_mask = work_dir / "estuary_clip_mask.tif"
    if _estuary_clip_mask.exists():
        cmd.append(f"--estuary-clip-mask={_estuary_clip_mask}")
    _estuary_transition_mask = work_dir / "estuary_transition_mask.tif"
    if _estuary_transition_mask.exists():
        cmd.append(f"--estuary-transition-mask={_estuary_transition_mask}")
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
    if cfg.river_wse_profile_enabled:
        cmd.append("--wse-profile-enabled")
    else:
        cmd.append("--no-wse-profile")
    cmd.append(f"--wse-profile-window={cfg.river_wse_profile_window}")
    cmd.append(f"--wse-profile-min-n={cfg.river_wse_profile_min_n}")
    if cfg.river_wse_profile_monotonic:
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
    if cfg.river_manning_mode != "off":
        cmd.append(f"--manning-mode={cfg.river_manning_mode}")
        if cfg.river_manning_q_cms is not None:
            cmd.append(f"--manning-q-cms={cfg.river_manning_q_cms}")
        if cfg.river_manning_q_field:
            cmd.append(f"--manning-q-field={cfg.river_manning_q_field}")
        cmd.append(f"--manning-n={cfg.river_manning_n}")
        cmd.append(f"--manning-region={cfg.river_manning_region}")
        cmd.append(f"--manning-min-confidence={cfg.river_manning_min_confidence}")
        cmd.append(f"--manning-max-weight={cfg.river_manning_max_weight}")
        cmd.append(f"--manning-backwater-slope-thresh={cfg.river_manning_backwater_slope_thresh}")
        if cfg.river_manning_dist_to_mouth_field:
            cmd.append(f"--manning-dist-to-mouth-field={cfg.river_manning_dist_to_mouth_field}")
        cmd.append(f"--manning-dist-to-mouth-km-max={cfg.river_manning_dist_to_mouth_km_max}")



    # Optional: Regional hydraulic geometry curve prior
    if cfg.river_regional_curve_enabled:
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

    # ── Channel template system ──
    _append_channel_template_args(cmd, cfg, river_dir)

    if cfg.river_continuous_buffer_m is not None:
        cmd.append(f"--continuous-buffer-m={cfg.river_continuous_buffer_m}")
    _append_river_soundings_args(cmd, cfg, include_calib_args=False, include_mode_args=True)

    rc, out, err = run_command(cmd, cwd=script_dir, prefix="[RIVER] ")
    cmd_str = " ".join(str(c) for c in cmd)
    rc, err = _enforce_template_disable_runtime_contract(cfg, rc, out, err)
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

    
    return bed_tif, channel_mask_tif, xs_meta_json, xs_acct_json


def _build_constraint_summary(cfg: "BathyConfig", report: Dict[str, Any]) -> None:
    """Summarise XS constraint metadata into report['river']['constraints_summary'].

    Reads the constraint sidecar already placed in report by xs_infer, checks
    which requirements (soundings, slope, drainage-area) the run satisfied, and
    records a self-contained summary dict.
    """
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
            "requirement": cfg.require_river_constraints,
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

        req = str(cfg.require_river_constraints or "none").lower().strip()
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
        log.debug("Failed to write constraint summary; continuing.", exc_info=True)



def _normalize_channel_template_setting(cfg: "BathyConfig", report: Dict[str, Any]) -> bool:
    """Normalize the XS channel-template setting for river runs."""
    requested = enforce_channel_template_invariant(
        cfg,
        enabled_attr="river_channel_template_enabled",
        requested_attr="river_channel_template_requested",
        forced_attr="river_channel_template_forced_enabled",
        reason_attr="river_channel_template_forced_reason",
        log_info=lambda msg: log.info("[RIVER] %s", msg),
        context_label="channel template",
        reason="channel template enablement is normalized and audited for river runs; explicit disable is allowed.",
        force_enabled=False,
    )
    report.setdefault("river", {}).setdefault("notes", {})["channel_template_enablement"] = {
        "requested_enabled": requested,
        "effective_enabled": bool(getattr(cfg, "river_channel_template_enabled", False)),
        "requested_attr_recorded": hasattr(cfg, "river_channel_template_requested"),
        "forced_attr_recorded": hasattr(cfg, "river_channel_template_forced_enabled"),
        "reason": getattr(cfg, "river_channel_template_forced_reason", None),
    }
    return requested

def _build_structured_helper_rasters(
    *,
    cfg: "BathyConfig",
    river_dem: Path,
    channel_mask_tif: Path,
    bed_tif: Path,
    depth_tif: Path,
    logger,
) -> Dict[str, Any]:
    """Seed structured-stage helper rasters directly from authoritative DEM support.

    This deliberately avoids the upstream hybrid precursor. The helper bed is simply the
    authoritative DEM clipped to the river corridor; the structured channel frame/scaffold/surface
    path then becomes the active estimator that writes the primary river products.
    """
    import rasterio
    import numpy as np
    from rasterio.enums import Resampling

    receipt: Dict[str, Any] = {
        "mode": "authoritative_dem_clipped",
        "river_dem": str(river_dem),
        "channel_mask": str(channel_mask_tif),
        "helper_bed": str(bed_tif),
        "helper_depth": str(depth_tif),
    }
    with rasterio.open(river_dem) as dem_ds, rasterio.open(channel_mask_tif) as mask_ds:
        profile = dem_ds.profile.copy()
        profile.update(dtype='float32', compress='deflate')
        nodata = float(profile.get('nodata', getattr(cfg, 'river_nodata', -9999.0)) or getattr(cfg, 'river_nodata', -9999.0))
        profile['nodata'] = nodata
        mask = mask_ds.read(1, out_shape=(dem_ds.height, dem_ds.width), resampling=Resampling.nearest)
        dem = dem_ds.read(1).astype('float32')
        dem_nodata = dem_ds.nodata
        valid = np.isfinite(dem)
        if dem_nodata is not None:
            valid &= ~np.isclose(dem, float(dem_nodata))
        inside = mask > 0
        bed = np.full(dem.shape, nodata, dtype='float32')
        bed[inside & valid] = dem[inside & valid]
        with rasterio.open(bed_tif, 'w', **profile) as dst:
            dst.write(bed, 1)
    depth_recovery = compute_depth_from_bed_and_dem(
        bed_tif,
        river_dem,
        depth_tif,
        depth_sign='negative_down',
        channel_mask_tif=channel_mask_tif,
    )
    depth_recovery_contract_path = bed_tif.parent / 'river_depth_surface_recovery_contract.json'
    write_json(depth_recovery_contract_path, depth_recovery)
    receipt['depth_surface_recovery'] = depth_recovery
    receipt['depth_surface_recovery_contract_path'] = str(depth_recovery_contract_path)
    receipt['helper_valid_pixels'] = int(np.count_nonzero(np.isfinite(bed) & ~np.isclose(bed, nodata)))
    logger.info('[RIVER][STRUCTURED] Built helper rasters directly from authoritative DEM and channel mask: valid=%d', receipt['helper_valid_pixels'])
    return receipt


def run_river(cfg: BathyConfig, report: Dict[str, Any]) -> Optional[Path]:
    log.info("River pipeline (method=%s)", cfg.river_method)
    # Ensure report structure exists (avoid KeyError if caller created only partial report dict)
    if report is not None:
        report.setdefault("river", {})
        report["river"].setdefault("steps", {})

    if cfg.river_dem is None:
        auto_dem = ensure_river_dem_auto(cfg, report)
        if auto_dem is None:
            log.error("[RIVER] Missing --river-dem (and auto-build failed)")
            report["river"] = {"status": "failed", "reason": "missing river_dem"}
            return None
        cfg.river_dem = auto_dem

    plan = determine_river_execution_plan(
        cfg=cfg,
        report=report,
        ensure_dir_fn=ensure_dir,
        normalize_channel_template_setting_fn=_normalize_channel_template_setting,
        logger=log,
    )
    river_dir = plan.river_dir
    script_dir = plan.script_dir
    cache_dir = plan.cache_dir
    work_dir = plan.work_dir
    raw_hydro_cache = plan.raw_hydro_cache
    cached_bed_tif = plan.cached_bed_tif
    cached_depth_tif = plan.cached_depth_tif
    # Step 1: river network (shared with early guidance-domain planning)
    log.info("[RIVER] Step 1: Extracting river network...")
    network_gpkg = _ensure_river_network_artifact(
        cfg,
        report,
        script_dir=script_dir,
        cache_dir=cache_dir,
        work_dir=work_dir,
        raw_hydro_cache=raw_hydro_cache,
    )
    if network_gpkg is None:
        return None


    requested_river_method = plan.requested_method
    river_method = plan.effective_method
    river_exec = report.setdefault("river", {}).setdefault("execution_receipts", {})
    river_exec["river_method_requested"] = requested_river_method
    river_exec["structured_mode_requested"] = requested_river_method == "structured"
    river_exec["river_v2_requested"] = requested_river_method in RIVER_V2_METHODS
    river_exec["absolute_bed_fallback_allowed"] = False
    river_exec["absolute_bed_fallback_removed"] = True
    river_exec.setdefault("legacy_xs_inputs_used", False)
    river_exec.setdefault("legacy_xs_inputs_detected", [])

    river_exec["river_method_executed"] = river_method
    river_exec["river_v2_active"] = river_method in RIVER_V2_METHODS
    river_exec["structured_mode_active"] = river_method == "structured"
    river_exec["legacy_xs_inputs_expected"] = river_method in {"hybrid", "xs"}
    river_exec["legacy_xs_inputs_permitted"] = river_method in {"hybrid", "xs"}

    # These are only written by the XS method (else branch below), but are referenced after all
    # method branches when capturing constraint meta/accounting. Initialise to None so that
    # hybrid and skeleton runs don't hit NameError.
    xs_meta_json: Optional[Path] = None
    xs_acct_json: Optional[Path] = None
    # Rationale: XS is most defensible on mainstem and most failure-prone at dense tributary junctions.
    # This preserves continuity along the mainstem while avoiding tributary overlap artifacts.

    def _build_domain_masks(_work_dir: Path, *, strict: bool = False):
        """Build channel/open-water/mainstem masks (DEM-aligned).

        If strict=True, missing channel mask is treated as fatal.
        """
        channel_mask_tif = _work_dir / "river_channel_mask.tif"
        open_water_mask_tif = _work_dir / "open_water_mask.tif"
        mainstem_mask_tif = _work_dir / "mainstem_mask.tif"

        precomputed = report.get("guidance_domains", {}).get("outputs", {}) if isinstance(report, dict) else {}
        precomputed_channel = precomputed.get("river_channel_mask") or getattr(cfg, "river_channel_mask", None)
        precomputed_open = precomputed.get("open_water_mask")
        precomputed_mainstem = precomputed.get("mainstem_mask")
        precomputed_estuary = precomputed.get("estuary_clip_mask")
        precomputed_transition = precomputed.get("estuary_transition_mask")
        if precomputed_channel and Path(precomputed_channel).exists():
            shutil.copy2(Path(precomputed_channel), channel_mask_tif)
            if precomputed_open and Path(precomputed_open).exists():
                shutil.copy2(Path(precomputed_open), open_water_mask_tif)
            if precomputed_mainstem and Path(precomputed_mainstem).exists():
                shutil.copy2(Path(precomputed_mainstem), mainstem_mask_tif)
            if precomputed_estuary and Path(precomputed_estuary).exists():
                shutil.copy2(Path(precomputed_estuary), _work_dir / "estuary_clip_mask.tif")
                report.setdefault("river", {}).setdefault("outputs", {})["estuary_clip_mask"] = str(_work_dir / "estuary_clip_mask.tif")
            if precomputed_transition and Path(precomputed_transition).exists():
                shutil.copy2(Path(precomputed_transition), _work_dir / "estuary_transition_mask.tif")
                report.setdefault("river", {}).setdefault("outputs", {})["estuary_transition"] = str(_work_dir / "estuary_transition_mask.tif")
            try:
                wm = Path(getattr(cfg, "waffles_with_nhd_mask", None) or "")
                if wm.exists():
                    report.setdefault("river", {}).setdefault("outputs", {})["waffles_water_mask"] = str(wm)
                om = Path(getattr(cfg, "waffles_ocean_mask", None) or "")
                if om.exists():
                    report.setdefault("river", {}).setdefault("outputs", {})["waffles_ocean_mask"] = str(om)
            except (OSError, TypeError, ValueError):
                log.debug("ignored", exc_info=True)
            report.setdefault("river", {}).setdefault("steps", {})["domain_mask"] = {
                "status": "success",
                "returncode": 0,
                "command": "precomputed_guidance_domains",
                "stdout_tail": "",
                "stderr_tail": "",
            }
            cfg.river_domain_mask_for_fusion = Path(precomputed.get("river_guidance_domain_mask") or getattr(cfg, "river_guidance_domain_mask", channel_mask_tif))
            return channel_mask_tif, open_water_mask_tif, mainstem_mask_tif

        # Reuse WAFFLES coastline masks across runs for the same AOI + resolution + module params.
        cache_masks_shared = cfg.cache_root.resolve() / "masks"
        ensure_dir(cache_masks_shared)
        cache_masks_run = Path(cfg.derived_cache_root) / "masks"
        ensure_dir(cache_masks_run)
        aoi_buf = str(cfg.aoi)
        inc_arcsec = float(cfg.waffles_inc_arcsec or 1.0)

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
        except (OSError, RuntimeError, ValueError) as e:
            log.warning("[WAFFLES] Ocean-only mask unavailable; ocean bleed protection degraded: %s", e)
            ocean_mask = None
            report.setdefault("river", {}).setdefault("masking", {})["ocean_mask_warning"] = str(e)

        try:
            if ocean_mask is None:
                raise RuntimeError("Cannot build canonical with-NHD mask without ocean-only mask")
            with_nhd_mask = _build_canonical_with_nhd_mask(
                Path(ocean_mask),
                Path(network_gpkg),
                cache_masks_run / "waffles_coastline_with_nhd.tif",
                log=log,
            )
        except (OSError, RuntimeError, ValueError) as e:
            log.warning("[WAFFLES] Canonical with-NHD mask unavailable: %s", e)
            with_nhd_mask = None
            report.setdefault("river", {}).setdefault("masking", {})["with_nhd_mask_warning"] = str(e)


        if cfg.strict:
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
            f"--channel-buffer-m={cfg.river_channel_buffer_m}",
            f"--max-channel-width-m={cfg.river_max_channel_width_m}",
            f"--mainstem-method={cfg.river_mainstem_method}",
            f"--mainstem-solve-layer={cfg.river_mainstem_solve_layer}",
            f"--mainstem-min-order={cfg.river_mainstem_min_order}",
            f"--max-mainstem-width-m={cfg.river_max_mainstem_width_m}",
        ]

        chan_src = str(cfg.river_channel_source or 'auto').strip().lower()
        if chan_src not in ('auto', 'nhdarea', 'corridor'):
            log.warning("[RIVER] Unknown river_channel_source=%r, defaulting to 'auto'.", chan_src)
            chan_src = 'auto'
        cmd.append(f"--channel-source={chan_src}")

        nhd_allow = cfg.river_nhdarea_allow_ftype
        if nhd_allow is None:
            nhd_allow = "460"
        cmd.append(f"--nhdarea-allow-ftype={nhd_allow}")
        nhd_allow_fcode = cfg.river_nhdarea_allow_fcode
        if nhd_allow_fcode:
            cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")

        if (chan_src in ('auto', 'nhdarea')) and cfg.river_use_nhdarea:
            cmd.append(f"--nhdarea-gpkg={network_gpkg}")
            cmd.append(f"--nhdarea-layer={cfg.river_nhdarea_layer}")

        if ocean_mask and Path(ocean_mask).exists():
            cmd.append(f"--ocean-mask={ocean_mask}")
        oke = float(cfg.river_ocean_keep_dist_m or 0.0)
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
            report.setdefault("river", {}).setdefault("outputs", {})["waffles_ocean_mask"] = str(ocean_mask) if ocean_mask and Path(ocean_mask).exists() else None
        except (OSError, TypeError, ValueError, KeyError):
            log.debug("ignored", exc_info=True)

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
            if cfg.strict:
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
            except (ImportError, OSError, ValueError, RuntimeError) as e:
                report.setdefault("river", {}).setdefault("masking", {})["channel_mask_validation_warning"] = str(e)
                if cfg.strict:
                    log.error("[RIVER] Channel mask validation failed (strict): %s", e)
                    raise
                log.warning("[RIVER] Channel mask validation warning (non-strict): %s", e)

            try:
                cfg.river_domain_mask_for_fusion = Path(channel_mask_tif)
            except OSError:
                log.debug("ignored", exc_info=True)

        return channel_mask_tif, open_water_mask_tif, mainstem_mask_tif


    # Phase C: hybrid merge moved to hybrid_merge.merge_hybrid_river_bed

    def _count_valid_pixels(rp: Path, nodata_val: float) -> int:
        try:
            import rasterio
            import numpy as np
            with rasterio.open(rp) as ds:
                nod = ds.nodata if ds.nodata is not None else nodata_val
                total = 0
                for _, w in ds.block_windows(1):
                    a = ds.read(1, window=w)
                    total += int(np.count_nonzero(np.isfinite(a) & (a != nod)))
                return int(total)
        except Exception:
            log.debug("_count_valid_pixels: suppressed exception", exc_info=True)
            return -1


    def _apply_river_v2_result_to_report(*, report, v2_result, v2_context):
            river_report = report.setdefault("river", {})
            outputs_root = report.setdefault("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
            v2_outputs = dict(v2_result.aux_outputs or {})
            locked_surface = v2_outputs.get("river_v2_primary_surface_authoritative_applied") or v2_outputs.get("river_primary_surface_authoritative_applied")
            primary_surface = v2_outputs.get("river_v2_primary_surface") or v2_outputs.get("river_primary_surface")

            river_outputs = _register_river_v2_outputs(outputs_root=outputs_root, v2_outputs=v2_outputs)
            if primary_surface:
                river_outputs["river_primary_surface"] = str(primary_surface)
            if v2_result.success and locked_surface:
                river_outputs = _set_river_v2_final_guidance_surface(
                    river_outputs=river_outputs,
                    locked_surface=str(locked_surface),
                )

            river_report["v2_pipeline"] = v2_result.to_dict()
            river_report["simple_stage_status"] = _build_simple_river_status_from_v2(report=report, v2_result=v2_result, v2_context=v2_context)
            river_report["status"] = "success" if v2_result.success else "failed"
            river_report["status_family"] = river_report["status"]
            river_report["execution_mode"] = "river_v2_pipeline"
            river_report["outputs"] = dict(river_outputs)
            river_report["notes"] = {
                "river_v2": {
                    "routing": "authoritative_applied_primary_surface_participates_in_final_dem_route",
                    "river_raster_returned": bool(v2_result.success),
                    "pipeline_version": v2_outputs.get("pipeline_version", str(PIPELINE_VERSION)),
                    "river_method_selected": "v2",
                    "river_path_used": "river_v2_only",
                    "legacy_river_path_participated": False,
                    "summary_path": v2_outputs.get("river_v2_summary"),
                    "primary_surface_path": primary_surface,
                    "primary_surface_authoritative_applied_path": locked_surface,
                    "stage_products_overview": v2_outputs.get("river_v2_stage_products_overview"),
                }
            }
            river_report["v2_route_contract"] = {
                "active": bool(v2_result.success),
                "active_stage": "river_primary_surface_authoritative_applied" if v2_result.success else None,
                "active_river_guidance_surface": str(locked_surface) if v2_result.success and locked_surface else None,
                "legacy_river_final_route_participation_blocked": bool(v2_result.success),
                "runtime_enforced": True,
                "river_method_selected": "v2",
                "river_path_used": "river_v2_only",
                "legacy_river_path_participated": False,
                "pipeline_version": v2_outputs.get("pipeline_version", str(PIPELINE_VERSION)),
                "stage_products_overview": v2_outputs.get("river_v2_stage_products_overview"),
            }

            return Path(locked_surface) if v2_result.success and locked_surface else None


    if river_method in RIVER_V2_METHODS:
        channel_mask_tif, _, _ = _build_domain_masks(work_dir, strict=False)
        authoritative_outputs = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
        v2_out_dir = ensure_dir(Path(cfg.out_dir) / "river_v2")
        v2_bank_guidance_inputs = resolve_river_v2_bank_guidance_inputs(report, v2_out_dir)
        v2_support_policy = _resolve_river_v2_authoritative_support_policy(cfg=cfg, authoritative_outputs=authoritative_outputs)
        v2_context = RiverV2Context(
            cfg=cfg,
            report=report,
            out_dir=v2_out_dir,
            network_gpkg=Path(network_gpkg),
            river_dem_path=Path(cfg.river_dem),
            channel_mask_path=Path(channel_mask_tif) if channel_mask_tif is not None else None,
            authoritative_bed_path=(
                Path(report.get("outputs", {}).get("river_authoritative_soundings"))
                if isinstance(report.get("outputs", {}), dict) and report.get("outputs", {}).get("river_authoritative_soundings")
                else (
                    Path(getattr(cfg, "river_authoritative_soundings", "") or "")
                    if getattr(cfg, "river_authoritative_soundings", None)
                    else (Path(getattr(cfg, "river_authoritative_bed", "") or "") if getattr(cfg, "river_authoritative_bed", None) else None)
                )
            ),
            authoritative_base_path=(
                Path(authoritative_outputs.get("authoritative_base"))
                if authoritative_outputs.get("authoritative_base")
                else (Path(getattr(cfg, "authoritative_base", "") or "") if getattr(cfg, "authoritative_base", None) else None)
            ),
            aligned_authoritative_base_path=(
                Path(authoritative_outputs.get("aligned_authoritative_base"))
                if authoritative_outputs.get("aligned_authoritative_base")
                else None
            ),
            baseline_interpolated_path=(
                Path(authoritative_outputs.get("baseline_cudem_interpolation"))
                if authoritative_outputs.get("baseline_cudem_interpolation")
                else ((Path(cfg.out_dir) / "final" / "authoritative_base_aligned.tif") if (Path(cfg.out_dir) / "final" / "authoritative_base_aligned.tif").exists() else None)
            ),
            authoritative_support_coverage_path=Path(authoritative_outputs.get("authoritative_support_coverage")) if authoritative_outputs.get("authoritative_support_coverage") else None,
            authoritative_dem_mode=str(v2_support_policy.get("authoritative_dem_mode", "mixed_requires_metadata")),
            trusted_support_mode=str(v2_support_policy.get("trusted_support_mode", "low_support_no_trusted_support")),
            trusted_support_artifact_path=Path(v2_support_policy["trusted_support_artifact_path"]) if v2_support_policy.get("trusted_support_artifact_path") else None,
            authoritative_support_policy_warning=v2_support_policy.get("authoritative_support_policy_warning"),
            support_policy_source=v2_support_policy.get("support_policy_source"),
            bank_wse_edge_guidance_path=Path(v2_bank_guidance_inputs["bank_wse_edge_guidance_path"]) if v2_bank_guidance_inputs.get("bank_wse_edge_guidance_path") else None,
            bank_materialization_diagnostics_path=Path(v2_bank_guidance_inputs["bank_materialization_diagnostics_path"]) if v2_bank_guidance_inputs.get("bank_materialization_diagnostics_path") else None,
            vertical_reference=str(getattr(cfg, "vertical_datum", None) or getattr(cfg, "vertical_datum_name", None) or "unknown"),
            centerline_spacing_m=float(getattr(cfg, "river_centerline_sample_spacing_m", 0.0) or 0.0) or None,
        )
        v2_result = run_river_v2_pipeline(v2_context)
        v2_locked_surface = _apply_river_v2_result_to_report(report=report, v2_result=v2_result, v2_context=v2_context)
        if v2_result.success and v2_locked_surface is not None:
            log.info("[RIVER][V2] Pipeline complete: stages=%d summary=%s", len(v2_result.stage_results), v2_result.aux_outputs.get("river_v2_summary", str(v2_context.paths.pipeline_summary)))
            return v2_locked_surface
        log.error("[RIVER][V2] Pipeline failed at stage=%s error=%s", v2_result.failed_stage, v2_result.error)
        return None

    if river_method == "v1":
        river_raster_v1 = run_river_v1_stage(
            cfg=cfg,
            report=report,
            river_dir=river_dir,
            work_dir=work_dir,
            network_gpkg=network_gpkg,
            build_domain_masks_fn=_build_domain_masks,
            load_support_points_fn=_load_river_authoritative_support_points,
            logger=log,
        )
        log.info("[RIVER][V1] Minimal pipeline produced one primary river surface for fusion: %s", river_raster_v1)
        return Path(river_raster_v1)

    if river_method == "structured":
        cfg.river_allow_absolute_bed_fallback = False
        cfg._write_river_guidance_artifacts = _write_river_guidance_artifacts
        return run_structured_river_stage(
            cfg=cfg,
            report=report,
            river_dir=river_dir,
            work_dir=work_dir,
            cache_dir=cache_dir,
            logger=log,
            build_domain_masks_fn=_build_domain_masks,
            estuary_clip_fn=_clip_channel_mask_for_estuary,
            build_structured_helper_rasters_fn=_build_structured_helper_rasters,
            write_guidance_artifacts_fn=_write_guidance_artifacts_with_reporting,
            build_constraint_summary_fn=_build_constraint_summary,
            apply_depth_metadata_fn=apply_depth_metadata,
            apply_elevation_metadata_fn=apply_elevation_metadata,
        )

    if river_method == "hybrid":
        hybrid_result = run_hybrid_river_stage(
            cfg=cfg,
            report=report,
            script_dir=script_dir,
            work_dir=work_dir,
            cache_dir=cache_dir,
            network_gpkg=network_gpkg,
            build_domain_masks_fn=_build_domain_masks,
            estuary_clip_fn=_clip_channel_mask_for_estuary,
            run_command_fn=run_command,
            validate_xs_artifacts_fn=validate_xs_artifacts,
            validate_soundings_subset_fn=_validate_soundings_subset,
            append_river_soundings_args_fn=_append_river_soundings_args,
            build_river_skeleton_command_fn=_build_river_skeleton_command,
            authoritative_passthrough_args_fn=_authoritative_passthrough_args,
            record_authoritative_child_passthrough_fn=_record_authoritative_child_passthrough,
            logger=log,
        )
        bed_tif = hybrid_result.merged_bed_tif
        channel_mask_tif = hybrid_result.channel_mask_tif
        mainstem_mask_tif = hybrid_result.mainstem_mask_tif
        report.setdefault('river', {}).setdefault('outputs', {}).update(hybrid_result.as_report_outputs())
        report.setdefault('river', {}).setdefault('hybrid_combine', {}).update(hybrid_result.merge_receipt)
        report.setdefault('river', {}).setdefault('hybrid_stage', {
            'status': 'success',
            'xs_gpkg': str(hybrid_result.xs_gpkg),
            'soundings_subset_path': str(hybrid_result.soundings_subset_path),
            'merged_bed_tif': str(hybrid_result.merged_bed_tif),
            'hybrid_merge_receipt_json': str(hybrid_result.hybrid_merge_receipt_json),
        })

    if river_method == "hybrid":
        # HYBRID handled above (bed_tif already built)
        pass
    elif river_method == "skeleton":
        result = _run_river_skeleton(
            cfg, report, network_gpkg, work_dir, cached_bed_tif, script_dir, river_dir,
        )
        if result is None:
            return None
        bed_tif, channel_mask_tif = result

    else:
        result = _run_river_xs(
            cfg, report, network_gpkg, work_dir, cached_bed_tif, script_dir, river_dir,
        )
        if result is None:
            return None
        bed_tif, channel_mask_tif, xs_meta_json, xs_acct_json = result

    report["river"]["status"] = "success"
    report["river"]["status_family"] = "success"
    report["river"]["execution_mode"] = "hybrid_full" if river_method == "hybrid" else river_method
    log.info("[RIVER] Success: %s", bed_tif)

    # Template cache persistence: if a template was built by XS/hybrid, save to shared cache
    # so subsequent runs with the same AOI+network reuse it without recomputation.
    try:
        _template_artifacts = _resolve_channel_template_artifacts(cfg, work_dir=work_dir, river_dir=river_dir)
        _tpl_src = _template_artifacts["template_json"] if _template_artifacts["template_json"].exists() else None
        if _tpl_src is not None and cfg.river_channel_template_enabled:
            _tpl_cache_dir = ensure_dir(cfg.cache_root / "channel_template" / _channel_template_cache_key(cfg))
            _tpl_cache_dst = _tpl_cache_dir / "channel_template.json"
            if not _tpl_cache_dst.exists():
                shutil.copy2(_tpl_src, _tpl_cache_dst)
                log.info("[RIVER] Template cached: %s", _tpl_cache_dst)
            report.setdefault("river", {}).setdefault("outputs", {})["channel_template_cache"] = str(_tpl_cache_dst)
    except Exception:
        log.debug("[RIVER] Template cache write failed", exc_info=True)

    # Mask the final river bed raster with the clipped channel mask.
    # The skeleton's Gaussian smoothing and XS interpolation can extend
    # slightly beyond the channel mask boundary, creating thin strips of
    # river data in the estuary zone.  Re-masking ensures the river output
    # strictly respects the clipped channel domain.
    try:
        if channel_mask_tif is not None and Path(channel_mask_tif).exists() and Path(bed_tif).exists():
            import rasterio as _rio_mask
            with _rio_mask.open(channel_mask_tif) as cm:
                ch_arr = cm.read(1)
            with _rio_mask.open(bed_tif) as bt:
                bed_arr = bt.read(1).astype("float32")
                bed_prof = bt.profile.copy()
                bed_nod = bt.nodata if bt.nodata is not None else float(cfg.river_nodata)
            outside_channel = (ch_arr == 0)
            n_masked = int((np.isfinite(bed_arr) & (bed_arr != bed_nod) & outside_channel).sum())
            if n_masked > 0:
                bed_arr[outside_channel] = bed_nod
                bed_prof.update(dtype="float32")
                with _rio_mask.open(bed_tif, "w", **bed_prof) as dst:
                    dst.write(bed_arr, 1)
                log.info("[RIVER] Post-mask: removed %d bed-elevation pixels outside clipped channel mask", n_masked)
    except Exception:
        log.debug("[RIVER] Post-mask failed", exc_info=True)

    # Capture explicit XS constraint meta + accounting (no guessing).
    try:
        import json as _json
        if xs_meta_json is not None and xs_meta_json.exists():
            meta = _json.loads(xs_meta_json.read_text(encoding="utf-8"))
            report.setdefault("river", {}).setdefault("constraints", {})["xs"] = meta.get("constraints", {})
            report.setdefault("river", {}).setdefault("outputs", {})["xs_constraint_meta"] = str(xs_meta_json)
        if xs_acct_json is not None and xs_acct_json.exists():
            acct = _json.loads(xs_acct_json.read_text(encoding="utf-8"))
            report.setdefault("river", {}).setdefault("constraint_accounting", {})["xs"] = acct
            report.setdefault("river", {}).setdefault("outputs", {})["xs_constraint_accounting"] = str(xs_acct_json)
    except Exception:
        log.debug("Failed to read XS constraint meta/accounting; continuing.", exc_info=True)

    # Gap-fill: fill channel-mask pixels where XS inference left nodata using a
    # lightweight skeleton depth prior (bank-derived WSE, width→depth, r^exp profile).
    # This eliminates the coverage gaps between NHD area polygons.
    try:
        # Use channel template coefficients if available (learned from measured XS)
        _gf_a = cfg.river_mv_a0
        _gf_b = cfg.river_mv_bw
        _gf_exp = cfg.river_shape_exp
        if cfg.river_channel_template_enabled:
            try:
                _gf_tpl, _gf_data = _load_channel_template_json(cfg, work_dir=work_dir, river_dir=river_dir)
                if str(_gf_data.get("depth_fit_source", "")) == "local_power_law":
                    _gf_a = float(_gf_data.get("depth_a", _gf_a))
                    _gf_b = float(_gf_data.get("depth_b", _gf_b))
                    _gf_exp = float(_gf_data.get("shape_exponent", _gf_exp))
                    log.info("[GAP-FILL] Using channel template from %s: a=%.4f b=%.3f exp=%.3f", _gf_tpl, _gf_a, _gf_b, _gf_exp)
            except RuntimeError as exc:
                log.debug("[GAP-FILL] Failed to load template systematically; using defaults. %s", exc, exc_info=True)

        if river_method == "hybrid":
            _gap_filled = 0
            log.info("[RIVER] Gap-fill skipped for hybrid method: skeleton prior demoted from peer bed product.")
            report.setdefault("river", {}).setdefault("gap_fill", {})["status"] = "skipped_hybrid_phase4"
        else:
            _gap_filled = _gap_fill_bed_with_skeleton_prior(
                bed_tif=bed_tif,
                dem_tif=Path(cfg.river_dem),
                channel_mask_tif=Path(channel_mask_tif) if channel_mask_tif else None,
                a=_gf_a,
                b=_gf_b,
                shape_exp=_gf_exp,
                dmin=cfg.river_dmax_min_m,
                dmax=cfg.river_dmax_max_m,
                nodata=float(cfg.river_nodata),
                logger=log,
                confidence_tif=river_dir / "river_gap_fill_confidence.tif" if river_dir else None,
                confidence_sigma_m=float(getattr(cfg, "river_channel_template_distance_sigma_m", 2000.0) or 2000.0),
            )
            if _gap_filled > 0:
                log.info("[RIVER] Gap-fill: filled %d channel pixels with skeleton depth prior.", _gap_filled)
                report.setdefault("river", {}).setdefault("gap_fill", {})["n_filled"] = _gap_filled
    except Exception:
        log.debug("[RIVER] Gap-fill failed; continuing with sparse coverage.", exc_info=True)

    # Materialize into run output folder for convenience
    # Compute depth relative to DEM terrain surface (negative down): depth = bed_elev - dem
    try:
        depth_surface_recovery = compute_depth_from_bed_and_dem(bed_tif, Path(cfg.river_dem), cached_depth_tif, depth_sign="negative_down", channel_mask_tif=(Path(channel_mask_tif) if channel_mask_tif is not None else None))
        depth_surface_recovery_contract_path = river_dir / "river_depth_surface_recovery_contract.json"
        write_json(depth_surface_recovery_contract_path, depth_surface_recovery)
        report.setdefault("river", {}).setdefault("execution_receipts", {})["depth_surface_recovery"] = depth_surface_recovery
        report.setdefault("river", {}).setdefault("outputs", {})["depth_surface_recovery_contract"] = str(depth_surface_recovery_contract_path)
    except Exception as e:
        log.error("[RIVER] Failed to compute depth from bed elevation and DEM: %s", e)
        report["river"]["status"] = "failed"
        return None

    # Sanitize cached river rasters before any downstream delivery / warping. This strips NaN/Inf,
    # float32-max sentinels, and obviously impossible extremes that can leak through sparse raster math
    # or reprojection edge cases and then appear as invalid values in GIS.
    try:
        depth_stats = sanitize_raster_values(Path(cached_depth_tif), nodata=cfg.river_nodata, min_valid=-100.0, max_valid=100.0)
        bed_stats = sanitize_raster_values(Path(bed_tif), nodata=cfg.river_nodata, min_valid=-500.0, max_valid=500.0)
        report.setdefault("river", {}).setdefault("sanitation", {})["cached_depth"] = depth_stats
        report.setdefault("river", {}).setdefault("sanitation", {})["cached_bed"] = bed_stats
        log.info(
            "[RIVER] Sanitized cached rasters: depth(valid=%d nf=%d extreme=%d range=%d) | bed(valid=%d nf=%d extreme=%d range=%d)",
            int(depth_stats.get("valid_pixels", 0)),
            int(depth_stats.get("replaced_nonfinite", 0)),
            int(depth_stats.get("replaced_extreme", 0)),
            int(depth_stats.get("replaced_range", 0)),
            int(bed_stats.get("valid_pixels", 0)),
            int(bed_stats.get("replaced_nonfinite", 0)),
            int(bed_stats.get("replaced_extreme", 0)),
            int(bed_stats.get("replaced_range", 0)),
        )
    except Exception:
        log.debug("[RIVER] Cached raster sanitation failed", exc_info=True)

    # Final hard guarantee: river outputs (bed + depth) must be nodata outside the river channel domain.
    try:
        _maskp = None
        if cfg.river_domain_mask_for_fusion:
            mp = Path(cfg.river_domain_mask_for_fusion)
            if mp.exists():
                _maskp = mp
        if _maskp is None and channel_mask_tif is not None:
            try:
                mp2 = Path(channel_mask_tif)
                if mp2.exists():
                    _maskp = mp2
            except OSError:
                log.debug("ignored", exc_info=True)
        if _maskp is not None:
            nval = cfg.river_nodata
            ok_bed = _clip_raster_to_mask(Path(bed_tif), _maskp, inside_value=1, invert=False, nodata=nval)
            ok_dep = _clip_raster_to_mask(Path(cached_depth_tif), _maskp, inside_value=1, invert=False, nodata=nval)
            report.setdefault("river", {}).setdefault("masking", {})["channel_mask_clip_bed"] = bool(ok_bed)
            report.setdefault("river", {}).setdefault("masking", {})["channel_mask_clip_depth"] = bool(ok_dep)
    except Exception:
        log.debug("ignored", exc_info=True)


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
                channel_overlap_frac = None
                try:
                    import rasterio as _rio
                    import numpy as _np
                    with _rio.open(channel_mask_tif) as _cm, _rio.open(wm) as _wm:
                        _cm_arr = _cm.read(1)
                        _wm_arr = _wm.read(1, out_shape=(_cm.height, _cm.width), resampling=_rio.enums.Resampling.nearest)
                        _channel = (_cm_arr == 1)
                        _overlap = _channel & (_wm_arr == 0)
                        channel_total = int(_channel.sum())
                        channel_overlap_frac = float(_overlap.sum() / max(1, channel_total)) if channel_total > 0 else 0.0
                except Exception:
                    log.debug('[RIVER] WAFFLES/channel overlap check failed', exc_info=True)
                if water_px is not None and water_px <= 0:
                    log.warning("[RIVER] Skipping WAFFLES mask (no water pixels in AOI): %s", wm)
                elif channel_overlap_frac is not None and channel_overlap_frac < 0.10:
                    log.warning(
                        "[RIVER] Skipping WAFFLES mask because it overlaps too little of the river corridor "
                        "(channel_overlap_frac=%.3f). This usually means the selected WAFFLES mask is ocean-only "
                        "or otherwise unsuitable for inland river delivery: %s",
                        channel_overlap_frac,
                        wm,
                    )
                else:
                    md_ok = _mask_raster_to_waffles(cached_depth_tif, wm, nodata=-9999.0)
                    mb_ok = _mask_raster_to_waffles(cached_bed_tif, wm, nodata=-9999.0)
                    report.setdefault("river", {}).setdefault("masking", {})["waffles_mask"] = str(wm)
                    report.setdefault("river", {}).setdefault("masking", {})["waffles_channel_overlap_frac"] = channel_overlap_frac
                    report.setdefault("river", {}).setdefault("masking", {})["masked_depth"] = bool(md_ok)
                    report.setdefault("river", {}).setdefault("masking", {})["masked_bed"] = bool(mb_ok)
    except Exception:
        log.debug("ignored", exc_info=True)

    # Strict safety: river deliverables must contain valid pixels and must not collapse into
    # a near-constant diagnostic surface. This catches invalid-but-finite outputs that would
    # otherwise survive nodata checks and appear as unusable rasters downstream.
    try:
        import rasterio
        import numpy as np

        def _summarize_valid(rp: Path, nodata_val: float, *, min_valid: float | None = None, max_valid: float | None = None) -> dict:
            stats = {"valid": 0, "min": np.nan, "max": np.nan, "p01": np.nan, "p99": np.nan, "frac_neg": np.nan}
            if not rp.exists():
                return stats
            vals = []
            with rasterio.open(rp) as ds:
                nod = ds.nodata
                if nod is None:
                    nod = nodata_val
                for _, w in ds.block_windows(1):
                    a = ds.read(1, window=w).astype("float32")
                    bad = nodata_mask(a, nod)
                    if min_valid is not None:
                        bad |= np.isfinite(a) & (a < float(min_valid))
                    if max_valid is not None:
                        bad |= np.isfinite(a) & (a > float(max_valid))
                    m = np.isfinite(a) & (~bad)
                    if np.any(m):
                        vals.append(a[m].astype("float64"))
            if not vals:
                return stats
            v = np.concatenate(vals)
            stats["valid"] = int(v.size)
            stats["min"] = float(np.min(v))
            stats["max"] = float(np.max(v))
            stats["p01"] = float(np.percentile(v, 1.0))
            stats["p99"] = float(np.percentile(v, 99.0))
            stats["frac_neg"] = float(np.mean(v < 0.0))
            return stats

        nd = cfg.river_nodata
        depth_summary = _summarize_valid(cached_depth_tif, nd, min_valid=-100.0, max_valid=100.0)
        bed_summary = _summarize_valid(cached_bed_tif, nd, min_valid=-500.0, max_valid=500.0)
        report.setdefault("river", {}).setdefault("validation", {})["cached_depth_summary"] = depth_summary
        report.setdefault("river", {}).setdefault("validation", {})["cached_bed_summary"] = bed_summary

        if int(depth_summary.get("valid", 0)) <= 0:
            raise RuntimeError("River depth raster has no valid pixels after masking; outputs would be all nodata.")
        if int(bed_summary.get("valid", 0)) <= 0:
            raise RuntimeError("River bed elevation raster has no valid pixels after masking; outputs would be all nodata.")
        bed_valid = int(bed_summary.get("valid", 0))
        depth_valid = int(depth_summary.get("valid", 0))
        min_depth_valid_from_bed = int(max(50, 0.50 * bed_valid))

        def _collect_depth_coverage_diagnostics() -> dict:
            diag = {
                "bed_valid": bed_valid,
                "depth_valid": depth_valid,
                "min_expected_depth_valid": min_depth_valid_from_bed,
                "channel_mask_path": str(channel_mask_tif) if channel_mask_tif is not None else None,
            }
            try:
                import rasterio
                import numpy as np
                with rasterio.open(Path(cached_bed_tif)) as bed_ds, rasterio.open(Path(cached_depth_tif)) as depth_ds, rasterio.open(Path(cfg.river_dem)) as dem_ds:
                    bed_arr = sanitize_array(bed_ds.read(1), bed_ds.nodata, dtype="float32")
                    depth_arr = sanitize_array(depth_ds.read(1), depth_ds.nodata, dtype="float32")
                    dem_arr = sanitize_array(
                        dem_ds.read(1, out_shape=(bed_ds.height, bed_ds.width), resampling=rasterio.enums.Resampling.bilinear),
                        dem_ds.nodata,
                        dtype="float32",
                    )
                    channel = np.ones(bed_arr.shape, dtype=bool)
                    if channel_mask_tif is not None and Path(channel_mask_tif).exists():
                        with rasterio.open(Path(channel_mask_tif)) as cm_ds:
                            cm = cm_ds.read(1, out_shape=(bed_ds.height, bed_ds.width), resampling=rasterio.enums.Resampling.nearest)
                        channel = cm > 0
                    bed_ok = channel & np.isfinite(bed_arr)
                    depth_ok = channel & np.isfinite(depth_arr)
                    dem_ok = channel & np.isfinite(dem_arr)
                    bed_and_dem_ok = bed_ok & dem_ok
                    diag.update({
                        "channel_pixels": int(np.count_nonzero(channel)),
                        "bed_valid_in_channel": int(np.count_nonzero(bed_ok)),
                        "dem_valid_in_channel": int(np.count_nonzero(dem_ok)),
                        "bed_dem_overlap_valid": int(np.count_nonzero(bed_and_dem_ok)),
                        "depth_valid_in_channel": int(np.count_nonzero(depth_ok)),
                        "depth_vs_bed_dem_overlap_ratio": float(np.count_nonzero(depth_ok) / max(1, np.count_nonzero(bed_and_dem_ok))),
                    })
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                log.debug("[RIVER] Failed to collect sparse depth diagnostics", exc_info=True)
            return diag

        if depth_valid < min_depth_valid_from_bed:
            sparse_diag = _collect_depth_coverage_diagnostics()
            report.setdefault("river", {}).setdefault("validation", {})["sparse_depth_diagnostics"] = sparse_diag
            log.warning(
                "[RIVER] Depth coverage is sparse relative to bed coverage; attempting support-based recovery "
                "before failing (depth_valid=%d bed_valid=%d min_expected=%d dem_overlap=%s channel_pixels=%s)",
                depth_valid,
                bed_valid,
                min_depth_valid_from_bed,
                sparse_diag.get("bed_dem_overlap_valid"),
                sparse_diag.get("channel_pixels"),
            )
            recovery = _recover_river_depth_from_support(
                cfg=cfg,
                depth_tif=Path(cached_depth_tif),
                bed_tif=Path(cached_bed_tif),
                channel_mask_tif=Path(channel_mask_tif) if channel_mask_tif is not None else None,
                logger=log,
            )
            report.setdefault("river", {}).setdefault("validation", {})["depth_recovery_sparse"] = recovery
            depth_summary = _summarize_valid(cached_depth_tif, nd, min_valid=-100.0, max_valid=100.0)
            report.setdefault("river", {}).setdefault("validation", {})["cached_depth_summary"] = depth_summary
            depth_valid = int(depth_summary.get("valid", 0))
            if depth_valid < min_depth_valid_from_bed:
                sparse_diag = _collect_depth_coverage_diagnostics()
                report.setdefault("river", {}).setdefault("validation", {})["sparse_depth_diagnostics_post_recovery"] = sparse_diag
                raise RuntimeError(
                    "River depth raster retained far too few valid pixels relative to river bed coverage "
                    f"(depth_valid={depth_valid}, bed_valid={bed_valid}, "
                    f"min_expected_depth_valid={min_depth_valid_from_bed}, "
                    f"bed_dem_overlap_valid={sparse_diag.get('bed_dem_overlap_valid')}, "
                    f"channel_pixels={sparse_diag.get('channel_pixels')}). "
                    "This indicates depth derivation is still collapsing after support-based recovery. "
                    "Inspect river.validation.sparse_depth_diagnostics in the run report."
                )

        depth_span = float(depth_summary["p99"] - depth_summary["p01"]) if np.isfinite(depth_summary["p99"]) and np.isfinite(depth_summary["p01"]) else np.nan
        bed_span = float(bed_summary["p99"] - bed_summary["p01"]) if np.isfinite(bed_summary["p99"]) and np.isfinite(bed_summary["p01"]) else np.nan
        depth_degenerate = bool(np.isfinite(depth_span) and depth_span < 0.05)
        depth_not_negative = bool(np.isfinite(depth_summary["frac_neg"]) and depth_summary["frac_neg"] < 0.01 and float(depth_summary["max"]) <= 1.0)
        if depth_degenerate or depth_not_negative:
            recovery = _recover_river_depth_from_support(
                cfg=cfg,
                depth_tif=Path(cached_depth_tif),
                bed_tif=Path(cached_bed_tif),
                channel_mask_tif=Path(channel_mask_tif) if channel_mask_tif is not None else None,
                logger=log,
            )
            report.setdefault("river", {}).setdefault("validation", {})["depth_recovery"] = recovery
            if recovery.get("recovered"):
                depth_summary = _summarize_valid(cached_depth_tif, nd, min_valid=-100.0, max_valid=100.0)
                report.setdefault("river", {}).setdefault("validation", {})["cached_depth_summary"] = depth_summary
                depth_span = float(depth_summary["p99"] - depth_summary["p01"]) if np.isfinite(depth_summary["p99"]) and np.isfinite(depth_summary["p01"]) else np.nan
                depth_degenerate = bool(np.isfinite(depth_span) and depth_span < 0.05)
                depth_not_negative = bool(np.isfinite(depth_summary["frac_neg"]) and depth_summary["frac_neg"] < 0.01 and float(depth_summary["max"]) <= 1.0)
        if depth_degenerate:
            raise RuntimeError(
                f"River depth raster collapsed to a near-constant surface (p01={depth_summary['p01']:.3f}, p99={depth_summary['p99']:.3f})."
            )
        if np.isfinite(bed_span) and bed_span < 0.05:
            raise RuntimeError(
                f"River bed raster collapsed to a near-constant surface (p01={bed_summary['p01']:.3f}, p99={bed_summary['p99']:.3f})."
            )
        if depth_not_negative:
            raise RuntimeError(
                f"River depth raster is not behaving like negative-down terrain depth (frac_neg={depth_summary['frac_neg']:.3f}, min={depth_summary['min']:.3f}, max={depth_summary['max']:.3f})."
            )

        log.info(
            "[RIVER] Cached raster validation: depth(valid=%d p01=%.3f p99=%.3f frac_neg=%.3f) | bed(valid=%d p01=%.3f p99=%.3f)",
            int(depth_summary["valid"]), float(depth_summary["p01"]), float(depth_summary["p99"]), float(depth_summary["frac_neg"]),
            int(bed_summary["valid"]), float(bed_summary["p01"]), float(bed_summary["p99"]),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
        log.error("%s", e)
        report["river"]["status"] = "failed"
        raise

    out_depth = river_dir / "river_depth_terrain_patch.tif"
    out_bed = river_dir / "river_bottom_navd88_patch.tif"
    for src, dst in [(cached_depth_tif, out_depth), (bed_tif, out_bed)]:
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)
        except (AttributeError, NotImplementedError, OSError, shutil.Error):
            log.debug("bathy_main: suppressed exception", exc_info=True)
            shutil.copy(src, dst)

    try:
        apply_depth_metadata(out_depth, depth_reference="terrain_surface")
    except (OSError, RuntimeError, TypeError, ValueError):
        log.debug("ignored", exc_info=True)
    try:
        apply_elevation_metadata(out_bed, vertical_datum="NAVD88")
    except (OSError, RuntimeError, TypeError, ValueError):
        log.debug("ignored", exc_info=True)

    report.setdefault("river", {}).setdefault("outputs", {}).update(
        {
            "depth_terrain": str(out_depth),
            "bottom_elevation": str(out_bed),
            "bottom_elevation_internal_helper": str(out_bed),
        }
    )

    _write_guidance_artifacts_with_reporting(
        writer=_write_river_guidance_artifacts,
        cfg=cfg,
        out_bed=Path(out_bed),
        out_depth=Path(out_depth),
        channel_mask_tif=(Path(channel_mask_tif) if channel_mask_tif is not None else None),
        river_dir=river_dir,
        report=report,
        logger=log,
    )

    _build_constraint_summary(cfg, report)

    return out_depth

# -----------------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------------

def _find_sdb_guidance_artifact(sdb_dir: Path, key: str, suffix: str) -> Optional[Path]:
    """Resolve an SDB guidance raster from the explicit SDB manifest first, then a deterministic sibling path.

    This keeps the bathy_main side no-guess when the manifest is present, but remains usable with
    older manifests that only recorded the depth raster.
    """
    if not sdb_dir.exists():
        return None
    manifest = sdb_dir / "artifacts_sdb.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            rel = data.get(key)
            if isinstance(rel, str) and rel.strip():
                p = (sdb_dir / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
                if p.exists():
                    return p
            depth_rel = data.get("sdb_guidance_active") or data.get("depth_raster")
            if isinstance(depth_rel, str) and depth_rel.strip():
                dp = (sdb_dir / depth_rel).resolve() if not os.path.isabs(depth_rel) else Path(depth_rel).resolve()
                cand = dp.with_name(dp.stem + suffix + dp.suffix)
                if cand.exists():
                    return cand
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            log.debug("Failed to resolve SDB guidance artifact %s from manifest", key, exc_info=True)
    return None


def _grid_pixel_size_m(transform, crs, ref_lat_deg: Optional[float] = None) -> float:
    """Return a representative pixel size in meters for support-distance calculations."""
    try:
        dx = abs(float(getattr(transform, "a", 0.0) or 0.0))
        dy = abs(float(getattr(transform, "e", 0.0) or 0.0))
    except Exception:
        log.debug("_grid_pixel_size_m: suppressed exception", exc_info=True)
        dx = dy = 0.0
    px = max((dx + dy) / 2.0, 0.0)
    try:
        is_geographic = bool(getattr(crs, "is_geographic", False))
    except Exception:
        log.debug("_grid_pixel_size_m: suppressed exception", exc_info=True)
        is_geographic = False
    if not is_geographic:
        return max(px, 1.0)
    lat = float(ref_lat_deg if ref_lat_deg is not None else 0.0)
    lat_rad = np.deg2rad(lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2.0 * lat_rad) + 1.175 * np.cos(4.0 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3.0 * lat_rad)
    dx_m = dx * max(abs(m_per_deg_lon), 1.0)
    dy_m = dy * max(abs(m_per_deg_lat), 1.0)
    return max((dx_m + dy_m) / 2.0, 1.0)


def _compute_support_distance_density_guidance(
    locked: np.ndarray,
    auth: np.ndarray,
    *,
    pixel_size_m: float,
    support_decay_m: float,
    density_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper around authoritative_conditioning helper."""
    return _ac_compute_support_distance_density_guidance(
        locked,
        auth,
        pixel_size_m=pixel_size_m,
        support_decay_m=support_decay_m,
        density_radius_m=density_radius_m,
    )


def _compute_river_anchor_support_fields(
    river_anchor: np.ndarray,
    river_guidance_weight: Optional[np.ndarray],
    river_domain: np.ndarray,
    *,
    pixel_size_m: float,
    density_radius_m: float,
    scaffold_transition_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper around authoritative_conditioning helper."""
    return _ac_compute_river_anchor_support_fields(
        river_anchor=river_anchor,
        river_guidance_weight=river_guidance_weight,
        river_domain=river_domain,
        pixel_size_m=pixel_size_m,
        density_radius_m=density_radius_m,
        scaffold_transition_m=scaffold_transition_m,
    )


def _compute_coastal_sdb_support_confidence(
    sdb_domain: np.ndarray,
    sdb_guidance_weight: Optional[np.ndarray],
    sdb_trusted_interior: Optional[np.ndarray],
    support_distance_m: np.ndarray,
    support_density: np.ndarray,
    *,
    support_transition_m: float = 600.0,
) -> np.ndarray:
    """Compatibility wrapper around authoritative_conditioning helper."""
    return _ac_compute_coastal_sdb_support_confidence(
        sdb_domain=sdb_domain,
        sdb_guidance_weight=sdb_guidance_weight,
        sdb_trusted_interior=sdb_trusted_interior,
        support_distance_m=support_distance_m,
        support_density=support_density,
        support_transition_m=support_transition_m,
    )


def _support_weighted_condition_arrays(
    *,
    candidate: Optional[np.ndarray] = None,
    background_surface: Optional[np.ndarray] = None,
    auth: np.ndarray,
    sdb_depth_guidance: Optional[np.ndarray] = None,
    river_depth_guidance: Optional[np.ndarray] = None,
    primary_river_guidance_surface: Optional[np.ndarray] = None,
    sdb_guide_points_path: Optional[str] = None,
    river_guide_points_path: Optional[str] = None,
    guidance_template_raster: Optional[str] = None,
    sdb_ok: np.ndarray,
    river_ok: np.ndarray,
    sdb_gw: Optional[np.ndarray],
    sdb_ti: Optional[np.ndarray],
    river_gw: Optional[np.ndarray],
    river_ti: Optional[np.ndarray],
    river_support: Optional[np.ndarray],
    river_support_depth: Optional[np.ndarray],
    estuary_transition: Optional[np.ndarray] = None,
    river_corridor_mask: Optional[np.ndarray] = None,
    river_bank_influence: Optional[np.ndarray] = None,
    river_bank_elevation: Optional[np.ndarray] = None,
    river_bank_pair_weight: Optional[np.ndarray] = None,
    river_bank_continuity_weight: Optional[np.ndarray] = None,
    river_bank_graph_confidence: Optional[np.ndarray] = None,
    river_bank_confluence_damping: Optional[np.ndarray] = None,
    river_bank_estuary_side_decay: Optional[np.ndarray] = None,
    river_centerline_elevation: Optional[np.ndarray] = None,
    river_centerline_influence: Optional[np.ndarray] = None,
    river_centerline_stationing: Optional[np.ndarray] = None,
    river_channel_surface: Optional[np.ndarray] = None,
    river_channel_surface_confidence: Optional[np.ndarray] = None,
    river_channel_surface_source_class: Optional[np.ndarray] = None,
    river_channel_surface_support_count: Optional[np.ndarray] = None,
    river_channel_surface_authoritative_lock_scope: Optional[np.ndarray] = None,
    river_channel_surface_authoritative_lock_applied: Optional[np.ndarray] = None,
    river_longitudinal_profile_local_authoritative_reconciliation: Optional[np.ndarray] = None,
    river_longitudinal_profile_local_authoritative_reconciliation_influence: Optional[np.ndarray] = None,
    river_xs_support_elevation: Optional[np.ndarray] = None,
    river_xs_support_weight: Optional[np.ndarray] = None,
    sdb_uncertainty: Optional[np.ndarray] = None,
    river_uncertainty: Optional[np.ndarray] = None,
    pixel_size_m: float,
    support_decay_m: float,
    support_density_radius_m: float,
    coastal_sdb_support_transition_m: float,
    river_anchor_density_radius_m: float,
    river_scaffold_transition_m: float,
    river_aniso_along_scale_m: float = 500.0,
    river_aniso_cross_scale_m: float = 30.0,
    river_contract_mode: str = "canonical_v322",
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    """Compatibility wrapper around authoritative_conditioning helper."""
    return _ac_support_weighted_condition_arrays(
        candidate=candidate,
        background_surface=background_surface,
        auth=auth,
        sdb_depth_guidance=sdb_depth_guidance,
        river_depth_guidance=river_depth_guidance,
        primary_river_guidance_surface=primary_river_guidance_surface,
        sdb_guide_points_path=sdb_guide_points_path,
        river_guide_points_path=river_guide_points_path,
        guidance_template_raster=guidance_template_raster,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_gw=sdb_gw,
        sdb_ti=sdb_ti,
        river_gw=river_gw,
        river_ti=river_ti,
        river_support=river_support,
        river_support_depth=river_support_depth,
        estuary_transition=estuary_transition,
        river_corridor_mask=river_corridor_mask,
        river_bank_influence=river_bank_influence,
        river_bank_elevation=river_bank_elevation,
        river_bank_pair_weight=river_bank_pair_weight,
        river_bank_continuity_weight=river_bank_continuity_weight,
        river_bank_graph_confidence=river_bank_graph_confidence,
        river_bank_confluence_damping=river_bank_confluence_damping,
        river_bank_estuary_side_decay=river_bank_estuary_side_decay,
        river_centerline_elevation=river_centerline_elevation,
        river_centerline_influence=river_centerline_influence,
        river_centerline_stationing=river_centerline_stationing,
        river_channel_surface=river_channel_surface,
        river_channel_surface_confidence=river_channel_surface_confidence,
        river_channel_surface_source_class=river_channel_surface_source_class,
        river_channel_surface_support_count=river_channel_surface_support_count,
        river_channel_surface_authoritative_lock_scope=river_channel_surface_authoritative_lock_scope,
        river_channel_surface_authoritative_lock_applied=river_channel_surface_authoritative_lock_applied,
        river_longitudinal_profile_local_authoritative_reconciliation=river_longitudinal_profile_local_authoritative_reconciliation,
        river_longitudinal_profile_local_authoritative_reconciliation_influence=river_longitudinal_profile_local_authoritative_reconciliation_influence,
        river_xs_support_elevation=river_xs_support_elevation,
        river_xs_support_weight=river_xs_support_weight,
        sdb_uncertainty=sdb_uncertainty,
        river_uncertainty=river_uncertainty,
        pixel_size_m=pixel_size_m,
        support_decay_m=support_decay_m,
        support_density_radius_m=support_density_radius_m,
        coastal_sdb_support_transition_m=coastal_sdb_support_transition_m,
        river_anchor_density_radius_m=river_anchor_density_radius_m,
        river_scaffold_transition_m=river_scaffold_transition_m,
        river_aniso_along_scale_m=river_aniso_along_scale_m,
        river_aniso_cross_scale_m=river_aniso_cross_scale_m,
        river_contract_mode=river_contract_mode,
        **legacy_kwargs,
    )


def _build_source_aware_candidate_arrays(
    *,
    legacy_candidate: Optional[np.ndarray],
    sdb_candidate: Optional[np.ndarray],
    river_candidate: Optional[np.ndarray],
    sdb_ok: np.ndarray,
    river_ok: np.ndarray,
    sdb_guidance_weight: Optional[np.ndarray],
    sdb_trusted_interior: Optional[np.ndarray],
    river_guidance_weight: Optional[np.ndarray],
    river_trusted_interior: Optional[np.ndarray],
    estuary_transition: Optional[np.ndarray] = None,
    allow_legacy_backstop: bool = True,
) -> Dict[str, Any]:
    """Compatibility wrapper around authoritative_conditioning helper."""
    return _ac_build_source_aware_candidate_arrays(
        legacy_candidate=legacy_candidate,
        sdb_candidate=sdb_candidate,
        river_candidate=river_candidate,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_guidance_weight=sdb_guidance_weight,
        sdb_trusted_interior=sdb_trusted_interior,
        river_guidance_weight=river_guidance_weight,
        river_trusted_interior=river_trusted_interior,
        estuary_transition=estuary_transition,
        allow_legacy_backstop=allow_legacy_backstop,
    )




def _resolve_existing_output_path(mapping: Any, key: str) -> Optional[Path]:
    """Resolve a file path from a mapping-like outputs object conservatively."""
    if not isinstance(mapping, dict):
        return None
    cand = mapping.get(key)
    if not cand:
        return None
    try:
        p = Path(str(cand))
    except (TypeError, ValueError):
        return None
    try:
        return p if p.exists() else None
    except OSError:
        return None


def _condition_final_to_authoritative_base(
    cfg: BathyConfig,
    candidate_path: Optional[Path],
    provenance_path: Optional[Path],
    report: Dict[str, Any],
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Build the final support-aware DEM route via dedicated final-route stages."""
    paths = collect_final_route_inputs(cfg=cfg, candidate_path=candidate_path, report=report)
    if paths is None:
        return candidate_path, provenance_path, None, None, None, None

    guidance = assemble_guidance_inputs(
        cfg=cfg,
        paths=paths,
        candidate_path=candidate_path,
        provenance_path=provenance_path,
        report=report,
    )
    terrain = run_deterministic_terrain_stage(
        guidance=guidance,
        template_path=str(paths.template_path),
        receipt_path=str(paths.terrain_receipt_path),
    )
    return write_final_route_outputs(
        cfg=cfg,
        paths=paths,
        guidance=guidance,
        terrain=terrain,
        candidate_path=candidate_path,
        provenance_path=provenance_path,
        report=report,
    )




def _write_single_source_diagnostic_with_cudem_background(
    cfg: BathyConfig,
    report: Dict[str, Any],
    *,
    source_raster: Path,
    out_depth: Path,
    mode: str,
) -> Dict[str, Any]:
    """Write a full-AOI diagnostic candidate for single-source fusion modes.

    When a cached CUDEM baseline interpolation exists, use it as the background
    surface and overlay the finite single-source raster values onto that grid.
    This keeps ``bathy_combined_depth.tif`` physically interpretable in river-only
    or sdb-only runs and prevents sparse source rasters from surfacing nodata-
    driven artifacts as if they were valid bathymetry.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    ensure_dir(out_depth.parent)
    baseline_path = _resolve_baseline_cudem_interpolation(cfg, report)
    info: Dict[str, Any] = {
        "mode": mode,
        "source_raster": str(source_raster),
        "baseline_cudem_interpolation": str(baseline_path) if baseline_path is not None else None,
        "used_baseline_background": False,
    }

    if baseline_path is not None and Path(baseline_path).exists():
        with rasterio.open(baseline_path) as bg_ds:
            bg_arr = sanitize_array(bg_ds.read(1), bg_ds.nodata, dtype="float32")
            out_nodata = float(bg_ds.nodata if bg_ds.nodata is not None else -9999.0)
            profile = bg_ds.profile.copy()
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.update(dtype="float32", count=1, nodata=out_nodata, compress="DEFLATE", predictor=2, tiled=False)

            overlay = np.full((bg_ds.height, bg_ds.width), out_nodata, dtype="float32")
            with rasterio.open(source_raster) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=overlay,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=resolve_reproject_nodata_value(src.nodata, default=-9999.0),
                    dst_transform=bg_ds.transform,
                    dst_crs=bg_ds.crs,
                    dst_nodata=out_nodata,
                    resampling=Resampling.bilinear,
                )
            overlay = sanitize_array(overlay, out_nodata, dtype="float32")
            out_arr = bg_arr.copy()
            take_overlay = np.isfinite(overlay)
            if np.any(take_overlay):
                out_arr[take_overlay] = overlay[take_overlay]
            out_write = np.where(np.isfinite(out_arr), out_arr, out_nodata).astype("float32")
            with rasterio.open(out_depth, "w", **profile) as dst:
                dst.write(out_write, 1)
            info["used_baseline_background"] = True
            info["background_grid"] = str(baseline_path)
            info["overlay_valid_pixels"] = int(np.count_nonzero(np.isfinite(overlay)))
            info["output_valid_pixels"] = int(np.count_nonzero(np.isfinite(out_arr)))
            return info

    with rasterio.open(source_raster) as src:
        arr = sanitize_array(src.read(1), src.nodata, dtype="float32")
        out_nodata = float(src.nodata if src.nodata is not None else -9999.0)
        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(dtype="float32", count=1, nodata=out_nodata, compress="DEFLATE", predictor=2, tiled=False)
        out_write = np.where(np.isfinite(arr), arr, out_nodata).astype("float32")
        with rasterio.open(out_depth, "w", **profile) as dst:
            dst.write(out_write, 1)
        info["output_valid_pixels"] = int(np.count_nonzero(np.isfinite(arr)))
        return info

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
        diag = _write_single_source_diagnostic_with_cudem_background(
            cfg,
            report,
            source_raster=Path(river_raster),
            out_depth=out_depth,
            mode="river_only",
        )
        report["fusion"] = {"status": "success", "mode": "river_only", "outputs": {"depth": str(out_depth)}, "diagnostic_candidate": diag}
        return out_depth

    if river_raster is None and sdb_raster is not None:
        diag = _write_single_source_diagnostic_with_cudem_background(
            cfg,
            report,
            source_raster=Path(sdb_raster),
            out_depth=out_depth,
            mode="sdb_only",
        )
        report["fusion"] = {"status": "success", "mode": "sdb_only", "outputs": {"depth": str(out_depth)}, "diagnostic_candidate": diag}
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

                # 0.0 on land is a mask artifact, not valid depth
                invalid |= ((sdb_arr == 0.0) & (land == 1))

                sdb_arr2 = sdb_arr.copy()
                sdb_arr2[invalid] = prof["nodata"]

                with rasterio.open(str(sanitized), "w", **prof) as dst:
                    dst.write(sdb_arr2.astype("float32"), 1)

            sdb_fuse_path = sanitized
            report.setdefault("fusion", {}).setdefault("inputs_sanitized", {})["sdb_landmask_applied"] = str(lm)
    except Exception:
        log.debug("ignored", exc_info=True)
    # Heuristic: if the SDB raster is overwhelmingly literal 0.0 (common failure mode),
    # treat 0.0 as nodata for fusion so river can gap-fill.
    try:
        from osgeo import gdal
        gdal.UseExceptions()
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
        log.debug("ignored", exc_info=True)


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
            log.debug("ignored", exc_info=True)

        if pri == "river":
            primary, secondary = riv_a, sdb_a
        else:
            primary, secondary = sdb_a, riv_a

        out = primary.copy()
        take = valid_mask(secondary, nodata) & (~valid_mask(out, nodata))
        out[take] = secondary[take]

        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(out.astype(np.float32), 1)


    def _gdal_union_overlay(sdb_path: Path, river_path: Path, out_path: Path, template_path: Path, pri: str) -> None:
        """GDAL-based union overlay fallback (no rasterio dependency).
        Aligns both rasters onto the template grid, then fills primary nodata with secondary values.
        """
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            import numpy as np
        except (ImportError, ModuleNotFoundError) as ee:
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
            log.debug("ignored", exc_info=True)

        # ensure finite
        sdb_a[~np.isfinite(sdb_a)] = nodata
        riv_a[~np.isfinite(riv_a)] = nodata

        if pri == "river":
            primary, secondary = riv_a, sdb_a
        else:
            primary, secondary = sdb_a, riv_a

        out = primary.copy()
        take = valid_mask(secondary, nodata) & (~valid_mask(out, nodata))
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
        river_domain_mask_for_fusion = _resolve_river_domain_mask_for_fusion(cfg, report)

        if river_domain_mask_for_fusion is not None and river_domain_mask_for_fusion.exists():
            report.setdefault("fusion", {}).setdefault("domain", {})["river_domain_mask"] = str(river_domain_mask_for_fusion)
            report["fusion"]["domain"]["river_overrides_sdb_in_domain"] = True
        else:
            report.setdefault("fusion", {}).setdefault("domain", {})["river_domain_mask"] = None
            report["fusion"]["domain"]["river_overrides_sdb_in_domain"] = False
        cfg_fuse = FusionConfig(
            sdb_raster=Path(sdb_fuse_path) if sdb_fuse_path else None,
            river_raster=Path(river_fuse_path) if river_fuse_path else None,
            measured_raster=(Path(cfg.authoritative_base) if getattr(cfg, "authoritative_base", None) and Path(cfg.authoritative_base).exists() else None),
            dem_raster=None,
            river_domain_mask=river_domain_mask_for_fusion,
            ocean_domain_mask=cfg.ocean_domain_mask_for_fusion,
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
        out_depth, out_prov = _copy_fusion_outputs(res, out_depth, out_prov)

        report["fusion"] = {
            "status": "success",
            "mode": str(cfg.fusion_strategy),
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
            _ensure_river_contribution(
                union_overlay_fn=_simple_union_overlay,
                gdal_union_overlay_fn=_gdal_union_overlay,
                sdb_fuse_path=Path(sdb_fuse_path),
                river_fuse_path=Path(river_fuse_path),
                out_depth=Path(out_depth),
                out_prov=Path(out_prov) if out_prov else None,
                template=Path(template),
                pri=pri,
                report=report,
                logger=log,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError) as e:
            log.debug("Optional river-contribution validation failed: %s", e, exc_info=True)


        # Apply river guidance artifacts as downstream fusion controls.
        _apply_guidance_controls_with_reporting(
            apply_fn=_apply_river_guidance_to_fused_output,
            out_depth=Path(out_depth),
            river_fuse_path=Path(river_fuse_path) if river_fuse_path else None,
            sdb_fuse_path=Path(sdb_fuse_path) if sdb_fuse_path else None,
            out_prov=Path(out_prov) if out_prov else None,
            report=report,
            estuary_max_weight=float(cfg.estuary_max_weight),
            logger=log,
        )

        # ------------------------------------------------------------------
        # Enforce authoritative XYZ constraints in the final fused raster.
        # This is a hard "burn-in": at pixels containing authoritative soundings,
        # the output bed elevation must equal the sounding bed elevation exactly.
        # ------------------------------------------------------------------
        try:
            xyz_paths = _normalize_multi_path_value(getattr(cfg, 'river_soundings', None))
            _burn_authoritative_xyz_into_final(
                out_depth=Path(out_depth),
                xyz_paths=xyz_paths,
                report=report,
                logger=log,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError, KeyError) as _burn_e:
            report.setdefault('fusion', {}).setdefault('constraints', {})['xyz_burned_into_final'] = False
            report['fusion']['constraints']['xyz_burn_error'] = str(_burn_e)

        # --- Residual correction: smooth the fused output toward authoritative XYZ ---
        # After all fusion and exact burn-in, apply a Gaussian-smoothed residual
        # correction that gently pulls the prediction surface toward measured depths
        # in the gaps between exact-overwrite pixels.
        try:
            _apply_residual_correction_with_reporting(
                cfg=cfg,
                out_depth=Path(out_depth),
                report=report,
                logger=log,
                load_support_points_fn=_load_river_authoritative_support_points,
                apply_residual_correction_fn=_apply_residual_correction,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError, KeyError) as e:
            log.debug("Optional residual correction on fused output failed: %s", e, exc_info=True)

        return Path(out_depth)

    except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError, KeyError, ImportError) as e:
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
        except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError) as ee2:
            report["fusion"]["note"] = f"Union overlay fallback also failed; priority copy used. union_error={ee2}"

        # Priority copy as last resort
        if pri == "river":
            shutil.copy2(str(river_raster), str(out_depth))
        else:
            shutil.copy2(str(sdb_raster), str(out_depth))
        return out_depth


def _load_yaml_config(config_paths: "List[str]") -> "Dict[str, Any]":
    """Load and merge YAML config files (later files override earlier ones).

    Returns a flat dict of argparse-compatible key=value pairs by flattening
    the nested YAML sections.  Section keys are prefixed to avoid collisions
    (e.g., ``estuary.max_weight`` becomes ``estuary_max_weight``).
    """
    import yaml  # PyYAML — available in most scientific Python environments

    # Map from YAML section.key to argparse dest name
    _SECTION_PREFIX = {
        "core": "",
        "tiling": "",
        "model_bank": "sdb_",
        "sentinel2": "",
        "icesat2": "",
        "training": "",
        "prediction": "",
        "river_network": "river_",
        "river_method": "river_",
        "river_priors": "river_",
        "estuary": "estuary_",
        "fusion": "fusion_",
    }
    # Special mappings where YAML key != argparse dest
    _KEY_MAP = {
        "model_bank.enabled": "sdb_model_bank_enabled",
        "model_bank.bank": "sdb_model_bank",
        "model_bank.max_samples": "sdb_bank_max_samples",
        "model_bank.seed": "sdb_bank_seed",
        "model_bank.retrain_min_new": "sdb_bank_retrain_min_new",
        "estuary.transition_m": "estuary_transition_m",
        "estuary.max_weight": "estuary_max_weight",
        "estuary.width_ratio_thresh": "estuary_width_ratio_thresh",
        "estuary.connect_dist_m": "estuary_connect_dist_m",
        "river_network.dem_auto": "river_dem_auto",
        "river_network.dem_source": "river_dem_source",
        "river_network.dem_res_m": "river_dem_res_m",
        "river_method.method": "river_method",
        "core.sdb_mode": "sdb_mode",
        "core.convert_sdb_to_navd88": "convert_sdb_to_navd88",
    }

    merged: Dict[str, Any] = {}
    for cp in config_paths:
        p = Path(cp)
        if not p.exists():
            log.warning("[CONFIG] Config file not found: %s", cp)
            continue
        with open(p, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            continue
        for section, values in doc.items():
            if not isinstance(values, dict):
                continue
            prefix = _SECTION_PREFIX.get(section, section + "_")
            for key, val in values.items():
                full_key = f"{section}.{key}"
                dest = _KEY_MAP.get(full_key)
                if dest is None:
                    dest = f"{prefix}{key}" if prefix else key
                merged[dest] = val
        log.info("[CONFIG] Loaded: %s (%d values)", cp, sum(1 for s in doc.values() if isinstance(s, dict) for _ in s))
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Coastal + River Bathymetry Pipeline", conflict_handler="resolve")

    # --- Config file (Level 2 interface) ---
    p.add_argument("--config", dest="config_files", action="append", default=[],
                   help="YAML config file(s). May be specified multiple times; later files override earlier. "
                        "CLI args always override config file values. "
                        "Example: --config=config/default.yaml --config=config/us_east_coast.yaml")

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
            "Example: --buff 0.10 expands width/height by 10%% (5%% each side). Use 0 for no buffer."
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
    p.add_argument(
        "--nested-aoi-compare-with-final-outputs",
        action="append",
        default=[],
        help=(
            "Path to another run's final_outputs.json for nested-AOI / overlap regression checks. "
            "May be specified multiple times. Explicit-only (no auto-discovery)."
        ),
    )
    p.add_argument(
        "--nested-aoi-compare-with-final-outputs-list",
        type=str,
        default=None,
        help=(
            "Path to a text file listing final_outputs.json paths (one per line; '#' comments allowed) "
            "for nested-AOI / overlap regression checks."
        ),
    )
    p.add_argument(
        "--nested-aoi-overlap-tolerance",
        type=float,
        default=1.0e-6,
        help="Tolerance for common-overlap regression identity checks (default 1e-6).",
    )
    p.add_argument(
        "--nested-aoi-trusted-tolerance",
        type=float,
        default=1.0e-6,
        help="Tolerance for trusted-interior regression identity checks (default 1e-6).",
    )
    p.add_argument("--validation-truth", default=None, help="Optional truth raster aligned or alignable to the final DEM grid for support-class validation metrics.")
    p.add_argument("--validation-case", action="append", default=[], help="Optional ablation case as NAME=PATH. Repeatable. Used with --validation-truth.")
    p.add_argument("--validation-case-manifest", default=None, help="Optional JSON file mapping validation case names to raster paths.")
    p.add_argument("--validation-guidance-baseline-case", default="baseline_cudem_interpolation", help="Baseline case name for optional guidance non-degradation evaluation.")
    p.add_argument("--validation-guidance-target-case", default="selected_final", help="Target case name for optional guidance non-degradation evaluation.")
    p.add_argument("--validation-require-guidance-non-degradation", action="store_true", help="Fail the run when the guidance target case is worse than the baseline in guidance-conditioned support families, if validation truth is provided.")
    p.add_argument("--validation-guidance-rmse-tolerance", type=float, default=0.0, help="Allowed RMSE degradation for guidance-conditioned support-family comparison before validation fails.")

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
    p.add_argument("--make-figs", action="store_true", default=False,
                   help="Generate presentation-ready figure PNGs under <out-dir>/figures from explicit workflow outputs.")
    p.add_argument("--save-intermediates", action="store_true", default=False,
                   help="Preserve intermediate artifacts (rasters/caches) by moving them under <out-dir>/<intermediates-dirname>/... (default: off; intermediates are deleted).")
    p.add_argument("--intermediates-dirname", default="debug",
                   help="Subfolder name under <out-dir> to store intermediates when --save-intermediates is enabled (default: debug).")
    p.add_argument("--methods", default="sdb,river,fuse")
    p.add_argument("--review-guidance-domains-only", action="store_true", default=False,
                   help="Build and stage the pre-inference river/SDB guidance domains, then stop before any SDB or river inference runs.")
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
    p.add_argument("--river-dem-res-m", type=float, default=0.0,
                   help="Target resolution (meters) for the auto-built river DEM in working CRS. Default 0 = derive from authoritative_base grid when available; otherwise fall back to 10 m.")
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
        help=("Target CRS/compound CRS passed to dlim -P for --extra-xyz-cudem outputs. "
              "Default: epsg:4269+5714 = NAD83 + MSL height. Set this to your effective optical-training datum "
              "(for example an MTL compound CRS) when you want provider soundings fetched directly in that datum."),
    )
    p.add_argument(
        "--extra-xyz-cudem-source-vdatum",
        default=None,
        help=("Optional source compound CRS passed to dlim -J for --extra-xyz-cudem outputs. "
              "Use this when the provider soundings should be treated as a known source datum before conversion, "
              "for example NAVD88 -> MTL during fetch."),
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
    
    # River network parameters
    p.add_argument("--river-hydrography-source", default="arcgis", choices=["arcgis","arcgis_tnm","tnm"],
                   help="Hydrography acquisition for river network: arcgis (default) queries ArcGIS REST only; arcgis_tnm uses TNM as a fallback; tnm prefers TNM (and may still fall back to ArcGIS).")
    p.add_argument("--no-tnm", action="store_true", help="Disable TNM download (if using local flowlines)")
    p.add_argument("--tnm-dataset", default="NHDPlusHR")
    p.add_argument("--snap-m", type=float, default=30.0)

    # Optional drainage-area raster fallback (e.g., MERIT Hydro UPA)
    p.add_argument(
    "--river-da-raster",
    default=None,
    help=(
        "Optional drainage-area raster to sample when NHDPlus drainage area is unavailable. "
        "Example: MERIT Hydro upstream area (UPA)."
    ),
    )
    p.add_argument("--river-da-raster-band", type=int, default=1, help="Band index for --river-da-raster (1-based).")
    p.add_argument(
    "--river-da-raster-units",
    default="km2",
    choices=["km2", "m2"],
    help=("Units stored in --river-da-raster. 'km2' is typical for MERIT UPA; set explicitly."),
    )
    
    
    # River bathymetry method selection
    p.add_argument("--river-method", choices=["hybrid","skeleton", "xs", "structured", "v1", "v2", "simple_v2"], default="structured",
               help=("River bathy method. 'structured' (default) runs the channel frame/scaffold/surface path as the primary river estimator. 'hybrid' runs XS only on the mainstem and uses the skeleton method elsewhere, then combines them so mainstem depths are continuous. 'skeleton' uses a raster distance-transform channel skeleton (recommended for dense tributaries/meanders/tidal channels). 'xs' uses cross-sections everywhere (more artifact-prone at junctions)."))
    # Skeleton (distance-transform) parameters: domain mask (river vs ocean) + channel profile
    p.add_argument("--river-channel-buffer-m", type=float, default=400.0,
               help="Buffer around NHD flowlines used to define candidate river corridor (meters).")
    p.add_argument("--river-max-channel-width-m", type=float, default=600.0,
               help="Maximum channel width allowed inside the corridor (meters). Prevents filling open bays/ocean.")
    p.add_argument("--river-mainstem-method", choices=["dominant_trunk", "stream_order"], default="dominant_trunk",
               help="How to identify the canonical mainstem corridor for river guidance. dominant_trunk solves an outlet-connected weighted trunk path on the river network; stream_order keeps the legacy threshold proxy.")
    p.add_argument("--river-mainstem-solve-layer", default="auto",
               help="Layer used for dominant trunk solving. 'auto' prefers major_system_network, then mainstem_solve_network, then rivers_aoi, then rivers_clip.")
    p.add_argument("--river-mainstem-min-order", type=int, default=5,
               help="Legacy stream order threshold for allowing larger widths when --river-mainstem-method=stream_order.")
    p.add_argument("--river-max-mainstem-width-m", type=float, default=2500.0,
               help="Maximum channel width allowed for mainstem corridor pixels (meters).")
    p.add_argument("--river-guidance-bank-margin-m", type=float, default=3.0,
               help="Exclude near-bank pixels from the early river guidance domain mask by this interior margin in meters (default: 3.0).")

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
    p.add_argument("--river-domain-min-water-corridor-frac", dest="river_domain_min_water_corridor_frac", type=float, default=0.02,
               help="Minimum acceptable effective-water∩corridor overlap fraction after harmonization (default 0.02).")
    p.add_argument("--river-domain-min-channel-corridor-frac", dest="river_domain_min_channel_corridor_frac", type=float, default=0.001,
               help="Minimum acceptable channel∩corridor overlap fraction for river-domain validation (default 0.001).")
    p.add_argument("--river-domain-min-channel-pixels", dest="river_domain_min_channel_pixels", type=int, default=1,
               help="Minimum acceptable number of river channel pixels after domain harmonization (default 1).")
    p.add_argument("--river-domain-hard-fail", dest="river_domain_hard_fail", action="store_true", default=False,
               help="Fail the run when river-domain validation fails instead of only recording the failure.")
    p.add_argument("--river-ocean-keep-dist-m", dest="river_ocean_keep_dist_m", type=float, default=0.0,
               help="Allow ocean-connected water within this distance (m) of flowlines when building the river channel mask. Useful for tidal river mouths/estuaries where the mainstem is classified as ocean water. 0 disables.")
    p.add_argument("--estuary-transition-m", dest="estuary_transition_m", type=float, default=500.0,
               help="Ocean-connectivity distance (m) used to filter inland false-positive estuary detections. "
                    "Estuary pixels detected by width-ratio or low-slope must also be within this distance of the ocean to be clipped. 0 disables.")
    p.add_argument("--estuary-max-weight", dest="estuary_max_weight", type=float, default=0.25,
               help="Maximum river guidance weight (0..1) allowed in the estuary transition zone. "
                    "Lower values make the transition more conservative (less river influence near coast). Default 0.25.")
    p.add_argument("--estuary-width-ratio-thresh", dest="estuary_width_ratio_thresh", type=float, default=3.0,
               help="Width ratio threshold for estuary detection. Channel pixels wider than this multiple "
                    "of the median channel width are classified as estuarine widening. Default 3.0.")
    p.add_argument("--estuary-connect-dist-m", dest="estuary_connect_dist_m", type=float, default=200.0,
               help="Max distance (m) from ocean boundary to retain estuary detections. "
                    "Prevents inland low-slope reaches from being falsely classified as estuary. Default 200.")
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
    p.add_argument(
        "--river-xs-profile-shape",
        choices=["parabolic", "linear_trapezoid", "cosine_trapezoid"],
        default="parabolic",
        help=(
            "Cross-section depth profile family used inside the XS solver. 'parabolic' gives a rounded U-shaped "
            "profile for unconstrained XS; 'cosine_trapezoid' keeps a width-constrained trapezoid with smooth cosine side slopes."
        ),
    )
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
    p.add_argument("--river-enable-1d-energy-solver", dest="river_enable_1d_energy_solver", action="store_true",
               help="Enable XS reach-scale 1D energy-consistent depth solver (Option A). Default: off.")
    p.add_argument("--no-river-1d-energy-solver", dest="river_enable_1d_energy_solver", action="store_false",
               help="Disable river 1D energy solver.")
    p.set_defaults(river_enable_1d_energy_solver=False)
    p.add_argument("--river-energy-allow-dem-proxy-wse", dest="river_energy_allow_dem_proxy_wse", action="store_true",
               help="Allow 1D energy solver to run even when WSE anchoring is DEM/topo proxy only (no observed stage). Default: off (safety gate).")
    p.set_defaults(river_energy_allow_dem_proxy_wse=False)

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

    # ── Channel template system ──────────────────────────────────────────────
    p.add_argument("--river-centerline-influence-scale", type=float, default=1.0,
                   help="Multiplier on local centerline-core half-width: >1 broadens, <1 narrows, and very large values saturate near the bank edge rather than amplifying strength indefinitely (default: 1.0).")
    p.add_argument("--river-channel-template-enabled", action="store_true", default=False,
                   help="Enable channel template learning from sounding/DEM-calibrated XS. When enabled, this builds a normalized cross-section shape template and a locally-calibrated width→max_depth relation when sufficient measured sections are available. Disabled by default.")
    p.add_argument("--no-river-channel-template", dest="river_channel_template_enabled",
                   action="store_false", help="Disable channel template for this river run.")
    p.add_argument("--river-channel-template-min-xs", type=int, default=3,
                   help="Minimum measured XS required to build a channel template.")
    p.add_argument("--river-channel-template-fit-min-xs", type=int, default=5,
                   help="Minimum measured XS required for local width→depth power-law fit.")
    p.add_argument("--river-disable-xs-influence", action="store_true",
                   help="Turn off XS-derived river influence and use longitudinal/centerline/backbone guidance only in structured river stages.")
    p.add_argument("--river-channel-template-n-bins", type=int, default=50,
                   help="Number of normalized bins used in the learned channel template.")
    p.add_argument("--river-channel-template-min-depth-m", type=float, default=0.3,
                   help="Minimum measured depth retained when learning a channel template.")
    p.add_argument("--river-channel-template-distance-sigma-m", type=float, default=2000.0,
                   help="Along-network support decay scale (m) for channel-template influence and dense guidance weighting.")
    p.add_argument("--river-channel-template-estuary-buffer-m", type=float, default=500.0,
                   help="Estuary transition distance (m) used when tapering template guidance near the estuary.")
    p.add_argument("--river-channel-template-junction-buffer-m", type=float, default=120.0,
                   help="Confluence/junction buffer distance (m) used when screening template support near junctions.")
    p.add_argument("--river-channel-template-width-depth-ratio-max", type=float, default=None,
                   help="Optional maximum allowed width/depth ratio for measured profiles retained in the template fit.")
    p.add_argument("--river-channel-template-loo-max-rmse-norm", type=float, default=0.25,
                   help="Median normalized leave-one-out RMSE threshold above which the local width→depth fit is downgraded to fallback.")
    p.add_argument("--river-channel-template-loo-max-dmax-error-m", type=float, default=1.5,
                   help="Median absolute leave-one-out Dmax error threshold (m) above which the local width→depth fit is downgraded to fallback.")

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

    # WAFFLES domain-detection overrides
    p.add_argument("--waffles-min-water-fraction", type=float, default=0.001, metavar="FRAC",
                   help="Minimum water fraction in WAFFLES mask required to enable SDB or river "
                        "(default 0.001). Set to 0.0 to bypass the gate entirely — useful when "
                        "WAFFLES returns an all-land mask for a valid inland river AOI.")
    p.add_argument("--force-waffles-masks", action="store_true", default=False,
                   help="Delete and regenerate cached WAFFLES coastline masks. Use when a prior run "
                        "cached an all-land mask that is incorrect for the AOI.")

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
    p.add_argument("--authoritative-base", default=None,
                   help="Optional hard-locked measured-constrained raster. Finite cells are preserved exactly and only its nodata gaps are eligible for conditioned fill.")
    p.add_argument("--authoritative-base-auto", action="store_true", default=True,
                   help="Automatically build and AOI-cache authoritative_base.tif from NOAA CUDEM tile index + spatial metadata under <cache-root>/authoritative_base/. Enabled by default; use --no-authoritative-base-auto to disable, or --authoritative-base /path/to/file.tif to override.")
    p.add_argument("--no-authoritative-base-auto", dest="authoritative_base_auto", action="store_false",
                   help="Disable automatic AOI-cached authoritative_base materialization.")
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
    p.add_argument("--authoritative-support-decay-m", type=float, default=300.0,
                   help="Distance scale (meters) controlling how quickly inferred guidance can dominate away from authoritative_base hard control.")
    p.add_argument("--authoritative-support-density-radius-m", type=float, default=250.0,
                   help="Neighborhood radius (meters) used to estimate local authoritative support density for support-aware conditioning.")
    p.add_argument("--coastal-sdb-support-transition-m", type=float, default=600.0,
                   help="Distance scale (meters) controlling when stable optical SDB support can dominate in estuary/nearshore gaps away from hard control.")
    p.add_argument("--river-anchor-density-radius-m", type=float, default=200.0,
                   help="Neighborhood radius (meters) used to estimate local river anchor density during final conditioning.")
    p.add_argument("--river-scaffold-transition-m", type=float, default=800.0,
                   help="Distance scale (meters) controlling when river guidance becomes channel-structure-guidance dominant away from river anchors.")
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

    # Optional workflow benchmark against holdout points
    p.add_argument("--benchmark-holdout", default=None,
                   help="Optional holdout points/candidate pool file for baseline-vs-final workflow benchmark.")
    p.add_argument("--benchmark-auto-holdout", action="store_true", default=False,
                   help="Automatically derive a spatial holdout set from --benchmark-holdout or authoritative support artifacts and benchmark against it.")
    p.add_argument("--benchmark-holdout-frac", type=float, default=0.2,
                   help="Fraction of benchmark candidate points to place into the automatic holdout set (default 0.2).")
    p.add_argument("--benchmark-holdout-min-points", type=int, default=2000,
                   help="Minimum number of automatic holdout points to select when --benchmark-auto-holdout is used.")
    p.add_argument("--benchmark-holdout-seed", type=int, default=42,
                   help="Random seed for automatic benchmark holdout block selection.")
    p.add_argument("--benchmark-points-epsg", type=int, default=None,
                   help="Optional EPSG code for benchmark holdout point coordinates. If omitted, workflow will try to infer it automatically.")
    p.add_argument("--benchmark-x-col", default=None,
                   help="Optional x/lon column name for benchmark holdout file.")
    p.add_argument("--benchmark-y-col", default=None,
                   help="Optional y/lat column name for benchmark holdout file.")
    p.add_argument("--benchmark-z-col", default=None,
                   help="Optional z/depth column name for benchmark holdout file.")
    p.add_argument("--benchmark-baseline-raster", default=None,
                   help="Optional override path for baseline raster used in workflow benchmark.")
    p.add_argument("--benchmark-final-raster", default=None,
                   help="Optional override path for final raster used in workflow benchmark.")
    p.add_argument("--river-withheld-support-csv", default=None,
                   help="Optional CSV of deterministic river support points to exclude from authoritative river support generation for withheld-support benchmarking.")

    # Run health gates
    p.add_argument("--strict", action="store_true", default=False,
                   help="Fail the run if contract tests or output sanity checks fail.")

    # --- Apply YAML config defaults before parsing CLI ---
    # Parse only --config first (without erroring on unknown args)
    pre_args, _ = p.parse_known_args()
    config_files = getattr(pre_args, "config_files", []) or []
    if config_files:
        try:
            yaml_defaults = _load_yaml_config(config_files)
            if yaml_defaults:
                # Set argparse defaults from config — CLI args will override
                p.set_defaults(**{k: v for k, v in yaml_defaults.items()
                                  if hasattr(pre_args, k) or k in p._option_string_actions
                                  or any(k == a.dest for a in p._actions)})
        except Exception as e:
            log.warning("[CONFIG] Failed to load YAML config: %s", e)

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

    # Minimal retained outputs by default: final deliverables + consolidated river debug report + screen log.
    minimal_keep_keys = {
        "final_output",
        "combined_warped",
        "combined",
        "combined_hillshade",
        "river_warped",
        "river",
        "river_workflow_debug_report",
        "river_primary_surface",
        "river_support_points",
        "river_centerline_points",
        "centerline_elevation",
        "bank_points",
        "xs_bank_qc_points",
        "xs_bank_qc_summary",
        "bank_elevation_xs",
        "bank_influence",
        "stage_divergence_debug_dir",
        "stage_divergence_debug_manifest",
        "stage_divergence_receipt",
        "stage_semantics_receipt",
        "stage_divergence_summary_csv",
        "stage_divergence_cell_trace_csv",
        "stage_debug_01_authoritative_aligned",
        "stage_debug_02_river_primary_surface_handoff",
        "stage_debug_03_conditioned_before_final_route",
        "stage_debug_04_conditioned_after_harmonize_conditioned_elevation",
        "stage_debug_05_final_route_input",
        "stage_debug_06_dem_enhanced_written",
        "traceability_manifest_json",
        "traceability_manifest_md",
        "traceability_contract",
        "workflow_input_output_trace",
    }

    def _collect_minimal_output_paths(report_obj):
        out_paths = set()
        if not isinstance(report_obj, dict):
            return out_paths

        def _collect_from_outputs_dict(outputs_obj):
            if not isinstance(outputs_obj, dict):
                return
            for key in minimal_keep_keys:
                value = outputs_obj.get(key)
                if isinstance(value, str) and _is_probably_path(value):
                    out_paths.add(value)

        _collect_from_outputs_dict(report_obj.get("outputs"))

        river_obj = report_obj.get("river")
        if isinstance(river_obj, dict):
            _collect_from_outputs_dict(river_obj.get("outputs"))

        authoritative_obj = report_obj.get("authoritative_base")
        if isinstance(authoritative_obj, dict):
            _collect_from_outputs_dict(authoritative_obj.get("outputs"))

        return out_paths

    # Prefer unified report if present, else fall back to bathy_report.
    # Retain only the minimal explicit deliverables from those reports.
    report_paths = [out_dir / "unified_bathy_report.json", out_dir / "bathy_report.json"]
    for rp in report_paths:
        if rp.exists() and rp.is_file():
            try:
                robj = json.loads(rp.read_text(encoding="utf-8"))
                for s in _collect_minimal_output_paths(robj):
                    try:
                        pp = Path(s)
                        keep_abs.add(pp if pp.is_absolute() else (out_dir / pp).resolve())
                    except Exception:
                        log.debug("_walk: suppressed exception", exc_info=True)
                        continue
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue

    # Also honor only the minimal retained outputs recorded in the current in-memory report.
    try:
        if isinstance(report, dict):
            for s in _collect_minimal_output_paths(report):
                try:
                    pp = Path(s)
                    keep_abs.add(pp if pp.is_absolute() else (out_dir / pp).resolve())
                except Exception:
                    log.debug("_walk(report): suppressed exception", exc_info=True)
                    continue
    except Exception:
        log.debug("ignored", exc_info=True)

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
            log.debug("ignored", exc_info=True)

    # Only enforce policy if we produced a final deliverable
    final_ok = False

    # Prefer explicit outputs recorded in reports/manifest
    if keep_abs:
        for pth in list(keep_abs):
            try:
                if Path(pth).exists():
                    final_ok = True
                    break
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
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
                if not ap.exists():
                    continue
                try:
                    rel = ap.resolve().relative_to(out_dir.resolve())
                except Exception:
                    log.debug("_copy_rel: suppressed exception", exc_info=True)
                    # If output is outside out_dir, copy into tmp/_external/ for inspection
                    rel = Path("_external") / ap.name
                dst = tmp / rel
                if ap.is_dir():
                    if dst.exists():
                        shutil.rmtree(str(dst), ignore_errors=True)
                    shutil.copytree(str(ap), str(dst))
                    continue
                if not ap.is_file():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(ap), str(dst))
            except Exception as e:
                log.warning('[OUTPUT] Failed to stage output %s (%s). Continuing.', str(ap), e)

        # Keep only screen logs under run_logs.
        run_logs = out_dir / "run_logs"
        if run_logs.exists() and run_logs.is_dir():
            staged_any_screen_log = False
            for log_path in sorted(run_logs.glob("screen_*.log")):
                dst = tmp / "run_logs" / log_path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(log_path), str(dst))
                staged_any_screen_log = True
            if not staged_any_screen_log:
                log.info("[OUTPUT] No screen log found under run_logs to retain.")

        # If saving intermediates, move everything except tmp and intermediates dir under intermediates.
        intermediates_dir = out_dir / str(cfg.intermediates_dirname or "debug")

        if cfg.save_intermediates:
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
                try:
                    if child.is_symlink() or child.is_file():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        shutil.rmtree(str(child), ignore_errors=False)
                except Exception as e:
                    log.debug("bathy_main: suppressed exception", exc_info=True)
                    # One more quiet best-effort pass avoids noisy false alarms from transient files.
                    try:
                        if child.is_dir():
                            shutil.rmtree(str(child), ignore_errors=True)
                        else:
                            child.unlink(missing_ok=True)
                    except Exception:
                        log.debug('[OUTPUT] Final cleanup pass failed for %s', str(child), exc_info=True)
                    if child.exists():
                        log.warning('[OUTPUT] Failed to delete %s (%s). Continuing.', str(child), e)
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






def _enable_gdal_exceptions_best_effort() -> None:
    """Silence GDAL 4 deprecation warnings and make failures deterministic when bindings exist."""
    try:
        from osgeo import gdal
        gdal.UseExceptions()
    except Exception:
        log.debug("_enable_gdal_exceptions_best_effort: suppressed exception", exc_info=True)
        return


def _file_sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        log.debug("_file_sha256: suppressed exception", exc_info=True)
        return None


def _record_dem_enhanced_touch(out_dir: Path, *, action: str, path: Path, source: Optional[Path] = None, note: Optional[str] = None) -> None:
    try:
        combined_dir = ensure_dir(Path(out_dir) / "combined")
        touch_log = combined_dir / "dem_enhanced_touch_log.jsonl"
        payload = {
            "action": str(action),
            "path": str(Path(path)),
            "exists": bool(Path(path).exists()),
            "source": str(source) if source is not None else None,
            "source_exists": bool(Path(source).exists()) if source is not None else None,
            "path_sha256": _file_sha256(Path(path)) if Path(path).exists() else None,
            "source_sha256": _file_sha256(Path(source)) if source is not None and Path(source).exists() else None,
            "note": note,
            "is_symlink": bool(Path(path).is_symlink()),
        }
        with touch_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        log.info("[FINAL_OUTPUT][TRACE] %s: %s%s", action, path, f" <- {source}" if source is not None else "")
    except Exception:
        log.debug("_record_dem_enhanced_touch: suppressed exception", exc_info=True)


def _replace_with_symlink_or_copy(src: Path, dst: Path, *, out_dir: Optional[Path] = None, action: str = "replace_with_copy") -> None:
    """Replace a deliverable with a plain file copy.

    This helper intentionally avoids symlinks so final user-facing outputs and stable sidecars
    are straightforward on disk.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    try:
        if src == dst or (dst.exists() and os.path.samefile(src, dst)):
            if out_dir is not None and dst.name == "DEM_enhanced.tif":
                _record_dem_enhanced_touch(out_dir, action=f"{action}_noop", path=dst, source=src)
            return
    except Exception:
        log.debug("_replace_with_symlink_or_copy: samefile check suppressed exception", exc_info=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if out_dir is not None and dst.name == "DEM_enhanced.tif":
        _record_dem_enhanced_touch(out_dir, action=f"{action}_copy", path=dst, source=src)

def _verify_dem_enhanced_single_source_of_truth(cfg, report, *, final_native: Optional[Path | str], final_for_user: Optional[Path | str]) -> Optional[str]:
    """Verify that DEM_enhanced.tif is the same raster content as the internal conditioned DEM.

    The simple final-path rule is:
    1. final_route_outputs_stage writes combined/conditioned_final_dem_internal.tif
    2. that raster is copied once to combined/DEM_enhanced.tif
    3. later stages verify identity only and must not mutate the final DEM

    File-level hashes are allowed to differ because the deliverable may carry a different ROLE tag
    than the internal source-of-truth raster. The identity contract is therefore based on raster
    content and grid equivalence, not whole-file byte identity.
    """
    combined_dir = ensure_dir(Path(cfg.out_dir) / "combined")
    final_path = combined_dir / "DEM_enhanced.tif"
    receipt = combined_dir / "dem_enhanced_identity_receipt.json"
    report.setdefault("outputs", {})["final_dem_user_stable"] = str(final_path)
    report.setdefault("outputs", {})["final_depth_user_stable"] = str(final_path)
    report.setdefault("outputs", {})["dem_enhanced_touch_log"] = str(combined_dir / "dem_enhanced_touch_log.jsonl")
    report.setdefault("outputs", {})["dem_enhanced_identity_receipt"] = str(receipt)

    def _raster_identity_summary(src_path: Path, dst_path: Path) -> dict:
        import numpy as _np
        import rasterio as _rio
        summary = {
            "same_shape": False,
            "same_transform": False,
            "same_crs": False,
            "same_nodata": False,
            "same_dtype": False,
            "same_values": False,
            "source_role": None,
            "final_role": None,
            "max_abs_diff_m": None,
            "value_diff_pixels": None,
        }
        with _rio.open(src_path) as src_ds, _rio.open(dst_path) as dst_ds:
            summary["source_role"] = src_ds.tags().get("ROLE")
            summary["final_role"] = dst_ds.tags().get("ROLE")
            summary["same_shape"] = (src_ds.width == dst_ds.width and src_ds.height == dst_ds.height)
            summary["same_transform"] = tuple(src_ds.transform) == tuple(dst_ds.transform)
            summary["same_crs"] = str(src_ds.crs) == str(dst_ds.crs)
            summary["same_nodata"] = float(src_ds.nodata) == float(dst_ds.nodata)
            summary["same_dtype"] = tuple(src_ds.dtypes) == tuple(dst_ds.dtypes)
            src_arr = src_ds.read(1)
            dst_arr = dst_ds.read(1)
            if src_arr.shape == dst_arr.shape:
                same = _np.array_equal(src_arr, dst_arr)
                summary["same_values"] = bool(same)
                if not same:
                    diff = src_arr.astype(_np.float64) - dst_arr.astype(_np.float64)
                    finite = _np.isfinite(diff)
                    if finite.any():
                        summary["max_abs_diff_m"] = float(_np.max(_np.abs(diff[finite])))
                        summary["value_diff_pixels"] = int(_np.count_nonzero(diff[finite] != 0.0))
                    else:
                        summary["max_abs_diff_m"] = None
                        summary["value_diff_pixels"] = 0
            return summary

    try:
        auth_outputs = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}).get("outputs", {}), dict) else {}
        internal_source = Path(str(auth_outputs.get("conditioned_final_dem_internal"))) if auth_outputs.get("conditioned_final_dem_internal") else None
        debug_source = Path(str(auth_outputs.get("stage_debug_06_dem_enhanced_written"))) if auth_outputs.get("stage_debug_06_dem_enhanced_written") else None
        source = None
        source_kind = None
        if internal_source is not None and internal_source.exists():
            source = internal_source
            source_kind = "conditioned_final_dem_internal"
        elif debug_source is not None and debug_source.exists():
            source = debug_source
            source_kind = "stage_debug_06_dem_enhanced_written"
        elif final_native is not None and Path(str(final_native)).exists():
            source = Path(str(final_native))
            source_kind = "final_native"
        elif final_for_user is not None and Path(str(final_for_user)).exists():
            source = Path(str(final_for_user))
            source_kind = "final_for_user"
        if not final_path.exists():
            raise RuntimeError(f"DEM_enhanced missing at expected final path: {final_path}")
        if source is None:
            raise RuntimeError("No valid source available for DEM_enhanced identity verification")
        _record_dem_enhanced_touch(cfg.out_dir, action="identity_verification_start", path=final_path, source=source)
        if final_path.is_symlink():
            raise RuntimeError(f"DEM_enhanced must be a real raster, not a symlink: {final_path}")
        src_hash = _file_sha256(source)
        dst_hash = _file_sha256(final_path)
        identity = _raster_identity_summary(source, final_path)
        ok = bool(
            identity.get("same_shape")
            and identity.get("same_transform")
            and identity.get("same_crs")
            and identity.get("same_nodata")
            and identity.get("same_dtype")
            and identity.get("same_values")
        )
        write_json(receipt, {
            "final_path": str(final_path),
            "source": str(source),
            "source_kind": source_kind,
            "source_sha256": src_hash,
            "final_sha256": dst_hash,
            "match": ok,
            "writer": "final_route_outputs_stage",
            "verification_mode": "raster_content_no_rewrite",
            "identity": identity,
        })
        if not ok:
            raise RuntimeError(f"DEM_enhanced mismatch: source={source} final={final_path}")
        _record_dem_enhanced_touch(cfg.out_dir, action="identity_verification_ok", path=final_path, source=source)
        return str(final_path)
    except Exception:
        log.exception("[FINAL_OUTPUT] DEM_enhanced identity verification failed")
        raise

def _run_perspecto_hillshade_best_effort(raster_path: Path, *, report: Optional[Dict[str, Any]] = None, command_key: Optional[str] = None, path_key: Optional[str] = None, label: str = "HILLSHADE") -> Optional[Path]:
    """Create a perspecto hillshade beside a raster when the executable is available."""
    try:
        raster_path = Path(raster_path)
        if not raster_path.exists():
            return None
        exe = shutil.which("perspecto")
        if exe is None:
            log.info("[%s] perspecto not available; skipping hillshade creation.", label)
            return None
        cmd = [exe, raster_path.name]
        subprocess.run(cmd, cwd=str(raster_path.parent), check=True)
        hillshade_path = raster_path.with_name(f"{raster_path.stem}_hillshade.tif")
        if report is not None:
            outputs = report.setdefault("outputs", {})
            if command_key:
                outputs[command_key] = f"{Path(exe).name} {raster_path.name}"
            if path_key and hillshade_path.exists():
                outputs[path_key] = str(hillshade_path)
        log.info("[%s] Created hillshade via: %s %s", label, Path(exe).name, raster_path.name)
        return hillshade_path if hillshade_path.exists() else None
    except subprocess.CalledProcessError as exc:
        log.warning("[%s] perspecto failed for %s: returncode=%s", label, raster_path if 'raster_path' in locals() else '<unknown>', exc.returncode)
        return None
    except Exception:
        log.debug("_run_perspecto_hillshade_best_effort: suppressed exception", exc_info=True)
        return None


def _create_combined_hillshade_best_effort(cfg, report) -> None:
    """Create a hillshade for the user-facing combined raster using perspecto."""
    combined = report.get("outputs", {}).get("combined_warped") or report.get("outputs", {}).get("final")
    if not combined:
        return
    _run_perspecto_hillshade_best_effort(
        Path(combined),
        report=report,
        command_key="combined_hillshade_command",
        path_key="combined_hillshade",
        label="HILLSHADE",
    )


def _create_comparison_hillshades_best_effort(cfg, report) -> None:
    outputs = report.get("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
    specs = [
        ("final_comparison_navd88", "final_comparison_navd88_hillshade", "final_comparison_navd88_hillshade_command", "COMPARE_HILLSHADE"),
        ("baseline_comparison_navd88", "baseline_comparison_navd88_hillshade", "baseline_comparison_navd88_hillshade_command", "COMPARE_HILLSHADE"),
        ("final_comparison_navd88_all", "final_comparison_navd88_all_hillshade", "final_comparison_navd88_all_hillshade_command", "COMPARE_HILLSHADE"),
        ("baseline_comparison_navd88_all", "baseline_comparison_navd88_all_hillshade", "baseline_comparison_navd88_all_hillshade_command", "COMPARE_HILLSHADE"),
    ]
    for raster_key, hillshade_key, command_key, label in specs:
        raster = outputs.get(raster_key)
        if not raster:
            continue
        _run_perspecto_hillshade_best_effort(
            Path(raster),
            report=report,
            command_key=command_key,
            path_key=hillshade_key,
            label=label,
        )


def _create_final_visual_products_best_effort(cfg, report) -> None:
    _create_combined_hillshade_best_effort(cfg, report)
    _create_comparison_hillshades_best_effort(cfg, report)


def _write_simple_hillshade_fallback(raster_path: Path, hillshade_path: Path) -> Optional[Path]:
    try:
        import rasterio
        import numpy as np
        raster_path = Path(raster_path)
        hillshade_path = Path(hillshade_path)
        if not raster_path.exists():
            return None
        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            profile = src.profile.copy()
            if nodata is not None:
                arr[arr == nodata] = np.nan
            valid = np.isfinite(arr)
            if not np.any(valid):
                return None
            filled = arr.copy()
            fill_value = float(np.nanmedian(arr[valid]))
            filled[~valid] = fill_value
            x, y = np.gradient(filled)
            slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
            aspect = np.arctan2(-x, y)
            az = np.deg2rad(315.0)
            alt = np.deg2rad(45.0)
            shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
            shaded = ((np.clip(shaded, -1.0, 1.0) + 1.0) * 127.5).astype(np.float32)
            shaded[~valid] = np.nan
            profile.pop('blockxsize', None)
            profile.pop('blockysize', None)
            profile.pop('BLOCKXSIZE', None)
            profile.pop('BLOCKYSIZE', None)
            profile.update(driver='GTiff', dtype='float32', count=1, nodata=np.nan, compress='deflate', tiled=False)
            hillshade_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(hillshade_path, 'w', **profile) as dst:
                dst.write(shaded, 1)
                dst.update_tags(VALUE_TYPE='hillshade', ROLE='visual_comparison_hillshade')
        return hillshade_path if hillshade_path.exists() else None
    except Exception:
        log.debug('_write_simple_hillshade_fallback: suppressed exception', exc_info=True)
        return None


def _ensure_hillshade_best_effort(raster_path: Path, *, report: Optional[Dict[str, Any]] = None, command_key: Optional[str] = None, path_key: Optional[str] = None, label: str = 'HILLSHADE') -> Optional[Path]:
    hillshade_path = _run_perspecto_hillshade_best_effort(raster_path, report=report, command_key=command_key, path_key=path_key, label=label)
    if hillshade_path is not None and hillshade_path.exists():
        return hillshade_path
    fallback_path = Path(raster_path).with_name(f"{Path(raster_path).stem}_hillshade.tif")
    fallback = _write_simple_hillshade_fallback(Path(raster_path), fallback_path)
    if fallback is not None and report is not None and path_key:
        report.setdefault('outputs', {})[path_key] = str(fallback)
    return fallback


def _sync_v2_primary_surface_outputs_best_effort(cfg, report) -> None:
    try:
        outputs = report.setdefault('outputs', {}) if isinstance(report.get('outputs', {}), dict) else {}
        river_method = str((report.get('config', {}) or {}).get('river_method') or getattr(cfg, 'river_method', '') or '').lower()
        if river_method not in {'v2', 'simple_v2'}:
            return
        out_dir = Path(cfg.out_dir)
        stage_primary = out_dir / 'river_v2' / 'river_primary_surface.tif'
        combined_primary = out_dir / 'combined' / 'river_primary_surface.tif'
        if not stage_primary.exists():
            return
        combined_primary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_primary, combined_primary)
        outputs['river_primary_surface'] = str(combined_primary)
        outputs['river_primary_surface_v2_stage'] = str(stage_primary)
    except Exception:
        log.debug('_sync_v2_primary_surface_outputs_best_effort: suppressed exception', exc_info=True)


def _resolve_final_baseline_source(cfg, report) -> Path | None:
    out_dir = Path(cfg.out_dir)
    outputs = report.setdefault("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
    baseline_candidates = [
        Path(outputs.get("final_folder_authoritative_base_source") or ""),
        out_dir / "final" / "authoritative_base_aligned.tif",
        out_dir / "_external" / "cudem_baseline_interpolation.tif",
        Path(outputs.get("baseline_comparison_navd88_all") or ""),
        Path(outputs.get("baseline_comparison_navd88") or ""),
        Path(outputs.get("cudem_baseline_interpolation") or ""),
        Path(outputs.get("baseline_interpolated_dem") or ""),
        Path(outputs.get("final_route_baseline_interpolated") or ""),
        out_dir / "combined" / "cudem_baseline_interpolation.tif",
        out_dir / "combined" / "cudem_baseline_interpolation_comparison_all.tif",
        out_dir / "combined" / "cudem_baseline_interpolation_comparison.tif",
        out_dir / "combined" / "baseline_comparison_navd88_all.tif",
        out_dir / "combined" / "baseline_comparison_navd88.tif",
    ]
    combined_dir = out_dir / "combined"
    if combined_dir.exists():
        baseline_candidates.extend(sorted(combined_dir.glob("cudem_baseline_interpolation_comparison_*_all.tif")))
        baseline_candidates.extend(sorted(combined_dir.glob("cudem_baseline_interpolation_*_all.tif")))
        baseline_candidates.extend(sorted(combined_dir.glob("cudem_baseline_interpolation_comparison_*.tif")))
        baseline_candidates.extend(sorted(combined_dir.glob("cudem_baseline_interpolation_*.tif")))
        baseline_candidates.extend(sorted(combined_dir.glob("baseline_comparison_navd88*.tif")))
    seen = set()
    for candidate in baseline_candidates:
        try:
            candidate = Path(candidate)
        except Exception:
            continue
        key = str(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def _materialize_final_output_contract(cfg, report, *, baseline_only: bool = False) -> Dict[str, Path]:
    out_dir = Path(cfg.out_dir)
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    outputs = report.setdefault("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}

    baseline_src = _resolve_final_baseline_source(cfg, report)
    if baseline_src is None:
        raise RuntimeError("Missing baseline interpolated DEM source for final comparison folder")
    baseline_final = final_dir / "authoritative_base_aligned.tif"
    if baseline_src.resolve() != baseline_final.resolve():
        shutil.copy2(baseline_src, baseline_final)
    outputs["final_folder_authoritative_base"] = str(baseline_final)
    outputs["final_folder_authoritative_base_source"] = str(baseline_src)
    baseline_hs = _ensure_hillshade_best_effort(baseline_final, label="FINAL_COMPARE_HILLSHADE")
    if baseline_hs is None or not Path(baseline_hs).exists():
        raise RuntimeError("Missing final baseline hillshade for final comparison folder")
    outputs["final_folder_authoritative_base_hillshade"] = str(baseline_hs)

    products = {
        "authoritative_base_aligned": baseline_final,
        "authoritative_base_aligned_hillshade": Path(baseline_hs),
    }
    if baseline_only:
        return products

    dem_enhanced_src = Path(outputs.get("combined_warped") or outputs.get("final") or (out_dir / "combined" / "DEM_enhanced.tif"))
    if not dem_enhanced_src.exists():
        raise RuntimeError("Missing DEM_enhanced source for final comparison folder")
    dem_final = final_dir / "DEM_enhanced.tif"
    if dem_enhanced_src.resolve() != dem_final.resolve():
        shutil.copy2(dem_enhanced_src, dem_final)
    outputs["final_folder_dem_enhanced"] = str(dem_final)
    outputs["final_folder_dem_enhanced_source"] = str(dem_enhanced_src)
    dem_hs = _ensure_hillshade_best_effort(dem_final, label="FINAL_COMPARE_HILLSHADE")
    if dem_hs is None or not Path(dem_hs).exists():
        raise RuntimeError("Missing DEM_enhanced hillshade for final comparison folder")
    outputs["final_folder_dem_enhanced_hillshade"] = str(dem_hs)
    products.update({
        "DEM_enhanced": dem_final,
        "DEM_enhanced_hillshade": Path(dem_hs),
        "DEM_enhanced_source": dem_enhanced_src,
    })
    return products


def _write_final_output_receipt(cfg, report) -> Path:
    out_dir = Path(cfg.out_dir)
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    outputs = report.setdefault("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}

    products = _materialize_final_output_contract(cfg, report, baseline_only=False)
    baseline_src = Path(outputs.get("final_folder_authoritative_base_source"))
    dem_enhanced_src = Path(outputs.get("final_folder_dem_enhanced_source"))
    required = {
        "authoritative_base_aligned": products["authoritative_base_aligned"],
        "authoritative_base_aligned_hillshade": products["authoritative_base_aligned_hillshade"],
        "DEM_enhanced": products["DEM_enhanced"],
        "DEM_enhanced_hillshade": products["DEM_enhanced_hillshade"],
    }
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    receipt = {
        "contract": "final_output_folder_v1",
        "pipeline_version": PIPELINE_VERSION,
        "final_dir": str(final_dir),
        "baseline_source": str(baseline_src),
        "dem_enhanced_source": str(dem_enhanced_src),
        "required_outputs": {name: str(path) for name, path in required.items()},
        "missing_outputs": missing,
        "valid": not missing,
    }
    receipt_path = final_dir / "final_output_receipt.json"
    write_json(receipt_path, receipt)
    outputs["final_output_receipt"] = str(receipt_path)
    if missing:
        raise RuntimeError(f"Final output contract missing files: {missing}")
    return receipt_path


def _ensure_final_baseline_outputs_best_effort(cfg, report) -> None:
    try:
        _materialize_final_output_contract(cfg, report, baseline_only=True)
    except Exception:
        log.debug("_ensure_final_baseline_outputs_best_effort: suppressed exception", exc_info=True)


def _populate_final_comparison_folder_best_effort(cfg, report) -> None:
    try:
        _write_final_output_receipt(cfg, report)
    except Exception:
        log.debug('_populate_final_comparison_folder_best_effort: suppressed exception', exc_info=True)

def _reproject_final_outputs(cfg, final, report, sdb_raster, river_raster, fatal_errors):
    """Reproject final rasters to the configured output SRS while preserving DEM_enhanced as single-writer output.

    The final DEM at combined/DEM_enhanced.tif must never be rewritten, clipped, tapered,
    or otherwise mutated here. If it already matches the requested CRS, this stage records
    that fact and returns. If it does not, a sidecar context raster may be created on a
    different path, but the final DEM path remains the one written by final_route_outputs_stage.
    """
    final_for_user = final
    try:
        dst_srs = cfg.final_out_srs
        try:
            m = re.search(r"(\d{4,6})", str(dst_srs))
            dst_tag = f"epsg{m.group(1)}" if m else "dst"
        except Exception:
            log.debug("_reproject_final_outputs: suppressed exception", exc_info=True)
            dst_tag = "dst"
        try:
            m2 = re.search(r"(\d{4,6})", str(dst_srs))
            dst_horiz_epsg = int(m2.group(1)) if m2 else None
        except Exception:
            log.debug("_reproject_final_outputs: suppressed exception", exc_info=True)
            dst_horiz_epsg = None
        dst_is_4269 = (dst_horiz_epsg == 4269)

        if final:
            final_p = Path(final)
            combined_dir = ensure_dir(cfg.out_dir / "combined")
            out_name = f"{final_p.stem}_{dst_tag}.tif"
            final_same_crs = raster_crs_matches(final_p, dst_srs)
            preserve_final_dem_postwrite = False

            if final_same_crs:
                warped = final_p
                report.setdefault("outputs", {})["combined_warped_context"] = str(final_p)
                report.setdefault("outputs", {})["combined_warped"] = str(final_p)
                report.setdefault("outputs", {})["combined_warped_sidecar_only"] = None
                conditioned_user = combined_dir / "DEM_enhanced.tif"
                if final_p.resolve() == conditioned_user.resolve():
                    preserve_final_dem_postwrite = True
                    final_for_user = str(conditioned_user)
                    report.setdefault("outputs", {})["final_dem_user_stable"] = str(conditioned_user)
                    report.setdefault("outputs", {})["final_depth_user_stable"] = str(conditioned_user)
                    report.setdefault("outputs", {})["combined_warped_postwrite_skipped"] = "single_writer_final_dem_preserved"
                else:
                    final_for_user = str(final_p)
            else:
                warped = warp_raster_to_srs(final_p, combined_dir / out_name, dst_srs)

            if warped and (not final_same_crs):
                report.setdefault("outputs", {})["combined_warped_context"] = str(warped)
                bbox_tile = _parse_aoi_bbox(cfg.aoi_tile or cfg.aoi)
                deliverable = Path(warped)
                try:
                    if bbox_tile and dst_is_4269:
                        _crop_raster_extent_to_bbox(deliverable, bbox_tile, nodata=cfg.final_nodata)
                    report.setdefault("outputs", {})["combined_warped"] = str(deliverable)
                except Exception:
                    log.debug("ignored", exc_info=True)
                    report.setdefault("outputs", {})["combined_warped"] = str(deliverable)

                final_for_user = str(final_p)
                try:
                    conditioned_user = combined_dir / "DEM_enhanced.tif"
                    if final_p.resolve() == conditioned_user.resolve():
                        final_for_user = str(conditioned_user)
                        report.setdefault("outputs", {})["final_dem_user_stable"] = str(conditioned_user)
                        report.setdefault("outputs", {})["final_depth_user_stable"] = str(conditioned_user)
                    else:
                        report.setdefault("outputs", {})["combined_warped_sidecar_only"] = str(deliverable)
                except Exception:
                    log.debug("ignored", exc_info=True)
            elif not warped:
                msg = f"Final warp failed: {final_p.name} -> {out_name} ({dst_srs})"
                fatal_errors.append(msg)
                report.setdefault("outputs", {})["combined_warped"] = None
                log.error("[WARP] %s", msg)

            if warped:
                try:
                    bbox = _parse_aoi_bbox(cfg.aoi_tile or cfg.aoi)
                    deliver = Path(report.get("outputs", {}).get("combined_warped") or warped)
                    if (not preserve_final_dem_postwrite) and bbox and dst_is_4269 and deliver.exists() and (not _is_final_dem_user_raster(deliver, cfg.out_dir)):
                        try:
                            if cfg.tile_edge_taper_enabled and float(cfg.tile_edge_taper_km or 0.0) > 0:
                                m = _compute_edge_band_metrics_epsg4269(
                                    deliver,
                                    bbox,
                                    band_km=float(cfg.tile_edge_metrics_band_km or 2.0),
                                    smooth_sigma_km=float(cfg.tile_edge_smooth_sigma_km or 10.0),
                                    nodata=cfg.final_nodata,
                                )
                                report.setdefault("seams", {})["combined_edge_metrics"] = m
                                tapered = _apply_tile_edge_taper_epsg4269(
                                    deliver,
                                    bbox,
                                    taper_km=float(cfg.tile_edge_taper_km or 2.0),
                                    nodata=cfg.final_nodata,
                                )
                                report.setdefault("outputs", {})["combined_warped"] = str(tapered)
                            else:
                                m = _compute_edge_band_metrics_epsg4269(
                                    deliver,
                                    bbox,
                                    band_km=float(cfg.tile_edge_metrics_band_km or 2.0),
                                    smooth_sigma_km=float(cfg.tile_edge_smooth_sigma_km or 10.0),
                                    nodata=cfg.final_nodata,
                                )
                                report.setdefault("seams", {})["combined_edge_metrics"] = m
                        except Exception:
                            log.debug("ignored", exc_info=True)
                except Exception:
                    log.debug("ignored", exc_info=True)

        if sdb_raster:
            p = Path(sdb_raster)
            if p.exists():
                out = warp_raster_to_srs(p, ensure_dir(cfg.out_dir / "sdb") / f"{p.stem}_{dst_tag}.tif", dst_srs)
                if out:
                    report.setdefault("outputs", {})["sdb_warped"] = str(out)
        if river_raster:
            p = Path(river_raster)
            if p.exists():
                out = warp_raster_to_srs(p, ensure_dir(cfg.out_dir / "river") / f"{p.stem}_{dst_tag}.tif", dst_srs)
                if out:
                    report.setdefault("outputs", {})["river_warped"] = str(out)
    except Exception as e:
        msg = f"Final output reprojection failed: {e}"
        fatal_errors.append(msg)
        log.exception("[WARP] %s", msg)
    return final_for_user




def _build_authoritative_comparison_navd88_deliverable(*, cfg, report, reference_raster: Path, enhanced_raster: Path, dst_tag: str, dst_srs: str) -> Optional[Path]:
    """Create apples-to-apples NAVD88 comparison deliverables.

    Outputs:
    - bathy_cudem_enhanced_navd88_<tag>.tif: river-domain-only comparison raster
    - bathy_cudem_enhanced_navd88_<tag>_all.tif: full-AOI comparison raster
    - cudem_baseline_interpolation_<tag>.tif: river-domain-only baseline CUDEM raster
    - cudem_baseline_interpolation_<tag>_all.tif: full-AOI baseline CUDEM raster
    """
    try:
        import rasterio
        import numpy as np
        from rasterio.warp import reproject, Resampling
    except ImportError:
        log.debug("[COMPARE_NAVD88] rasterio unavailable", exc_info=True)
        return None

    if not reference_raster.exists() or not enhanced_raster.exists():
        return None

    ab_out = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    baseline_src = _resolve_baseline_cudem_interpolation(cfg, report)
    auth_src = ab_out.get("aligned_authoritative_base")
    support_src = ab_out.get("support_class")
    guidance_src = ab_out.get("guidance_influence")
    if baseline_src is None or auth_src is None or support_src is None:
        log.warning("[COMPARE_NAVD88] Missing baseline/auth/support inputs; skipping comparison deliverable.")
        return None

    try:
        auth_src_p = Path(str(auth_src))
        support_src_p = Path(str(support_src))
        guidance_src_p = Path(str(guidance_src)) if guidance_src else None
        baseline_src = Path(str(baseline_src))
    except (TypeError, ValueError, OSError):
        log.warning("[COMPARE_NAVD88] Invalid input paths; skipping comparison deliverable.")
        return None
    if not (baseline_src.exists() and auth_src_p.exists() and support_src_p.exists()):
        log.warning("[COMPARE_NAVD88] Required input rasters missing on disk; skipping comparison deliverable.")
        return None

    from support_classes import SupportClass

    combined_dir = ensure_dir(cfg.out_dir / "combined")
    baseline_all_path = combined_dir / f"cudem_baseline_interpolation_{dst_tag}_all.tif"
    if baseline_all_path.exists() or baseline_all_path.is_symlink():
        try:
            baseline_all_path.unlink()
        except OSError:
            pass
    warped_baseline = warp_raster_to_srs(baseline_src, baseline_all_path, dst_srs, write_depth_metadata=False, resample="near")
    if warped_baseline is None:
        return None
    baseline_all_path = Path(warped_baseline)

    with rasterio.open(baseline_all_path) as ref_ds:
        ref_profile = ref_ds.profile.copy()
        ref_profile.update(dtype="float32", count=1, nodata=float(getattr(cfg, "final_nodata", -9999.0)), compress="deflate")
        ref_shape = (ref_ds.height, ref_ds.width)
        ref_transform = ref_ds.transform
        ref_crs = ref_ds.crs

        def _warp_float(src_path: Path, *, src_nodata=None, resampling=Resampling.bilinear):
            arr = np.full(ref_shape, np.float32(np.nan), dtype=np.float32)
            with rasterio.open(src_path) as src_ds:
                reproject(
                    source=rasterio.band(src_ds, 1),
                    destination=arr,
                    src_transform=src_ds.transform,
                    src_crs=src_ds.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    src_nodata=src_ds.nodata if src_nodata is None else src_nodata,
                    dst_nodata=np.float32(np.nan),
                    resampling=resampling,
                )
            return arr

        baseline = _warp_float(baseline_all_path, resampling=Resampling.nearest)
        auth = _warp_float(auth_src_p, resampling=Resampling.nearest)
        enhanced = _warp_float(enhanced_raster, resampling=Resampling.bilinear)
        guidance = _warp_float(guidance_src_p, resampling=Resampling.nearest) if guidance_src_p and guidance_src_p.exists() else np.zeros(ref_shape, dtype=np.float32)

        support = np.zeros(ref_shape, dtype=np.uint8)
        with rasterio.open(support_src_p) as src_ds:
            reproject(
                source=rasterio.band(src_ds, 1),
                destination=support,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                src_nodata=src_ds.nodata,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

        river_mask = None
        river_mask_path = Path(cfg.river_channel_mask) if getattr(cfg, "river_channel_mask", None) else None
        if river_mask_path and river_mask_path.exists():
            river_mask = np.zeros(ref_shape, dtype=np.uint8)
            with rasterio.open(river_mask_path) as src_ds:
                reproject(
                    source=rasterio.band(src_ds, 1),
                    destination=river_mask,
                    src_transform=src_ds.transform,
                    src_crs=src_ds.crs,
                    src_nodata=src_ds.nodata,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )

    locked = support == int(SupportClass.AUTHORITATIVE_LOCKED)
    out_all = baseline.astype(np.float32, copy=True)
    if not np.any(np.isfinite(out_all)):
        out_all = auth.astype(np.float32, copy=True)
    auth_valid_locked = locked & np.isfinite(auth)
    if np.any(auth_valid_locked):
        out_all[auth_valid_locked] = auth[auth_valid_locked]
    enhancement_domain = (~locked) & np.isfinite(enhanced) & (np.nan_to_num(guidance, nan=0.0) > 0.0)
    if np.any(enhancement_domain):
        out_all[enhancement_domain] = enhanced[enhancement_domain]
    unresolved = ~np.isfinite(out_all)
    if np.any(unresolved & np.isfinite(auth)):
        fill = unresolved & np.isfinite(auth)
        out_all[fill] = auth[fill]
        unresolved = ~np.isfinite(out_all)
    if np.any(unresolved & np.isfinite(enhanced)):
        fill = unresolved & np.isfinite(enhanced)
        out_all[fill] = enhanced[fill]

    nodata = np.float32(ref_profile["nodata"])
    def _write_float_raster(dst_path: Path, arr: np.ndarray, *, role: str, policy: str) -> Path:
        if dst_path.exists() or dst_path.is_symlink():
            try:
                dst_path.unlink()
            except OSError:
                pass
        out_write = arr.astype(np.float32, copy=True)
        out_write[~np.isfinite(out_write)] = nodata
        with rasterio.open(dst_path, "w", **ref_profile) as dst:
            dst.write(out_write, 1)
            dst.update_tags(
                VALUE_TYPE="elevation",
                UNITS="meters",
                SIGN_CONVENTION="relative_to_datum",
                VERTICAL_SEMANTICS="absolute_elevation",
                VERTICAL_DATUM="NAVD88",
                VERTICAL_DATUM_EPSG="5703",
                ROLE=role,
                COMPARISON_POLICY=policy,
                OUTPUT_SRS=str(dst_srs),
            )
        return dst_path

    comparison_all_path = combined_dir / f"bathy_cudem_enhanced_comparison_navd88_{dst_tag}_all.tif"
    baseline_all_out = combined_dir / f"cudem_baseline_interpolation_comparison_{dst_tag}_all.tif"
    _write_float_raster(baseline_all_out, baseline, role="baseline_cudem_interpolation_all", policy="baseline_original_full_aoi")
    _write_float_raster(comparison_all_path, out_all, role="authoritative_aligned_enhanced_comparison_all", policy="baseline_cudem_background_plus_enhanced_nonlocked_guidance")

    river_domain_path = combined_dir / f"bathy_cudem_enhanced_comparison_navd88_{dst_tag}.tif"
    baseline_domain_path = combined_dir / f"cudem_baseline_interpolation_comparison_{dst_tag}.tif"
    river_domain_mask = (river_mask == 1) if river_mask is not None else enhancement_domain
    out_domain = out_all.astype(np.float32, copy=True)
    baseline_domain = baseline.astype(np.float32, copy=True)
    if river_domain_mask is not None:
        out_domain[~river_domain_mask] = np.nan
        baseline_domain[~river_domain_mask] = np.nan
    _write_float_raster(baseline_domain_path, baseline_domain, role="baseline_cudem_interpolation_river_domain", policy="baseline_original_river_domain")
    _write_float_raster(river_domain_path, out_domain, role="authoritative_aligned_enhanced_comparison_river_domain", policy="baseline_cudem_background_plus_enhanced_nonlocked_guidance_river_domain")

    report.setdefault("outputs", {})["final_comparison_navd88"] = str(river_domain_path)
    report.setdefault("outputs", {})["baseline_comparison_navd88"] = str(baseline_domain_path)
    report.setdefault("outputs", {})["final_comparison_navd88_all"] = str(comparison_all_path)
    report.setdefault("outputs", {})["baseline_comparison_navd88_all"] = str(baseline_all_out)
    report.setdefault("outputs", {})["combined_comparison_navd88"] = str(river_domain_path)
    report.setdefault("outputs", {})["combined_comparison_navd88_all"] = str(comparison_all_path)
    report.setdefault("outputs", {})["final_comparison_navd88_reference"] = str(baseline_all_out)
    report.setdefault("outputs", {})["final_comparison_navd88_enhanced_source"] = str(enhanced_raster)
    report.setdefault("outputs", {})["final_comparison_navd88_policy"] = {
        "background": str(baseline_src),
        "authoritative": str(auth_src_p),
        "support_class": str(support_src_p),
        "guidance_influence": str(guidance_src_p) if guidance_src_p and guidance_src_p.exists() else None,
        "enhanced_source": str(enhanced_raster),
        "locked_pixels": int(np.count_nonzero(locked)),
        "enhancement_pixels": int(np.count_nonzero(enhancement_domain)),
        "river_domain_pixels": int(np.count_nonzero(river_domain_mask)) if river_domain_mask is not None else 0,
    }
    return comparison_all_path

def _link_or_copy_file(src: Path, dst: Path) -> Optional[Path]:
    """Create a lightweight packaged pointer to an existing artifact."""
    try:
        src = Path(src).resolve()
        if not src.exists():
            return None
        ensure_dir(dst.parent)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            rel = os.path.relpath(str(src), str(dst.parent.resolve()))
            dst.symlink_to(rel)
        except Exception:
            log.debug("_link_or_copy_file: suppressed exception", exc_info=True)
            shutil.copy2(src, dst)
        return dst
    except Exception:
        log.debug("[COMPARE] Failed linking/copying %s -> %s", src, dst, exc_info=True)
        return None


def _materialize_masked_raster_copy(src: Path, dst: Path, *, mask: Optional[Path], inside_value: int, nodata: float) -> Optional[Path]:
    """Copy or link a raster and optionally clip it to a mask in-place."""
    try:
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            return None
        ensure_dir(dst.parent)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)
        if mask is not None and Path(mask).exists():
            _clip_raster_to_mask_reproject(dst, Path(mask), inside_value=inside_value, invert=False, nodata=nodata)
        return dst
    except Exception:
        log.debug("_materialize_masked_raster_copy: suppressed exception", exc_info=True)
        return None


def _resolve_original_cudem_full_aoi_baseline(cfg: "BathyConfig", report: Dict[str, Any]) -> Optional[Path]:
    """Resolve the original AOI-wide CUDEM baseline mosaic from the authoritative-base stage."""
    info = report.get("authoritative_base_auto", {}) if isinstance(report.get("authoritative_base_auto"), dict) else {}
    cand = info.get("baseline_cudem_interpolation")
    if cand:
        try:
            p = Path(str(cand))
        except (TypeError, ValueError, OSError):
            p = None
        if p is not None and p.exists():
            log.info("[COMPARE] Using original full-AOI CUDEM baseline mosaic: %s", p)
            return p
    auth = getattr(cfg, "authoritative_base", None)
    if auth:
        try:
            sibling = Path(auth).with_name("cudem_baseline_interpolation.tif")
            if sibling.exists():
                log.info("[COMPARE] Using original full-AOI CUDEM baseline sibling: %s", sibling)
                return sibling
        except (TypeError, ValueError, OSError):
            log.debug("[COMPARE] Failed resolving baseline sibling next to authoritative_base", exc_info=True)
    log.warning("[COMPARE] Could not resolve original full-AOI CUDEM baseline mosaic.")
    return None


def _resolve_baseline_cudem_interpolation(cfg: "BathyConfig", report: Dict[str, Any]) -> Optional[Path]:
    return _resolve_original_cudem_full_aoi_baseline(cfg, report)


def _write_authoritative_cache_receipt(cfg: "BathyConfig", report: Dict[str, Any]) -> Optional[Path]:
    return _fr_write_authoritative_cache_receipt(cfg, report)


def _write_comparison_summary(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    packaged: Dict[str, Optional[str]],
    out_dir: Path,
) -> Optional[Path]:
    return _fr_write_comparison_summary(cfg, report, packaged, out_dir, logger=log)


def _write_final_support_regime_audit(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    *,
    final_native: Optional[Path],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
) -> Optional[Path]:
    return _fsra_write_final_support_regime_audit(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance, logger=log)


def _write_support_provenance_summary(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    *,
    final_native: Optional[Path],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
) -> Optional[Path]:
    return _pr_write_support_provenance_summary(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance, logger=log)


def _write_final_dem_selection_receipt(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    *,
    final_native: Optional[Path],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
) -> Optional[Path]:
    return _fr_write_final_dem_selection_receipt(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)


def _finalize_sdb_run(sdb_dir: Path, report: Dict[str, Any]) -> Optional[Path]:
    return _finalize_sdb_run_impl(sdb_dir=sdb_dir, report=report, apply_depth_metadata=apply_depth_metadata, logger=log)


def _resolve_river_domain_mask_for_fusion(cfg: "BathyConfig", report: Dict[str, Any]) -> Optional[Path]:
    return _resolve_river_domain_mask_for_fusion_impl(cfg, report)


def _copy_fusion_outputs(res, out_depth: Path, out_prov: Optional[Path]):
    return _copy_fusion_outputs_impl(res, out_depth, out_prov)


def _emit_optional_artifacts(report_path, io_json, io_md, report, logger) -> None:
    try:
        from flight_recorder import emit_artifact_written
        emit_artifact_written(report_path, kind="json", role="bathy_report")
        if io_json is not None:
            emit_artifact_written(io_json, kind="json", role="io_manifest")
        if io_md is not None:
            emit_artifact_written(io_md, kind="md", role="io_manifest")
        _emit_artifacts_from_report(report)
    except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as e:
        logger.debug("Optional flight-recorder emit failed: %s", e)


def _run_gapfill_stage(*, cfg: "BathyConfig", args, report: Dict[str, Any], log, final, final_provenance, river_raster, authoritative_aligned, authoritative_eligible_fill_mask):
    if not (cfg.gapfill_enabled and final):
        return final, final_provenance
    try:
        from gapfill_intelligent import gapfill_depth_raster, GapfillConfig

        hq_files: List[str] = []
        if cfg.gapfill_hq:
            hq_files = [str(p) for p in cfg.gapfill_hq if p]
        else:
            ex = getattr(args, "extra_xyz", None)
            if ex:
                if isinstance(ex, (list, tuple)):
                    hq_files = [str(p) for p in ex if p]
                else:
                    hq_files = [str(ex)]

        if not hq_files:
            log.warning("[GAPFILL] Enabled but no HQ points provided (--gapfill-hq or --extra-xyz). Skipping.")
            return final, final_provenance

        combined_dir = ensure_dir(cfg.out_dir / "combined")
        out_gap = combined_dir / "bathy_combined_depth_gapfill.tif"
        out_sig = combined_dir / "bathy_combined_depth_gapfill_sigma.tif"
        out_prov = combined_dir / "bathy_combined_depth_gapfill_provenance.tif"

        wm: Optional[Path] = None
        if cfg.gapfill_water_mask and Path(cfg.gapfill_water_mask).exists():
            wm = Path(cfg.gapfill_water_mask)
        else:
            wm = _find_latest_waffles_mask(cfg.cache_root)

        xs_gpkg = cfg.out_dir / "river" / "river_xs_params.gpkg"
        if not xs_gpkg.exists():
            xs_gpkg = None
        river_mask = Path(river_raster) if river_raster and Path(river_raster).exists() else None

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

        gapfill_stats = gapfill_depth_raster(
            prior_raster=Path(final), hq_point_files=hq_files, out_raster=out_gap, out_sigma=out_sig, out_provenance=out_prov,
            cfg=gcfg, prior_sigma_raster=cfg.gapfill_prior_sigma_raster, water_mask_raster=wm, river_mask_raster=river_mask,
            bank_elev_raster=cfg.gapfill_bank_elev_raster, xs_params_gpkg=xs_gpkg, target_mask_raster=authoritative_eligible_fill_mask,
            lock_raster=authoritative_aligned, logger=log,
        )

        report.setdefault("gapfill", {})["status"] = "applied"
        report["gapfill"]["stats"] = gapfill_stats
        report["gapfill"]["outputs"] = {"depth": str(out_gap), "sigma": str(out_sig), "provenance": str(out_prov)}
        if cfg.gapfill_output_cudem_xyz and "cudem_xyz" in gapfill_stats:
            report["gapfill"]["outputs"]["cudem_xyz"] = gapfill_stats["cudem_xyz"]
        return out_gap, out_prov
    except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as e:
        report.setdefault("gapfill", {})["status"] = "failed"
        report["gapfill"]["error"] = str(e)
        log.warning("[GAPFILL] Failed: %s", e)
        return final, final_provenance


def _write_comparison_package(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    *,
    final_native: Optional[Path],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
) -> Optional[Path]:
    return _fr_write_comparison_package(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance, logger=log)


def _write_explicit_final_outputs_manifest(
    cfg: "BathyConfig",
    report: Dict[str, Any],
    *,
    final_native: Optional[Path],
    final_for_user: Optional[Path | str],
    final_provenance: Optional[Path | str],
) -> Optional[Path]:
    _ensure_bundle_a_route_contract_state(report)
    return _fr_write_explicit_final_outputs_manifest(cfg, report, final_native=final_native, final_for_user=final_for_user, final_provenance=final_provenance)


def _run_seam_comparisons(args, cfg, report, run_id, final, final_for_user, report_path):
    """Run explicit seam and nested-AOI regression checks and write receipts."""
    try:
        run_postrun_regression_stage(
            args=args,
            cfg=cfg,
            report=report,
            run_id=run_id,
            final=final,
            final_for_user=final_for_user,
            report_path=report_path,
            logger=log,
        )
        try:
            write_json(report_path, report)
        except OSError:
            log.debug("ignored", exc_info=True)
    except RuntimeError:
        raise
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as e:
        log.debug("Optional seam/nested-AOI regression step failed; continuing: %s", e, exc_info=True)


def _finalize_run(cfg, report, args, final, final_for_user, final_provenance, run_id, fatal_errors):
    """Write run summaries, run contract tests, and apply output retention policy."""
    from finalize_run_helpers import (
        NONFATAL_FINALIZE_EXCEPTIONS,
        apply_energy_solver_top_level_summary,
        build_and_log_human_summary,
        run_runtime_sign_semantics,
        write_detailed_run_summaries,
        write_river_workflow_debug_file,
    )
    # Human-friendly, scan-friendly console summary
    try:
        build_and_log_human_summary(cfg=cfg, report=report, logger=log)
    except NONFATAL_FINALIZE_EXCEPTIONS as e:
        log.debug("Human summary skipped: %s", e)

    # Echo river energy-solver status in the top-level summary so users do not
    # need to hunt through XS logs to confirm whether physics stabilization applied.
    try:
        apply_energy_solver_top_level_summary(cfg=cfg, report=report, logger=log)
    except NONFATAL_FINALIZE_EXCEPTIONS as e:
        log.debug("Energy solver top-level summary skipped: %s", e)

    # Runtime sign/elevation semantic contracts (Bundle A): report by default, fail in strict mode.
    try:
        run_runtime_sign_semantics(cfg=cfg, report=report, logger=log)
    except RuntimeError:
        raise
    except NONFATAL_FINALIZE_EXCEPTIONS as e:
        log.warning("[CONTRACT][SIGN_SEMANTICS] Runtime semantic contract evaluation failed: %s", e, exc_info=True)

    # Write detailed run summaries (technical/scientific/human) using the in-memory report
    try:
        write_detailed_run_summaries(cfg=cfg, report=report, logger=log)
    except NONFATAL_FINALIZE_EXCEPTIONS as e:
        log.debug("Run summary file write skipped: %s", e)



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
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError, TypeError) as e:
            log.error("[STRICT][SANITY] %s", e)
            raise



    # Output contract: ensure standard filenames exist (metrics/debug rely on these)
    try:
        _ensure_output_contract(
            Path(getattr(args, "out_dir")),
            Path(getattr(args, "cache_root", Path(getattr(args, "out_dir")) / "cache")),
        )
    except (OSError, RuntimeError, ValueError, TypeError):
        log.debug("ignored", exc_info=True)

    # Final domain policy: clip final products based on enabled methods
    try:
        _apply_final_domain_policy(cfg, cfg.out_dir, Path(cfg.derived_cache_root))
    except (OSError, RuntimeError, ValueError, TypeError):
        log.debug("ignored", exc_info=True)

    try:
        _write_final_output_receipt(cfg, report)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError, TypeError):
        log.debug("ignored", exc_info=True)

    try:
        if bool(getattr(cfg, "make_figs", False)):
            from presentation_figures import generate_presentation_figures
            generate_presentation_figures(cfg, report, logger=log)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError, TypeError) as e:
        log.warning("[FIGURES] Presentation figure generation failed: %s", e, exc_info=True)

    # Optional workflow benchmark using holdout points.
    try:
        postrun_context = build_final_postrun_context(
            cfg=cfg,
            args=args,
            report=report,
            run_id=run_id,
            final_native=final,
            final_for_user=final_for_user,
            final_provenance=final_provenance,
            fatal_errors=fatal_errors,
        )
        run_postrun_benchmark_stage(
            context=postrun_context,
            logger=log,
            run_workflow_benchmark_fn=_run_workflow_benchmark,
        )
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError, TypeError) as e:
        log.error("[BENCHMARK] %s", e, exc_info=True)
        bench = report.setdefault("benchmark", {})
        bench.update({
            "status": "failed",
            "error": str(e),
            "exception_type": type(e).__name__,
            "requested": bool(getattr(args, "benchmark_holdout", None)) or bool(getattr(args, "benchmark_auto_holdout", False)),
        })
        report.setdefault("postrun_failures", []).append({
            "stage": "benchmark",
            "error": str(e),
            "exception_type": type(e).__name__,
        })

    try:
        river_debug_path = write_river_workflow_debug_file(out_dir=cfg.out_dir, report=report, logger=log)
        report.setdefault("outputs", {})["river_workflow_debug_report"] = river_debug_path
    except NONFATAL_FINALIZE_EXCEPTIONS as e:
        log.warning("[RIVER_DEBUG_REPORT] Failed to write river workflow debug report: %s", e, exc_info=True)

    try:
        if final:
            final_for_user = _verify_dem_enhanced_single_source_of_truth(
                cfg,
                report,
                final_native=final,
                final_for_user=final_for_user,
            )
    except Exception as e:
        fatal_errors.append(f"DEM_enhanced identity verification failed: {e}")

    try:
        actual_trace = _write_workflow_actual_trace(out_dir=cfg.out_dir, report=report)
        report.setdefault("outputs", {})["workflow_input_output_trace"] = str(actual_trace)
    except Exception as e:
        fatal_errors.append(f"Workflow input/output trace generation failed: {e}")

    try:
        io_json, io_md = write_io_manifest(cfg.out_dir, report)
        report.setdefault("outputs", {})["io_manifest_json"] = str(io_json)
        report.setdefault("outputs", {})["io_manifest_md"] = str(io_md)
    except Exception as e:
        fatal_errors.append(f"Final IO manifest refresh failed: {e}")

    try:
        actual_trace = _write_workflow_actual_trace(out_dir=cfg.out_dir, report=report)
        report.setdefault("outputs", {})["workflow_input_output_trace"] = str(actual_trace)
    except Exception as e:
        fatal_errors.append(f"Workflow input/output trace refresh failed: {e}")

    try:
        trace_json, trace_md, trace_contract, _trace_manifest, trace_validation = _write_traceability_manifest(cfg.out_dir, report)
        report.setdefault("outputs", {})["traceability_manifest_json"] = str(trace_json)
        report.setdefault("outputs", {})["traceability_manifest_md"] = str(trace_md)
        report.setdefault("outputs", {})["traceability_contract"] = str(trace_contract)
        if not bool(trace_validation.get("ok", False)):
            fatal_errors.append(
                "Traceability contract failed: "
                f"missing={trace_validation.get('missing_required_artifacts', [])} "
                f"final_route_write_count={trace_validation.get('final_route_write_count')} "
                f"unexpected_actions={trace_validation.get('unexpected_actions', [])} "
                f"touch_paths_ok={trace_validation.get('touch_paths_ok')} "
                f"touch_path_mismatches={trace_validation.get('touch_path_mismatches', [])} "
                f"hash_match={trace_validation.get('hash_match')} "
                f"identity_receipt_ok={trace_validation.get('identity_receipt_ok')}"
            )
    except Exception as e:
        fatal_errors.append(f"Traceability manifest generation failed: {e}")

    # If any requested final reprojection outputs failed, treat the run as failed.
    if fatal_errors:
        report["status"] = "failed"
        report["fatal_errors"] = list(fatal_errors)
        for m in fatal_errors:
            log.error("%s", m)
    # Output retention policy (final outputs only by default; intermediates optional)
    if (not fatal_errors) and final:
        try:
            _apply_output_retention_policy(cfg, log, report, final_path=final, final_for_user_path=final_for_user)
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError, TypeError) as e:
            log.warning("Retention policy failed (leaving outputs as-is): %s", e)

    if fatal_errors:
        log.error("Pipeline failed.")
    else:
        log.info("Pipeline complete.")
    log.info("SDB: %s", report.get('sdb', {}).get('status', 'skipped'))
    log.info("River: %s (mode=%s)", report.get('river', {}).get('status', 'skipped'), report.get('river', {}).get('execution_mode', 'n/a'))
    log.info("Fusion: %s", report.get('fusion', {}).get('status', 'skipped'))
    log.info("Final output: %s", final_for_user or final or 'None')

    if fatal_errors:
        return 2
    return 0 if final else 2


def main() -> int:
    args = parse_args()
    run_id = _initialize_run_args(args, logger=log)
    _authoritative_cfg = _resolve_authoritative_base_args(args)
    _authoritative_base_auto = bool(_authoritative_cfg.get("authoritative_base_auto", False))
    _authoritative_base_path = _authoritative_cfg.get("authoritative_base_path")

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
            validation_truth=Path(args.validation_truth).resolve() if args.validation_truth else None,
            validation_case_specs=list(getattr(args, "validation_case", []) or []),
            validation_case_manifest=Path(args.validation_case_manifest).resolve() if args.validation_case_manifest else None,
            validation_guidance_baseline_case=str(getattr(args, "validation_guidance_baseline_case", "baseline_cudem_interpolation")),
            validation_guidance_target_case=str(getattr(args, "validation_guidance_target_case", "selected_final")),
            validation_require_guidance_non_degradation=bool(getattr(args, "validation_require_guidance_non_degradation", False)),
            validation_guidance_rmse_tolerance=float(getattr(args, "validation_guidance_rmse_tolerance", 0.0) or 0.0),
            require_river_constraints=str(getattr(args, "require_river_constraints", "none")),
            river_dem_auto=args.river_dem_auto,
            river_dem_source=args.river_dem_source,
            river_dem_res_m=args.river_dem_res_m,
            extra_xyz_crs=args.extra_xyz_crs,
            sdb_authoritative_extra_xyz=None,
            river_authoritative_soundings=None,
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

            river_enable_1d_energy_solver=bool(getattr(args, "river_enable_1d_energy_solver", False)),
            river_energy_allow_dem_proxy_wse=bool(getattr(args, "river_energy_allow_dem_proxy_wse", False)),

            river_manning_mode=str(getattr(args, "river_manning_mode", "off") or "off"),
            river_manning_q_cms=getattr(args, "river_manning_q_cms", None),
            river_manning_q_field=getattr(args, "river_manning_q_field", None),
            river_manning_n=float(getattr(args, "river_manning_n", 0.035) or 0.035),
            river_manning_region=str(getattr(args, "river_manning_region", "default") or "default"),
            river_manning_min_confidence=float(getattr(args, "river_manning_min_confidence", 0.30) or 0.30),
            river_manning_max_weight=float(getattr(args, "river_manning_max_weight", 0.60) or 0.60),
            river_manning_backwater_slope_thresh=float(getattr(args, "river_manning_backwater_slope_thresh", 1e-4) or 1e-4),
            river_manning_dist_to_mouth_field=getattr(args, "river_manning_dist_to_mouth_field", None),
            river_manning_dist_to_mouth_km_max=float(getattr(args, "river_manning_dist_to_mouth_km_max", 10.0) or 10.0),
    
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
            river_da_raster=(Path(args.river_da_raster) if getattr(args, 'river_da_raster', None) else None),
            river_da_raster_band=int(getattr(args, 'river_da_raster_band', 1) or 1),
            river_da_raster_units=str(getattr(args, 'river_da_raster_units', 'km2') or 'km2'),
            tnm_enable=((args.river_hydrography_source in ('arcgis_tnm','tnm')) and (not args.no_tnm)),
            tnm_dataset=args.tnm_dataset,
            snap_m=args.snap_m,
    
    
            river_method=args.river_method,
            river_channel_buffer_m=args.river_channel_buffer_m,
            river_max_channel_width_m=args.river_max_channel_width_m,
            river_mainstem_method=str(getattr(args, "river_mainstem_method", "dominant_trunk") or "dominant_trunk"),
            river_mainstem_solve_layer=str(getattr(args, "river_mainstem_solve_layer", "auto") or "auto"),
            river_mainstem_min_order=args.river_mainstem_min_order,
            river_max_mainstem_width_m=args.river_max_mainstem_width_m,
            river_guidance_bank_margin_m=float(getattr(args, 'river_guidance_bank_margin_m', 3.0) or 3.0),
            river_withheld_support_csv=(Path(args.river_withheld_support_csv) if getattr(args, 'river_withheld_support_csv', None) else None),
            river_use_nhdarea=bool(getattr(args, "river_use_nhdarea", True)),
            river_nhdarea_layer=getattr(args, "river_nhdarea_layer", "nhdarea_clip"),
            river_domain_min_water_corridor_frac=float(getattr(args, "river_domain_min_water_corridor_frac", 0.02)),
            river_domain_min_channel_corridor_frac=float(getattr(args, "river_domain_min_channel_corridor_frac", 0.001)),
            river_domain_min_channel_pixels=int(getattr(args, "river_domain_min_channel_pixels", 1)),
            river_domain_hard_fail=bool(getattr(args, "river_domain_hard_fail", False)),
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
            river_xs_profile_shape=getattr(args, "river_xs_profile_shape", "parabolic"),
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
            waffles_min_water_fraction=float(getattr(args, "waffles_min_water_fraction", 0.001)),
            force_waffles_masks=bool(getattr(args, "force_waffles_masks", False)),
    
            gapfill_enabled=bool(getattr(args, "gapfill_enabled", False)),
            gapfill_hq=list(getattr(args, "gapfill_hq", None)) if getattr(args, "gapfill_hq", None) else None,
            gapfill_water_mask=Path(getattr(args, "gapfill_water_mask", "")) if getattr(args, "gapfill_water_mask", None) else None,
            authoritative_base=_authoritative_base_path,
            authoritative_base_auto=_authoritative_base_auto,
            authoritative_base_tile_index_url=str(getattr(args, "authoritative_base_tile_index_url", "") or ""),
            authoritative_base_spatial_meta_url=str(getattr(args, "authoritative_base_spatial_meta_url", "") or ""),
            authoritative_base_missing_meta_policy=str(getattr(args, "authoritative_base_missing_meta_policy", "skip") or "skip"),
            authoritative_base_force_rebuild=bool(getattr(args, "authoritative_base_force_rebuild", False)),
            authoritative_base_tile_url_field=getattr(args, "authoritative_base_tile_url_field", None),
            authoritative_support_decay_m=float(getattr(args, "authoritative_support_decay_m", 300.0) or 300.0),
            authoritative_support_density_radius_m=float(getattr(args, "authoritative_support_density_radius_m", 250.0) or 250.0),
            coastal_sdb_support_transition_m=float(getattr(args, "coastal_sdb_support_transition_m", 600.0) or 600.0),
            river_anchor_density_radius_m=float(getattr(args, "river_anchor_density_radius_m", 200.0) or 200.0),
            river_scaffold_transition_m=float(getattr(args, "river_scaffold_transition_m", 800.0) or 800.0),
            river_centerline_influence_scale=float(getattr(args, "river_centerline_influence_scale", 1.0) or 1.0),
            river_disable_xs_influence=bool(getattr(args, "river_disable_xs_influence", False)),
            river_network_halo_km=float(getattr(args, "river_network_halo_km", 2.0) or 0.0),
            river_trusted_halo_m=float(getattr(args, "river_trusted_halo_m", 60.0) or 0.0),
            gapfill_method=getattr(args, "gapfill_method", "rbf"),
            gapfill_river_smooth_sigma_m=float(getattr(args, "gapfill_river_smooth_sigma", 500.0)),
            gapfill_prior_sigma_raster=Path(getattr(args, "gapfill_prior_sigma", "")) if getattr(args, "gapfill_prior_sigma", None) else None,
            gapfill_bank_elev_raster=Path(getattr(args, "gapfill_bank_elev", "")) if getattr(args, "gapfill_bank_elev", None) else None,
            gapfill_output_cudem_xyz=bool(getattr(args, "gapfill_cudem_xyz", False)),
            # ── Regional curve (was missing from constructor — used defaults) ──
            river_regional_curve_enabled=bool(getattr(args, "river_regional_curve_enabled", False)),
            # ── Channel template system ──
            river_channel_template_enabled=bool(getattr(args, "river_channel_template_enabled", False)),
            river_channel_template_min_xs=int(getattr(args, "river_channel_template_min_xs", 3)),
            river_channel_template_fit_min_xs=int(getattr(args, "river_channel_template_fit_min_xs", 5)),
            river_channel_template_n_bins=int(getattr(args, "river_channel_template_n_bins", 50)),
            river_channel_template_min_depth_m=float(getattr(args, "river_channel_template_min_depth_m", 0.3)),
            river_channel_template_distance_sigma_m=float(getattr(args, "river_channel_template_distance_sigma_m", 2000.0)),
            river_channel_template_estuary_buffer_m=float(getattr(args, "river_channel_template_estuary_buffer_m", 500.0)),
            river_channel_template_junction_buffer_m=float(getattr(args, "river_channel_template_junction_buffer_m", 120.0)),
            river_channel_template_width_depth_ratio_max=(float(getattr(args, "river_channel_template_width_depth_ratio_max")) if getattr(args, "river_channel_template_width_depth_ratio_max", None) is not None else None),
            river_channel_template_loo_max_rmse_norm=float(getattr(args, "river_channel_template_loo_max_rmse_norm", 0.25)),
            river_channel_template_loo_max_dmax_error_m=float(getattr(args, "river_channel_template_loo_max_dmax_error_m", 1.5)),
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

    try:
        _materialize_authoritative_base_if_requested(cfg, args)
    except (ImportError, FileNotFoundError, OSError, ValueError, KeyError, RuntimeError):
        log.error("[AUTHORITATIVE] Failed to materialize authoritative_base for AOI=%s", cfg.aoi, exc_info=True)
        raise

    log.info("Unified bathymetry pipeline starting.")
    log.info("AOI: %s", cfg.aoi)
    log.info("Date range: %s to %s", cfg.start_date, cfg.end_date)
    log.info("Methods: %s", cfg.methods)
    log.info("Priority: %s", cfg.priority)
    log.info("Output: %s", cfg.out_dir)

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pipeline_version": getattr(constants, "PIPELINE_VERSION", "unknown"),
        "config": {
            "aoi": cfg.aoi,
            "start": cfg.start_date,
            "end": cfg.end_date,
            "methods": list(cfg.methods or []),
            "priority": cfg.priority,
            "out_dir": str(cfg.out_dir),
        },
    }

    # ---------------------------------------------------------------------
    # Shared-domain stage (pass 1/2 architecture): build and summarize river/SDB
    # ownership before deciding which water methods should run.
    # ---------------------------------------------------------------------
    domain_inference: Dict[str, Any] = {}
    try:
        cfg.methods_requested = list(cfg.methods or [])
        _prepare_guidance_domains_for_run(cfg, report)
        effective_methods, _meta = _determine_effective_methods_from_domains(cfg, report)
        cfg.methods_effective = list(effective_methods)
        cfg.methods = list(effective_methods)
        skipped = (_meta or {}).get("skipped", {}) if isinstance(_meta, dict) else {}
        derived = (_meta or {}).get("derived_activation", {}) if isinstance(_meta, dict) else {}
        if skipped:
            log.info("[DOMAIN] Shared-domain activation: requested=%s -> effective=%s (skipped=%s; river_should_run=%s; sdb_should_run=%s)",
                     ",".join(cfg.methods_requested), ",".join(cfg.methods_effective),
                     ",".join([f"{k}:{v}" for k, v in skipped.items()]),
                     bool(derived.get("river_should_run", False)),
                     bool(derived.get("sdb_should_run", False)))
        else:
            log.info("[DOMAIN] Shared-domain activation: requested=%s -> effective=%s (river_should_run=%s; sdb_should_run=%s)",
                     ",".join(cfg.methods_requested), ",".join(cfg.methods_effective),
                     bool(derived.get("river_should_run", False)),
                     bool(derived.get("sdb_should_run", False)))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        log.error("Shared-domain activation failed; aborting before water-method execution.", exc_info=True)
        raise SystemExit(2)

    # ---------------------------------------------------------------------
    # External soundings (extra XYZ): only fetch/normalize if any water method
    # will actually run (shared-domain activation already decided above).
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
                        log.debug("bathy_main: suppressed exception", exc_info=True)
                        dlim_crs = "epsg:4269"

                auto_xyz, auto_rep = fetch_cudem_soundings_via_dlim(
                    aoi=cfg.aoi,
                    sources=requested,
                    cache_root=Path(cfg.cache_root),
                    out_crs=dlim_crs,
                    source_vdatum=getattr(args, "extra_xyz_cudem_source_vdatum", None),
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
                            aoi_bounds = _parse_aoi_bounds_deg(cfg.aoi)
                            reproj_path = _reproject_xyz_file(
                                xyz_path,
                                # Source CRS is the dlim output CRS (horizontal-only). Z is preserved.
                                src_crs=dlim_crs,
                                dst_crs=working_crs,
                                cache_dir=Path(cfg.cache_root) / "xyz",
                                aoi_bounds=aoi_bounds,
                                log_prefix="[XYZ][DLIM] ",
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
    if cfg.glint_correct and ("sdb" not in cfg.methods):
        log.info("[GLINT] --glint-correct set, but methods does not include 'sdb'; ignoring glint options for this run.")
        cfg.glint_correct = False

    # Fatal post-processing issues that should cause a non-zero exit even if a
    # "final" raster exists (e.g., requested reprojection outputs missing).
    fatal_errors: List[str] = []

    report.update({
        "timestamp": report.get("timestamp") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pipeline_version": getattr(constants, "PIPELINE_VERSION", "unknown"),
    })
    report["config"] = {
            "aoi": cfg.aoi,
            "aoi_tile": cfg.aoi_tile,
            "tile_bbox": cfg.tile_bbox if cfg.tile_bbox else None,
            "tile_buffer_km": float(cfg.tile_buffer_km or 0.0),
            "buff_frac": float(getattr(args, "buff", 0.0) or 0.0),
            "tile_edge_taper_enabled": cfg.tile_edge_taper_enabled,
            "tile_edge_taper_km": float(cfg.tile_edge_taper_km or 0.0),
            "tile_edge_smooth_sigma_km": float(cfg.tile_edge_smooth_sigma_km or 0.0),
            "sdb_model_bank_enabled": cfg.sdb_model_bank_enabled,
            "sdb_model_bank": cfg.sdb_model_bank,
            "sdb_bank_max_samples": cfg.sdb_bank_max_samples,
            "sdb_bank_seed": cfg.sdb_bank_seed,
            "sdb_bank_retrain_min_new": cfg.sdb_bank_retrain_min_new,
            "sdb_model_cache_enabled": cfg.sdb_model_cache_enabled,
            "sdb_model_cache_key": cfg.sdb_model_cache_key,
            "start": cfg.start_date,
            "end": cfg.end_date,
            "methods": cfg.methods,
            "priority": cfg.priority,
            "require_river_constraints": cfg.require_river_constraints,
            "out_dir": str(cfg.out_dir),
            "make_figs": bool(getattr(cfg, "make_figs", False)),
            "review_guidance_domains_only": bool(getattr(cfg, "review_guidance_domains_only", False)),
            "river_dem": str(cfg.river_dem) if cfg.river_dem else None,
            "working_srs": str(cfg.working_srs),
            "final_out_srs": str(cfg.final_out_srs),
            "depth_value_type": "elevation",
            "depth_units": "m",
            "depth_sign": None,
            "depth_reference": None,
            "authoritative_base": str(cfg.authoritative_base) if getattr(cfg, "authoritative_base", None) else None,
            "authoritative_base_auto": bool(getattr(cfg, "authoritative_base_auto", False)),
            "authoritative_base_missing_meta_policy": str(getattr(cfg, "authoritative_base_missing_meta_policy", "skip") or "skip"),
        }

    _prepare_authoritative_guidance_inputs(cfg, report=report)
    _prepare_authoritative_river_soundings(cfg, report=report)

    # Merge WAFFLES domain inference into the run report (if available)
    try:
        if isinstance(locals().get('domain_inference'), dict) and domain_inference:
            report.update(domain_inference)
    except Exception:
        log.debug("ignored", exc_info=True)

    # Include any auto-fetched CUDEM soundings in the run report
    if hasattr(args, "_xyz_auto_report"):
        report["xyz_auto"] = getattr(args, "_xyz_auto_report")
    if hasattr(args, "_authoritative_base_auto_report"):
        report["authoritative_base_auto"] = getattr(args, "_authoritative_base_auto_report")

    sdb_raster = None
    river_raster = None

    if bool(getattr(cfg, "review_guidance_domains_only", False)):
        report.setdefault("guidance_domains", {})["status"] = "review_only_stop"
        review_dir = report.get("guidance_domains", {}).get("review_dir")
        if review_dir:
            log.info("[DOMAIN] Review-only stop requested. Inspect guidance domains in %s before running inference.", review_dir)
        return _write_review_only_run_summary(cfg, report, run_id)

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
        req = str(cfg.require_river_constraints or "none").lower().strip()
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
        log.debug("Constraint guardrail check failed; continuing.", exc_info=True)

    # If fusion would have no sources after guardrails, skip it explicitly.
    return _execute_final_run_stage(
        cfg=cfg,
        args=args,
        report=report,
        log=log,
        fatal_errors=fatal_errors,
        sdb_raster=sdb_raster,
        river_raster=river_raster,
        river_for_fuse=river_for_fuse,
        river_excluded=river_excluded,
        fuse_fn=fuse,
        condition_fn=_condition_final_to_authoritative_base,
        reproject_fn=_reproject_final_outputs,
        write_bundle_fn=_write_final_reporting_bundle,
        finalize_run_fn=lambda cfg_, report_, args_, final_, final_for_user_, final_provenance_, fatal_errors_: _finalize_run(cfg_, report_, args_, final_, final_for_user_, final_provenance_, run_id, fatal_errors_),
        write_io_manifest_fn=write_io_manifest,
        emit_artifacts_fn=_emit_optional_artifacts,
        run_seam_comparisons_fn=lambda args_, cfg_, report_, final_, final_for_user_, report_path_: _run_seam_comparisons(args_, cfg_, report_, run_id, final_, final_for_user_, report_path_),
        conditioned_gapfill_fn=_run_gapfill_stage,
    )


if __name__ == "__main__":
    raise SystemExit(main())

