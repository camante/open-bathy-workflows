from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from final_reporting import evaluate_overlap_identity_checks
from seam_metrics import (
    compute_array_overlap_identity_metrics,
    compute_raster_overlap_identity_metrics,
    compute_raster_trusted_interior_identity_metrics,
    compute_vector_overlap_identity_metrics,
    compute_vector_trusted_interior_identity_metrics,
)
from support_classes import SUPPORT_CLASS_CODE_TO_NAME


_RASTER_OVERLAP_SPECS: Tuple[Tuple[str, str], ...] = (
    ("selected_final_depth", "selected_final_depth"),
    ("support_class", "support_class"),
    ("final_provenance_native", "final_provenance_native"),
    ("river_trusted_interior", "river_trusted_interior"),
    ("river_channel_surface_graph_mode", "river_channel_surface_graph_mode"),
    ("river_channel_surface_support_class", "river_channel_surface_support_class"),
    ("river_channel_surface_uncertainty", "river_channel_surface_uncertainty"),
    ("river_channel_surface_hard_lock", "river_channel_surface_hard_lock"),
    ("river_channel_surface_junction_constrained", "river_channel_surface_junction_constrained"),
    ("river_channel_surface_unsupported_span", "river_channel_surface_unsupported_span"),
    ("river_channel_surface_unsupported_regime", "river_channel_surface_unsupported_regime"),
    ("river_channel_surface_residual_to_candidate", "river_channel_surface_residual_to_candidate"),
    ("river_generalized_longitudinal_bed_base", "river_generalized_longitudinal_bed_base"),
    ("river_generalized_longitudinal_bed_reconciled", "river_generalized_longitudinal_bed_reconciled"),
    ("river_longitudinal_profile_local_authoritative_reconciliation", "river_longitudinal_profile_local_authoritative_reconciliation"),
    ("river_longitudinal_profile_local_authoritative_reconciliation_influence", "river_longitudinal_profile_local_authoritative_reconciliation_influence"),
)

_VECTOR_SPECS: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    (
        "river_graph_backbone_diagnostics",
        ("component_id", "station_m"),
        (
            "graph_backbone_z_m",
            "graph_hard_lock",
            "graph_junction_constrained",
            "graph_solution_mode",
            "graph_solver_support_class",
            "graph_uncertainty_class",
            "graph_residual_to_candidate_z_m",
            "graph_unsupported_span_m",
        ),
    ),
)


def _load_manifest(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))



def _existing(path: Optional[str | Path]) -> Optional[str]:
    if not path:
        return None
    p = Path(str(path))
    return str(p) if p.exists() else None



def _read_overlap_arrays(ds_a, ds_b):
    from rasterio.windows import from_bounds

    oxmin = max(ds_a.bounds.left, ds_b.bounds.left)
    oymin = max(ds_a.bounds.bottom, ds_b.bounds.bottom)
    oxmax = min(ds_a.bounds.right, ds_b.bounds.right)
    oymax = min(ds_a.bounds.top, ds_b.bounds.top)
    if oxmax <= oxmin or oymax <= oymin:
        return None
    wa = from_bounds(oxmin, oymin, oxmax, oymax, transform=ds_a.transform).round_offsets().round_lengths()
    wb = from_bounds(oxmin, oymin, oxmax, oymax, transform=ds_b.transform).round_offsets().round_lengths()
    arr_a = ds_a.read(1, window=wa)
    arr_b = ds_b.read(1, window=wb)
    ny = min(arr_a.shape[0], arr_b.shape[0])
    nx = min(arr_a.shape[1], arr_b.shape[1])
    if ny <= 0 or nx <= 0:
        return None
    return arr_a[:ny, :nx], arr_b[:ny, :nx]



def _compute_support_class_depth_identity_checks(
    *,
    depth_a: str,
    depth_b: str,
    support_a: str,
    support_b: str,
    trusted_a: Optional[str],
    trusted_b: Optional[str],
    neighbor_manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import rasterio

    overlap_rows: list[dict[str, Any]] = []
    trusted_rows: list[dict[str, Any]] = []
    with rasterio.open(depth_a) as dda, rasterio.open(depth_b) as ddb, rasterio.open(support_a) as sda, rasterio.open(support_b) as sdb:
        if dda.crs != ddb.crs or dda.crs != sda.crs or dda.crs != sdb.crs:
            return [
                {
                    "scope": "overlap",
                    "artifact": "selected_final_depth_by_support_class",
                    "neighbor_final_outputs_manifest": str(neighbor_manifest_path),
                    "status": "not_aligned",
                    "reason": "CRS mismatch among depth/support rasters",
                }
            ], []
        depth_pair = _read_overlap_arrays(dda, ddb)
        support_pair = _read_overlap_arrays(sda, sdb)
        if depth_pair is None or support_pair is None:
            return [
                {
                    "scope": "overlap",
                    "artifact": "selected_final_depth_by_support_class",
                    "neighbor_final_outputs_manifest": str(neighbor_manifest_path),
                    "status": "no_valid",
                    "reason": "no_overlap",
                }
            ], []
        depth_arr_a, depth_arr_b = depth_pair
        support_arr_a, support_arr_b = support_pair
        ny = min(depth_arr_a.shape[0], depth_arr_b.shape[0], support_arr_a.shape[0], support_arr_b.shape[0])
        nx = min(depth_arr_a.shape[1], depth_arr_b.shape[1], support_arr_a.shape[1], support_arr_b.shape[1])
        depth_arr_a = depth_arr_a[:ny, :nx]
        depth_arr_b = depth_arr_b[:ny, :nx]
        support_arr_a = support_arr_a[:ny, :nx].astype(np.int32, copy=False)
        support_arr_b = support_arr_b[:ny, :nx].astype(np.int32, copy=False)
        common_support = support_arr_a == support_arr_b
        trusted_mask = None
        if trusted_a and trusted_b:
            with rasterio.open(trusted_a) as tda, rasterio.open(trusted_b) as tdb:
                if tda.crs == tdb.crs == dda.crs:
                    trusted_pair = _read_overlap_arrays(tda, tdb)
                    if trusted_pair is not None:
                        ta, tb = trusted_pair
                        tny = min(ny, ta.shape[0], tb.shape[0])
                        tnx = min(nx, ta.shape[1], tb.shape[1])
                        depth_arr_a = depth_arr_a[:tny, :tnx]
                        depth_arr_b = depth_arr_b[:tny, :tnx]
                        support_arr_a = support_arr_a[:tny, :tnx]
                        support_arr_b = support_arr_b[:tny, :tnx]
                        common_support = common_support[:tny, :tnx]
                        trusted_mask = (ta[:tny, :tnx] > 0) & (tb[:tny, :tnx] > 0)
        for code, name in SUPPORT_CLASS_CODE_TO_NAME.items():
            mask = common_support & (support_arr_a == int(code))
            n_class = int(np.count_nonzero(mask))
            row = {
                "scope": "overlap",
                "artifact": "selected_final_depth_by_support_class",
                "neighbor_final_outputs_manifest": str(neighbor_manifest_path),
                "support_class_code": int(code),
                "support_class_name": str(name),
                "common_support_pixels": n_class,
            }
            if n_class <= 0:
                row.update({"status": "no_valid", "reason": "no_common_support_pixels"})
            else:
                out = compute_array_overlap_identity_metrics(
                    np.where(mask, depth_arr_a, np.nan),
                    np.where(mask, depth_arr_b, np.nan),
                )
                row.update(out)
            overlap_rows.append(row)
            if trusted_mask is not None:
                tmask = mask & trusted_mask
                trow = {
                    "scope": "trusted_interior",
                    "artifact": "selected_final_depth_by_support_class",
                    "neighbor_final_outputs_manifest": str(neighbor_manifest_path),
                    "support_class_code": int(code),
                    "support_class_name": str(name),
                    "common_support_pixels": n_class,
                    "trusted_common_support_pixels": int(np.count_nonzero(tmask)),
                }
                if not np.any(tmask):
                    trow.update({"status": "no_valid", "reason": "no_common_trusted_support_pixels"})
                else:
                    out = compute_array_overlap_identity_metrics(
                        np.where(tmask, depth_arr_a, np.nan),
                        np.where(tmask, depth_arr_b, np.nan),
                    )
                    trow.update(out)
                trusted_rows.append(trow)
    return overlap_rows, trusted_rows



def _run_pairwise_nested_aoi_checks(
    *,
    current_manifest: Dict[str, Any],
    neighbor_manifest: Dict[str, Any],
    neighbor_manifest_path: Path,
) -> Dict[str, Any]:
    overlap_checks: list[dict[str, Any]] = []
    trusted_checks: list[dict[str, Any]] = []

    current_trusted = _existing(current_manifest.get("river_trusted_interior"))
    neighbor_trusted = _existing(neighbor_manifest.get("river_trusted_interior"))

    for label, manifest_key in _RASTER_OVERLAP_SPECS:
        current_path = _existing(current_manifest.get(manifest_key))
        neighbor_path = _existing(neighbor_manifest.get(manifest_key))
        if not current_path or not neighbor_path:
            continue
        stats = compute_raster_overlap_identity_metrics(current_path, neighbor_path)
        stats.update({"artifact": label, "neighbor_final_outputs_manifest": str(neighbor_manifest_path)})
        overlap_checks.append(stats)
        if current_trusted and neighbor_trusted:
            trusted = compute_raster_trusted_interior_identity_metrics(current_path, neighbor_path, current_trusted, neighbor_trusted)
            trusted.update({"artifact": f"trusted_interior::{label}", "neighbor_final_outputs_manifest": str(neighbor_manifest_path)})
            trusted_checks.append(trusted)

    for label, key_fields, compare_fields in _VECTOR_SPECS:
        current_path = _existing(current_manifest.get(label))
        neighbor_path = _existing(neighbor_manifest.get(label))
        if current_path and neighbor_path:
            stats = compute_vector_overlap_identity_metrics(
                current_path,
                neighbor_path,
                key_fields=key_fields,
                compare_fields=compare_fields,
            )
            stats.update({"artifact": label, "neighbor_final_outputs_manifest": str(neighbor_manifest_path)})
            overlap_checks.append(stats)
            if current_trusted and neighbor_trusted:
                trusted = compute_vector_trusted_interior_identity_metrics(
                    current_path,
                    neighbor_path,
                    current_trusted,
                    neighbor_trusted,
                    key_fields=key_fields,
                    compare_fields=compare_fields,
                )
                trusted.update({"artifact": f"trusted_interior::{label}", "neighbor_final_outputs_manifest": str(neighbor_manifest_path)})
                trusted_checks.append(trusted)

    depth_a = _existing(current_manifest.get("selected_final_depth"))
    depth_b = _existing(neighbor_manifest.get("selected_final_depth"))
    support_a = _existing(current_manifest.get("support_class"))
    support_b = _existing(neighbor_manifest.get("support_class"))
    support_class_checks: list[dict[str, Any]] = []
    trusted_support_class_checks: list[dict[str, Any]] = []
    if depth_a and depth_b and support_a and support_b:
        support_class_checks, trusted_support_class_checks = _compute_support_class_depth_identity_checks(
            depth_a=depth_a,
            depth_b=depth_b,
            support_a=support_a,
            support_b=support_b,
            trusted_a=current_trusted,
            trusted_b=neighbor_trusted,
            neighbor_manifest_path=neighbor_manifest_path,
        )

    return {
        "neighbor_final_outputs_manifest": str(neighbor_manifest_path),
        "neighbor_aoi": neighbor_manifest.get("aoi"),
        "overlap_identity_checks": overlap_checks,
        "trusted_interior_identity_checks": trusted_checks,
        "support_class_depth_identity_checks": support_class_checks,
        "trusted_support_class_depth_identity_checks": trusted_support_class_checks,
    }



def run_nested_aoi_regression(
    *,
    current_final_outputs_manifest: str | Path,
    neighbor_final_outputs_manifests: Sequence[str | Path],
    overlap_tolerance: float = 1.0e-6,
    trusted_tolerance: float = 1.0e-6,
) -> Dict[str, Any]:
    current_manifest_path = Path(current_final_outputs_manifest)
    current_manifest = _load_manifest(current_manifest_path)
    comparisons = []
    all_overlap_checks = []
    all_trusted_checks = []
    all_class_checks = []
    all_trusted_class_checks = []
    for neighbor in neighbor_final_outputs_manifests:
        neighbor_path = Path(str(neighbor))
        if not neighbor_path.exists():
            comparisons.append({
                "neighbor_final_outputs_manifest": str(neighbor_path),
                "status": "error",
                "reason": "neighbor final_outputs manifest not found",
                "overlap_identity_checks": [],
                "trusted_interior_identity_checks": [],
                "support_class_depth_identity_checks": [],
                "trusted_support_class_depth_identity_checks": [],
            })
            continue
        neighbor_manifest = _load_manifest(neighbor_path)
        pair = _run_pairwise_nested_aoi_checks(
            current_manifest=current_manifest,
            neighbor_manifest=neighbor_manifest,
            neighbor_manifest_path=neighbor_path,
        )
        pair["overlap_identity_evaluation"] = evaluate_overlap_identity_checks(
            pair["overlap_identity_checks"], tolerance=float(overlap_tolerance)
        )
        pair["trusted_interior_identity_evaluation"] = evaluate_overlap_identity_checks(
            pair["trusted_interior_identity_checks"], tolerance=float(trusted_tolerance)
        )
        comparisons.append(pair)
        all_overlap_checks.extend(pair["overlap_identity_checks"])
        all_trusted_checks.extend(pair["trusted_interior_identity_checks"])
        all_class_checks.extend(pair.get("support_class_depth_identity_checks", []))
        all_trusted_class_checks.extend(pair.get("trusted_support_class_depth_identity_checks", []))

    overlap_eval = evaluate_overlap_identity_checks(all_overlap_checks, tolerance=float(overlap_tolerance))
    trusted_eval = evaluate_overlap_identity_checks(all_trusted_checks, tolerance=float(trusted_tolerance))
    return {
        "current_final_outputs_manifest": str(current_manifest_path),
        "current_aoi": current_manifest.get("aoi"),
        "neighbor_count": len(neighbor_final_outputs_manifests),
        "comparisons": comparisons,
        "overlap_identity_checks": all_overlap_checks,
        "trusted_interior_identity_checks": all_trusted_checks,
        "support_class_depth_identity_checks": all_class_checks,
        "trusted_support_class_depth_identity_checks": all_trusted_class_checks,
        "overlap_identity_evaluation": overlap_eval,
        "trusted_interior_identity_evaluation": trusted_eval,
        "all_overlap_identity_ok": overlap_eval.get("all_ok"),
        "all_trusted_interior_identity_ok": trusted_eval.get("all_ok"),
    }
