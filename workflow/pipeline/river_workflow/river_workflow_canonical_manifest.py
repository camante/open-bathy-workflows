from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import rasterio

from pipeline.river_workflow.river_workflow_context import RiverWorkflowContext


def sha256_file(path: Path | str | None) -> str | None:
    if path in (None, ""):
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raster_identity(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {"path": None, "exists": False, "sha256": None}
    p = Path(path)
    out: dict[str, Any] = {"path": str(p), "exists": p.exists(), "sha256": sha256_file(p)}
    if p.exists():
        with rasterio.open(p) as ds:
            out.update({
                "width": int(ds.width),
                "height": int(ds.height),
                "crs": str(ds.crs) if ds.crs is not None else None,
                "transform": [float(x) for x in tuple(ds.transform)[:6]],
                "bounds": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
                "nodata": None if ds.nodata is None else float(ds.nodata),
            })
    return out



def _receipt_hash(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {"path": None, "exists": False, "sha256": None}
    p = Path(path)
    return {"path": str(p), "exists": p.exists(), "sha256": sha256_file(p)}


def _read_json_or_none(path: Path | str | None) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _stage_receipt_paths(ctx: RiverWorkflowContext) -> dict[str, Path | None]:
    return {
        "solve_domain": getattr(ctx.paths, "solve_domain_receipt", None),
        "grids": getattr(ctx.paths, "grids_receipt", None),
        "authoritative_inputs": getattr(ctx.paths, "authoritative_receipt", None),
        "centerline_points": getattr(ctx.paths, "centerline_receipt", None),
        "centerline_wse_proxy": getattr(ctx.paths, "wse_proxy_receipt", None),
        "centerline_authoritative_bed": getattr(ctx.paths, "authoritative_bed_receipt", None),
        "centerline_observed_offset": getattr(ctx.paths, "observed_offset_receipt", None),
        "centerline_modeled_offset": getattr(ctx.paths, "modeled_offset_receipt", None),
        "centerline_bed_backbone": getattr(ctx.paths, "backbone_receipt", None),
        "river_corridor_solve": getattr(ctx.paths, "corridor_receipt", None),
        "river_primary_surface_solve": getattr(ctx.paths, "surface_receipt", None),
        "river_primary_surface_solve_locked": getattr(ctx.paths, "lock_receipt", None),
    }


def canonical_science_summary_from_stage_receipts(ctx: RiverWorkflowContext) -> dict[str, Any]:
    """Return compact parent-level science diagnostics from canonical receipts.

    The canonical parent manifest is the durable handoff for AOI exports.  It
    should carry enough read-only science context for exports to summarize WSE,
    offsets, backbone, and lock/support state without rerunning construction.
    """
    science_stages = {
        "wse_proxy": "centerline_wse_proxy",
        "observed_offset": "centerline_observed_offset",
        "modeled_offset": "centerline_modeled_offset",
        "backbone": "centerline_bed_backbone",
    }
    out: dict[str, Any] = {}
    receipt_paths = _stage_receipt_paths(ctx)
    for label, stage_name in science_stages.items():
        path = receipt_paths.get(stage_name)
        payload = _read_json_or_none(path)
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        science = summary.get("river_science") if isinstance(summary.get("river_science"), dict) else None
        if science is None and isinstance(payload.get("science"), dict):
            science = payload.get("science")
        if isinstance(science, dict):
            out[label] = science
    lock_payload = _read_json_or_none(receipt_paths.get("river_primary_surface_solve_locked"))
    if isinstance(lock_payload, dict):
        summary = lock_payload.get("summary") if isinstance(lock_payload.get("summary"), dict) else {}
        out["authoritative_lock"] = {
            key: summary.get(key)
            for key in ("locked_pixel_count", "finite_locked_surface_count", "measured_cells_changed_count")
            if summary.get(key) not in (None, "")
        }
    return out


def construction_stage_receipt_hashes(ctx: RiverWorkflowContext) -> dict[str, dict[str, Any]]:
    """Return hashes for canonical construction receipts that already exist."""
    return {stage: _receipt_hash(path) for stage, path in _stage_receipt_paths(ctx).items()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def canonical_solution_manifest_path(ctx: RiverWorkflowContext) -> Path:
    return ctx.paths.manifests_dir / "canonical_river_solution_manifest.json"


def public_canonical_solution_manifest_path(ctx: RiverWorkflowContext) -> Path:
    # ctx.paths.root is <run>/river_workflow in the active workflow.  The public
    # reports directory survives intermediate cleanup and is where comparison
    # tools should look for durable AOI/canonical identity metadata.
    return ctx.paths.root.parent / "reports" / "canonical_river_solution_manifest.json"


def canonical_solution_cache_manifest_path(ctx: RiverWorkflowContext) -> Path | None:
    """Return the cache-scoped canonical solution manifest used by AOI export-only runs."""
    bundle = ctx.linear_inputs
    if bundle is None:
        return None
    root = getattr(bundle, "canonical_solve_outputs_root", None)
    if root in (None, ""):
        return None
    return Path(root) / "canonical_river_solution_manifest.json"


def build_canonical_solution_manifest_payload(
    *,
    ctx: RiverWorkflowContext,
    canonical_parent_dem: Path,
    canonical_take_mask_path: Path | None,
    canonical_final_dem_path: Path | None,
    solve_grid_template_path: Path,
    authoritative_measured_path: Path,
    authoritative_support_mask_path: Path,
    baseline_background_path: Path,
    locked_surface_path: Path,
    run_contract_path: Path | None,
    composition_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = ctx.linear_inputs
    cache_key = None
    if bundle is not None:
        cache_key = getattr(bundle, "canonical_solve_cache_key", None)
        if cache_key in (None, ""):
            context_path = getattr(bundle, "canonical_source_context_path", None)
            if context_path not in (None, ""):
                try:
                    context_payload = json.loads(Path(context_path).read_text(encoding="utf-8"))
                    cache_key = context_payload.get("canonical_solve_cache_key") or context_payload.get("canonical_cache_key")
                except (OSError, ValueError, TypeError):
                    cache_key = None
        if cache_key in (None, ""):
            contract_path = getattr(bundle, "canonical_solve_contract_path", None)
            if contract_path not in (None, ""):
                try:
                    contract_payload = json.loads(Path(contract_path).read_text(encoding="utf-8"))
                    cache_key = contract_payload.get("canonical_solve_cache_key") or contract_payload.get("canonical_cache_key")
                except (OSError, ValueError, TypeError):
                    cache_key = None
    canonical_system_id = getattr(ctx, "canonical_system_id", None)
    if canonical_system_id in (None, "") and bundle is not None:
        canonical_system_id = getattr(bundle, "canonical_system_id", None)
    payload = {
        "schema_version": 2,
        "role": "canonical_river_solution",
        "execution_model": "canonical_build_then_aoi_export",
        "canonical_system_id": str(canonical_system_id) if canonical_system_id not in (None, "") else None,
        "canonical_cache_key": str(cache_key) if cache_key not in (None, "") else None,
        "canonical_solve_cache_key": str(cache_key) if cache_key not in (None, "") else None,
        "requested_solve_domain": str(ctx.requested_solve_domain) if ctx.requested_solve_domain is not None else None,
        "resolved_solve_domain": str(ctx.resolved_solve_domain) if ctx.resolved_solve_domain is not None else None,
        "canonical_solve_aoi": str(getattr(bundle, "canonical_solve_aoi", None) or ctx.canonical_domain_bounds or ctx.resolved_solve_domain or ""),
        "solve_domain_source": str(ctx.solve_domain_source),
        "projected_crs": str(ctx.projected_crs),
        "target_resolution_m": float(ctx.target_resolution_m),
        "run_contract_path": str(run_contract_path) if run_contract_path is not None else None,
        "source_context_path": str(getattr(bundle, "canonical_source_context_path", None)) if bundle is not None and getattr(bundle, "canonical_source_context_path", None) is not None else None,
        "canonical_parent_dem_path": str(canonical_parent_dem),
        "canonical_stage_class": "canonical_parent_finalization",
        "canonical_parent_dem_sha256": sha256_file(canonical_parent_dem),
        "canonical_parent_content_key": sha256_file(canonical_parent_dem),
        "canonical_comparison_key": sha256_file(canonical_parent_dem),
        "canonical_final_dem_path": str(canonical_final_dem_path) if canonical_final_dem_path is not None else None,
        "canonical_final_dem_sha256": sha256_file(canonical_final_dem_path),
        "canonical_take_mask_path": str(canonical_take_mask_path) if canonical_take_mask_path is not None else None,
        "canonical_take_mask_sha256": sha256_file(canonical_take_mask_path),
        "canonical_construction_stage_receipts": construction_stage_receipt_hashes(ctx),
        "canonical_science_summary": canonical_science_summary_from_stage_receipts(ctx),
        "canonical_science_summary_policy": {
            "source": "canonical_parent_stage_receipts",
            "aoi_exports_read_only": True,
            "aoi_exports_may_recompute_science": False,
        },
        "solve_grid": raster_identity(solve_grid_template_path),
        "canonical_parent_dem": raster_identity(canonical_parent_dem),
        "authoritative_measured_only": raster_identity(authoritative_measured_path),
        "authoritative_support_mask": raster_identity(authoritative_support_mask_path),
        "baseline_background": raster_identity(baseline_background_path),
        "locked_river_surface": raster_identity(locked_surface_path),
        "composition_summary": dict(composition_summary or {}),
        "canonical_parent_policy": {
            "parent_built_once_on_solve_grid": True,
            "aoi_outputs_must_be_exact_parent_subsets": True,
            "post_subset_modification_allowed": False,
        },
        "aoi_export_policy": {
            "aoi_runs_may_recompute_canonical_construction": False,
            "aoi_runs_must_subset_from_parent": True,
            "cache_is_implementation_detail": True,
            "manifest_handoff_required": True,
            "post_subset_modification_allowed": False,
            "exact_parent_window_identity_required": True,
        },
    }
    return payload


def write_canonical_solution_manifest(*, ctx: RiverWorkflowContext, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    private_path = _write_json(canonical_solution_manifest_path(ctx), payload)
    public_path = _write_json(public_canonical_solution_manifest_path(ctx), payload)
    cache_manifest = canonical_solution_cache_manifest_path(ctx)
    if cache_manifest is not None:
        cache_payload = dict(payload)
        canonical_final = cache_payload.get("canonical_final_dem_path")
        if canonical_final not in (None, ""):
            canonical_final_path = Path(str(canonical_final))
            cache_payload["canonical_parent_dem_path"] = str(canonical_final_path)
            cache_payload["canonical_parent_dem_sha256"] = sha256_file(canonical_final_path)
            cache_payload["canonical_parent_content_key"] = sha256_file(canonical_final_path)
            cache_payload["canonical_comparison_key"] = sha256_file(canonical_final_path)
            cache_payload["canonical_parent_dem"] = raster_identity(canonical_final_path)
        _write_json(cache_manifest, cache_payload)
    return private_path, public_path


def load_canonical_solution_manifest(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("role") != "canonical_river_solution":
        raise ValueError(f"invalid_canonical_solution_manifest_role:{payload.get('role')}")
    required = ["canonical_parent_dem_path", "canonical_parent_dem_sha256", "solve_grid"]
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise ValueError(f"invalid_canonical_solution_manifest_missing:{','.join(missing)}")
    return payload


__all__ = [
    "build_canonical_solution_manifest_payload",
    "construction_stage_receipt_hashes",
    "canonical_solution_manifest_path",
    "canonical_solution_cache_manifest_path",
    "load_canonical_solution_manifest",
    "public_canonical_solution_manifest_path",
    "raster_identity",
    "sha256_file",
    "write_canonical_solution_manifest",
]
