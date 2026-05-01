from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CANONICAL_RIVER_SOLVE_CACHE_CONTRACT_VERSION = "seamless_dem_parent_export_v20_backbone_anchor_conflict_reporting"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return {
            "exists": False,
        }
    return {
        "exists": True,
        "size": int(st.st_size),
        "sha256": _file_sha256(p),
    }


def build_canonical_network_identity_tag(
    *,
    river_system_id: str | None,
    canonical_solve_aoi: str,
    solve_stop_reason: str | None,
    selected_reach_count: int | None,
    selected_reach_length_m: float | None,
    canonical_domain_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = dict(canonical_domain_summary or {})
    return {
        "river_system_id": str(river_system_id) if river_system_id is not None else None,
        "canonical_solve_aoi": str(canonical_solve_aoi),
        "solve_stop_reason": str(solve_stop_reason) if solve_stop_reason is not None else None,
        "selected_reach_count": int(selected_reach_count) if selected_reach_count is not None else None,
        # selected_reach_length_m intentionally excluded from cache identity.
        # It is diagnostic but can vary by tiny geometry/driver roundoff.
        # Keep only stable solve-domain identity fields here.
        # Do not include export-seed-dependent counters like export_reach_count or
        # run-local path material, because those are not part of the canonical solve identity.
        "shared_system_selection_column": (
            str(summary.get("shared_system_selection_column"))
            if summary.get("shared_system_selection_column") is not None
            else None
        ),
    }


def canonical_river_solve_identity_payload(
    *,
    river_system_id: str | None,
    canonical_solve_aoi: str,
    projected_crs: str,
    target_resolution_m: float = 0.0,
    canonical_network_path: Path | None = None,
    canonical_network_tag: dict[str, Any] | str | None = None,
    authoritative_source_path: Path | None = None,
    baseline_source_path: Path | None = None,
    authoritative_source_tag: str | None = None,
    baseline_source_tag: str | None = None,
    workflow_contract_version: str = CANONICAL_RIVER_SOLVE_CACHE_CONTRACT_VERSION,
) -> dict[str, Any]:
    return {
        "identity_version": "canonical_solve_v3_contract_versioned",
        "workflow_contract_version": str(workflow_contract_version),
        "river_system_id": str(river_system_id) if river_system_id is not None else None,
        "canonical_solve_aoi": str(canonical_solve_aoi),
        "projected_crs": str(projected_crs),
        "target_resolution_m": float(target_resolution_m),
        "canonical_network": (
            canonical_network_tag
            if canonical_network_tag is not None
            else _content_fingerprint(canonical_network_path)
        ),
        "authoritative_source": _content_fingerprint(authoritative_source_path),
        "baseline_source": _content_fingerprint(baseline_source_path),
        "authoritative_source_tag": str(authoritative_source_tag) if authoritative_source_tag is not None else None,
        "baseline_source_tag": str(baseline_source_tag) if baseline_source_tag is not None else None,
    }


def canonical_river_solve_cache_key(
    *,
    river_system_id: str | None,
    canonical_solve_aoi: str,
    projected_crs: str,
    target_resolution_m: float = 0.0,
    canonical_network_path: Path | None = None,
    canonical_network_tag: dict[str, Any] | str | None = None,
    authoritative_source_path: Path | None = None,
    baseline_source_path: Path | None = None,
    authoritative_source_contract_path: Path | None = None,
    baseline_source_contract_path: Path | None = None,
    authoritative_source_tag: str | None = None,
    baseline_source_tag: str | None = None,
    workflow_contract_version: str = CANONICAL_RIVER_SOLVE_CACHE_CONTRACT_VERSION,
) -> str:
    payload = canonical_river_solve_identity_payload(
        river_system_id=river_system_id,
        canonical_solve_aoi=canonical_solve_aoi,
        projected_crs=projected_crs,
        target_resolution_m=target_resolution_m,
        canonical_network_path=canonical_network_path,
        canonical_network_tag=canonical_network_tag,
        authoritative_source_path=authoritative_source_path,
        baseline_source_path=baseline_source_path,
        authoritative_source_tag=authoritative_source_tag,
        baseline_source_tag=baseline_source_tag,
        workflow_contract_version=workflow_contract_version,
    )
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:20]


def canonical_river_solve_root(*, cache_root: Path, cache_key: str) -> Path:
    return Path(cache_root) / "river_workflow_canonical" / str(cache_key)


def write_canonical_river_solve_contract(
    path: Path,
    *,
    river_system_id: str | None,
    export_aoi: str,
    canonical_solve_aoi: str,
    projected_crs: str,
    target_resolution_m: float,
    canonical_network_path: Path,
    canonical_receipt_path: Path | None,
    authoritative_source_contract_path: Path | None,
    baseline_source_contract_path: Path | None,
    solve_bundle_root: Path,
    solve_outputs_root: Path,
    canonical_solve_grid_path: Path | None = None,
    canonical_solve_authoritative_measured_only_path: Path | None = None,
    canonical_solve_authoritative_support_mask_path: Path | None = None,
    canonical_solve_baseline_background_path: Path | None = None,
    canonical_solve_take_mask_path: Path | None = None,
    canonical_solve_final_dem_path: Path | None = None,
    canonical_solve_cache_manifest_path: Path | None = None,
    canonical_solve_cache_key: str | None = None,
    canonical_network_identity_tag: dict[str, Any] | None = None,
    authoritative_source_tag: str | None = None,
    baseline_source_tag: str | None = None,
    grid_policy: str = "canonical_solve_domain_fixed_for_system",
    resolution_policy: str = "round_to_0.01m",
    canonical_identity_path: Path | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "contract_type": "canonical_river_solve_contract",
        "workflow_contract_version": CANONICAL_RIVER_SOLVE_CACHE_CONTRACT_VERSION,
        "river_system_id": str(river_system_id) if river_system_id is not None else None,
        "export_aoi": str(export_aoi),
        "canonical_solve_aoi": str(canonical_solve_aoi),
        "projected_crs": str(projected_crs),
        "target_resolution_m": float(target_resolution_m),
        "grid_policy": str(grid_policy),
        "resolution_policy": str(resolution_policy),
        "canonical_solve_cache_key": str(canonical_solve_cache_key) if canonical_solve_cache_key is not None else None,
        "canonical_network_path": str(canonical_network_path),
        "canonical_receipt_path": str(canonical_receipt_path) if canonical_receipt_path is not None else None,
        "authoritative_source_contract_path": str(authoritative_source_contract_path) if authoritative_source_contract_path is not None else None,
        "baseline_source_contract_path": str(baseline_source_contract_path) if baseline_source_contract_path is not None else None,
        "solve_bundle_root": str(solve_bundle_root),
        "solve_outputs_root": str(solve_outputs_root),
        "canonical_solve_grid_path": str(canonical_solve_grid_path) if canonical_solve_grid_path is not None else None,
        "canonical_solve_authoritative_measured_only_path": str(canonical_solve_authoritative_measured_only_path) if canonical_solve_authoritative_measured_only_path is not None else None,
        "canonical_solve_authoritative_support_mask_path": str(canonical_solve_authoritative_support_mask_path) if canonical_solve_authoritative_support_mask_path is not None else None,
        "canonical_solve_baseline_background_path": str(canonical_solve_baseline_background_path) if canonical_solve_baseline_background_path is not None else None,
        "canonical_solve_take_mask_path": str(canonical_solve_take_mask_path) if canonical_solve_take_mask_path is not None else None,
        "canonical_solve_final_dem_path": str(canonical_solve_final_dem_path) if canonical_solve_final_dem_path is not None else None,
        "canonical_solve_cache_manifest_path": str(canonical_solve_cache_manifest_path) if canonical_solve_cache_manifest_path is not None else None,
        "canonical_network_identity_tag": dict(canonical_network_identity_tag) if canonical_network_identity_tag is not None else None,
        "authoritative_source_tag": str(authoritative_source_tag) if authoritative_source_tag is not None else None,
        "baseline_source_tag": str(baseline_source_tag) if baseline_source_tag is not None else None,
        "canonical_identity_path": str(canonical_identity_path) if canonical_identity_path is not None else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
