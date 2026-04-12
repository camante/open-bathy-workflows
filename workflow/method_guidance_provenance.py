from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean_string(value: Any) -> Optional[str]:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _resolve_existing_path(value: Any, *, out_root: Optional[Path]) -> Optional[str]:
    raw = _clean_string(value)
    if raw is None:
        return None
    path = Path(raw)
    candidates = [path]
    if out_root is not None and not path.is_absolute():
        candidates.insert(0, (out_root / path))
    for cand in candidates:
        try:
            if cand.exists():
                try:
                    if out_root is not None:
                        return str(cand.resolve().relative_to(out_root.resolve()))
                except ValueError:
                    pass
                return str(cand.resolve())
        except OSError:
            continue
    return None


def _load_manifest(path_value: Any, *, out_root: Optional[Path]) -> Dict[str, Any]:
    path_str = _resolve_existing_path(path_value, out_root=out_root)
    if not path_str:
        return {}
    try:
        path = Path(path_str) if Path(path_str).is_absolute() else (out_root / path_str if out_root is not None else Path(path_str))
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _artifact_role(manifest: Dict[str, Any], key: str) -> Optional[str]:
    roles = _as_dict(manifest.get("artifact_roles"))
    value = roles.get(key)
    return str(value) if isinstance(value, str) and value.strip() else None


def _shared_domain_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(_as_dict(_as_dict(report.get("shared_domain_stage")).get("summary")))


def _shared_domain_masks(report: Dict[str, Any]) -> Dict[str, Any]:
    summary = _shared_domain_summary(report)
    masks = _as_dict(summary.get("masks"))
    if masks:
        return masks
    return {
        "river_candidate_domain_mask": _as_dict(report.get("outputs")).get("river_candidate_domain_mask"),
        "river_active_domain_mask": _as_dict(report.get("outputs")).get("river_active_domain_mask"),
        "sdb_candidate_domain_mask": _as_dict(report.get("outputs")).get("sdb_candidate_domain_mask"),
    }


def _shared_activation(report: Dict[str, Any]) -> Dict[str, Any]:
    activation_truth = _as_dict(report.get("method_activation_truth"))
    if activation_truth:
        river = _as_dict(activation_truth.get("river"))
        sdb = _as_dict(activation_truth.get("sdb"))
        return {
            "river_should_run": bool(river.get("effective_should_run", False)),
            "sdb_should_run": bool(sdb.get("effective_should_run", False)),
        }
    summary = _shared_domain_summary(report)
    activation = _as_dict(summary.get("derived_activation"))
    if activation:
        return activation
    return _as_dict(_as_dict(_as_dict(report.get("guidance_domains")).get("activation")).get("derived_activation"))


def _river_architecture(report: Dict[str, Any], *, out_root: Optional[Path]) -> Dict[str, Any]:
    river = _as_dict(report.get("river"))
    outputs = _as_dict(river.get("outputs"))
    masks = _shared_domain_masks(report)
    activation = _shared_activation(report)
    manifest = _load_manifest(outputs.get("guidance_manifest"), out_root=out_root)

    candidate_domain = _resolve_existing_path(masks.get("river_candidate_domain_mask"), out_root=out_root)
    active_domain = _resolve_existing_path(masks.get("river_active_domain_mask"), out_root=out_root)
    auth_mask = _resolve_existing_path(outputs.get("authoritative_support"), out_root=out_root)
    auth_values = _resolve_existing_path(outputs.get("authoritative_support_depth"), out_root=out_root)

    raw_method_output = (
        _resolve_existing_path(outputs.get("depth_terrain_internal_helper"), out_root=out_root)
        or _resolve_existing_path(outputs.get("bottom_elevation_internal_helper"), out_root=out_root)
        or _resolve_existing_path(outputs.get("depth_terrain"), out_root=out_root)
    )
    locked_guidance_product = None
    active_guidance_key = None
    active_guidance_product = None
    for key in ("channel_surface", "guide_points", "centerline_points"):
        resolved = _resolve_existing_path(outputs.get(key), out_root=out_root)
        if resolved:
            active_guidance_key = key
            active_guidance_product = resolved
            break

    guidance_mode = (
        "channel_surface_guidance" if active_guidance_key == "channel_surface"
        else "structured_sparse_guidance" if active_guidance_key == "guide_points"
        else "structured_guidance"
    )
    active_product_role = _artifact_role(manifest, active_guidance_key) if active_guidance_key else None
    domain_should_run = bool(activation.get("river_should_run", False))

    status = str(river.get("status") or "").strip().lower()
    if status in {"success", "failed", "skipped"}:
        final_status = status
    elif active_guidance_product:
        final_status = "success"
    elif domain_should_run:
        final_status = "missing_active_guidance"
    else:
        final_status = "inactive"

    return {
        "candidate_domain_mask": candidate_domain,
        "active_domain_mask": active_domain,
        "authoritative_support_mask": auth_mask,
        "authoritative_support_values": auth_values,
        "authoritative_support_contract": None,
        "raw_method_output": raw_method_output,
        "locked_guidance_product": locked_guidance_product,
        "active_guidance_product": active_guidance_product,
        "guidance_mode": guidance_mode,
        "active_product_role": active_product_role,
        "domain_should_run": domain_should_run,
        "status": final_status,
    }


def _sdb_architecture(report: Dict[str, Any], *, out_root: Optional[Path]) -> Dict[str, Any]:
    sdb = _as_dict(report.get("sdb"))
    outputs = _as_dict(report.get("outputs"))
    artifacts = _as_dict(sdb.get("artifacts"))
    masks = _shared_domain_masks(report)
    activation = _shared_activation(report)
    manifest = _load_manifest(sdb.get("guidance_manifest") or artifacts.get("guidance_manifest"), out_root=out_root)

    candidate_domain = _resolve_existing_path(masks.get("sdb_candidate_domain_mask"), out_root=out_root)
    active_domain = (
        _resolve_existing_path(sdb.get("shared_domain_mask_used"), out_root=out_root)
        or candidate_domain
    )
    auth_mask = _resolve_existing_path(artifacts.get("authoritative_support_mask") or outputs.get("sdb_authoritative_support_mask"), out_root=out_root)
    auth_values = _resolve_existing_path(artifacts.get("authoritative_support_values") or outputs.get("sdb_authoritative_support_values"), out_root=out_root)
    auth_points = _resolve_existing_path(artifacts.get("authoritative_support_points") or outputs.get("sdb_authoritative_support_points"), out_root=out_root)
    auth_contract = _resolve_existing_path(artifacts.get("authoritative_support_contract") or outputs.get("sdb_authoritative_support_contract"), out_root=out_root)

    raw_method_output = _resolve_existing_path(sdb.get("raw_prediction_raster") or artifacts.get("raw_prediction_raster"), out_root=out_root)
    locked_guidance_product = _resolve_existing_path(sdb.get("locked_guidance_raster") or artifacts.get("sdb_locked_guidance_raster"), out_root=out_root)
    active_guidance_product = _resolve_existing_path(sdb.get("guidance_active") or artifacts.get("sdb_guidance_active") or artifacts.get("depth_raster"), out_root=out_root)

    active_product_role = (
        _artifact_role(manifest, "sdb_guidance_active")
        or _artifact_role(manifest, "depth_raster")
        or _clean_string(sdb.get("depth_raster_role"))
    )
    guidance_mode = _clean_string(sdb.get("guidance_mode")) or _clean_string(artifacts.get("guidance_mode")) or "guidance_first"
    domain_should_run = bool(sdb.get("shared_domain_activation_should_run", activation.get("sdb_should_run", False)))

    status = str(sdb.get("status") or "").strip().lower()
    if status in {"success", "failed", "skipped"}:
        final_status = status
    elif active_guidance_product:
        final_status = "success"
    elif domain_should_run:
        final_status = "missing_active_guidance"
    else:
        final_status = "inactive"

    return {
        "candidate_domain_mask": candidate_domain,
        "active_domain_mask": active_domain,
        "authoritative_support_mask": auth_mask,
        "authoritative_support_values": auth_values,
        "authoritative_support_points": auth_points,
        "authoritative_support_contract": auth_contract,
        "raw_method_output": raw_method_output,
        "locked_guidance_product": locked_guidance_product,
        "active_guidance_product": active_guidance_product,
        "guidance_mode": guidance_mode,
        "active_product_role": active_product_role,
        "domain_should_run": domain_should_run,
        "status": final_status,
    }


def apply_parallel_method_guidance_summary(*, report: Dict[str, Any], out_root: Optional[str | Path] = None) -> Dict[str, Any]:
    root = Path(out_root) if out_root is not None else None
    parallel = {
        "river": _river_architecture(report, out_root=root),
        "sdb": _sdb_architecture(report, out_root=root),
    }
    report["parallel_method_guidance"] = parallel
    report.setdefault("river", {})["guidance_architecture"] = dict(parallel["river"])
    report.setdefault("sdb", {})["guidance_architecture"] = dict(parallel["sdb"])
    return parallel
