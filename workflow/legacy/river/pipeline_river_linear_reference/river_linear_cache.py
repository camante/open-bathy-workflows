from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

WORKFLOW_CONTRACT_VERSION = "seamless_dem_parent_export_v20_backbone_anchor_conflict_reporting"


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    return value


def _file_fingerprint(path: Path | str | None) -> dict[str, Any]:
    if path in (None, ""):
        return {"path": None, "exists": False, "sha256": None}
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "sha256": None}
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(p), "exists": True, "sha256": h.hexdigest()}


def fingerprint_authoritative_source(ctx: Any) -> dict[str, Any]:
    bundle = getattr(ctx, "linear_inputs", None)
    return {
        "measured_only": _file_fingerprint(getattr(bundle, "canonical_solve_authoritative_measured_only_path", None)),
        "support_mask": _file_fingerprint(getattr(bundle, "canonical_solve_authoritative_support_mask_path", None)),
        "baseline_background": _file_fingerprint(getattr(bundle, "canonical_solve_baseline_background_path", None)),
    }


def fingerprint_river_network(ctx: Any) -> dict[str, Any]:
    bundle = getattr(ctx, "linear_inputs", None)
    return {
        "network": _file_fingerprint(getattr(bundle, "canonical_solve_network_source_gpkg", None)),
        "solve_aoi": str(getattr(bundle, "canonical_solve_aoi", None) or getattr(ctx, "resolved_solve_domain", None) or ""),
    }


def fingerprint_solve_grid(ctx: Any) -> dict[str, Any]:
    bundle = getattr(ctx, "linear_inputs", None)
    return {
        "grid_template": _file_fingerprint(getattr(bundle, "canonical_solve_grid_template_path", None)),
        "target_resolution_m": float(getattr(ctx, "target_resolution_m", 0.0) or 0.0),
        "projected_crs": str(getattr(ctx, "projected_crs", "") or ""),
    }


def _cache_key_payload(value: Any) -> Any:
    """Return a path-independent payload for canonical cache key hashing."""
    if isinstance(value, Path):
        return None
    if isinstance(value, dict):
        # File paths are useful receipt metadata, but they must not participate in
        # the scientific cache identity because they can vary by out-dir/cache-root.
        return {str(k): _cache_key_payload(v) for k, v in value.items() if str(k) != "path"}
    if isinstance(value, (list, tuple)):
        return [_cache_key_payload(v) for v in value]
    return value


def canonical_cache_key_from_metadata(metadata: Mapping[str, Any]) -> str:
    payload = json.dumps(_cache_key_payload(dict(metadata)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_canonical_cache_metadata(ctx: Any) -> dict[str, Any]:
    metadata = {
        "canonical_system_id": str(getattr(ctx, "canonical_system_id", "") or ""),
        "authoritative_source_fingerprint": fingerprint_authoritative_source(ctx),
        "river_network_fingerprint": fingerprint_river_network(ctx),
        "solve_grid_fingerprint": fingerprint_solve_grid(ctx),
        "workflow_contract_version": WORKFLOW_CONTRACT_VERSION,
        "critical_science_parameters": {
            "canonical_max_trace_km": float(getattr(ctx, "canonical_max_trace_km", 0.0) or 0.0),
            "target_resolution_m": float(getattr(ctx, "target_resolution_m", 0.0) or 0.0),
        },
        "identity_excludes": ["user_aoi", "out_dir", "timestamp", "temporary_directory", "run_id"],
    }
    metadata["canonical_cache_key"] = canonical_cache_key_from_metadata(metadata)
    return metadata


def canonical_cache_key(ctx: Any) -> str:
    return str(build_canonical_cache_metadata(ctx)["canonical_cache_key"])


def validate_cached_parent_dem(*, ctx: Any, parent_dem: Path | str, parent_receipt: Path | str) -> None:
    parent = Path(parent_dem)
    receipt = Path(parent_receipt)
    if not parent.exists():
        raise RuntimeError(f"cached_parent_dem_missing:{parent}")
    if not receipt.exists():
        raise RuntimeError(f"cached_parent_receipt_missing:{receipt}")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    cached = payload.get("canonical_cache") if isinstance(payload, dict) else None
    if not isinstance(cached, dict):
        raise RuntimeError("cached_parent_receipt_missing_canonical_cache")
    expected = build_canonical_cache_metadata(ctx)
    for key in ("canonical_system_id", "canonical_cache_key", "workflow_contract_version"):
        if str(cached.get(key)) != str(expected.get(key)):
            raise RuntimeError(f"cached_parent_{key}_mismatch:{cached.get(key)}:{expected.get(key)}")


__all__ = [
    "WORKFLOW_CONTRACT_VERSION",
    "build_canonical_cache_metadata",
    "canonical_cache_key",
    "canonical_cache_key_from_metadata",
    "fingerprint_authoritative_source",
    "fingerprint_river_network",
    "fingerprint_solve_grid",
    "validate_cached_parent_dem",
]
