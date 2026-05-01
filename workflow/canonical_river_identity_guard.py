from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


_IDENTITY_FIELDS = (
    "canonical_system_id",
    "canonical_solve_aoi",
    "canonical_solve_stop_reason",
    "canonical_selected_reach_count",
    "projected_crs",
    "target_resolution_m",
    "resolution_policy",
    "authoritative_source_tag",
    "baseline_source_tag",
    "authoritative_routing_policy",
)

_MATERIALIZED_FIELDS = (
    "canonical_solve_grid_template",
    "canonical_solve_authoritative_measured_only",
    "canonical_solve_authoritative_support_mask",
    "canonical_solve_baseline_background",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def canonical_identity_signature(identity_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path/export-free identity used to compare parent solves.

    The signature intentionally excludes export_aoi, run id, output directories, and
    raster paths. Adjacent AOI exports from the same canonical river system must
    resolve to the same signature and cache key before the detailed river stages run.
    """
    return {key: identity_payload.get(key) for key in _IDENTITY_FIELDS}


def materialized_artifact_hashes(
    *,
    canonical_solve_grid_template_path: Path,
    canonical_solve_authoritative_measured_only_path: Path,
    canonical_solve_authoritative_support_mask_path: Path,
    canonical_solve_baseline_background_path: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        "canonical_solve_grid_template": Path(canonical_solve_grid_template_path),
        "canonical_solve_authoritative_measured_only": Path(canonical_solve_authoritative_measured_only_path),
        "canonical_solve_authoritative_support_mask": Path(canonical_solve_authoritative_support_mask_path),
        "canonical_solve_baseline_background": Path(canonical_solve_baseline_background_path),
    }
    out: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        out[key] = {
            "name": path.name,
            "exists": path.exists(),
            "size": int(path.stat().st_size) if path.exists() else None,
            "sha256": _sha256_file(path),
        }
    return out


def register_canonical_identity_or_raise(
    *,
    registry_path: Path,
    identity_path: Path,
    receipt_path: Path,
    canonical_solve_cache_key: str,
    canonical_solve_grid_template_path: Path,
    canonical_solve_authoritative_measured_only_path: Path,
    canonical_solve_authoritative_support_mask_path: Path,
    canonical_solve_baseline_background_path: Path,
) -> Path:
    identity_payload = _read_json(Path(identity_path))
    signature = canonical_identity_signature(identity_payload)
    signature_hash = _stable_hash(signature)
    artifact_hashes = materialized_artifact_hashes(
        canonical_solve_grid_template_path=canonical_solve_grid_template_path,
        canonical_solve_authoritative_measured_only_path=canonical_solve_authoritative_measured_only_path,
        canonical_solve_authoritative_support_mask_path=canonical_solve_authoritative_support_mask_path,
        canonical_solve_baseline_background_path=canonical_solve_baseline_background_path,
    )
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = _read_json(registry_path)
    else:
        registry = {"schema_version": 1, "identities": {}}
    identities = registry.setdefault("identities", {})
    previous = identities.get(signature_hash)
    current = {
        "identity_signature": signature,
        "identity_signature_hash": signature_hash,
        "canonical_solve_cache_key": str(canonical_solve_cache_key),
        "canonical_solve_identity_path": str(identity_path),
        "materialized_artifact_hashes": artifact_hashes,
    }
    status = "registered"
    if previous is not None:
        previous_key = str(previous.get("canonical_solve_cache_key"))
        previous_hashes = previous.get("materialized_artifact_hashes") or {}
        mismatched = []
        for key in _MATERIALIZED_FIELDS:
            if (previous_hashes.get(key) or {}).get("sha256") != (artifact_hashes.get(key) or {}).get("sha256"):
                mismatched.append(key)
        if previous_key != str(canonical_solve_cache_key):
            if mismatched:
                receipt = {
                    "status": "failed",
                    "failure": "river_workflow_canonical_identity_drift",
                    "reason": "same canonical identity changed cache key and materialized canonical inputs",
                    "mismatched_materialized_artifacts": mismatched,
                    "previous": previous,
                    "current": current,
                }
                Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
                Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                raise RuntimeError(
                    "river_workflow_canonical_identity_drift:"
                    f"signature={signature_hash}:previous_cache_key={previous_key}:"
                    f"current_cache_key={canonical_solve_cache_key}:"
                    f"mismatched_materialized_artifacts={','.join(mismatched)}:receipt={receipt_path}"
                )
            current["previous_canonical_solve_cache_key"] = previous_key
            current["cache_key_rotation_policy"] = (
                "allowed_same_canonical_identity_same_materialized_inputs_new_workflow_contract"
            )
            identities[signature_hash] = current
            registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            status = "cache_key_rotated_for_workflow_contract"
        elif mismatched:
            receipt = {
                "status": "failed",
                "failure": "river_workflow_canonical_identity_drift",
                "reason": "same canonical identity/cache key produced different canonical materialized inputs",
                "mismatched_materialized_artifacts": mismatched,
                "previous": previous,
                "current": current,
            }
            Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
            Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise RuntimeError(
                "river_workflow_canonical_identity_drift:"
                f"signature={signature_hash}:cache_key={canonical_solve_cache_key}:"
                f"mismatched_materialized_artifacts={','.join(mismatched)}:receipt={receipt_path}"
            )
        else:
            status = "matched_existing"
    else:
        identities[signature_hash] = current
        registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "status": status,
        "failure": None,
        "registry_path": str(registry_path),
        "identity_signature_hash": signature_hash,
        "canonical_solve_cache_key": str(canonical_solve_cache_key),
        "identity_signature": signature,
        "materialized_artifact_hashes": artifact_hashes,
    }
    Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Path(receipt_path)


__all__ = [
    "canonical_identity_signature",
    "materialized_artifact_hashes",
    "register_canonical_identity_or_raise",
]
