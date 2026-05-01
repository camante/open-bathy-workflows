"""Canonical river-solve cache handoff helpers.

This module intentionally treats the cache as an implementation detail.  It only
records and reads explicit canonical-solve manifests; it never constructs or
modifies river products.  AOI export code may use these manifests to decide that
an existing canonical parent can be subset directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import hashlib


def _sha256_file(path: Path | str | None) -> str | None:
    if path in (None, ""):
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_candidates(bundle: Any) -> list[Path]:
    candidates: list[Path] = []
    for attr in (
        "canonical_solve_cache_manifest_path",
        "canonical_solution_manifest_path",
    ):
        p = _as_path(getattr(bundle, attr, None))
        if p is not None:
            candidates.append(p)

    root = _as_path(getattr(bundle, "canonical_solve_outputs_root", None))
    if root is not None:
        candidates.extend([
            root / "canonical_river_solution_manifest.json",
            root / "canonical_solve_cache_manifest.json",
        ])

    # De-duplicate while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _payload_has_existing_parent(payload: Mapping[str, Any]) -> bool:
    for key in ("canonical_parent_dem_path", "canonical_final_dem_path", "canonical_solve_final_dem_path"):
        value = payload.get(key)
        if value not in (None, "") and Path(str(value)).is_file():
            return True
    parent = payload.get("canonical_parent_dem")
    if isinstance(parent, Mapping):
        value = parent.get("path")
        if value not in (None, "") and Path(str(value)).is_file():
            return True
    return False


def load_canonical_solve_cache(bundle: Any) -> dict[str, Any] | None:
    """Load a cache manifest only when it points to an existing parent product.

    Returning ``None`` means the caller must run canonical construction.  The
    function does not attempt source discovery outside the explicit bundle
    manifest paths, because hidden discovery is exactly the kind of glue that
    makes AOI/export behavior hard to debug.
    """
    if bundle is None:
        return None
    for manifest_path in _manifest_candidates(bundle):
        if not manifest_path.is_file():
            continue
        payload = _read_json(manifest_path)
        if not payload:
            continue
        if not _payload_has_existing_parent(payload):
            continue
        out = dict(payload)
        out.setdefault("manifest_path", str(manifest_path))
        out.setdefault("cache_manifest_path", str(manifest_path))
        cache_key = (
            out.get("canonical_solve_cache_key")
            or out.get("canonical_cache_key")
            or getattr(bundle, "canonical_solve_cache_key", None)
        )
        if cache_key not in (None, ""):
            out["canonical_solve_cache_key"] = str(cache_key)
            out.setdefault("canonical_cache_key", str(cache_key))
        return out
    return None


def write_canonical_solve_cache(*, bundle: Any, **stage_results: Any) -> Path | None:
    """Write a lightweight canonical-stage cache receipt.

    This is not the public AOI handoff manifest.  The full
    ``canonical_river_solution_manifest.json`` is written later when the parent
    DEM exists.  This receipt is useful for traceability during a canonical build
    and is deliberately ignored by ``load_canonical_solve_cache`` until a real
    parent DEM manifest exists.
    """
    if bundle is None:
        return None
    manifest_path = _as_path(getattr(bundle, "canonical_solve_cache_manifest_path", None))
    if manifest_path is None:
        root = _as_path(getattr(bundle, "canonical_solve_outputs_root", None))
        if root is None:
            return None
        manifest_path = root / "canonical_solve_cache_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _result_path(obj: Any, *names: str) -> str | None:
        if obj is None:
            return None
        for name in names:
            value = getattr(obj, name, None)
            if value not in (None, ""):
                return str(value)
        return None

    artifacts = {
        "centerline_points": _result_path(stage_results.get("centerline_result"), "centerline_points_path"),
        "centerline_wse_proxy": _result_path(stage_results.get("wse_result"), "centerline_wse_proxy_points_path"),
        "centerline_authoritative_bed": _result_path(stage_results.get("authoritative_bed_result"), "centerline_authoritative_bed_points_path"),
        "centerline_observed_offset": _result_path(stage_results.get("observed_offset_result"), "centerline_observed_offset_points_path"),
        "centerline_modeled_offset": _result_path(stage_results.get("modeled_offset_result"), "centerline_modeled_offset_points_path"),
        "centerline_bed_backbone": _result_path(stage_results.get("backbone_result"), "centerline_bed_backbone_points_path"),
        "river_corridor_solve": _result_path(stage_results.get("corridor_result"), "river_corridor_solve_path"),
        "river_primary_surface_solve": _result_path(stage_results.get("surface_result"), "river_primary_surface_solve_path"),
        "river_primary_surface_solve_locked": _result_path(stage_results.get("lock_result"), "river_primary_surface_solve_locked_path"),
    }
    stage_identity_artifacts = {
        key: {"path": value, "sha256": _sha256_file(value), "exists": Path(value).is_file() if value else False}
        for key, value in artifacts.items()
        if value not in (None, "")
    }
    payload = {
        "schema_version": 1,
        "role": "canonical_solve_stage_cache_receipt",
        "cache_is_implementation_detail": True,
        "canonical_solve_cache_key": str(getattr(bundle, "canonical_solve_cache_key", "") or "") or None,
        "canonical_system_id": str(getattr(bundle, "canonical_system_id", "") or "") or None,
        "canonical_solve_aoi": str(getattr(bundle, "canonical_solve_aoi", "") or "") or None,
        "canonical_parent_dem_path": str(getattr(bundle, "canonical_solve_final_dem_path", "") or "") or None,
        "artifacts": artifacts,
        "stage_identity_artifacts": stage_identity_artifacts,
        "load_policy": "ignored_until_canonical_parent_manifest_exists",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


__all__ = ["load_canonical_solve_cache", "write_canonical_solve_cache"]
