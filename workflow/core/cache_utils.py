#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cache_utils.py — Deterministic, exact-match cache fingerprints for the SDB pipeline.

Goal
----
Reuse cached artifacts ONLY when the *exact same* parameter set (and inputs/code, if enabled)
was used to create them.

Core ideas
----------
- Canonical JSON serialization (stable ordering, normalized types)
- Parameter fingerprint: SHA256(canonical_json(params))
- Input fingerprint: fast (size+mtime) by default, optional strict (file SHA256)
- Optional code fingerprint: script name + mtime_ns (or strict hash)
- Artifact cache key: SHA256(canonical_json({stage, params_fp, inputs, code_fp}))[:20]
- Sidecar metadata JSON stored next to outputs (or in out_dir)

This module is dependency-light. rasterio is optional (only used for grid fingerprints).
"""


import logging
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

log = logging.getLogger("cache_utils")

# -----------------------------------------------------------------------------
# Canonical JSON
# -----------------------------------------------------------------------------

_JSONABLE = Union[str, int, float, bool, None, Dict[str, Any], list]

def _normalize_for_json(obj: Any) -> _JSONABLE:
    """Normalize Python objects to JSON-serializable primitives deterministically."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple, set)):
        return [_normalize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        # sort keys for determinism
        return {str(k): _normalize_for_json(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    # Numpy types, dataclasses, etc. -> stable string
    try:
        import numpy as np  # optional
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return _normalize_for_json(obj.tolist())
    except Exception:
        log.debug("ignored", exc_info=True)
    return str(obj)

def canonical_json(obj: Any) -> str:
    """Deterministic JSON string (sorted keys, compact separators)."""
    norm = _normalize_for_json(obj)
    return json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

# -----------------------------------------------------------------------------
# Hash helpers
# -----------------------------------------------------------------------------

def sha256_hex(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def sha1_hex(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()

# -----------------------------------------------------------------------------
# Parameter fingerprinting
# -----------------------------------------------------------------------------

def fingerprint_params(params: Dict[str, Any]) -> str:
    """
    Exact-match parameter fingerprint.
    Any change in values (or keys) yields a new fingerprint.
    """
    return sha256_hex(canonical_json(params))

# -----------------------------------------------------------------------------
# File / input fingerprinting
# -----------------------------------------------------------------------------

def fingerprint_file(path: Union[str, Path], *, strict: bool = False, include_mtime: bool = False, chunk_bytes: int = 4 * 1024 * 1024) -> Dict[str, Any]:
    """
    Fingerprint a file.

    strict=False (default): fast fingerprint based on path and size only
    strict=True           : includes SHA256 content hash (streamed)
    include_mtime=False   : exclude mtime from fingerprint (default, for cache stability)
    include_mtime=True    : include mtime_ns in fingerprint

    Notes:
    - Fast fingerprints (size-only) are usually sufficient if content changes always change size.
    - Strict mode is recommended for publication-grade provenance.
    - mtime is excluded by default because copying files changes mtime but not content.
    """
    p = Path(path)
    st = p.stat()
    out: Dict[str, Any] = {
        "path": str(p),
        "size": int(st.st_size),
    }
    if include_mtime:
        out["mtime_ns"] = int(st.st_mtime_ns)
    if strict:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                b = f.read(chunk_bytes)
                if not b:
                    break
                h.update(b)
        out["sha256"] = h.hexdigest()
    return out

def fingerprint_text(text: str) -> Dict[str, Any]:
    """Fingerprint an in-memory text payload (useful for URLs, STAC query payloads, etc.)."""
    return {"sha256": sha256_hex(text), "len": len(text)}

# -----------------------------------------------------------------------------
# Raster grid fingerprint (optional)
# -----------------------------------------------------------------------------

def fingerprint_raster_grid(raster_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Fingerprint a raster grid definition (CRS, transform, width/height).
    Requires rasterio. If unavailable, raises ImportError.
    """
    import rasterio  # type: ignore

    p = Path(raster_path)
    with rasterio.open(p) as ds:
        tr = ds.transform
        crs = ds.crs.to_string() if ds.crs else None
        return {
            "path": str(p),
            "crs": crs,
            "transform": [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f],
            "width": int(ds.width),
            "height": int(ds.height),
        }

# -----------------------------------------------------------------------------
# Code fingerprinting
# -----------------------------------------------------------------------------

def fingerprint_code(path: Union[str, Path], *, strict: bool = False) -> str:
    """
    Fingerprint code that produced an artifact.

    strict=False: "filename:mtime_ns"
    strict=True : SHA256(file_contents) + ":filename"
    """
    p = Path(path)
    if strict:
        return sha256_hex(p.read_bytes()) + f":{p.name}"
    st = p.stat()
    return f"{p.name}:{int(st.st_mtime_ns)}"

# -----------------------------------------------------------------------------
# Artifact cache keys + metadata
# -----------------------------------------------------------------------------

def artifact_cache_key(
    *,
    stage: str,
    params: Dict[str, Any],
    inputs: Optional[Dict[str, Any]] = None,
    code_fp: Optional[str] = None,
    key_len: int = 20,
) -> str:
    """
    Compute the deterministic cache key for an artifact.

    The key depends on:
      - stage name
      - parameter fingerprint (exact-match)
      - inputs (file fingerprints, URLs, grid fingerprints, etc.)
      - code fingerprint (optional but recommended)

    Returns a short hex string (default 20 chars).
    """
    payload = {
        "stage": stage,
        "params_fp": fingerprint_params(params),
        "inputs": inputs or {},
        "code_fp": code_fp or "",
    }
    return sha256_hex(canonical_json(payload))[: int(key_len)]

def meta_payload(
    *,
    cache_key: str,
    stage: str,
    params: Dict[str, Any],
    inputs: Optional[Dict[str, Any]] = None,
    code_fp: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the metadata payload written to each cache entry."""
    from datetime import datetime, timezone
    d: Dict[str, Any] = {
        "cache_key": cache_key,
        "stage": stage,
        "params": _normalize_for_json(params),
        "inputs": _normalize_for_json(inputs or {}),
        "code_fingerprint": code_fp or "",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        d["extra"] = _normalize_for_json(extra)
    return d

def read_meta(meta_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read metadata JSON; return None if missing or invalid."""
    p = Path(meta_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        log.debug("read_meta: suppressed exception", exc_info=True)
        return None

def write_meta(meta_path: Union[str, Path], payload: Dict[str, Any]) -> Path:
    """Write metadata JSON (pretty, sorted)."""
    p = Path(meta_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p

def cache_hit(
    meta_path: Union[str, Path],
    *,
    expected_key: str,
    require_exact_params: bool = True,
    expected_params: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Determine whether an artifact cache hit is valid.

    Returns (hit, reason).

    - Always checks cache_key match.
    - If require_exact_params=True and expected_params is provided, also checks
      canonical_json(meta["params"]) == canonical_json(expected_params).
      This provides an additional safeguard if you ever change how cache_key is computed.
    """
    meta = read_meta(meta_path)
    if not meta:
        return (False, "no_meta")
    have_key = str(meta.get("cache_key", ""))
    if have_key != expected_key:
        return (False, "key_mismatch")
    if require_exact_params and expected_params is not None:
        have_params = meta.get("params", {})
        if canonical_json(have_params) != canonical_json(expected_params):
            return (False, "params_mismatch")
    return (True, "hit")

# -----------------------------------------------------------------------------
# Convenience: build + validate in one go
# -----------------------------------------------------------------------------

@dataclass
class ArtifactCache:
    """
    Small helper to standardize cache checks.

    Example:
        ac = ArtifactCache(stage="s2", meta_path=out_dir/"S2_COMPOSITE_META.json",
                           params=params, inputs=inputs, code_fp=code_fp)
        hit, reason = ac.is_hit()
        if hit: ...
        else: rebuild; ac.write(extra={...})
    """
    stage: str
    meta_path: Path
    params: Dict[str, Any]
    inputs: Dict[str, Any]
    code_fp: str = ""
    key_len: int = 20

    def expected_key(self) -> str:
        return artifact_cache_key(
            stage=self.stage,
            params=self.params,
            inputs=self.inputs,
            code_fp=self.code_fp,
            key_len=self.key_len,
        )

    def is_hit(self, *, require_exact_params: bool = True) -> Tuple[bool, str]:
        return cache_hit(
            self.meta_path,
            expected_key=self.expected_key(),
            require_exact_params=require_exact_params,
            expected_params=self.params if require_exact_params else None,
        )

    def write(self, *, extra: Optional[Dict[str, Any]] = None) -> Path:
        payload = meta_payload(
            cache_key=self.expected_key(),
            stage=self.stage,
            params=self.params,
            inputs=self.inputs,
            code_fp=self.code_fp,
            extra=extra,
        )
        return write_meta(self.meta_path, payload)
