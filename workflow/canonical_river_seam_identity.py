from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_canonical_identity(contract_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(contract_path).read_text())
    out: dict[str, Any] = {
        'identity_version': payload.get('identity_version'),
        'canonical_solve_cache_key': payload.get('canonical_solve_cache_key'),
        'solve_bundle_root': payload.get('solve_bundle_root'),
        'solve_outputs_root': payload.get('solve_outputs_root'),
        'canonical_solve_aoi': payload.get('canonical_solve_aoi'),
        'projected_crs': payload.get('projected_crs'),
        'target_resolution_m': payload.get('target_resolution_m'),
        'canonical_network_identity_tag': payload.get('canonical_network_identity_tag'),
        'authoritative_source_tag': payload.get('authoritative_source_tag'),
        'baseline_source_tag': payload.get('baseline_source_tag'),
    }
    for key in (
        'canonical_solve_grid_path',
        'canonical_solve_authoritative_measured_only_path',
        'canonical_solve_authoritative_support_mask_path',
        'canonical_solve_baseline_background_path',
    ):
        value = payload.get(key)
        out[key] = value
        out[f'{key}_sha256'] = sha256_file(Path(value)) if value and Path(value).exists() else None
    return out


def compare_touching_horizontal_border(north_raster: Path, south_raster: Path, *, atol: float = 1e-9) -> dict[str, Any]:
    with rasterio.open(north_raster) as nsrc, rasterio.open(south_raster) as ssrc:
        n = nsrc.read(1)
        s = ssrc.read(1)
        north_edge = n[-1, :]
        south_edge = s[0, :]
        overlap = min(north_edge.shape[0], south_edge.shape[0])
        if overlap <= 0:
            return {'overlap_cols': 0, 'mean_abs_diff': None, 'max_abs_diff': None, 'allclose': False}
        a = north_edge[:overlap].astype(float)
        b = south_edge[:overlap].astype(float)
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            return {'overlap_cols': overlap, 'mean_abs_diff': None, 'max_abs_diff': None, 'allclose': False}
        diff = np.abs(a[mask] - b[mask])
        return {
            'overlap_cols': int(overlap),
            'finite_pairs': int(mask.sum()),
            'mean_abs_diff': float(diff.mean()),
            'max_abs_diff': float(diff.max()),
            'allclose': bool(np.all(diff <= atol)),
        }
