from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def load_river_domain_summary(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def evaluate_river_domain_summary(
    summary: Mapping[str, Any] | None,
    *,
    min_effective_water_corridor_overlap_frac: float = 0.02,
    min_channel_corridor_overlap_frac: float = 0.001,
    min_channel_pixels: int = 1,
) -> Dict[str, Any]:
    payload = dict(summary or {})
    checks: Dict[str, Any] = {}
    failures = []
    warnings = []

    def _num(key: str, default: float = 0.0) -> float:
        try:
            return float(payload.get(key, default) or default)
        except (TypeError, ValueError):
            return float(default)

    def _int(key: str, default: int = 0) -> int:
        try:
            return int(payload.get(key, default) or default)
        except (TypeError, ValueError):
            return int(default)

    corridor_pixels = _int('corridor_pixels', 0)
    channel_pixels = _int('channel_pixels', 0)
    requested = str(payload.get('channel_source_requested') or 'auto')
    effective = str(payload.get('channel_source_effective') or 'unknown')
    effective_water_source = str(payload.get('effective_water_source') or 'unknown')
    harmonized = bool(payload.get('water_mask_harmonization_applied', False))
    water_overlap = _num('effective_water_corridor_overlap_frac', 0.0)
    channel_overlap = _num('channel_corridor_overlap_frac', 0.0)
    nhd_overlap = payload.get('nhdarea_corridor_overlap_frac')
    nhd_effective = bool(payload.get('nhdarea_effective', False))

    if corridor_pixels <= 0:
        checks['corridor_pixels'] = {'ok': False, 'value': corridor_pixels, 'reason': 'no corridor pixels'}
        failures.append({'check': 'corridor_pixels', 'reason': 'no corridor pixels'})
    else:
        checks['corridor_pixels'] = {'ok': True, 'value': corridor_pixels}

    water_ok = corridor_pixels <= 0 or water_overlap >= float(min_effective_water_corridor_overlap_frac)
    checks['effective_water_corridor_overlap'] = {
        'ok': water_ok,
        'value': water_overlap,
        'threshold': float(min_effective_water_corridor_overlap_frac),
    }
    if not water_ok:
        failures.append({'check': 'effective_water_corridor_overlap', 'value': water_overlap, 'threshold': float(min_effective_water_corridor_overlap_frac)})

    channel_pixels_ok = channel_pixels >= int(min_channel_pixels)
    checks['channel_pixels'] = {
        'ok': channel_pixels_ok,
        'value': channel_pixels,
        'threshold': int(min_channel_pixels),
    }
    if not channel_pixels_ok:
        failures.append({'check': 'channel_pixels', 'value': channel_pixels, 'threshold': int(min_channel_pixels)})

    channel_overlap_ok = corridor_pixels <= 0 or channel_overlap >= float(min_channel_corridor_overlap_frac)
    checks['channel_corridor_overlap'] = {
        'ok': channel_overlap_ok,
        'value': channel_overlap,
        'threshold': float(min_channel_corridor_overlap_frac),
    }
    if not channel_overlap_ok:
        failures.append({'check': 'channel_corridor_overlap', 'value': channel_overlap, 'threshold': float(min_channel_corridor_overlap_frac)})

    source_policy_ok = not (requested == 'nhdarea' and effective != 'nhdarea')
    checks['channel_source_policy'] = {
        'ok': source_policy_ok,
        'requested': requested,
        'effective': effective,
    }
    if not source_policy_ok:
        failures.append({'check': 'channel_source_policy', 'requested': requested, 'effective': effective})

    if harmonized:
        warnings.append({'check': 'water_mask_harmonization_applied', 'effective_water_source': effective_water_source})
    if nhd_overlap is not None and not nhd_effective:
        warnings.append({'check': 'nhdarea_not_effective', 'nhdarea_corridor_overlap_frac': nhd_overlap})

    return {
        'ok': len(failures) == 0,
        'checks': checks,
        'warnings': warnings,
        'failures': failures,
        'requested_channel_source': requested,
        'effective_channel_source': effective,
        'effective_water_source': effective_water_source,
        'water_mask_harmonization_applied': harmonized,
    }


__all__ = [
    'load_river_domain_summary',
    'evaluate_river_domain_summary',
]
