from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

LOG = logging.getLogger(__name__)


def _read_json(pathlike: str | Path | None) -> dict[str, Any]:
    if not pathlike:
        return {}
    path = Path(pathlike)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOG.debug("river diagnostics: failed reading %s", path, exc_info=True)
        return {}


def _num(payload: dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in payload and payload.get(key) is not None:
            try:
                return float(payload[key])
            except Exception:
                continue
    return None


def _extract_xs_diag(xs_meta: dict[str, Any]) -> dict[str, Any]:
    total = _num(xs_meta, 'n_xs_total', 'xs_total', 'xs_lines', 'n_total')
    valid = _num(xs_meta, 'n_valid_xs', 'xs_valid', 'n_used', 'n_xs_used')
    auth_cal = _num(xs_meta, 'n_authoritative_calibrated_xs', 'authoritative_calibrated_xs', 'n_authoritative_calibrated')
    simple_manning = _num(xs_meta, 'n_q2_regional_xs', 'q2_regional_xs', 'n_simplified_manning_fallback', 'simplified_manning_fallback_xs')
    out = {
        'xs_total': int(total) if total is not None else None,
        'xs_valid': int(valid) if valid is not None else None,
        'authoritative_calibrated_xs': int(auth_cal) if auth_cal is not None else None,
        'simplified_manning_fallback_xs': int(simple_manning) if simple_manning is not None else None,
        'wse_anchor_source': xs_meta.get('wse_anchor_source'),
        'energy_solver_enabled': xs_meta.get('energy_solver_enabled'),
    }
    if total and valid is not None:
        out['xs_valid_fraction'] = float(valid) / float(total) if float(total) > 0 else None
    if valid and auth_cal is not None:
        out['authoritative_calibration_fraction'] = float(auth_cal) / float(valid) if float(valid) > 0 else None
    return out


def _extract_reverticalization_diag(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        'xs_reverticalized': int(receipt.get('xs_count', receipt.get('xs', 0) or 0)) if receipt else None,
        'reverticalized_point_count': int(receipt.get('point_count', 0) or 0) if receipt else None,
        'reverticalization_shift_p95_m': _num(receipt, 'shift_p95_m', 'shift_p95'),
    }


def _extract_channel_surface_diag(contract: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cells = _num(contract, 'populated_cells')
    auth = _num(contract, 'authoritative_cells')
    xs = _num(contract, 'xs_profile_cells')
    bank = _num(contract, 'bank_prior_cells')
    resolved = _num(contract, 'resolved_bed_cells')
    if cells is not None:
        out['channel_surface_populated_cells'] = int(cells)
        denom = max(float(cells), 1.0)
        if auth is not None:
            out['channel_surface_authoritative_fraction'] = float(auth) / denom
        if xs is not None:
            out['channel_surface_xs_profile_fraction'] = float(xs) / denom
        if bank is not None:
            out['channel_surface_bank_prior_fraction'] = float(bank) / denom
        if resolved is not None:
            out['channel_surface_resolved_bed_fraction'] = float(resolved) / denom
    return out


def _extract_stage_support_diag(receipt: dict[str, Any]) -> dict[str, Any]:
    counts = receipt.get('class_counts', {}) if isinstance(receipt.get('class_counts'), dict) else {}
    none = float(counts.get('none', counts.get('0', 0)) or 0)
    weak = float(counts.get('weak_dem_proxy', counts.get('5', 0)) or 0)
    bank = float(counts.get('bank_stage', counts.get('10', 0)) or 0)
    prof = float(counts.get('bank_profile', counts.get('20', 0)) or 0)
    total = none + weak + bank + prof
    out = {
        'skeleton_stage_total_pixels': int(total) if total > 0 else None,
        'skeleton_stage_weak_dem_proxy_fraction': (weak / total) if total > 0 else None,
        'skeleton_stage_bank_stage_fraction': (bank / total) if total > 0 else None,
        'skeleton_stage_bank_profile_fraction': (prof / total) if total > 0 else None,
    }
    return out


def _apply_optional_gates(payload: dict[str, Any], cfg: Any) -> None:
    failures: list[str] = []
    xvf = payload.get('xs_valid_fraction')
    if xvf is not None:
        thr = getattr(cfg, 'river_diag_min_xs_valid_fraction', None)
        if thr is not None and float(xvf) < float(thr):
            failures.append(f'xs_valid_fraction<{thr}')
    rp95 = payload.get('reverticalization_shift_p95_m')
    if rp95 is not None:
        thr = getattr(cfg, 'river_diag_max_reverticalization_p95_m', None)
        if thr is not None and float(rp95) > float(thr):
            failures.append(f'reverticalization_shift_p95_m>{thr}')
    xsfrac = payload.get('channel_surface_xs_profile_fraction')
    if xsfrac is not None:
        thr = getattr(cfg, 'river_diag_max_xs_profile_fraction', None)
        if thr is not None and float(xsfrac) > float(thr):
            failures.append(f'channel_surface_xs_profile_fraction>{thr}')
    authfrac = payload.get('channel_surface_authoritative_fraction')
    if authfrac is not None:
        thr = getattr(cfg, 'river_diag_min_authoritative_fraction', None)
        if thr is not None and float(authfrac) < float(thr):
            failures.append(f'channel_surface_authoritative_fraction<{thr}')
    weakfrac = payload.get('skeleton_stage_weak_dem_proxy_fraction')
    if weakfrac is not None:
        thr = getattr(cfg, 'river_diag_max_weak_stage_fraction', None)
        if thr is not None and float(weakfrac) > float(thr):
            failures.append(f'skeleton_stage_weak_dem_proxy_fraction>{thr}')
    payload['ok'] = len(failures) == 0
    payload['failures'] = failures
    if failures:
        raise RuntimeError('river runtime diagnostics gates failed: ' + ', '.join(failures))


def build_river_runtime_diagnostics(*, river_dir: str | Path, river_outputs: Dict[str, Any], report: Dict[str, Any], cfg: Any, logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    river_dir = Path(river_dir)
    outputs = river_outputs if isinstance(river_outputs, dict) else {}
    run_receipt = _read_json(outputs.get('river_run_receipt'))
    xs_meta_path = run_receipt.get('xs_mainstem_constraint_meta') if isinstance(run_receipt, dict) else None
    xs_meta = _read_json(xs_meta_path)
    revertical = _read_json(outputs.get('xs_reverticalization_receipt'))
    channel_contract = _read_json(outputs.get('channel_surface_contract'))

    stage_receipt_path = None
    skeleton_receipt = _read_json(outputs.get('skeleton_receipt'))
    if isinstance(skeleton_receipt, dict):
        stage_receipt_path = skeleton_receipt.get('stage_support_receipt') or skeleton_receipt.get('outputs', {}).get('stage_support_receipt')
    if not stage_receipt_path and isinstance(run_receipt, dict):
        srec = _read_json(run_receipt.get('skeleton_receipt'))
        if isinstance(srec, dict):
            stage_receipt_path = srec.get('stage_support_receipt') or srec.get('outputs', {}).get('stage_support_receipt')
    stage_receipt = _read_json(stage_receipt_path)

    payload: Dict[str, Any] = {
        'run_id': report.get('run_id'),
        'river_dir': str(river_dir),
    }
    payload.update(_extract_xs_diag(xs_meta))
    payload.update(_extract_reverticalization_diag(revertical))
    payload.update(_extract_channel_surface_diag(channel_contract))
    payload.update(_extract_stage_support_diag(stage_receipt))
    payload['energy_solver_requested'] = bool(getattr(cfg, 'river_enable_1d_energy_solver', False))

    payload['ok'] = True
    payload['failures'] = []
    enforce = bool(getattr(cfg, 'river_diag_enforce', False))
    if enforce:
        _apply_optional_gates(payload, cfg)
    else:
        try:
            _apply_optional_gates(payload, type('Cfg', (), {})())
        except Exception:
            pass
        payload['ok'] = True
        payload['failures'] = []

    out_path = river_dir / 'river_runtime_diagnostics.json'
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    (logger or LOG).info('[RIVER][DIAG] Wrote runtime diagnostics: %s', out_path)
    return {'runtime_diagnostics': str(out_path)}
