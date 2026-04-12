from __future__ import annotations

import json
from pathlib import Path

from river_workflow_diagnostics import build_river_runtime_diagnostics


class _Cfg:
    river_enable_1d_energy_solver = False
    river_diag_enforce = False


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding='utf-8')
    return str(path)


def test_runtime_diagnostics_collects_key_metrics(tmp_path: Path):
    river_dir = tmp_path / 'river'
    river_dir.mkdir()

    xs_meta = river_dir / 'xs_meta.json'
    _write_json(xs_meta, {
        'n_xs_total': 200,
        'n_valid_xs': 120,
        'n_authoritative_calibrated_xs': 90,
        'n_q2_regional_xs': 50,
        'wse_anchor_source': 'dem_proxy',
        'energy_solver_enabled': False,
    })
    skeleton_receipt = river_dir / 'skeleton_receipt.json'
    stage_receipt = river_dir / 'stage_support_receipt.json'
    _write_json(stage_receipt, {'class_counts': {'none': 10, 'weak_dem_proxy': 20, 'bank_stage': 30, 'bank_profile': 40}})
    _write_json(skeleton_receipt, {'stage_support_receipt': str(stage_receipt)})
    run_receipt = river_dir / 'river_run_receipt.json'
    _write_json(run_receipt, {'xs_mainstem_constraint_meta': str(xs_meta), 'skeleton_receipt': str(skeleton_receipt)})
    revertical = river_dir / 'river_xs_reverticalization_receipt.json'
    _write_json(revertical, {'xs_count': 100, 'point_count': 7000, 'shift_p95_m': 4.5})
    channel_contract = river_dir / 'river_channel_surface_contract.json'
    _write_json(channel_contract, {'populated_cells': 1000, 'authoritative_cells': 100, 'xs_profile_cells': 800, 'bank_prior_cells': 50, 'resolved_bed_cells': 50})

    outputs = {
        'river_run_receipt': str(run_receipt),
        'xs_reverticalization_receipt': str(revertical),
        'channel_surface_contract': str(channel_contract),
        'skeleton_receipt': str(skeleton_receipt),
    }
    result = build_river_runtime_diagnostics(river_dir=river_dir, river_outputs=outputs, report={'run_id': 'r1'}, cfg=_Cfg())
    diag = json.loads(Path(result['runtime_diagnostics']).read_text(encoding='utf-8'))
    assert diag['xs_total'] == 200
    assert diag['xs_valid'] == 120
    assert abs(diag['xs_valid_fraction'] - 0.6) < 1e-6
    assert abs(diag['channel_surface_xs_profile_fraction'] - 0.8) < 1e-6
    assert abs(diag['channel_surface_authoritative_fraction'] - 0.1) < 1e-6
    assert abs(diag['skeleton_stage_weak_dem_proxy_fraction'] - 0.2) < 1e-6
