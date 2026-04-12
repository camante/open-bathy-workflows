from pathlib import Path

import numpy as np
import pandas as pd

from river_effectiveness_receipts import write_effectiveness_receipts, write_science_effect_summary


def _nodes() -> pd.DataFrame:
    rows = []
    for station, support_class, template, channel_protected, inner_rebuildable in [
        (0.0, 'measured_xs_partial', 'hybrid_xs', True, False),
        (10.0, 'bank_only_low_confidence', 'generic_symmetric', False, True),
        (20.0, 'supported_transition', 'hybrid_longitudinal', False, True),
    ]:
        for role in ['left_bank', 'left_inner', 'thalweg', 'right_inner', 'right_bank']:
            applied = station == 10.0 and role in {'left_inner', 'thalweg', 'right_inner'}
            rows.append({
                'component_id': 'main',
                'station_m': station,
                'node_role': role,
                'xs_support_template_class': support_class,
                'xs_template_type': template,
                'station_provenance_class': 'true_measured' if station == 0.0 else ('indirect_xs' if station == 20.0 else 'bank_only'),
                'station_true_measured_xs_fraction': 0.5 if station == 0.0 else 0.0,
                'station_indirect_xs_fraction': 0.4 if station == 20.0 else 0.0,
                'station_residual_xs_fraction': 0.0,
                'station_authoritative_fraction': 0.2 if role in {'left_bank', 'right_bank'} else 0.0,
                'station_authoritative_channel_fraction': 0.35 if station == 0.0 else 0.0,
                'station_authoritative_bank_fraction': 1.0 if role in {'left_bank', 'right_bank'} else 0.0,
                'station_bank_protected': role in {'left_bank', 'right_bank'},
                'station_channel_protected': channel_protected,
                'station_inner_rebuildable': inner_rebuildable,
                'primary_surface_rebuild_applied': applied,
                'primary_surface_rebuild_delta_m': -0.8 if applied else 0.0,
                'primary_surface_rebuild_weight': 0.65 if applied else 0.0,
            })
    return pd.DataFrame(rows)


def test_effectiveness_receipts_capture_weak_support_changes(tmp_path: Path):
    nodes = _nodes()
    outputs, summary = write_effectiveness_receipts(nodes, river_dir=tmp_path)
    assert summary['available'] is True
    assert summary['weak_support_station_count'] == 2
    assert summary['weak_support_rebuild_eligible_station_count'] == 2
    assert summary['weak_support_changed_station_count'] == 1
    assert summary['changed_channel_node_count'] == 3
    assert summary['rebuild_suppression_reason_counts']['true_measured_xs'] >= 1
    station_csv = Path(outputs['river_effectiveness_station_summary'])
    receipt_json = Path(outputs['river_effectiveness_receipt'])
    assert station_csv.exists()
    assert receipt_json.exists()
    station_df = pd.read_csv(station_csv)
    changed = station_df.loc[np.isclose(station_df['station_m'].to_numpy(dtype=float), 10.0)]
    assert bool(changed['station_changed'].iloc[0])
    assert changed['rebuild_eligibility_reason'].iloc[0] == 'eligible'


def test_science_effect_summary_rolls_up_receipts(tmp_path: Path):
    nodes = _nodes()
    _, eff_summary = write_effectiveness_receipts(nodes, river_dir=tmp_path)
    outputs, summary = write_science_effect_summary(
        river_dir=tmp_path,
        effectiveness_summary=eff_summary,
        primary_surface_rebuild_summary={'adjusted_node_count': 3, 'changed_channel_core_node_count': 2, 'rebuild_mode_counts': {'thalweg_only_rebuild': 1}},
        xs_realism_summary={'adjusted_node_count': 4, 'template_type_counts': {'generic_symmetric': 1}},
        tendency_summary={'adjusted_node_count': 5, 'support_regime_counts': {'longitudinal_weak_support': 2}},
        render_mode_counts={'full_scaffold': 1},
        render_reason_counts={'weak_support': 1},
        fast_render_component_ids=[],
        measured_xs_frame_receipt={'stations_with_true_measured_xs_qualified': 2, 'stations_with_authoritative_xs_inner_support': 3, 'stations_with_authoritative_bank_only_support': 1, 'measured_xs_activation_from_bank_only_forbidden': True},
        measured_xs_scaffold_receipt={'nodes_marked_true_measured_xs_post_filter': 4, 'measured_xs_activation_contract_ok': True, 'measured_xs_activation_from_bank_only_forbidden': True, 'qualified_station_without_measured_nodes_count': 0},
    )
    assert summary['available'] is True
    assert summary['science_engaged'] is True
    assert summary['weak_support_station_count'] == 2
    assert summary['primary_surface_rebuild_changed_channel_core_node_count'] == 2
    assert summary['measured_xs_post_filter_node_count'] == 4
    assert summary['measured_xs_qualified_station_count'] == 2
    assert summary['measured_xs_activation_contract_ok'] is True
    path = Path(outputs['river_science_effect_summary'])
    assert path.exists()
