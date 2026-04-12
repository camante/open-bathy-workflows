import json
from pathlib import Path

from workflow_run_diagnosis import write_run_diagnosis


def test_write_run_diagnosis_writes_connected_diagnosis_files(tmp_path: Path):
    (tmp_path / 'io_manifest.json').write_text(json.dumps({'inputs': [], 'outputs': []}), encoding='utf-8')
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    bench_dir = tmp_path / 'benchmark'
    bench_dir.mkdir()

    transition = {
        'available': True,
        'candidate_cell_count': 20,
        'nonzero_cell_count': 0,
    }
    backbone = {
        'available': True,
        'candidate_station_count': 5,
        'adjusted_station_count': 0,
        'weight_summary': {'mean': 0.4},
    }
    triage = {
        'available': True,
        'receipts': {
            'centerline_agreement': {'roughness_ratio_p95_abs': 6.5},
            'role_agreement': {'available': False, 'reason': 'missing node targets'},
            'section_target_agreement': {'available': False, 'reason': 'missing target surface'},
        },
    }
    focus = {
        'available': True,
        'hard_problem_holdout_blind': True,
        'hard_problem_blind_reason': 'all holdout points are authoritative_locked',
    }
    paths = {
        'transition': river_dir / 'transition.json',
        'backbone': river_dir / 'backbone.json',
        'triage': bench_dir / 'triage.json',
        'focus': bench_dir / 'focus.json',
    }
    paths['transition'].write_text(json.dumps(transition), encoding='utf-8')
    paths['backbone'].write_text(json.dumps(backbone), encoding='utf-8')
    paths['triage'].write_text(json.dumps(triage), encoding='utf-8')
    paths['focus'].write_text(json.dumps(focus), encoding='utf-8')

    report = {
        'river': {'outputs': {
            'channel_surface_authoritative_transition_summary': str(paths['transition']),
            'backbone_smoothing_summary': str(paths['backbone']),
        }},
        'outputs': {
            'benchmark_river_receipt_triage_summary_json': str(paths['triage']),
            'benchmark_river_primary_focus_summary_json': str(paths['focus']),
        },
        'benchmark': {'status': 'ran', 'requested': True},
    }

    outputs = write_run_diagnosis(out_dir=tmp_path, report=report)
    assert Path(outputs['workflow_accuracy_anomalies_json']).exists()
    assert Path(outputs['workflow_first_bad_artifact_summary_json']).exists()
    assert Path(outputs['workflow_run_diagnosis_summary_json']).exists()
    assert Path(outputs['workflow_run_diagnosis_summary_txt']).exists()

    anomalies = json.loads(Path(outputs['workflow_accuracy_anomalies_json']).read_text(encoding='utf-8'))
    ids = {row['anomaly_id'] for row in anomalies}
    assert 'authoritative_transition_inert' in ids
    assert 'backbone_smoothing_inert' in ids
    assert 'centerline_roughness_high' in ids

    first_bad = json.loads(Path(outputs['workflow_first_bad_artifact_summary_json']).read_text(encoding='utf-8'))
    assert first_bad['available'] is True
    assert first_bad['first_bad_stage'] == 'river'
    assert first_bad['failure_mode'] == 'authoritative_transition_inert'

    summary_txt = Path(outputs['workflow_run_diagnosis_summary_txt']).read_text(encoding='utf-8')
    assert 'FIRST BAD ARTIFACT' in summary_txt
    assert 'authoritative_transition_inert' in summary_txt



def test_run_diagnosis_flags_inert_thalweg_and_section_target_effects(tmp_path: Path):
    (tmp_path / 'io_manifest.json').write_text(json.dumps({'inputs': [], 'outputs': []}), encoding='utf-8')
    river_dir = tmp_path / 'river'
    river_dir.mkdir()

    effect = {
        'available': True,
        'selected_for_thalweg_render_component_count': 3,
        'candidate_pixel_count': 120,
        'final_changed_pixel_count': 0,
        'section_target_geometry_count': 50,
        'section_target_applied_pixel_count': 0,
    }
    effect_path = river_dir / 'effect.json'
    effect_path.write_text(json.dumps(effect), encoding='utf-8')

    report = {
        'river': {'outputs': {
            'channel_surface_effect_summary': str(effect_path),
        }},
        'outputs': {},
        'benchmark': {'status': 'skipped', 'requested': False, 'reason': 'no_holdout_flag'},
    }

    outputs = write_run_diagnosis(out_dir=tmp_path, report=report)
    anomalies = json.loads(Path(outputs['workflow_accuracy_anomalies_json']).read_text(encoding='utf-8'))
    ids = {row['anomaly_id'] for row in anomalies}
    assert 'thalweg_render_inert' in ids
    assert 'section_target_inert' in ids

def test_first_bad_artifact_prefers_earliest_bad_stage_and_stage_trace_anomalies(tmp_path: Path):
    (tmp_path / 'io_manifest.json').write_text(json.dumps({'inputs': [], 'outputs': []}), encoding='utf-8')
    (tmp_path / 'final.tif').write_text('x', encoding='utf-8')

    report = {
        'river': {
            'status': 'failed',
            'reason': 'transition collapse',
            'outputs': {'river_final': str(tmp_path / 'final.tif')},
        },
        'benchmark': {
            'status': 'skipped',
            'requested': False,
            'reason': 'no_holdout_flag',
        },
        'outputs': {},
    }

    outputs = write_run_diagnosis(out_dir=tmp_path, report=report)
    anomalies = json.loads(Path(outputs['workflow_accuracy_anomalies_json']).read_text(encoding='utf-8'))
    ids = {row['anomaly_id'] for row in anomalies}
    assert 'stage_failed::river' in ids

    first_bad = json.loads(Path(outputs['workflow_first_bad_artifact_summary_json']).read_text(encoding='utf-8'))
    assert first_bad['first_bad_stage'] == 'river'
    assert first_bad['failure_mode'] == 'stage_failed::river'
    assert first_bad['recommended_fix_module'] == 'river'


def test_run_diagnosis_flags_missing_and_orphan_stage_outputs(tmp_path: Path):
    (tmp_path / 'io_manifest.json').write_text(json.dumps({'inputs': [], 'outputs': []}), encoding='utf-8')
    existing = tmp_path / 'existing.txt'
    existing.write_text('ok', encoding='utf-8')

    report = {
        'river': {
            'status': 'ran',
            'module': 'river_channel_surface.py',
            'outputs': {
                'missing_receipt': str(tmp_path / 'missing.json'),
                'existing_receipt': str(existing),
            },
        },
        'outputs': {},
    }

    outputs = write_run_diagnosis(out_dir=tmp_path, report=report)
    anomalies = json.loads(Path(outputs['workflow_accuracy_anomalies_json']).read_text(encoding='utf-8'))
    ids = {row['anomaly_id'] for row in anomalies}
    assert 'stage_missing_outputs::river' in ids
    assert 'stage_orphan_outputs::river' in ids

    first_bad = json.loads(Path(outputs['workflow_first_bad_artifact_summary_json']).read_text(encoding='utf-8'))
    assert first_bad['first_bad_stage'] == 'river'
    assert first_bad['failure_mode'] == 'stage_missing_outputs::river'
    assert first_bad['first_bad_artifact'] == str(tmp_path / 'missing.json')



def test_run_diagnosis_counts_success_stage_as_active_branch(tmp_path: Path):
    (tmp_path / 'io_manifest.json').write_text(json.dumps({'inputs': [], 'outputs': []}), encoding='utf-8')
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    final = tmp_path / 'final.txt'
    final.write_text('ok', encoding='utf-8')
    report = {
        'workflow_execution_state': {
            'status': 'ran',
            'outputs': {'execution_receipt': str(final)},
        },
        'river': {
            'status': 'success',
            'outputs': {'river_final': str(final)},
        },
        'outputs': {},
    }
    outputs = write_run_diagnosis(out_dir=tmp_path, report=report)
    summary = json.loads(Path(outputs['workflow_run_diagnosis_summary_json']).read_text(encoding='utf-8'))
    assert summary['primary_active_branch'] == 'river'
    assert 'river' in summary['stages_ran']



def test_run_diagnosis_deprioritizes_terminal_orphan_outputs_as_first_bad(tmp_path: Path):
    (tmp_path / 'io_manifest.json').write_text(json.dumps({'inputs': [], 'outputs': []}), encoding='utf-8')
    terminal = tmp_path / 'terminal.txt'
    terminal.write_text('ok', encoding='utf-8')
    river_dir = tmp_path / 'river'
    river_dir.mkdir()
    transition = river_dir / 'transition.json'
    transition.write_text(json.dumps({'available': True, 'candidate_cell_count': 5, 'nonzero_cell_count': 0}), encoding='utf-8')
    report = {
        'workflow_execution_state': {'status': 'ran', 'outputs': {'state_receipt': str(terminal)}},
        'river': {'status': 'success', 'outputs': {'channel_surface_authoritative_transition_summary': str(transition)}},
        'outputs': {},
    }
    outputs = write_run_diagnosis(out_dir=tmp_path, report=report)
    first_bad = json.loads(Path(outputs['workflow_first_bad_artifact_summary_json']).read_text(encoding='utf-8'))
    assert first_bad['failure_mode'] == 'authoritative_transition_inert'
    assert first_bad['first_bad_stage'] == 'river'
