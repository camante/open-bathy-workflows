from __future__ import annotations

import json
from pathlib import Path

from workflow_reports_hub import write_reports_hub, _reconcile_manifest_with_reports_dir



def test_write_reports_hub_copies_key_outputs(tmp_path: Path):
    out_dir = tmp_path / 'run'
    out_dir.mkdir()
    (out_dir / 'bathy_report.json').write_text('{}', encoding='utf-8')
    (out_dir / 'io_manifest.json').write_text('{}', encoding='utf-8')
    diag = out_dir / 'reports'
    diag.mkdir()
    (diag / 'WORKFLOW_INPUT_OUTPUT_TRACE.txt').write_text('trace', encoding='utf-8')
    (diag / 'workflow_run_diagnosis_summary.json').write_text('{}', encoding='utf-8')
    (diag / 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt').write_text('summary', encoding='utf-8')
    active_eval = out_dir / 'benchmark_river_active_evaluation_summary.json'
    active_eval.write_text('{}', encoding='utf-8')
    mode = out_dir / 'benchmark_river_mode_summary.json'
    mode.write_text('{}', encoding='utf-8')
    withheld = out_dir / 'benchmark_river_withheld_support_receipt.json'
    withheld.write_text('{}', encoding='utf-8')
    primary_summary = out_dir / 'river_primary_guidance_summary.json'
    primary_summary.write_text(json.dumps({'active_product_name': 'river_primary_surface'}), encoding='utf-8')
    primary_contract = out_dir / 'river_primary_surface_contract.json'
    primary_contract.write_text(json.dumps({'ok': True, 'failures': []}), encoding='utf-8')
    outputs_receipt = out_dir / 'final_route_outputs_receipt.json'
    outputs_receipt.write_text(json.dumps({'artifact_roles': {'conditioned_depth': 'primary', 'river_primary_guidance_summary': 'diagnostic_only'}}), encoding='utf-8')
    river = out_dir / 'river_backbone_smoothing_summary.json'
    river.write_text('{}', encoding='utf-8')
    primary_summary = out_dir / 'river_primary_guidance_summary.json'
    primary_summary.write_text('{}', encoding='utf-8')
    primary_contract = out_dir / 'river_primary_surface_contract.json'
    primary_contract.write_text(json.dumps({'ok': True, 'failures': []}), encoding='utf-8')
    outputs_receipt = out_dir / 'final_route_outputs_receipt.json'
    outputs_receipt.write_text(json.dumps({'artifact_roles': {'conditioned_depth': 'primary', 'river_primary_guidance_summary': 'diagnostic_only'}}), encoding='utf-8')
    report = {
        'outputs': {
            'workflow_input_output_trace': str(diag / 'WORKFLOW_INPUT_OUTPUT_TRACE.txt'),
            'workflow_run_diagnosis_summary_json': str(diag / 'workflow_run_diagnosis_summary.json'),
            'workflow_run_diagnosis_summary_txt': str(diag / 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt'),
            'benchmark_river_active_evaluation_summary_json': str(active_eval),
            'benchmark_river_mode_summary_json': str(mode),
            'benchmark_river_withheld_support_receipt_json': str(withheld),
            'river_primary_guidance_summary': str(primary_summary),
            'river_primary_surface_contract': str(primary_contract),
        },
        'authoritative_base': {'outputs': {'river_primary_guidance_summary': str(primary_summary), 'river_primary_surface_contract': str(primary_contract), 'final_route_receipt': str(out_dir / 'final_route_receipt.json')}},
        'final_dem_route': {'stage_receipts': {'outputs': str(outputs_receipt)}},
        'river': {'outputs': {'backbone_smoothing_summary': str(river)}}
    }
    outputs = write_reports_hub(out_dir=out_dir, report=report)
    assert Path(outputs['reports_readme']).exists()
    manifest = json.loads(Path(outputs['reports_manifest']).read_text(encoding='utf-8'))
    labels = {x['label'] for x in manifest['copied']}
    assert 'workflow_input_output_trace' in labels
    assert 'benchmark_river_active_evaluation_summary_json' in labels
    assert 'benchmark_river_mode_summary_json' in labels
    assert 'benchmark_river_withheld_support_receipt_json' in labels
    assert 'backbone_smoothing_summary' in labels
    assert 'river_primary_guidance_summary' in labels
    assert 'river_primary_surface_contract' in labels
    assert 'river_active_runtime_summary' in labels
    assert 'river_active_shape_summary' in labels
    assert 'river_active_summary' in labels
    priority = {item['file']: item for item in manifest['priority_status']}
    assert priority['benchmark_river_active_evaluation_summary.json']['present'] is True
    assert priority['benchmark_river_active_evaluation_summary.json']['present'] is True
    assert priority['benchmark_river_mode_summary.json']['present'] is True
    assert priority['benchmark_river_withheld_support_receipt.json']['present'] is True
    assert priority['river_active_summary.json']['present'] is True
    assert priority['river_active_runtime_summary.json']['present'] is True
    assert priority['river_primary_guidance_summary.json']['present'] is True
    assert priority['river_primary_surface_contract.json']['present'] is True
    assert priority['river_active_shape_summary.json']['present'] is True



def test_write_reports_hub_copies_unavailable_river_receipts(tmp_path: Path):
    out_dir = tmp_path / 'run'
    out_dir.mkdir()
    (out_dir / 'bathy_report.json').write_text('{}', encoding='utf-8')
    (out_dir / 'io_manifest.json').write_text('{}', encoding='utf-8')
    reports_dir = out_dir / 'reports'
    reports_dir.mkdir()
    (reports_dir / 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt').write_text('summary', encoding='utf-8')
    (reports_dir / 'workflow_run_diagnosis_summary.json').write_text('{}', encoding='utf-8')
    role = out_dir / 'river_role_unavailable.json'
    role.write_text(json.dumps({'available': False, 'reason': 'no_finite_target_comparisons'}), encoding='utf-8')
    section = out_dir / 'river_section_unavailable.json'
    section.write_text(json.dumps({'available': False, 'reason': 'missing_geometry'}), encoding='utf-8')
    width = out_dir / 'river_width_unavailable.json'
    width.write_text(json.dumps({'available': False, 'reason': 'no_backbone_led_inner_nodes'}), encoding='utf-8')
    report = {
        'outputs': {
            'workflow_run_diagnosis_summary_json': str(reports_dir / 'workflow_run_diagnosis_summary.json'),
            'workflow_run_diagnosis_summary_txt': str(reports_dir / 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt'),
        },
        'river': {'outputs': {
            'channel_surface_role_agreement_summary': str(role),
            'channel_surface_section_target_agreement_summary': str(section),
            'centerline_width_propagation_summary': str(width),
        }},
    }
    write_reports_hub(out_dir=out_dir, report=report)
    assert (out_dir / 'reports' / 'river_role_agreement_summary.json').exists()
    assert (out_dir / 'reports' / 'river_section_target_agreement_summary.json').exists()
    assert (out_dir / 'reports' / 'river_centerline_width_propagation_summary.json').exists()





def test_write_reports_hub_builds_primary_river_runtime_summary(tmp_path: Path):
    out_dir = tmp_path / 'run'
    out_dir.mkdir()
    (out_dir / 'bathy_report.json').write_text('{}', encoding='utf-8')
    (out_dir / 'io_manifest.json').write_text('{}', encoding='utf-8')
    reports_dir = out_dir / 'reports'
    reports_dir.mkdir()
    (reports_dir / 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt').write_text('summary', encoding='utf-8')
    (reports_dir / 'workflow_run_diagnosis_summary.json').write_text('{}', encoding='utf-8')
    primary_summary = out_dir / 'river_primary_guidance_summary.json'
    primary_summary.write_text(json.dumps({'active_product_name': 'river_primary_surface', 'primary_builder_mode': 'channel_surface_scaffold', 'primary_surface_contract_ok': True, 'degraded_mode_active': False, 'continuity_safeguard_used': False}), encoding='utf-8')
    primary_contract = out_dir / 'river_primary_surface_contract.json'
    primary_contract.write_text(json.dumps({'ok': True, 'failures': []}), encoding='utf-8')
    outputs_receipt = out_dir / 'final_route_outputs_receipt.json'
    outputs_receipt.write_text(json.dumps({'artifact_roles': {'conditioned_depth': 'primary', 'river_primary_guidance_summary': 'diagnostic_only'}}), encoding='utf-8')
    final_route_receipt = out_dir / 'final_route_receipt.json'
    final_route_receipt.write_text(json.dumps({'artifact_roles': {'final_depth': 'primary', 'river_primary_surface': 'diagnostic_only'}}), encoding='utf-8')
    report = {
        'outputs': {
            'workflow_run_diagnosis_summary_json': str(reports_dir / 'workflow_run_diagnosis_summary.json'),
            'workflow_run_diagnosis_summary_txt': str(reports_dir / 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt'),
            'river_primary_guidance_summary': str(primary_summary),
            'river_primary_surface_contract': str(primary_contract),
        },
        'authoritative_base': {'outputs': {'final_route_receipt': str(final_route_receipt)}},
        'final_dem_route': {'stage_receipts': {'outputs': str(outputs_receipt)}},
        'river': {'outputs': {}},
    }
    write_reports_hub(out_dir=out_dir, report=report)
    runtime_summary = json.loads((out_dir / 'reports' / 'river_active_runtime_summary.json').read_text(encoding='utf-8'))
    assert runtime_summary['active_product_name'] == 'river_primary_surface'
    assert runtime_summary['primary_surface_contract_ok'] is True
    assert runtime_summary['artifact_role'] == 'diagnostic_only'
    assert runtime_summary['dominant_runtime_issue'] == 'river_runtime_contract_ok'
    assert 'conditioned_depth' in runtime_summary['primary_artifact_role_labels']
    assert 'river_primary_guidance_summary' in runtime_summary['diagnostic_artifact_role_labels']

def test_write_reports_hub_readme_and_manifest_surface_priority_status(tmp_path: Path):
    out_dir = tmp_path / 'run'
    out_dir.mkdir()
    (out_dir / 'bathy_report.json').write_text('{}', encoding='utf-8')
    (out_dir / 'io_manifest.json').write_text('{}', encoding='utf-8')
    reports_dir = out_dir / 'reports'
    reports_dir.mkdir()
    (reports_dir / 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt').write_text('summary', encoding='utf-8')
    (reports_dir / 'workflow_run_diagnosis_summary.json').write_text('{}', encoding='utf-8')
    triage = out_dir / 'benchmark_river_receipt_triage_summary.json'
    triage.write_text('{}', encoding='utf-8')
    focus = out_dir / 'benchmark_river_primary_focus_summary.json'
    focus.write_text('{}', encoding='utf-8')
    active_eval = out_dir / 'benchmark_river_active_evaluation_summary.json'
    active_eval.write_text('{}', encoding='utf-8')
    mode = out_dir / 'benchmark_river_mode_summary.json'
    mode.write_text('{}', encoding='utf-8')
    withheld = out_dir / 'benchmark_river_withheld_support_receipt.json'
    withheld.write_text('{}', encoding='utf-8')
    primary_summary = out_dir / 'river_primary_guidance_summary.json'
    primary_summary.write_text(json.dumps({'active_product_name': 'river_primary_surface'}), encoding='utf-8')
    primary_contract = out_dir / 'river_primary_surface_contract.json'
    primary_contract.write_text(json.dumps({'ok': True, 'failures': []}), encoding='utf-8')
    outputs_receipt = out_dir / 'final_route_outputs_receipt.json'
    outputs_receipt.write_text(json.dumps({'artifact_roles': {'conditioned_depth': 'primary', 'river_primary_guidance_summary': 'diagnostic_only'}}), encoding='utf-8')
    report = {
        'outputs': {
            'workflow_run_diagnosis_summary_json': str(reports_dir / 'workflow_run_diagnosis_summary.json'),
            'workflow_run_diagnosis_summary_txt': str(reports_dir / 'WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt'),
            'benchmark_river_receipt_triage_summary_json': str(triage),
            'benchmark_river_primary_focus_summary_json': str(focus),
            'benchmark_river_active_evaluation_summary_json': str(active_eval),
            'benchmark_river_mode_summary_json': str(mode),
            'benchmark_river_withheld_support_receipt_json': str(withheld),
            'river_primary_guidance_summary': str(primary_summary),
            'river_primary_surface_contract': str(primary_contract),
        },
        'authoritative_base': {'outputs': {'river_primary_guidance_summary': str(primary_summary), 'river_primary_surface_contract': str(primary_contract)}},
        'final_dem_route': {'stage_receipts': {'outputs': str(outputs_receipt)}},
        'river': {'outputs': {}},
    }
    outputs = write_reports_hub(out_dir=out_dir, report=report)
    manifest = json.loads(Path(outputs['reports_manifest']).read_text(encoding='utf-8'))
    readme = Path(outputs['reports_readme']).read_text(encoding='utf-8')

    priority = {item['file']: item for item in manifest['priority_status']}
    assert priority['benchmark_river_active_evaluation_summary.json']['present'] is True
    assert priority['benchmark_river_mode_summary.json']['present'] is True
    assert priority['benchmark_river_withheld_support_receipt.json']['present'] is True
    assert priority['river_active_summary.json']['present'] is True
    assert priority['river_active_runtime_summary.json']['present'] is True
    assert priority['river_backbone_smoothing_summary.json']['present'] is False
    assert priority['river_backbone_smoothing_summary.json']['reason'] == 'not_reported'
    assert 'benchmark_river_active_evaluation_summary.json' in readme
    assert 'benchmark_river_mode_summary.json' in readme
    assert 'river_active_shape_summary.json' in readme
    assert 'river_centerline_width_propagation_summary.json' in readme
    assert 'Missing priority files:' in readme


def test_reconcile_manifest_with_reports_dir_drops_missing_copied_dest(tmp_path: Path):
    reports_dir = tmp_path / 'reports'
    reports_dir.mkdir()
    manifest = {
        'reports_dir': str(reports_dir),
        'copied': [
            {'label': 'river_backbone_smoothing_summary.json', 'source': 'src.json', 'dest': str(reports_dir / 'river_backbone_smoothing_summary.json'), 'action': 'copied'},
        ],
        'missing': [],
    }
    reconciled = _reconcile_manifest_with_reports_dir(manifest=manifest)
    assert reconciled['copied'] == []
    assert reconciled['missing'][0]['reason'] == 'manifest_dest_missing'


def test_reports_hub_manifest_includes_artifact_roles(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "bathy_report.json").write_text("{}", encoding="utf-8")
    (out_dir / "io_manifest.json").write_text("{}", encoding="utf-8")
    reports_dir = out_dir / "reports"
    reports_dir.mkdir()
    (reports_dir / "workflow_run_diagnosis_summary.json").write_text("{}", encoding="utf-8")
    (reports_dir / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt").write_text("summary", encoding="utf-8")
    outputs = write_reports_hub(out_dir=out_dir, report={"outputs": {"workflow_run_diagnosis_summary_json": str(reports_dir / "workflow_run_diagnosis_summary.json"), "workflow_run_diagnosis_summary_txt": str(reports_dir / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt")}})
    manifest = json.loads(Path(outputs["reports_manifest"]).read_text(encoding="utf-8"))
    copied = {entry["label"]: entry for entry in manifest["copied"]}
    assert copied["workflow_run_diagnosis_summary_json"]["artifact_role"] == "primary"


def test_reports_hub_demotes_benchmark_drilldowns_to_diagnostic_only(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "bathy_report.json").write_text("{}", encoding="utf-8")
    (out_dir / "io_manifest.json").write_text("{}", encoding="utf-8")
    reports_dir = out_dir / "reports"
    reports_dir.mkdir()
    (reports_dir / "workflow_run_diagnosis_summary.json").write_text("{}", encoding="utf-8")
    (reports_dir / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt").write_text("summary", encoding="utf-8")
    active_eval = out_dir / "benchmark_river_active_evaluation_summary.json"
    active_eval.write_text("{}", encoding="utf-8")
    mode = out_dir / "benchmark_river_mode_summary.json"
    mode.write_text("{}", encoding="utf-8")
    triage = out_dir / "benchmark_river_receipt_triage_summary.json"
    triage.write_text("{}", encoding="utf-8")
    focus = out_dir / "benchmark_river_primary_focus_summary.json"
    focus.write_text("{}", encoding="utf-8")
    outputs = write_reports_hub(out_dir=out_dir, report={"outputs": {
        "workflow_run_diagnosis_summary_json": str(reports_dir / "workflow_run_diagnosis_summary.json"),
        "workflow_run_diagnosis_summary_txt": str(reports_dir / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt"),
        "benchmark_river_active_evaluation_summary_json": str(active_eval),
        "benchmark_river_mode_summary_json": str(mode),
        "benchmark_river_receipt_triage_summary_json": str(triage),
        "benchmark_river_primary_focus_summary_json": str(focus),
    }})
    manifest = json.loads(Path(outputs["reports_manifest"]).read_text(encoding="utf-8"))
    copied = {entry["label"]: entry for entry in manifest["copied"]}
    assert copied["benchmark_river_active_evaluation_summary_json"]["artifact_role"] == "diagnostic_only"
    assert copied["benchmark_river_mode_summary_json"]["artifact_role"] == "diagnostic_only"
    assert copied["benchmark_river_receipt_triage_summary_json"]["artifact_role"] == "diagnostic_only"
    assert copied["benchmark_river_primary_focus_summary_json"]["artifact_role"] == "diagnostic_only"


def test_reports_hub_generates_primary_river_shape_summary(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "bathy_report.json").write_text("{}", encoding="utf-8")
    (out_dir / "io_manifest.json").write_text("{}", encoding="utf-8")
    reports_dir = out_dir / "reports"
    reports_dir.mkdir()
    (reports_dir / "workflow_run_diagnosis_summary.json").write_text("{}", encoding="utf-8")
    (reports_dir / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt").write_text("summary", encoding="utf-8")
    backbone = out_dir / "river_backbone_smoothing_summary.json"
    backbone.write_text(json.dumps({"available": True, "adjusted_station_count": 0, "reason": "prepared"}), encoding="utf-8")
    width = out_dir / "river_centerline_width_propagation_summary.json"
    width.write_text(json.dumps({"available": True, "backbone_led_station_count": 4, "bank_margin_damping_station_count": 2}), encoding="utf-8")
    role = out_dir / "river_role_agreement_summary.json"
    role.write_text(json.dumps({"available": True, "reason": "prepared", "weakest_role": "bank_edge", "lateral_failure_mode": "bank_vs_inner_shape"}), encoding="utf-8")
    section = out_dir / "river_section_target_agreement_summary.json"
    section.write_text(json.dumps({"available": True, "reason": "prepared", "weakest_role_class": "bank_edge"}), encoding="utf-8")
    outputs = write_reports_hub(out_dir=out_dir, report={"outputs": {
        "workflow_run_diagnosis_summary_json": str(reports_dir / "workflow_run_diagnosis_summary.json"),
        "workflow_run_diagnosis_summary_txt": str(reports_dir / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt"),
    }, "river": {"outputs": {
        "backbone_smoothing_summary": str(backbone),
        "centerline_width_propagation_summary": str(width),
        "channel_surface_role_agreement_summary": str(role),
        "channel_surface_section_target_agreement_summary": str(section),
    }}})
    manifest = json.loads(Path(outputs["reports_manifest"]).read_text(encoding="utf-8"))
    copied = {entry["label"]: entry for entry in manifest["copied"]}
    assert copied["river_active_shape_summary"]["artifact_role"] == "diagnostic_only"
    assert copied["backbone_smoothing_summary"]["artifact_role"] == "diagnostic_only"
    summary = json.loads((out_dir / "reports" / "river_active_shape_summary.json").read_text(encoding="utf-8"))
    assert summary["artifact_role"] == "diagnostic_only"
    assert summary["dominant_remaining_issue"] == "backbone_smoothing_inert"
    assert summary["suggested_next_action"] == "strengthen_backbone_smoothing"
    assert summary["diagnostic_receipt_roles"]["river_backbone_smoothing_summary.json"] == "diagnostic_only"



def test_reports_hub_generates_primary_river_summary(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "bathy_report.json").write_text("{}", encoding="utf-8")
    (out_dir / "io_manifest.json").write_text("{}", encoding="utf-8")
    reports_dir = out_dir / "reports"
    reports_dir.mkdir()
    (reports_dir / "workflow_run_diagnosis_summary.json").write_text("{}", encoding="utf-8")
    (reports_dir / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt").write_text("summary", encoding="utf-8")
    active_eval = out_dir / "benchmark_river_active_evaluation_summary.json"
    active_eval.write_text(json.dumps({
        "available": True,
        "active_river_benchmark_mode": "support_aware_primary",
        "active_river_benchmark_decision_basis": "unsupported_support_aware_science",
        "hard_problem_holdout_blind": True,
        "river_specific_withheld_support_benchmark_recommended": True,
        "withheld_support_plan_available": False,
        "withheld_support_cli_flag": "--river-withheld-support-csv benchmark/river_withheld_support_points.csv",
        "suggested_next_action": "run_withheld_support_benchmark",
    }), encoding="utf-8")
    primary_summary = out_dir / "river_primary_guidance_summary.json"
    primary_summary.write_text(json.dumps({
        "active_product_name": "river_primary_surface",
        "primary_builder_mode": "channel_surface_scaffold",
        "primary_surface_contract_ok": True,
        "degraded_mode_active": False,
        "continuity_safeguard_used": False,
    }), encoding="utf-8")
    primary_contract = out_dir / "river_primary_surface_contract.json"
    primary_contract.write_text(json.dumps({"ok": True, "failures": []}), encoding="utf-8")
    outputs_receipt = out_dir / "final_route_outputs_receipt.json"
    outputs_receipt.write_text(json.dumps({'artifact_roles': {'conditioned_depth': 'primary', 'river_primary_guidance_summary': 'diagnostic_only'}}), encoding='utf-8')
    final_route_receipt = out_dir / 'final_route_receipt.json'
    final_route_receipt.write_text(json.dumps({'artifact_roles': {'final_depth': 'primary', 'river_primary_surface': 'diagnostic_only'}}), encoding='utf-8')
    backbone = out_dir / "river_backbone_smoothing_summary.json"
    backbone.write_text(json.dumps({"available": True, "adjusted_station_count": 3}), encoding="utf-8")
    width = out_dir / "river_centerline_width_propagation_summary.json"
    width.write_text(json.dumps({"available": True, "backbone_led_station_count": 4, "bank_margin_damping_station_count": 1}), encoding="utf-8")
    role = out_dir / "river_role_agreement_summary.json"
    role.write_text(json.dumps({"available": True, "weakest_role": "bank_edge", "lateral_failure_mode": "bank_vs_inner_shape"}), encoding="utf-8")
    section = out_dir / "river_section_target_agreement_summary.json"
    section.write_text(json.dumps({"available": True, "weakest_role_class": "bank_edge"}), encoding="utf-8")
    outputs = write_reports_hub(out_dir=out_dir, report={"outputs": {
        "workflow_run_diagnosis_summary_json": str(reports_dir / "workflow_run_diagnosis_summary.json"),
        "workflow_run_diagnosis_summary_txt": str(reports_dir / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt"),
        "benchmark_river_active_evaluation_summary_json": str(active_eval),
        "river_primary_guidance_summary": str(primary_summary),
        "river_primary_surface_contract": str(primary_contract),
    }, "authoritative_base": {"outputs": {"final_route_receipt": str(final_route_receipt)}}, "final_dem_route": {"stage_receipts": {"outputs": str(outputs_receipt)}}, "river": {"outputs": {
        "backbone_smoothing_summary": str(backbone),
        "centerline_width_propagation_summary": str(width),
        "channel_surface_role_agreement_summary": str(role),
        "channel_surface_section_target_agreement_summary": str(section),
    }}})
    manifest = json.loads(Path(outputs["reports_manifest"]).read_text(encoding="utf-8"))
    copied = {entry["label"]: entry for entry in manifest["copied"]}
    assert copied["river_active_summary"]["artifact_role"] == "primary"
    assert copied["river_active_runtime_summary"]["artifact_role"] == "diagnostic_only"
    assert copied["river_active_shape_summary"]["artifact_role"] == "diagnostic_only"
    assert copied["benchmark_river_active_evaluation_summary_json"]["artifact_role"] == "diagnostic_only"
    summary = json.loads((out_dir / "reports" / "river_active_summary.json").read_text(encoding="utf-8"))
    assert summary["artifact_role"] == "primary"
    assert summary["primary_signal"] == "river_shape"
    assert summary["dominant_remaining_issue"] == "bank_vs_inner_shape"
    assert summary["active_river_benchmark_mode"] == "support_aware_primary"
    assert summary["drilldown_receipt_paths"]["river_active_runtime_summary.json"] is not None

