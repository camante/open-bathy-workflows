from __future__ import annotations

import json
from pathlib import Path

from debug_bundle import write_debug_bundle


def test_write_debug_bundle_collects_key_run_files(tmp_path: Path):
    out_dir = tmp_path / "output"
    run_logs = out_dir / "run_logs"
    run_logs.mkdir(parents=True)
    run_id = "bathy_20260331T000000Z_12345"

    suffix = run_id.removeprefix("bathy_")
    for name in [
        f"screen_bathy_{suffix}.log",
        f"run_bathy_{suffix}.log",
        f"flight_recorder_bathy_{suffix}.jsonl",
        f"run_summary_bathy_{suffix}.json",
    ]:
        (run_logs / name).write_text(name, encoding="utf-8")

    (out_dir / "bathy_report.json").write_text("{}", encoding="utf-8")
    (out_dir / "unified_bathy_report.json").write_text("{}", encoding="utf-8")
    
    river_dir = out_dir / "river"
    river_dir.mkdir()
    (river_dir / "river_guidance_manifest.json").write_text("{}", encoding="utf-8")
    (river_dir / "river_channel_frame_contract.json").write_text("{}", encoding="utf-8")
    (river_dir / "river_longitudinal_profile_summary.json").write_text("{}", encoding="utf-8")
    (river_dir / "river_longitudinal_profile_coverage.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    report = {
        "river": {
            "outputs": {
                "guidance_manifest": str(river_dir / "river_guidance_manifest.json"),
                "channel_frame_contract": str(river_dir / "river_channel_frame_contract.json"),
                "longitudinal_profile_summary": str(river_dir / "river_longitudinal_profile_summary.json"),
                "longitudinal_profile_coverage": str(river_dir / "river_longitudinal_profile_coverage.csv"),
            }
        },
        "outputs": {
            "selected_final": str(out_dir / "final.tif"),
        },
    }

    bundle_dir = write_debug_bundle(out_dir, report=report, run_id=run_id)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    copied_labels = {item["label"] for item in manifest["copied_files"]}
    assert "screen_log" in copied_labels
    assert "run_summary_json" in copied_labels
    assert "bathy_report" in copied_labels
    assert "channel_frame_contract" in copied_labels
    assert (bundle_dir / "README.txt").exists()
    assert any(item["label"] == "selected_final" for item in manifest["spatial_output_references"])


def test_write_debug_bundle_does_not_report_missing_legacy_run_logs(tmp_path: Path):
    out_dir = tmp_path / "output"
    run_logs = out_dir / "run_logs"
    run_logs.mkdir(parents=True)
    run_id = "bathy_20260331T000000Z_99999"
    suffix = run_id.removeprefix("bathy_")
    for name in [
        f"screen_bathy_{suffix}.log",
        f"run_bathy_{suffix}.log",
        f"flight_recorder_bathy_{suffix}.jsonl",
        f"run_summary_bathy_{suffix}.json",
    ]:
        (run_logs / name).write_text(name, encoding="utf-8")
    report = {"river": {"outputs": {}}, "outputs": {}}
    bundle_dir = write_debug_bundle(out_dir, report=report, run_id=run_id)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    missing_labels = {item.get("label") for item in manifest["missing_or_skipped"]}
    assert all(not str(label).endswith("_legacy") for label in missing_labels if label is not None)


def test_write_debug_bundle_collects_role_contract_and_benchmark_receipts(tmp_path: Path):
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    run_logs = out_dir / "run_logs"
    run_logs.mkdir(parents=True)
    run_id = "bathy_20260402T000000Z_11111"
    (run_logs / f"screen_bathy_{run_id}.log").write_text("screen", encoding="utf-8")
    (out_dir / "bathy_report.json").write_text("{}", encoding="utf-8")

    river_dir = out_dir / "river"
    river_dir.mkdir()
    role_contract = river_dir / "authoritative_river_role_contract.json"
    role_contract.write_text("{}", encoding="utf-8")
    role_csv = river_dir / "authoritative_river_support.csv"
    role_csv.write_text("x,y,depth_m,source\n", encoding="utf-8")
    (river_dir / "river_channel_surface_longitudinal_smoothing_summary.json").write_text("{}", encoding="utf-8")
    bench_dir = out_dir / "benchmark"
    bench_dir.mkdir()
    (bench_dir / "benchmark_summary.json").write_text("{}", encoding="utf-8")
    (bench_dir / "benchmark_authoritative_role_validation_summary.json").write_text("{}", encoding="utf-8")
    (bench_dir / "benchmark_river_receipt_triage_summary.json").write_text("{}", encoding="utf-8")
    (bench_dir / "benchmark_river_primary_focus_summary.json").write_text("{}", encoding="utf-8")
    role_code = river_dir / "authoritative_role_code.tif"
    role_code.write_text("placeholder", encoding="utf-8")

    report = {
        "authoritative_base": {
            "river_guidance": {
                "path": str(role_csv),
                "role_contract": str(role_contract),
                "role_code_raster": str(role_code),
            }
        },
        "outputs": {
            "benchmark_summary_json": str(bench_dir / "benchmark_summary.json"),
            "benchmark_authoritative_role_validation_summary_json": str(bench_dir / "benchmark_authoritative_role_validation_summary.json"),
            "benchmark_river_receipt_triage_summary_json": str(bench_dir / "benchmark_river_receipt_triage_summary.json"),
            "benchmark_river_primary_focus_summary_json": str(bench_dir / "benchmark_river_primary_focus_summary.json"),
        },
        "river": {"outputs": {"channel_surface_longitudinal_smoothing_summary": str(river_dir / "river_channel_surface_longitudinal_smoothing_summary.json")}},
    }

    bundle_dir = write_debug_bundle(out_dir, report=report, run_id=run_id)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    copied_labels = {item["label"] for item in manifest["copied_files"]}
    assert "authoritative_river_role_contract" in copied_labels
    assert "benchmark_authoritative_role_validation_summary_json" in copied_labels
    assert "benchmark_river_receipt_triage_summary_json" in copied_labels
    assert "benchmark_river_primary_focus_summary_json" in copied_labels
    assert "channel_surface_longitudinal_smoothing_summary" in copied_labels
    assert any(item["label"] == "authoritative_river_role_code_raster" for item in manifest["spatial_output_references"])


def test_write_debug_bundle_collects_workflow_trace_when_reported(tmp_path: Path):
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    run_logs = out_dir / "run_logs"
    run_logs.mkdir(parents=True)
    run_id = "bathy_20260403T000000Z_22222"
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True)
    trace_path = reports_dir / "WORKFLOW_INPUT_OUTPUT_TRACE.txt"
    trace_path.write_text("trace", encoding="utf-8")
    report = {
        "outputs": {
            "workflow_input_output_trace": str(trace_path),
        },
        "river": {"outputs": {}},
    }
    bundle_dir = write_debug_bundle(out_dir, report=report, run_id=run_id)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    copied_labels = {item["label"] for item in manifest["copied_files"]}
    assert "workflow_input_output_trace" in copied_labels
    assert (bundle_dir / "reports" / "WORKFLOW_INPUT_OUTPUT_TRACE.txt").exists()


def test_write_debug_bundle_collects_connected_diagnostics_reports(tmp_path: Path):
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    run_id = "bathy_20260404T000000Z_33333"
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True)
    for name in [
        "workflow_stage_trace.json",
        "workflow_stage_trace_lines.jsonl",
        "workflow_file_graph.json",
        "WORKFLOW_EXPLANATION_REPORT.txt",
        "workflow_accuracy_anomalies.json",
        "workflow_first_bad_artifact_summary.json",
        "workflow_run_diagnosis_summary.json",
        "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt",
    ]:
        (reports_dir / name).write_text("{}", encoding="utf-8")
    report = {
        "outputs": {
            "workflow_stage_trace_json": str(out_dir / "reports" / "workflow_stage_trace.json"),
            "workflow_stage_trace_jsonl": str(out_dir / "reports" / "workflow_stage_trace_lines.jsonl"),
            "workflow_file_graph_json": str(out_dir / "reports" / "workflow_file_graph.json"),
            "workflow_explanation_report": str(out_dir / "reports" / "WORKFLOW_EXPLANATION_REPORT.txt"),
            "workflow_accuracy_anomalies_json": str(out_dir / "reports" / "workflow_accuracy_anomalies.json"),
            "workflow_first_bad_artifact_summary_json": str(out_dir / "reports" / "workflow_first_bad_artifact_summary.json"),
            "workflow_run_diagnosis_summary_json": str(out_dir / "reports" / "workflow_run_diagnosis_summary.json"),
            "workflow_run_diagnosis_summary_txt": str(out_dir / "reports" / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt"),
        },
        "river": {"outputs": {}},
    }
    bundle_dir = write_debug_bundle(out_dir, report=report, run_id=run_id)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    copied_labels = {item["label"] for item in manifest["copied_files"]}
    assert "workflow_stage_trace_json" in copied_labels
    assert "workflow_stage_trace_jsonl" in copied_labels
    assert "workflow_file_graph_json" in copied_labels
    assert "workflow_explanation_report" in copied_labels
    assert "workflow_accuracy_anomalies_json" in copied_labels
    assert "workflow_first_bad_artifact_summary_json" in copied_labels
    assert "workflow_run_diagnosis_summary_json" in copied_labels
    assert "workflow_run_diagnosis_summary_txt" in copied_labels
    assert (bundle_dir / "reports" / "WORKFLOW_EXPLANATION_REPORT.txt").exists()
    assert (bundle_dir / "reports" / "WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt").exists()
