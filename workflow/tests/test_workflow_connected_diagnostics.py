import json
from pathlib import Path

from workflow_connected_diagnostics import (
    build_connected_diagnostics_payload,
    write_connected_diagnostics,
)


def test_write_connected_diagnostics_writes_stage_trace_file_graph_and_explanation(tmp_path: Path):
    io_manifest = {
        "inputs": ["/data/source_a.tif", str(tmp_path / "upstream" / "river_support.csv")],
        "outputs": [str(tmp_path / "river" / "river_surface.tif"), str(tmp_path / "benchmark" / "benchmark_summary.json")],
    }
    (tmp_path / "io_manifest.json").write_text(json.dumps(io_manifest), encoding="utf-8")
    report = {
        "river": {
            "status": "ran",
            "reason": "used weak-support thalweg render",
            "source_support_csv": str(tmp_path / "upstream" / "river_support.csv"),
            "outputs": {
                "river_surface": str(tmp_path / "river" / "river_surface.tif"),
                "effect_summary": str(tmp_path / "river" / "river_effect_summary.json"),
            },
            "candidate_pixel_count": 10,
            "changed_pixel_count": 8,
        },
        "benchmark": {
            "requested": False,
            "reason": "no_holdout_flag",
            "summary_json": str(tmp_path / "benchmark" / "benchmark_summary.json"),
        },
        "outputs": {
            "workflow_input_output_trace": str(tmp_path / "reports" / "WORKFLOW_INPUT_OUTPUT_TRACE.txt"),
        },
    }
    outputs = write_connected_diagnostics(out_dir=tmp_path, report=report)
    assert Path(outputs["workflow_stage_trace_json"]).exists()
    assert Path(outputs["workflow_stage_trace_jsonl"]).exists()
    assert Path(outputs["workflow_file_graph_json"]).exists()
    assert Path(outputs["workflow_explanation_report"]).exists()
    explanation = Path(outputs["workflow_explanation_report"]).read_text(encoding="utf-8")
    assert "RIVER WORKFLOW" in explanation
    assert "status: ran" in explanation
    assert "BENCHMARK" in explanation
    assert "reason: no_holdout_flag" in explanation
    stage_trace = json.loads(Path(outputs["workflow_stage_trace_json"]).read_text(encoding="utf-8"))
    river_stage = next(row for row in stage_trace if row["stage_name"] == "river")
    assert any(item["path"].endswith("river_support.csv") for item in river_stage["input_files"])
    assert any(item["path"].endswith("river_surface.tif") for item in river_stage["output_files"])
    assert report["outputs"]["workflow_stage_trace_json"] == outputs["workflow_stage_trace_json"]


def test_build_connected_diagnostics_payload_connects_output_to_downstream_consumer(tmp_path: Path):
    io_manifest = {
        "inputs": [],
        "outputs": [str(tmp_path / "river" / "river_surface.tif")],
    }
    (tmp_path / "io_manifest.json").write_text(json.dumps(io_manifest), encoding="utf-8")
    shared_path = str(tmp_path / "river" / "river_surface.tif")
    report = {
        "river": {"outputs": {"river_surface": shared_path}},
        "benchmark": {"reference_surface": shared_path},
    }
    payload = build_connected_diagnostics_payload(out_dir=tmp_path, report=report)
    stages = payload["stage_trace"]
    river_stage = next(row for row in stages if row["stage_name"] == "river")
    benchmark_stage = next(row for row in stages if row["stage_name"] == "benchmark")
    assert "benchmark" in river_stage["downstream_consumers"]
    assert "river" in benchmark_stage["upstream_dependencies"]
    artifacts = payload["file_graph"]["artifacts"]
    artifact = next(row for row in artifacts if row["path"] == shared_path)
    assert artifact["producers"] == ["river"]
    assert artifact["consumers"] == ["benchmark"]


def test_connected_diagnostics_marks_missing_and_orphan_outputs(tmp_path: Path):
    missing_path = str(tmp_path / "river" / "missing_surface.tif")
    existing_orphan = tmp_path / "river" / "diagnostic_only.json"
    existing_orphan.parent.mkdir(parents=True, exist_ok=True)
    existing_orphan.write_text("{}", encoding="utf-8")
    report = {
        "river": {
            "outputs": {
                "river_surface": missing_path,
                "diagnostic_only": str(existing_orphan),
            },
        },
    }
    payload = build_connected_diagnostics_payload(out_dir=tmp_path, report=report)
    river_stage = next(row for row in payload["stage_trace"] if row["stage_name"] == "river")
    assert river_stage["missing_output_count"] == 1
    assert any(item["path"] == missing_path for item in river_stage["missing_output_files"])
    assert river_stage["orphan_output_count"] == 2
    file_graph = payload["file_graph"]["artifacts"]
    missing_art = next(row for row in file_graph if row["path"] == missing_path)
    assert missing_art["exists"] is False
    diag_art = next(row for row in file_graph if row["path"] == str(existing_orphan))
    assert diag_art["exists"] is True
