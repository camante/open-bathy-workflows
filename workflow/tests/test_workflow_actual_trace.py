import json
from pathlib import Path

from workflow_actual_trace import build_workflow_actual_trace_text, write_workflow_actual_trace


def test_write_workflow_actual_trace_uses_actual_paths(tmp_path: Path):
    io_manifest = {
        "run_id": "r1",
        "out_dir": str(tmp_path),
        "inputs": ["/data/input_a.tif", "/data/input_b.gpkg"],
        "outputs": [str(tmp_path / "final.tif"), str(tmp_path / "river" / "river_surface.tif")],
    }
    (tmp_path / "io_manifest.json").write_text(json.dumps(io_manifest), encoding="utf-8")
    report = {
        "run_id": "r1",
        "config": {"aoi": "1/2/3/4", "start_date": "2025-01-01", "end_date": "2026-01-01", "methods": ["river"]},
        "workflow_execution_state": {
            "final_outputs": {
                "final_native": str(tmp_path / "final_native.tif"),
                "final_for_user": str(tmp_path / "final_for_user.tif"),
                "final_provenance": str(tmp_path / "final_provenance.tif"),
            }
        },
        "river": {
            "outputs": {
                "channel_surface": str(tmp_path / "river" / "river_channel_surface.tif"),
                "longitudinal_profile": str(tmp_path / "river" / "river_longitudinal_profile.csv"),
            }
        },
        "outputs": {
            "final_depth_user": str(tmp_path / "final_for_user.tif"),
        },
    }
    path = write_workflow_actual_trace(out_dir=tmp_path, report=report)
    text = path.read_text(encoding="utf-8")
    assert "INPUTS OBSERVED DURING THIS RUN" in text
    assert "/data/input_a.tif" in text
    assert str(tmp_path / "river" / "river_channel_surface.tif") in text
    assert "river.outputs.channel_surface" in text
    assert report["outputs"]["workflow_input_output_trace"] == str(path)


def test_build_workflow_actual_trace_text_without_io_manifest(tmp_path: Path):
    report = {
        "config": {"aoi": "aoi"},
        "outputs": {"final_depth_user": str(tmp_path / "final.tif")},
    }
    text = build_workflow_actual_trace_text(out_dir=tmp_path, report=report)
    assert "STAGE-BY-STAGE FILE TRACE" in text
    assert str(tmp_path / "final.tif") in text
