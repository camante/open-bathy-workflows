import json
from pathlib import Path

from traceability_manifest import build_traceability_manifest, validate_traceability_manifest
from workflow_actual_trace import write_workflow_actual_trace


def _write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_traceability_manifest_requires_real_final_dem_and_identity_receipt(tmp_path: Path):
    final_dem = _write_bytes(tmp_path / "combined" / "DEM_enhanced.tif", b"same-raster")
    debug_written = _write_bytes(tmp_path / "combined" / "debug_final_route" / "06_DEM_enhanced_written.tif", b"same-raster")
    touch_log = tmp_path / "combined" / "dem_enhanced_touch_log.jsonl"
    touch_log.write_text(
        "\n".join(
            [
                json.dumps({"action": "final_route_write", "path": str(final_dem)}),
                json.dumps({"action": "identity_verification_start", "path": str(final_dem), "source": str(debug_written)}),
                json.dumps({"action": "identity_verification_ok", "path": str(final_dem), "source": str(debug_written)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "combined" / "dem_enhanced_identity_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "final_path": str(final_dem),
                "source": str(debug_written),
                "source_sha256": "x",
                "final_sha256": "x",
                "match": True,
                "writer": "final_route_outputs_stage",
                "verification_mode": "hash_only_no_rewrite",
            }
        ),
        encoding="utf-8",
    )
    _write_bytes(tmp_path / "RIVER_WORKFLOW_DEBUG.txt", b"ok")
    _write_bytes(tmp_path / "combined" / "debug_final_route_manifest.json", b"{}")
    (tmp_path / "io_manifest.json").write_text(json.dumps({"inputs": ["/data/a.tif"], "outputs": [str(final_dem)]}), encoding="utf-8")
    (tmp_path / "io_manifest.md").write_text("# io\n", encoding="utf-8")
    report = {"outputs": {"final_dem_user_stable": str(final_dem), "dem_enhanced_touch_log": str(touch_log), "dem_enhanced_identity_receipt": str(receipt), "river_workflow_debug_report": str(tmp_path / "RIVER_WORKFLOW_DEBUG.txt")}, "authoritative_base": {"outputs": {"stage_debug_06_dem_enhanced_written": str(debug_written), "debug_final_route_manifest": str(tmp_path / "combined" / "debug_final_route_manifest.json")}}}
    write_workflow_actual_trace(out_dir=tmp_path, report=report)

    manifest = build_traceability_manifest(tmp_path, report)
    validation = validate_traceability_manifest(manifest)

    assert manifest["single_writer_contract"]["final_dem_path_ok"] is True
    assert manifest["single_writer_contract"]["identity_receipt_ok"] is True
    assert manifest["io_observed"]["input_count"] == 1
    assert validation["ok"] is True


def test_traceability_manifest_flags_unexpected_postwrite_touch(tmp_path: Path):
    final_dem = _write_bytes(tmp_path / "combined" / "DEM_enhanced.tif", b"same-raster")
    debug_written = _write_bytes(tmp_path / "combined" / "debug_final_route" / "06_DEM_enhanced_written.tif", b"same-raster")
    touch_log = tmp_path / "combined" / "dem_enhanced_touch_log.jsonl"
    touch_log.write_text(
        "\n".join(
            [
                json.dumps({"action": "final_route_write", "path": str(final_dem)}),
                json.dumps({"action": "replace_with_copy_copy", "path": str(final_dem), "source": str(debug_written)}),
                json.dumps({"action": "identity_verification_ok", "path": str(final_dem), "source": str(debug_written)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "combined" / "dem_enhanced_identity_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "final_path": str(final_dem),
                "source": str(debug_written),
                "source_sha256": "x",
                "final_sha256": "x",
                "match": True,
                "writer": "final_route_outputs_stage",
                "verification_mode": "hash_only_no_rewrite",
            }
        ),
        encoding="utf-8",
    )
    _write_bytes(tmp_path / "RIVER_WORKFLOW_DEBUG.txt", b"ok")
    _write_bytes(tmp_path / "combined" / "debug_final_route_manifest.json", b"{}")
    (tmp_path / "io_manifest.json").write_text(json.dumps({"inputs": [], "outputs": [str(final_dem)]}), encoding="utf-8")
    (tmp_path / "io_manifest.md").write_text("# io\n", encoding="utf-8")
    report = {"outputs": {"final_dem_user_stable": str(final_dem), "dem_enhanced_touch_log": str(touch_log), "dem_enhanced_identity_receipt": str(receipt), "river_workflow_debug_report": str(tmp_path / "RIVER_WORKFLOW_DEBUG.txt")}, "authoritative_base": {"outputs": {"stage_debug_06_dem_enhanced_written": str(debug_written), "debug_final_route_manifest": str(tmp_path / "combined" / "debug_final_route_manifest.json")}}}
    write_workflow_actual_trace(out_dir=tmp_path, report=report)

    manifest = build_traceability_manifest(tmp_path, report)
    validation = validate_traceability_manifest(manifest)

    assert manifest["single_writer_contract"]["unexpected_actions"] == ["replace_with_copy_copy"]
    assert validation["ok"] is False


def test_traceability_manifest_flags_touch_on_wrong_path(tmp_path: Path):
    final_dem = _write_bytes(tmp_path / "combined" / "DEM_enhanced.tif", b"same-raster")
    debug_written = _write_bytes(tmp_path / "combined" / "debug_final_route" / "06_DEM_enhanced_written.tif", b"same-raster")
    wrong_path = tmp_path / "combined" / "DEM_enhanced_context_epsg4269.tif"
    _write_bytes(wrong_path, b"same-raster")
    touch_log = tmp_path / "combined" / "dem_enhanced_touch_log.jsonl"
    touch_log.write_text(
        "\n".join(
            [
                json.dumps({"action": "final_route_write", "path": str(wrong_path)}),
                json.dumps({"action": "identity_verification_start", "path": str(final_dem), "source": str(debug_written)}),
                json.dumps({"action": "identity_verification_ok", "path": str(final_dem), "source": str(debug_written)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "combined" / "dem_enhanced_identity_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "final_path": str(final_dem),
                "source": str(debug_written),
                "source_sha256": "x",
                "final_sha256": "x",
                "match": True,
                "writer": "final_route_outputs_stage",
                "verification_mode": "hash_only_no_rewrite",
            }
        ),
        encoding="utf-8",
    )
    _write_bytes(tmp_path / "RIVER_WORKFLOW_DEBUG.txt", b"ok")
    _write_bytes(tmp_path / "combined" / "debug_final_route_manifest.json", b"{}")
    (tmp_path / "io_manifest.json").write_text(json.dumps({"inputs": [], "outputs": [str(final_dem)]}), encoding="utf-8")
    (tmp_path / "io_manifest.md").write_text("# io\n", encoding="utf-8")
    report = {"outputs": {"final_dem_user_stable": str(final_dem), "dem_enhanced_touch_log": str(touch_log), "dem_enhanced_identity_receipt": str(receipt), "river_workflow_debug_report": str(tmp_path / "RIVER_WORKFLOW_DEBUG.txt")}, "authoritative_base": {"outputs": {"stage_debug_06_dem_enhanced_written": str(debug_written), "debug_final_route_manifest": str(tmp_path / "combined" / "debug_final_route_manifest.json")}}}
    write_workflow_actual_trace(out_dir=tmp_path, report=report)

    manifest = build_traceability_manifest(tmp_path, report)
    validation = validate_traceability_manifest(manifest)

    assert manifest["single_writer_contract"]["touch_paths_ok"] is False
    assert str(wrong_path) in manifest["single_writer_contract"]["touch_path_mismatches"]
    assert validation["ok"] is False


def test_workflow_actual_trace_uses_refreshed_io_manifest_outputs(tmp_path: Path):
    from workflow_actual_trace import build_workflow_actual_trace_text
    from io_artifacts import write_io_manifest

    combined = tmp_path / "combined"
    reports = tmp_path / "reports"
    combined.mkdir(parents=True)
    reports.mkdir(parents=True)
    final_dem = combined / "DEM_enhanced.tif"
    final_dem.write_bytes(b"final")
    trace_path = reports / "WORKFLOW_INPUT_OUTPUT_TRACE.txt"

    report = {
        "run_id": "run-1",
        "out_dir": str(tmp_path),
        "outputs": {
            "final_dem_user_stable": str(final_dem),
            "workflow_input_output_trace": str(trace_path),
        },
    }
    write_io_manifest(tmp_path, report)
    text = build_workflow_actual_trace_text(out_dir=tmp_path, report=report)
    assert str(trace_path) in text
    assert str(final_dem) in text
