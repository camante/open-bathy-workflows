from pathlib import Path


def test_active_river_workflow_forwards_required_direct_runner_callbacks():
    text = Path("active_pipeline.py").read_text(encoding="utf-8")
    assert "run_river_workflow_details(" in text
    for required in [
        'logger=ctx.log',
        'script_dir=callbacks.get("script_dir")',
        'ensure_dir_fn=callbacks.get("ensure_dir_fn")',
        'detect_working_srs_fn=callbacks.get("detect_working_srs_fn")',
        'estimate_raster_pixel_size_m_for_dst_crs_fn=callbacks.get("estimate_raster_pixel_size_m_for_dst_crs_fn")',
        'resolve_river_shared_source_artifacts_fn=callbacks.get("resolve_river_shared_source_artifacts_fn")',
        'prepare_river_canonical_source_bundle_fn=callbacks.get("prepare_river_canonical_source_bundle_fn")',
        'return_river_workflow_details=True',
    ]:
        assert required in text


def test_active_river_workflow_does_not_call_direct_runner_without_callbacks():
    text = Path("active_pipeline.py").read_text(encoding="utf-8")
    bad_call = "run_river_workflow_details(cfg, ctx.report, river_inputs_override=None)"
    assert bad_call not in text
