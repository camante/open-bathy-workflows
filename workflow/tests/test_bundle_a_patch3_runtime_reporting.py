from pathlib import Path
from types import SimpleNamespace

from river_execution_plan import determine_river_execution_plan
from river_guidance import build_river_guidance_manifest
from sdb_guidance import build_sdb_guidance_manifest
from workflow_actual_trace import build_workflow_actual_trace_text


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _noop_normalize(cfg, report):
    return None


def test_river_execution_plan_exposes_route_state(tmp_path):
    cfg = SimpleNamespace(
        out_dir=tmp_path,
        derived_cache_root=tmp_path / "derived",
        cache_root=tmp_path / "cache",
        river_method="structured",
        river_channel_template_enabled=False,
        run_id="run1",
        aoi_tile="tile",
        aoi="aoi",
        start_date="2025-01-01",
        end_date="2025-12-31",
        working_srs="EPSG:32619",
        river_dem_source="src",
        extra_xyz_cudem=None,
    )
    report = {"river": {}}
    plan = determine_river_execution_plan(
        cfg=cfg,
        report=report,
        ensure_dir_fn=_ensure_dir,
        normalize_channel_template_setting_fn=_noop_normalize,
        logger=SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None),
    )
    assert plan.route_mode == "legacy_structured_transition"
    assert plan.target_contract_mode == "simple_river_plan_v1"
    assert "river_centerline" in plan.simple_stage_status


def test_river_guidance_manifest_includes_route_modes_and_stage_status(tmp_path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    centerline = river_dir / "river_centerline_points.gpkg"
    centerline.write_text("x", encoding="utf-8")
    report = {
        "river": {
            "outputs": {"centerline_points": str(centerline)},
            "guidance": {},
        }
    }
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert manifest["current_route_mode"] == "legacy_structured_transition"
    assert manifest["target_route_mode"] == "simple_river_plan_v1"
    assert "simple_river_stage_status" in manifest
    assert manifest["simple_river_stage_status"]["river_centerline"]["legacy_equivalent"] == "centerline_points"


def test_sdb_guidance_manifest_includes_route_modes(tmp_path):
    depth = tmp_path / "sdb_depth.tif"
    depth.write_text("x", encoding="utf-8")
    manifest = build_sdb_guidance_manifest(out_root=tmp_path, depth_raster=depth, args=SimpleNamespace(authoritative_base=""))
    assert manifest["current_route_mode"] == "legacy_structured_transition"
    assert manifest["target_route_mode"] == "simple_river_plan_v1"
    assert manifest["final_dem_policy"]["final_dem_filename"] == "DEM_enhanced.tif"


def test_workflow_actual_trace_reports_route_modes_and_stage_progress(tmp_path):
    report = {
        "run_id": "run1",
        "config": {"aoi": "a", "start": "2025-01-01", "end": "2025-12-31"},
        "workflow_execution_state": {"current_route_mode": "legacy_structured_transition", "target_route_mode": "simple_river_plan_v1"},
        "final_dem_contract": {
            "current_route_mode": "legacy_structured_transition",
            "target_route_mode": "simple_river_plan_v1",
            "simple_river_stage_status": {
                "river_centerline": {
                    "status": "implemented_via_existing_pipeline",
                    "implemented": True,
                    "canonical_filename": "river_centerline_points.gpkg",
                    "legacy_equivalent": "centerline_points",
                }
            },
            "legacy_transitional_artifacts_present": ["legacy_structured_river_guidance"],
        },
    }
    text = build_workflow_actual_trace_text(out_dir=tmp_path, report=report)
    assert "current_route_mode: legacy_structured_transition" in text
    assert "target_route_mode: simple_river_plan_v1" in text
    assert "SIMPLE RIVER TARGET STAGE STATUS" in text
