from workflow_actual_trace import build_workflow_actual_trace_text


def test_workflow_actual_trace_reports_river_v2_pass3_progress(tmp_path):
    report = {
        "config": {"aoi": "x", "start": "2025-01-01", "end": "2026-01-01", "methods": ["river"]},
        "river": {
            "v2_pass3": {
                "stage_status": {
                    "river_centerline": {"implemented": True, "status": "implemented", "canonical_filename": "river_centerline_points.gpkg", "output_artifact": str(tmp_path / "river_centerline_points.gpkg")},
                    "centerline_wse_proxy": {"implemented": True, "status": "implemented", "canonical_filename": "centerline_wse_proxy_points.gpkg", "output_artifact": str(tmp_path / "centerline_wse_proxy_points.gpkg")},
                    "river_primary_surface": {"implemented": False, "status": "not_run", "canonical_filename": "river_primary_surface.tif"},
                }
            }
        },
    }
    text = build_workflow_actual_trace_text(out_dir=tmp_path, report=report)
    assert "river_v2_pass3_progress: 2/3 implemented" in text
    assert "RIVER V2 PASS3 STAGE STATUS" in text


def test_workflow_actual_trace_reports_river_v2_pass4_progress(tmp_path):
    report = {
        "config": {"aoi": "x", "start": "2025-01-01", "end": "2026-01-01", "methods": ["river"]},
        "river": {
            "v2_pass4": {
                "stage_status": {
                    "river_centerline": {"implemented": True, "status": "implemented", "canonical_filename": "river_centerline_points.gpkg", "output_artifact": str(tmp_path / "river_centerline_points.gpkg")},
                    "river_primary_surface_authoritative_applied": {"implemented": True, "status": "implemented", "canonical_filename": "river_primary_surface_authoritative_applied.tif", "output_artifact": str(tmp_path / "river_primary_surface_authoritative_applied.tif")},
                }
            }
        },
    }
    text = build_workflow_actual_trace_text(out_dir=tmp_path, report=report)
    assert "river_v2_pass4_progress: 2/2 implemented" in text
    assert "RIVER V2 PASS4 STAGE STATUS" in text


def test_workflow_actual_trace_reports_v2_final_route_contract(tmp_path):
    report = {
        "config": {"aoi": "x", "start": "2025-01-01", "end": "2026-01-01", "methods": ["river"]},
        "final_dem_contract": {
            "final_route_contract": {
                "river_v2_final_route_contract": {
                    "active": True,
                    "active_stage": "river_primary_surface_authoritative_applied",
                    "active_river_guidance_surface": str(tmp_path / "river_primary_surface_authoritative_applied.tif"),
                    "legacy_river_final_route_participation_blocked": True,
                    "runtime_enforced": True,
                }
            }
        },
    }
    text = build_workflow_actual_trace_text(out_dir=tmp_path, report=report)
    assert "river_v2_final_route_active: True" in text
    assert "river_v2_final_route_stage: river_primary_surface_authoritative_applied" in text
    assert "river_v2_legacy_final_route_participation_blocked: True" in text
