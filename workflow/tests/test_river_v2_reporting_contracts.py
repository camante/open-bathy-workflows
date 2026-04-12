from pathlib import Path

from final_dem_contract import build_final_dem_contract_summary
from river_guidance import build_river_guidance_manifest


def test_final_dem_contract_summary_marks_v2_locked_surface_active(tmp_path):
    locked = tmp_path / "river_primary_surface_authoritative_applied.tif"
    locked.write_text("x", encoding="utf-8")
    guidance_manifest = tmp_path / "river_guidance_manifest.json"
    guidance_manifest.write_text("{}", encoding="utf-8")
    report = {
        "river": {
            "v2_pass4": {"success": True, "execution_mode": "river_v2_pass4_authoritative_lock_ready_for_final_dem"},
            "outputs": {
                "guidance_manifest": str(guidance_manifest),
                "primary_river_guidance_surface": str(locked),
            },
        },
        "authoritative_base": {
            "candidate_generation": {
                "stats": {},
                "backstop_policy": {"legacy_candidate_enabled": False},
            }
        },
        "final_dem_route": {
            "route_mode": "river_v2"
        },
    }
    summary = build_final_dem_contract_summary(report)
    assert summary["guidance_roles"]["dense_river_depth"] == "authoritative_applied_final_route_input"
    assert summary["final_route_contract"]["river_v2_final_route_contract"]["active"] is True


def test_river_guidance_manifest_includes_v2_pass4_and_final_route_participation(tmp_path):
    out_root = tmp_path
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    locked = tmp_path / "river_primary_surface_authoritative_applied.tif"
    locked.write_text("x", encoding="utf-8")
    report = {
        "river": {
            "outputs": {
                "primary_river_guidance_surface": str(locked),
            },
            "v2_pass4": {
                "execution_mode": "river_v2_pass4_authoritative_lock_ready_for_final_dem",
                "success": True,
                "stage_status": {
                    "river_primary_surface_authoritative_applied": {
                        "implemented": True,
                        "status": "implemented",
                        "canonical_filename": "river_primary_surface_authoritative_applied.tif",
                        "output_artifact": str(locked),
                    }
                },
                "stage_results": {},
            },
        }
    }
    manifest = build_river_guidance_manifest(out_root=out_root, river_dir=river_dir, report=report)
    assert manifest["river_v2_pass4"]["success"] is True
    assert manifest["river_v2_final_route_participation"]["active"] is True
    assert manifest["river_v2_final_route_participation"]["active_stage"] == "river_primary_surface_authoritative_applied"


def test_final_dem_contract_summary_carries_runtime_enforced_flag(tmp_path):
    locked = tmp_path / "river_primary_surface_authoritative_applied.tif"
    locked.write_text("x", encoding="utf-8")
    report = {
        "river": {
            "v2_pass4": {"success": True, "execution_mode": "river_v2_pass4_authoritative_lock_ready_for_final_dem"},
            "v2_route_contract": {"runtime_enforced": True},
            "outputs": {"primary_river_guidance_surface": str(locked)},
        },
        "authoritative_base": {
            "candidate_generation": {"stats": {}, "backstop_policy": {"legacy_candidate_enabled": False}}
        },
        "final_dem_route": {"route_mode": "river_v2"},
    }
    summary = build_final_dem_contract_summary(report)
    assert summary["final_route_contract"]["river_v2_final_route_contract"]["runtime_enforced"] is True
