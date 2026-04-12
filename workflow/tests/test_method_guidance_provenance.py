from pathlib import Path

from method_guidance_provenance import apply_parallel_method_guidance_summary


def _write_json(path: Path, payload: dict) -> Path:
    import json

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_apply_parallel_method_guidance_summary_builds_parallel_architecture(tmp_path: Path):
    river_candidate = tmp_path / "river_candidate.tif"
    river_candidate.write_text("x", encoding="utf-8")
    river_active = tmp_path / "river_active.tif"
    river_active.write_text("x", encoding="utf-8")
    river_auth_mask = tmp_path / "river_auth_mask.tif"
    river_auth_mask.write_text("x", encoding="utf-8")
    river_auth_depth = tmp_path / "river_auth_depth.tif"
    river_auth_depth.write_text("x", encoding="utf-8")
    river_channel_surface = tmp_path / "river_channel_surface.tif"
    river_channel_surface.write_text("x", encoding="utf-8")
    river_helper = tmp_path / "river_depth_helper.tif"
    river_helper.write_text("x", encoding="utf-8")
    river_manifest = _write_json(
        tmp_path / "river_guidance_manifest.json",
        {"artifact_roles": {"channel_surface": "channel_fitted_river_surface"}},
    )

    sdb_candidate = tmp_path / "sdb_candidate.tif"
    sdb_candidate.write_text("x", encoding="utf-8")
    sdb_active = tmp_path / "sdb_active.tif"
    sdb_active.write_text("x", encoding="utf-8")
    sdb_raw = tmp_path / "sdb_raw.tif"
    sdb_raw.write_text("x", encoding="utf-8")
    sdb_locked = tmp_path / "sdb_locked.tif"
    sdb_locked.write_text("x", encoding="utf-8")
    sdb_auth_mask = tmp_path / "sdb_auth_mask.tif"
    sdb_auth_mask.write_text("x", encoding="utf-8")
    sdb_auth_values = tmp_path / "sdb_auth_values.tif"
    sdb_auth_values.write_text("x", encoding="utf-8")
    sdb_auth_points = tmp_path / "sdb_auth_points.gpkg"
    sdb_auth_points.write_text("x", encoding="utf-8")
    sdb_auth_contract = tmp_path / "sdb_auth_contract.json"
    sdb_auth_contract.write_text("{}", encoding="utf-8")
    sdb_manifest = _write_json(
        tmp_path / "sdb_guidance_manifest.json",
        {"artifact_roles": {"sdb_guidance_active": "active_guidance"}},
    )

    report = {
        "outputs": {
            "river_candidate_domain_mask": str(river_candidate),
            "river_active_domain_mask": str(river_active),
            "sdb_candidate_domain_mask": str(sdb_candidate),
        },
        "shared_domain_stage": {
            "summary": {
                "derived_activation": {"river_should_run": True, "sdb_should_run": True},
                "masks": {
                    "river_candidate_domain_mask": str(river_candidate),
                    "river_active_domain_mask": str(river_active),
                    "sdb_candidate_domain_mask": str(sdb_candidate),
                },
            }
        },
        "river": {
            "status": "success",
            "outputs": {
                "guidance_manifest": str(river_manifest),
                "authoritative_support": str(river_auth_mask),
                "authoritative_support_depth": str(river_auth_depth),
                "depth_terrain_internal_helper": str(river_helper),
                "channel_surface": str(river_channel_surface),
            },
        },
        "sdb": {
            "status": "success",
            "guidance_manifest": str(sdb_manifest),
            "guidance_active": str(sdb_active),
            "raw_prediction_raster": str(sdb_raw),
            "locked_guidance_raster": str(sdb_locked),
            "guidance_mode": "authoritative_locked_guidance",
            "shared_domain_mask_used": str(sdb_candidate),
            "shared_domain_activation_should_run": True,
            "artifacts": {
                "authoritative_support_mask": str(sdb_auth_mask),
                "authoritative_support_values": str(sdb_auth_values),
                "authoritative_support_points": str(sdb_auth_points),
                "authoritative_support_contract": str(sdb_auth_contract),
            },
        },
    }

    parallel = apply_parallel_method_guidance_summary(report=report, out_root=tmp_path)

    assert parallel["river"]["active_guidance_product"] == "river_channel_surface.tif"
    assert parallel["river"]["raw_method_output"] == "river_depth_helper.tif"
    assert parallel["river"]["active_product_role"] == "channel_fitted_river_surface"
    assert parallel["river"]["domain_should_run"] is True
    assert parallel["sdb"]["active_guidance_product"] == "sdb_active.tif"
    assert parallel["sdb"]["locked_guidance_product"] == "sdb_locked.tif"
    assert parallel["sdb"]["authoritative_support_contract"] == "sdb_auth_contract.json"
    assert report["river"]["guidance_architecture"] == parallel["river"]
    assert report["sdb"]["guidance_architecture"] == parallel["sdb"]


def test_apply_parallel_method_guidance_summary_marks_missing_active_guidance_when_domain_should_run(tmp_path: Path):
    river_candidate = tmp_path / "river_candidate.tif"
    river_candidate.write_text("x", encoding="utf-8")
    river_active = tmp_path / "river_active.tif"
    river_active.write_text("x", encoding="utf-8")
    sdb_candidate = tmp_path / "sdb_candidate.tif"
    sdb_candidate.write_text("x", encoding="utf-8")

    report = {
        "shared_domain_stage": {
            "summary": {
                "derived_activation": {"river_should_run": True, "sdb_should_run": True},
                "masks": {
                    "river_candidate_domain_mask": str(river_candidate),
                    "river_active_domain_mask": str(river_active),
                    "sdb_candidate_domain_mask": str(sdb_candidate),
                },
            }
        },
        "river": {"outputs": {}},
        "sdb": {"artifacts": {}},
    }

    parallel = apply_parallel_method_guidance_summary(report=report, out_root=tmp_path)

    assert parallel["river"]["status"] == "missing_active_guidance"
    assert parallel["sdb"]["status"] == "missing_active_guidance"
