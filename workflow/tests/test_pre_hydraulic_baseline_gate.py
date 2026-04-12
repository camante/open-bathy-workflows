import json
from pathlib import Path

from pre_hydraulic_baseline_gate import evaluate_pre_hydraulic_baseline


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def test_pre_hydraulic_baseline_gate_passes_clean_inland_run(tmp_path: Path):
    coverage_csv = tmp_path / "river_longitudinal_profile_coverage.csv"
    coverage_csv.write_text("profile_id,station_m,profile_support_class\n1,0,centerline_supported\n", encoding="utf-8")

    profile_summary = {
        "coverage_summary": {
            "unsupported_fraction": 0.0,
            "class_counts": {"centerline_supported": 1},
        }
    }
    frame_contract = {
        "metrics": {
            "station_count": 10,
            "component_count": 3,
            "duplicate_station_rows_collapsed": 0,
        }
    }
    stability = {"all_trusted_interior_identity_ok": True}

    profile_summary_path = _write_json(tmp_path / "profile_summary.json", profile_summary)
    frame_contract_path = _write_json(tmp_path / "frame_contract.json", frame_contract)
    stability_path = _write_json(tmp_path / "stability.json", stability)

    report = {
        "river": {
            "status": "success",
            "outputs": {
                "longitudinal_profile_coverage": str(coverage_csv),
                "longitudinal_profile_summary": profile_summary_path,
                "channel_frame_contract": frame_contract_path,
            },
            "execution_receipts": {
                "xs_support_status": {"status": "no_xs_source"},
            },
        },
        "final_dem_route": {
            "authoritative_first_contract": {
                "ok": True,
                "locked_changed_count": 0,
            },
            "manifest_contract": {
                "all_manifest_contracts_valid": True,
            },
        },
        "sdb": {"status": "inactive"},
    }
    run_summary = {
        "stats": {
            "bathy_report": report,
            "status": {"sdb": "inactive"},
        }
    }
    run_summary_path = _write_json(tmp_path / "run_summary.json", run_summary)

    payload = evaluate_pre_hydraulic_baseline(
        report_json=run_summary_path,
        river_stability_summary_json=stability_path,
        expected_sdb_mode="inactive",
        require_trusted_interior_identity=True,
        max_profile_unsupported_fraction=0.1,
    )

    assert payload["all_required_checks_ok"] is True
    assert payload["failure_count"] == 0


def test_pre_hydraulic_baseline_gate_fails_duplicate_frame_and_invalid_manifest(tmp_path: Path):
    coverage_csv = tmp_path / "river_longitudinal_profile_coverage.csv"
    coverage_csv.write_text("profile_id,station_m,profile_support_class\n1,0,unsupported\n", encoding="utf-8")
    profile_summary = {"coverage_summary": {"unsupported_fraction": 0.4}}
    frame_contract = {"metrics": {"duplicate_station_rows_collapsed": 9, "station_count": 10, "component_count": 1}}

    profile_summary_path = _write_json(tmp_path / "profile_summary.json", profile_summary)
    frame_contract_path = _write_json(tmp_path / "frame_contract.json", frame_contract)

    report = {
        "river": {
            "status": "success",
            "outputs": {
                "longitudinal_profile_coverage": str(coverage_csv),
                "longitudinal_profile_summary": profile_summary_path,
                "channel_frame_contract": frame_contract_path,
            },
            "execution_receipts": {},
        },
        "final_dem_route": {
            "authoritative_first_contract": {
                "ok": False,
                "locked_changed_count": 2,
            },
            "manifest_contract": {
                "all_manifest_contracts_valid": False,
            },
        },
        "sdb": {"status": "inactive"},
    }
    report_path = _write_json(tmp_path / "report.json", report)

    payload = evaluate_pre_hydraulic_baseline(
        report_json=report_path,
        expected_sdb_mode="inactive",
        max_profile_unsupported_fraction=0.1,
    )

    assert payload["all_required_checks_ok"] is False
    names = {f["name"] for f in payload["failures"]}
    assert "authoritative_lock_ok" in names
    assert "manifest_contract_valid" in names
    assert "xs_support_status_receipt_present" in names
    assert "channel_frame_duplicate_station_rows_collapsed" in names
    assert "longitudinal_profile_unsupported_fraction" in names


def test_pre_hydraulic_baseline_gate_resolves_relative_artifact_paths_from_report_location(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    coverage_csv = run_dir / "river_longitudinal_profile_coverage.csv"
    coverage_csv.write_text("profile_id,station_m,profile_support_class\n1,0,centerline_supported\n", encoding="utf-8")
    profile_summary_path = run_dir / "profile_summary.json"
    profile_summary_path.write_text(json.dumps({"coverage_summary": {"unsupported_fraction": 0.0}}), encoding="utf-8")
    frame_contract_path = run_dir / "frame_contract.json"
    frame_contract_path.write_text(json.dumps({"metrics": {"duplicate_station_rows_collapsed": 0, "station_count": 2, "component_count": 1}}), encoding="utf-8")

    report = {
        "river": {
            "status": "success",
            "outputs": {
                "longitudinal_profile_coverage": "river_longitudinal_profile_coverage.csv",
                "longitudinal_profile_summary": "profile_summary.json",
                "channel_frame_contract": "frame_contract.json",
            },
            "execution_receipts": {
                "xs_support_status": {"status": "inactive"},
            },
        },
        "final_dem_route": {
            "authoritative_first_contract": {"ok": True, "locked_changed_count": 0},
            "manifest_contract": {"all_manifest_contracts_valid": True},
        },
        "sdb": {"status": "inactive"},
    }
    report_path = _write_json(run_dir / "run_summary.json", report)

    payload = evaluate_pre_hydraulic_baseline(report_json=report_path, expected_sdb_mode="inactive")
    assert payload["all_required_checks_ok"] is True
