from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_phase9_policy_manifest_is_verify_only_and_focused() -> None:
    payload = json.loads(_read("validation/manifests/river_validation_policy_manifest.json"))
    policy = payload["policy"]
    assert policy["verify_only"] is True
    assert policy["does_not_change_construction"] is True
    assert policy["does_not_drive_routing"] is True
    assert policy["does_not_repair_outputs"] is True
    assert policy["does_not_select_dem_source"] is True
    assert policy["suppress_unconfigured_placeholder_validation"] is True
    assert payload["allowed_questions"] == [
        "Did AOI export exactly match the canonical parent window?",
        "Did final DEM equal AOI export?",
        "Did authoritative lock preserve measured cells?",
        "Did WSE/backbone behave physically enough to inspect?",
        "Did north/south use the same parent?",
        "What was the first failing artifact?",
    ]


def test_optional_validation_helpers_suppress_unconfigured_placeholders() -> None:
    for rel in [
        "validation/postrun_benchmark_stage.py",
        "validation/postrun_regression_stage.py",
        "validation/scientific_validation_stage.py",
        "validation/validation_invariance_framework.py",
    ]:
        text = _read(rel)
        assert "suppressed_optional_validation_payload" in text
        assert "not_run" in text or "suppressed_optional_validation_payload" in text
        assert "may_reroute_or_repair_outputs" not in text or "False" in text


def test_final_reporting_does_not_publish_not_run_validation_as_primary_output() -> None:
    text = _read("reporting/final_reporting.py")
    assert 'payload.get("status") == "not_run"' in text
    assert 'optional_validation' in text
    assert 'scientific_validation_summary"] = str(sci_path)' in text
    assert 'if sci_payload.get("status") == "not_run"' in text


def test_phase9_policy_registered_in_active_boundary() -> None:
    payload = json.loads(_read("active_river_modules.json"))
    policy = payload.get("active_validation_policy")
    assert isinstance(policy, dict)
    assert policy["phase"] == "phase9"
    assert policy["verify_only"] is True
    assert policy["does_not_change_construction"] is True
    assert policy["suppressed_when_not_configured"] == [
        "benchmark",
        "postrun_regression",
        "validation_invariance",
        "scientific_validation",
    ]


def test_phase9_modified_modules_parse() -> None:
    for rel in [
        "validation/river_validation_policy.py",
        "validation/postrun_benchmark_stage.py",
        "validation/postrun_regression_stage.py",
        "validation/scientific_validation_stage.py",
        "validation/validation_invariance_framework.py",
        "reporting/final_reporting.py",
    ]:
        ast.parse(_read(rel), filename=rel)
