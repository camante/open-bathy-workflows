"""Phase 0-9 cleanup consistency checks.

These checks make sure the cleanup documentation, active module manifest, and
legacy-boundary manifest stay synchronized. They are static/read-only and do
not execute geospatial workflow code.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "active_river_modules.json"
LEGACY = ROOT / "legacy_river_quarantine_manifest.json"
DOC = ROOT / "docs" / "ACTIVE_RIVER_MODULES.md"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cleanup_policy_lists_every_phase_0_to_9() -> None:
    policy = _json(ACTIVE)["cleanup_policy"]
    missing = [f"phase_{i}" for i in range(10) if f"phase_{i}" not in policy]
    assert not missing, f"cleanup policy missing phases: {missing}"


def test_reporting_files_are_not_labeled_as_parent_export_identity_helpers() -> None:
    active = _json(ACTIVE)
    identity = set(active["active_parent_export_identity"])
    reporting = set(active["active_verify_reporting_boundary_files"])
    assert reporting == {
        "pipeline/run_summary.py",
        "reporting/final_reporting.py",
        "tools/compare_aoi_exports.py",
    }
    assert identity.isdisjoint(reporting)


def test_active_and_legacy_boundary_reporting_lists_match() -> None:
    active = _json(ACTIVE)
    legacy = _json(LEGACY)
    assert active["active_verify_reporting_boundary_files"] == legacy["active_verify_reporting_boundary_files"]


def test_active_docs_record_phase2_and_phase0_to_9_scope() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "Phase 0–9 cleanup policy" in text
    assert "Phase 2 route ownership reporting boundary" in text
    assert "active_route = canonical_river_parent_export" in text
    assert "Active verify/reporting boundary files" in text
