"""Phase 6 legacy-boundary guards for the active river route.

These tests are static only. They protect the canonical-parent/AOI-export
construction path from re-entering older river workflow modules while cleanup is
in progress. They do not execute geospatial code or run the workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MANIFEST = ROOT / "active_river_modules.json"
LEGACY_MANIFEST = ROOT / "legacy_river_quarantine_manifest.json"
DOC_PATH = ROOT / "docs" / "ACTIVE_RIVER_MODULES.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _import_tops(path: Path) -> set[str]:
    # Lightweight static scan: enough for top-level import guards and much
    # faster than walking very large ASTs in packaged workflow tests.
    tops: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("import "):
            rest = line[len("import "):].split("#", 1)[0]
            for part in rest.split(","):
                name = part.strip().split(" as ", 1)[0].strip()
                if name:
                    tops.add(name.split(".")[0])
        elif line.startswith("from "):
            rest = line[len("from "):].split(" import ", 1)[0].strip()
            if rest and not rest.startswith("."):
                tops.add(rest.split(".")[0])
    return tops
def test_phase6_legacy_boundary_manifest_exists_and_is_verify_only() -> None:
    active = _load_json(ACTIVE_MANIFEST)
    assert active["legacy_boundary_manifest"] == "legacy_river_quarantine_manifest.json"

    legacy = _load_json(LEGACY_MANIFEST)
    assert legacy["policy"]["verify_only_static_guard"] is True
    assert legacy["policy"]["delete_or_move_legacy_modules_in_this_phase"] is False
    assert legacy["policy"]["does_not_change_dem_construction"] is True
    assert legacy["policy"]["does_not_change_science"] is True
    assert legacy["policy"]["does_not_change_aoi_export"] is True
    assert legacy["policy"]["does_not_change_final_materialization"] is True


def test_phase6_hard_quarantine_paths_are_not_missing_from_manifest() -> None:
    legacy = _load_json(LEGACY_MANIFEST)
    missing: list[str] = []
    for rel in legacy["hard_quarantine_paths"]:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
    assert not missing, "legacy quarantine manifest points to missing paths:\n" + "\n".join(missing)


def test_phase6_active_construction_core_does_not_import_hard_quarantine_modules() -> None:
    legacy = _load_json(LEGACY_MANIFEST)
    hard_modules = set(legacy["hard_quarantine_modules"])
    active_files = legacy["active_construction_boundary_files"] + legacy.get("active_verify_reporting_boundary_files", [])

    violations: list[str] = []
    for rel in active_files:
        path = ROOT / rel
        assert path.exists(), f"active construction boundary file is missing: {rel}"
        if path.suffix != ".py":
            continue
        hits = sorted(_import_tops(path) & hard_modules)
        for hit in hits:
            violations.append(f"{rel} imports hard-quarantine module {hit}")

    assert not violations, "active river construction re-enters legacy workflow modules:\n" + "\n".join(violations)


def test_phase6_transitional_utility_is_explicitly_allowlisted() -> None:
    legacy = _load_json(LEGACY_MANIFEST)
    transitional = legacy["active_transitional_utilities"]
    assert transitional == [
        {
            "path": "river_structured_scaffold.py",
            "imported_by": "pipeline/river_workflow/river_workflow_stage_centerline.py",
            "allowed_imports": ["build_centerline_points"],
            "cleanup_target": "extract build_centerline_points and any strictly required helpers into pipeline/river_workflow before quarantining the remaining historical scaffold module",
        }
    ]

    stage_text = (ROOT / "pipeline/river_workflow/river_workflow_stage_centerline.py").read_text(encoding="utf-8")
    assert "from river_structured_scaffold import build_centerline_points" in stage_text

    doc = DOC_PATH.read_text(encoding="utf-8")
    assert "Phase 6 legacy-boundary guard" in doc
    assert "active construction files and verify/reporting files must not import hard-quarantine river workflow modules" in doc
    assert "river_structured_scaffold.py" in doc
    assert "helper extraction" in doc


def test_phase6_reporting_files_are_not_labeled_as_construction_stages() -> None:
    legacy = _load_json(LEGACY_MANIFEST)
    construction = set(legacy["active_construction_boundary_files"])
    reporting = set(legacy.get("active_verify_reporting_boundary_files", []))
    assert reporting == {
        "pipeline/run_summary.py",
        "reporting/final_reporting.py",
        "tools/compare_aoi_exports.py",
    }
    assert construction.isdisjoint(reporting)
