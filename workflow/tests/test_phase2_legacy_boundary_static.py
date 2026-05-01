"""Phase 2 active river legacy-boundary guards.

These checks are intentionally static.  Phase 2 should clarify which files are
active bridge files and which files are inactive legacy/compatibility modules
without changing DEM construction, science, AOI export, or final materialization.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MANIFEST = ROOT / "active_river_modules.json"
LEGACY_MANIFEST = ROOT / "legacy_river_quarantine_manifest.json"
DOC_PATH = ROOT / "docs" / "RIVER_WORKFLOW_PHASE2_LEGACY_BOUNDARY.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase2_active_runner_is_bridge_not_legacy() -> None:
    active = _load_json(ACTIVE_MANIFEST)
    legacy = _load_json(LEGACY_MANIFEST)

    bridge_files = set(active["active_entrypoint_bridge"]["files"])
    assert bridge_files == {"river_runner.py", "river_runner_contract.py"}
    assert bridge_files.issubset(set(active["entrypoint_orchestration"]))

    assert bridge_files.isdisjoint(set(active["legacy_or_quarantine_candidates"]))
    assert "river_runner" not in set(legacy["hard_quarantine_modules"])
    assert "river_runner_contract" not in set(legacy["hard_quarantine_modules"])
    assert bridge_files.isdisjoint(set(legacy["hard_quarantine_paths"]))


def test_phase2_compatibility_wrappers_remain_legacy_surface() -> None:
    active = _load_json(ACTIVE_MANIFEST)
    legacy = _load_json(LEGACY_MANIFEST)

    compat = {"legacy/river/archive_root_scripts/linear_v1_runner.py", "legacy/river/archive_root_scripts/linear_v1_runner_contract.py"}
    assert compat.issubset(set(active["legacy_or_quarantine_candidates"]))
    assert compat.issubset(set(legacy["hard_quarantine_paths"]))
    assert {"linear_v1_runner", "linear_v1_runner_contract"}.issubset(set(legacy["hard_quarantine_modules"]))


def test_phase2_boundary_doc_records_bridge_policy() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "River Workflow Phase 2 Legacy Boundary" in text
    assert "river_runner.py" in text
    assert "active bridge files" in text
    assert "must not implement an alternate river workflow" in text
    assert "canonical parent DEM -> AOI export DEM -> final user DEM" in text
