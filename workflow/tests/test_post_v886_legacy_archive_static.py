from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "legacy" / "river" / "archive_root_scripts"
ARCHIVED_ROOT_FILES = {
    "linear_v1_runner.py",
    "linear_v1_runner_contract.py",
    "river_primary_surface_rebuild.py",
    "river_channel_scaffold.py",
    "river_channel_surface.py",
    "river_channel_template.py",
    "river_graph_backbone_solver.py",
    "river_xs_realism.py",
    "river_bank_guidance.py",
    "river_bank_longitudinal_fit.py",
}


def test_hard_quarantine_root_scripts_are_archived() -> None:
    assert ARCHIVE_DIR.is_dir()
    for name in ARCHIVED_ROOT_FILES:
        assert not (ROOT / name).exists(), f"legacy script still at repo root: {name}"
        assert (ARCHIVE_DIR / name).is_file(), f"archived legacy script missing: {name}"


def test_active_manifest_records_archive_cleanup() -> None:
    manifest = json.loads((ROOT / "active_river_modules.json").read_text(encoding="utf-8"))
    cleanup = manifest.get("legacy_archive_completed", {})
    assert cleanup.get("archive_dir") == "legacy/river/archive_root_scripts"
    assert cleanup.get("active_route_unchanged") is True
    moved = set(cleanup.get("moved_root_scripts", []))
    for name in ARCHIVED_ROOT_FILES:
        assert f"legacy/river/archive_root_scripts/{name}" in moved


def test_active_core_does_not_import_archived_legacy_modules() -> None:
    manifest = json.loads((ROOT / "active_river_modules.json").read_text(encoding="utf-8"))
    active_files = (
        manifest["active_river_core"]
        + manifest["active_parent_export_identity"]
        + manifest.get("active_verify_reporting_boundary_files", [])
    )
    archived_module_names = {Path(name).stem for name in ARCHIVED_ROOT_FILES}
    violations: list[str] = []
    for rel in active_files:
        path = ROOT / rel
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8")
        for module in sorted(archived_module_names):
            if f"from {module} import" in text or f"import {module}" in text:
                violations.append(f"{rel} imports archived legacy module {module}")
    assert not violations, "\n".join(violations)


def test_archive_doc_exists() -> None:
    text = (ROOT / "docs" / "RIVER_WORKFLOW_POST_V886_LEGACY_ARCHIVE_CLEANUP.md").read_text(encoding="utf-8")
    assert "canonical parent river solution" in text
    assert "legacy/river/archive_root_scripts" in text
    assert "river_structured_scaffold.py" in text
