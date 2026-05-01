"""Phase 1 active river module-boundary guards.

The goal is to document and protect the active river path before deleting or
moving legacy code. These checks do not execute geospatial code.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "active_river_modules.json"
DOC_PATH = ROOT / "docs" / "ACTIVE_RIVER_MODULES.md"

DISALLOWED_ACTIVE_CORE_IMPORT_TOPS = {
    "linear_v1_runner",
    "linear_v1_runner_contract",
    "river_runner",
    "river_runner_contract",
    "river_workflow_entry",
    "river_primary_surface_rebuild",
    "river_channel_scaffold",
    "river_channel_surface",
    "river_channel_template",
    "river_graph_backbone_solver",
    "river_xs_realism",
    "river_bank_guidance",
    "river_bank_longitudinal_fit",
}


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


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
def test_phase1_boundary_manifest_and_doc_exist() -> None:
    assert MANIFEST_PATH.is_file()
    assert DOC_PATH.is_file()
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "Active River Module Boundary" in text
    assert "shared solve domain" in text
    assert "canonical parent river solution" in text
    assert "exact AOI export/window" in text
    assert "one input artifact -> one transformation -> one output artifact -> one receipt/check" in text


def test_phase1_manifest_paths_exist() -> None:
    manifest = _manifest()
    path_keys = [
        "entrypoint_orchestration",
        "active_river_core",
        "active_parent_export_identity",
        "active_verify_reporting_boundary_files",
        "verify_only_tools",
    ]
    missing: list[str] = []
    for key in path_keys:
        for rel in manifest[key]:
            if not (ROOT / rel).exists():
                missing.append(f"{key}:{rel}")
    assert not missing, "active river boundary points to missing files:\n" + "\n".join(missing)


def test_phase1_active_core_does_not_import_quarantined_workflows() -> None:
    manifest = _manifest()
    active_files = (
        manifest["active_river_core"]
        + manifest["active_parent_export_identity"]
        + manifest.get("active_verify_reporting_boundary_files", [])
    )
    violations: list[str] = []
    for rel in active_files:
        path = ROOT / rel
        if path.suffix != ".py":
            continue
        hits = sorted(_import_tops(path) & DISALLOWED_ACTIVE_CORE_IMPORT_TOPS)
        for hit in hits:
            violations.append(f"{rel} imports {hit}")
    assert not violations, "active core imports quarantined workflow modules:\n" + "\n".join(violations)


def test_phase1_transitional_utility_is_documented_not_hidden() -> None:
    manifest = _manifest()
    transitional = set(manifest.get("active_transitional_utilities", []))
    assert "river_structured_scaffold.py" in transitional
    doc = DOC_PATH.read_text(encoding="utf-8")
    assert "Active verify/reporting boundary files" in doc
    assert "Phase 2 route ownership reporting boundary" in doc
    assert "Active transitional utilities" in doc
    assert "river_structured_scaffold.py" in doc
    assert "extract the small helper surface" in doc
