from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_package_restores_required_river_support_modules() -> None:
    required = [
        "river_withheld_support.py",
        "river_primary_surface_contract.py",
        "river_station_target_contract.py",
        "river_section_tendency.py",
        "river_longitudinal_tendency.py",
        "river_source_semantics.py",
        "river_target_contract.py",
        "river_support_semantics.py",
        "river_support_uncertainty.py",
        "legacy/river/archive_root_scripts/river_xs_realism.py",
        "river_prediction_confidence.py",
        "legacy/river/archive_root_scripts/river_primary_surface_rebuild.py",
        "river_effectiveness_receipts.py",
        "legacy/river/archive_root_scripts/river_graph_backbone_solver.py",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    assert missing == []


def test_active_legacy_unavailable_scaffold_stub_removed_from_centerline_path() -> None:
    scaffold_text = (ROOT / "river_structured_scaffold.py").read_text(encoding="utf-8")
    assert "legacy_structured_scaffold_function_unavailable:build_centerline_points" not in scaffold_text
    assert "def build_centerline_points" in scaffold_text


def test_channel_scaffold_graph_backbone_module_is_not_missing() -> None:
    text = (ROOT / "legacy/river/archive_root_scripts/river_channel_scaffold.py").read_text(encoding="utf-8")
    assert "from legacy.river.archive_root_scripts.river_graph_backbone_solver import" in text
    solver = (ROOT / "legacy/river/archive_root_scripts/river_graph_backbone_solver.py").read_text(encoding="utf-8")
    for name in [
        "def _build_component_station_candidates",
        "def _solve_component_backbone",
        "def _solve_network_backbone",
        "def _build_junction_groups",
        "def summarize_graph_physical_plausibility",
    ]:
        assert name in solver
