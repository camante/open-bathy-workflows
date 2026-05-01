from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_active_pipeline_imports_numbered_stage_modules() -> None:
    text = (ROOT / "pipeline/river_workflow/river_workflow_pipeline.py").read_text(encoding="utf-8")
    expected = [
        "stage_01_solve_domain",
        "stage_02_grids",
        "stage_03_authoritative_source",
        "stage_04_centerline",
        "stage_05_wse_proxy",
        "stage_06_authoritative_bed",
        "stage_07_observed_offset",
        "stage_08_modeled_offset",
        "stage_09_bed_backbone",
        "stage_10_river_corridor",
        "stage_11_primary_surface",
        "stage_12_authoritative_lock",
        "stage_13_aoi_export",
        "stage_14_final_products",
    ]
    for name in expected:
        assert f"pipeline.river_workflow.{name}" in text


def test_no_broad_exception_handlers_in_active_river_construction_files() -> None:
    active_files = list((ROOT / "pipeline/river_workflow").glob("*.py"))
    active_files += [ROOT / "active_pipeline.py", ROOT / "river_runner.py", ROOT / "river_workflow.py"]
    offenders: list[str] = []
    for path in active_files:
        text = path.read_text(encoding="utf-8")
        if "except Exception" in text or "except:" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_stage_error_types_are_available() -> None:
    text = (ROOT / "pipeline/river_workflow/river_workflow_stage_errors.py").read_text(encoding="utf-8")
    assert "class RiverWorkflowStageError" in text
    assert "class RiverWorkflowInputError" in text
