"""Repo-level runtime metadata and operator-facing contracts.

This module is the single source of truth for the simplified repo story:
- the single built-in river workflow name
- the absence of method-style aliases in the active river workflow
- which files define the normal run operator contract
- which outputs are debug-only and should not be treated as default run outputs
"""

from __future__ import annotations

from typing import FrozenSet, Tuple

ACTIVE_RIVER_WORKFLOW: str = "river_workflow"
COMPATIBILITY_DEFAULT_RIVER_WORKFLOW: str = "river_workflow"
SUPPORTED_RIVER_WORKFLOWS: Tuple[str, ...] = (
    "river_workflow",
)
# There is one built-in active river workflow. Historical method-like names are
# quarantined in legacy code and must not normalize into active runtime routes.
LEGACY_RIVER_WORKFLOWS: FrozenSet[str] = frozenset()
RIVER_WORKFLOW_ALIASES: FrozenSet[str] = frozenset()
NORMAL_RUN_PRIMARY_FILES: Tuple[str, ...] = (
    "RUN_OVERVIEW.txt",
    "bathy_report.json",
    "io_manifest.json",
)
NORMAL_RUN_REPORTS_FILES: Tuple[str, ...] = (
    "reports/README_FIRST.txt",
    "reports/WORKFLOW_RUN_DIAGNOSIS_SUMMARY.txt",
    "reports/WORKFLOW_INPUT_OUTPUT_TRACE.txt",
    "reports/river_active_summary.json",
    "reports/river_science_chain_summary.json",
    # Durable canonical identity metadata must survive normal cleanup so AOI
    # comparison can prove that exports came from the same canonical parent.
    "reports/canonical_river_solution_manifest.json",
    "reports/aoi_export_identity.json",
    "reports/RIVER_ARCHITECTURE_SUMMARY.txt",
    "reports/run_summary.json",
    "reports/run_summary.txt",
    "reports/WSE_CURRENT_PATH_AUDIT.txt",
    "reports/WSE_DIRECTION_AUDIT.txt",
    "reports/WSE_PROXY_AUDIT.txt",
    "reports/wse_artifacts/wse_artifact_manifest.json",
    "reports/wse_artifacts/06a_centerline_wse_support_points.gpkg",
    "reports/wse_artifacts/06b_centerline_wse_trend_points.gpkg",
    "reports/wse_artifacts/06c_centerline_wse_pre_smooth_points.gpkg",
    "reports/wse_artifacts/06_centerline_wse_proxy_points.gpkg",
    "reports/OFFSET_CURRENT_PATH_AUDIT.txt",
    "reports/OFFSET_OBSERVED_QC_AUDIT.txt",
    "reports/offset_artifacts/offset_artifact_manifest.json",
    "reports/offset_artifacts/08_centerline_observed_offset_points.gpkg",
    "reports/offset_artifacts/08a_observed_offset_candidates_qc.gpkg",
    "reports/OFFSET_MODEL_AUDIT.txt",
    "reports/BED_BACKBONE_AUDIT.txt",
    "reports/offset_artifacts/09_centerline_modeled_offset_points.gpkg",
    "reports/offset_artifacts/10_centerline_bed_backbone_points.gpkg",
    "reports/offset_artifacts/modeled_offset_artifact_manifest.json",
    "reports/offset_artifacts/bed_backbone_artifact_manifest.json",
    "reports/LATERAL_CURRENT_PATH_AUDIT.txt",
    "reports/LATERAL_CROSS_SECTION_AUDIT.txt",
    "reports/lateral_artifacts/lateral_artifact_manifest.json",
    "reports/lateral_artifacts/river_lateral_distance_to_centerline.tif",
    "reports/lateral_artifacts/river_lateral_distance_to_bank.tif",
    "reports/lateral_artifacts/river_lateral_normalized_position.tif",
    "reports/lateral_artifacts/river_lateral_taper_weight.tif",
    "reports/lateral_artifacts/river_channel_width_points.gpkg",
    "reports/lateral_artifacts/lateral_audit_cross_sections.gpkg",
    "reports/lateral_artifacts/11_river_primary_surface_solve.tif",
)
NORMAL_RUN_OPTIONAL_LOG_GLOB: str = "run_logs/screen_*.log"
DEBUG_ONLY_OUTPUTS: FrozenSet[str] = frozenset(
    {
        "contract",
        "receipts",
        "manifests",
        "RIVER_WORKFLOW_DEBUG.txt",
        "WORKFLOW_INPUT_OUTPUT_TRACE.txt",
        "unified_bathy_report.json",
        "io_manifest.md",
    }
)
LEGACY_CODE_DIRS: Tuple[str, ...] = (
    "legacy/river",
    "legacy/debug",
)
VALIDATION_CODE_DIRS: Tuple[str, ...] = (
    "validation",
)
REPORTING_CODE_DIRS: Tuple[str, ...] = (
    "reporting",
)
DEBUG_CODE_DIRS: Tuple[str, ...] = ()
CORE_CODE_DIRS: Tuple[str, ...] = (
    "core",
)
PIPELINE_CODE_DIRS: Tuple[str, ...] = (
    "pipeline",
    "pipeline/river_shared_solve",
    "pipeline/river_workflow",
    "pipeline/final_route",
    "pipeline/final_dem",
)

PRIMARY_DOCS: Tuple[str, ...] = (
    "README.md",
    "docs/ACTIVE_WORKFLOW.md",
    "docs/RUN_OUTPUTS.md",
    "docs/REPO_MAP.md",
    "docs/LEGACY_WORKFLOWS.md",
)


def normalize_active_river_workflow_name(value: str | None) -> str:
    label = str(value or "").strip()
    if not label:
        return ACTIVE_RIVER_WORKFLOW
    if label == ACTIVE_RIVER_WORKFLOW:
        return ACTIVE_RIVER_WORKFLOW
    return label
