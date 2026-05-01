from __future__ import annotations

from typing import Any, Mapping, Sequence

WORKFLOW_STAGE_ORDER = [
    "run_contract",
    "solve_domain",
    "grids",
    "authoritative_inputs",
    "centerline_points",
    "centerline_wse_proxy",
    "centerline_authoritative_bed",
    "centerline_observed_offset",
    "centerline_modeled_offset",
    "centerline_bed_backbone",
    "river_corridor_solve",
    "river_primary_surface_solve",
    "river_primary_surface_solve_locked",
    "river_export_handoff",
    "final_dem",
]

LINEAR_STAGE_VALIDATION_CHECKS: dict[str, list[str]] = {
    "run_contract": [
        "validate_run_contract",
    ],
    "solve_domain": [
        "validate_linear_canonical_network_gpkg",
        "validate_solve_aoi_json",
    ],
    "grids": [
        "validate_grid_template:solve",
        "validate_grid_template:export",
        "validate_solve_contains_export",
        "validate_grid_alignment",
    ],
    "authoritative_inputs": [
        "validate_raster_matches_template:solve_authoritative",
        "validate_support_mask_matches_measured_only:solve",
        "validate_raster_matches_template:export_authoritative",
        "validate_support_mask_matches_measured_only:export",
        "validate_background_covers_measured_only",
    ],
    "centerline_points": [
        "validate_centerline_points_gpkg",
        "validate_centerline_points_overlap_template",
    ],
    "centerline_wse_proxy": [
        "validate_wse_proxy_points_gpkg",
    ],
    "centerline_authoritative_bed": [
        "validate_authoritative_bed_points_gpkg",
    ],
    "centerline_observed_offset": [
        "validate_observed_offset_points_gpkg",
    ],
    "centerline_modeled_offset": [
        "validate_modeled_offset_points_gpkg",
    ],
    "centerline_bed_backbone": [
        "validate_backbone_points_gpkg",
    ],
    "river_corridor_solve": [
        "validate_mask_raster",
    ],
    "river_primary_surface_solve": [
        "validate_surface_raster",
    ],
    "river_primary_surface_solve_locked": [
        "validate_locked_surface_raster",
    ],
    "river_export_handoff": [
        "validate_raster_matches_template:river_guidance_export",
        "validate_mask_raster:river_corridor_mask_export",
        "validate_mask_raster:river_support_mask_export",
        "validate_take_mask_raster",
    ],
    "final_dem": [
        "validate_final_dem_raster",
    ],
}



# Read-only first-wrong-artifact guidance for active river stages. These
# records are diagnostic metadata only; they must never be used to reroute,
# repair, or choose alternate DEM products.
LINEAR_STAGE_DIAGNOSTIC_CONTRACTS: dict[str, dict[str, str]] = {
    "run_contract": {
        "primary_input": "requested run configuration",
        "primary_output": "00_run_contract.json",
        "critical_invariant": "Frozen run configuration declares the active linear river stage order and final DEM ownership before construction begins.",
        "first_wrong_artifact_hint": "If this fails, inspect the command/config handoff before any river artifact.",
    },
    "solve_domain": {
        "primary_input": "export AOI, requested solve domain, and canonical river network source",
        "primary_output": "01_canonical_solve_network.gpkg and 01_canonical_solve_aoi.json",
        "critical_invariant": "The solve domain is canonical for the river system and contains the export AOI without using AOI-local construction as the parent solution.",
        "first_wrong_artifact_hint": "If downstream AOIs disagree, first inspect the canonical solve network and solve AOI receipts.",
    },
    "grids": {
        "primary_input": "canonical solve AOI and export AOI",
        "primary_output": "02_solve_grid_template.tif and 02_export_grid_template.tif",
        "critical_invariant": "Solve and export grids are explicit, aligned, and stable before rasters are built or exported.",
        "first_wrong_artifact_hint": "If rasters have shape/transform drift, first inspect the solve/export grid templates.",
    },
    "authoritative_inputs": {
        "primary_input": "canonical grid templates plus authoritative/baseline source rasters",
        "primary_output": "measured-only authoritative rasters, support masks, and baseline background rasters",
        "critical_invariant": "Measured authoritative support is separated from baseline/background interpolation before river modeling.",
        "first_wrong_artifact_hint": "If authoritative lock or bed sampling looks wrong, first inspect measured-only authoritative rasters and support masks.",
    },
    "centerline_points": {
        "primary_input": "canonical solve network and solve grid",
        "primary_output": "05_centerline_points.gpkg",
        "critical_invariant": "Centerline stationing is canonical on the solve domain and is not recomputed separately for each export AOI.",
        "first_wrong_artifact_hint": "If WSE, offsets, or backbone are spatially misplaced, first inspect centerline_points.",
    },
    "centerline_wse_proxy": {
        "primary_input": "centerline points plus WSE/support evidence from measured/bank-support sources",
        "primary_output": "06_centerline_wse_proxy_points.gpkg",
        "critical_invariant": "WSE proxy is a smooth downstream water-surface guide derived before bed offsets and remains diagnostic/read-only in validation.",
        "first_wrong_artifact_hint": "If bed elevations have the wrong longitudinal trend, first inspect WSE proxy support/trend/pre-smooth/final points.",
    },
    "centerline_authoritative_bed": {
        "primary_input": "centerline points and measured-only authoritative bed raster",
        "primary_output": "07_centerline_authoritative_bed_points.gpkg",
        "critical_invariant": "Authoritative bed samples come only from measured/support cells, not from interpolated baseline background.",
        "first_wrong_artifact_hint": "If observed offsets are implausible, first inspect centerline_authoritative_bed_points.",
    },
    "centerline_observed_offset": {
        "primary_input": "WSE proxy points and authoritative bed points",
        "primary_output": "08_centerline_observed_offset_points.gpkg",
        "critical_invariant": "Observed offset is computed explicitly as WSE proxy minus measured authoritative bed where both are supported.",
        "first_wrong_artifact_hint": "If modeled offsets are wrong, first inspect observed_offset support counts and offset ranges.",
    },
    "centerline_modeled_offset": {
        "primary_input": "observed offset points and centerline stationing",
        "primary_output": "09_centerline_modeled_offset_points.gpkg",
        "critical_invariant": "Modeled offset fills the canonical centerline from explicit observed support or one documented low-support mode.",
        "first_wrong_artifact_hint": "If the bed backbone is too flat/deep/shallow, first inspect modeled_offset mode, count, and range.",
    },
    "centerline_bed_backbone": {
        "primary_input": "WSE proxy points and modeled offset points",
        "primary_output": "10_centerline_bed_backbone_points.gpkg",
        "critical_invariant": "Bed backbone is derived as WSE proxy minus modeled offset and is the sparse longitudinal structure for the river surface.",
        "first_wrong_artifact_hint": "If the dense surface has the wrong river shape, first inspect bed_backbone points before the raster surface.",
    },
    "river_corridor_solve": {
        "primary_input": "canonical solve grid and channel/corridor geometry",
        "primary_output": "11_river_corridor_solve.tif",
        "critical_invariant": "The solve-domain river corridor is explicit and carried forward; downstream stages do not rediscover the channel mask.",
        "first_wrong_artifact_hint": "If river guidance appears outside/inside the wrong area, first inspect river_corridor_solve.",
    },
    "river_primary_surface_solve": {
        "primary_input": "bed backbone and solve-domain river corridor",
        "primary_output": "11_river_primary_surface_solve.tif",
        "critical_invariant": "The primary dense river surface is generated once on the canonical solve grid from the backbone/corridor artifacts.",
        "first_wrong_artifact_hint": "If the dense river DEM looks wrong before lock/export, first inspect river_primary_surface_solve.",
    },
    "river_primary_surface_solve_locked": {
        "primary_input": "primary river surface and measured-only authoritative support mask",
        "primary_output": "12_river_primary_surface_solve_locked.tif",
        "critical_invariant": "Authoritative measured cells are locked after the river guidance surface is built.",
        "first_wrong_artifact_hint": "If measured cells changed, first inspect the locked surface and authoritative support mask.",
    },
    "river_export_handoff": {
        "primary_input": "locked canonical river surface, export grid, corridor/support/take masks",
        "primary_output": "13/14 export-grid river guidance and masks",
        "critical_invariant": "Export artifacts are subsets/warps of canonical solve artifacts and do not rerun river construction.",
        "first_wrong_artifact_hint": "If AOI products differ from the parent, first inspect river export handoff before final DEM materialization.",
    },
    "final_dem": {
        "primary_input": "AOI export DEM/guidance, support mask, take mask, and baseline background",
        "primary_output": "combined/DEM_enhanced.tif",
        "critical_invariant": "The user-facing final DEM is written once from the named AOI export artifact; no AOI-local solve or final patching is allowed.",
        "first_wrong_artifact_hint": "If final output is wrong but AOI export is correct, inspect only final materialization and writer receipts.",
    },
}

LINEAR_ARTIFACT_SPECS: dict[str, dict[str, str]] = {
    "run_contract": {
        "name": "00_run_contract.json",
        "class": "contract",
        "meaning": "Frozen linear river run inputs and stage policy.",
    },
    "canonical_solve_network": {
        "name": "01_canonical_solve_network.gpkg",
        "class": "solve_domain",
        "meaning": "Canonical connected solve reach traced from export AOI.",
    },
    "canonical_solve_aoi": {
        "name": "01_canonical_solve_aoi.json",
        "class": "solve_domain",
        "meaning": "Solve-domain bounds in EPSG:4269 and trace metadata.",
    },
    "solve_grid_template": {
        "name": "02_solve_grid_template.tif",
        "class": "solve_grid",
        "meaning": "Explicit solve-grid template for all solve-domain rasters.",
    },
    "export_grid_template": {
        "name": "02_export_grid_template.tif",
        "class": "export_grid",
        "meaning": "Explicit export-grid template for all export-domain rasters.",
    },
    "solve_authoritative_base_measured_only": {
        "name": "03_solve_authoritative_base_measured_only.tif",
        "class": "solve_authoritative",
        "meaning": "Solve-grid measured-only authoritative base; nodata where unsupported.",
    },
    "solve_authoritative_support_mask": {
        "name": "03_solve_authoritative_support_mask.tif",
        "class": "solve_support_mask",
        "meaning": "Solve-grid boolean authoritative support mask derived from measured-only support.",
    },
    "export_authoritative_base_measured_only": {
        "name": "03_export_authoritative_base_measured_only.tif",
        "class": "export_authoritative",
        "meaning": "Export-grid measured-only authoritative base; nodata where unsupported.",
    },
    "export_authoritative_support_mask": {
        "name": "03_export_authoritative_support_mask.tif",
        "class": "export_support_mask",
        "meaning": "Export-grid boolean authoritative support mask derived from measured-only support.",
    },
    "export_baseline_background": {
        "name": "03_export_baseline_background.tif",
        "class": "export_background",
        "meaning": "Export-grid baseline background raster; never treated as authoritative support.",
    },
    "centerline_points": {
        "name": "05_centerline_points.gpkg",
        "class": "solve_sparse_points",
        "meaning": "Solve-domain centerline scaffold points with stationing and component identifiers.",
    },

    "centerline_wse_support_points": {
        "name": "06a_centerline_wse_support_points.gpkg",
        "class": "solve_sparse_points",
        "meaning": "Solve-domain raw WSE support evidence sampled from the canonical measured-only authoritative raster before interpolation or monotone smoothing.",
    },
    "centerline_wse_trend_points": {
        "name": "06b_centerline_wse_trend_points.gpkg",
        "class": "solve_sparse_points",
        "meaning": "Solve-domain WSE trend/interpolation values derived from support evidence before rolling median smoothing and monotone enforcement.",
    },
    "centerline_wse_pre_smooth_points": {
        "name": "06c_centerline_wse_pre_smooth_points.gpkg",
        "class": "solve_sparse_points",
        "meaning": "Solve-domain WSE values after local smoothing/floor protection and before final monotone enforcement.",
    },
    "centerline_wse_proxy_points": {
        "name": "06_centerline_wse_proxy_points.gpkg",
        "class": "solve_sparse_points",
        "meaning": "Solve-domain centerline water-surface proxy points derived from authoritative support raster sampling.",
    },
    "centerline_authoritative_bed_points": {
        "name": "07_centerline_authoritative_bed_points.gpkg",
        "class": "solve_sparse_points",
        "meaning": "Solve-domain centerline authoritative bed samples intersected from measured-only support raster.",
    },
    "centerline_observed_offset_points": {
        "name": "08_centerline_observed_offset_points.gpkg",
        "class": "solve_sparse_points",
        "meaning": "Solve-domain observed offsets computed as WSE proxy minus authoritative bed at matched centerline points.",
    },
    "centerline_modeled_offset_points": {
        "name": "09_centerline_modeled_offset_points.gpkg",
        "class": "solve_sparse_points",
        "meaning": "Solve-domain modeled offsets across the full centerline derived from observed offsets with one explicit solve-domain prior fallback.",
    },
    "centerline_bed_backbone_points": {
        "name": "10_centerline_bed_backbone_points.gpkg",
        "class": "solve_sparse_points",
        "meaning": "Solve-domain sparse centerline bed backbone computed as WSE proxy minus modeled offset and smoothed monotonically downstream.",
    },
    "river_primary_surface_solve": {
        "name": "11_river_primary_surface_solve.tif",
        "class": "solve_surface",
        "meaning": "Solve-grid dense river primary surface created directly from the sparse backbone over the solve-domain corridor.",
    },
    "river_corridor_solve": {
        "name": "11_river_corridor_solve.tif",
        "class": "solve_corridor_mask",
        "meaning": "Solve-grid explicit river corridor mask written alongside the primary surface and carried forward to export without rediscovery.",
    },
    "river_primary_surface_solve_locked": {
        "name": "12_river_primary_surface_solve_locked.tif",
        "class": "solve_surface_locked",
        "meaning": "Solve-grid dense river primary surface after one authoritative lock step using the explicit solve support mask.",
    },
    "river_guidance_export": {
        "name": "13_river_guidance_export.tif",
        "class": "export_guidance",
        "meaning": "Export-grid river guidance raster subset from the solve-grid primary surface.",
    },
    "river_corridor_mask_export": {
        "name": "13_river_corridor_mask_export.tif",
        "class": "export_corridor_mask",
        "meaning": "Export-grid explicit river corridor mask carried forward directly from the solve-grid corridor artifact.",
    },
    "river_support_mask_export": {
        "name": "13_river_support_mask_export.tif",
        "class": "export_river_support_mask",
        "meaning": "Export-grid river support mask copied from the explicit measured authoritative export support mask.",
    },
    "river_take_mask_export": {
        "name": "14_river_take_mask_export.tif",
        "class": "export_take_mask",
        "meaning": "Export-grid river take mask defined as corridor mask and not support mask.",
    },
    "dem_enhanced_final": {
        "name": "15_DEM_enhanced.tif",
        "class": "final_dem",
        "meaning": "Final export-grid DEM written once using measured authoritative support first, river guidance on the take mask, and baseline background elsewhere.",
    },
}

REQUIRED_RUN_CONTRACT_FIELDS = {
    "run_id",
    "workflow_name",
    "export_aoi",
    "requested_solve_domain",
    "resolved_solve_domain",
    "solve_domain_source",
    "projected_crs",
    "target_resolution_m",
    "canonical_max_trace_km",
    "shared_solve",
    "stage_order",
    "artifacts",
}


# Canonical/AOI/final product-role contract.
# These names are intentionally centralized so the active workflow cannot
# silently treat an internal river raster as the user-facing final DEM.
CANONICAL_PARENT_DEM_ROLE = "canonical_parent_dem"
AOI_EXPORT_DEM_ROLE = "aoi_export_dem"
FINAL_USER_DEM_ROLE = "final_user_dem"
FINAL_USER_DEM_RELATIVE_PATH = "combined/DEM_enhanced.tif"
FINAL_DEM_MATERIALIZER_ROLE = "final_dem_materializer"


def assert_final_dem_source_role(source_role: str) -> None:
    """Require the final user DEM to be materialized from an AOI export role."""
    if str(source_role) != AOI_EXPORT_DEM_ROLE:
        raise ValueError(f"final_dem_invalid_source_role:{source_role}")


def assert_materialization_receipt_valid(receipt: Mapping[str, Any]) -> None:
    """Validate the strict Bundle-A final DEM materialization receipt."""
    if not isinstance(receipt, Mapping):
        raise ValueError("final_dem_materialization_receipt_not_mapping")
    if receipt.get("stage") != "final_dem_materialization":
        raise ValueError("final_dem_materialization_receipt_invalid_stage")
    assert_final_dem_source_role(str(receipt.get("source_role")))
    if receipt.get("destination_role") != FINAL_USER_DEM_ROLE:
        raise ValueError(f"final_dem_invalid_destination_role:{receipt.get('destination_role')}")
    if receipt.get("writer_role") != FINAL_DEM_MATERIALIZER_ROLE:
        raise ValueError(f"final_dem_invalid_writer_role:{receipt.get('writer_role')}")
    if bool(receipt.get("aoi_local_solve")):
        raise ValueError("final_dem_materialization_used_aoi_local_solve")


def assert_no_aoi_local_final_solve(receipt: Mapping[str, Any]) -> None:
    """Reject receipts that indicate local interpolation/blending wrote the final DEM."""
    if not isinstance(receipt, Mapping):
        raise ValueError("final_dem_receipt_not_mapping")
    if bool(receipt.get("aoi_local_solve")) or bool(receipt.get("terrain_interpolation")) or bool(receipt.get("blending")):
        raise ValueError("final_dem_aoi_local_mutation_detected")



def assert_single_final_dem_writer(final_receipts: Sequence[Mapping[str, Any]]) -> None:
    """Require exactly one receipt for the stable final user DEM writer."""
    writer_receipts = []
    for receipt in final_receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError("final_dem_writer_receipt_not_mapping")
        if receipt.get("destination_role") == FINAL_USER_DEM_ROLE or receipt.get("writer_role") == FINAL_DEM_MATERIALIZER_ROLE:
            writer_receipts.append(receipt)
    if len(writer_receipts) != 1:
        raise ValueError(f"final_dem_single_writer_violation:{len(writer_receipts)}")
    assert_materialization_receipt_valid(writer_receipts[0])


def assert_aoi_export_identity_passed(identity_receipt: Mapping[str, Any]) -> None:
    """Require the AOI export identity check against the canonical parent to pass."""
    if not isinstance(identity_receipt, Mapping):
        raise ValueError("aoi_identity_receipt_not_mapping")
    if identity_receipt.get("stage") != "aoi_identity":
        raise ValueError(f"aoi_identity_invalid_stage:{identity_receipt.get('stage')}")
    if not bool(identity_receipt.get("checked_against_parent")):
        raise ValueError("aoi_identity_not_checked_against_parent")
    if not bool(identity_receipt.get("passed")):
        raise ValueError(f"aoi_identity_failed:{identity_receipt.get('failure_reason')}")



def diagnostic_contract_for_stage(stage_name: str) -> dict[str, str]:
    """Return read-only first-wrong-artifact guidance for an active stage."""
    contract = LINEAR_STAGE_DIAGNOSTIC_CONTRACTS.get(stage_name, {})
    return {str(k): str(v) for k, v in contract.items()}


def artifact_name(key: str) -> str:
    spec = LINEAR_ARTIFACT_SPECS.get(key)
    if spec is None:
        raise KeyError(f"unknown_linear_artifact:{key}")
    return str(spec["name"])


def validate_stage_output_name(key: str, path_like: Any) -> None:
    expected = artifact_name(key)
    actual = str(path_like).replace("\\", "/").rsplit("/", 1)[-1]
    if actual != expected:
        raise ValueError(f"linear_artifact_name_mismatch:{key}:{actual}!={expected}")


def validation_checks_for_stage(stage_name: str) -> list[str]:
    return list(LINEAR_STAGE_VALIDATION_CHECKS.get(stage_name, []))


def validate_run_contract(payload: Mapping[str, Any]) -> None:
    linear_input_bundle = payload.get("linear_input_bundle")
    if not isinstance(linear_input_bundle, Mapping):
        raise ValueError("linear_run_contract_missing_linear_input_bundle")
    for key in (
        "solve_source_aoi",
        "export_aoi",
        "projected_crs",
        "target_resolution_m",
        "resolved_solve_domain",
        "solve_domain_source",
        "solve_network_gpkg",
        "export_network_gpkg",
        "solve_authoritative_source_path",
        "export_authoritative_source_path",
        "export_baseline_source_path",
        "authoritative_source_contract_path",
        "baseline_source_contract_path",
    ):
        value = linear_input_bundle.get(key)
        if value in (None, ""):
            raise ValueError(f"linear_run_contract_missing_input_bundle_field:{key}")
    missing = sorted(REQUIRED_RUN_CONTRACT_FIELDS.difference(payload.keys()))
    if missing:
        raise ValueError(f"linear_run_contract_missing_fields:{','.join(missing)}")
    stage_order = list(payload.get("stage_order") or [])
    if stage_order != WORKFLOW_STAGE_ORDER:
        raise ValueError("linear_run_contract_invalid_stage_order")
    shared_solve = payload.get("shared_solve")
    if not isinstance(shared_solve, Mapping):
        raise ValueError("linear_run_contract_invalid_shared_solve")
    if "reused" not in shared_solve:
        raise ValueError("linear_run_contract_missing_shared_solve_field:reused")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("linear_run_contract_invalid_artifacts")
    for key in (
        "run_contract",
        "canonical_solve_network",
        "canonical_solve_aoi",
        "solve_grid_template",
        "export_grid_template",
        "solve_authoritative_base_measured_only",
        "solve_authoritative_support_mask",
        "export_authoritative_base_measured_only",
        "export_authoritative_support_mask",
        "export_baseline_background",
        "centerline_points",
        "centerline_wse_proxy_points",
        "centerline_authoritative_bed_points",
        "centerline_observed_offset_points",
        "centerline_modeled_offset_points",
        "centerline_bed_backbone_points",
        "river_primary_surface_solve",
        "river_corridor_solve",
        "river_primary_surface_solve_locked",
        "river_guidance_export",
        "river_corridor_mask_export",
        "river_support_mask_export",
        "river_take_mask_export",
        "dem_enhanced_final",
    ):
        if key not in artifacts:
            raise ValueError(f"linear_run_contract_missing_artifact:{key}")
        validate_stage_output_name(key, artifacts[key])


__all__ = [
    "AOI_EXPORT_DEM_ROLE",
    "CANONICAL_PARENT_DEM_ROLE",
    "FINAL_DEM_MATERIALIZER_ROLE",
    "FINAL_USER_DEM_RELATIVE_PATH",
    "FINAL_USER_DEM_ROLE",
    "LINEAR_ARTIFACT_SPECS",
    "WORKFLOW_STAGE_ORDER",
    "LINEAR_STAGE_DIAGNOSTIC_CONTRACTS",
    "LINEAR_STAGE_VALIDATION_CHECKS",
    "REQUIRED_RUN_CONTRACT_FIELDS",
    "artifact_name",
    "diagnostic_contract_for_stage",
    "assert_aoi_export_identity_passed",
    "assert_final_dem_source_role",
    "assert_materialization_receipt_valid",
    "assert_no_aoi_local_final_solve",
    "assert_single_final_dem_writer",
    "validate_run_contract",
    "validate_stage_output_name",
    "validation_checks_for_stage",
]
