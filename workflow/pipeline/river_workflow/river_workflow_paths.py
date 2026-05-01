from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.paths import ensure_dir
from pipeline.river_workflow.river_workflow_contract import FINAL_USER_DEM_RELATIVE_PATH, artifact_name


@dataclass(frozen=True)
class RiverWorkflowPaths:
    root: Path
    contract_dir: Path
    solve_domain_dir: Path
    grids_dir: Path
    authoritative_dir: Path
    sparse_points_dir: Path
    surfaces_dir: Path
    export_dir: Path
    final_dir: Path
    receipts_dir: Path
    manifests_dir: Path
    run_contract: Path
    canonical_solve_network: Path
    canonical_solve_domain_json: Path
    canonical_solve_aoi: Path
    solve_grid_template: Path
    export_grid_template: Path
    solve_authoritative_base_measured_only: Path
    solve_authoritative_support_mask: Path
    export_authoritative_base_measured_only: Path
    export_authoritative_support_mask: Path
    export_baseline_background: Path
    solve_baseline_background: Path
    centerline_points: Path
    centerline_wse_support_points: Path
    centerline_wse_trend_points: Path
    centerline_wse_pre_smooth_points: Path
    centerline_wse_proxy_points: Path
    centerline_authoritative_bed_points: Path
    centerline_observed_offset_points: Path
    centerline_modeled_offset_points: Path
    centerline_bed_backbone_points: Path
    river_primary_surface_solve: Path
    river_corridor_solve: Path
    river_primary_surface_solve_locked: Path
    river_guidance_export: Path
    river_corridor_mask_export: Path
    river_support_mask_export: Path
    river_take_mask_export: Path
    dem_enhanced_final: Path
    canonical_parent_dem: Path
    canonical_parent_receipt: Path
    aoi_export_dem: Path
    aoi_export_receipt: Path
    final_user_dem: Path
    final_materialization_receipt: Path
    solve_domain_receipt: Path
    grids_receipt: Path
    authoritative_receipt: Path
    centerline_receipt: Path
    wse_proxy_receipt: Path
    authoritative_bed_receipt: Path
    observed_offset_receipt: Path
    modeled_offset_receipt: Path
    backbone_receipt: Path
    corridor_receipt: Path
    surface_receipt: Path
    lock_receipt: Path
    export_receipt: Path
    final_dem_receipt: Path
    bundle_manifest: Path
    canonical_solve_identity_manifest: Path
    canonical_stage_identity_manifest: Path
    canonical_cache_consistency_receipt: Path
    adjacent_aoi_verification_receipt: Path
    river_verification_summary: Path
    final_dem_writer_receipt: Path
    receipt_purpose_manifest: Path
    canonical_system_identity_receipt: Path
    trace_summary: Path

    def stage_receipt_path(self, stage_name: str) -> Path:
        return self.receipts_dir / f"{stage_name}_receipt.json"


def build_river_workflow_paths(out_dir: str | Path, *, write_diagnostics: bool = True, write_core_receipts: bool = True) -> RiverWorkflowPaths:
    root = ensure_dir(Path(out_dir))
    ensure_contract_dirs = bool(write_diagnostics or write_core_receipts)
    contract_dir = (ensure_dir(root / "contract") if ensure_contract_dirs else (root / "contract"))
    solve_domain_dir = ensure_dir(root / "solve_domain")
    grids_dir = ensure_dir(root / "grids")
    authoritative_dir = ensure_dir(root / "authoritative")
    sparse_points_dir = ensure_dir(root / "sparse_points")
    surfaces_dir = ensure_dir(root / "surfaces")
    export_dir = ensure_dir(root / "export")
    final_dir = ensure_dir(root / "final")
    receipts_dir = (ensure_dir(root / "receipts") if ensure_contract_dirs else (root / "receipts"))
    manifests_dir = (ensure_dir(root / "manifests") if ensure_contract_dirs else (root / "manifests"))
    return RiverWorkflowPaths(
        root=root,
        contract_dir=contract_dir,
        solve_domain_dir=solve_domain_dir,
        grids_dir=grids_dir,
        authoritative_dir=authoritative_dir,
        sparse_points_dir=sparse_points_dir,
        surfaces_dir=surfaces_dir,
        export_dir=export_dir,
        final_dir=final_dir,
        receipts_dir=receipts_dir,
        manifests_dir=manifests_dir,
        run_contract=contract_dir / artifact_name("run_contract"),
        canonical_solve_network=solve_domain_dir / artifact_name("canonical_solve_network"),
        canonical_solve_domain_json=receipts_dir / "canonical_solve_domain_receipt.json",
        canonical_solve_aoi=solve_domain_dir / artifact_name("canonical_solve_aoi"),
        solve_grid_template=grids_dir / artifact_name("solve_grid_template"),
        export_grid_template=grids_dir / artifact_name("export_grid_template"),
        solve_authoritative_base_measured_only=authoritative_dir / artifact_name("solve_authoritative_base_measured_only"),
        solve_authoritative_support_mask=authoritative_dir / artifact_name("solve_authoritative_support_mask"),
        export_authoritative_base_measured_only=authoritative_dir / artifact_name("export_authoritative_base_measured_only"),
        export_authoritative_support_mask=authoritative_dir / artifact_name("export_authoritative_support_mask"),
        export_baseline_background=authoritative_dir / artifact_name("export_baseline_background"),
        solve_baseline_background=authoritative_dir / "03_solve_baseline_background.tif",
        centerline_points=sparse_points_dir / artifact_name("centerline_points"),
        centerline_wse_support_points=sparse_points_dir / artifact_name("centerline_wse_support_points"),
        centerline_wse_trend_points=sparse_points_dir / artifact_name("centerline_wse_trend_points"),
        centerline_wse_pre_smooth_points=sparse_points_dir / artifact_name("centerline_wse_pre_smooth_points"),
        centerline_wse_proxy_points=sparse_points_dir / artifact_name("centerline_wse_proxy_points"),
        centerline_authoritative_bed_points=sparse_points_dir / artifact_name("centerline_authoritative_bed_points"),
        centerline_observed_offset_points=sparse_points_dir / artifact_name("centerline_observed_offset_points"),
        centerline_modeled_offset_points=sparse_points_dir / artifact_name("centerline_modeled_offset_points"),
        centerline_bed_backbone_points=sparse_points_dir / artifact_name("centerline_bed_backbone_points"),
        river_primary_surface_solve=surfaces_dir / artifact_name("river_primary_surface_solve"),
        river_corridor_solve=surfaces_dir / artifact_name("river_corridor_solve"),
        river_primary_surface_solve_locked=surfaces_dir / artifact_name("river_primary_surface_solve_locked"),
        river_guidance_export=export_dir / artifact_name("river_guidance_export"),
        river_corridor_mask_export=export_dir / artifact_name("river_corridor_mask_export"),
        river_support_mask_export=export_dir / artifact_name("river_support_mask_export"),
        river_take_mask_export=export_dir / artifact_name("river_take_mask_export"),
        dem_enhanced_final=final_dir / artifact_name("dem_enhanced_final"),
        canonical_parent_dem=final_dir / "canonical_parent_dem.tif",
        canonical_parent_receipt=receipts_dir / "canonical_parent_dem_receipt.json",
        aoi_export_dem=export_dir / "DEM_enhanced_export.tif",
        aoi_export_receipt=receipts_dir / "aoi_export_dem_receipt.json",
        final_user_dem=root / FINAL_USER_DEM_RELATIVE_PATH,
        final_materialization_receipt=receipts_dir / "final_dem_materialization_receipt.json",
        solve_domain_receipt=receipts_dir / "solve_domain_receipt.json",
        grids_receipt=receipts_dir / "grids_receipt.json",
        authoritative_receipt=receipts_dir / "authoritative_receipt.json",
        centerline_receipt=receipts_dir / "centerline_receipt.json",
        wse_proxy_receipt=receipts_dir / "wse_proxy_receipt.json",
        authoritative_bed_receipt=receipts_dir / "authoritative_bed_receipt.json",
        observed_offset_receipt=receipts_dir / "observed_offset_receipt.json",
        modeled_offset_receipt=receipts_dir / "modeled_offset_receipt.json",
        backbone_receipt=receipts_dir / "backbone_receipt.json",
        corridor_receipt=receipts_dir / "corridor_receipt.json",
        surface_receipt=receipts_dir / "surface_receipt.json",
        lock_receipt=receipts_dir / "lock_receipt.json",
        export_receipt=receipts_dir / "export_receipt.json",
        final_dem_receipt=receipts_dir / "final_dem_receipt.json",
        bundle_manifest=manifests_dir / "linear_pipeline_manifest.json",
        canonical_solve_identity_manifest=manifests_dir / "canonical_solve_identity.json",
        canonical_stage_identity_manifest=manifests_dir / "canonical_stage_identity.json",
        canonical_cache_consistency_receipt=manifests_dir / "canonical_cache_consistency.json",
        adjacent_aoi_verification_receipt=manifests_dir / "adjacent_aoi_verification.json",
        river_verification_summary=manifests_dir / "river_verification_summary.json",
        final_dem_writer_receipt=manifests_dir / "final_dem_writer_receipt.json",
        receipt_purpose_manifest=manifests_dir / "receipt_purpose_manifest.json",
        canonical_system_identity_receipt=receipts_dir / "canonical_system_identity_receipt.json",
        trace_summary=manifests_dir / "linear_trace_summary.json",
    )


__all__ = ["RiverWorkflowPaths", "build_river_workflow_paths"]
