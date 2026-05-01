from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RiverV2Paths:
    root: Path
    support_dir: Path
    stage_products_dir: Path
    stage_products_overview: Path
    stage_trace: Path
    resolved_inputs_manifest: Path
    preflight_receipt: Path
    system_support_decision: Path
    canonical_solve_domain_json: Path
    canonical_solve_network: Path
    downstream_support_search_json: Path
    downstream_support_search_gpkg: Path
    solve_support_network: Path
    solve_support_coverage: Path
    canonical_solve_channel_mask: Path
    canonical_solve_channel_mask_summary: Path
    canonical_solve_support_coverage: Path
    canonical_authoritative_sampling_source_raster: Path
    canonical_authoritative_base: Path
    canonical_inputs_receipt_json: Path
    solve_channel_mask: Path
    solve_channel_mask_summary: Path
    river_centerline_points: Path
    river_centerline_points_export: Path
    river_centerline_points_receipt: Path
    centerline_wse_support_points: Path
    centerline_wse_trend_points: Path
    centerline_wse_pre_smooth_points: Path
    centerline_wse_stage_diagnostics: Path
    centerline_wse_stage_summary: Path
    centerline_wse_proxy_points: Path
    centerline_wse_support_points_receipt: Path
    centerline_wse_proxy_points_receipt: Path
    authoritative_sampling_support: Path
    authoritative_sampling_support_receipt: Path
    authoritative_sampling_support_diagnostics: Path
    centerline_authoritative_bed_points: Path
    centerline_authoritative_bed_diagnostics: Path
    centerline_authoritative_bed_points_receipt: Path
    centerline_observed_offset_points: Path
    centerline_observed_offset_points_receipt: Path
    component_offset_transfer_priors: Path
    component_offset_transfer_priors_receipt: Path
    centerline_offset_modeled_points: Path
    centerline_offset_modeled_points_receipt: Path
    component_stream_summary: Path
    component_stream_summary_receipt: Path
    river_centerline_bed_backbone_points: Path
    river_centerline_bed_backbone_points_receipt: Path
    river_system_bed_backbone_points: Path
    river_system_bed_backbone_points_receipt: Path
    river_centerline_bed_backbone_dense_points: Path
    river_centerline_bed_backbone_dense_points_receipt: Path
    river_centerline_bed_backbone_raw_points: Path
    river_centerline_bed_backbone_barrier_points: Path
    river_centerline_bed_backbone_barrier_audit: Path
    river_primary_surface_domain: Path
    primary_surface_support_receipt: Path
    river_primary_surface_solve_domain: Path
    river_primary_surface_solve_domain_receipt: Path
    river_primary_surface: Path
    river_primary_surface_receipt: Path
    river_primary_surface_authoritative_applied_solve_domain: Path
    river_primary_surface_authoritative_applied_solve_domain_receipt: Path
    river_primary_surface_authoritative_applied: Path
    river_primary_surface_authoritative_applied_receipt: Path
    river_export_subset_receipt: Path
    river_primary_surface_authoritative_lock_diff: Path
    river_primary_surface_trusted_export: Path
    river_trusted_interior: Path
    river_admissibility: Path
    river_guidance_weight: Path
    river_trusted_interior_summary: Path
    pipeline_summary: Path
    pass1_summary: Path
    pass2_summary: Path
    pass3_summary: Path
    pass4_summary: Path
    bank_guidance_summary: Path
    wse_support_source_raster: Path
    bank_wse_profile_summary: Path
    bank_wse_edge_guidance: Path
    bank_influence: Path
    authoritative_sampling_source_raster: Path
    authoritative_sampling_source_summary: Path
    authoritative_sampling_raster: Path
    authoritative_measured_only_projected: Path
    authoritative_bed_support_points: Path
    authoritative_sampling_summary: Path


def build_river_v2_paths(out_dir: str | Path) -> RiverV2Paths:
    root = Path(out_dir)
    support_dir = root / "_support"
    stage_products_dir = root / "stage_products"
    return RiverV2Paths(
        root=root,
        support_dir=support_dir,
        stage_products_dir=stage_products_dir,
        stage_products_overview=stage_products_dir / "PIPELINE_OVERVIEW.json",
        stage_trace=stage_products_dir / "PIPELINE_STAGE_TRACE.json",
        resolved_inputs_manifest=root / "resolved_inputs.json",
        preflight_receipt=root / "preflight_receipt.json",
        system_support_decision=root / "system_support_decision.json",
        canonical_solve_domain_json=root / "canonical_solve_domain.json",
        canonical_solve_network=root / "canonical_solve_network.gpkg",
        downstream_support_search_json=root / "downstream_support_search.json",
        downstream_support_search_gpkg=root / "downstream_support_search.gpkg",
        solve_support_network=root / "solve_support_network.gpkg",
        solve_support_coverage=root / "solve_support_coverage.gpkg",
        canonical_solve_channel_mask=root / "canonical_solve_channel_mask.tif",
        canonical_solve_channel_mask_summary=root / "canonical_solve_channel_mask_summary.json",
        canonical_solve_support_coverage=root / "canonical_solve_support_coverage.gpkg",
        canonical_authoritative_sampling_source_raster=support_dir / "canonical_authoritative_sampling_source_raster.tif",
        canonical_authoritative_base=root / "canonical_authoritative_base.tif",
        canonical_inputs_receipt_json=root / "canonical_inputs_receipt.json",
        solve_channel_mask=root / "solve_channel_mask.tif",
        solve_channel_mask_summary=root / "solve_channel_mask_summary.json",
        river_centerline_points=root / "river_centerline_points.gpkg",
        river_centerline_points_export=root / "river_centerline_points_export.gpkg",
        river_centerline_points_receipt=root / "river_centerline_points_receipt.json",
        centerline_wse_support_points=root / "centerline_wse_support_points.gpkg",
        centerline_wse_trend_points=root / "centerline_wse_trend_points.gpkg",
        centerline_wse_pre_smooth_points=root / "centerline_wse_pre_smooth_points.gpkg",
        centerline_wse_stage_diagnostics=root / "centerline_wse_stage_diagnostics.json",
        centerline_wse_stage_summary=root / "centerline_wse_stage_summary.txt",
        centerline_wse_proxy_points=root / "centerline_wse_proxy_points.gpkg",
        centerline_wse_support_points_receipt=root / "centerline_wse_support_points_receipt.json",
        centerline_wse_proxy_points_receipt=root / "centerline_wse_proxy_points_receipt.json",
        authoritative_sampling_support=root / "authoritative_sampling_support.tif",
        authoritative_sampling_support_receipt=root / "authoritative_sampling_support_receipt.json",
        authoritative_sampling_support_diagnostics=root / "authoritative_sampling_support_diagnostics.json",
        centerline_authoritative_bed_points=root / "centerline_authoritative_bed_points.gpkg",
        centerline_authoritative_bed_diagnostics=root / "centerline_authoritative_bed_diagnostics.json",
        centerline_authoritative_bed_points_receipt=root / "centerline_authoritative_bed_points_receipt.json",
        centerline_observed_offset_points=root / "centerline_observed_offset_points.gpkg",
        centerline_observed_offset_points_receipt=root / "centerline_observed_offset_points_receipt.json",
        component_offset_transfer_priors=root / "component_offset_transfer_priors.gpkg",
        component_offset_transfer_priors_receipt=root / "component_offset_transfer_priors_receipt.json",
        centerline_offset_modeled_points=root / "centerline_offset_modeled_points.gpkg",
        centerline_offset_modeled_points_receipt=root / "centerline_offset_modeled_points_receipt.json",
        component_stream_summary=root / "river_v2_component_stream_summary.gpkg",
        component_stream_summary_receipt=root / "river_v2_component_stream_summary_receipt.json",
        river_centerline_bed_backbone_points=root / "river_centerline_bed_backbone_points.gpkg",
        river_centerline_bed_backbone_points_receipt=root / "river_centerline_bed_backbone_points_receipt.json",
        river_system_bed_backbone_points=root / "river_system_bed_backbone_points.gpkg",
        river_system_bed_backbone_points_receipt=root / "river_system_bed_backbone_points_receipt.json",
        river_centerline_bed_backbone_dense_points=root / "river_centerline_bed_backbone_dense_points.gpkg",
        river_centerline_bed_backbone_dense_points_receipt=root / "river_centerline_bed_backbone_dense_points_receipt.json",
        river_centerline_bed_backbone_raw_points=root / "river_centerline_bed_backbone_raw_points.gpkg",
        river_centerline_bed_backbone_barrier_points=root / "river_centerline_bed_backbone_barrier_points.gpkg",
        river_centerline_bed_backbone_barrier_audit=root / "river_centerline_bed_backbone_barrier_audit.json",
        river_primary_surface_domain=root / "river_primary_surface_domain.tif",
        primary_surface_support_receipt=root / "river_primary_surface_support_receipt.json",
        river_primary_surface_solve_domain=root / "river_primary_surface_solve_domain.tif",
        river_primary_surface_solve_domain_receipt=root / "river_primary_surface_solve_domain_receipt.json",
        river_primary_surface=root / "river_primary_surface.tif",
        river_primary_surface_receipt=root / "river_primary_surface_receipt.json",
        river_primary_surface_authoritative_applied_solve_domain=root / "river_primary_surface_authoritative_applied_solve_domain.tif",
        river_primary_surface_authoritative_applied_solve_domain_receipt=root / "river_primary_surface_authoritative_applied_solve_domain_receipt.json",
        river_primary_surface_authoritative_applied=root / "river_primary_surface_authoritative_applied.tif",
        river_primary_surface_authoritative_applied_receipt=root / "river_primary_surface_authoritative_applied_receipt.json",
        river_export_subset_receipt=root / "river_export_subset_receipt.json",
        river_primary_surface_authoritative_lock_diff=root / "river_primary_surface_authoritative_applied_diff_before_overwrite.tif",
        river_primary_surface_trusted_export=root / "river_primary_surface_trusted_export.tif",
        river_trusted_interior=root / "river_trusted_interior.tif",
        river_admissibility=root / "river_admissibility.tif",
        river_guidance_weight=root / "river_guidance_weight.tif",
        river_trusted_interior_summary=root / "river_trusted_interior_summary.json",
        pipeline_summary=root / "river_v2_summary.json",
        pass1_summary=root / "river_v2_pass1_summary.json",
        pass2_summary=root / "river_v2_pass2_summary.json",
        pass3_summary=root / "river_v2_pass3_summary.json",
        pass4_summary=root / "river_v2_pass4_summary.json",
        bank_guidance_summary=support_dir / "river_bank_guidance_summary.json",
        wse_support_source_raster=support_dir / "river_wse_support_source_raster.tif",
        bank_wse_profile_summary=support_dir / "river_bank_wse_proxy_profile_summary.csv",
        bank_wse_edge_guidance=support_dir / "river_bank_wse_edge_guidance.tif",
        bank_influence=support_dir / "river_bank_influence.tif",
        authoritative_sampling_source_raster=support_dir / "authoritative_sampling_source_raster.tif",
        authoritative_sampling_source_summary=support_dir / "authoritative_sampling_source_summary.json",
        authoritative_sampling_raster=support_dir / "authoritative_base_projected_for_river_sampling.tif",
        authoritative_measured_only_projected=support_dir / "authoritative_measured_only_projected.tif",
        authoritative_bed_support_points=support_dir / "authoritative_bed_support_points.csv",
        authoritative_sampling_summary=support_dir / "authoritative_measured_only_projected_summary.json",
    )
