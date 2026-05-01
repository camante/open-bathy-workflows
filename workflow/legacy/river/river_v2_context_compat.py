from __future__ import annotations

from pathlib import Path

from legacy.river.river_v2_context import RiverV2Context


def apply_direct_context_compatibility_defaults(ctx: RiverV2Context) -> RiverV2Context:
    """Apply narrow defaults for direct stage/unit-test contexts.

    Production River v2 runs should not depend on these fallbacks. They exist only
    for isolated tests or developer-constructed contexts that bypass bathy_main.py.
    """
    if ctx.export_channel_mask_path is None and ctx.export_aoi is None and ctx.channel_mask_path is not None:
        ctx.export_channel_mask_path = Path(ctx.channel_mask_path)

    if ctx.canonical_network_gpkg is None and ctx.network_gpkg is not None:
        ctx.canonical_network_gpkg = Path(ctx.network_gpkg)
    if ctx.canonical_channel_mask_path is None and ctx.channel_mask_path is not None:
        ctx.canonical_channel_mask_path = Path(ctx.channel_mask_path)
    if ctx.canonical_authoritative_base_path is None and ctx.authoritative_base_path is not None:
        ctx.canonical_authoritative_base_path = Path(ctx.authoritative_base_path)
    if ctx.canonical_authoritative_sampling_source_raster_path is None and ctx.authoritative_sampling_source_raster_path is not None:
        ctx.canonical_authoritative_sampling_source_raster_path = Path(ctx.authoritative_sampling_source_raster_path)
    if ctx.canonical_support_coverage_path is None and ctx.authoritative_support_coverage_path is not None:
        ctx.canonical_support_coverage_path = Path(ctx.authoritative_support_coverage_path)

    trusted_mode = str(ctx.trusted_support_mode or "").strip().lower()
    if not ctx.support_policy_source:
        if str(ctx.authoritative_dem_mode or "mixed_requires_metadata").strip().lower() == "measured_only" and ctx.authoritative_base_path is not None:
            ctx.trusted_support_mode = "all_finite_cells_trusted"
            ctx.support_policy_source = "direct_context_measured_only_default"
        elif ctx.trusted_support_artifact_path is not None or ctx.authoritative_support_coverage_path is not None:
            if ctx.trusted_support_artifact_path is None and ctx.authoritative_support_coverage_path is not None:
                ctx.trusted_support_artifact_path = Path(ctx.authoritative_support_coverage_path)
            ctx.trusted_support_mode = "metadata_proven_only"
            ctx.support_policy_source = "direct_context_metadata_support_default"
        elif ctx.authoritative_base_path is not None:
            ctx.trusted_support_mode = "all_finite_cells_trusted"
            ctx.support_policy_source = "direct_context_compatibility_default"
        else:
            ctx.trusted_support_mode = trusted_mode or "low_support_no_trusted_support"
            ctx.support_policy_source = "direct_context_low_support_default"
    elif ctx.trusted_support_mode == "metadata_proven_only" and ctx.trusted_support_artifact_path is None and ctx.authoritative_support_coverage_path is not None:
        ctx.trusted_support_artifact_path = Path(ctx.authoritative_support_coverage_path)
    return ctx


__all__ = ["apply_direct_context_compatibility_defaults"]
