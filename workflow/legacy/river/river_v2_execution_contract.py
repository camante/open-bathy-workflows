from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_resolved_inputs import ResolvedRiverV2Inputs


@dataclass(frozen=True)
class RiverV2ExecutionContract:
    solve_network_gpkg: Path
    solve_channel_mask_path: Path
    solve_authoritative_lock_base_path: Path | None
    solve_authoritative_sampling_source_raster_path: Path | None
    river_dem_path: Path | None
    authoritative_sampling_raster_path: Path | None
    authoritative_bed_support_points_path: Path | None
    centerline_spacing_m: float
    min_stream_order: int
    wse_support_source_raster_path: Path | None
    wse_support_source_contract: str | None
    export_channel_mask_path: Path | None
    export_aoi: str | None
    canonical_solve_aoi: str
    support_status: str
    canonical_system_id: str | None


def build_river_v2_execution_contract(ctx: RiverV2Context, inputs: ResolvedRiverV2Inputs) -> RiverV2ExecutionContract:
    solve_network = inputs.solve_network_gpkg
    solve_mask = inputs.solve_channel_mask_path
    solve_aoi = str(inputs.solve_aoi or getattr(ctx, "canonical_solve_aoi", None) or "").strip()
    if solve_network is None:
        raise RuntimeError('river_v2_execution_contract_missing_solve_network_gpkg')
    if solve_mask is None:
        raise RuntimeError('river_v2_execution_contract_missing_solve_channel_mask_path')
    if not solve_aoi:
        raise RuntimeError('river_v2_execution_contract_missing_canonical_solve_aoi')
    return RiverV2ExecutionContract(
        solve_network_gpkg=Path(solve_network),
        solve_channel_mask_path=Path(solve_mask),
        solve_authoritative_lock_base_path=Path(inputs.solve_authoritative_lock_base_path) if inputs.solve_authoritative_lock_base_path is not None else None,
        solve_authoritative_sampling_source_raster_path=Path(inputs.solve_authoritative_sampling_source_raster_path) if inputs.solve_authoritative_sampling_source_raster_path is not None else None,
        river_dem_path=Path(inputs.river_dem_path) if inputs.river_dem_path is not None else None,
        authoritative_sampling_raster_path=Path(inputs.authoritative_sampling_raster_path) if inputs.authoritative_sampling_raster_path is not None else None,
        authoritative_bed_support_points_path=Path(inputs.authoritative_bed_support_points_path) if inputs.authoritative_bed_support_points_path is not None else None,
        centerline_spacing_m=float(inputs.centerline_spacing_m),
        min_stream_order=int(inputs.min_stream_order),
        wse_support_source_raster_path=Path(inputs.wse_support_source_raster_path) if inputs.wse_support_source_raster_path is not None else None,
        wse_support_source_contract=inputs.wse_support_source_contract,
        export_channel_mask_path=Path(inputs.export_channel_mask_path) if inputs.export_channel_mask_path is not None else None,
        export_aoi=str(getattr(ctx, 'export_aoi', None) or '').strip() or None,
        canonical_solve_aoi=solve_aoi,
        support_status=str(getattr(ctx, 'system_support_status', 'unknown') or 'unknown'),
        canonical_system_id=str(getattr(ctx, 'canonical_system_id', None)) if getattr(ctx, 'canonical_system_id', None) is not None else None,
    )
