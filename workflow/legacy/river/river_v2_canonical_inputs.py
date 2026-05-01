from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from legacy.river.river_v2_authoritative_sampling_source import materialize_authoritative_sampling_source_for_solve_aoi
from legacy.river.river_v2_context import RiverV2Context
from legacy.river.river_v2_solve_domain import materialize_solve_channel_mask_from_network
from authoritative_guidance import build_projected_authoritative_raster, build_projected_authoritative_sampling_raster


def _rewrite_local_artifact_from_source(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        rel = os.path.relpath(source, destination.parent)
        destination.symlink_to(rel)
    except Exception:
        shutil.copy2(source, destination)


def _existing_path(value: Any) -> Path | None:
    if value in (None, '', False):
        return None
    path = Path(value)
    return path if path.exists() else None


def _resolve_solve_domain_projection(ctx: RiverV2Context) -> tuple[str, float] | None:
    river_dem = _existing_path(ctx.river_dem_path)
    if river_dem is None:
        return None
    import rasterio

    with rasterio.open(river_dem) as ds:
        if ds.crs is None:
            return None
        res = abs(float(ds.transform.a)) if ds.transform is not None else None
        if res is None or not res > 0:
            return None
        return ds.crs.to_string(), float(res)


def _materialize_solve_domain_explicit_raster(
    ctx: RiverV2Context,
    *,
    source: Path,
    destination: Path,
    solve_aoi: str,
    logger: logging.Logger | None,
    mode: str,
) -> Path:
    grid = _resolve_solve_domain_projection(ctx)
    if not solve_aoi or grid is None:
        _rewrite_local_artifact_from_source(source=source, destination=destination)
        return destination if destination.exists() else source
    dst_crs, res_m = grid
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        if mode == 'sampling_source':
            build_projected_authoritative_sampling_raster(
                source,
                destination,
                aoi=solve_aoi,
                dst_crs=str(dst_crs),
                res_m=float(res_m),
                logger=logger,
            )
        elif mode == 'lock_base':
            build_projected_authoritative_raster(
                source,
                destination,
                aoi=solve_aoi,
                dst_crs=str(dst_crs),
                res_m=float(res_m),
                logger=logger,
                resampling_method='near',
            )
        else:
            raise ValueError(f'unknown_solve_domain_raster_mode:{mode}')
    except ValueError:
        _rewrite_local_artifact_from_source(source=source, destination=destination)
    return destination if destination.exists() else source


def materialize_canonical_authoritative_sampling_source(ctx: RiverV2Context, *, logger: logging.Logger | None = None) -> Path | None:
    solve_aoi = str(ctx.canonical_solve_aoi or ctx.export_aoi or getattr(ctx.cfg, 'aoi', '') or '')
    source = materialize_authoritative_sampling_source_for_solve_aoi(ctx, solve_aoi=solve_aoi, logger=logger)
    if source is None:
        ctx.canonical_authoritative_sampling_source_raster_path = None
        return None
    source = Path(source)
    out_path = ctx.paths.canonical_authoritative_sampling_source_raster
    ctx.canonical_authoritative_sampling_source_raster_path = _materialize_solve_domain_explicit_raster(
        ctx,
        source=source,
        destination=out_path,
        solve_aoi=solve_aoi,
        logger=logger,
        mode='sampling_source',
    )
    return ctx.canonical_authoritative_sampling_source_raster_path


def materialize_canonical_authoritative_base(ctx: RiverV2Context, *, logger: logging.Logger | None = None) -> Path | None:
    solve_aoi = str(ctx.canonical_solve_aoi or ctx.export_aoi or getattr(ctx.cfg, 'aoi', '') or '')
    source = (
        _existing_path(getattr(ctx, 'solve_aoi_authoritative_base_path', None))
        or _existing_path(ctx.authoritative_base_path)
        or _existing_path(ctx.aligned_authoritative_base_path)
        or _existing_path(ctx.canonical_authoritative_sampling_source_raster_path)
    )
    if source is None:
        ctx.canonical_authoritative_base_path = None
        return None
    out_path = ctx.paths.canonical_authoritative_base
    ctx.canonical_authoritative_base_path = _materialize_solve_domain_explicit_raster(
        ctx,
        source=source,
        destination=out_path,
        solve_aoi=solve_aoi,
        logger=logger,
        mode='lock_base',
    )
    return ctx.canonical_authoritative_base_path


def materialize_canonical_support_coverage(ctx: RiverV2Context) -> Path | None:
    source = (
        _existing_path(getattr(ctx, 'solve_aoi_authoritative_support_coverage_path', None))
        or _existing_path(ctx.trusted_support_artifact_path)
        or _existing_path(ctx.authoritative_support_coverage_path)
    )
    if source is None:
        ctx.canonical_support_coverage_path = None
        return None
    out_path = ctx.paths.canonical_solve_support_coverage
    _rewrite_local_artifact_from_source(source=source, destination=out_path)
    ctx.canonical_support_coverage_path = out_path if out_path.exists() else source
    if ctx.trusted_support_artifact_path is None:
        ctx.trusted_support_artifact_path = ctx.canonical_support_coverage_path
    return ctx.canonical_support_coverage_path


def materialize_canonical_solve_channel_mask(ctx: RiverV2Context) -> Path:
    template = (
        _existing_path(ctx.canonical_authoritative_sampling_source_raster_path)
        or _existing_path(ctx.canonical_authoritative_base_path)
        or _existing_path(ctx.river_dem_path)
        or _existing_path(ctx.authoritative_base_path)
    )
    if template is None:
        raise RuntimeError('river_v2_canonical_channel_mask_missing_template_raster')
    mask_path, _ = materialize_solve_channel_mask_from_network(
        network_gpkg=Path(ctx.canonical_network_gpkg or ctx.network_gpkg),
        template_raster_path=template,
        out_path=ctx.paths.canonical_solve_channel_mask,
        summary_path=ctx.paths.canonical_solve_channel_mask_summary,
    )
    ctx.canonical_channel_mask_path = Path(mask_path)
    return ctx.canonical_channel_mask_path


def _write_canonical_inputs_receipt(ctx: RiverV2Context) -> Path:
    payload = {
        'export_aoi': str(ctx.export_aoi or ''),
        'canonical_solve_aoi': str(ctx.canonical_solve_aoi or ''),
        'canonical_system_id': str(ctx.canonical_system_id or ''),
        'canonical_network_gpkg': str(ctx.canonical_network_gpkg) if ctx.canonical_network_gpkg is not None else None,
        'canonical_support_coverage_path': str(ctx.canonical_support_coverage_path) if ctx.canonical_support_coverage_path is not None else None,
        'canonical_support_empty': ctx.canonical_support_coverage_path is None,
        'canonical_authoritative_sampling_source_raster_path': str(ctx.canonical_authoritative_sampling_source_raster_path) if ctx.canonical_authoritative_sampling_source_raster_path is not None else None,
        'canonical_authoritative_base_path': str(ctx.canonical_authoritative_base_path) if ctx.canonical_authoritative_base_path is not None else None,
        'solve_aoi_authoritative_base_path': str(ctx.solve_aoi_authoritative_base_path) if getattr(ctx, 'solve_aoi_authoritative_base_path', None) is not None else None,
        'solve_aoi_authoritative_support_coverage_path': str(ctx.solve_aoi_authoritative_support_coverage_path) if getattr(ctx, 'solve_aoi_authoritative_support_coverage_path', None) is not None else None,
        'canonical_channel_mask_path': str(ctx.canonical_channel_mask_path) if ctx.canonical_channel_mask_path is not None else None,
    }
    path = ctx.paths.canonical_inputs_receipt_json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    ctx.canonical_inputs_receipt_path = path
    return path


def materialize_canonical_river_v2_inputs(ctx: RiverV2Context, *, logger: logging.Logger | None = None) -> RiverV2Context:
    materialize_canonical_authoritative_sampling_source(ctx, logger=logger)
    materialize_canonical_authoritative_base(ctx, logger=logger)
    materialize_canonical_support_coverage(ctx)
    materialize_canonical_solve_channel_mask(ctx)
    _write_canonical_inputs_receipt(ctx)
    return ctx


__all__ = [
    'materialize_canonical_river_v2_inputs',
    'materialize_canonical_authoritative_sampling_source',
    'materialize_canonical_authoritative_base',
    'materialize_canonical_support_coverage',
    'materialize_canonical_solve_channel_mask',
]
