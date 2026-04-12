from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy import ndimage as ndi

from cache_utils import artifact_cache_key, canonical_json, fingerprint_raster_grid
from river_masking import clip_channel_mask_for_estuary, ensure_waffles_coastline_mask

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuidanceDomainPaths:
    cache_dir: Path
    run_dir: Path
    review_dir: Path
    manifest_json: Path
    ocean_mask: Path
    ocean_connectivity_mask: Path
    with_nhd_water_mask: Path
    river_water_support_mask: Path
    river_channel_mask: Path
    effective_water_mask: Path
    corridor_mask: Path
    nhdarea_mask: Path
    open_water_mask: Path
    mainstem_mask: Path
    estuary_clip_mask: Path
    estuary_transition_mask: Path
    nhd_sea_ocean_mask: Path
    river_guidance_domain_mask: Path
    sdb_guidance_domain_mask: Path
    river_domain_policy_json: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "cache_dir": str(self.cache_dir),
            "run_dir": str(self.run_dir),
            "review_dir": str(self.review_dir),
            "manifest_json": str(self.manifest_json),
            "ocean_mask": str(self.ocean_mask),
            "ocean_connectivity_mask": str(self.ocean_connectivity_mask),
            "with_nhd_water_mask": str(self.with_nhd_water_mask),
            "river_water_support_mask": str(self.river_water_support_mask),
            "river_channel_mask": str(self.river_channel_mask),
            "effective_water_mask": str(self.effective_water_mask),
            "corridor_mask": str(self.corridor_mask),
            "nhdarea_mask": str(self.nhdarea_mask),
            "open_water_mask": str(self.open_water_mask),
            "mainstem_mask": str(self.mainstem_mask),
            "estuary_clip_mask": str(self.estuary_clip_mask),
            "estuary_transition_mask": str(self.estuary_transition_mask),
            "nhd_sea_ocean_mask": str(self.nhd_sea_ocean_mask),
            "river_guidance_domain_mask": str(self.river_guidance_domain_mask),
            "sdb_guidance_domain_mask": str(self.sdb_guidance_domain_mask),
            "river_domain_policy_json": str(self.river_domain_policy_json),
        }




def make_guidance_domain_cfg(source: Any, **overrides: Any) -> Any:
    """Return a minimal config object for shared guidance-domain planning.

    This centralizes the default domain-policy values so bathy_main,
    guidance_domains_main, and direct sdb_main calls all resolve the same
    planning contract unless the caller overrides it explicitly.
    """
    values = {
        "aoi": getattr(source, "aoi", None),
        "cache_root": Path(getattr(source, "cache_root")),
        "waffles_inc_arcsec": float(getattr(source, "waffles_inc_arcsec", 1.0) or 1.0),
        "river_channel_buffer_m": float(getattr(source, "river_channel_buffer_m", 400.0) or 400.0),
        "river_max_channel_width_m": float(getattr(source, "river_max_channel_width_m", 600.0) or 600.0),
        "river_mainstem_method": str(getattr(source, "river_mainstem_method", "dominant_trunk") or "dominant_trunk"),
        "river_mainstem_solve_layer": str(getattr(source, "river_mainstem_solve_layer", "auto") or "auto"),
        "river_mainstem_min_order": int(getattr(source, "river_mainstem_min_order", 5) or 5),
        "river_max_mainstem_width_m": float(getattr(source, "river_max_mainstem_width_m", 2500.0) or 2500.0),
        "river_channel_source": str(getattr(source, "river_channel_source", "auto") or "auto"),
        "river_use_nhdarea": bool(getattr(source, "river_use_nhdarea", True)),
        "river_nhdarea_layer": str(getattr(source, "river_nhdarea_layer", "nhdarea_clip") or "nhdarea_clip"),
        "river_nhdarea_allow_ftype": str(getattr(source, "river_nhdarea_allow_ftype", "460") or "460"),
        "river_nhdarea_allow_fcode": getattr(source, "river_nhdarea_allow_fcode", None),
        "river_ocean_keep_dist_m": float(getattr(source, "river_ocean_keep_dist_m", 0.0) or 0.0),
        "estuary_width_ratio_thresh": float(getattr(source, "estuary_width_ratio_thresh", 3.0) or 3.0),
        "estuary_transition_m": float(getattr(source, "estuary_transition_m", 500.0) or 500.0),
        "estuary_connect_dist_m": float(getattr(source, "estuary_connect_dist_m", 200.0) or 200.0),
    }
    values.update(overrides)
    if values.get("aoi") is None:
        raise ValueError("Guidance-domain config requires an AOI.")
    return type("GuidanceDomainConfig", (), values)()

REQUIRED_KEYS = (
    "ocean_mask",
    "ocean_connectivity_mask",
    "with_nhd_water_mask",
    "river_water_support_mask",
    "river_channel_mask",
    "effective_water_mask",
    "corridor_mask",
    "open_water_mask",
    "mainstem_mask",
    "estuary_clip_mask",
    "estuary_transition_mask",
    "nhd_sea_ocean_mask",
    "river_guidance_domain_mask",
    "sdb_guidance_domain_mask",
    "river_domain_policy_json",
)

OPTIONAL_DEBUG_KEYS = (
    "diag_01_water_footprint",
    "diag_02_effective_water_mask",
    "diag_03_corridor_mask",
    "diag_04_channel_candidate_water_and_corridor",
    "diag_05_structure_seed_nhdarea_or_mainstem",
    "diag_06_component_labels",
    "diag_07_component_accept_mask",
    "diag_08_channel_after_component_acceptance",
    "diag_09_channel_after_morphology",
    "diag_10_river_channel_mask_final",
    "diag_construction_receipt_json",
)


def _stage(src: Path, dst: Path, *, refresh: bool = True) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if refresh and (dst.exists() or dst.is_symlink()):
        dst.unlink()
    elif dst.exists() or dst.is_symlink():
        return dst
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _expected_paths(base_dir: Path) -> Dict[str, Path]:
    return {
        "manifest_json": base_dir / "guidance_domains_manifest.json",
        "ocean_mask": base_dir / "waffles_ocean_only_mask.tif",
        "ocean_connectivity_mask": base_dir / "waffles_ocean_connectivity_mask.tif",
        "with_nhd_water_mask": base_dir / "waffles_with_nhd_water_mask.tif",
        "river_water_support_mask": base_dir / "river_water_support_mask.tif",
        "river_channel_mask": base_dir / "river_channel_mask.tif",
        "effective_water_mask": base_dir / "effective_water_mask.tif",
        "corridor_mask": base_dir / "corridor_mask.tif",
        "nhdarea_mask": base_dir / "nhdarea_mask.tif",
        "open_water_mask": base_dir / "open_water_mask.tif",
        "mainstem_mask": base_dir / "mainstem_mask.tif",
        "estuary_clip_mask": base_dir / "estuary_clip_mask.tif",
        "estuary_transition_mask": base_dir / "estuary_transition_mask.tif",
        "nhd_sea_ocean_mask": base_dir / "nhd_sea_ocean_mask.tif",
        "river_guidance_domain_mask": base_dir / "river_guidance_domain_mask.tif",
        "sdb_guidance_domain_mask": base_dir / "sdb_guidance_domain_mask.tif",
        "river_domain_policy_json": base_dir / "river_domain_policy.json",
        "diag_01_water_footprint": base_dir / "01_water_footprint.tif",
        "diag_02_effective_water_mask": base_dir / "02_effective_water_mask.tif",
        "diag_03_corridor_mask": base_dir / "03_corridor_mask.tif",
        "diag_04_channel_candidate_water_and_corridor": base_dir / "04_channel_candidate_water_and_corridor.tif",
        "diag_05_structure_seed_nhdarea_or_mainstem": base_dir / "05_structure_seed_nhdarea_or_mainstem.tif",
        "diag_06_component_labels": base_dir / "06_component_labels.tif",
        "diag_07_component_accept_mask": base_dir / "07_component_accept_mask.tif",
        "diag_08_channel_after_component_acceptance": base_dir / "08_channel_after_component_acceptance.tif",
        "diag_09_channel_after_morphology": base_dir / "09_channel_after_morphology.tif",
        "diag_10_river_channel_mask_final": base_dir / "10_river_channel_mask_final.tif",
        "diag_construction_receipt_json": base_dir / "river_channel_mask_construction_receipt.json",
    }


def _paths_from_dir(base_dir: Path) -> GuidanceDomainPaths:
    expected = _expected_paths(base_dir)
    return GuidanceDomainPaths(
        cache_dir=base_dir,
        run_dir=base_dir,
        review_dir=base_dir,
        manifest_json=expected["manifest_json"],
        ocean_mask=expected["ocean_mask"],
        ocean_connectivity_mask=expected["ocean_connectivity_mask"],
        with_nhd_water_mask=expected["with_nhd_water_mask"],
        river_water_support_mask=expected["river_water_support_mask"],
        river_channel_mask=expected["river_channel_mask"],
        effective_water_mask=expected["effective_water_mask"],
        corridor_mask=expected["corridor_mask"],
        nhdarea_mask=expected["nhdarea_mask"],
        open_water_mask=expected["open_water_mask"],
        mainstem_mask=expected["mainstem_mask"],
        estuary_clip_mask=expected["estuary_clip_mask"],
        estuary_transition_mask=expected["estuary_transition_mask"],
        nhd_sea_ocean_mask=expected["nhd_sea_ocean_mask"],
        river_guidance_domain_mask=expected["river_guidance_domain_mask"],
        sdb_guidance_domain_mask=expected["sdb_guidance_domain_mask"],
        river_domain_policy_json=expected["river_domain_policy_json"],
    )


def _manifest_complete(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    outputs = payload.get("outputs", {})
    if not isinstance(outputs, dict):
        return False
    for key in REQUIRED_KEYS:
        value = outputs.get(key)
        if not value:
            return False
        path = Path(str(value))
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.exists():
            return False
    return True


def _domain_params(cfg: Any, *, river_dem: Path, river_gpkg: Path) -> Dict[str, Any]:
    return {
        "aoi": str(cfg.aoi),
        "waffles_inc_arcsec": float(getattr(cfg, "waffles_inc_arcsec", 1.0) or 1.0),
        "river_dem_grid": fingerprint_raster_grid(river_dem),
        "river_network_gpkg": str(river_gpkg.resolve()),
        "river_channel_buffer_m": float(getattr(cfg, "river_channel_buffer_m", 400.0) or 400.0),
        "river_max_channel_width_m": float(getattr(cfg, "river_max_channel_width_m", 600.0) or 600.0),
        "river_mainstem_method": str(getattr(cfg, "river_mainstem_method", "dominant_trunk") or "dominant_trunk"),
        "river_mainstem_solve_layer": str(getattr(cfg, "river_mainstem_solve_layer", "auto") or "auto"),
        "river_mainstem_min_order": int(getattr(cfg, "river_mainstem_min_order", 5) or 5),
        "river_max_mainstem_width_m": float(getattr(cfg, "river_max_mainstem_width_m", 2500.0) or 2500.0),
        "river_channel_source": str(getattr(cfg, "river_channel_source", "auto") or "auto"),
        "river_use_nhdarea": bool(getattr(cfg, "river_use_nhdarea", True)),
        "river_nhdarea_layer": str(getattr(cfg, "river_nhdarea_layer", "nhdarea_clip") or "nhdarea_clip"),
        "river_nhdarea_allow_ftype": str(getattr(cfg, "river_nhdarea_allow_ftype", "460") or "460"),
        "river_nhdarea_allow_fcode": str(getattr(cfg, "river_nhdarea_allow_fcode", "") or ""),
        "river_ocean_keep_dist_m": float(getattr(cfg, "river_ocean_keep_dist_m", 0.0) or 0.0),
        "estuary_width_ratio_thresh": float(getattr(cfg, "estuary_width_ratio_thresh", 3.0) or 3.0),
        "estuary_transition_m": float(getattr(cfg, "estuary_transition_m", 500.0) or 500.0),
        "estuary_connect_dist_m": float(getattr(cfg, "estuary_connect_dist_m", 200.0) or 200.0),
        "river_guidance_bank_margin_m": float(getattr(cfg, "river_guidance_bank_margin_m", 3.0) or 3.0),
    }


def compute_guidance_domain_cache_key(cfg: Any, *, river_dem: Path, river_gpkg: Path) -> str:
    params = _domain_params(cfg, river_dem=river_dem, river_gpkg=river_gpkg)
    return artifact_cache_key(
        stage="guidance_domains",
        params=params,
        inputs={
            "river_gpkg": str(river_gpkg.resolve()),
            "river_dem": fingerprint_raster_grid(river_dem),
        },
        code_fp=f"{Path(__file__).name}:{int(Path(__file__).stat().st_mtime_ns)}",
    )


def _build_canonical_with_nhd_mask(ocean_mask_tif: Path, river_gpkg: Path, out_tif: Path, *, logger: logging.Logger) -> Path:
    ocean_mask_tif = Path(ocean_mask_tif)
    river_gpkg = Path(river_gpkg)
    out_tif = Path(out_tif)
    if not ocean_mask_tif.exists():
        raise RuntimeError(f"Ocean-only WAFFLES mask missing: {ocean_mask_tif}")
    if not river_gpkg.exists():
        raise RuntimeError(f"River network GeoPackage missing: {river_gpkg}")

    with rasterio.open(ocean_mask_tif) as ds:
        ocean = ds.read(1)
        profile = ds.profile.copy()
        transform = ds.transform
        crs = ds.crs
        shape = (ds.height, ds.width)
    ocean_water = ocean == 0

    available_layers = set(gpd.list_layers(river_gpkg).name.tolist()) if hasattr(gpd, "list_layers") else set()
    candidate_layers = ["nhdarea_clip"] if available_layers else ["nhdarea_clip"]
    shapes = []
    nonempty_layers = []
    half_cell = 0.5 * max(abs(float(transform.a)), abs(float(transform.e)))
    for layer in candidate_layers:
        try:
            gdf = gpd.read_file(river_gpkg, layer=layer)
        except Exception:
            LOG.debug("_build_canonical_with_nhd_mask: suppressed exception", exc_info=True)
            continue
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise RuntimeError(f"River network layer {layer} has no CRS: {river_gpkg}")
        if str(gdf.crs) != str(crs):
            gdf = gdf.to_crs(crs)
        gdf = gdf[gdf.geometry.notnull()].copy()
        if gdf.empty:
            continue
        poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        gdf = gdf.loc[poly_mask].copy()
        if gdf.empty:
            continue
        if np.isfinite(half_cell) and half_cell > 0.0:
            keep_index = []
            buffered_geoms = []
            for idx, geom in zip(gdf.index, gdf.geometry):
                if geom is None or geom.is_empty:
                    continue
                try:
                    buffered = geom.buffer(float(half_cell), cap_style=1, join_style=2)
                except Exception:
                    LOG.debug("_build_canonical_with_nhd_mask: suppressed exception", exc_info=True)
                    buffered = geom
                if buffered is not None and not buffered.is_empty:
                    keep_index.append(idx)
                    buffered_geoms.append(buffered)
            gdf = gdf.loc[keep_index].copy()
            gdf["geometry"] = buffered_geoms
            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
        if gdf.empty:
            continue
        nonempty_layers.append(layer)
        shapes.extend((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    if not shapes:
        raise RuntimeError(
            f"No usable polygonal inland-water layer found in {river_gpkg}. Required layer: nhdarea_clip"
        )

    inland = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    rook = ndi.generate_binary_structure(2, 1)
    queen = ndi.generate_binary_structure(2, 2)
    ocean_edge = ndi.binary_dilation(ocean_water, structure=rook) & ~ocean_water
    inland_edge = ndi.binary_dilation(inland, structure=rook) & ~inland
    bridge_cells = ocean_edge & inland_edge
    direct_touch_pixels = int(np.count_nonzero(ndi.binary_dilation(ocean_water, structure=rook) & inland))
    diagonal_touch_pixels = int(np.count_nonzero(ndi.binary_dilation(ocean_water, structure=queen) & inland))
    bridge_pixel_count = 0
    if direct_touch_pixels == 0 and diagonal_touch_pixels > 0:
        inland = inland | bridge_cells
        bridge_pixel_count = int(np.count_nonzero(bridge_cells))
        direct_touch_pixels = int(np.count_nonzero(ndi.binary_dilation(ocean_water, structure=rook) & inland))

    full_water = ocean_water | inland
    added_pixels = int(np.count_nonzero(full_water & ~ocean_water))
    passthrough_ocean_only = bool(added_pixels == 0)
    if passthrough_ocean_only:
        logger.info(
            "[DOMAIN] Canonical with-NHD mask adds zero inland-water pixels beyond ocean-only mask; using ocean-only mask as canonical water domain for this AOI."
        )
        full_water = ocean_water.copy()
    out = np.where(full_water, 0, 1).astype("uint8")
    profile.update(driver="GTiff", dtype="uint8", count=1, compress="DEFLATE", nodata=None)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    if int(profile.get("width", 0)) >= 16 and int(profile.get("height", 0)) >= 16:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    else:
        profile.update(tiled=False)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out, 1)

    diagnostics = {
        "ocean_only_mask": str(ocean_mask_tif),
        "river_gpkg": str(river_gpkg),
        "layers_used": nonempty_layers,
        "half_cell_buffer": float(half_cell),
        "added_inland_pixels": int(added_pixels),
        "passthrough_ocean_only": bool(passthrough_ocean_only),
        "direct_touch_pixels": int(direct_touch_pixels),
        "diagonal_touch_pixels": int(diagonal_touch_pixels),
        "bridge_pixel_count": int(bridge_pixel_count),
    }
    out_tif.with_name(out_tif.stem + "_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "[DOMAIN] Built canonical with-NHD water mask: %s (added_inland_pixels=%d passthrough_ocean_only=%s)",
        out_tif,
        added_pixels,
        str(bool(passthrough_ocean_only)).lower(),
    )
    return out_tif


def _build_template_aligned_river_water_mask(source_with_nhd_mask: Path, river_gpkg: Path, template_raster: Path, out_tif: Path, *, logger: logging.Logger) -> Path:
    """Build a river-domain water support mask directly on the river template grid.

    The river domain should be planned from a water mask that is valid on the same grid
    used to rasterize the corridor/mainstem products. We therefore reproject the shared
    WAFFLES with-NHD mask to the river template and explicitly union the prepared
    NHDArea river polygons from river_network.gpkg. This is not a fallback; it is the
    intended early-domain water support contract for river-domain construction.
    """
    with rasterio.open(template_raster) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        crs = ref.crs
        shape = (ref.height, ref.width)

    water_dst = np.ones(shape, dtype=np.uint8)
    with rasterio.open(source_with_nhd_mask) as src:
        src_nodata = src.nodata
        if src_nodata in (0, 1):
            src_nodata = None
        reproject(
            source=rasterio.band(src, 1),
            destination=water_dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=1,
            resampling=Resampling.nearest,
        )
    full_water = water_dst == 0

    try:
        nhd = gpd.read_file(river_gpkg, layer='nhdarea_clip')
    except Exception as exc:
        LOG.debug("_build_template_aligned_river_water_mask: suppressed exception", exc_info=True)
        raise RuntimeError(f"Failed to load nhdarea_clip from {river_gpkg}: {exc}") from exc
    if nhd is not None and not nhd.empty:
        if nhd.crs is None:
            raise RuntimeError(f"NHDArea layer has no CRS: {river_gpkg}:nhdarea_clip")
        if str(nhd.crs) != str(crs):
            nhd = nhd.to_crs(crs)
        nhd = nhd[nhd.geometry.notnull() & (~nhd.geometry.is_empty)].copy()
        nhd = nhd.loc[nhd.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if not nhd.empty:
            river_polys = rasterize(
                [(geom, 1) for geom in nhd.geometry if geom is not None and not geom.is_empty],
                out_shape=shape,
                transform=transform,
                fill=0,
                default_value=1,
                all_touched=True,
                dtype='uint8',
            ).astype(bool)
            full_water |= river_polys
            logger.info('[DOMAIN] River template water-support mask augmented with NHDArea polygons (pixels=%d)', int(river_polys.sum()))

    profile.pop('blockxsize', None)
    profile.pop('blockysize', None)
    profile.pop('tiled', None)
    profile.update(dtype='uint8', count=1, nodata=1, compress='deflate')
    arr = np.where(full_water, 0, 1).astype('uint8')
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(arr, 1)
    return out_tif


def _write_zero_mask_like(template_path: Path, out_path: Path) -> Path:
    with rasterio.open(template_path) as ds:
        profile = ds.profile.copy()
        arr = np.zeros((ds.height, ds.width), dtype=np.uint8)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    return out_path


def _clip_mainstem_mask(mainstem_mask_path: Path, estuary_clip_mask_path: Path, *, logger: logging.Logger) -> None:
    if not mainstem_mask_path.exists() or not estuary_clip_mask_path.exists():
        return
    with rasterio.open(estuary_clip_mask_path) as ec:
        estuary = ec.read(1)
    with rasterio.open(mainstem_mask_path) as ms:
        arr = ms.read(1)
        profile = ms.profile.copy()
    before = int(np.count_nonzero(arr > 0))
    arr[estuary > 0] = 0
    after = int(np.count_nonzero(arr > 0))
    if after == before:
        return
    profile.update(dtype="uint8", nodata=0)
    with rasterio.open(mainstem_mask_path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8), 1)
    logger.info("[DOMAIN] Clipped mainstem mask by estuary clip: %d -> %d pixels", before, after)


def _write_river_guidance_domain_mask(*, channel_mask_path: Path, out_path: Path, bank_margin_m: float, logger: logging.Logger, estuary_transition_mask_path: Optional[Path] = None) -> Dict[str, int | float]:
    with rasterio.open(channel_mask_path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
        transform = ds.transform
        crs = ds.crs
    channel = arr > 0
    transition_excluded = 0
    if estuary_transition_mask_path and Path(estuary_transition_mask_path).exists():
        try:
            transition = np.zeros_like(arr, dtype=np.uint8)
            with rasterio.open(estuary_transition_mask_path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=transition,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata,
                    dst_nodata=0,
                )
            transition_mask = transition > 0
            transition_excluded = int(np.count_nonzero(channel & transition_mask))
            if transition_excluded > 0:
                channel &= ~transition_mask
        except (OSError, ValueError, RuntimeError):
            logger.debug('[DOMAIN] Failed to apply estuary transition exclusion to river guidance domain', exc_info=True)
    before = int(np.count_nonzero(channel))
    if before <= 0:
        _write_zero_mask_like(channel_mask_path, out_path)
        return {"input_pixels": 0, "output_pixels": 0, "bank_margin_m": float(bank_margin_m), "components_preserved_by_core": 0, "transition_pixels_excluded": 0}

    safe_margin_m = max(float(bank_margin_m or 0.0), 0.0)
    if safe_margin_m <= 0.0:
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        profile.pop('tiled', None)
        profile.update(dtype='uint8', count=1, nodata=0, compress='deflate')
        out = channel.astype(np.uint8)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(out, 1)
        return {"input_pixels": before, "output_pixels": before, "bank_margin_m": 0.0, "effective_bank_margin_m": 0.0, "components_preserved_by_core": 0, "transition_pixels_excluded": int(transition_excluded)}

    px_size_x = abs(float(transform.a))
    px_size_y = abs(float(transform.e))
    if crs is not None and getattr(crs, 'is_geographic', False):
        mid_lat = 0.0
        try:
            mid_lat = float(transform.f + (profile['height'] * transform.e * 0.5))
        except Exception:
            LOG.debug("_write_river_guidance_domain_mask: suppressed exception", exc_info=True)
            LOG.debug("guidance_domains: suppressed exception", exc_info=True)
            mid_lat = 0.0
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = meters_per_deg_lat * max(np.cos(np.deg2rad(mid_lat)), 1e-6)
        px_size_m = max(px_size_x * meters_per_deg_lon, px_size_y * meters_per_deg_lat, 1e-6)
    else:
        px_size_m = max(px_size_x, px_size_y, 1e-6)

    from scipy import ndimage as ndi
    effective_margin_m = max(safe_margin_m + (0.5 * px_size_m), 1.05 * px_size_m)
    dist_m = ndi.distance_transform_edt(channel, sampling=px_size_m).astype(np.float32)
    core = channel & (dist_m > effective_margin_m)
    labels, ncomp = ndi.label(channel, structure=np.ones((3, 3), dtype=np.uint8))
    preserved = 0
    if ncomp > 0:
        for idx in range(1, ncomp + 1):
            comp = labels == idx
            if not np.any(comp):
                continue
            if np.any(core & comp):
                continue
            comp_dist = np.where(comp, dist_m, -np.inf)
            max_dist = float(np.nanmax(comp_dist))
            if not np.isfinite(max_dist) or max_dist <= 0.0:
                core |= comp
                preserved += 1
                continue
            ridge = comp & np.isclose(dist_m, max_dist)
            if not np.any(ridge):
                ridge = comp
            core |= ridge
            preserved += 1

    profile.pop('blockxsize', None)
    profile.pop('blockysize', None)
    profile.pop('tiled', None)
    profile.update(dtype='uint8', count=1, nodata=0, compress='deflate')
    out = core.astype(np.uint8)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(out, 1)
    after = int(np.count_nonzero(out))
    logger.info('[DOMAIN] Built river guidance domain with estuary-transition exclusion=%d px and bank exclusion margin %.1f m (effective %.2f m): %d -> %d pixels', transition_excluded, safe_margin_m, effective_margin_m, before, after)
    return {
        'input_pixels': before,
        'output_pixels': after,
        'bank_margin_m': float(safe_margin_m),
        'effective_bank_margin_m': float(effective_margin_m),
        'components_preserved_by_core': int(preserved),
        'transition_pixels_excluded': int(transition_excluded),
    }




def _detect_ftype_column(gdf: gpd.GeoDataFrame) -> str | None:
    for name in ("FType", "FTYPE", "ftype", "FeatureType", "FEATURETYPE", "featuretype"):
        if name in gdf.columns:
            return name
    return None


def _detect_fcode_column(gdf: gpd.GeoDataFrame) -> str | None:
    for name in ("FCode", "FCODE", "fcode", "FeatureCode", "FEATURECODE", "featurecode"):
        if name in gdf.columns:
            return name
    return None


def _detect_name_column(gdf: gpd.GeoDataFrame) -> str | None:
    for name in ("GNIS_NAME", "gnis_name", "Name", "NAME", "name"):
        if name in gdf.columns:
            return name
    return None


def _parse_numeric_code(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series.astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    except Exception:
        return pd.Series(np.nan, index=series.index)


def _write_nhd_sea_ocean_mask(*, template_path: Path, river_gpkg: Path, out_path: Path, logger: logging.Logger) -> Dict[str, int | str | None]:
    """Write a template-aligned coastal/ocean handoff mask from NHD polygons.

    Primary selectors:
      - Sea/Ocean: FCODE 44500 or FTYPE 445
    Supplemental estuary selector:
      - Bay/Inlet: FCODE 31200

    Bay/Inlet is included so the shared estuary/coastal handoff can bridge the
    river-to-ocean transition even when the open-ocean polygon alone does not
    fully cover the estuarine interface on the template grid.
    """
    layers = []
    try:
        import fiona
        layers = list(fiona.listlayers(str(river_gpkg)))
    except Exception:
        logger.debug('[DOMAIN] Failed listing river_gpkg layers for NHD coastal/ocean mask', exc_info=True)
    for layer_name in ('nhdarea_aoi', 'nhdarea_clip'):
        if layer_name not in layers:
            continue
        gdf = gpd.read_file(river_gpkg, layer=layer_name)
        if gdf.empty:
            continue
        ftype_col = _detect_ftype_column(gdf)
        fcode_col = _detect_fcode_column(gdf)
        name_col = _detect_name_column(gdf)
        if ftype_col is None and fcode_col is None and name_col is None:
            continue
        gdf = gdf[gdf.geometry.notnull()].copy()
        if gdf.empty:
            continue
        poly_mask = gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        gdf = gdf.loc[poly_mask].copy()
        if gdf.empty:
            continue
        ftype_num = _parse_numeric_code(gdf[ftype_col]) if ftype_col is not None else pd.Series(np.nan, index=gdf.index)
        fcode_num = _parse_numeric_code(gdf[fcode_col]) if fcode_col is not None else pd.Series(np.nan, index=gdf.index)
        name_ser = gdf[name_col].astype(str).str.lower() if name_col is not None else pd.Series('', index=gdf.index)

        ocean_mask = (fcode_num == 44500) | (ftype_num == 445)
        estuary_mask = (fcode_num == 31200)
        name_mask = name_ser.str.contains(r'\bocean\b|\bsea\b|\bbay\b|\binlet\b|\bestuary\b', regex=True, na=False)
        selected = gdf.loc[ocean_mask | estuary_mask | ((~(ocean_mask | estuary_mask)) & name_mask)].copy()
        if selected.empty:
            continue
        with rasterio.open(template_path) as ref:
            profile = ref.profile.copy()
            transform = ref.transform
            crs = ref.crs
            shape = (ref.height, ref.width)
        if selected.crs is None:
            raise RuntimeError(f'NHD coastal/ocean layer {layer_name} has no CRS: {river_gpkg}')
        if str(selected.crs) != str(crs):
            selected = selected.to_crs(crs)
        shapes = [(geom, 1) for geom in selected.geometry if geom is not None and not geom.is_empty]
        arr = np.zeros(shape, dtype=np.uint8)
        if shapes:
            arr = rasterize(shapes, out_shape=shape, transform=transform, fill=0, default_value=1, all_touched=False, dtype='uint8')
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        profile.pop('tiled', None)
        profile.update(dtype='uint8', count=1, nodata=0, compress='deflate')
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(arr.astype(np.uint8), 1)
        return {
            'layer': layer_name,
            'ftype_column': ftype_col,
            'fcode_column': fcode_col,
            'name_column': name_col,
            'polygon_count': int(len(selected)),
            'ocean_polygon_count': int(np.count_nonzero(ocean_mask.loc[selected.index])),
            'bay_inlet_polygon_count': int(np.count_nonzero(estuary_mask.loc[selected.index])),
            'pixel_count': int(np.count_nonzero(arr > 0)),
            'selected_codes': {'primary_ocean_fcode': 44500, 'primary_ocean_ftype': 445, 'supplemental_estuary_fcode': 31200},
        }
    _write_zero_mask_like(template_path, out_path)
    return {'layer': None, 'ftype_column': None, 'fcode_column': None, 'name_column': None, 'polygon_count': 0, 'ocean_polygon_count': 0, 'bay_inlet_polygon_count': 0, 'pixel_count': 0}


def _union_mask_in_place(*, base_mask_path: Path, add_mask_path: Path, logger: logging.Logger) -> Dict[str, int]:
    with rasterio.open(base_mask_path) as base_ds:
        base = base_ds.read(1)
        profile = base_ds.profile.copy()
        transform = base_ds.transform
        crs = base_ds.crs
    add = np.zeros_like(base, dtype=np.uint8)
    with rasterio.open(add_mask_path) as add_ds:
        reproject(
            source=rasterio.band(add_ds, 1),
            destination=add,
            src_transform=add_ds.transform,
            src_crs=add_ds.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=add_ds.nodata,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    before = int(np.count_nonzero(base > 0))
    combined = ((base > 0) | (add > 0)).astype(np.uint8)
    after = int(np.count_nonzero(combined > 0))
    added = int(np.count_nonzero((combined > 0) & ~(base > 0)))
    if added > 0:
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        profile.pop('tiled', None)
        profile.update(dtype='uint8', count=1, nodata=0, compress='deflate')
        with rasterio.open(base_mask_path, 'w', **profile) as dst:
            dst.write(combined, 1)
        logger.info('[DOMAIN] Unioned supplemental coastal mask into %s: %d -> %d pixels (+%d)', base_mask_path.name, before, after, added)
    return {'before_pixels': before, 'after_pixels': after, 'added_pixels': added}


def _subtract_mask_in_place(*, base_mask_path: Path, remove_mask_path: Path, logger: logging.Logger) -> Dict[str, int]:
    with rasterio.open(base_mask_path) as base_ds:
        base = base_ds.read(1)
        profile = base_ds.profile.copy()
        transform = base_ds.transform
        crs = base_ds.crs
    rem = np.zeros_like(base, dtype=np.uint8)
    with rasterio.open(remove_mask_path) as rem_ds:
        reproject(
            source=rasterio.band(rem_ds, 1),
            destination=rem,
            src_transform=rem_ds.transform,
            src_crs=rem_ds.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=rem_ds.nodata,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    before = int(np.count_nonzero(base > 0))
    out = ((base > 0) & ~(rem > 0)).astype(np.uint8)
    after = int(np.count_nonzero(out > 0))
    removed = int(np.count_nonzero((base > 0) & (rem > 0)))
    if removed > 0:
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        profile.pop('tiled', None)
        profile.update(dtype='uint8', count=1, nodata=0, compress='deflate')
        with rasterio.open(base_mask_path, 'w', **profile) as dst:
            dst.write(out, 1)
        logger.info('[DOMAIN] Subtracted supplemental coastal mask from %s: %d -> %d pixels (-%d)', base_mask_path.name, before, after, removed)
    return {'before_pixels': before, 'after_pixels': after, 'removed_pixels': removed}


def _write_ocean_connectivity_mask(*, template_path: Path, ocean_mask_path: Path, sea_ocean_mask_path: Optional[Path], out_path: Path) -> Path:
    with rasterio.open(template_path) as ref:
        ocean_dst = np.full((ref.height, ref.width), 255, dtype=np.uint8)
        with rasterio.open(ocean_mask_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=ocean_dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.nearest,
                dst_nodata=255,
            )
        sea_ocean_dst = np.zeros((ref.height, ref.width), dtype=np.uint8)
        if sea_ocean_mask_path is not None and Path(sea_ocean_mask_path).exists():
            with rasterio.open(sea_ocean_mask_path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=sea_ocean_dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.nearest,
                    dst_nodata=0,
                )
        water = (ocean_dst == 0) | (sea_ocean_dst > 0)
        profile = ref.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
        profile.update(dtype="uint8", nodata=255, compress="deflate")
        data = np.where(water, 0, 255).astype("uint8")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
    return out_path


def _template_ocean_water_pixels(template_path: Path, ocean_mask_path: Path) -> int:
    with rasterio.open(template_path) as ref:
        ocean_dst = np.full((ref.height, ref.width), 255, dtype=np.uint8)
        with rasterio.open(ocean_mask_path) as src:
            src_nodata = src.nodata
            if src_nodata in (0, 1):
                src_nodata = None
            reproject(
                source=rasterio.band(src, 1),
                destination=ocean_dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=255,
                resampling=Resampling.nearest,
            )
    return int(np.count_nonzero(ocean_dst == 0))


def _write_empty_sdb_domain_mask(*, template_path: Path, out_path: Path) -> Path:
    with rasterio.open(template_path) as ref:
        arr = np.ones((ref.height, ref.width), dtype=np.uint8)
        profile = ref.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(dtype="uint8", count=1, nodata=1, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    return out_path

def _write_sdb_domain_mask(*, template_path: Path, ocean_mask_path: Path, estuary_clip_mask_path: Path, sea_ocean_mask_path: Optional[Path], out_path: Path) -> Path:
    with rasterio.open(template_path) as ref:
        ocean_dst = np.full((ref.height, ref.width), 255, dtype=np.uint8)
        with rasterio.open(ocean_mask_path) as src:
            src_nodata = src.nodata
            if src_nodata in (0, 1):
                src_nodata = None
            reproject(
                source=rasterio.band(src, 1),
                destination=ocean_dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=255,
                resampling=Resampling.nearest,
            )
        estuary_dst = np.zeros((ref.height, ref.width), dtype=np.uint8)
        with rasterio.open(estuary_clip_mask_path) as src:
            src_nodata = src.nodata
            if src_nodata in (0, 1):
                src_nodata = None
            reproject(
                source=rasterio.band(src, 1),
                destination=estuary_dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
        sea_ocean_dst = np.zeros((ref.height, ref.width), dtype=np.uint8)
        if sea_ocean_mask_path is not None and Path(sea_ocean_mask_path).exists():
            with rasterio.open(sea_ocean_mask_path) as src:
                src_nodata = src.nodata
                if src_nodata in (0, 1):
                    src_nodata = None
                reproject(
                    source=rasterio.band(src, 1),
                    destination=sea_ocean_dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
        water = (ocean_dst == 0) | (estuary_dst > 0) | (sea_ocean_dst > 0)
        arr = np.where(water, 0, 1).astype(np.uint8)
        profile = ref.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(dtype="uint8", count=1, nodata=1, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    return out_path


def _build_river_domain_masks(cfg: Any, *, river_dem: Path, river_gpkg: Path, ocean_mask: Optional[Path], with_nhd_mask: Path, outputs: GuidanceDomainPaths, logger: logging.Logger, inland_only: bool = False) -> None:
    script_path = Path(__file__).with_name("river_domain_mask.py")
    aligned_water_mask = outputs.river_water_support_mask
    _build_template_aligned_river_water_mask(with_nhd_mask, river_gpkg, river_dem, aligned_water_mask, logger=logger)
    cmd = [
        sys.executable,
        str(script_path),
        f"--river-gpkg={river_gpkg}",
        f"--template-raster={river_dem}",
        f"--water-mask={aligned_water_mask}",
        f"--water-mask-role={'river_support' if inland_only else 'waffles_with_nhd'}",
        f"--out-channel-mask={outputs.river_channel_mask}",
        f"--out-open-water-mask={outputs.open_water_mask}",
        f"--out-mainstem-mask={outputs.mainstem_mask}",
        f"--out-policy-json={outputs.river_domain_policy_json}",
        f"--out-effective-water-mask={outputs.effective_water_mask}",
        f"--out-corridor-mask={outputs.corridor_mask}",
        f"--out-nhdarea-mask={outputs.nhdarea_mask}",
        "--write-debug",
        f"--channel-buffer-m={float(getattr(cfg, 'river_channel_buffer_m', 400.0) or 400.0)}",
        f"--max-channel-width-m={float(getattr(cfg, 'river_max_channel_width_m', 600.0) or 600.0)}",
        f"--mainstem-method={str(getattr(cfg, 'river_mainstem_method', 'dominant_trunk') or 'dominant_trunk').strip().lower()}",
        f"--mainstem-solve-layer={str(getattr(cfg, 'river_mainstem_solve_layer', 'auto') or 'auto').strip()}",
        f"--mainstem-min-order={int(getattr(cfg, 'river_mainstem_min_order', 5) or 5)}",
        f"--max-mainstem-width-m={float(getattr(cfg, 'river_max_mainstem_width_m', 2500.0) or 2500.0)}",
        f"--channel-source={str(getattr(cfg, 'river_channel_source', 'auto') or 'auto').strip().lower()}",
        f"--nhdarea-allow-ftype={str(getattr(cfg, 'river_nhdarea_allow_ftype', '460') or '460')}",
    ]
    if (not inland_only) and ocean_mask is not None:
        cmd.append(f"--ocean-mask={ocean_mask}")
    nhd_allow_fcode = str(getattr(cfg, 'river_nhdarea_allow_fcode', '') or '').strip()
    if nhd_allow_fcode:
        cmd.append(f"--nhdarea-allow-fcode={nhd_allow_fcode}")
    if bool(getattr(cfg, 'river_use_nhdarea', True)) and str(getattr(cfg, 'river_channel_source', 'auto') or 'auto').strip().lower() in ('auto', 'nhdarea'):
        cmd.append(f"--nhdarea-gpkg={river_gpkg}")
        cmd.append(f"--nhdarea-layer={str(getattr(cfg, 'river_nhdarea_layer', 'nhdarea_clip') or 'nhdarea_clip')}")
    ocean_keep = float(getattr(cfg, 'river_ocean_keep_dist_m', 0.0) or 0.0)
    if ocean_keep > 0.0:
        cmd.append(f"--ocean-keep-dist-m={ocean_keep}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "River guidance-domain build failed:\n"
            f"command={' '.join(cmd)}\n"
            f"stdout={proc.stdout[-2000:]}\n"
            f"stderr={proc.stderr[-2000:]}"
        )
    logger.info("[DOMAIN] Built river domain masks: %s", outputs.river_channel_mask)




def _enforce_handoff_on_mask(*, base_mask_path: Path, remove_mask_path: Optional[Path] = None, add_mask_path: Optional[Path] = None, valid_value: int = 1, invalid_value: int = 0, logger: logging.Logger) -> Dict[str, int]:
    stats = {"removed_pixels": 0, "added_pixels": 0}
    with rasterio.open(base_mask_path) as src:
        arr = src.read(1).astype(np.uint8)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    base_valid = arr == valid_value
    if remove_mask_path is not None and Path(remove_mask_path).exists():
        rm = np.zeros_like(arr, dtype=np.uint8)
        with rasterio.open(remove_mask_path) as src:
            reproject(source=rasterio.band(src,1), destination=rm, src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata, dst_transform=transform, dst_crs=crs, dst_nodata=0, resampling=Resampling.nearest)
        removed = base_valid & (rm > 0)
        stats["removed_pixels"] = int(np.count_nonzero(removed))
        arr[removed] = invalid_value
    if add_mask_path is not None and Path(add_mask_path).exists():
        ad = np.zeros_like(arr, dtype=np.uint8)
        with rasterio.open(add_mask_path) as src:
            reproject(source=rasterio.band(src,1), destination=ad, src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata, dst_transform=transform, dst_crs=crs, dst_nodata=0, resampling=Resampling.nearest)
        add = ad > 0
        stats["added_pixels"] = int(np.count_nonzero((arr != valid_value) & add))
        arr[add] = valid_value
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(dtype="uint8", count=1, nodata=invalid_value, compress="deflate")
    with rasterio.open(base_mask_path, "w", **profile) as dst:
        dst.write(arr, 1)
    if stats["removed_pixels"] or stats["added_pixels"]:
        logger.info("[DOMAIN] Enforced shared handoff on %s: %s", base_mask_path.name, stats)
    return stats


def _validate_shared_domain_handoff(*, river_guidance_domain_path: Path, sdb_domain_path: Path, estuary_clip_mask_path: Path, sea_ocean_mask_path: Path, logger: logging.Logger) -> Dict[str, int]:
    with rasterio.open(river_guidance_domain_path) as src:
        river = src.read(1) == 1
        transform = src.transform
        crs = src.crs
    with rasterio.open(sdb_domain_path) as src:
        sdb = src.read(1) == 0
    handoff = np.zeros(river.shape, dtype=np.uint8)
    for mask_path in (estuary_clip_mask_path, sea_ocean_mask_path):
        if mask_path is not None and Path(mask_path).exists():
            tmp = np.zeros_like(handoff, dtype=np.uint8)
            with rasterio.open(mask_path) as src:
                reproject(source=rasterio.band(src,1), destination=tmp, src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata, dst_transform=transform, dst_crs=crs, dst_nodata=0, resampling=Resampling.nearest)
            handoff = np.maximum(handoff, tmp)
    handoff_bool = handoff > 0
    river_overlap = int(np.count_nonzero(river & handoff_bool))
    sdb_handoff = int(np.count_nonzero(sdb & handoff_bool))
    total_handoff = int(np.count_nonzero(handoff_bool))
    if river_overlap:
        logger.warning("[DOMAIN] Shared handoff overlap remained in river guidance domain: %d pixels", river_overlap)
    return {"handoff_pixels": total_handoff, "river_handoff_overlap_pixels": river_overlap, "sdb_handoff_pixels": sdb_handoff}



def _count_mask_pixels(mask_path: Path, *, water_value: int = 1) -> int:
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
    return int(np.count_nonzero(arr == water_value))


def _write_domain_summary(summary_path: Path, *, paths: GuidanceDomainPaths, manifest_path: Path) -> Path:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    diagnostics = manifest.get("diagnostics", {})
    summary = {
        "stage": "shared_domain_generation",
        "manifest_json": str(manifest_path),
        "review_dir": str(paths.review_dir),
        "masks": {
            "ocean_mask": str(paths.ocean_mask),
            "ocean_connectivity_mask": str(paths.ocean_connectivity_mask),
            "base_water_mask": str(paths.with_nhd_water_mask),
            "river_candidate_domain_mask": str(paths.river_channel_mask),
            "estuary_handoff_mask": str(paths.estuary_clip_mask),
            "river_active_domain_mask": str(paths.river_guidance_domain_mask),
            "sdb_candidate_domain_mask": str(paths.sdb_guidance_domain_mask),
            "estuary_transition_mask": str(paths.estuary_transition_mask),
            "nhd_sea_ocean_mask": str(paths.nhd_sea_ocean_mask),
        },
        "pixel_counts": {
            "ocean_mask_water_pixels": _count_mask_pixels(paths.ocean_mask, water_value=0),
            "ocean_connectivity_water_pixels": _count_mask_pixels(paths.ocean_connectivity_mask, water_value=0),
            "base_water_pixels": _count_mask_pixels(paths.with_nhd_water_mask, water_value=0),
            "river_candidate_pixels": _count_mask_pixels(paths.river_channel_mask, water_value=1),
            "estuary_handoff_pixels": _count_mask_pixels(paths.estuary_clip_mask, water_value=1),
            "river_active_pixels": _count_mask_pixels(paths.river_guidance_domain_mask, water_value=1),
            "sdb_candidate_pixels": _count_mask_pixels(paths.sdb_guidance_domain_mask, water_value=0),
            "estuary_transition_pixels": _count_mask_pixels(paths.estuary_transition_mask, water_value=1),
            "nhd_sea_ocean_pixels": _count_mask_pixels(paths.nhd_sea_ocean_mask, water_value=1),
        },
        "derived_activation": {
            "river_should_run": _count_mask_pixels(paths.river_guidance_domain_mask, water_value=1) > 0,
            "sdb_should_run": _count_mask_pixels(paths.sdb_guidance_domain_mask, water_value=0) > 0,
        },
        "diagnostics": diagnostics,
        "definitions": {
            "base_water_mask": "Shared water/context mask built from template-aligned WAFFLES with-NHD water.",
            "ocean_connectivity_mask": "Ocean/coastal-connected water from WAFFLES ocean mask plus NHD coastal/ocean and estuary supplementation.",
            "river_candidate_domain_mask": "River-owned candidate water before the final river-guidance near-bank exclusion margin.",
            "estuary_handoff_mask": "Shared river-to-SDB handoff mask derived from estuary/coastal logic.",
            "river_active_domain_mask": "Final river guidance domain consumed by downstream river stages.",
            "sdb_candidate_domain_mask": "Hydrologically/coastally eligible SDB domain before later optical/data admissibility.",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def _mask_shape(path: Path) -> Tuple[int, int]:
    with rasterio.open(path) as ds:
        return int(ds.height), int(ds.width)


def _count_overlap(mask_a: Path, *, value_a: int, mask_b: Path, value_b: int) -> int:
    with rasterio.open(mask_a) as dsa, rasterio.open(mask_b) as dsb:
        if dsa.height != dsb.height or dsa.width != dsb.width or dsa.transform != dsb.transform:
            raise ValueError(f"Mask grid mismatch for overlap count: {mask_a} vs {mask_b}")
        a = dsa.read(1)
        b = dsb.read(1)
    return int(np.count_nonzero((a == value_a) & (b == value_b)))


def _count_unexpected_sdb_pixels(mask_sdb: Path, *, value_sdb: int, ocean_connectivity_mask: Path, ocean_value: int, estuary_mask: Path, estuary_value: int) -> int:
    with rasterio.open(mask_sdb) as dss, rasterio.open(ocean_connectivity_mask) as dso, rasterio.open(estuary_mask) as dse:
        if (
            dss.height != dso.height or dss.width != dso.width or dss.transform != dso.transform or
            dss.height != dse.height or dss.width != dse.width or dss.transform != dse.transform
        ):
            raise ValueError(
                f"Mask grid mismatch for SDB subset check: {mask_sdb} vs {ocean_connectivity_mask} vs {estuary_mask}"
            )
        sdb = dss.read(1)
        ocean = dso.read(1)
        estuary = dse.read(1)
    allowed = (ocean == ocean_value) | (estuary == estuary_value)
    return int(np.count_nonzero((sdb == value_sdb) & (~allowed)))


def _write_domain_validation(validation_path: Path, *, paths: GuidanceDomainPaths, summary_path: Path) -> Path:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    pixel_counts = summary.get("pixel_counts", {}) if isinstance(summary, dict) else {}
    checks = []

    def _add_check(name: str, ok: bool, details: Dict[str, Any]) -> None:
        checks.append({"name": name, "ok": bool(ok), "details": details})

    mask_specs = [
        ("ocean_connectivity_mask", paths.ocean_connectivity_mask),
        ("base_water_mask", paths.with_nhd_water_mask),
        ("river_candidate_domain_mask", paths.river_channel_mask),
        ("estuary_handoff_mask", paths.estuary_clip_mask),
        ("river_active_domain_mask", paths.river_guidance_domain_mask),
        ("sdb_candidate_domain_mask", paths.sdb_guidance_domain_mask),
    ]
    shapes = {}
    for name, path in mask_specs:
        exists = Path(path).exists()
        shp = _mask_shape(path) if exists else None
        shapes[name] = shp
        _add_check(f"exists::{name}", exists, {"path": str(path), "shape": shp})

    grid_shapes = [shp for shp in shapes.values() if shp is not None]
    _add_check("grid_consistent", len(set(grid_shapes)) <= 1, {"shapes": shapes})

    river_active_overlap = _count_overlap(paths.river_guidance_domain_mask, value_a=1, mask_b=paths.estuary_clip_mask, value_b=1)
    _add_check(
        "river_excludes_estuary_handoff",
        river_active_overlap == 0,
        {"overlap_pixels": int(river_active_overlap)},
    )

    estuary_not_in_sdb = _count_overlap(paths.estuary_clip_mask, value_a=1, mask_b=paths.sdb_guidance_domain_mask, value_b=1)
    # sdb water convention is 0 for included, so value_b=1 means estuary pixels not included in sdb candidate
    _add_check(
        "sdb_includes_estuary_handoff",
        estuary_not_in_sdb == 0,
        {"missing_pixels": int(estuary_not_in_sdb)},
    )

    river_candidate_pixels = int(pixel_counts.get("river_candidate_pixels", 0) or 0)
    river_active_pixels = int(pixel_counts.get("river_active_pixels", 0) or 0)
    _add_check(
        "river_active_not_larger_than_candidate",
        river_active_pixels <= river_candidate_pixels,
        {"river_candidate_pixels": river_candidate_pixels, "river_active_pixels": river_active_pixels},
    )

    sdb_candidate_pixels = int(pixel_counts.get("sdb_candidate_pixels", 0) or 0)
    estuary_handoff_pixels = int(pixel_counts.get("estuary_handoff_pixels", 0) or 0)
    ocean_connectivity_pixels = int(pixel_counts.get("ocean_connectivity_water_pixels", 0) or 0)
    sdb_unexpected_pixels = _count_unexpected_sdb_pixels(
        paths.sdb_guidance_domain_mask,
        value_sdb=0,
        ocean_connectivity_mask=paths.ocean_connectivity_mask,
        ocean_value=0,
        estuary_mask=paths.estuary_clip_mask,
        estuary_value=1,
    )
    _add_check(
        "sdb_candidate_supported_by_connectivity_or_handoff",
        sdb_unexpected_pixels == 0,
        {
            "unexpected_pixels": int(sdb_unexpected_pixels),
            "sdb_candidate_pixels": sdb_candidate_pixels,
            "ocean_connectivity_pixels": ocean_connectivity_pixels,
            "estuary_handoff_pixels": estuary_handoff_pixels,
        },
    )

    ok = all(bool(check.get("ok", False)) for check in checks)
    payload = {
        "stage": "shared_domain_validation",
        "summary_json": str(summary_path),
        "ok": bool(ok),
        "checks": checks,
    }
    validation_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return validation_path
def _write_manifest(paths: GuidanceDomainPaths, *, cache_key: str, params: Dict[str, Any], diagnostics: Dict[str, Any]) -> Path:
    payload = {
        "cache_key": cache_key,
        "stage": "guidance_domains",
        "params": params,
        "outputs": {k: str(v) for k, v in paths.as_dict().items() if k not in {"cache_dir", "run_dir"}},
        "diagnostics": diagnostics,
    }
    paths.manifest_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return paths.manifest_json


def ensure_guidance_domains(cfg: Any, *, river_dem: Path, river_gpkg: Path, derived_cache_root: Path, review_root: Optional[Path] = None, report: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> GuidanceDomainPaths:
    active_log = logger or LOG
    river_dem = Path(river_dem)
    river_gpkg = Path(river_gpkg)
    derived_cache_root = Path(derived_cache_root)
    cache_root = Path(cfg.cache_root)
    cache_key = compute_guidance_domain_cache_key(cfg, river_dem=river_dem, river_gpkg=river_gpkg)
    shared_dir = cache_root / "guidance_domains" / cache_key
    run_dir = derived_cache_root / "guidance_domains"
    review_dir = Path(review_root) if review_root is not None else run_dir
    if review_root is not None and review_dir.name != "guidance_domains":
        review_dir = review_dir / "guidance_domains"
    shared_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    shared_paths = _paths_from_dir(shared_dir)
    run_paths = _paths_from_dir(run_dir)
    review_paths = _paths_from_dir(review_dir)
    object.__setattr__(run_paths, "run_dir", run_dir)
    object.__setattr__(run_paths, "cache_dir", shared_dir)
    object.__setattr__(run_paths, "review_dir", review_dir)

    params = _domain_params(cfg, river_dem=river_dem, river_gpkg=river_gpkg)
    if not _manifest_complete(shared_paths.manifest_json):
        masks_cache = cache_root / "masks"
        masks_cache.mkdir(parents=True, exist_ok=True)
        ocean_cache = ensure_waffles_coastline_mask(
            masks_cache,
            str(cfg.aoi),
            inc_arcsec=float(getattr(cfg, "waffles_inc_arcsec", 1.0) or 1.0),
            want_nhd=False,
            want_lakes=False,
            prefix="waffles_coastline_ocean_only",
            logger=active_log,
        )
        if shared_paths.ocean_mask.exists():
            shared_paths.ocean_mask.unlink()
        _stage(Path(ocean_cache), shared_paths.ocean_mask)
        sea_ocean_diag = _write_nhd_sea_ocean_mask(
            template_path=river_dem,
            river_gpkg=river_gpkg,
            out_path=shared_paths.nhd_sea_ocean_mask,
            logger=active_log,
        )
        _write_ocean_connectivity_mask(
            template_path=river_dem,
            ocean_mask_path=shared_paths.ocean_mask,
            sea_ocean_mask_path=shared_paths.nhd_sea_ocean_mask,
            out_path=shared_paths.ocean_connectivity_mask,
        )
        _build_canonical_with_nhd_mask(shared_paths.ocean_connectivity_mask, river_gpkg, shared_paths.with_nhd_water_mask, logger=active_log)
        ocean_pixels_on_template = _template_ocean_water_pixels(river_dem, shared_paths.ocean_connectivity_mask)
        inland_only = ocean_pixels_on_template == 0
        if inland_only:
            active_log.info("[DOMAIN] Inland AOI detected: no ocean-connected water on river template grid after Sea/Ocean handoff; building river-only guidance domain and writing empty SDB domain.")
        _build_river_domain_masks(
            cfg,
            river_dem=river_dem,
            river_gpkg=river_gpkg,
            ocean_mask=None if inland_only else shared_paths.ocean_connectivity_mask,
            with_nhd_mask=shared_paths.with_nhd_water_mask,
            outputs=shared_paths,
            logger=active_log,
            inland_only=inland_only,
        )
        temp_report: Dict[str, Any] = {}
        removed, estuary_path = clip_channel_mask_for_estuary(
            shared_paths.river_channel_mask,
            cfg,
            ocean_mask_path=shared_paths.ocean_connectivity_mask,
            report=temp_report,
            logger=active_log,
        )
        if estuary_path is None or not shared_paths.estuary_clip_mask.exists():
            _write_zero_mask_like(shared_paths.river_channel_mask, shared_paths.estuary_clip_mask)
        if not shared_paths.estuary_transition_mask.exists():
            _write_zero_mask_like(shared_paths.river_channel_mask, shared_paths.estuary_transition_mask)
        sea_ocean_union = _union_mask_in_place(
            base_mask_path=shared_paths.estuary_clip_mask,
            add_mask_path=shared_paths.nhd_sea_ocean_mask,
            logger=active_log,
        )
        sea_ocean_channel_trim = _subtract_mask_in_place(
            base_mask_path=shared_paths.river_channel_mask,
            remove_mask_path=shared_paths.nhd_sea_ocean_mask,
            logger=active_log,
        )
        sea_ocean_open_water_trim = _subtract_mask_in_place(
            base_mask_path=shared_paths.open_water_mask,
            remove_mask_path=shared_paths.nhd_sea_ocean_mask,
            logger=active_log,
        )
        _clip_mainstem_mask(shared_paths.mainstem_mask, shared_paths.estuary_clip_mask, logger=active_log)
        # Enforce the core invariant before staging/review: mainstem must be a subset of the retained river channel.
        with rasterio.open(shared_paths.mainstem_mask) as _ms_ds, rasterio.open(shared_paths.river_channel_mask) as _ch_ds:
            _ms = _ms_ds.read(1)
            _ch = _ch_ds.read(1)
            _profile = _ms_ds.profile.copy()
        _synced = ((_ms == 1) & (_ch == 1)).astype("uint8")
        _profile.pop("blockxsize", None)
        _profile.pop("blockysize", None)
        _profile.pop("tiled", None)
        _profile.update(dtype="uint8", nodata=0, compress="deflate")
        with rasterio.open(shared_paths.mainstem_mask, "w", **_profile) as _dst:
            _dst.write(_synced, 1)
        river_guidance_diag = _write_river_guidance_domain_mask(
            channel_mask_path=shared_paths.river_channel_mask,
            out_path=shared_paths.river_guidance_domain_mask,
            bank_margin_m=float(getattr(cfg, 'river_guidance_bank_margin_m', 3.0) or 3.0),
            logger=active_log,
            estuary_transition_mask_path=shared_paths.estuary_transition_mask,
        )
        river_guidance_handoff_enforcement = _enforce_handoff_on_mask(
            base_mask_path=shared_paths.river_guidance_domain_mask,
            remove_mask_path=shared_paths.estuary_clip_mask,
            valid_value=1,
            invalid_value=0,
            logger=active_log,
        )
        if inland_only:
            _write_empty_sdb_domain_mask(template_path=river_dem, out_path=shared_paths.sdb_guidance_domain_mask)
            sdb_desc = "empty inland-only SDB domain (no ocean-connected water on template grid)"
            sdb_handoff_enforcement = {"removed_pixels": 0, "added_pixels": 0}
        else:
            _write_sdb_domain_mask(
                template_path=river_dem,
                ocean_mask_path=shared_paths.ocean_connectivity_mask,
                estuary_clip_mask_path=shared_paths.estuary_clip_mask,
                sea_ocean_mask_path=shared_paths.nhd_sea_ocean_mask,
                out_path=shared_paths.sdb_guidance_domain_mask,
            )
            sdb_handoff_enforcement = _enforce_handoff_on_mask(
                base_mask_path=shared_paths.sdb_guidance_domain_mask,
                add_mask_path=shared_paths.estuary_clip_mask,
                valid_value=0,
                invalid_value=1,
                logger=active_log,
            )
            sdb_desc = "ocean-connected water after Sea/Ocean handoff plus shared river-excluded estuary/coastal area"
        handoff_validation = _validate_shared_domain_handoff(
            river_guidance_domain_path=shared_paths.river_guidance_domain_mask,
            sdb_domain_path=shared_paths.sdb_guidance_domain_mask,
            estuary_clip_mask_path=shared_paths.estuary_clip_mask,
            sea_ocean_mask_path=shared_paths.nhd_sea_ocean_mask,
            logger=active_log,
        )
        diagnostics = {
            "inland_only": bool(inland_only),
            "sdb_domain_reason": "inland_no_ocean_water_after_handoff" if inland_only else "ocean_connected_plus_estuary_handoff",
            "ocean_pixels_on_template": int(ocean_pixels_on_template),
            "ocean_connectivity_mask": str(shared_paths.ocean_connectivity_mask),
            "river_estuary_pixels_removed": int(removed),
            "nhd_sea_ocean_diagnostics": sea_ocean_diag,
            "nhd_sea_ocean_union": sea_ocean_union,
            "nhd_sea_ocean_channel_trim": sea_ocean_channel_trim,
            "nhd_sea_ocean_open_water_trim": sea_ocean_open_water_trim,
            "river_guidance_domain_diagnostics": river_guidance_diag,
            "domain_contract": {
                "river_guidance_domain": "NHD-constrained river channel after shared estuary/coastal handoff subtraction and near-bank exclusion margin",
                "sdb_guidance_domain": sdb_desc,
                "estuary_signal": "width_ratio_plus_nhd_coastal_ocean_44500_plus_bay_inlet_31200",
            },
            "temp_report": temp_report,
        }
        _write_manifest(shared_paths, cache_key=cache_key, params=params, diagnostics=diagnostics)
        active_log.info("[DOMAIN] Guidance domains built in shared cache: %s", shared_dir)
    else:
        active_log.info("[DOMAIN] Reusing cached guidance domains: %s", shared_dir)

    expected_shared = _expected_paths(shared_dir)
    expected_run = _expected_paths(run_dir)
    expected_review = _expected_paths(review_dir)
    for key, src in expected_shared.items():
        if key == "manifest_json":
            continue
        if key in OPTIONAL_DEBUG_KEYS and not src.exists():
            continue
        dst = expected_run[key]
        _stage(src, dst)
        review_dst = expected_review[key]
        _stage(src, review_dst)
    shutil.copy2(shared_paths.manifest_json, run_paths.manifest_json)

    review_manifest = json.loads(shared_paths.manifest_json.read_text(encoding="utf-8"))
    review_outputs = {k: str(v) for k, v in review_paths.as_dict().items() if k not in {"cache_dir", "run_dir", "review_dir"}}
    review_manifest["run_outputs"] = {k: str(v) for k, v in run_paths.as_dict().items() if k not in {"cache_dir", "run_dir", "review_dir"}}
    review_manifest["review_outputs"] = review_outputs
    review_manifest["inspection"] = {
        "review_before_inference": True,
        "review_dir": str(review_dir),
        "manifest_json": str(review_paths.manifest_json),
        "river_domain_to_check": str(review_paths.river_guidance_domain_mask),
        "sdb_domain_to_check": str(review_paths.sdb_guidance_domain_mask),
        "supporting_masks": {
            "ocean_mask": str(review_paths.ocean_mask),
        "ocean_connectivity_mask": str(review_paths.ocean_connectivity_mask),
            "with_nhd_water_mask": str(review_paths.with_nhd_water_mask),
            "river_water_support_mask": str(review_paths.river_water_support_mask),
            "river_channel_mask": str(review_paths.river_channel_mask),
            "effective_water_mask": str(review_paths.effective_water_mask),
            "corridor_mask": str(review_paths.corridor_mask),
            "nhdarea_mask": str(review_paths.nhdarea_mask),
            "open_water_mask": str(review_paths.open_water_mask),
            "mainstem_mask": str(review_paths.mainstem_mask),
            "estuary_clip_mask": str(review_paths.estuary_clip_mask),
            "estuary_transition_mask": str(review_paths.estuary_transition_mask),
        },
        "construction_diagnostics": {
            "receipt_json": str(Path(review_dir) / "river_channel_mask_construction_receipt.json"),
            "01_water_footprint": str(Path(review_dir) / "01_water_footprint.tif"),
            "02_effective_water_mask": str(Path(review_dir) / "02_effective_water_mask.tif"),
            "03_corridor_mask": str(Path(review_dir) / "03_corridor_mask.tif"),
            "04_channel_candidate_water_and_corridor": str(Path(review_dir) / "04_channel_candidate_water_and_corridor.tif"),
            "05_structure_seed_nhdarea_or_mainstem": str(Path(review_dir) / "05_structure_seed_nhdarea_or_mainstem.tif"),
            "06_component_labels": str(Path(review_dir) / "06_component_labels.tif"),
            "07_component_accept_mask": str(Path(review_dir) / "07_component_accept_mask.tif"),
            "08_channel_after_component_acceptance": str(Path(review_dir) / "08_channel_after_component_acceptance.tif"),
            "09_channel_after_morphology": str(Path(review_dir) / "09_channel_after_morphology.tif"),
            "10_river_channel_mask_final": str(Path(review_dir) / "10_river_channel_mask_final.tif"),
        },
    }
    review_paths.manifest_json.write_text(json.dumps(review_manifest, indent=2, sort_keys=True), encoding="utf-8")

    domain_review_dir = review_dir.parent / "domains" if review_dir.name == "guidance_domains" else review_dir / "domains"
    domain_review_dir.mkdir(parents=True, exist_ok=True)
    domain_review_manifest = domain_review_dir / "domain_manifest.json"
    domain_review_summary = domain_review_dir / "domain_summary.json"
    shutil.copy2(review_paths.manifest_json, domain_review_manifest)
    _write_domain_summary(domain_review_summary, paths=review_paths, manifest_path=review_paths.manifest_json)
    domain_review_validation = _write_domain_validation(domain_review_dir / "domain_validation.json", paths=review_paths, summary_path=domain_review_summary)
    domain_run_dir = run_dir.parent / "domains" if run_dir.name == "guidance_domains" else run_dir / "domains"
    domain_run_dir.mkdir(parents=True, exist_ok=True)
    domain_run_manifest = domain_run_dir / "domain_manifest.json"
    domain_run_summary = domain_run_dir / "domain_summary.json"
    shutil.copy2(run_paths.manifest_json, domain_run_manifest)
    _write_domain_summary(domain_run_summary, paths=run_paths, manifest_path=run_paths.manifest_json)
    domain_run_validation = _write_domain_validation(domain_run_dir / "domain_validation.json", paths=run_paths, summary_path=domain_run_summary)

    if report is not None:
        report.setdefault("guidance_domains", {}).update({
            "cache_key": cache_key,
            "cache_dir": str(shared_dir),
            "run_dir": str(run_dir),
            "review_dir": str(review_dir),
            "manifest_json": str(run_paths.manifest_json),
            "review_manifest_json": str(review_paths.manifest_json),
            "domain_manifest_json": str(domain_run_manifest),
            "domain_summary_json": str(domain_run_summary),
            "domain_validation_json": str(domain_run_validation),
            "review_domain_manifest_json": str(domain_review_manifest),
            "review_domain_summary_json": str(domain_review_summary),
            "review_domain_validation_json": str(domain_review_validation),
            "outputs": {k: str(v) for k, v in run_paths.as_dict().items() if k not in {"cache_dir", "run_dir", "review_dir"}},
            "review_outputs": review_outputs,
        })

    return run_paths


__all__ = [
    "GuidanceDomainPaths",
    "compute_guidance_domain_cache_key",
    "ensure_guidance_domains",
    "make_guidance_domain_cfg",
]
