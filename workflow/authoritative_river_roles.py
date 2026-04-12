from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage as ndi

LOG = logging.getLogger(__name__)


ROLE_BED_CORE = "authoritative_bed_core"
ROLE_BED_INNER = "authoritative_bed_inner"
ROLE_BANK_MARGIN = "authoritative_bank_margin"
ROLE_AMBIGUOUS = "authoritative_overbank_or_ambiguous"

ROLE_TO_CODE = {
    ROLE_AMBIGUOUS: 0,
    ROLE_BANK_MARGIN: 1,
    ROLE_BED_INNER: 2,
    ROLE_BED_CORE: 3,
}
CODE_TO_ROLE = {int(v): str(k) for k, v in ROLE_TO_CODE.items()}


def role_to_code(role: Any) -> int:
    return int(ROLE_TO_CODE.get(str(role), 0))


def code_to_role(code: Any) -> str:
    try:
        idx = int(code)
    except Exception:
        return ROLE_AMBIGUOUS
    return str(CODE_TO_ROLE.get(idx, ROLE_AMBIGUOUS))


def _pixel_size_m(profile: Dict[str, Any]) -> float:
    transform = profile["transform"]
    crs = profile.get("crs")
    px_size_x = abs(float(transform.a))
    px_size_y = abs(float(transform.e))
    if crs is not None and getattr(crs, "is_geographic", False):
        try:
            mid_lat = float(transform.f + (profile["height"] * transform.e * 0.5))
        except Exception:
            mid_lat = 0.0
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = meters_per_deg_lat * max(np.cos(np.deg2rad(mid_lat)), 1e-6)
        return max(px_size_x * meters_per_deg_lon, px_size_y * meters_per_deg_lat, 1e-6)
    return max(px_size_x, px_size_y, 1e-6)


def _read_bool_mask_aligned(mask_path: Path | str | None, *, template_profile: Dict[str, Any]) -> np.ndarray:
    shape = (int(template_profile["height"]), int(template_profile["width"]))
    if mask_path is None:
        return np.zeros(shape, dtype=bool)
    path = Path(mask_path)
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    out = np.zeros(shape, dtype=np.uint8)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=0,
        )
    return out > 0


def _component_max_distance(channel_mask: np.ndarray, dist_to_bank_m: np.ndarray) -> np.ndarray:
    labels, ncomp = ndi.label(channel_mask, structure=np.ones((3, 3), dtype=np.uint8))
    if ncomp <= 0:
        return np.full(channel_mask.shape, np.nan, dtype=np.float32)
    out = np.full(channel_mask.shape, np.nan, dtype=np.float32)
    for idx in range(1, ncomp + 1):
        comp = labels == idx
        if not np.any(comp):
            continue
        dmax = float(np.nanmax(dist_to_bank_m[comp])) if np.any(np.isfinite(dist_to_bank_m[comp])) else float("nan")
        if np.isfinite(dmax) and dmax > 0.0:
            out[comp] = np.float32(dmax)
    return out


def build_authoritative_river_role_arrays(
    *,
    template_profile: Dict[str, Any],
    river_channel_mask_path: Path | str | None,
    river_guidance_domain_mask_path: Path | str | None = None,
    estuary_clip_mask_path: Path | str | None = None,
    mainstem_mask_path: Path | str | None = None,
    bank_margin_m: float = 3.0,
) -> Dict[str, np.ndarray | float]:
    """Classify authoritative river-support cells into coarse bed/bank roles.

    This is a geometry-first pass for the broad authoritative-raster support export.
    It preserves role information at the first handoff without yet relying on the
    later XS/station semantics.
    """
    shape = (int(template_profile["height"]), int(template_profile["width"]))
    channel_mask = _read_bool_mask_aligned(river_channel_mask_path, template_profile=template_profile)
    guidance_mask = _read_bool_mask_aligned(river_guidance_domain_mask_path, template_profile=template_profile)
    estuary_mask = _read_bool_mask_aligned(estuary_clip_mask_path, template_profile=template_profile)
    mainstem_mask = _read_bool_mask_aligned(mainstem_mask_path, template_profile=template_profile)
    px_size_m = _pixel_size_m(template_profile)
    bank_margin_m = max(float(bank_margin_m or 0.0), 0.0)
    effective_bank_margin_m = max(bank_margin_m + (0.5 * px_size_m), 1.05 * px_size_m)
    core_min_m = max(2.0 * bank_margin_m, 1.5 * px_size_m, effective_bank_margin_m + px_size_m)

    if np.any(channel_mask):
        dist_to_bank_m = ndi.distance_transform_edt(channel_mask, sampling=px_size_m).astype(np.float32)
    else:
        dist_to_bank_m = np.zeros(shape, dtype=np.float32)
    component_half_width_est_m = _component_max_distance(channel_mask, dist_to_bank_m)
    normalized_channel_position = np.zeros(shape, dtype=np.float32)
    valid_norm = channel_mask & np.isfinite(component_half_width_est_m) & (component_half_width_est_m > 0.0)
    normalized_channel_position[valid_norm] = (
        dist_to_bank_m[valid_norm] / component_half_width_est_m[valid_norm]
    ).astype(np.float32)
    normalized_channel_position = np.clip(normalized_channel_position, 0.0, 1.0).astype(np.float32)

    role = np.full(shape, ROLE_AMBIGUOUS, dtype=object)
    role_confidence = np.zeros(shape, dtype=np.float32)

    in_channel = channel_mask & ~estuary_mask
    bankish = in_channel & (
        (~guidance_mask & channel_mask)
        | (dist_to_bank_m <= effective_bank_margin_m)
        | (normalized_channel_position <= 0.25)
    )
    coreish = in_channel & (~bankish) & (dist_to_bank_m >= core_min_m) & (normalized_channel_position >= 0.55)
    innerish = in_channel & (~bankish) & (~coreish)

    role[bankish] = ROLE_BANK_MARGIN
    role[innerish] = ROLE_BED_INNER
    role[coreish] = ROLE_BED_CORE

    role_confidence[bankish] = 0.70
    role_confidence[innerish] = 0.72
    role_confidence[coreish] = 0.86
    role_confidence[estuary_mask & channel_mask] = 0.25
    role_confidence[~channel_mask] = 0.10
    role_confidence[mainstem_mask & coreish] = np.maximum(role_confidence[mainstem_mask & coreish], 0.90)

    return {
        "role": role,
        "role_confidence": role_confidence.astype(np.float32),
        "distance_to_bank_m": dist_to_bank_m.astype(np.float32),
        "component_half_width_est_m": component_half_width_est_m.astype(np.float32),
        "normalized_channel_position": normalized_channel_position.astype(np.float32),
        "inside_channel_mask": channel_mask.astype(np.uint8),
        "inside_river_guidance_domain": guidance_mask.astype(np.uint8),
        "inside_mainstem_mask": mainstem_mask.astype(np.uint8),
        "inside_estuary_clip": estuary_mask.astype(np.uint8),
        "pixel_size_m": float(px_size_m),
        "bank_margin_m": float(bank_margin_m),
        "effective_bank_margin_m": float(effective_bank_margin_m),
        "core_min_m": float(core_min_m),
    }


__all__ = [
    "ROLE_BED_CORE",
    "ROLE_BED_INNER",
    "ROLE_BANK_MARGIN",
    "ROLE_AMBIGUOUS",
    "ROLE_TO_CODE",
    "CODE_TO_ROLE",
    "role_to_code",
    "code_to_role",
    "build_authoritative_river_role_arrays",
]
