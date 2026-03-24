from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
from pathlib import Path
import logging

import numpy as np

from provenance_schema import ProvenanceClass
from support_classes import SupportClass, build_regime_masks, regime_array_from_masks
from river_bank_guidance import (
    compute_bank_distance_influence,
    compute_bank_elevation_surface_from_authoritative,
)

log = logging.getLogger("terrain_interpolator")


@dataclass(frozen=True)
class TerrainInterpolationConfig:
    pixel_size_m: float
    support_decay_m: float = 300.0
    support_density_radius_m: float = 250.0
    coastal_sdb_support_transition_m: float = 600.0
    river_anchor_density_radius_m: float = 200.0
    river_scaffold_transition_m: float = 800.0
    anchor_uncertainty_floor_m: float = 0.10
    anchor_uncertainty_growth_per_m: float = 0.002
    river_aniso_along_scale_m: float = 500.0
    river_aniso_cross_scale_m: float = 30.0
    use_inverse_variance_blend_when_available: bool = True

    def sanitized(self) -> "TerrainInterpolationConfig":
        safe_pixel = max(float(self.pixel_size_m or 0.0), 1.0)
        return TerrainInterpolationConfig(
            pixel_size_m=safe_pixel,
            support_decay_m=max(float(self.support_decay_m or 0.0), safe_pixel),
            support_density_radius_m=max(float(self.support_density_radius_m or 0.0), safe_pixel),
            coastal_sdb_support_transition_m=max(float(self.coastal_sdb_support_transition_m or 0.0), 1.0),
            river_anchor_density_radius_m=max(float(self.river_anchor_density_radius_m or 0.0), safe_pixel),
            river_scaffold_transition_m=max(float(self.river_scaffold_transition_m or 0.0), safe_pixel),
            anchor_uncertainty_floor_m=max(float(self.anchor_uncertainty_floor_m or 0.0), 0.0),
            anchor_uncertainty_growth_per_m=max(float(self.anchor_uncertainty_growth_per_m or 0.0), 0.0),
            river_aniso_along_scale_m=max(float(self.river_aniso_along_scale_m or 0.0), safe_pixel),
            river_aniso_cross_scale_m=max(float(self.river_aniso_cross_scale_m or 0.0), max(0.5 * safe_pixel, 1.0)),
            use_inverse_variance_blend_when_available=bool(self.use_inverse_variance_blend_when_available),
        )

    def as_dict(self) -> Dict[str, float]:
        return asdict(self.sanitized())


@dataclass(frozen=True)
class TerrainInterpolationInputs:
    auth: np.ndarray
    sdb_ok: np.ndarray
    river_ok: np.ndarray
    candidate: Optional[np.ndarray] = None
    sdb_depth_guidance: Optional[np.ndarray] = None
    river_depth_guidance: Optional[np.ndarray] = None
    sdb_guide_points_path: Optional[str] = None
    river_guide_points_path: Optional[str] = None
    guidance_template_raster: Optional[str] = None
    sdb_gw: Optional[np.ndarray] = None
    sdb_ti: Optional[np.ndarray] = None
    river_gw: Optional[np.ndarray] = None
    river_ti: Optional[np.ndarray] = None
    river_support: Optional[np.ndarray] = None
    river_support_depth: Optional[np.ndarray] = None
    estuary_transition: Optional[np.ndarray] = None
    river_corridor_mask: Optional[np.ndarray] = None
    river_bank_influence: Optional[np.ndarray] = None
    river_bank_elevation: Optional[np.ndarray] = None
    river_bank_pair_weight: Optional[np.ndarray] = None
    river_bank_continuity_weight: Optional[np.ndarray] = None
    river_bank_graph_confidence: Optional[np.ndarray] = None
    river_bank_confluence_damping: Optional[np.ndarray] = None
    river_bank_estuary_side_decay: Optional[np.ndarray] = None
    river_centerline_elevation: Optional[np.ndarray] = None
    river_centerline_influence: Optional[np.ndarray] = None
    river_centerline_stationing: Optional[np.ndarray] = None
    river_longitudinal_profile_elevation: Optional[np.ndarray] = None
    river_longitudinal_profile_uncertainty: Optional[np.ndarray] = None
    river_longitudinal_profile_influence: Optional[np.ndarray] = None
    river_xs_support_elevation: Optional[np.ndarray] = None
    river_xs_support_weight: Optional[np.ndarray] = None
    sdb_uncertainty: Optional[np.ndarray] = None
    river_uncertainty: Optional[np.ndarray] = None


def _as_float32(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return None if arr is None else np.asarray(arr, dtype=np.float32)




def _float32_or_nan(shape: tuple[int, ...], arr: Optional[np.ndarray]) -> np.ndarray:
    if arr is None:
        return np.full(shape, np.nan, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def _float32_or_zero(shape: tuple[int, ...], arr: Optional[np.ndarray]) -> np.ndarray:
    if arr is None:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)




def _resolve_native_guidance_arrays(inputs: TerrainInterpolationInputs) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    sdb_depth_guidance = _as_float32(inputs.sdb_depth_guidance)
    river_depth_guidance = _as_float32(inputs.river_depth_guidance)
    template = inputs.guidance_template_raster

    if (inputs.sdb_guide_points_path or inputs.river_guide_points_path) and not template:
        raise ValueError("guidance_template_raster is required when native guide-point paths are provided")

    if template:
        template_path = Path(template)
        if (inputs.sdb_guide_points_path or inputs.river_guide_points_path) and (not template_path.exists()):
            raise ValueError(f"guidance_template_raster does not exist: {template_path}")
        if inputs.sdb_guide_points_path and sdb_depth_guidance is None:
            try:
                from sdb_guidance import rasterize_sdb_guide_points_to_template
                sdb_depth_guidance = _as_float32(rasterize_sdb_guide_points_to_template(inputs.sdb_guide_points_path, template_path, logger=log))
            except Exception:
                log.debug("[TERRAIN] Failed to rasterize native SDB guide points inside interpolator.", exc_info=True)
        if inputs.river_guide_points_path and river_depth_guidance is None:
            try:
                from river_guidance import rasterize_river_guide_points_to_template
                river_depth_guidance = _as_float32(rasterize_river_guide_points_to_template(inputs.river_guide_points_path, template_path, logger=log))
            except Exception:
                log.debug("[TERRAIN] Failed to rasterize native river guide points inside interpolator.", exc_info=True)
    return sdb_depth_guidance, river_depth_guidance

def _validate_shapes(inputs: TerrainInterpolationInputs) -> None:
    expected = np.asarray(inputs.auth).shape
    named = {
        "candidate": inputs.candidate,
        "sdb_depth_guidance": inputs.sdb_depth_guidance,
        "river_depth_guidance": inputs.river_depth_guidance,
        "auth": inputs.auth,
        "sdb_ok": inputs.sdb_ok,
        "river_ok": inputs.river_ok,
        "sdb_gw": inputs.sdb_gw,
        "sdb_ti": inputs.sdb_ti,
        "river_gw": inputs.river_gw,
        "river_ti": inputs.river_ti,
        "river_support": inputs.river_support,
        "river_support_depth": inputs.river_support_depth,
        "estuary_transition": inputs.estuary_transition,
        "river_corridor_mask": inputs.river_corridor_mask,
        "river_bank_influence": inputs.river_bank_influence,
        "river_bank_elevation": inputs.river_bank_elevation,
        "river_bank_pair_weight": inputs.river_bank_pair_weight,
        "river_bank_continuity_weight": inputs.river_bank_continuity_weight,
        "river_bank_graph_confidence": inputs.river_bank_graph_confidence,
        "river_bank_confluence_damping": inputs.river_bank_confluence_damping,
        "river_bank_estuary_side_decay": inputs.river_bank_estuary_side_decay,
        "river_centerline_elevation": inputs.river_centerline_elevation,
        "river_centerline_influence": inputs.river_centerline_influence,
        "river_centerline_stationing": inputs.river_centerline_stationing,
        "river_longitudinal_profile_elevation": inputs.river_longitudinal_profile_elevation,
        "river_longitudinal_profile_uncertainty": inputs.river_longitudinal_profile_uncertainty,
        "river_longitudinal_profile_influence": inputs.river_longitudinal_profile_influence,
        "river_xs_support_elevation": inputs.river_xs_support_elevation,
        "river_xs_support_weight": inputs.river_xs_support_weight,
        "sdb_uncertainty": inputs.sdb_uncertainty,
        "river_uncertainty": inputs.river_uncertainty,
    }
    for name, arr in named.items():
        if arr is None:
            continue
        arr_shape = np.asarray(arr).shape
        if arr_shape != expected:
            raise ValueError(f"{name} shape {arr_shape} does not match expected shape {expected}")



def _validate_resolved_guidance_shapes(expected: tuple[int, ...], **named_arrays: Optional[np.ndarray]) -> None:
    for name, arr in named_arrays.items():
        if arr is None:
            continue
        arr_shape = np.asarray(arr).shape
        if arr_shape != expected:
            raise ValueError(f"{name} resolved shape {arr_shape} does not match expected shape {expected}")

def compute_support_distance_density_guidance(
    locked: np.ndarray,
    auth: np.ndarray,
    *,
    pixel_size_m: float,
    support_decay_m: float,
    density_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import distance_transform_edt, uniform_filter

    locked = np.asarray(locked, dtype=bool)
    auth = np.asarray(auth, dtype=np.float32)
    safe_pixel = max(float(pixel_size_m or 0.0), 1.0)
    safe_decay = max(float(support_decay_m or 0.0), safe_pixel)
    safe_radius = max(float(density_radius_m or 0.0), safe_pixel)

    if locked.size == 0:
        empty = np.zeros_like(auth, dtype=np.float32)
        return empty, empty, empty, empty

    if np.any(locked):
        dist_px, nearest = distance_transform_edt(~locked, return_indices=True)
        support_distance_m = (dist_px.astype(np.float32) * safe_pixel).astype(np.float32)
        nearest_auth = auth[nearest[0], nearest[1]].astype(np.float32)
    else:
        support_distance_m = np.full(auth.shape, np.inf, dtype=np.float32)
        nearest_auth = np.full(auth.shape, np.nan, dtype=np.float32)

    density_px = max(int(round(safe_radius / safe_pixel)), 1)
    win = (2 * density_px) + 1
    support_density = uniform_filter(locked.astype(np.float32), size=win, mode="nearest").astype(np.float32)
    support_density = np.clip(support_density, 0.0, 1.0)

    dist_factor = 1.0 - np.exp(-np.nan_to_num(support_distance_m, nan=np.inf, posinf=np.inf) / safe_decay)
    density_factor = 1.0 - support_density
    guidance_influence = np.clip(0.05 + 0.95 * (0.65 * dist_factor + 0.35 * density_factor), 0.0, 1.0).astype(np.float32)
    guidance_influence[locked] = 0.0
    return support_distance_m, support_density, guidance_influence, nearest_auth


def compute_river_anchor_support_fields(
    *,
    river_anchor: np.ndarray,
    river_guidance_weight: Optional[np.ndarray],
    river_domain: np.ndarray,
    pixel_size_m: float,
    density_radius_m: float,
    scaffold_transition_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import distance_transform_edt, uniform_filter

    river_anchor = np.asarray(river_anchor, dtype=bool)
    river_domain = np.asarray(river_domain, dtype=bool)
    safe_pixel = max(float(pixel_size_m or 0.0), 1.0)
    safe_radius = max(float(density_radius_m or 0.0), safe_pixel)
    safe_transition = max(float(scaffold_transition_m or 0.0), safe_pixel)

    if np.any(river_anchor):
        dist_px = distance_transform_edt(~river_anchor)
        river_anchor_distance_m = (dist_px.astype(np.float32) * safe_pixel).astype(np.float32)
    else:
        river_anchor_distance_m = np.full(river_domain.shape, np.inf, dtype=np.float32)

    density_px = max(int(round(safe_radius / safe_pixel)), 1)
    win = (2 * density_px) + 1
    river_anchor_density = uniform_filter(river_anchor.astype(np.float32), size=win, mode="nearest").astype(np.float32)
    river_anchor_density = np.clip(river_anchor_density, 0.0, 1.0)

    if river_guidance_weight is not None:
        river_guidance_weight = np.clip(np.nan_to_num(river_guidance_weight, nan=0.0), 0.0, 1.0).astype(np.float32)
    else:
        river_guidance_weight = np.zeros(river_domain.shape, dtype=np.float32)
    dist_factor = 1.0 - np.exp(-np.nan_to_num(river_anchor_distance_m, nan=np.inf, posinf=np.inf) / safe_transition)
    density_factor = 1.0 - river_anchor_density
    scaffold = np.clip((0.55 * dist_factor) + (0.30 * density_factor) + (0.15 * river_guidance_weight), 0.0, 1.0).astype(np.float32)
    scaffold[river_anchor] = 0.0
    scaffold[~river_domain] = 0.0
    return river_anchor_distance_m, river_anchor_density, scaffold




def _nearest_surface_from_values(valid_mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    valid_mask = np.asarray(valid_mask, dtype=bool)
    values = np.asarray(values, dtype=np.float32)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if not np.any(valid_mask):
        return out
    _, nearest = distance_transform_edt(~valid_mask, return_indices=True)
    out[...] = values[nearest[0], nearest[1]].astype(np.float32)
    return out


def _nearest_surface_within_domain(
    valid_mask: np.ndarray,
    values: np.ndarray,
    domain_mask: np.ndarray,
    *,
    centerline_mask: Optional[np.ndarray] = None,
    stationing_raster: Optional[np.ndarray] = None,
    along_scale_m: float = 500.0,
    cross_scale_m: float = 30.0,
    pixel_size_m: float = 1.0,
) -> np.ndarray:
    """Nearest-value handoff constrained to a logical domain.

    If a centerline transport mask and stationing raster are available, values
    are projected to the centerline and propagated primarily by along-channel
    station distance with a secondary source-side lateral penalty. This is more
    river-appropriate than pure Euclidean nearest in meanders and across-bank
    situations.
    """
    from scipy.ndimage import distance_transform_edt

    domain_mask = np.asarray(domain_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool) & domain_mask
    values = np.asarray(values, dtype=np.float32)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if not np.any(valid_mask):
        return out

    centerline_ok = None
    if centerline_mask is not None:
        centerline_ok = np.asarray(centerline_mask, dtype=bool) & domain_mask
        if not np.any(centerline_ok):
            centerline_ok = None

    stationing = None if stationing_raster is None else np.asarray(stationing_raster, dtype=np.float32)
    if centerline_ok is None or stationing is None or not np.any(np.isfinite(stationing[centerline_ok])):
        domain_out = _nearest_surface_from_values(valid_mask, values)
        out[domain_mask] = domain_out[domain_mask]
        return out

    dist_to_centerline, nearest_centerline = distance_transform_edt(~centerline_ok, return_indices=True)
    src_rows, src_cols = np.nonzero(valid_mask)
    proj_rows = nearest_centerline[0][src_rows, src_cols]
    proj_cols = nearest_centerline[1][src_rows, src_cols]
    src_station = stationing[proj_rows, proj_cols].astype(np.float32)
    src_lateral = (dist_to_centerline[src_rows, src_cols].astype(np.float32) * np.float32(max(float(pixel_size_m), 1e-6)))
    src_vals = values[src_rows, src_cols].astype(np.float32)
    finite_src = np.isfinite(src_station) & np.isfinite(src_vals)
    if not np.any(finite_src):
        domain_out = _nearest_surface_from_values(valid_mask, values)
        out[domain_mask] = domain_out[domain_mask]
        return out
    src_station = src_station[finite_src]
    src_lateral = src_lateral[finite_src]
    src_vals = src_vals[finite_src]

    order = np.argsort(src_station, kind='mergesort')
    src_station = src_station[order]
    src_lateral = src_lateral[order]
    src_vals = src_vals[order]

    center_rows, center_cols = np.nonzero(centerline_ok & np.isfinite(stationing))
    if center_rows.size == 0:
        domain_out = _nearest_surface_from_values(valid_mask, values)
        out[domain_mask] = domain_out[domain_mask]
        return out

    q_station = stationing[center_rows, center_cols].astype(np.float32)
    ins = np.searchsorted(src_station, q_station)
    k = min(8, int(src_station.size))
    safe_along = max(float(along_scale_m), 1.0)
    safe_cross = max(float(cross_scale_m), 1.0)
    centerline_vals = np.full(values.shape, np.nan, dtype=np.float32)
    for rr, cc, sq, ii in zip(center_rows.tolist(), center_cols.tolist(), q_station.tolist(), ins.tolist()):
        lo = max(0, ii - k)
        hi = min(int(src_station.size), ii + k)
        cand_s = src_station[lo:hi]
        cand_lat = src_lateral[lo:hi]
        cand_val = src_vals[lo:hi]
        if cand_s.size == 0:
            continue
        d_eff = np.sqrt(((np.abs(cand_s - sq) / safe_along) ** 2) + ((cand_lat / safe_cross) ** 2))
        centerline_vals[rr, cc] = np.float32(cand_val[int(np.argmin(d_eff))])

    centerline_valid = centerline_ok & np.isfinite(centerline_vals)
    if not np.any(centerline_valid):
        domain_out = _nearest_surface_from_values(valid_mask, values)
        out[domain_mask] = domain_out[domain_mask]
        return out

    routed = _nearest_surface_from_values(centerline_valid, centerline_vals)
    out[domain_mask] = routed[domain_mask]
    return out




def _confidence_to_variance_ratio(confidence: np.ndarray) -> np.ndarray:
    conf = np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0).astype(np.float32)
    ratio = np.full(conf.shape, np.inf, dtype=np.float32)
    positive = conf > 0.0
    if np.any(positive):
        safe_conf = np.clip(conf[positive], np.float32(1.0e-6), np.float32(1.0 - 1.0e-6)).astype(np.float32)
        ratio[positive] = ((1.0 - safe_conf) / safe_conf).astype(np.float32)
    return ratio


def _variance_ratio_to_confidence(variance_ratio: np.ndarray) -> np.ndarray:
    ratio = np.asarray(variance_ratio, dtype=np.float32)
    conf = np.zeros(ratio.shape, dtype=np.float32)
    finite = np.isfinite(ratio) & (ratio >= 0.0)
    if np.any(finite):
        conf[finite] = (1.0 / (1.0 + ratio[finite])).astype(np.float32)
    conf[np.isinf(ratio)] = 0.0
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def combine_structural_confidence(*confidence_cues: Optional[np.ndarray]) -> np.ndarray:
    available: list[np.ndarray] = []
    shape = None
    for cue in confidence_cues:
        if cue is None:
            continue
        arr = np.clip(np.nan_to_num(np.asarray(cue, dtype=np.float32), nan=0.0), 0.0, 1.0).astype(np.float32)
        if shape is None:
            shape = arr.shape
        positive = arr > 0.0
        if np.any(positive):
            ratio = _confidence_to_variance_ratio(arr)
            ratio[~positive] = np.nan
            available.append(ratio)
    if shape is None:
        raise ValueError('combine_structural_confidence requires at least one array-shaped cue')
    if not available:
        return np.zeros(shape, dtype=np.float32)
    stacked = np.stack(available, axis=0).astype(np.float32)
    finite = np.isfinite(stacked)
    count = np.sum(finite, axis=0).astype(np.int32)
    total = np.where(finite, stacked, 0.0).sum(axis=0, dtype=np.float32)
    mean_ratio = np.full(shape, np.inf, dtype=np.float32)
    use = count > 0
    if np.any(use):
        mean_ratio[use] = (total[use] / count[use].astype(np.float32)).astype(np.float32)
    return _variance_ratio_to_confidence(mean_ratio)


def confidence_to_guidance_uncertainty(
    *,
    anchor_uncertainty: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    sigma_anchor = np.asarray(anchor_uncertainty, dtype=np.float32)
    ratio = _confidence_to_variance_ratio(confidence)
    out = np.full(sigma_anchor.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(sigma_anchor) & np.isfinite(ratio)
    if np.any(valid):
        out[valid] = (sigma_anchor[valid] * np.sqrt(ratio[valid])).astype(np.float32)
    zero_conf = np.isfinite(sigma_anchor) & np.isinf(ratio)
    if np.any(zero_conf):
        out[zero_conf] = np.inf
    return out.astype(np.float32)


def _normalized_distance(distance_m: Optional[np.ndarray], scale_m: float, *, default_when_missing: float = 1.0) -> np.ndarray:
    safe_scale = max(float(scale_m or 0.0), 1.0)
    if distance_m is None:
        return np.full((), np.float32(default_when_missing), dtype=np.float32)
    arr = np.nan_to_num(np.asarray(distance_m, dtype=np.float32), nan=safe_scale * 3.0, posinf=safe_scale * 3.0).astype(np.float32)
    return np.clip(arr / safe_scale, 0.0, 3.0).astype(np.float32)



def compute_value_spread(*value_arrays: Optional[np.ndarray]) -> np.ndarray:
    shape = None
    arrays: list[np.ndarray] = []
    for arr in value_arrays:
        if arr is None:
            continue
        vals = np.asarray(arr, dtype=np.float32)
        if shape is None:
            shape = vals.shape
        arrays.append(vals)
    if shape is None:
        raise ValueError('compute_value_spread requires at least one array-shaped value input')
    if not arrays:
        return np.zeros(shape, dtype=np.float32)
    stack = np.stack(arrays, axis=0).astype(np.float32)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0).astype(np.int32)
    spread = np.zeros(shape, dtype=np.float32)
    use = count > 1
    if np.any(use):
        safe = np.where(finite, stack, np.nan).astype(np.float32)
        spread[use] = np.nanstd(safe[:, use], axis=0).astype(np.float32)
    spread[~np.isfinite(spread)] = 0.0
    return spread.astype(np.float32)



def compute_physically_explicit_guidance_uncertainty(
    *,
    anchor_uncertainty: np.ndarray,
    distance_m: Optional[np.ndarray],
    transition_m: float,
    support_density: Optional[np.ndarray],
    quality: Optional[np.ndarray],
    disagreement_m: Optional[np.ndarray],
    floor_m: float,
    distance_weight_m: float,
    density_weight_m: float,
    quality_weight_m: float,
    disagreement_weight: float,
    min_anchor_fraction: float,
    extra_penalty_m: Optional[np.ndarray] = None,
) -> np.ndarray:
    sigma_anchor = np.asarray(anchor_uncertainty, dtype=np.float32)
    out = np.full(sigma_anchor.shape, np.nan, dtype=np.float32)
    dist_term = _normalized_distance(distance_m, transition_m)
    if np.ndim(dist_term) == 0:
        dist_term = np.full(sigma_anchor.shape, dist_term, dtype=np.float32)
    if support_density is None:
        density_term = np.ones(sigma_anchor.shape, dtype=np.float32)
    else:
        density_term = (1.0 - np.clip(np.nan_to_num(np.asarray(support_density, dtype=np.float32), nan=0.0), 0.0, 1.0)).astype(np.float32)
    if quality is None:
        quality_term = np.ones(sigma_anchor.shape, dtype=np.float32)
    else:
        quality_term = (1.0 - np.clip(np.nan_to_num(np.asarray(quality, dtype=np.float32), nan=0.0), 0.0, 1.0)).astype(np.float32)
    if disagreement_m is None:
        spread_term = np.zeros(sigma_anchor.shape, dtype=np.float32)
    else:
        spread_term = np.nan_to_num(np.asarray(disagreement_m, dtype=np.float32), nan=0.0, posinf=0.0).astype(np.float32)
    sigma = (
        float(floor_m)
        + (float(distance_weight_m) * dist_term)
        + (float(density_weight_m) * density_term)
        + (float(quality_weight_m) * quality_term)
        + (float(disagreement_weight) * spread_term)
    ).astype(np.float32)
    if extra_penalty_m is not None:
        sigma = (sigma + np.nan_to_num(np.asarray(extra_penalty_m, dtype=np.float32), nan=0.0, posinf=0.0)).astype(np.float32)
    finite_anchor = np.isfinite(sigma_anchor)
    if np.any(finite_anchor):
        sigma[finite_anchor] = np.maximum(sigma[finite_anchor], (sigma_anchor[finite_anchor] * float(min_anchor_fraction)).astype(np.float32))
    sigma[~np.isfinite(sigma)] = np.nan
    return sigma.astype(np.float32)


def compute_anchor_uncertainty(
    support_distance_m: np.ndarray,
    *,
    locked: np.ndarray,
    floor_m: float,
    growth_per_m: float,
) -> np.ndarray:
    safe_dist = np.nan_to_num(support_distance_m, nan=np.inf, posinf=np.inf).astype(np.float32)
    out = (float(floor_m) + (safe_dist * float(growth_per_m))).astype(np.float32)
    out[locked] = 0.0
    return out


def combine_conditioning_uncertainty(
    *,
    anchor_uncertainty: np.ndarray,
    guidance_uncertainty: np.ndarray,
    guidance_influence: np.ndarray,
    conditioned: np.ndarray,
    locked: np.ndarray,
) -> np.ndarray:
    blend_w = np.clip(np.nan_to_num(guidance_influence, nan=0.0), 0.0, 1.0).astype(np.float32)
    sigma_anchor = np.asarray(anchor_uncertainty, dtype=np.float32)
    sigma_guidance = np.asarray(guidance_uncertainty, dtype=np.float32)
    anchor_term = np.zeros_like(sigma_anchor, dtype=np.float32)
    guidance_term = np.zeros_like(sigma_guidance, dtype=np.float32)
    use_anchor = np.isfinite(sigma_anchor) & (blend_w < 1.0)
    use_guidance = np.isfinite(sigma_guidance) & (blend_w > 0.0)
    anchor_term[use_anchor] = (((1.0 - blend_w[use_anchor]) ** 2) * (sigma_anchor[use_anchor] ** 2)).astype(np.float32)
    guidance_term[use_guidance] = ((blend_w[use_guidance] ** 2) * (sigma_guidance[use_guidance] ** 2)).astype(np.float32)
    combined = np.sqrt(anchor_term + guidance_term).astype(np.float32)
    missing = (~use_anchor & (blend_w < 1.0)) | (~use_guidance & (blend_w > 0.0))
    combined[missing & np.isfinite(conditioned)] = np.nan
    combined[locked] = 0.0
    combined[~np.isfinite(conditioned)] = np.nan
    return combined.astype(np.float32)



def compute_inverse_variance_blend_weights(
    *,
    anchor_uncertainty: np.ndarray,
    guidance_uncertainty: np.ndarray,
    fallback_guidance_influence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-level blend weights and availability mask.

    Weight is the guidance-side blend fraction. When both uncertainties are finite,
    use inverse-variance weighting: w = sigma_anchor^2 / (sigma_anchor^2 + sigma_guidance^2).
    Otherwise fall back to the provided heuristic influence.
    """
    fallback = np.clip(np.nan_to_num(fallback_guidance_influence, nan=0.0), 0.0, 1.0).astype(np.float32)
    sigma_anchor = np.asarray(anchor_uncertainty, dtype=np.float32)
    sigma_guidance = np.asarray(guidance_uncertainty, dtype=np.float32)
    have_both = np.isfinite(sigma_anchor) & np.isfinite(sigma_guidance)
    out = fallback.copy()
    used_inverse_variance = np.zeros_like(have_both, dtype=bool)
    if np.any(have_both):
        var_anchor = np.square(sigma_anchor[have_both], dtype=np.float32)
        var_guidance = np.square(sigma_guidance[have_both], dtype=np.float32)
        denom = var_anchor + var_guidance
        valid = denom > 0.0
        if np.any(valid):
            idx = np.flatnonzero(have_both)
            tgt = idx[valid]
            out.flat[tgt] = (var_anchor[valid] / denom[valid]).astype(np.float32)
            used_inverse_variance.flat[tgt] = True
        if np.any(~valid):
            # If both variances are zero, preserve the fallback heuristic rather than divide by zero.
            pass
    return np.clip(out, 0.0, 1.0).astype(np.float32), used_inverse_variance

def compute_coastal_sdb_support_confidence(
    *,
    sdb_domain: np.ndarray,
    sdb_guidance_weight: Optional[np.ndarray],
    sdb_trusted_interior: Optional[np.ndarray],
    support_distance_m: np.ndarray,
    support_density: np.ndarray,
    support_transition_m: float = 600.0,
) -> np.ndarray:
    sdb_domain = np.asarray(sdb_domain, dtype=bool)
    out = np.zeros(sdb_domain.shape, dtype=np.float32)
    if out.size == 0 or not np.any(sdb_domain):
        return out

    optical = np.clip(np.nan_to_num(sdb_guidance_weight, nan=0.0), 0.0, 1.0).astype(np.float32) if sdb_guidance_weight is not None else np.zeros(sdb_domain.shape, dtype=np.float32)
    if sdb_trusted_interior is not None:
        optical = np.maximum(optical, 0.70 * (np.asarray(sdb_trusted_interior) > 0).astype(np.float32))
    safe_transition = max(float(support_transition_m or 0.0), 1.0)
    safe_dist = np.nan_to_num(support_distance_m, nan=np.inf, posinf=np.inf).astype(np.float32)
    dist_factor = (1.0 - np.exp(-safe_dist / safe_transition)).astype(np.float32)
    density_factor = (1.0 - np.clip(np.nan_to_num(support_density, nan=0.0), 0.0, 1.0)).astype(np.float32)
    conf = np.clip((0.20 + 0.80 * optical) * (0.30 + 0.70 * dist_factor) * (0.25 + 0.75 * density_factor), 0.0, 1.0).astype(np.float32)
    conf[~sdb_domain] = 0.0
    return conf


def interpolate_support_aware_surface(
    *,
    inputs: TerrainInterpolationInputs,
    config: TerrainInterpolationConfig,
) -> Dict[str, Any]:
    cfg = config.sanitized()
    _validate_shapes(inputs)

    auth = np.asarray(inputs.auth, dtype=np.float32)
    candidate = _as_float32(inputs.candidate)
    sdb_depth_guidance, river_depth_guidance = _resolve_native_guidance_arrays(inputs)
    work_shape = auth.shape
    _validate_resolved_guidance_shapes(
        work_shape,
        sdb_depth_guidance=sdb_depth_guidance,
        river_depth_guidance=river_depth_guidance,
    )
    sdb_ok = np.asarray(inputs.sdb_ok, dtype=bool)
    river_ok = np.asarray(inputs.river_ok, dtype=bool)
    sdb_gw = _as_float32(inputs.sdb_gw)
    sdb_ti = None if inputs.sdb_ti is None else np.asarray(inputs.sdb_ti)
    river_gw = _as_float32(inputs.river_gw)
    river_ti = None if inputs.river_ti is None else np.asarray(inputs.river_ti)
    river_support = None if inputs.river_support is None else np.asarray(inputs.river_support)
    river_support_depth = _as_float32(inputs.river_support_depth)
    river_corridor_mask = np.asarray(inputs.river_corridor_mask, dtype=bool) if inputs.river_corridor_mask is not None else None
    river_bank_influence_input = _as_float32(inputs.river_bank_influence)
    river_bank_elevation_input = _as_float32(inputs.river_bank_elevation)
    river_bank_pair_weight = _as_float32(inputs.river_bank_pair_weight)
    river_bank_continuity_weight = _as_float32(inputs.river_bank_continuity_weight)
    river_bank_graph_confidence = _as_float32(inputs.river_bank_graph_confidence)
    river_bank_confluence_damping = _as_float32(inputs.river_bank_confluence_damping)
    river_bank_estuary_side_decay = _as_float32(inputs.river_bank_estuary_side_decay)
    river_centerline_elevation = _float32_or_nan(work_shape, inputs.river_centerline_elevation)
    river_centerline_influence = _float32_or_zero(work_shape, inputs.river_centerline_influence)
    river_centerline_stationing = _float32_or_nan(work_shape, inputs.river_centerline_stationing)
    river_longitudinal_profile_elevation = _float32_or_nan(work_shape, inputs.river_longitudinal_profile_elevation)
    river_longitudinal_profile_uncertainty = _as_float32(inputs.river_longitudinal_profile_uncertainty)
    river_longitudinal_profile_influence = _float32_or_zero(work_shape, inputs.river_longitudinal_profile_influence)
    river_xs_support_elevation = _float32_or_nan(work_shape, inputs.river_xs_support_elevation)
    river_xs_support_weight = _float32_or_zero(work_shape, inputs.river_xs_support_weight)
    sdb_uncertainty = _as_float32(inputs.sdb_uncertainty)
    river_uncertainty = _as_float32(inputs.river_uncertainty)

    has_river_centerline_elevation = inputs.river_centerline_elevation is not None
    has_river_longitudinal_profile_elevation = inputs.river_longitudinal_profile_elevation is not None
    has_river_centerline_influence = inputs.river_centerline_influence is not None
    has_river_xs_support_elevation = inputs.river_xs_support_elevation is not None
    has_river_xs_support_weight = inputs.river_xs_support_weight is not None

    locked = np.isfinite(auth)
    gap = ~locked
    estuary_transition = (
        np.asarray(inputs.estuary_transition, dtype=bool)
        if inputs.estuary_transition is not None
        else np.zeros_like(gap, dtype=bool)
    )

    support = np.zeros_like(auth, dtype=np.uint8)
    support[locked] = int(SupportClass.AUTHORITATIVE_LOCKED)
    support[gap] = int(SupportClass.ANCHORED_INTERPOLATION)

    support_distance_m, support_density, base_guidance_influence, nearest_auth = compute_support_distance_density_guidance(
        locked,
        auth,
        pixel_size_m=cfg.pixel_size_m,
        support_decay_m=cfg.support_decay_m,
        density_radius_m=cfg.support_density_radius_m,
    )
    anchor_uncertainty = compute_anchor_uncertainty(
        support_distance_m,
        locked=locked,
        floor_m=cfg.anchor_uncertainty_floor_m,
        growth_per_m=cfg.anchor_uncertainty_growth_per_m,
    )
    coastal_sdb_confidence = compute_coastal_sdb_support_confidence(
        sdb_domain=(gap & sdb_ok),
        sdb_guidance_weight=sdb_gw,
        sdb_trusted_interior=sdb_ti,
        support_distance_m=support_distance_m,
        support_density=support_density,
        support_transition_m=cfg.coastal_sdb_support_transition_m,
    )

    river_anchor = ((np.asarray(river_support) > 0) if river_support is not None else np.zeros_like(gap, dtype=bool))
    if (not np.any(river_anchor)) and river_ti is not None:
        river_anchor = np.asarray(river_ti) > 0
    river_corridor = np.asarray(river_corridor_mask, dtype=bool) if river_corridor_mask is not None else np.asarray(river_ok, dtype=bool)
    river_domain = gap & river_ok
    river_bank_edge, river_bank_distance_m, river_bank_influence = compute_bank_distance_influence(
        river_corridor,
        pixel_size_m=cfg.pixel_size_m,
        full_influence_m=0.0,
        zero_influence_m=max(cfg.river_scaffold_transition_m * 0.18, 60.0),
    )
    if river_bank_influence_input is not None:
        river_bank_influence = np.maximum(river_bank_influence.astype(np.float32), np.clip(np.nan_to_num(river_bank_influence_input, nan=0.0), 0.0, 1.0).astype(np.float32))
        river_bank_influence[~river_corridor] = 0.0
    river_bank_elevation = compute_bank_elevation_surface_from_authoritative(
        auth,
        river_corridor,
        max_bank_distance_m=max(cfg.river_scaffold_transition_m * 0.35, 120.0),
        bank_distance_m=river_bank_distance_m,
    )
    if river_bank_elevation_input is not None:
        use_xs_bank = river_corridor & np.isfinite(river_bank_elevation_input)
        if np.any(use_xs_bank):
            river_bank_elevation[use_xs_bank] = river_bank_elevation_input[use_xs_bank].astype(np.float32)
    river_bank_structural_confidence = combine_structural_confidence(
        river_bank_influence,
        river_bank_pair_weight,
        river_bank_continuity_weight,
        river_bank_graph_confidence,
    )
    if river_bank_confluence_damping is not None:
        confluence = np.clip(np.nan_to_num(river_bank_confluence_damping, nan=1.0), 0.0, 1.0).astype(np.float32)
        river_bank_structural_confidence = np.clip(river_bank_structural_confidence * confluence, 0.0, 1.0).astype(np.float32)
    if river_bank_estuary_side_decay is not None:
        est_decay = np.clip(np.nan_to_num(river_bank_estuary_side_decay, nan=1.0), 0.0, 1.0).astype(np.float32)
        river_bank_structural_confidence = np.clip(river_bank_structural_confidence * est_decay, 0.0, 1.0).astype(np.float32)
    river_bank_influence = river_bank_structural_confidence.astype(np.float32)
    river_bank_influence[~river_corridor] = 0.0
    river_anchor_distance_m, river_anchor_density, river_scaffold_confidence = compute_river_anchor_support_fields(
        river_anchor=river_anchor,
        river_guidance_weight=river_gw,
        river_domain=river_domain,
        pixel_size_m=cfg.pixel_size_m,
        density_radius_m=cfg.river_anchor_density_radius_m,
        scaffold_transition_m=cfg.river_scaffold_transition_m,
    )
    river_scaffold_dominant = river_domain & (river_scaffold_confidence >= 0.60)

    support[gap & sdb_ok & ~river_ok] = int(SupportClass.GUIDANCE_CONDITIONED_SDB)
    support[gap & river_ok] = int(SupportClass.GUIDANCE_CONDITIONED_RIVER)
    support[river_scaffold_dominant] = int(SupportClass.SCAFFOLD_INFERRED)

    fallback_guidance_confidence = np.clip(0.20 + 0.80 * base_guidance_influence, 0.0, 1.0).astype(np.float32)
    guidance_influence = fallback_guidance_confidence.copy()
    top_level_guidance_uncertainty = compute_physically_explicit_guidance_uncertainty(
        anchor_uncertainty=anchor_uncertainty,
        distance_m=support_distance_m,
        transition_m=cfg.support_decay_m,
        support_density=support_density,
        quality=fallback_guidance_confidence,
        disagreement_m=None,
        floor_m=max(cfg.anchor_uncertainty_floor_m, 0.15),
        distance_weight_m=0.35,
        density_weight_m=0.25,
        quality_weight_m=0.40,
        disagreement_weight=0.0,
        min_anchor_fraction=0.75,
    )

    if np.any(sdb_ok):
        sdb_local = np.clip(np.nan_to_num(sdb_gw, nan=0.0), 0.0, 1.0).astype(np.float32) if sdb_gw is not None else np.zeros_like(auth, dtype=np.float32)
        if sdb_ti is not None:
            sdb_local = np.maximum(sdb_local, 0.65 * (np.asarray(sdb_ti) > 0).astype(np.float32))
        sdb_domain_confidence = combine_structural_confidence(
            sdb_local,
            coastal_sdb_confidence.astype(np.float32),
        )
        if not np.any(sdb_domain_confidence[gap & sdb_ok] > 0.0):
            sdb_domain_confidence = fallback_guidance_confidence.copy()
        sdb_domain_confidence[~(gap & sdb_ok)] = 0.0
        guidance_influence[sdb_ok] = sdb_domain_confidence[sdb_ok]
        sdb_top_level_uncertainty = compute_physically_explicit_guidance_uncertainty(
            anchor_uncertainty=anchor_uncertainty,
            distance_m=support_distance_m,
            transition_m=cfg.coastal_sdb_support_transition_m,
            support_density=support_density,
            quality=sdb_domain_confidence,
            disagreement_m=None,
            floor_m=max(cfg.anchor_uncertainty_floor_m, 0.20),
            distance_weight_m=0.30,
            density_weight_m=0.25,
            quality_weight_m=0.70,
            disagreement_weight=0.0,
            min_anchor_fraction=0.45,
            extra_penalty_m=np.where(estuary_transition, np.float32(0.15), np.float32(0.0)).astype(np.float32),
        )
        top_level_guidance_uncertainty[sdb_ok] = sdb_top_level_uncertainty[sdb_ok]

    if has_river_centerline_influence:
        river_centerline_influence = np.clip(np.nan_to_num(river_centerline_influence, nan=0.0), 0.0, 1.0).astype(np.float32)
        river_centerline_influence[~river_corridor] = 0.0
    if inputs.river_longitudinal_profile_influence is not None:
        river_longitudinal_profile_influence = np.clip(np.nan_to_num(river_longitudinal_profile_influence, nan=0.0), 0.0, 1.0).astype(np.float32)
        river_longitudinal_profile_influence[~river_corridor] = 0.0
    if has_river_xs_support_weight:
        river_xs_support_weight = np.clip(np.nan_to_num(river_xs_support_weight, nan=0.0), 0.0, 1.0).astype(np.float32)
        river_xs_support_weight[~river_corridor] = 0.0

    river_transport_centerline = river_domain & (
        np.isfinite(river_longitudinal_profile_elevation)
        | np.isfinite(river_centerline_elevation)
        | (river_centerline_influence > 0.0)
        | np.isfinite(river_xs_support_elevation)
        | (river_xs_support_weight > 0.0)
    )
    if not np.any(river_transport_centerline):
        river_transport_centerline = None

    river_structural_spread_m = compute_value_spread(
        river_support_depth,
        river_longitudinal_profile_elevation if has_river_longitudinal_profile_elevation else None,
        river_centerline_elevation if has_river_centerline_elevation else None,
        river_xs_support_elevation if has_river_xs_support_elevation else None,
        river_bank_elevation,
        river_depth_guidance,
    )
    river_structural_spread_m[~river_corridor] = 0.0

    if np.any(river_ok):
        river_local = np.clip(np.nan_to_num(river_gw, nan=0.0), 0.0, 1.0).astype(np.float32) if river_gw is not None else np.clip(base_guidance_influence, 0.0, 1.0).astype(np.float32)
        if river_ti is not None:
            river_local = np.maximum(river_local, 0.70 * (np.asarray(river_ti) > 0).astype(np.float32))
        river_domain_confidence = combine_structural_confidence(
            river_local,
            river_scaffold_confidence.astype(np.float32),
        )
        if not np.any(river_domain_confidence[river_ok] > 0.0):
            river_domain_confidence = fallback_guidance_confidence.copy()
        if np.any(river_corridor):
            river_domain_confidence = np.clip(river_domain_confidence * (1.0 - (0.40 * river_bank_influence)), 0.0, 1.0).astype(np.float32)
        estuary_cap = np.where(estuary_transition, 0.85, 1.0).astype(np.float32)
        river_domain_confidence = np.minimum(river_domain_confidence, estuary_cap).astype(np.float32)
        river_domain_confidence[~river_ok] = 0.0
        # River-domain guidance confidence should be owned by the river-specific logic.
        # Using max(fallback, river_confidence) lets the generic support-distance field
        # overwhelm bank/centerline/estuary adjustments, which makes the river modifiers
        # effectively no-ops in corridor tests.
        guidance_influence[river_ok] = river_domain_confidence[river_ok]
        river_top_level_uncertainty = compute_physically_explicit_guidance_uncertainty(
            anchor_uncertainty=anchor_uncertainty,
            distance_m=river_anchor_distance_m,
            transition_m=cfg.river_scaffold_transition_m,
            support_density=river_anchor_density,
            quality=river_domain_confidence,
            disagreement_m=river_structural_spread_m,
            floor_m=max(cfg.anchor_uncertainty_floor_m, 0.18),
            distance_weight_m=0.30,
            density_weight_m=0.20,
            quality_weight_m=0.45,
            disagreement_weight=0.35,
            min_anchor_fraction=0.40,
            extra_penalty_m=np.where(estuary_transition, np.float32(0.20), np.float32(0.0)).astype(np.float32),
        )
        top_level_guidance_uncertainty[river_ok] = river_top_level_uncertainty[river_ok]

    # Keep scaffold-dominant support-class labeling separate from the active blend weight.
    # Forcing a near-1.0 guidance weight here suppresses the river-bank modifiers and makes
    # continuity / confluence damping ineffective in the actual conditioned surface.
    guidance_influence[locked] = 0.0
    top_level_guidance_uncertainty[locked] = 0.0

    anchor_surface = nearest_auth.astype(np.float32)
    river_channel_anchor_surface = np.full_like(auth, np.nan, dtype=np.float32)

    # River corridors must prefer in-channel support and in-channel hard control over the
    # globally nearest authoritative pixel.  Otherwise nearby bank/topography cells can
    # bleed across the corridor and flatten the channel geometry.
    if river_support_depth is not None:
        river_support_depth = np.asarray(river_support_depth, dtype=np.float32)
        river_support_valid = gap & np.isfinite(river_support_depth) & river_domain
        if np.any(river_support_valid):
            river_channel_anchor_surface = _nearest_surface_within_domain(
                river_support_valid,
                river_support_depth,
                river_domain,
                centerline_mask=river_transport_centerline,
                stationing_raster=river_centerline_stationing,
                along_scale_m=cfg.river_aniso_along_scale_m,
                cross_scale_m=cfg.river_aniso_cross_scale_m,
                pixel_size_m=cfg.pixel_size_m,
            )

    river_authoritative_valid = locked & river_domain & np.isfinite(auth)
    if np.any(river_authoritative_valid):
        river_authoritative_surface = _nearest_surface_within_domain(
            river_authoritative_valid,
            auth,
            river_domain,
            centerline_mask=river_transport_centerline,
            stationing_raster=river_centerline_stationing,
            along_scale_m=cfg.river_aniso_along_scale_m,
            cross_scale_m=cfg.river_aniso_cross_scale_m,
            pixel_size_m=cfg.pixel_size_m,
        )
        use_authoritative_surface = river_domain & ~np.isfinite(river_channel_anchor_surface) & np.isfinite(river_authoritative_surface)
        if np.any(use_authoritative_surface):
            river_channel_anchor_surface[use_authoritative_surface] = river_authoritative_surface[use_authoritative_surface]

    use_river_anchor_surface = river_domain & np.isfinite(river_channel_anchor_surface)
    if np.any(use_river_anchor_surface):
        anchor_surface[use_river_anchor_surface] = river_channel_anchor_surface[use_river_anchor_surface]

    river_bank_constrained = river_domain & (~river_anchor) & np.isfinite(river_bank_elevation) & (river_bank_influence > 0.0)
    river_bank_uncertainty = compute_physically_explicit_guidance_uncertainty(
        anchor_uncertainty=anchor_uncertainty,
        distance_m=river_bank_distance_m,
        transition_m=max(cfg.river_scaffold_transition_m * 0.35, 120.0),
        support_density=river_anchor_density,
        quality=river_bank_influence,
        disagreement_m=river_structural_spread_m,
        floor_m=max(cfg.anchor_uncertainty_floor_m, 0.20),
        distance_weight_m=0.25,
        density_weight_m=0.15,
        quality_weight_m=0.30,
        disagreement_weight=0.35,
        min_anchor_fraction=0.35,
    )
    if np.any(river_bank_constrained):
        edge_mix, _ = compute_inverse_variance_blend_weights(
            anchor_uncertainty=anchor_uncertainty[river_bank_constrained],
            guidance_uncertainty=river_bank_uncertainty[river_bank_constrained],
            fallback_guidance_influence=river_bank_influence[river_bank_constrained],
        )
        anchor_vals = anchor_surface[river_bank_constrained].astype(np.float32)
        bank_vals = river_bank_elevation[river_bank_constrained].astype(np.float32)
        finite_anchor = np.isfinite(anchor_vals)
        mixed_vals = bank_vals.copy()
        mixed_vals[finite_anchor] = (
            ((1.0 - edge_mix[finite_anchor]) * anchor_vals[finite_anchor])
            + (edge_mix[finite_anchor] * bank_vals[finite_anchor])
        ).astype(np.float32)
        anchor_surface[river_bank_constrained] = mixed_vals

    river_longitudinal_profile_constrained = river_domain & (~river_anchor) & np.isfinite(river_longitudinal_profile_elevation)
    if inputs.river_longitudinal_profile_influence is not None:
        river_longitudinal_profile_constrained &= (river_longitudinal_profile_influence > 0.0)
    river_longitudinal_profile_confidence = combine_structural_confidence(
        river_longitudinal_profile_influence if inputs.river_longitudinal_profile_influence is not None else river_scaffold_confidence,
    )
    river_longitudinal_profile_confidence[~river_corridor] = 0.0
    river_longitudinal_profile_sigma = _float32_or_nan(work_shape, river_longitudinal_profile_uncertainty)
    if np.any(river_longitudinal_profile_constrained):
        lp_guidance_uncertainty = np.where(
            np.isfinite(river_longitudinal_profile_sigma),
            river_longitudinal_profile_sigma,
            compute_physically_explicit_guidance_uncertainty(
                anchor_uncertainty=anchor_uncertainty,
                distance_m=river_anchor_distance_m,
                transition_m=cfg.river_scaffold_transition_m,
                support_density=river_anchor_density,
                quality=river_longitudinal_profile_confidence,
                disagreement_m=river_structural_spread_m,
                floor_m=max(cfg.anchor_uncertainty_floor_m, 0.12),
                distance_weight_m=0.20,
                density_weight_m=0.10,
                quality_weight_m=0.20,
                disagreement_weight=0.35,
                min_anchor_fraction=0.25,
            ),
        )
        lp_mix, _ = compute_inverse_variance_blend_weights(
            anchor_uncertainty=anchor_uncertainty[river_longitudinal_profile_constrained],
            guidance_uncertainty=lp_guidance_uncertainty[river_longitudinal_profile_constrained],
            fallback_guidance_influence=river_longitudinal_profile_confidence[river_longitudinal_profile_constrained],
        )
        anchor_vals = anchor_surface[river_longitudinal_profile_constrained].astype(np.float32)
        lp_vals = river_longitudinal_profile_elevation[river_longitudinal_profile_constrained].astype(np.float32)
        finite_anchor = np.isfinite(anchor_vals)
        mixed_vals = lp_vals.copy()
        mixed_vals[finite_anchor] = (((1.0 - lp_mix[finite_anchor]) * anchor_vals[finite_anchor]) + (lp_mix[finite_anchor] * lp_vals[finite_anchor])).astype(np.float32)
        anchor_surface[river_longitudinal_profile_constrained] = mixed_vals

    river_centerline_constrained = river_domain & (~river_anchor) & np.isfinite(river_centerline_elevation)
    if has_river_centerline_influence:
        river_centerline_constrained &= (river_centerline_influence > 0.0)
    river_centerline_confidence = combine_structural_confidence(
        river_centerline_influence if has_river_centerline_influence else river_scaffold_confidence,
    )
    river_centerline_confidence[~river_corridor] = 0.0
    river_centerline_uncertainty = compute_physically_explicit_guidance_uncertainty(
        anchor_uncertainty=anchor_uncertainty,
        distance_m=river_anchor_distance_m,
        transition_m=cfg.river_scaffold_transition_m,
        support_density=river_anchor_density,
        quality=river_centerline_confidence,
        disagreement_m=river_structural_spread_m,
        floor_m=max(cfg.anchor_uncertainty_floor_m, 0.15),
        distance_weight_m=0.25,
        density_weight_m=0.15,
        quality_weight_m=0.25,
        disagreement_weight=0.45,
        min_anchor_fraction=0.30,
    )
    if np.any(river_centerline_constrained):
        cl_mix, _ = compute_inverse_variance_blend_weights(
            anchor_uncertainty=anchor_uncertainty[river_centerline_constrained],
            guidance_uncertainty=river_centerline_uncertainty[river_centerline_constrained],
            fallback_guidance_influence=river_centerline_confidence[river_centerline_constrained],
        )
        anchor_vals = anchor_surface[river_centerline_constrained].astype(np.float32)
        cl_vals = river_centerline_elevation[river_centerline_constrained].astype(np.float32)
        finite_anchor = np.isfinite(anchor_vals)
        mixed_vals = cl_vals.copy()
        mixed_vals[finite_anchor] = (((1.0 - cl_mix[finite_anchor]) * anchor_vals[finite_anchor]) + (cl_mix[finite_anchor] * cl_vals[finite_anchor])).astype(np.float32)
        anchor_surface[river_centerline_constrained] = mixed_vals

    river_xs_constrained = river_domain & (~river_anchor) & np.isfinite(river_xs_support_elevation)
    if has_river_xs_support_weight:
        river_xs_constrained &= (river_xs_support_weight > 0.0)
    river_xs_confidence = combine_structural_confidence(
        river_xs_support_weight if has_river_xs_support_weight else river_scaffold_confidence,
    )
    river_xs_confidence[~river_corridor] = 0.0
    river_xs_uncertainty = compute_physically_explicit_guidance_uncertainty(
        anchor_uncertainty=anchor_uncertainty,
        distance_m=river_anchor_distance_m,
        transition_m=max(cfg.river_scaffold_transition_m * 0.75, cfg.pixel_size_m),
        support_density=river_anchor_density,
        quality=river_xs_confidence,
        disagreement_m=river_structural_spread_m,
        floor_m=max(cfg.anchor_uncertainty_floor_m, 0.12),
        distance_weight_m=0.20,
        density_weight_m=0.10,
        quality_weight_m=0.20,
        disagreement_weight=0.35,
        min_anchor_fraction=0.25,
    )
    if np.any(river_xs_constrained):
        xs_mix, _ = compute_inverse_variance_blend_weights(
            anchor_uncertainty=anchor_uncertainty[river_xs_constrained],
            guidance_uncertainty=river_xs_uncertainty[river_xs_constrained],
            fallback_guidance_influence=river_xs_confidence[river_xs_constrained],
        )
        anchor_vals = anchor_surface[river_xs_constrained].astype(np.float32)
        xs_vals = river_xs_support_elevation[river_xs_constrained].astype(np.float32)
        finite_anchor = np.isfinite(anchor_vals)
        mixed_vals = xs_vals.copy()
        mixed_vals[finite_anchor] = (((1.0 - xs_mix[finite_anchor]) * anchor_vals[finite_anchor]) + (xs_mix[finite_anchor] * xs_vals[finite_anchor])).astype(np.float32)
        anchor_surface[river_xs_constrained] = mixed_vals

    # Guidance surface is now assembled directly inside the interpolator rather than
    # requiring a prebuilt dense source-aware candidate raster. This keeps the active
    # final route centered on authoritative base + support classes + guidance artifacts.
    guidance_surface = np.full_like(auth, np.nan, dtype=np.float32)
    guidance_uncertainty = np.full_like(auth, np.nan, dtype=np.float32)
    guidance_uncertainty_source_available = np.zeros_like(auth, dtype=bool)
    if candidate is not None:
        # The source-aware candidate is the current dense guidance backstop. When explicit
        # domain guidance rasters are absent, let both SDB and river domains draw from the
        # candidate under their own domain masks rather than silently discarding it for river.
        if sdb_depth_guidance is None:
            sdb_depth_guidance = candidate
        if river_depth_guidance is None:
            river_depth_guidance = candidate
    if river_depth_guidance is not None:
        river_direct = river_domain & np.isfinite(river_depth_guidance)
        if np.any(river_direct):
            guidance_surface[river_direct] = river_depth_guidance[river_direct]
            if river_uncertainty is not None:
                guidance_uncertainty[river_direct] = river_uncertainty[river_direct]
                guidance_uncertainty_source_available[river_direct] = np.isfinite(river_uncertainty[river_direct])
    sdb_direct = gap & sdb_ok & (~river_ok | estuary_transition)
    if sdb_depth_guidance is not None:
        sdb_direct &= np.isfinite(sdb_depth_guidance)
        if np.any(sdb_direct):
            guidance_surface[sdb_direct] = sdb_depth_guidance[sdb_direct]
            if sdb_uncertainty is not None:
                guidance_uncertainty[sdb_direct] = sdb_uncertainty[sdb_direct]
                guidance_uncertainty_source_available[sdb_direct] = np.isfinite(sdb_uncertainty[sdb_direct])

    structured_river_guidance = river_domain & np.isnan(guidance_surface)
    if np.any(structured_river_guidance):
        if river_support_depth is not None:
            use_support_depth = structured_river_guidance & np.isfinite(river_support_depth)
            if np.any(use_support_depth):
                guidance_surface[use_support_depth] = river_support_depth[use_support_depth]
                river_support_confidence = combine_structural_confidence(
                    np.where(np.isfinite(river_support_depth), np.float32(1.0), np.float32(0.0)).astype(np.float32),
                    river_scaffold_confidence.astype(np.float32),
                )
                river_support_uncertainty = compute_physically_explicit_guidance_uncertainty(
                    anchor_uncertainty=anchor_uncertainty,
                    distance_m=river_anchor_distance_m,
                    transition_m=cfg.river_scaffold_transition_m,
                    support_density=river_anchor_density,
                    quality=river_support_confidence,
                    disagreement_m=river_structural_spread_m,
                    floor_m=max(cfg.anchor_uncertainty_floor_m, 0.10),
                    distance_weight_m=0.15,
                    density_weight_m=0.10,
                    quality_weight_m=0.15,
                    disagreement_weight=0.30,
                    min_anchor_fraction=0.20,
                )
                guidance_uncertainty[use_support_depth] = river_support_uncertainty[use_support_depth]
                guidance_uncertainty_source_available[use_support_depth] = np.isfinite(river_support_uncertainty[use_support_depth])
        use_profile = structured_river_guidance & np.isnan(guidance_surface) & np.isfinite(river_longitudinal_profile_elevation)
        if inputs.river_longitudinal_profile_influence is not None:
            use_profile &= (river_longitudinal_profile_influence > 0.0)
        if np.any(use_profile):
            guidance_surface[use_profile] = river_longitudinal_profile_elevation[use_profile]
            if river_longitudinal_profile_uncertainty is not None:
                guidance_uncertainty[use_profile] = river_longitudinal_profile_uncertainty[use_profile]
                guidance_uncertainty_source_available[use_profile] = np.isfinite(river_longitudinal_profile_uncertainty[use_profile])
        use_centerline = structured_river_guidance & np.isnan(guidance_surface) & np.isfinite(river_centerline_elevation)
        if has_river_centerline_influence:
            use_centerline &= (river_centerline_influence > 0.0)
        if np.any(use_centerline):
            guidance_surface[use_centerline] = river_centerline_elevation[use_centerline]
            guidance_uncertainty[use_centerline] = river_centerline_uncertainty[use_centerline]
            guidance_uncertainty_source_available[use_centerline] = np.isfinite(river_centerline_uncertainty[use_centerline])
        use_xs = structured_river_guidance & np.isnan(guidance_surface) & np.isfinite(river_xs_support_elevation)
        if has_river_xs_support_weight:
            use_xs &= (river_xs_support_weight > 0.0)
        if np.any(use_xs):
            guidance_surface[use_xs] = river_xs_support_elevation[use_xs]
            guidance_uncertainty[use_xs] = river_xs_uncertainty[use_xs]
            guidance_uncertainty_source_available[use_xs] = np.isfinite(river_xs_uncertainty[use_xs])
        use_bank = structured_river_guidance & np.isnan(guidance_surface) & np.isfinite(river_bank_elevation)
        if np.any(use_bank):
            guidance_surface[use_bank] = river_bank_elevation[use_bank]
            guidance_uncertainty[use_bank] = river_bank_uncertainty[use_bank]
            guidance_uncertainty_source_available[use_bank] = np.isfinite(river_bank_uncertainty[use_bank])

    top_level_blend_w = np.clip(guidance_influence, 0.0, 1.0).astype(np.float32)
    ivar_available = np.zeros_like(top_level_blend_w, dtype=bool)

    conditioned = np.full_like(auth, np.nan, dtype=np.float32)
    conditioned[locked] = auth[locked]
    take_any = gap & np.isfinite(guidance_surface)
    synth_top_level_uncertainty = np.isfinite(guidance_surface) & ~np.isfinite(guidance_uncertainty) & np.isfinite(top_level_guidance_uncertainty)
    if np.any(synth_top_level_uncertainty):
        guidance_uncertainty[synth_top_level_uncertainty] = top_level_guidance_uncertainty[synth_top_level_uncertainty]
        guidance_uncertainty_source_available[synth_top_level_uncertainty] = True
    if np.any(np.isfinite(guidance_surface) & ~np.isfinite(guidance_uncertainty)):
        fill_mask = np.isfinite(guidance_surface) & ~np.isfinite(guidance_uncertainty)
        guidance_uncertainty[fill_mask] = np.maximum(anchor_uncertainty[fill_mask], np.float32(1.0))
    if cfg.use_inverse_variance_blend_when_available:
        ivar_guidance_uncertainty = np.where(guidance_uncertainty_source_available, guidance_uncertainty, np.nan).astype(np.float32)
        top_level_blend_w, ivar_available = compute_inverse_variance_blend_weights(
            anchor_uncertainty=anchor_uncertainty,
            guidance_uncertainty=ivar_guidance_uncertainty,
            fallback_guidance_influence=top_level_blend_w,
        )
    if np.any(take_any):
        conditioned[take_any] = ((1.0 - top_level_blend_w[take_any]) * anchor_surface[take_any] + top_level_blend_w[take_any] * guidance_surface[take_any]).astype(np.float32)
    guidance_influence = top_level_blend_w.astype(np.float32)

    remaining_gap = gap & ~np.isfinite(conditioned)
    support_note = "terrain_interpolator_support_weighted"
    if np.any(remaining_gap):
        try:
            from scipy.ndimage import distance_transform_edt
            river_remaining_gap = remaining_gap & river_domain
            if np.any(river_remaining_gap):
                river_valid = np.isfinite(conditioned) & river_domain
                if np.any(river_valid):
                    river_backstop = _nearest_surface_within_domain(
                        river_valid,
                        conditioned,
                        river_domain,
                        centerline_mask=river_transport_centerline,
                        stationing_raster=river_centerline_stationing,
                        along_scale_m=cfg.river_aniso_along_scale_m,
                        cross_scale_m=cfg.river_aniso_cross_scale_m,
                        pixel_size_m=cfg.pixel_size_m,
                    )
                    take_river_backstop = river_remaining_gap & np.isfinite(river_backstop)
                    if np.any(take_river_backstop):
                        conditioned[take_river_backstop] = river_backstop[take_river_backstop]
                        support[take_river_backstop] = np.where(
                            river_scaffold_dominant[take_river_backstop],
                            int(SupportClass.SCAFFOLD_INFERRED),
                            int(SupportClass.GUIDANCE_CONDITIONED_RIVER),
                        ).astype(np.uint8)
                        guidance_influence[take_river_backstop] = np.maximum(guidance_influence[take_river_backstop], 0.35).astype(np.float32)
                        support_note = "terrain_interpolator_river_channel_constrained_backstop"
                        remaining_gap = gap & ~np.isfinite(conditioned)

            if np.any(remaining_gap):
                valid = np.isfinite(conditioned)
                if np.any(valid):
                    _, nearest = distance_transform_edt(~valid, return_indices=True)
                    conditioned[remaining_gap] = conditioned[nearest[0][remaining_gap], nearest[1][remaining_gap]]
                    support[remaining_gap] = np.where(
                        river_scaffold_dominant[remaining_gap],
                        int(SupportClass.SCAFFOLD_INFERRED),
                        int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL),
                    ).astype(np.uint8)
                    guidance_influence[remaining_gap] = np.maximum(guidance_influence[remaining_gap], 0.20).astype(np.float32)
                    if support_note == "terrain_interpolator_support_weighted":
                        support_note = "terrain_interpolator_with_nearest_backstop"
                    else:
                        support_note = f"{support_note}; global_backstop"
                else:
                    support_note = "terrain_interpolator_no_valid_guidance_surface"
        except (ImportError, RuntimeError, ValueError):
            log.debug("[TERRAIN] Nearest backstop fill failed; attempting deterministic hard continuity fallback.", exc_info=True)
            support_note = "terrain_interpolator_backstop_failed"

    remaining_gap = gap & ~np.isfinite(conditioned)
    if np.any(remaining_gap):
        fallback_surface = None
        if np.any(np.isfinite(anchor_surface)):
            fallback_surface = anchor_surface
            fallback_label = "anchor_surface"
        elif np.any(np.isfinite(guidance_surface)):
            fallback_surface = guidance_surface
            fallback_label = "guidance_surface"
        elif np.any(np.isfinite(nearest_auth)):
            fallback_surface = nearest_auth
            fallback_label = "nearest_authoritative"
        else:
            fallback_label = None

        if fallback_surface is not None:
            take_hard = remaining_gap & np.isfinite(fallback_surface)
            if np.any(take_hard):
                conditioned[take_hard] = fallback_surface[take_hard].astype(np.float32)
                support[take_hard] = np.where(
                    river_scaffold_dominant[take_hard],
                    int(SupportClass.SCAFFOLD_INFERRED),
                    int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL),
                ).astype(np.uint8)
                guidance_influence[take_hard] = np.maximum(guidance_influence[take_hard], 0.10).astype(np.float32)
                support_note = f"{support_note}; hard_continuity_fallback={fallback_label}"
                remaining_gap = gap & ~np.isfinite(conditioned)

    if np.any(gap & ~np.isfinite(conditioned)):
        unresolved = int(np.count_nonzero(gap & ~np.isfinite(conditioned)))
        raise RuntimeError(f"terrain interpolator failed to produce continuous output; unresolved_gap_pixels={unresolved}")

    finite_guidance_uncertainty = np.isfinite(guidance_uncertainty)
    conditioned_uncertainty = combine_conditioning_uncertainty(
        anchor_uncertainty=anchor_uncertainty,
        guidance_uncertainty=guidance_uncertainty,
        guidance_influence=guidance_influence,
        conditioned=conditioned,
        locked=locked,
    )

    eligible = gap & np.isfinite(guidance_surface)
    provenance = np.zeros_like(support, dtype=np.uint8)
    provenance[locked] = int(ProvenanceClass.AUTHORITATIVE_LOCKED)
    provenance[(support == int(SupportClass.ANCHORED_INTERPOLATION)) & np.isfinite(conditioned)] = int(ProvenanceClass.ANCHORED_INTERPOLATION)
    provenance[(support == int(SupportClass.GUIDANCE_CONDITIONED_SDB)) & np.isfinite(conditioned)] = int(ProvenanceClass.SDB_CONDITIONED_FILL)
    provenance[(support == int(SupportClass.GUIDANCE_CONDITIONED_RIVER)) & np.isfinite(conditioned)] = int(ProvenanceClass.RIVER_CONDITIONED_FILL)
    provenance[(support == int(SupportClass.SCAFFOLD_INFERRED)) & np.isfinite(conditioned)] = int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL)
    provenance[(support == int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)) & np.isfinite(conditioned)] = int(ProvenanceClass.LOW_CONFIDENCE_FILL)

    regime = regime_array_from_masks(
        build_regime_masks(
            locked=locked,
            sdb_ok=(gap & sdb_ok & ~river_ok),
            river_ok=(gap & river_ok),
            estuary_transition=(gap & estuary_transition),
        )
    ).astype(np.uint8)

    return {
        "locked": locked,
        "gap": gap,
        "eligible": eligible,
        "support": support,
        "support_distance_m": support_distance_m,
        "support_density": support_density,
        "guidance_influence": guidance_influence,
        "coastal_sdb_confidence": coastal_sdb_confidence,
        "river_anchor_distance_m": river_anchor_distance_m,
        "river_anchor_density": river_anchor_density,
        "river_scaffold_confidence": river_scaffold_confidence,
        "river_bank_edge": river_bank_edge.astype(np.uint8),
        "river_bank_distance_m": river_bank_distance_m.astype(np.float32),
        "river_bank_influence": river_bank_influence.astype(np.float32),
        "river_bank_elevation": river_bank_elevation.astype(np.float32),
        "river_bank_continuity_weight": np.clip(np.nan_to_num(river_bank_continuity_weight, nan=0.0), 0.0, 1.0).astype(np.float32) if river_bank_continuity_weight is not None else np.zeros_like(auth, dtype=np.float32),
        "river_bank_graph_confidence": np.clip(np.nan_to_num(river_bank_graph_confidence, nan=0.0), 0.0, 1.0).astype(np.float32) if river_bank_graph_confidence is not None else np.zeros_like(auth, dtype=np.float32),
        "river_bank_confluence_damping": np.clip(np.nan_to_num(river_bank_confluence_damping, nan=1.0), 0.0, 1.0).astype(np.float32) if river_bank_confluence_damping is not None else np.ones_like(auth, dtype=np.float32),
        "river_bank_estuary_side_decay": np.clip(np.nan_to_num(river_bank_estuary_side_decay, nan=1.0), 0.0, 1.0).astype(np.float32) if river_bank_estuary_side_decay is not None else np.ones_like(auth, dtype=np.float32),
        "river_centerline_elevation": river_centerline_elevation.astype(np.float32),
        "river_longitudinal_profile_elevation": river_longitudinal_profile_elevation.astype(np.float32),
        "river_centerline_influence": np.clip(np.nan_to_num(river_centerline_influence, nan=0.0), 0.0, 1.0).astype(np.float32),
        "river_xs_support_elevation": river_xs_support_elevation.astype(np.float32),
        "river_xs_support_weight": np.clip(np.nan_to_num(river_xs_support_weight, nan=0.0), 0.0, 1.0).astype(np.float32),
        "guidance_surface": guidance_surface.astype(np.float32),
        "anchor_uncertainty": anchor_uncertainty.astype(np.float32),
        "guidance_uncertainty": guidance_uncertainty.astype(np.float32),
        "conditioned_uncertainty": conditioned_uncertainty.astype(np.float32),
        "conditioned": conditioned,
        "provenance": provenance,
        "regime": regime,
        "support_note": (support_note + ("; top_level_blend=inverse_variance" if np.any(ivar_available & take_any) else "; top_level_blend=heuristic_fallback")).strip('; '),
        "config": cfg.as_dict(),
    }


__all__ = [
    "TerrainInterpolationConfig",
    "TerrainInterpolationInputs",
    "compute_support_distance_density_guidance",
    "compute_river_anchor_support_fields",
    "compute_coastal_sdb_support_confidence",
    "compute_anchor_uncertainty",
    "combine_conditioning_uncertainty",
    "compute_inverse_variance_blend_weights",
    "interpolate_support_aware_surface",
]
