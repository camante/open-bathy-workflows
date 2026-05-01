"""Support-aware terrain interpolation helpers.

This module contains the deterministic array-level helpers imported by
``authoritative_conditioning``.  It does not discover inputs or create hidden
fallback routes; callers pass every array explicitly and measured authoritative
cells remain hard control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from support_classes import SupportClass, build_regime_masks, regime_array_from_masks


@dataclass(frozen=True)
class TerrainInterpolationConfig:
    pixel_size_m: float
    support_decay_m: float
    support_density_radius_m: float
    coastal_sdb_support_transition_m: float
    river_anchor_density_radius_m: float
    river_scaffold_transition_m: float
    river_aniso_along_scale_m: float = 500.0
    river_aniso_cross_scale_m: float = 30.0
    use_inverse_variance_blend_when_available: bool = True


@dataclass(frozen=True)
class TerrainInterpolationInputs:
    auth: np.ndarray
    background_surface: Optional[np.ndarray] = None
    candidate: Optional[np.ndarray] = None
    sdb_depth_guidance: Optional[np.ndarray] = None
    river_depth_guidance: Optional[np.ndarray] = None
    primary_river_guidance_surface: Optional[np.ndarray] = None
    sdb_guide_points_path: Optional[str] = None
    river_guide_points_path: Optional[str] = None
    guidance_template_raster: Optional[str] = None
    sdb_ok: Optional[np.ndarray] = None
    river_ok: Optional[np.ndarray] = None
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
    river_channel_surface: Optional[np.ndarray] = None
    river_channel_surface_confidence: Optional[np.ndarray] = None
    river_channel_surface_source_class: Optional[np.ndarray] = None
    river_channel_surface_support_count: Optional[np.ndarray] = None
    river_channel_surface_authoritative_lock_scope: Optional[np.ndarray] = None
    river_channel_surface_authoritative_lock_applied: Optional[np.ndarray] = None
    river_channel_surface_prediction_support_confidence: Optional[np.ndarray] = None
    river_channel_surface_measured_anchor_fraction: Optional[np.ndarray] = None
    river_channel_surface_structure_only_fraction: Optional[np.ndarray] = None
    river_channel_surface_low_support_caution: Optional[np.ndarray] = None
    river_channel_surface_prediction_admissibility: Optional[np.ndarray] = None
    river_longitudinal_profile_elevation: Optional[np.ndarray] = None
    river_longitudinal_profile_uncertainty: Optional[np.ndarray] = None
    river_longitudinal_profile_influence: Optional[np.ndarray] = None
    river_longitudinal_profile_local_authoritative_reconciliation: Optional[np.ndarray] = None
    river_longitudinal_profile_local_authoritative_reconciliation_influence: Optional[np.ndarray] = None
    river_xs_support_elevation: Optional[np.ndarray] = None
    river_xs_support_weight: Optional[np.ndarray] = None
    sdb_uncertainty: Optional[np.ndarray] = None
    river_uncertainty: Optional[np.ndarray] = None
    require_explicit_bank_guidance: bool = False
    river_contract_mode: str = "canonical_v322"


def _finite(arr: Any) -> np.ndarray:
    return np.isfinite(np.asarray(arr))


def _zeros(shape: tuple[int, ...], dtype: Any = np.float32) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def _distance_and_density(mask: np.ndarray, *, pixel_size_m: float, density_radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.ndimage import distance_transform_edt, uniform_filter
    except Exception as exc:  # pragma: no cover - environment preflight should normally catch scipy
        raise RuntimeError("terrain_interpolator_requires_scipy_ndimage") from exc

    support = np.asarray(mask, dtype=bool)
    dist = distance_transform_edt(~support).astype(np.float32) * float(max(pixel_size_m, 1e-6))
    radius_px = max(1, int(round(float(max(density_radius_m, pixel_size_m)) / float(max(pixel_size_m, 1e-6)))))
    size = max(1, 2 * radius_px + 1)
    density = uniform_filter(support.astype(np.float32), size=size, mode="nearest")
    return dist.astype(np.float32), np.clip(density, 0.0, 1.0).astype(np.float32)


def compute_support_distance_density_guidance(
    locked: np.ndarray,
    auth: np.ndarray,
    *,
    pixel_size_m: float,
    support_decay_m: float,
    density_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = _finite(auth) | _finite(locked)
    dist, density = _distance_and_density(support, pixel_size_m=pixel_size_m, density_radius_m=density_radius_m)
    decay = max(float(support_decay_m), float(pixel_size_m), 1e-6)
    guidance = np.exp(-dist / decay).astype(np.float32) * np.maximum(density, support.astype(np.float32))
    trusted = support | (dist <= float(pixel_size_m) * 0.5)
    return dist, density, np.clip(guidance, 0.0, 1.0).astype(np.float32), trusted.astype(bool)


def compute_river_anchor_support_fields(
    *,
    river_anchor: np.ndarray,
    river_guidance_weight: Optional[np.ndarray],
    river_domain: np.ndarray,
    pixel_size_m: float,
    density_radius_m: float,
    scaffold_transition_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = np.asarray(river_domain).shape
    domain = np.asarray(river_domain, dtype=bool)
    anchor = np.asarray(river_anchor, dtype=np.float32)
    anchor_valid = np.isfinite(anchor) & domain
    dist, density = _distance_and_density(anchor_valid, pixel_size_m=pixel_size_m, density_radius_m=density_radius_m)
    transition = max(float(scaffold_transition_m), float(pixel_size_m), 1e-6)
    proximity = np.exp(-dist / transition).astype(np.float32)
    gw = np.asarray(river_guidance_weight, dtype=np.float32) if river_guidance_weight is not None else _zeros(shape)
    confidence = np.clip(np.maximum(gw, density) * proximity, 0.0, 1.0).astype(np.float32)
    support_depth = np.where(anchor_valid, anchor, np.nan).astype(np.float32)
    return confidence, support_depth, dist.astype(np.float32)


def compute_coastal_sdb_support_confidence(
    *,
    sdb_domain: np.ndarray,
    sdb_guidance_weight: Optional[np.ndarray],
    sdb_trusted_interior: Optional[np.ndarray],
    support_distance_m: np.ndarray,
    support_density: np.ndarray,
    support_transition_m: float = 600.0,
) -> np.ndarray:
    domain = np.asarray(sdb_domain, dtype=bool)
    shape = domain.shape
    gw = np.asarray(sdb_guidance_weight, dtype=np.float32) if sdb_guidance_weight is not None else _zeros(shape)
    trusted = np.asarray(sdb_trusted_interior, dtype=bool) if sdb_trusted_interior is not None else np.zeros(shape, dtype=bool)
    dist = np.asarray(support_distance_m, dtype=np.float32)
    density = np.asarray(support_density, dtype=np.float32)
    transition = max(float(support_transition_m), 1e-6)
    proximity = np.exp(-np.maximum(dist, 0.0) / transition).astype(np.float32)
    confidence = np.maximum(gw, trusted.astype(np.float32)) * np.maximum(density, proximity)
    confidence[~domain] = 0.0
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def compute_anchor_uncertainty(confidence: np.ndarray, *, min_uncertainty_m: float = 0.15, max_uncertainty_m: float = 5.0) -> np.ndarray:
    conf = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    return (float(max_uncertainty_m) - conf * (float(max_uncertainty_m) - float(min_uncertainty_m))).astype(np.float32)


def combine_conditioning_uncertainty(*arrays: Optional[np.ndarray]) -> np.ndarray | None:
    valid_arrays = [np.asarray(a, dtype=np.float32) for a in arrays if a is not None]
    if not valid_arrays:
        return None
    stacked = np.stack(valid_arrays, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmin(stacked, axis=0).astype(np.float32)


def interpolate_support_aware_surface(*, inputs: TerrainInterpolationInputs, config: TerrainInterpolationConfig) -> dict[str, Any]:
    auth = np.asarray(inputs.auth, dtype=np.float32)
    shape = auth.shape
    background = np.asarray(inputs.background_surface, dtype=np.float32) if inputs.background_surface is not None else np.full(shape, np.nan, dtype=np.float32)
    candidate = np.asarray(inputs.candidate, dtype=np.float32) if inputs.candidate is not None else np.full(shape, np.nan, dtype=np.float32)
    sdb = np.asarray(inputs.sdb_depth_guidance, dtype=np.float32) if inputs.sdb_depth_guidance is not None else np.full(shape, np.nan, dtype=np.float32)
    river_source = inputs.primary_river_guidance_surface if inputs.primary_river_guidance_surface is not None else inputs.river_depth_guidance
    river = np.asarray(river_source, dtype=np.float32) if river_source is not None else np.full(shape, np.nan, dtype=np.float32)

    sdb_ok = np.asarray(inputs.sdb_ok, dtype=bool) if inputs.sdb_ok is not None else np.zeros(shape, dtype=bool)
    river_ok = np.asarray(inputs.river_ok, dtype=bool) if inputs.river_ok is not None else np.zeros(shape, dtype=bool)
    estuary = np.asarray(inputs.estuary_transition, dtype=bool) if inputs.estuary_transition is not None else np.zeros(shape, dtype=bool)

    out = background.copy()
    support_class = np.full(shape, int(SupportClass.UNSUPPORTED), dtype=np.uint8)
    background_valid = np.isfinite(background)
    support_class[background_valid] = int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)

    candidate_valid = np.isfinite(candidate)
    out[candidate_valid] = candidate[candidate_valid]
    support_class[candidate_valid] = int(SupportClass.ANCHORED_INTERPOLATION)

    sdb_valid = np.isfinite(sdb) & sdb_ok & (~river_ok | estuary)
    out[sdb_valid] = sdb[sdb_valid]
    support_class[sdb_valid] = int(SupportClass.GUIDANCE_CONDITIONED_SDB)

    river_valid = np.isfinite(river) & river_ok
    out[river_valid] = river[river_valid]
    support_class[river_valid] = int(SupportClass.GUIDANCE_CONDITIONED_RIVER)

    auth_valid = np.isfinite(auth)
    out[auth_valid] = auth[auth_valid]
    support_class[auth_valid] = int(SupportClass.AUTHORITATIVE_LOCKED)

    regime_masks = build_regime_masks(auth_valid, sdb_ok, river_ok, estuary_transition=estuary, shape=shape)
    regime_class = regime_array_from_masks(regime_masks)
    confidence = np.zeros(shape, dtype=np.float32)
    confidence[background_valid] = 0.2
    confidence[candidate_valid] = 0.45
    confidence[sdb_valid] = 0.6
    confidence[river_valid] = 0.7
    confidence[auth_valid] = 1.0
    uncertainty = compute_anchor_uncertainty(confidence)

    return {
        "conditioned": out.astype(np.float32),
        "surface": out.astype(np.float32),
        "support_class": support_class,
        "regime_class": regime_class,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "stats": {
            "authoritative_locked_pixels": int(np.count_nonzero(auth_valid)),
            "river_guidance_pixels": int(np.count_nonzero(river_valid & ~auth_valid)),
            "sdb_guidance_pixels": int(np.count_nonzero(sdb_valid & ~auth_valid & ~river_valid)),
            "candidate_pixels": int(np.count_nonzero(candidate_valid)),
            "background_pixels": int(np.count_nonzero(background_valid)),
            "finite_output_pixels": int(np.count_nonzero(np.isfinite(out))),
        },
        "contract": {
            "authoritative_cells_locked": True,
            "hidden_source_discovery": False,
            "river_contract_mode": inputs.river_contract_mode,
            "use_inverse_variance_blend_when_available": bool(config.use_inverse_variance_blend_when_available),
        },
    }


__all__ = [
    "TerrainInterpolationConfig",
    "TerrainInterpolationInputs",
    "combine_conditioning_uncertainty",
    "compute_anchor_uncertainty",
    "compute_coastal_sdb_support_confidence",
    "compute_river_anchor_support_fields",
    "compute_support_distance_density_guidance",
    "interpolate_support_aware_surface",
]
