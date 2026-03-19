from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
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

    def sanitized(self) -> "TerrainInterpolationConfig":
        safe_pixel = max(float(self.pixel_size_m or 0.0), 1.0)
        return TerrainInterpolationConfig(
            pixel_size_m=safe_pixel,
            support_decay_m=max(float(self.support_decay_m or 0.0), safe_pixel),
            support_density_radius_m=max(float(self.support_density_radius_m or 0.0), safe_pixel),
            coastal_sdb_support_transition_m=max(float(self.coastal_sdb_support_transition_m or 0.0), 1.0),
            river_anchor_density_radius_m=max(float(self.river_anchor_density_radius_m or 0.0), safe_pixel),
            river_scaffold_transition_m=max(float(self.river_scaffold_transition_m or 0.0), safe_pixel),
        )

    def as_dict(self) -> Dict[str, float]:
        return asdict(self.sanitized())


@dataclass(frozen=True)
class TerrainInterpolationInputs:
    candidate: np.ndarray
    auth: np.ndarray
    sdb_ok: np.ndarray
    river_ok: np.ndarray
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
    river_xs_support_elevation: Optional[np.ndarray] = None
    river_xs_support_weight: Optional[np.ndarray] = None


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


def _validate_shapes(inputs: TerrainInterpolationInputs) -> None:
    base = np.asarray(inputs.candidate)
    expected = base.shape
    named = {
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
        "river_xs_support_elevation": inputs.river_xs_support_elevation,
        "river_xs_support_weight": inputs.river_xs_support_weight,
    }
    for name, arr in named.items():
        if arr is None:
            continue
        arr_shape = np.asarray(arr).shape
        if arr_shape != expected:
            raise ValueError(f"{name} shape {arr_shape} does not match candidate shape {expected}")


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


def _nearest_surface_within_domain(valid_mask: np.ndarray, values: np.ndarray, domain_mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbor handoff constrained to a logical domain.

    This is used to prevent cross-bank bleed in river corridors: river-domain
    cells can only inherit from other valid river-domain cells, never from the
    nearest non-river authoritative bank pixel.
    """
    domain_mask = np.asarray(domain_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool) & domain_mask
    values = np.asarray(values, dtype=np.float32)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    if not np.any(valid_mask):
        return out
    domain_out = _nearest_surface_from_values(valid_mask, values)
    out[domain_mask] = domain_out[domain_mask]
    return out

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

    candidate = np.asarray(inputs.candidate, dtype=np.float32)
    auth = np.asarray(inputs.auth, dtype=np.float32)
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
    river_centerline_elevation = _float32_or_nan(candidate.shape, inputs.river_centerline_elevation)
    river_centerline_influence = _float32_or_zero(candidate.shape, inputs.river_centerline_influence)
    river_xs_support_elevation = _float32_or_nan(candidate.shape, inputs.river_xs_support_elevation)
    river_xs_support_weight = _float32_or_zero(candidate.shape, inputs.river_xs_support_weight)

    has_river_centerline_elevation = inputs.river_centerline_elevation is not None
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

    support = np.zeros_like(candidate, dtype=np.uint8)
    support[locked] = int(SupportClass.AUTHORITATIVE_LOCKED)
    support[gap] = int(SupportClass.ANCHORED_INTERPOLATION)

    support_distance_m, support_density, base_guidance_influence, nearest_auth = compute_support_distance_density_guidance(
        locked,
        auth,
        pixel_size_m=cfg.pixel_size_m,
        support_decay_m=cfg.support_decay_m,
        density_radius_m=cfg.support_density_radius_m,
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
    if river_bank_pair_weight is not None:
        river_bank_influence = np.clip(
            river_bank_influence * (0.55 + (0.45 * np.clip(np.nan_to_num(river_bank_pair_weight, nan=0.0), 0.0, 1.0))),
            0.0,
            1.0,
        ).astype(np.float32)
        river_bank_influence[~river_corridor] = 0.0
    if river_bank_continuity_weight is not None:
        cont = np.clip(np.nan_to_num(river_bank_continuity_weight, nan=0.0), 0.0, 1.0).astype(np.float32)
        river_bank_influence = np.clip(
            river_bank_influence * (0.60 + (0.40 * cont)),
            0.0,
            1.0,
        ).astype(np.float32)
        river_bank_influence[~river_corridor] = 0.0
    if river_bank_graph_confidence is not None:
        graph_conf = np.clip(np.nan_to_num(river_bank_graph_confidence, nan=0.0), 0.0, 1.0).astype(np.float32)
        river_bank_influence = np.clip(
            river_bank_influence * (0.45 + (0.55 * graph_conf)),
            0.0,
            1.0,
        ).astype(np.float32)
        river_bank_influence[~river_corridor] = 0.0
    if river_bank_confluence_damping is not None:
        confluence = np.clip(np.nan_to_num(river_bank_confluence_damping, nan=1.0), 0.0, 1.0).astype(np.float32)
        river_bank_influence = np.clip(river_bank_influence * confluence, 0.0, 1.0).astype(np.float32)
        river_bank_influence[~river_corridor] = 0.0
    if river_bank_estuary_side_decay is not None:
        est_decay = np.clip(np.nan_to_num(river_bank_estuary_side_decay, nan=1.0), 0.0, 1.0).astype(np.float32)
        river_bank_influence = np.clip(river_bank_influence * est_decay, 0.0, 1.0).astype(np.float32)
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

    fallback_influence = np.clip(0.20 + 0.80 * base_guidance_influence, 0.0, 1.0).astype(np.float32)
    guidance_influence = fallback_influence.copy()

    if np.any(sdb_ok):
        sdb_local = np.clip(np.nan_to_num(sdb_gw, nan=0.0), 0.0, 1.0).astype(np.float32) if sdb_gw is not None else np.zeros_like(candidate, dtype=np.float32)
        if sdb_ti is not None:
            sdb_local = np.maximum(sdb_local, 0.65 * (np.asarray(sdb_ti) > 0).astype(np.float32))
        sdb_signal = np.maximum(sdb_local, coastal_sdb_confidence.astype(np.float32))
        sdb_influence = np.clip(
            0.35 + (0.65 * base_guidance_influence * (0.15 + 0.85 * sdb_signal)),
            0.10,
            1.0,
        ).astype(np.float32)
        guidance_influence[sdb_ok] = sdb_influence[sdb_ok]

    if has_river_centerline_influence:
        river_centerline_influence = np.clip(np.nan_to_num(river_centerline_influence, nan=0.0), 0.0, 1.0).astype(np.float32)
        river_centerline_influence[~river_corridor] = 0.0
    if has_river_xs_support_weight:
        river_xs_support_weight = np.clip(np.nan_to_num(river_xs_support_weight, nan=0.0), 0.0, 1.0).astype(np.float32)
        river_xs_support_weight[~river_corridor] = 0.0

    if np.any(river_ok):
        river_local = np.clip(np.nan_to_num(river_gw, nan=0.0), 0.0, 1.0).astype(np.float32) if river_gw is not None else np.clip(base_guidance_influence, 0.0, 1.0).astype(np.float32)
        if river_ti is not None:
            river_local = np.maximum(river_local, 0.70 * (np.asarray(river_ti) > 0).astype(np.float32))
        river_signal = np.maximum(river_local, river_scaffold_confidence.astype(np.float32))
        river_influence = np.clip(0.75 + (0.25 * river_signal), 0.75, 1.0).astype(np.float32)
        if np.any(river_corridor):
            river_influence = np.clip(river_influence * (1.0 - (0.40 * river_bank_influence)), 0.55, 1.0).astype(np.float32)
        estuary_cap = np.where(estuary_transition, 0.85, 1.0).astype(np.float32)
        river_influence = np.minimum(river_influence, estuary_cap).astype(np.float32)
        guidance_influence[river_ok] = np.maximum(guidance_influence[river_ok], river_influence[river_ok])

    guidance_influence[river_scaffold_dominant] = np.maximum(
        guidance_influence[river_scaffold_dominant],
        np.clip(0.85 + 0.15 * river_scaffold_confidence[river_scaffold_dominant], 0.85, 1.0),
    ).astype(np.float32)
    guidance_influence[locked] = 0.0

    anchor_surface = nearest_auth.astype(np.float32)
    river_channel_anchor_surface = np.full_like(candidate, np.nan, dtype=np.float32)

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
            )

    river_authoritative_valid = locked & river_domain & np.isfinite(auth)
    if np.any(river_authoritative_valid):
        river_authoritative_surface = _nearest_surface_within_domain(
            river_authoritative_valid,
            auth,
            river_domain,
        )
        use_authoritative_surface = river_domain & ~np.isfinite(river_channel_anchor_surface) & np.isfinite(river_authoritative_surface)
        if np.any(use_authoritative_surface):
            river_channel_anchor_surface[use_authoritative_surface] = river_authoritative_surface[use_authoritative_surface]

    river_candidate_valid = river_domain & np.isfinite(candidate)
    if np.any(river_candidate_valid):
        river_candidate_surface = _nearest_surface_within_domain(
            river_candidate_valid,
            candidate,
            river_domain,
        )
        use_candidate_surface = river_domain & ~np.isfinite(river_channel_anchor_surface) & np.isfinite(river_candidate_surface)
        if np.any(use_candidate_surface):
            river_channel_anchor_surface[use_candidate_surface] = river_candidate_surface[use_candidate_surface]

    use_river_anchor_surface = river_domain & np.isfinite(river_channel_anchor_surface)
    if np.any(use_river_anchor_surface):
        anchor_surface[use_river_anchor_surface] = river_channel_anchor_surface[use_river_anchor_surface]

    river_bank_constrained = river_domain & (~river_anchor) & np.isfinite(river_bank_elevation) & (river_bank_influence > 0.0)
    if np.any(river_bank_constrained):
        edge_mix = np.clip(river_bank_influence[river_bank_constrained], 0.0, 1.0).astype(np.float32)
        if river_bank_continuity_weight is not None:
            cont_mix = np.clip(np.nan_to_num(river_bank_continuity_weight[river_bank_constrained], nan=0.0), 0.0, 1.0).astype(np.float32)
            edge_mix = np.clip(edge_mix * (0.60 + (0.40 * cont_mix)), 0.0, 1.0).astype(np.float32)
        if river_bank_graph_confidence is not None:
            graph_mix = np.clip(np.nan_to_num(river_bank_graph_confidence[river_bank_constrained], nan=0.0), 0.0, 1.0).astype(np.float32)
            edge_mix = np.clip(edge_mix * (0.45 + (0.55 * graph_mix)), 0.0, 1.0).astype(np.float32)
        if river_bank_confluence_damping is not None:
            conf_mix = np.clip(np.nan_to_num(river_bank_confluence_damping[river_bank_constrained], nan=1.0), 0.0, 1.0).astype(np.float32)
            edge_mix = np.clip(edge_mix * conf_mix, 0.0, 1.0).astype(np.float32)
        if river_bank_estuary_side_decay is not None:
            est_mix = np.clip(np.nan_to_num(river_bank_estuary_side_decay[river_bank_constrained], nan=1.0), 0.0, 1.0).astype(np.float32)
            edge_mix = np.clip(edge_mix * est_mix, 0.0, 1.0).astype(np.float32)
        anchor_vals = anchor_surface[river_bank_constrained].astype(np.float32)
        bank_vals = river_bank_elevation[river_bank_constrained].astype(np.float32)
        finite_anchor = np.isfinite(anchor_vals)
        mixed_vals = bank_vals.copy()
        mixed_vals[finite_anchor] = (
            ((1.0 - edge_mix[finite_anchor]) * anchor_vals[finite_anchor])
            + (edge_mix[finite_anchor] * bank_vals[finite_anchor])
        ).astype(np.float32)
        anchor_surface[river_bank_constrained] = mixed_vals

    river_centerline_constrained = river_domain & (~river_anchor) & np.isfinite(river_centerline_elevation)
    if has_river_centerline_influence:
        river_centerline_constrained &= (river_centerline_influence > 0.0)
    if np.any(river_centerline_constrained):
        cl_mix = np.clip(river_centerline_influence[river_centerline_constrained], 0.0, 1.0).astype(np.float32) if has_river_centerline_influence else np.full(int(np.sum(river_centerline_constrained)), 0.35, dtype=np.float32)
        cl_mix = np.clip(0.15 + (0.55 * cl_mix), 0.0, 0.70).astype(np.float32)
        anchor_vals = anchor_surface[river_centerline_constrained].astype(np.float32)
        cl_vals = river_centerline_elevation[river_centerline_constrained].astype(np.float32)
        finite_anchor = np.isfinite(anchor_vals)
        mixed_vals = cl_vals.copy()
        mixed_vals[finite_anchor] = (((1.0 - cl_mix[finite_anchor]) * anchor_vals[finite_anchor]) + (cl_mix[finite_anchor] * cl_vals[finite_anchor])).astype(np.float32)
        anchor_surface[river_centerline_constrained] = mixed_vals

    river_xs_constrained = river_domain & (~river_anchor) & np.isfinite(river_xs_support_elevation)
    if has_river_xs_support_weight:
        river_xs_constrained &= (river_xs_support_weight > 0.0)
    if np.any(river_xs_constrained):
        xs_mix = np.clip(river_xs_support_weight[river_xs_constrained], 0.0, 1.0).astype(np.float32) if has_river_xs_support_weight else np.full(int(np.sum(river_xs_constrained)), 0.40, dtype=np.float32)
        xs_mix = np.clip(0.20 + (0.60 * xs_mix), 0.0, 0.80).astype(np.float32)
        anchor_vals = anchor_surface[river_xs_constrained].astype(np.float32)
        xs_vals = river_xs_support_elevation[river_xs_constrained].astype(np.float32)
        finite_anchor = np.isfinite(anchor_vals)
        mixed_vals = xs_vals.copy()
        mixed_vals[finite_anchor] = (((1.0 - xs_mix[finite_anchor]) * anchor_vals[finite_anchor]) + (xs_mix[finite_anchor] * xs_vals[finite_anchor])).astype(np.float32)
        anchor_surface[river_xs_constrained] = mixed_vals

    conditioned = np.full_like(candidate, np.nan, dtype=np.float32)
    conditioned[locked] = auth[locked]
    take_any = gap & np.isfinite(candidate)
    if np.any(take_any):
        blend_w = np.clip(guidance_influence, 0.0, 1.0).astype(np.float32)
        conditioned[take_any] = ((1.0 - blend_w[take_any]) * anchor_surface[take_any] + blend_w[take_any] * candidate[take_any]).astype(np.float32)

    remaining_gap = gap & ~np.isfinite(conditioned)
    support_note = "terrain_interpolator_support_weighted"
    if np.any(remaining_gap):
        try:
            from scipy.ndimage import distance_transform_edt
            river_remaining_gap = remaining_gap & river_domain
            if np.any(river_remaining_gap):
                river_valid = np.isfinite(conditioned) & river_domain
                if np.any(river_valid):
                    river_backstop = _nearest_surface_within_domain(river_valid, conditioned, river_domain)
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
                    support_note = "terrain_interpolator_no_valid_candidate"
        except (ImportError, RuntimeError, ValueError):
            log.debug("[TERRAIN] Nearest backstop fill failed; residual nodata may remain.", exc_info=True)
            support_note = "terrain_interpolator_backstop_failed"

    eligible = gap & np.isfinite(candidate)
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
        "river_bank_continuity_weight": np.clip(np.nan_to_num(river_bank_continuity_weight, nan=0.0), 0.0, 1.0).astype(np.float32) if river_bank_continuity_weight is not None else np.zeros_like(candidate, dtype=np.float32),
        "river_bank_graph_confidence": np.clip(np.nan_to_num(river_bank_graph_confidence, nan=0.0), 0.0, 1.0).astype(np.float32) if river_bank_graph_confidence is not None else np.zeros_like(candidate, dtype=np.float32),
        "river_bank_confluence_damping": np.clip(np.nan_to_num(river_bank_confluence_damping, nan=1.0), 0.0, 1.0).astype(np.float32) if river_bank_confluence_damping is not None else np.ones_like(candidate, dtype=np.float32),
        "river_bank_estuary_side_decay": np.clip(np.nan_to_num(river_bank_estuary_side_decay, nan=1.0), 0.0, 1.0).astype(np.float32) if river_bank_estuary_side_decay is not None else np.ones_like(candidate, dtype=np.float32),
        "river_centerline_elevation": river_centerline_elevation.astype(np.float32),
        "river_centerline_influence": np.clip(np.nan_to_num(river_centerline_influence, nan=0.0), 0.0, 1.0).astype(np.float32),
        "river_xs_support_elevation": river_xs_support_elevation.astype(np.float32),
        "river_xs_support_weight": np.clip(np.nan_to_num(river_xs_support_weight, nan=0.0), 0.0, 1.0).astype(np.float32),
        "conditioned": conditioned,
        "provenance": provenance,
        "regime": regime,
        "support_note": support_note,
        "config": cfg.as_dict(),
    }


__all__ = [
    "TerrainInterpolationConfig",
    "TerrainInterpolationInputs",
    "compute_support_distance_density_guidance",
    "compute_river_anchor_support_fields",
    "compute_coastal_sdb_support_confidence",
    "interpolate_support_aware_surface",
]
