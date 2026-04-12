from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
from pathlib import Path
import logging
import gc

import numpy as np

from nodata_utils import sanitize_for_output, array_valid_mask
from memory_diag import memory_checkpoint

from provenance_schema import ProvenanceClass
from support_classes import SupportClass, build_regime_masks, regime_array_from_masks
from river_bank_guidance import (
    compute_bank_distance_influence,
    compute_bank_elevation_surface_from_authoritative,
)
from river_primary_surface_contract import validate_river_primary_surface_contract
from channel_core_preservation import (
    compute_channel_core_preserve_mask,
    apply_channel_core_preservation,
    build_channel_core_preservation_diagnostics,
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
    background_surface: Optional[np.ndarray] = None
    sdb_depth_guidance: Optional[np.ndarray] = None
    river_depth_guidance: Optional[np.ndarray] = None
    primary_river_guidance_surface: Optional[np.ndarray] = None
    river_contract_mode: str = "canonical_v322"
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


def _as_float32(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return None if arr is None else np.asarray(arr, dtype=np.float32)




def _float32_or_nan(shape: tuple[int, ...], arr: Optional[np.ndarray]) -> np.ndarray:
    if arr is None:
        # Placeholder-only layers do not need float32 precision; using float16
        # halves memory for the many optional river rasters that are absent in
        # coastal AOIs.
        return np.full(shape, np.nan, dtype=np.float16)
    return sanitize_for_output(arr, dtype=np.float32)


def _float32_or_zero(shape: tuple[int, ...], arr: Optional[np.ndarray]) -> np.ndarray:
    if arr is None:
        return np.zeros(shape, dtype=np.float16)
    out = sanitize_for_output(arr, dtype=np.float32)
    out[~np.isfinite(out)] = 0.0
    return out






_GUIDANCE_VERIFY_ZERO_KEYS = (
    "sdb_gw",
    "sdb_ti",
    "river_gw",
    "river_ti",
    "river_bank_influence",
    "river_centerline_influence",
    "river_longitudinal_profile_influence",
    "river_longitudinal_profile_local_authoritative_reconciliation_influence",
    "river_xs_support_weight",
)

_GUIDANCE_VERIFY_NAN_KEYS = (
    "sdb_depth_guidance",
    "river_depth_guidance",
    "primary_river_guidance_surface",
    "river_support_depth",
    "river_bank_elevation",
    "river_centerline_elevation",
    "river_centerline_stationing",
    "river_channel_surface",
    "river_channel_surface_confidence",
    "river_longitudinal_profile_elevation",
    "river_longitudinal_profile_uncertainty",
    "river_longitudinal_profile_local_authoritative_reconciliation",
    "river_xs_support_elevation",
    "sdb_uncertainty",
    "river_uncertainty",
)


def _verify_locked_guidance_exclusion(
    *,
    locked: np.ndarray,
    zero_arrays: Dict[str, tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    nan_arrays: Dict[str, tuple[Optional[np.ndarray], Optional[np.ndarray]]],
) -> Dict[str, Any]:
    locked = np.asarray(locked, dtype=bool)
    payload: Dict[str, Any] = {
        "locked_pixels": int(np.count_nonzero(locked)),
        "arrays": {},
        "ok": True,
        "violations": [],
    }
    if not np.any(locked):
        return payload

    def _active_locked(mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return locked
        return locked & np.asarray(mask, dtype=bool)

    for name, (arr, active_mask) in zero_arrays.items():
        if arr is None:
            continue
        active_locked = _active_locked(active_mask)
        vals = np.asarray(arr)
        if vals.dtype == np.bool_:
            bad = active_locked & vals
        else:
            bad = active_locked & np.isfinite(vals) & (np.abs(vals) > 0.0)
        count = int(np.count_nonzero(bad))
        payload["arrays"][name] = {"policy": "zero_on_locked", "violating_pixels": count}
        if count:
            payload["ok"] = False
            payload["violations"].append(f"{name}:{count}")

    for name, (arr, active_mask) in nan_arrays.items():
        if arr is None:
            continue
        active_locked = _active_locked(active_mask)
        vals = np.asarray(arr)
        bad = active_locked & np.isfinite(vals)
        count = int(np.count_nonzero(bad))
        payload["arrays"][name] = {"policy": "nan_on_locked", "violating_pixels": count}
        if count:
            payload["ok"] = False
            payload["violations"].append(f"{name}:{count}")

    if not payload["ok"]:
        raise RuntimeError(
            "terrain interpolator received subordinate guidance inside authoritative locked cells: "
            + ", ".join(payload["violations"])
        )
    return payload

def _resolve_native_guidance_arrays(inputs: TerrainInterpolationInputs) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    sdb_depth_guidance = _as_float32(inputs.sdb_depth_guidance)
    contract_mode = str(getattr(inputs, "river_contract_mode", "canonical_v322") or "canonical_v322")
    canonical_mode = contract_mode.startswith("canonical_")
    primary_input_source = "none"
    legacy_alias_rejected = False
    if inputs.primary_river_guidance_surface is not None:
        primary_input_source = "primary_river_guidance_surface"
        river_depth_guidance = _as_float32(inputs.primary_river_guidance_surface)
    elif inputs.river_depth_guidance is not None:
        if canonical_mode:
            primary_input_source = "rejected_legacy_river_depth_guidance_alias"
            river_depth_guidance = None
            legacy_alias_rejected = True
        else:
            primary_input_source = "legacy_river_depth_guidance_alias"
            river_depth_guidance = _as_float32(inputs.river_depth_guidance)
    else:
        river_depth_guidance = None
    if (primary_input_source == "none") and inputs.river_guide_points_path:
        if canonical_mode:
            primary_input_source = "rejected_native_river_guide_points_path"
        else:
            primary_input_source = "native_river_guide_points_rasterized"
    template = inputs.guidance_template_raster

    requires_template = bool(inputs.sdb_guide_points_path) or bool(inputs.river_guide_points_path and (not canonical_mode))
    if requires_template and not template:
        raise ValueError("guidance_template_raster is required when native guide-point paths are provided")

    if template:
        template_path = Path(template)
        if requires_template and (not template_path.exists()):
            raise ValueError(f"guidance_template_raster does not exist: {template_path}")
        if inputs.sdb_guide_points_path and sdb_depth_guidance is None:
            try:
                from sdb_guidance import rasterize_sdb_guide_points_to_template
                sdb_depth_guidance = _as_float32(rasterize_sdb_guide_points_to_template(inputs.sdb_guide_points_path, template_path, logger=log))
            except Exception:
                log.debug("[TERRAIN] Failed to rasterize native SDB guide points inside interpolator.", exc_info=True)
        if inputs.river_guide_points_path and river_depth_guidance is None and (not canonical_mode):
            try:
                from river_guidance import rasterize_river_guide_points_to_template
                river_depth_guidance = _as_float32(rasterize_river_guide_points_to_template(inputs.river_guide_points_path, template_path, logger=log))
                if (river_depth_guidance is not None) and np.any(np.isfinite(river_depth_guidance)):
                    primary_input_source = "native_river_guide_points_rasterized"
                if river_depth_guidance is None or not np.any(np.isfinite(river_depth_guidance)):
                    log.warning("[TERRAIN] Native river guide points rasterization produced no finite cells: %s", inputs.river_guide_points_path)
            except Exception as exc:
                log.warning("[TERRAIN] Failed to rasterize native river guide points inside interpolator: %s", exc, exc_info=True)
    return sdb_depth_guidance, river_depth_guidance, primary_input_source

def _validate_shapes(inputs: TerrainInterpolationInputs) -> None:
    expected = np.asarray(inputs.auth).shape
    named = {
        "candidate": inputs.candidate,
        "background_surface": inputs.background_surface,
        "sdb_depth_guidance": inputs.sdb_depth_guidance,
        "river_depth_guidance": inputs.river_depth_guidance,
        "primary_river_guidance_surface": inputs.primary_river_guidance_surface,
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
        "river_channel_surface_authoritative_lock_scope": inputs.river_channel_surface_authoritative_lock_scope,
        "river_channel_surface_authoritative_lock_applied": inputs.river_channel_surface_authoritative_lock_applied,
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


def _compute_authoritative_first_contract(*, conditioned: np.ndarray, auth: np.ndarray, locked: np.ndarray, background_surface: Optional[np.ndarray], baseline_exact_domain: np.ndarray, support: np.ndarray) -> Dict[str, Any]:
    conditioned = np.asarray(conditioned, dtype=np.float32)
    auth = np.asarray(auth, dtype=np.float32)
    locked = np.asarray(locked, dtype=bool)
    baseline_exact_domain = np.asarray(baseline_exact_domain, dtype=bool)
    support = np.asarray(support)
    locked_valid = locked & np.isfinite(auth) & np.isfinite(conditioned)
    locked_diff = np.abs(conditioned - auth)
    locked_violation = locked_valid & (locked_diff > np.float32(1e-6))
    background_exact = baseline_exact_domain & np.isfinite(conditioned)
    background_violation = np.zeros_like(background_exact, dtype=bool)
    background_diff = np.zeros_like(conditioned, dtype=np.float32)
    if background_surface is not None:
        bg = np.asarray(background_surface, dtype=np.float32)
        background_exact &= np.isfinite(bg)
        background_diff = np.abs(conditioned - bg).astype(np.float32)
        background_violation = background_exact & (background_diff > np.float32(1e-6))
    low_conf_outside_guidance = background_exact & (support == int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL))
    return {
        "locked_cell_count": int(np.count_nonzero(locked_valid)),
        "locked_changed_count": int(np.count_nonzero(locked_violation)),
        "locked_max_abs_diff_m": float(np.nanmax(locked_diff[locked_valid])) if np.any(locked_valid) else 0.0,
        "background_exact_cell_count": int(np.count_nonzero(background_exact)),
        "background_changed_count": int(np.count_nonzero(background_violation)),
        "background_max_abs_diff_m": float(np.nanmax(background_diff[background_exact])) if np.any(background_exact) else 0.0,
        "low_confidence_fill_outside_guidance_count": int(np.count_nonzero(low_conf_outside_guidance)),
        "ok": bool((not np.any(locked_violation)) and (not np.any(background_violation)) and (not np.any(low_conf_outside_guidance))),
    }


def _enforce_authoritative_first_contract(*, conditioned: np.ndarray, auth: np.ndarray, locked: np.ndarray, background_surface: Optional[np.ndarray], baseline_exact_domain: np.ndarray, support: np.ndarray, guidance_influence: np.ndarray) -> Dict[str, Any]:
    conditioned = np.array(conditioned, dtype=np.float32, copy=True)
    support = np.array(support, dtype=np.uint8, copy=True)
    guidance_influence = np.array(guidance_influence, dtype=np.float32, copy=True)
    locked = np.asarray(locked, dtype=bool)
    baseline_exact_domain = np.asarray(baseline_exact_domain, dtype=bool)
    if np.any(locked):
        conditioned[locked] = np.asarray(auth, dtype=np.float32)[locked]
        support[locked] = int(SupportClass.AUTHORITATIVE_LOCKED)
        guidance_influence[locked] = 0.0
    if background_surface is not None and np.any(baseline_exact_domain):
        bg = np.asarray(background_surface, dtype=np.float32)
        take_bg = baseline_exact_domain & np.isfinite(bg) & (~locked)
        if np.any(take_bg):
            conditioned[take_bg] = bg[take_bg]
            support[take_bg] = int(SupportClass.ANCHORED_INTERPOLATION)
            guidance_influence[take_bg] = 0.0
    contract = _compute_authoritative_first_contract(
        conditioned=conditioned,
        auth=auth,
        locked=locked,
        background_surface=background_surface,
        baseline_exact_domain=baseline_exact_domain,
        support=support,
    )
    if not contract["ok"]:
        raise RuntimeError(
            "authoritative-first contract violated: "
            f"locked_changed={contract['locked_changed_count']} "
            f"background_changed={contract['background_changed_count']} "
            f"low_conf_outside_guidance={contract['low_confidence_fill_outside_guidance_count']}"
        )
    return {
        "conditioned": conditioned,
        "support": support,
        "guidance_influence": guidance_influence,
        "contract": contract,
    }

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


def _localize_routed_surface_by_station_window(
    seed_mask: np.ndarray,
    domain_mask: np.ndarray,
    *,
    centerline_mask: Optional[np.ndarray] = None,
    stationing_raster: Optional[np.ndarray] = None,
    along_window_m: float = 250.0,
) -> np.ndarray:
    """Return a 0-1 weight that limits support to a local along-channel neighborhood.

    This is intended for local structure like cross-sections, which should refine
    nearby channel shape but should not behave like a full-domain transported bed
    surface. If centerline stationing is unavailable, a simple Euclidean fallback
    is used instead.
    """
    from scipy.ndimage import distance_transform_edt, label

    domain_mask = np.asarray(domain_mask, dtype=bool)
    seed_mask = np.asarray(seed_mask, dtype=bool) & domain_mask
    out = np.zeros(domain_mask.shape, dtype=np.float32)
    if not np.any(seed_mask):
        return out

    safe_window = max(float(along_window_m), 1.0)
    centerline_ok = None
    if centerline_mask is not None:
        centerline_ok = np.asarray(centerline_mask, dtype=bool) & domain_mask
        if not np.any(centerline_ok):
            centerline_ok = None
    stationing = None if stationing_raster is None else np.asarray(stationing_raster, dtype=np.float32)
    if centerline_ok is None or stationing is None or not np.any(np.isfinite(stationing[centerline_ok])):
        dist = distance_transform_edt(~seed_mask).astype(np.float32)
        out[domain_mask] = np.clip(1.0 - (dist[domain_mask] / np.float32(safe_window)), 0.0, 1.0).astype(np.float32)
        return out

    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    component_labels, ncomp = label(centerline_ok.astype(np.uint8), structure=structure)
    if ncomp <= 0:
        dist = distance_transform_edt(~seed_mask).astype(np.float32)
        out[domain_mask] = np.clip(1.0 - (dist[domain_mask] / np.float32(safe_window)), 0.0, 1.0).astype(np.float32)
        return out

    safe_lateral = max(np.float32(0.25 * safe_window), np.float32(1.0))
    for comp_idx in range(1, int(ncomp) + 1):
        comp_centerline = centerline_ok & (component_labels == comp_idx)
        if not np.any(comp_centerline):
            continue
        comp_domain = domain_mask.copy()
        dist_to_centerline, nearest_centerline = distance_transform_edt(~comp_centerline, return_indices=True)
        nearest_comp = component_labels[nearest_centerline[0], nearest_centerline[1]] == comp_idx
        comp_domain &= nearest_comp
        if not np.any(comp_domain):
            continue

        comp_seed = seed_mask & comp_domain
        if not np.any(comp_seed):
            continue
        src_rows, src_cols = np.nonzero(comp_seed)
        proj_rows = nearest_centerline[0][src_rows, src_cols]
        proj_cols = nearest_centerline[1][src_rows, src_cols]
        src_station = stationing[proj_rows, proj_cols].astype(np.float32)
        finite_src = np.isfinite(src_station)
        if not np.any(finite_src):
            continue
        src_station = np.sort(src_station[finite_src], kind='mergesort')

        q_rows, q_cols = np.nonzero(comp_domain & np.isfinite(stationing))
        if q_rows.size == 0:
            continue
        q_station = stationing[q_rows, q_cols].astype(np.float32)
        ins = np.searchsorted(src_station, q_station)
        q_lateral = dist_to_centerline[q_rows, q_cols].astype(np.float32)
        query_weight = np.zeros(q_rows.shape, dtype=np.float32)
        for i, (sq, ii, lat) in enumerate(zip(q_station.tolist(), ins.tolist(), q_lateral.tolist())):
            best = np.inf
            if ii < int(src_station.size):
                best = min(best, abs(float(src_station[ii]) - float(sq)))
            if ii > 0:
                best = min(best, abs(float(src_station[ii - 1]) - float(sq)))
            along_weight = max(0.0, 1.0 - (best / safe_window))
            lateral_weight = float(np.exp(-0.5 * ((max(float(lat), 0.0) / float(safe_lateral)) ** 2)))
            query_weight[i] = np.float32(along_weight * lateral_weight)
        out[q_rows, q_cols] = np.maximum(out[q_rows, q_cols], np.clip(query_weight, 0.0, 1.0).astype(np.float32))
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
    from scipy.ndimage import distance_transform_edt, label

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

    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    component_labels, ncomp = label(centerline_ok.astype(np.uint8), structure=structure)
    if ncomp <= 0:
        domain_out = _nearest_surface_from_values(valid_mask, values)
        out[domain_mask] = domain_out[domain_mask]
        return out

    safe_along = max(float(along_scale_m), 1.0)
    safe_cross = max(float(cross_scale_m), 1.0)
    safe_pix = np.float32(max(float(pixel_size_m), 1.0e-6))

    for comp_idx in range(1, int(ncomp) + 1):
        comp_centerline = centerline_ok & (component_labels == comp_idx)
        if not np.any(comp_centerline):
            continue

        dist_to_centerline, nearest_centerline = distance_transform_edt(~comp_centerline, return_indices=True)
        nearest_comp = component_labels[nearest_centerline[0], nearest_centerline[1]] == comp_idx
        comp_domain = domain_mask & nearest_comp
        comp_valid = valid_mask & comp_domain
        if not np.any(comp_valid):
            continue

        src_rows, src_cols = np.nonzero(comp_valid)
        proj_rows = nearest_centerline[0][src_rows, src_cols]
        proj_cols = nearest_centerline[1][src_rows, src_cols]
        src_station = stationing[proj_rows, proj_cols].astype(np.float32)
        src_lateral = dist_to_centerline[src_rows, src_cols].astype(np.float32) * safe_pix
        src_vals = values[src_rows, src_cols].astype(np.float32)
        finite_src = np.isfinite(src_station) & np.isfinite(src_vals)
        if not np.any(finite_src):
            continue
        src_station = src_station[finite_src]
        src_lateral = src_lateral[finite_src]
        src_vals = src_vals[finite_src]

        order = np.argsort(src_station, kind='mergesort')
        src_station = src_station[order]
        src_lateral = src_lateral[order]
        src_vals = src_vals[order]

        center_rows, center_cols = np.nonzero(comp_centerline & np.isfinite(stationing))
        if center_rows.size == 0:
            continue
        q_station = stationing[center_rows, center_cols].astype(np.float32)
        ins = np.searchsorted(src_station, q_station)
        k = min(8, int(src_station.size))
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

        centerline_valid = comp_centerline & np.isfinite(centerline_vals)
        if not np.any(centerline_valid):
            continue

        ref_rows, ref_cols = np.nonzero(centerline_valid)
        ref_station = stationing[ref_rows, ref_cols].astype(np.float32)
        ref_vals = centerline_vals[ref_rows, ref_cols].astype(np.float32)
        order = np.argsort(ref_station, kind='mergesort')
        ref_station = ref_station[order]
        ref_vals = ref_vals[order]
        uniq_station, inverse = np.unique(ref_station, return_inverse=True)
        agg_vals = np.full(uniq_station.shape, np.nan, dtype=np.float32)
        for i in range(uniq_station.size):
            sel = inverse == i
            vals_i = ref_vals[sel]
            vals_i = vals_i[np.isfinite(vals_i)]
            if vals_i.size:
                agg_vals[i] = np.float32(np.nanmedian(vals_i))
        finite_profile = np.isfinite(agg_vals)
        if not np.any(finite_profile):
            continue
        uniq_station = uniq_station[finite_profile]
        agg_vals = agg_vals[finite_profile]
        if uniq_station.size == 0:
            continue

        qmask = comp_domain & np.isfinite(stationing)
        if not np.any(qmask):
            continue
        q_rows, q_cols = np.nonzero(qmask)
        q_station = stationing[q_rows, q_cols].astype(np.float32)
        step = float(np.nanmedian(np.diff(uniq_station))) if uniq_station.size >= 2 else max(float(pixel_size_m), 1.0)
        tol = max(step * 1.5, max(float(pixel_size_m), 1.0))
        inside = (q_station >= (float(uniq_station[0]) - tol)) & (q_station <= (float(uniq_station[-1]) + tol))
        if not np.any(inside):
            continue
        q_rows_i = q_rows[inside]
        q_cols_i = q_cols[inside]
        q_station_i = q_station[inside]
        interp_vals = np.interp(q_station_i, uniq_station, agg_vals).astype(np.float32)

        ins_i = np.searchsorted(uniq_station, q_station_i)
        best_along = np.full(q_station_i.shape, np.inf, dtype=np.float32)
        right = ins_i < int(uniq_station.size)
        if np.any(right):
            best_along[right] = np.minimum(best_along[right], np.abs(q_station_i[right] - uniq_station[ins_i[right]]).astype(np.float32))
        left = ins_i > 0
        if np.any(left):
            best_along[left] = np.minimum(best_along[left], np.abs(q_station_i[left] - uniq_station[ins_i[left] - 1]).astype(np.float32))

        lateral_m = dist_to_centerline[q_rows_i, q_cols_i].astype(np.float32) * safe_pix
        max_gap_m = max(3.0 * safe_along, 4.0 * step, 50.0)
        keep = best_along <= np.float32(max_gap_m)
        if np.any(keep):
            locality = np.exp(-0.5 * np.square(best_along[keep] / safe_along)).astype(np.float32)
            locality *= np.exp(-0.5 * np.square(lateral_m[keep] / safe_cross)).astype(np.float32)
            vals_keep = interp_vals[keep]
            vals_keep[locality < np.float32(0.02)] = np.nan
            out[q_rows_i[keep], q_cols_i[keep]] = vals_keep.astype(np.float32)
    return out






def _blend_surfaces_by_confidence(
    base_values: np.ndarray,
    base_confidence: np.ndarray,
    new_values: np.ndarray,
    new_confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(base_values, dtype=np.float32).copy()
    confidence = np.clip(np.nan_to_num(np.asarray(base_confidence, dtype=np.float32), nan=0.0), 0.0, 1.0).astype(np.float32).copy()
    new_vals = np.asarray(new_values, dtype=np.float32)
    new_conf = np.clip(np.nan_to_num(np.asarray(new_confidence, dtype=np.float32), nan=0.0), 0.0, 1.0).astype(np.float32)
    take = np.isfinite(new_vals) & (new_conf > 0.0)
    if not np.any(take):
        return values, confidence
    missing = take & ~np.isfinite(values)
    if np.any(missing):
        values[missing] = new_vals[missing]
        confidence[missing] = new_conf[missing]
    overlap = take & np.isfinite(values) & ~missing
    if np.any(overlap):
        base_w = np.clip(confidence[overlap], 0.0, 1.0).astype(np.float32)
        new_w = np.clip(new_conf[overlap], 0.0, 1.0).astype(np.float32)
        total = (base_w + new_w).astype(np.float32)
        valid = total > 0.0
        if np.any(valid):
            merged = values[overlap].astype(np.float32)
            merged[valid] = (((base_w[valid] * values[overlap][valid]) + (new_w[valid] * new_vals[overlap][valid])) / total[valid]).astype(np.float32)
            values[overlap] = merged
            confidence[overlap] = np.maximum(base_w, new_w).astype(np.float32)
    return values.astype(np.float32), confidence.astype(np.float32)


def _compute_river_channel_terrain_response(
    *,
    base_confidence: np.ndarray,
    prediction_support_confidence: Optional[np.ndarray],
    measured_anchor_fraction: Optional[np.ndarray],
    structure_only_fraction: Optional[np.ndarray],
    low_support_caution: Optional[np.ndarray],
    prediction_admissibility: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = np.clip(np.nan_to_num(np.asarray(base_confidence, dtype=np.float32), nan=0.0), 0.0, 1.0).astype(np.float32)
    tuned_confidence = np.clip(base, 0.0, 0.98).astype(np.float32)
    terrain_blend = np.clip(0.82 + (0.16 * base), 0.82, 0.98).astype(np.float32)
    cautious_structure = np.zeros(base.shape, dtype=bool)
    channel_core_preserve = np.zeros(base.shape, dtype=bool)

    support_conf = None if prediction_support_confidence is None else np.asarray(prediction_support_confidence, dtype=np.float32)
    measured = None if measured_anchor_fraction is None else np.asarray(measured_anchor_fraction, dtype=np.float32)
    structure = None if structure_only_fraction is None else np.asarray(structure_only_fraction, dtype=np.float32)
    caution = None if low_support_caution is None else np.asarray(low_support_caution, dtype=np.float32)
    admiss = None if prediction_admissibility is None else np.asarray(prediction_admissibility, dtype=np.float32)

    has_prediction = np.zeros(base.shape, dtype=bool)
    for arr in (support_conf, measured, structure, caution, admiss):
        if arr is not None:
            has_prediction |= np.isfinite(arr)
    if not np.any(has_prediction):
        return tuned_confidence, terrain_blend, cautious_structure, channel_core_preserve

    support_conf_v = np.where(np.isfinite(support_conf), np.clip(support_conf, 0.0, 1.0), base).astype(np.float32) if support_conf is not None else base.copy()
    measured_v = np.where(np.isfinite(measured), np.clip(measured, 0.0, 1.0), 0.0).astype(np.float32) if measured is not None else np.zeros(base.shape, dtype=np.float32)
    structure_v = np.where(np.isfinite(structure), np.clip(structure, 0.0, 1.0), 0.0).astype(np.float32) if structure is not None else np.zeros(base.shape, dtype=np.float32)
    caution_v = np.where(np.isfinite(caution), caution > 0.5, False) if caution is not None else np.zeros(base.shape, dtype=bool)
    admiss_v = np.where(np.isfinite(admiss), admiss > 0.5, True) if admiss is not None else np.ones(base.shape, dtype=bool)

    tuned_factor = np.clip(
        0.55
        + (0.30 * support_conf_v)
        + (0.15 * measured_v)
        - (0.18 * structure_v)
        - (0.18 * caution_v.astype(np.float32)),
        0.18,
        1.0,
    ).astype(np.float32)
    tuned_here = np.clip(base * tuned_factor, 0.0, 0.98).astype(np.float32)
    blend_here = np.clip(
        0.24
        + (0.44 * tuned_here)
        + (0.16 * measured_v)
        - (0.16 * structure_v)
        - (0.12 * caution_v.astype(np.float32)),
        0.12,
        0.94,
    ).astype(np.float32)
    inadmissible = ~admiss_v
    if np.any(inadmissible):
        inadmissible_conf_cap = np.clip(
            0.16
            + (0.18 * measured_v)
            + (0.08 * support_conf_v)
            - (0.14 * structure_v)
            - (0.12 * caution_v.astype(np.float32)),
            0.06,
            0.36,
        ).astype(np.float32)
        inadmissible_blend_cap = np.clip(
            0.12
            + (0.14 * measured_v)
            + (0.08 * support_conf_v)
            - (0.18 * structure_v)
            - (0.12 * caution_v.astype(np.float32)),
            0.06,
            0.22,
        ).astype(np.float32)
        very_inadmissible = inadmissible & caution_v & (structure_v >= 0.60) & (measured_v < 0.10) & (support_conf_v < 0.45)
        if np.any(very_inadmissible):
            inadmissible_conf_cap[very_inadmissible] = np.minimum(inadmissible_conf_cap[very_inadmissible], np.float32(0.22))
            inadmissible_blend_cap[very_inadmissible] = np.minimum(inadmissible_blend_cap[very_inadmissible], np.float32(0.14))
        tuned_here[inadmissible] = np.minimum(tuned_here[inadmissible], inadmissible_conf_cap[inadmissible]).astype(np.float32)
        blend_here[inadmissible] = np.minimum(blend_here[inadmissible], inadmissible_blend_cap[inadmissible]).astype(np.float32)

    channel_core_preserve = compute_channel_core_preserve_mask(
        prediction_support_confidence=support_conf_v,
        measured_anchor_fraction=measured_v,
        structure_only_fraction=structure_v,
        low_support_caution=caution_v.astype(np.uint8),
        prediction_admissibility=admiss_v.astype(np.uint8),
    )
    if np.any(channel_core_preserve):
        preserve_conf = np.clip(
            0.54
            + (0.18 * support_conf_v)
            + (0.08 * structure_v)
            - (0.06 * measured_v),
            0.52,
            0.88,
        ).astype(np.float32)
        preserve_blend = np.clip(
            0.58
            + (0.22 * support_conf_v)
            + (0.10 * structure_v)
            - (0.08 * measured_v),
            0.55,
            0.90,
        ).astype(np.float32)
        tuned_here[channel_core_preserve] = np.maximum(tuned_here[channel_core_preserve], preserve_conf[channel_core_preserve]).astype(np.float32)
        blend_here[channel_core_preserve] = np.maximum(blend_here[channel_core_preserve], preserve_blend[channel_core_preserve]).astype(np.float32)

    tuned_confidence[has_prediction] = tuned_here[has_prediction]
    terrain_blend[has_prediction] = blend_here[has_prediction]
    cautious_structure = has_prediction & (~admiss_v | caution_v | ((structure_v >= 0.65) & (support_conf_v < 0.45))) & (measured_v < 0.25) & (~channel_core_preserve)
    if np.any(inadmissible):
        cautious_structure |= has_prediction & inadmissible & (structure_v >= 0.45) & (measured_v < 0.20) & (~channel_core_preserve)
    return tuned_confidence, terrain_blend, cautious_structure.astype(bool), channel_core_preserve.astype(bool)


def _build_canonical_direct_primary_surface(
    *,
    river_primary_guidance_domain: np.ndarray,
    primary_river_guidance_surface: np.ndarray,
    river_bank_elevation: Optional[np.ndarray] = None,
    river_bank_influence: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    domain = np.asarray(river_primary_guidance_domain, dtype=bool)
    primary_surface = np.full(domain.shape, np.nan, dtype=np.float32)
    primary_confidence = np.zeros(domain.shape, dtype=np.float32)
    primary_source_class = np.zeros(domain.shape, dtype=np.uint8)
    support_count = np.zeros(domain.shape, dtype=np.uint8)
    xs_locality_out = np.zeros(domain.shape, dtype=np.float32)
    terrain_response_out = np.zeros(domain.shape, dtype=np.float32)
    cautious_structure_out = np.zeros(domain.shape, dtype=np.uint8)
    channel_core_preserve = np.zeros(domain.shape, dtype=np.uint8)
    direct_primary_surface = np.asarray(primary_river_guidance_surface, dtype=np.float32)
    direct_take = domain & np.isfinite(direct_primary_surface)
    if np.any(direct_take):
        primary_surface[direct_take] = direct_primary_surface[direct_take].astype(np.float32)
        primary_confidence[direct_take] = np.float32(0.9)
        terrain_response_out[direct_take] = np.float32(0.95)
        primary_source_class[direct_take] = np.uint8(5)
        support_count[direct_take] = np.uint8(1)

        if river_bank_elevation is not None and river_bank_influence is not None:
            bank_surface = np.asarray(river_bank_elevation, dtype=np.float32)
            bank_infl = np.clip(np.nan_to_num(river_bank_influence, nan=0.0), 0.0, 1.0).astype(np.float32)
            bank_take = direct_take & np.isfinite(bank_surface) & (bank_infl > np.float32(0.05))
            if np.any(bank_take):
                edge_weight = (np.float32(0.18) * np.square(bank_infl[bank_take])).astype(np.float32)
                edge_weight = np.clip(edge_weight, np.float32(0.0), np.float32(0.18)).astype(np.float32)
                primary_vals = primary_surface[bank_take].astype(np.float32)
                bank_vals = bank_surface[bank_take].astype(np.float32)
                primary_surface[bank_take] = (((np.float32(1.0) - edge_weight) * primary_vals) + (edge_weight * bank_vals)).astype(np.float32)
                primary_confidence[bank_take] = np.clip(primary_confidence[bank_take] * (np.float32(1.0) - (np.float32(0.08) * bank_infl[bank_take])), 0.0, 0.98).astype(np.float32)
    return (
        primary_surface.astype(np.float32),
        primary_confidence.astype(np.float32),
        primary_source_class.astype(np.uint8),
        support_count.astype(np.uint8),
        domain.astype(np.uint8),
        xs_locality_out.astype(np.float32),
        terrain_response_out.astype(np.float32),
        cautious_structure_out.astype(np.uint8),
        channel_core_preserve.astype(np.uint8),
    )


def _build_primary_river_surface(
    *,
    river_contract_mode: str,
    river_primary_guidance_domain: np.ndarray,
    primary_river_guidance_surface: Optional[np.ndarray],
    river_transport_centerline: Optional[np.ndarray],
    river_centerline_stationing: np.ndarray,
    river_channel_surface: np.ndarray,
    river_channel_surface_confidence: np.ndarray,
    river_channel_surface_source_class: np.ndarray,
    river_channel_surface_support_count: np.ndarray,
    river_channel_surface_prediction_support_confidence: Optional[np.ndarray],
    river_channel_surface_measured_anchor_fraction: Optional[np.ndarray],
    river_channel_surface_structure_only_fraction: Optional[np.ndarray],
    river_channel_surface_low_support_caution: Optional[np.ndarray],
    river_channel_surface_prediction_admissibility: Optional[np.ndarray],
    river_longitudinal_profile_elevation: np.ndarray,
    river_longitudinal_profile_confidence: np.ndarray,
    river_longitudinal_profile_influence: np.ndarray,
    river_centerline_elevation: np.ndarray,
    river_centerline_confidence: np.ndarray,
    river_centerline_influence: np.ndarray,
    river_xs_support_elevation: np.ndarray,
    river_xs_confidence: np.ndarray,
    river_xs_support_weight: np.ndarray,
    river_bank_elevation: np.ndarray,
    river_bank_influence: np.ndarray,
    pixel_size_m: float,
    along_scale_m: float,
    cross_scale_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    domain = np.asarray(river_primary_guidance_domain, dtype=bool)
    contract_mode = str(river_contract_mode or 'canonical_v322')
    canonical_mode = contract_mode.startswith('canonical_')
    primary_surface = np.full(domain.shape, np.nan, dtype=np.float32)
    primary_confidence = np.zeros(domain.shape, dtype=np.float32)
    primary_source_class = np.zeros(domain.shape, dtype=np.uint8)
    support_count = np.zeros(domain.shape, dtype=np.uint8)
    xs_locality_out = np.zeros(domain.shape, dtype=np.float32)
    terrain_response_out = np.zeros(domain.shape, dtype=np.float32)
    cautious_structure_out = np.zeros(domain.shape, dtype=np.uint8)
    if not np.any(domain):
        return primary_surface, primary_confidence, primary_source_class, support_count, domain.astype(np.uint8), xs_locality_out, terrain_response_out, cautious_structure_out, np.zeros(domain.shape, dtype=np.uint8)

    # Phase 6: the structured channel surface is the only active river input to final conditioning.
    # Legacy longitudinal/centerline/XS/bank rasters remain diagnostics and are no longer merged
    # into the primary river surface when a channel surface is available.
    channel_take = domain & np.isfinite(river_channel_surface) & (np.clip(river_channel_surface_confidence, 0.0, 1.0) > 0.0)
    if np.any(channel_take):
        tuned_channel_confidence, terrain_response_blend, cautious_structure, channel_core_preserve = _compute_river_channel_terrain_response(
            base_confidence=river_channel_surface_confidence,
            prediction_support_confidence=river_channel_surface_prediction_support_confidence,
            measured_anchor_fraction=river_channel_surface_measured_anchor_fraction,
            structure_only_fraction=river_channel_surface_structure_only_fraction,
            low_support_caution=river_channel_surface_low_support_caution,
            prediction_admissibility=river_channel_surface_prediction_admissibility,
        )
        primary_surface[channel_take] = river_channel_surface[channel_take].astype(np.float32)
        primary_confidence[channel_take] = np.clip(tuned_channel_confidence[channel_take], 0.0, 0.98).astype(np.float32)
        terrain_response_out[channel_take] = terrain_response_blend[channel_take].astype(np.float32)
        cautious_structure_out[channel_take] = cautious_structure[channel_take].astype(np.uint8)
        primary_source_class[channel_take] = np.uint8(5)
        support_count[channel_take] = np.clip(
            np.nan_to_num(river_channel_surface_support_count[channel_take], nan=0.0),
            1.0,
            8.0,
        ).astype(np.uint8)
        primary_surface[~domain] = np.nan
        primary_confidence[~domain] = 0.0
        primary_source_class[~domain] = np.uint8(0)
        support_count[~domain] = np.uint8(0)
        return (
            primary_surface.astype(np.float32),
            primary_confidence.astype(np.float32),
            primary_source_class.astype(np.uint8),
            support_count.astype(np.uint8),
            domain.astype(np.uint8),
            xs_locality_out.astype(np.float32),
            terrain_response_out.astype(np.float32),
            cautious_structure_out.astype(np.uint8),
            channel_core_preserve.astype(np.uint8),
        )

    if canonical_mode:
        return (
            primary_surface.astype(np.float32),
            primary_confidence.astype(np.float32),
            primary_source_class.astype(np.uint8),
            support_count.astype(np.uint8),
            domain.astype(np.uint8),
            xs_locality_out.astype(np.float32),
            terrain_response_out.astype(np.float32),
            cautious_structure_out.astype(np.uint8),
            np.zeros(domain.shape, dtype=np.uint8),
        )

    # Fallback only when structured channel surface is absent.
    routed_kwargs = dict(
        domain_mask=domain,
        centerline_mask=river_transport_centerline,
        stationing_raster=river_centerline_stationing,
        along_scale_m=along_scale_m,
        cross_scale_m=cross_scale_m,
        pixel_size_m=pixel_size_m,
    )

    def _merge_component(values: np.ndarray, confidence: np.ndarray, *, source_code: int):
        nonlocal primary_surface, primary_confidence, primary_source_class, support_count
        vals = np.asarray(values, dtype=np.float32)
        conf = np.clip(np.nan_to_num(np.asarray(confidence, dtype=np.float32), nan=0.0), 0.0, 1.0).astype(np.float32)
        take = domain & np.isfinite(vals) & (conf > 0.0)
        if not np.any(take):
            return
        support_count[take] = np.clip(support_count[take].astype(np.int16) + 1, 0, 255).astype(np.uint8)
        missing = take & ~np.isfinite(primary_surface)
        if np.any(missing):
            primary_surface[missing] = vals[missing]
            primary_confidence[missing] = conf[missing]
            primary_source_class[missing] = np.uint8(source_code)
        overlap = take & np.isfinite(primary_surface) & ~missing
        if np.any(overlap):
            prev_vals = primary_surface[overlap].astype(np.float32)
            prev_conf = np.clip(primary_confidence[overlap], 0.0, 1.0).astype(np.float32)
            new_vals = vals[overlap].astype(np.float32)
            new_conf = conf[overlap].astype(np.float32)
            total = (prev_conf + new_conf).astype(np.float32)
            valid = total > 0.0
            if np.any(valid):
                merged = prev_vals.copy()
                merged[valid] = (((prev_conf[valid] * prev_vals[valid]) + (new_conf[valid] * new_vals[valid])) / total[valid]).astype(np.float32)
                primary_surface[overlap] = merged
                primary_confidence[overlap] = np.maximum(prev_conf, new_conf).astype(np.float32)
                stronger = new_conf > prev_conf
                if np.any(stronger):
                    overlap_idx = np.flatnonzero(overlap)
                    primary_source_class.flat[overlap_idx[stronger]] = np.uint8(source_code)

    lp_seed = domain & np.isfinite(river_longitudinal_profile_elevation) & (np.clip(river_longitudinal_profile_influence, 0.0, 1.0) >= np.float32(0.05))
    if np.any(lp_seed):
        lp_routed = _nearest_surface_within_domain(lp_seed, river_longitudinal_profile_elevation, **routed_kwargs)
        lp_conf = np.clip(np.maximum(river_longitudinal_profile_confidence, river_longitudinal_profile_influence), 0.0, 1.0).astype(np.float32)
        _merge_component(lp_routed, lp_conf, source_code=1)

    cl_seed = domain & np.isfinite(river_centerline_elevation) & (np.clip(river_centerline_influence, 0.0, 1.0) > 0.0)
    if np.any(cl_seed):
        cl_core = np.clip(river_centerline_influence, 0.0, 1.0).astype(np.float32)
        cl_routed = _nearest_surface_within_domain(cl_seed, river_centerline_elevation, **routed_kwargs)
        cl_routed[cl_core <= 0.0] = np.nan
        cl_conf = np.clip(river_centerline_confidence * cl_core, 0.0, 0.92).astype(np.float32)
        _merge_component(cl_routed, cl_conf, source_code=2)

    primary_surface[~domain] = np.nan
    primary_confidence[~domain] = 0.0
    primary_source_class[~domain] = np.uint8(0)
    support_count[~domain] = np.uint8(0)
    return (
        primary_surface.astype(np.float32),
        primary_confidence.astype(np.float32),
        primary_source_class.astype(np.uint8),
        support_count.astype(np.uint8),
        domain.astype(np.uint8),
        xs_locality_out.astype(np.float32),
        terrain_response_out.astype(np.float32),
        cautious_structure_out.astype(np.uint8),
        np.zeros(domain.shape, dtype=np.uint8),
    )

def _tokenize_support_note(support_note: str | None) -> list[str]:
    return [part.strip() for part in str(support_note or '').split(';') if str(part).strip()]


def _build_primary_river_guidance_summary(*,
    primary_surface: np.ndarray,
    primary_domain: np.ndarray,
    primary_contract: dict[str, Any],
    support_note: str | None,
    primary_input_surface_source: str,
    river_contract_mode: str,
    canonical_direct_primary_mode: bool = False,
    legacy_structured_take_bypassed: bool = False,
    primary_builder_inputs_simplified: bool = False,
    canonical_direct_primary_single_path: bool = False,
    canonical_direct_primary_weighting_single_path: bool = False,
    canonical_direct_primary_downstream_sidepaths_bypassed: bool = False,
    canonical_direct_primary_single_write_path: bool = False,
    canonical_bank_input_finite_pixels: int = 0,
    canonical_bank_influence_active_pixels: int = 0,
    canonical_bank_shaped_pixels: int = 0,
) -> dict[str, Any]:
    domain = np.asarray(primary_domain, dtype=bool)
    surface = np.asarray(primary_surface, dtype=np.float32)
    finite = domain & np.isfinite(surface)
    metrics = primary_contract.get('metrics', {}) if isinstance(primary_contract, dict) else {}
    source_counts = metrics.get('source_class_counts', {}) if isinstance(metrics.get('source_class_counts', {}), dict) else {}
    continuity_tokens = [tok for tok in _tokenize_support_note(support_note) if ('backstop' in tok) or ('hard_continuity_fallback' in tok)]
    contract_mode = str(river_contract_mode or 'canonical_v322')
    canonical_mode = contract_mode.startswith('canonical_')
    deprecated_alias_used = primary_input_surface_source == 'legacy_river_depth_guidance_alias'
    deprecated_alias_rejected = primary_input_surface_source == 'rejected_legacy_river_depth_guidance_alias'
    deprecated_guide_points_used = primary_input_surface_source == 'native_river_guide_points_rasterized'
    deprecated_guide_points_rejected = primary_input_surface_source == 'rejected_native_river_guide_points_path'
    dominant_source = 'none'
    if source_counts:
        dominant_source = max(source_counts.items(), key=lambda kv: (int(kv[1] or 0), str(kv[0])))[0]
    structured_pixels = int(source_counts.get('channel_surface_scaffold', 0) or 0)
    backbone_pixels = int(metrics.get('backbone_pixels', 0) or 0)
    direct_primary_used = primary_input_surface_source == 'primary_river_guidance_surface'
    degraded_reasons: list[str] = []
    if not bool(primary_contract.get('ok', False)):
        degraded_reasons.append('primary_surface_contract_failed')
    if int(np.count_nonzero(finite)) <= 0:
        degraded_reasons.append('no_primary_surface_coverage')
    if structured_pixels <= 0 and not direct_primary_used:
        degraded_reasons.append('structured_channel_surface_absent')
    if continuity_tokens:
        degraded_reasons.append('continuity_safeguard_active')
    if deprecated_alias_used:
        degraded_reasons.append('deprecated_primary_guidance_alias_used')
    if deprecated_alias_rejected:
        degraded_reasons.append('deprecated_primary_guidance_alias_rejected')
    if deprecated_guide_points_used:
        degraded_reasons.append('deprecated_native_river_guide_points_used')
    if deprecated_guide_points_rejected:
        degraded_reasons.append('deprecated_native_river_guide_points_rejected')
    if canonical_mode and structured_pixels <= 0 and not direct_primary_used:
        degraded_reasons.append('missing_channel_surface_primary_path')
    builder_mode = 'no_primary_surface'
    if direct_primary_used and int(np.count_nonzero(finite)) > 0:
        builder_mode = 'direct_primary_surface_alias'
    elif structured_pixels > 0:
        builder_mode = 'channel_surface_scaffold'
    elif backbone_pixels > 0:
        builder_mode = 'backbone_fallback'
    return {
        'active_product_name': 'river_primary_surface',
        'active_product_role': 'primary',
        'primary_input_surface_source': str(primary_input_surface_source or 'none'),
        'primary_input_contract_mode': contract_mode,
        'primary_input_contract_ok': bool((not deprecated_alias_rejected) and (not deprecated_guide_points_rejected)),
        'legacy_primary_guidance_alias_used': bool(deprecated_alias_used),
        'deprecated_primary_guidance_alias_rejected': bool(deprecated_alias_rejected),
        'deprecated_primary_guidance_alias_allowed': bool(deprecated_alias_used and (not canonical_mode)),
        'deprecated_native_river_guide_points_allowed': bool(deprecated_guide_points_used and (not canonical_mode)),
        'deprecated_native_river_guide_points_rejected': bool(deprecated_guide_points_rejected),
        'diagnostic_guidance_surface_name': 'guidance_surface',
        'diagnostic_guidance_surface_role': 'diagnostic_only',
        'primary_builder_mode': builder_mode,
        'dominant_primary_source_class': str(dominant_source),
        'primary_surface_contract_ok': bool(primary_contract.get('ok', False)),
        'primary_surface_domain_pixels': int(np.count_nonzero(domain)),
        'primary_surface_finite_pixels': int(np.count_nonzero(finite)),
        'primary_surface_coverage_fraction': float(metrics.get('coverage_fraction', 0.0) or 0.0),
        'primary_source_class_counts': {str(k): int(v) for k, v in source_counts.items()},
        'backbone_fraction': float(metrics.get('backbone_fraction', 0.0) or 0.0),
        'continuity_safeguard_used': bool(continuity_tokens),
        'continuity_safeguard_labels': continuity_tokens,
        'degraded_mode_active': bool(degraded_reasons),
        'degraded_mode_reasons': degraded_reasons,
        'canonical_direct_primary_mode': bool(canonical_direct_primary_mode),
        'legacy_structured_take_bypassed': bool(legacy_structured_take_bypassed),
        'primary_builder_inputs_simplified': bool(primary_builder_inputs_simplified),
        'canonical_direct_primary_single_path': bool(canonical_direct_primary_single_path),
        'canonical_direct_primary_weighting_single_path': bool(canonical_direct_primary_weighting_single_path),
        'canonical_direct_primary_downstream_sidepaths_bypassed': bool(canonical_direct_primary_downstream_sidepaths_bypassed),
        'canonical_direct_primary_single_write_path': bool(canonical_direct_primary_single_write_path),
        'canonical_direct_primary_explicit_bank_only': bool(canonical_direct_primary_mode),
        'canonical_direct_primary_no_domain_fallback': bool(canonical_direct_primary_mode),
        'canonical_bank_input_finite_pixels': int(canonical_bank_input_finite_pixels or 0),
        'canonical_bank_influence_active_pixels': int(canonical_bank_influence_active_pixels or 0),
        'canonical_bank_shaped_pixels': int(canonical_bank_shaped_pixels or 0),
        'canonical_bank_shaping_active': bool(int(canonical_bank_shaped_pixels or 0) > 0),
        'support_note_tokens': _tokenize_support_note(support_note),
    }


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
    memory_diagnostics = [memory_checkpoint("terrain_interpolator_start")]
    log.info("[MEMORY][TERRAIN] %s", memory_diagnostics[-1])

    auth = sanitize_for_output(inputs.auth, dtype=np.float32)
    candidate = None if inputs.candidate is None else sanitize_for_output(inputs.candidate, dtype=np.float32)
    background_surface = None if inputs.background_surface is None else sanitize_for_output(inputs.background_surface, dtype=np.float32)
    sdb_depth_guidance, river_depth_guidance, primary_river_input_source = _resolve_native_guidance_arrays(inputs)
    sdb_depth_guidance = None if sdb_depth_guidance is None else sanitize_for_output(sdb_depth_guidance, dtype=np.float32)
    river_depth_guidance = None if river_depth_guidance is None else sanitize_for_output(river_depth_guidance, dtype=np.float32)
    work_shape = auth.shape
    _validate_resolved_guidance_shapes(
        work_shape,
        sdb_depth_guidance=sdb_depth_guidance,
        river_depth_guidance=river_depth_guidance,
    )
    sdb_ok = np.asarray(inputs.sdb_ok, dtype=bool)
    river_ok = np.asarray(inputs.river_ok, dtype=bool)
    sdb_ok_input = sdb_ok.copy()
    river_ok_input = river_ok.copy()
    sdb_gw = None if inputs.sdb_gw is None else sanitize_for_output(inputs.sdb_gw, dtype=np.float32)
    sdb_ti = None if inputs.sdb_ti is None else np.asarray(inputs.sdb_ti)
    river_gw = None if inputs.river_gw is None else sanitize_for_output(inputs.river_gw, dtype=np.float32)
    river_ti = None if inputs.river_ti is None else np.asarray(inputs.river_ti)
    river_support = None if inputs.river_support is None else np.asarray(inputs.river_support)
    river_support_depth = None if inputs.river_support_depth is None else sanitize_for_output(inputs.river_support_depth, dtype=np.float32)
    river_corridor_mask = np.asarray(inputs.river_corridor_mask, dtype=bool) if inputs.river_corridor_mask is not None else None
    river_bank_influence_input = None if inputs.river_bank_influence is None else sanitize_for_output(inputs.river_bank_influence, dtype=np.float32)
    river_bank_elevation_input = None if inputs.river_bank_elevation is None else sanitize_for_output(inputs.river_bank_elevation, dtype=np.float32)
    explicit_bank_influence_input = None if river_bank_influence_input is None else river_bank_influence_input.copy()
    explicit_bank_elevation_input = None if river_bank_elevation_input is None else river_bank_elevation_input.copy()
    river_bank_pair_weight = None if inputs.river_bank_pair_weight is None else sanitize_for_output(inputs.river_bank_pair_weight, dtype=np.float32)
    river_bank_continuity_weight = None if inputs.river_bank_continuity_weight is None else sanitize_for_output(inputs.river_bank_continuity_weight, dtype=np.float32)
    river_bank_graph_confidence = None if inputs.river_bank_graph_confidence is None else sanitize_for_output(inputs.river_bank_graph_confidence, dtype=np.float32)
    river_bank_confluence_damping = None if inputs.river_bank_confluence_damping is None else sanitize_for_output(inputs.river_bank_confluence_damping, dtype=np.float32)
    river_bank_estuary_side_decay = None if inputs.river_bank_estuary_side_decay is None else sanitize_for_output(inputs.river_bank_estuary_side_decay, dtype=np.float32)
    river_centerline_elevation = _float32_or_nan(work_shape, inputs.river_centerline_elevation)
    river_centerline_influence = _float32_or_zero(work_shape, inputs.river_centerline_influence)
    river_centerline_stationing = _float32_or_nan(work_shape, inputs.river_centerline_stationing)
    river_channel_surface = _float32_or_nan(work_shape, inputs.river_channel_surface)
    river_channel_surface_confidence = _float32_or_zero(work_shape, inputs.river_channel_surface_confidence)
    river_channel_surface_support_count = _float32_or_zero(work_shape, inputs.river_channel_surface_support_count)
    river_channel_surface_authoritative_lock_scope = np.asarray(inputs.river_channel_surface_authoritative_lock_scope, dtype=bool) if inputs.river_channel_surface_authoritative_lock_scope is not None else np.zeros(work_shape, dtype=bool)
    river_channel_surface_authoritative_lock_applied = np.asarray(inputs.river_channel_surface_authoritative_lock_applied, dtype=bool) if inputs.river_channel_surface_authoritative_lock_applied is not None else np.zeros(work_shape, dtype=bool)
    river_channel_surface_prediction_support_confidence = _float32_or_nan(work_shape, inputs.river_channel_surface_prediction_support_confidence)
    river_channel_surface_measured_anchor_fraction = _float32_or_nan(work_shape, inputs.river_channel_surface_measured_anchor_fraction)
    river_channel_surface_structure_only_fraction = _float32_or_nan(work_shape, inputs.river_channel_surface_structure_only_fraction)
    river_channel_surface_low_support_caution = _float32_or_zero(work_shape, inputs.river_channel_surface_low_support_caution)
    river_channel_surface_prediction_admissibility = _float32_or_nan(work_shape, inputs.river_channel_surface_prediction_admissibility)
    river_longitudinal_profile_elevation = _float32_or_nan(work_shape, inputs.river_longitudinal_profile_elevation)
    river_longitudinal_profile_uncertainty = None if inputs.river_longitudinal_profile_uncertainty is None else sanitize_for_output(inputs.river_longitudinal_profile_uncertainty, dtype=np.float32)
    river_longitudinal_profile_influence = _float32_or_zero(work_shape, inputs.river_longitudinal_profile_influence)
    river_longitudinal_profile_local_authoritative_reconciliation = _float32_or_nan(work_shape, inputs.river_longitudinal_profile_local_authoritative_reconciliation)
    river_longitudinal_profile_local_authoritative_reconciliation_influence = _float32_or_zero(work_shape, inputs.river_longitudinal_profile_local_authoritative_reconciliation_influence)
    river_xs_support_elevation = _float32_or_nan(work_shape, inputs.river_xs_support_elevation)
    river_xs_support_weight = _float32_or_zero(work_shape, inputs.river_xs_support_weight)
    sdb_uncertainty = None if inputs.sdb_uncertainty is None else sanitize_for_output(inputs.sdb_uncertainty, dtype=np.float32)
    river_uncertainty = None if inputs.river_uncertainty is None else sanitize_for_output(inputs.river_uncertainty, dtype=np.float32)

    has_river_centerline_elevation = inputs.river_centerline_elevation is not None
    has_river_channel_surface = inputs.river_channel_surface is not None
    has_river_longitudinal_profile_elevation = inputs.river_longitudinal_profile_elevation is not None
    has_river_centerline_influence = inputs.river_centerline_influence is not None
    has_river_xs_support_elevation = inputs.river_xs_support_elevation is not None
    has_river_xs_support_weight = inputs.river_xs_support_weight is not None
    gc.collect()
    memory_diagnostics.append(memory_checkpoint(
        "terrain_interpolator_inputs_sanitized",
        rows=int(work_shape[0]),
        cols=int(work_shape[1]),
        auth_finite=int(np.count_nonzero(np.isfinite(auth))),
        sdb_guidance_finite=int(np.count_nonzero(np.isfinite(sdb_depth_guidance))) if sdb_depth_guidance is not None else 0,
        river_guidance_finite=int(np.count_nonzero(np.isfinite(river_depth_guidance))) if river_depth_guidance is not None else 0,
    ))
    log.info("[MEMORY][TERRAIN] %s", memory_diagnostics[-1])

    locked = array_valid_mask(auth)
    gap = ~locked

    guidance_locked_contract = _verify_locked_guidance_exclusion(
        locked=locked,
        zero_arrays={
            "river_bank_influence": (river_bank_influence_input, river_ok_input),
            "river_centerline_influence": (river_centerline_influence, river_ok_input),
            "river_channel_surface_confidence": (river_channel_surface_confidence, river_ok_input),
            "river_channel_surface_support_count": (river_channel_surface_support_count, river_ok_input),
            "river_channel_surface_prediction_support_confidence": (river_channel_surface_prediction_support_confidence, river_ok_input),
            "river_channel_surface_measured_anchor_fraction": (river_channel_surface_measured_anchor_fraction, river_ok_input),
            "river_channel_surface_structure_only_fraction": (river_channel_surface_structure_only_fraction, river_ok_input),
            "river_channel_surface_low_support_caution": (river_channel_surface_low_support_caution, river_ok_input),
            "river_longitudinal_profile_influence": (river_longitudinal_profile_influence, river_ok_input),
            "river_longitudinal_profile_local_authoritative_reconciliation_influence": (river_longitudinal_profile_local_authoritative_reconciliation_influence, river_ok_input),
            "river_xs_support_weight": (river_xs_support_weight, river_ok_input),
        },
        nan_arrays={
            "sdb_depth_guidance": (sdb_depth_guidance, sdb_ok_input),
            "river_depth_guidance": (river_depth_guidance, river_ok_input),
            "river_support_depth": (river_support_depth, river_ok_input),
            "river_centerline_elevation": (river_centerline_elevation, river_ok_input),
            "river_centerline_stationing": (river_centerline_stationing, river_ok_input),
            "river_channel_surface": (river_channel_surface, river_ok_input),
            "river_channel_surface_prediction_admissibility": (river_channel_surface_prediction_admissibility, river_ok_input),
            "river_longitudinal_profile_elevation": (river_longitudinal_profile_elevation, river_ok_input),
            "river_longitudinal_profile_local_authoritative_reconciliation": (river_longitudinal_profile_local_authoritative_reconciliation, river_ok_input),
            "river_longitudinal_profile_uncertainty": (river_longitudinal_profile_uncertainty, river_ok_input),
            "river_xs_support_elevation": (river_xs_support_elevation, river_ok_input),
            "sdb_uncertainty": (sdb_uncertainty, sdb_ok_input),
            "river_uncertainty": (river_uncertainty, river_ok_input),
        },
    )

    def _zero_locked(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        out = np.asarray(arr).copy()
        if out.dtype == np.bool_:
            out[locked] = False
        else:
            out[locked] = 0
        return out

    def _nan_locked(arr: Optional[np.ndarray], *, dtype=np.float32) -> Optional[np.ndarray]:
        if arr is None:
            return None
        out = np.asarray(arr, dtype=dtype).copy()
        out[locked] = np.nan
        return out

    # Subordinate guidance is never allowed to remain active inside authoritative-locked cells.
    # Mask it here before any downstream confidence/support logic so reproducibility and lock
    # invariants operate on the actual sanitized inputs rather than failing on pre-masked arrays.
    sdb_ok = sdb_ok & gap
    river_ok = river_ok & gap
    sdb_gw = _zero_locked(sdb_gw)
    sdb_ti = _zero_locked(sdb_ti)
    river_gw = _zero_locked(river_gw)
    river_ti = _zero_locked(river_ti)
    river_support = _zero_locked(river_support)
    river_support_depth = _nan_locked(river_support_depth)
    sdb_depth_guidance = _nan_locked(sdb_depth_guidance)
    river_depth_guidance = _nan_locked(river_depth_guidance)
    # Preserve explicit bank-shaping inputs through authoritative cells.
    # These are upstream boundary/WSE-style guides used to build the canonical
    # primary river surface. Masking them on locked cells erases valid
    # authoritative corridor-edge bank samples before the canonical builder can
    # use them.
    river_bank_influence_input = (
        np.clip(np.nan_to_num(river_bank_influence_input, nan=0.0), 0.0, 1.0).astype(np.float32)
        if river_bank_influence_input is not None else None
    )
    river_bank_elevation_input = (
        np.asarray(river_bank_elevation_input, dtype=np.float32).copy()
        if river_bank_elevation_input is not None else None
    )
    river_centerline_elevation = _nan_locked(river_centerline_elevation)
    river_centerline_influence = _zero_locked(river_centerline_influence)
    river_centerline_stationing = _nan_locked(river_centerline_stationing)
    river_channel_surface = _nan_locked(river_channel_surface)
    river_channel_surface_confidence = _zero_locked(river_channel_surface_confidence)
    river_channel_surface_support_count = _zero_locked(river_channel_surface_support_count)
    river_channel_surface_prediction_support_confidence = _zero_locked(river_channel_surface_prediction_support_confidence)
    river_channel_surface_measured_anchor_fraction = _zero_locked(river_channel_surface_measured_anchor_fraction)
    river_channel_surface_structure_only_fraction = _zero_locked(river_channel_surface_structure_only_fraction)
    river_channel_surface_low_support_caution = _zero_locked(river_channel_surface_low_support_caution)
    river_channel_surface_prediction_admissibility = _nan_locked(river_channel_surface_prediction_admissibility)
    river_longitudinal_profile_elevation = _nan_locked(river_longitudinal_profile_elevation)
    river_longitudinal_profile_uncertainty = _nan_locked(river_longitudinal_profile_uncertainty)
    river_longitudinal_profile_influence = _zero_locked(river_longitudinal_profile_influence)
    river_longitudinal_profile_local_authoritative_reconciliation = _nan_locked(river_longitudinal_profile_local_authoritative_reconciliation)
    river_longitudinal_profile_local_authoritative_reconciliation_influence = _zero_locked(river_longitudinal_profile_local_authoritative_reconciliation_influence)
    river_xs_support_elevation = _nan_locked(river_xs_support_elevation)
    river_xs_support_weight = _zero_locked(river_xs_support_weight)
    sdb_uncertainty = _nan_locked(sdb_uncertainty)
    river_uncertainty = _nan_locked(river_uncertainty)

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
    canonical_mode = str(getattr(inputs, 'river_contract_mode', 'canonical_v322') or 'canonical_v322').startswith('canonical_')
    river_depth_guidance_mask = np.zeros(work_shape, dtype=bool) if river_depth_guidance is None else np.isfinite(river_depth_guidance)
    allow_direct_primary_surface = (primary_river_input_source == 'primary_river_guidance_surface')
    canonical_direct_primary_mode = canonical_mode and allow_direct_primary_surface
    river_bank_edge, river_bank_distance_m, river_bank_influence = compute_bank_distance_influence(
        river_corridor,
        pixel_size_m=cfg.pixel_size_m,
        full_influence_m=0.0,
        zero_influence_m=max(cfg.river_scaffold_transition_m * 0.18, 60.0),
    )
    if canonical_direct_primary_mode:
        # Single-path canonical bank science: use only the explicit QC'd and
        # longitudinally smoothed bank product passed into the interpolator.
        # Do not synthesize a second generic bank surface or confidence field.
        river_bank_elevation = (
            np.asarray(explicit_bank_elevation_input, dtype=np.float32).copy()
            if explicit_bank_elevation_input is not None
            else np.full(work_shape, np.float32(np.nan), dtype=np.float32)
        )
        river_bank_influence = (
            np.clip(np.nan_to_num(explicit_bank_influence_input, nan=0.0), 0.0, 1.0).astype(np.float32)
            if explicit_bank_influence_input is not None
            else np.zeros(work_shape, dtype=np.float32)
        )
        # In canonical v1, explicit bank rasters are upstream shaping inputs,
        # not final-take guidance. Do not corridor-clip them here: the single
        # active domain check is the canonical primary-surface domain used later
        # during bank shaping. Corridor clipping here was erasing the valid bank
        # edge samples extracted from the authoritative DEM and making the bank
        # handoff appear empty downstream.
        river_bank_pair_weight = None
        river_bank_continuity_weight = None
        river_bank_graph_confidence = None
        river_bank_confluence_damping = None
        river_bank_estuary_side_decay = None
    else:
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
    river_bank_influence_input = None
    river_bank_pair_weight = None
    river_bank_continuity_weight = None
    river_bank_graph_confidence = None
    river_bank_confluence_damping = None
    river_bank_estuary_side_decay = None
    gc.collect()
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
    river_authoritative_locked = np.zeros(work_shape, dtype=bool)
    river_authoritative_lock_surface = np.full(work_shape, np.nan, dtype=np.float32)

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
    if inputs.river_longitudinal_profile_local_authoritative_reconciliation_influence is not None:
        river_longitudinal_profile_local_authoritative_reconciliation_influence = np.clip(np.nan_to_num(river_longitudinal_profile_local_authoritative_reconciliation_influence, nan=0.0), 0.0, 1.0).astype(np.float32)
        river_longitudinal_profile_local_authoritative_reconciliation_influence[~river_corridor] = 0.0
    if has_river_xs_support_weight:
        river_xs_support_weight = np.clip(np.nan_to_num(river_xs_support_weight, nan=0.0), 0.0, 1.0).astype(np.float32)
        river_xs_support_weight[~river_corridor] = 0.0
    if bool(getattr(cfg, "river_disable_xs_influence", False)):
        river_xs_support_elevation[:] = np.float32(np.nan)
        river_xs_support_weight[:] = np.float32(0.0)

    if canonical_direct_primary_mode:
        # Canonical direct-primary river conditioning must run through exactly one
        # river path. Blank all legacy river support surfaces and influence fields
        # here so they cannot re-enter later through builder, takeover, uncertainty,
        # or reporting side paths.
        river_support_depth = np.full(work_shape, np.float32(np.nan), dtype=np.float32) if river_support_depth is not None else None
        river_channel_surface = np.full(work_shape, np.float32(np.nan), dtype=np.float32)
        river_channel_surface_confidence = np.zeros(work_shape, dtype=np.float32)
        river_longitudinal_profile_elevation = np.full(work_shape, np.float32(np.nan), dtype=np.float32)
        river_longitudinal_profile_confidence = np.zeros(work_shape, dtype=np.float32)
        river_longitudinal_profile_influence = np.zeros(work_shape, dtype=np.float32)
        river_centerline_elevation = np.full(work_shape, np.float32(np.nan), dtype=np.float32)
        river_centerline_confidence = np.zeros(work_shape, dtype=np.float32)
        river_centerline_influence = np.zeros(work_shape, dtype=np.float32)
        river_xs_support_elevation = np.full(work_shape, np.float32(np.nan), dtype=np.float32)
        river_xs_confidence = np.zeros(work_shape, dtype=np.float32)
        river_xs_support_weight = np.zeros(work_shape, dtype=np.float32)
        # Bank guidance remains the single allowed scientific modifier in canonical
        # direct-primary mode, but it is applied only once upstream while building
        # river_primary_surface.  It must not re-enter through downstream side paths.
        river_support_depth_mask = np.zeros(work_shape, dtype=bool)
        river_structural_domain = river_domain & river_depth_guidance_mask
        river_transport_centerline = None
        river_gw_mask = np.zeros(work_shape, dtype=bool)
        river_ti_mask = np.zeros(work_shape, dtype=bool)
        candidate_mask = np.zeros(work_shape, dtype=bool)
    else:
        river_support_depth_mask = np.isfinite(river_support_depth) if river_support_depth is not None else np.zeros(work_shape, dtype=bool)
        river_structural_domain = river_domain & (
            river_corridor
            | river_anchor
            | river_support_depth_mask
            | np.isfinite(river_channel_surface)
            | np.isfinite(river_longitudinal_profile_elevation)
            | np.isfinite(river_centerline_elevation)
            | np.isfinite(river_xs_support_elevation)
        )

        river_transport_centerline = None
        station_centerline = river_domain & np.isfinite(river_centerline_stationing)
        if np.any(station_centerline):
            river_transport_centerline = station_centerline
        else:
            river_transport_centerline = river_domain & (
                np.isfinite(river_channel_surface)
                | np.isfinite(river_longitudinal_profile_elevation)
                | np.isfinite(river_centerline_elevation)
                | (river_centerline_influence > 0.0)
                | np.isfinite(river_xs_support_elevation)
                | (river_xs_support_weight > 0.0)
            )
            if not np.any(river_transport_centerline):
                river_transport_centerline = None

        river_gw_mask = np.zeros(work_shape, dtype=bool) if river_gw is None else (np.nan_to_num(river_gw, nan=0.0).astype(np.float32) >= np.float32(0.15))
        river_ti_mask = np.zeros(work_shape, dtype=bool) if river_ti is None else (np.asarray(river_ti) > 0)
        candidate_mask = np.zeros(work_shape, dtype=bool) if candidate is None else np.isfinite(candidate)
    if canonical_direct_primary_mode:
        # Canonical v1 terrain conditioning has one river eligibility path: finite direct
        # primary guidance inside unlocked gaps. Use the direct-primary raster itself as
        # the source of truth for eligibility rather than the broader river_ok mask, which
        # can be empty even when the validated v1 primary surface is finite on the working
        # grid. The direct-primary product is already a river-only surface upstream.
        river_primary_guidance_domain = gap & river_depth_guidance_mask
        river_structural_domain = river_primary_guidance_domain.copy()
        river_transport_centerline = None
        river_structural_spread_m = np.zeros(work_shape, dtype=np.float32)
    else:
        river_primary_guidance_domain = river_domain & (
            river_anchor
            | river_support_depth_mask
            | (np.isfinite(river_longitudinal_profile_elevation) & (river_longitudinal_profile_influence >= np.float32(0.05)))
            | (np.isfinite(river_centerline_elevation) & (river_centerline_influence >= np.float32(0.10)))
            | (np.isfinite(river_xs_support_elevation) & (river_xs_support_weight >= np.float32(0.10)))
            | river_gw_mask
            | river_ti_mask
        )
        if background_surface is None:
            river_primary_guidance_domain |= river_domain & (
                (river_bank_influence >= np.float32(0.10))
                | candidate_mask
                | river_depth_guidance_mask
            )
        river_primary_guidance_domain &= river_domain
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
        if not canonical_direct_primary_mode:
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

    # Canonical direct-primary mode must run through one river path only: the emitted
    # primary surface over the unlocked river gap domain. Do not build alternate river
    # anchor surfaces or bank-constrained guidance side paths in this mode.
    if not canonical_direct_primary_mode:
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

    if not canonical_direct_primary_mode:
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
        river_longitudinal_profile_reconciliation_boost = (
            river_longitudinal_profile_local_authoritative_reconciliation_influence
            if inputs.river_longitudinal_profile_local_authoritative_reconciliation_influence is not None
            else np.zeros(work_shape, dtype=np.float32)
        )
        river_longitudinal_profile_confidence = combine_structural_confidence(
            river_longitudinal_profile_influence if inputs.river_longitudinal_profile_influence is not None else river_scaffold_confidence,
            river_longitudinal_profile_reconciliation_boost,
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
            lp_guidance_uncertainty = np.maximum(lp_guidance_uncertainty * (1.0 - (0.35 * np.clip(river_longitudinal_profile_reconciliation_boost, 0.0, 1.0))), 0.05).astype(np.float32)
            lp_mix, _ = compute_inverse_variance_blend_weights(
                anchor_uncertainty=anchor_uncertainty[river_longitudinal_profile_constrained],
                guidance_uncertainty=lp_guidance_uncertainty[river_longitudinal_profile_constrained],
                fallback_guidance_influence=np.maximum(river_longitudinal_profile_confidence[river_longitudinal_profile_constrained], 0.82 + (0.10 * river_longitudinal_profile_reconciliation_boost[river_longitudinal_profile_constrained])).astype(np.float32),
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
        # domain guidance rasters are absent, let SDB draw from the candidate under its own
        # domain mask rather than silently discarding it. River guidance now owns the primary
        # in-channel surface; canonical modes do not allow the candidate to become a peer
        # primary river surface substitute inside the interpolator.
        if sdb_depth_guidance is None:
            sdb_depth_guidance = candidate
        if (not canonical_mode) and river_depth_guidance is None:
            river_depth_guidance = candidate

    if canonical_direct_primary_mode:
        river_primary_surface, river_primary_surface_confidence, river_primary_surface_source_class, river_primary_surface_support_count, river_primary_surface_domain_mask, river_primary_surface_xs_locality, river_primary_surface_terrain_response, river_primary_surface_cautious_structure, river_primary_surface_channel_core_preserve = _build_canonical_direct_primary_surface(
            river_primary_guidance_domain=river_primary_guidance_domain,
            primary_river_guidance_surface=river_depth_guidance,
            river_bank_elevation=explicit_bank_elevation_input,
            river_bank_influence=explicit_bank_influence_input,
        )
    else:
        river_primary_surface, river_primary_surface_confidence, river_primary_surface_source_class, river_primary_surface_support_count, river_primary_surface_domain_mask, river_primary_surface_xs_locality, river_primary_surface_terrain_response, river_primary_surface_cautious_structure, river_primary_surface_channel_core_preserve = _build_primary_river_surface(
            river_contract_mode=getattr(inputs, 'river_contract_mode', 'canonical_v322'),
            river_primary_guidance_domain=river_primary_guidance_domain,
            primary_river_guidance_surface=river_depth_guidance if primary_river_input_source == 'primary_river_guidance_surface' else None,
            river_transport_centerline=river_transport_centerline,
            river_centerline_stationing=river_centerline_stationing,
            river_channel_surface=river_channel_surface,
            river_channel_surface_confidence=river_channel_surface_confidence,
            river_channel_surface_source_class=inputs.river_channel_surface_source_class if inputs.river_channel_surface_source_class is not None else np.zeros(work_shape, dtype=np.uint8),
            river_channel_surface_support_count=river_channel_surface_support_count,
            river_channel_surface_prediction_support_confidence=river_channel_surface_prediction_support_confidence,
            river_channel_surface_measured_anchor_fraction=river_channel_surface_measured_anchor_fraction,
            river_channel_surface_structure_only_fraction=river_channel_surface_structure_only_fraction,
            river_channel_surface_low_support_caution=river_channel_surface_low_support_caution,
            river_channel_surface_prediction_admissibility=river_channel_surface_prediction_admissibility,
            river_longitudinal_profile_elevation=river_longitudinal_profile_elevation,
            river_longitudinal_profile_confidence=river_longitudinal_profile_confidence,
            river_longitudinal_profile_influence=river_longitudinal_profile_influence,
            river_centerline_elevation=river_centerline_elevation,
            river_centerline_confidence=river_centerline_confidence,
            river_centerline_influence=river_centerline_influence,
            river_xs_support_elevation=river_xs_support_elevation,
            river_xs_confidence=river_xs_confidence,
            river_xs_support_weight=river_xs_support_weight,
            river_bank_elevation=river_bank_elevation,
            river_bank_influence=river_bank_influence,
            pixel_size_m=cfg.pixel_size_m,
            along_scale_m=cfg.river_aniso_along_scale_m,
            cross_scale_m=cfg.river_aniso_cross_scale_m,
        )
    river_primary_surface_domain = np.asarray(river_primary_surface_domain_mask, dtype=bool) & np.isfinite(river_primary_surface)
    if canonical_direct_primary_mode and (not np.any(river_primary_surface_domain)):
        raise RuntimeError(
            "canonical direct-primary river path produced an empty primary surface domain; "
            f"gap_finite_direct_primary={int(np.count_nonzero(gap & river_depth_guidance_mask))} "
            f"river_ok_gap_finite_direct_primary={int(np.count_nonzero(gap & river_ok & river_depth_guidance_mask))}"
        )
    river_primary_surface_contract = validate_river_primary_surface_contract(
        primary_surface=river_primary_surface,
        primary_confidence=river_primary_surface_confidence,
        source_class=river_primary_surface_source_class,
        support_count=river_primary_surface_support_count,
        domain=np.asarray(river_primary_surface_domain_mask, dtype=bool),
        bank_influence=river_bank_influence,
        xs_locality=river_primary_surface_xs_locality,
    )
    if not river_primary_surface_contract.get("ok", False):
        raise RuntimeError(f"river primary surface contract failed: {river_primary_surface_contract}")
    river_active_guidance_domain = river_primary_surface_domain if canonical_direct_primary_mode else river_primary_guidance_domain
    primary_unc = np.clip(0.10 - (0.06 * np.clip(river_primary_surface_confidence, 0.0, 1.0)), 0.03, 0.10).astype(np.float32)
    if np.any(river_primary_surface_domain):
        guidance_surface[river_primary_surface_domain] = river_primary_surface[river_primary_surface_domain]
        guidance_uncertainty[river_primary_surface_domain] = primary_unc[river_primary_surface_domain]
        guidance_uncertainty_source_available[river_primary_surface_domain] = True
        guidance_influence[river_primary_surface_domain] = np.maximum(
            guidance_influence[river_primary_surface_domain],
            np.clip(river_primary_surface_terrain_response[river_primary_surface_domain], 0.0, 0.98).astype(np.float32),
        )
        top_level_guidance_uncertainty[river_primary_surface_domain] = np.where(
            np.isfinite(top_level_guidance_uncertainty[river_primary_surface_domain]),
            np.minimum(top_level_guidance_uncertainty[river_primary_surface_domain], primary_unc[river_primary_surface_domain]),
            primary_unc[river_primary_surface_domain],
        ).astype(np.float32)

    if canonical_direct_primary_mode:
        river_authoritative_lock_surface = np.full(work_shape, np.nan, dtype=np.float32)
        river_authoritative_locked = np.zeros(work_shape, dtype=bool)
    else:
        river_authoritative_lock_surface = np.where(
            np.isfinite(river_primary_surface),
            river_primary_surface,
            river_channel_surface,
        ).astype(np.float32)
        river_authoritative_locked = river_domain & river_channel_surface_authoritative_lock_applied & np.isfinite(river_authoritative_lock_surface)
        if np.any(river_authoritative_locked):
            guidance_surface[river_authoritative_locked] = river_authoritative_lock_surface[river_authoritative_locked]
            guidance_uncertainty[river_authoritative_locked] = 0.0
            guidance_uncertainty_source_available[river_authoritative_locked] = True
            guidance_influence[river_authoritative_locked] = 1.0
            top_level_guidance_uncertainty[river_authoritative_locked] = 0.0
        # Legacy/non-canonical river modes may still backfill finite direct guidance into
        # any unresolved river pixels. Canonical direct-primary mode must not take this
        # second write path; its river surface may only enter through river_primary_surface.
        if river_depth_guidance is not None:
            river_direct = river_primary_guidance_domain & np.isfinite(river_depth_guidance) & ~np.isfinite(guidance_surface)
            if np.any(river_direct):
                guidance_surface[river_direct] = river_depth_guidance[river_direct]
                if river_uncertainty is not None:
                    guidance_uncertainty[river_direct] = river_uncertainty[river_direct]
                    guidance_uncertainty_source_available[river_direct] = np.isfinite(river_uncertainty[river_direct])

    river_cautious_structure_mask = river_primary_surface_domain & (np.asarray(river_primary_surface_cautious_structure, dtype=bool)) & (~river_authoritative_locked)
    if np.any(river_cautious_structure_mask):
        support[river_cautious_structure_mask] = int(SupportClass.SCAFFOLD_INFERRED)

    sdb_direct = gap & sdb_ok & (~river_ok | estuary_transition)
    if sdb_depth_guidance is not None:
        sdb_direct &= np.isfinite(sdb_depth_guidance)
        if np.any(sdb_direct):
            guidance_surface[sdb_direct] = sdb_depth_guidance[sdb_direct]
            if sdb_uncertainty is not None:
                guidance_uncertainty[sdb_direct] = sdb_uncertainty[sdb_direct]
                guidance_uncertainty_source_available[sdb_direct] = np.isfinite(sdb_uncertainty[sdb_direct])

    river_longitudinal_profile_primary = np.zeros(work_shape, dtype=bool)
    lp_primary_blend_floor = np.zeros(work_shape, dtype=np.float32)
    if not canonical_direct_primary_mode:
        river_longitudinal_profile_primary = river_primary_guidance_domain & np.isfinite(river_longitudinal_profile_elevation) & (~river_authoritative_locked)
        if inputs.river_longitudinal_profile_influence is not None:
            river_longitudinal_profile_primary &= (river_longitudinal_profile_influence >= np.float32(0.05))
        lp_primary_blend_floor = np.zeros(work_shape, dtype=np.float32)
        if np.any(river_longitudinal_profile_primary):
            lp_primary = river_longitudinal_profile_primary
            lp_primary_blend_floor = np.clip(
                0.90 + (0.08 * river_longitudinal_profile_confidence) + (0.04 * river_longitudinal_profile_reconciliation_boost),
                0.90,
                0.99,
            ).astype(np.float32)
            if river_longitudinal_profile_uncertainty is not None:
                lp_unc_primary = np.where(
                    np.isfinite(river_longitudinal_profile_uncertainty),
                    river_longitudinal_profile_uncertainty,
                    np.clip((0.08 - (0.04 * river_longitudinal_profile_confidence)) * (1.0 - (0.35 * np.clip(river_longitudinal_profile_reconciliation_boost, 0.0, 1.0))), 0.02, 0.08),
                ).astype(np.float32)
                lp_unc_observed = np.isfinite(river_longitudinal_profile_uncertainty)
            else:
                lp_unc_primary = np.clip((0.08 - (0.04 * river_longitudinal_profile_confidence)) * (1.0 - (0.35 * np.clip(river_longitudinal_profile_reconciliation_boost, 0.0, 1.0))), 0.02, 0.08).astype(np.float32)
                lp_unc_observed = np.zeros(work_shape, dtype=bool)
            existing = np.isfinite(guidance_surface) & lp_primary
            if np.any(existing):
                lp_weight = np.clip(0.78 + (0.20 * river_longitudinal_profile_confidence[existing]) + (0.06 * river_longitudinal_profile_reconciliation_boost[existing]), 0.78, 0.99).astype(np.float32)
                guidance_surface[existing] = (((1.0 - lp_weight) * guidance_surface[existing]) + (lp_weight * river_longitudinal_profile_elevation[existing])).astype(np.float32)
                guidance_uncertainty[existing] = np.where(
                    np.isfinite(guidance_uncertainty[existing]),
                    np.minimum(guidance_uncertainty[existing], lp_unc_primary[existing]),
                    lp_unc_primary[existing],
                ).astype(np.float32)
                guidance_uncertainty_source_available[existing] = np.where(
                    np.isfinite(guidance_uncertainty[existing]),
                    guidance_uncertainty_source_available[existing] | lp_unc_observed[existing],
                    lp_unc_observed[existing],
                )
            fill_lp = lp_primary & ~np.isfinite(guidance_surface)
            if np.any(fill_lp):
                guidance_surface[fill_lp] = river_longitudinal_profile_elevation[fill_lp]
                guidance_uncertainty[fill_lp] = lp_unc_primary[fill_lp]
                guidance_uncertainty_source_available[fill_lp] = lp_unc_observed[fill_lp]
            if np.any(lp_primary):
                guidance_influence[lp_primary] = np.maximum(guidance_influence[lp_primary], lp_primary_blend_floor[lp_primary])
                top_level_guidance_uncertainty[lp_primary] = np.where(
                    np.isfinite(top_level_guidance_uncertainty[lp_primary]),
                    np.minimum(top_level_guidance_uncertainty[lp_primary], lp_unc_primary[lp_primary]),
                    lp_unc_primary[lp_primary],
                ).astype(np.float32)

        structured_river_guidance = river_primary_guidance_domain & np.isnan(guidance_surface)
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

    active_guidance_domain = sdb_ok | river_active_guidance_domain
    baseline_exact_domain = np.zeros(work_shape, dtype=bool)
    conditioned = np.full_like(auth, np.nan, dtype=np.float32)
    conditioned[locked] = auth[locked]
    if background_surface is not None:
        baseline_exact_domain = gap & (~active_guidance_domain) & np.isfinite(background_surface)
        if np.any(baseline_exact_domain):
            conditioned[baseline_exact_domain] = background_surface[baseline_exact_domain].astype(np.float32)
            support[baseline_exact_domain] = int(SupportClass.ANCHORED_INTERPOLATION)
            guidance_influence[baseline_exact_domain] = 0.0
    take_any = gap & active_guidance_domain & np.isfinite(guidance_surface)
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
    if np.any(river_longitudinal_profile_primary):
        top_level_blend_w[river_longitudinal_profile_primary] = np.maximum(
            top_level_blend_w[river_longitudinal_profile_primary],
            lp_primary_blend_floor[river_longitudinal_profile_primary],
        ).astype(np.float32)
    if np.any(river_authoritative_locked):
        top_level_blend_w[river_authoritative_locked] = 1.0
    canonical_take = np.zeros(work_shape, dtype=bool)
    if canonical_direct_primary_mode and np.any(river_active_guidance_domain):
        top_level_blend_w[river_active_guidance_domain] = 1.0
        canonical_take = take_any & river_active_guidance_domain
        if np.any(canonical_take):
            conditioned[canonical_take] = guidance_surface[canonical_take].astype(np.float32)
    blended_take = take_any & ~canonical_take
    if np.any(blended_take):
        conditioned[blended_take] = ((1.0 - top_level_blend_w[blended_take]) * anchor_surface[blended_take] + top_level_blend_w[blended_take] * guidance_surface[blended_take]).astype(np.float32)
    guidance_influence = top_level_blend_w.astype(np.float32)

    remaining_gap = gap & ~np.isfinite(conditioned)
    support_note = "terrain_interpolator_support_weighted"
    non_guidance_gap = remaining_gap & (~active_guidance_domain)
    if np.any(non_guidance_gap) and background_surface is not None:
        background_fill = non_guidance_gap & np.isfinite(background_surface)
        if np.any(background_fill):
            conditioned[background_fill] = background_surface[background_fill].astype(np.float32)
            support[background_fill] = int(SupportClass.ANCHORED_INTERPOLATION)
            guidance_influence[background_fill] = 0.0
            support_note = "terrain_interpolator_with_baseline_cudem_background"
            remaining_gap = gap & ~np.isfinite(conditioned)

    if np.any(remaining_gap):
        try:
            from scipy.ndimage import distance_transform_edt
            if not canonical_direct_primary_mode:
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
                if canonical_direct_primary_mode:
                    support_note = f"{support_note}; canonical_no_global_backstop"
                else:
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
        if canonical_direct_primary_mode:
            unresolved_canonical = remaining_gap & river_active_guidance_domain
            if np.any(unresolved_canonical):
                unresolved = int(np.count_nonzero(unresolved_canonical))
                raise RuntimeError(
                    "canonical direct-primary conditioning left unresolved active river pixels; "
                    f"unresolved_canonical_pixels={unresolved} support_note={support_note}"
                )
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

    if canonical_direct_primary_mode:
        channel_core_preservation = {"conditioned": conditioned, "guidance_influence": guidance_influence, "support": support, "receipt": {"preservation_applied_pixels": 0, "bypassed_for_canonical_direct_primary": True}}
    else:
        channel_core_preservation = apply_channel_core_preservation(
            conditioned_surface=conditioned,
            primary_surface=river_primary_surface,
            primary_domain=np.asarray(river_primary_surface_domain_mask, dtype=bool),
            channel_core_preserve=np.asarray(river_primary_surface_channel_core_preserve, dtype=bool),
            authoritative_locked=river_authoritative_locked,
            bank_influence=river_bank_influence,
            measured_anchor_fraction=river_channel_surface_measured_anchor_fraction,
            structure_only_fraction=river_channel_surface_structure_only_fraction,
            prediction_support_confidence=river_channel_surface_prediction_support_confidence,
            guidance_influence=guidance_influence,
            support=support,
        )
        conditioned = channel_core_preservation["conditioned"]
        if channel_core_preservation.get("guidance_influence") is not None:
            guidance_influence = np.asarray(channel_core_preservation["guidance_influence"], dtype=np.float32)
        if channel_core_preservation.get("support") is not None:
            support = np.asarray(channel_core_preservation["support"], dtype=np.uint8)
        if int((channel_core_preservation.get("receipt") or {}).get("preservation_applied_pixels", 0) or 0) > 0:
            support_note = f"{support_note}; channel_core_preservation_applied"

    # Enforce the authoritative/CUDEM background handoff exactly outside active
    # guidance domains.  This prevents the global nearest backstop from inventing
    # broad non-river/non-SDB surfaces where a baseline CUDEM surface already
    # exists and should remain the default terrain.
    enforced = _enforce_authoritative_first_contract(
        conditioned=conditioned,
        auth=auth,
        locked=locked,
        background_surface=background_surface,
        baseline_exact_domain=baseline_exact_domain,
        support=support,
        guidance_influence=guidance_influence,
    )
    conditioned = enforced["conditioned"]
    support = enforced["support"]
    guidance_influence = enforced["guidance_influence"]
    authoritative_first_contract = enforced["contract"]
    if (not canonical_direct_primary_mode) and np.any(river_authoritative_locked):
        conditioned[river_authoritative_locked] = river_authoritative_lock_surface[river_authoritative_locked]
        support[river_authoritative_locked] = int(SupportClass.AUTHORITATIVE_LOCKED)
        guidance_influence[river_authoritative_locked] = 0.0
    if background_surface is not None and np.any(baseline_exact_domain):
        if "baseline_cudem_background" not in support_note:
            support_note = f"{support_note}; enforce_baseline_cudem_background_outside_guidance" if support_note else "enforce_baseline_cudem_background_outside_guidance"

    if np.any(gap & ~np.isfinite(conditioned)):
        unresolved = int(np.count_nonzero(gap & ~np.isfinite(conditioned)))
        raise RuntimeError(f"terrain interpolator failed to produce continuous output; unresolved_gap_pixels={unresolved}")

    finite_guidance_uncertainty = np.isfinite(guidance_uncertainty)
    runtime_authoritative_locked = locked | river_authoritative_locked
    conditioned_uncertainty = combine_conditioning_uncertainty(
        anchor_uncertainty=anchor_uncertainty,
        guidance_uncertainty=guidance_uncertainty,
        guidance_influence=guidance_influence,
        conditioned=conditioned,
        locked=runtime_authoritative_locked,
    )

    eligible = take_any
    provenance = np.zeros_like(support, dtype=np.uint8)
    provenance[runtime_authoritative_locked] = int(ProvenanceClass.AUTHORITATIVE_LOCKED)
    provenance[(support == int(SupportClass.ANCHORED_INTERPOLATION)) & np.isfinite(conditioned)] = int(ProvenanceClass.ANCHORED_INTERPOLATION)
    provenance[(support == int(SupportClass.GUIDANCE_CONDITIONED_SDB)) & np.isfinite(conditioned)] = int(ProvenanceClass.SDB_CONDITIONED_FILL)
    provenance[(support == int(SupportClass.GUIDANCE_CONDITIONED_RIVER)) & np.isfinite(conditioned)] = int(ProvenanceClass.RIVER_CONDITIONED_FILL)
    provenance[(support == int(SupportClass.SCAFFOLD_INFERRED)) & np.isfinite(conditioned)] = int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL)
    provenance[(support == int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)) & np.isfinite(conditioned)] = int(ProvenanceClass.LOW_CONFIDENCE_FILL)

    channel_core_diagnostics = build_channel_core_preservation_diagnostics(
        primary_surface=river_primary_surface,
        conditioned_surface=conditioned,
        primary_domain=np.asarray(river_primary_surface_domain_mask, dtype=bool),
        channel_core_preserve=np.asarray(river_primary_surface_channel_core_preserve, dtype=bool),
        authoritative_locked=river_authoritative_locked,
        bank_influence=river_bank_influence,
        measured_anchor_fraction=river_channel_surface_measured_anchor_fraction,
        structure_only_fraction=river_channel_surface_structure_only_fraction,
        prediction_support_confidence=river_channel_surface_prediction_support_confidence,
    )
    channel_core_diagnostics["receipt"].update(channel_core_preservation.get("receipt") or {})

    regime = regime_array_from_masks(
        build_regime_masks(
            locked=locked,
            sdb_ok=(gap & sdb_ok & ~river_ok),
            river_ok=(gap & river_ok),
            estuary_transition=(gap & estuary_transition),
        )
    ).astype(np.uint8)
    canonical_bank_input_finite_pixels = 0
    canonical_bank_influence_active_pixels = 0
    canonical_bank_shaped_pixels = 0
    if canonical_direct_primary_mode:
        if explicit_bank_elevation_input is not None and explicit_bank_influence_input is not None:
            explicit_bank_active = np.clip(np.nan_to_num(explicit_bank_influence_input, nan=0.0), 0.0, 1.0) > np.float32(0.05)
            log.info(
                "[RIVER][GUIDANCE] Explicit bank handoff into terrain stage: finite_bank_pixels=%d active_bank_influence_pixels=%d primary_domain_finite_bank_pixels=%d",
                int(np.count_nonzero(np.isfinite(explicit_bank_elevation_input))),
                int(np.count_nonzero(explicit_bank_active)),
                int(np.count_nonzero(np.asarray(river_primary_surface_domain_mask, dtype=bool) & np.isfinite(explicit_bank_elevation_input))),
            )
        canonical_bank_domain = np.asarray(river_primary_surface_domain_mask, dtype=bool)
        canonical_bank_take = (
            canonical_bank_domain
            & np.isfinite(river_bank_elevation)
            & (np.clip(np.nan_to_num(river_bank_influence, nan=0.0), 0.0, 1.0) > np.float32(0.05))
        )
        canonical_bank_input_finite_pixels = int(np.count_nonzero(canonical_bank_domain & np.isfinite(river_bank_elevation)))
        canonical_bank_influence_active_pixels = int(np.count_nonzero(canonical_bank_take))
        if bool(getattr(inputs, "require_explicit_bank_guidance", False)):
            if canonical_bank_input_finite_pixels <= 0:
                raise RuntimeError(
                    "canonical direct-primary river path requires explicit bank elevations after handoff, but no finite bank pixels reached terrain conditioning"
                )
            if canonical_bank_influence_active_pixels <= 0:
                raise RuntimeError(
                    "canonical direct-primary river path requires active explicit bank influence after handoff, but no bank-influence pixels were active in terrain conditioning"
                )
        if river_depth_guidance is not None:
            canonical_bank_shaped_pixels = int(np.count_nonzero(
                canonical_bank_domain
                & np.isfinite(river_primary_surface)
                & np.isfinite(river_depth_guidance)
                & (np.abs(river_primary_surface - river_depth_guidance) > np.float32(1.0e-6))
            ))
        log.info(
            "[RIVER][GUIDANCE] Canonical bank shaping active in terrain stage: finite_bank_pixels=%d active_bank_influence_pixels=%d bank_shaped_pixels=%d",
            canonical_bank_input_finite_pixels,
            canonical_bank_influence_active_pixels,
            canonical_bank_shaped_pixels,
        )
    memory_diagnostics.append(memory_checkpoint(
        "terrain_interpolator_end",
        conditioned_finite=int(np.count_nonzero(np.isfinite(conditioned))),
        gap_pixels=int(np.count_nonzero(gap)),
        eligible_pixels=int(np.count_nonzero(eligible)),
        river_guidance_take_domain_pixels=int(np.count_nonzero(river_active_guidance_domain)),
        river_guidance_taken_pixels=int(np.count_nonzero(gap & river_active_guidance_domain & np.isfinite(guidance_surface))),
        background_taken_pixels=int(np.count_nonzero(baseline_exact_domain)),
        authoritative_locked_pixels=int(np.count_nonzero(locked)),
        canonical_unresolved_pixels_before_fallback=int(np.count_nonzero(gap & ~np.isfinite(conditioned))) if canonical_direct_primary_mode else 0,
        canonical_used_global_backstop=bool(False if canonical_direct_primary_mode else ('global_backstop' in support_note)),
        canonical_used_hard_continuity_fallback=bool(False if canonical_direct_primary_mode else ('hard_continuity_fallback' in support_note)),
        canonical_bank_input_finite_pixels=int(canonical_bank_input_finite_pixels),
        canonical_bank_influence_active_pixels=int(canonical_bank_influence_active_pixels),
        canonical_bank_shaped_pixels=int(canonical_bank_shaped_pixels),
    ))
    log.info("[MEMORY][TERRAIN] %s", memory_diagnostics[-1])

    primary_river_guidance_summary = _build_primary_river_guidance_summary(
        primary_surface=river_primary_surface,
        primary_domain=np.asarray(river_primary_surface_domain_mask, dtype=bool),
        primary_contract=river_primary_surface_contract,
        support_note=support_note,
        primary_input_surface_source=primary_river_input_source,
        river_contract_mode=getattr(inputs, 'river_contract_mode', 'canonical_v322'),
        canonical_direct_primary_mode=bool(canonical_direct_primary_mode),
        legacy_structured_take_bypassed=bool(canonical_direct_primary_mode),
        primary_builder_inputs_simplified=bool(canonical_direct_primary_mode),
        canonical_direct_primary_single_path=bool(canonical_direct_primary_mode),
        canonical_direct_primary_weighting_single_path=bool(canonical_direct_primary_mode),
        canonical_direct_primary_downstream_sidepaths_bypassed=bool(canonical_direct_primary_mode),
        canonical_direct_primary_single_write_path=bool(canonical_direct_primary_mode),
        canonical_bank_input_finite_pixels=canonical_bank_input_finite_pixels,
        canonical_bank_influence_active_pixels=canonical_bank_influence_active_pixels,
        canonical_bank_shaped_pixels=canonical_bank_shaped_pixels,
    )

    return {
        "locked": locked,
        "river_authoritative_locked": river_authoritative_locked.astype(np.uint8),
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
        "river_centerline_elevation": river_centerline_elevation.astype(np.float32) if has_river_centerline_elevation else None,
        "river_channel_surface": river_channel_surface.astype(np.float32) if has_river_channel_surface else None,
        "river_channel_surface_confidence": np.clip(np.nan_to_num(river_channel_surface_confidence, nan=0.0), 0.0, 1.0).astype(np.float32) if inputs.river_channel_surface_confidence is not None else None,
        "river_channel_surface_support_count": np.clip(np.nan_to_num(river_channel_surface_support_count, nan=0.0), 0.0, 255.0).astype(np.uint8) if inputs.river_channel_surface_support_count is not None else None,
        "river_channel_surface_prediction_support_confidence": np.clip(np.nan_to_num(river_channel_surface_prediction_support_confidence, nan=0.0), 0.0, 1.0).astype(np.float32) if inputs.river_channel_surface_prediction_support_confidence is not None else None,
        "river_channel_surface_measured_anchor_fraction": np.clip(np.nan_to_num(river_channel_surface_measured_anchor_fraction, nan=0.0), 0.0, 1.0).astype(np.float32) if inputs.river_channel_surface_measured_anchor_fraction is not None else None,
        "river_channel_surface_structure_only_fraction": np.clip(np.nan_to_num(river_channel_surface_structure_only_fraction, nan=0.0), 0.0, 1.0).astype(np.float32) if inputs.river_channel_surface_structure_only_fraction is not None else None,
        "river_channel_surface_low_support_caution": np.asarray(river_channel_surface_low_support_caution > 0.5, dtype=np.uint8) if inputs.river_channel_surface_low_support_caution is not None else None,
        "river_channel_surface_prediction_admissibility": np.asarray(np.nan_to_num(river_channel_surface_prediction_admissibility, nan=0.0) > 0.5, dtype=np.uint8) if inputs.river_channel_surface_prediction_admissibility is not None else None,
        "river_longitudinal_profile_elevation": river_longitudinal_profile_elevation.astype(np.float32) if has_river_longitudinal_profile_elevation else None,
        "river_longitudinal_profile_local_authoritative_reconciliation": river_longitudinal_profile_local_authoritative_reconciliation.astype(np.float32) if inputs.river_longitudinal_profile_local_authoritative_reconciliation is not None else None,
        "river_longitudinal_profile_local_authoritative_reconciliation_influence": river_longitudinal_profile_local_authoritative_reconciliation_influence.astype(np.float32) if inputs.river_longitudinal_profile_local_authoritative_reconciliation_influence is not None else None,
        "river_primary_surface": river_primary_surface.astype(np.float32),
        "river_primary_surface_confidence": river_primary_surface_confidence.astype(np.float32),
        "river_primary_surface_source_class": river_primary_surface_source_class.astype(np.uint8),
        "river_primary_surface_support_count": river_primary_surface_support_count.astype(np.uint8),
        "river_primary_surface_domain": np.asarray(river_primary_surface_domain_mask, dtype=np.uint8),
        "river_primary_surface_terrain_response": river_primary_surface_terrain_response.astype(np.float32),
        "river_primary_surface_cautious_structure": np.asarray(river_primary_surface_cautious_structure, dtype=np.uint8),
        "river_primary_surface_channel_core_preserve": np.asarray(river_primary_surface_channel_core_preserve, dtype=np.uint8),
        "river_channel_core_preservation_zone": np.asarray(channel_core_diagnostics["channel_core_preservation_zone"], dtype=np.uint8),
        "river_channel_core_prepost_delta": np.asarray(channel_core_diagnostics["channel_core_prepost_delta"], dtype=np.float32),
        "river_channel_core_bank_pull_risk": np.asarray(channel_core_diagnostics["channel_core_bank_pull_risk"], dtype=np.float32),
        "river_channel_core_preservation_receipt": channel_core_diagnostics["receipt"],
        "river_primary_surface_contract": river_primary_surface_contract,
        "river_primary_guidance_summary": primary_river_guidance_summary,
        "river_centerline_influence": np.clip(np.nan_to_num(river_centerline_influence, nan=0.0), 0.0, 1.0).astype(np.float32) if has_river_centerline_influence else None,
        "river_xs_support_elevation": river_xs_support_elevation.astype(np.float32) if has_river_xs_support_elevation else None,
        "river_xs_support_weight": np.clip(np.nan_to_num(river_xs_support_weight, nan=0.0), 0.0, 1.0).astype(np.float32) if has_river_xs_support_weight else None,
        "guidance_surface": guidance_surface.astype(np.float32),
        "background_surface": background_surface.astype(np.float32) if background_surface is not None else np.full_like(auth, np.nan, dtype=np.float32),
        "active_guidance_domain": active_guidance_domain.astype(np.uint8),
        "river_guidance_take_domain": river_active_guidance_domain.astype(np.uint8),
        "baseline_exact_domain": baseline_exact_domain.astype(np.uint8),
        "authoritative_first_contract": authoritative_first_contract,
        "guidance_locked_contract": guidance_locked_contract,
        "anchor_uncertainty": anchor_uncertainty.astype(np.float32),
        "guidance_uncertainty": guidance_uncertainty.astype(np.float32),
        "conditioned_uncertainty": conditioned_uncertainty.astype(np.float32),
        "conditioned": conditioned,
        "provenance": provenance,
        "regime": regime,
        "memory_diagnostics": memory_diagnostics,
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
