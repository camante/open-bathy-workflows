from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from terrain_interpolator import (
    TerrainInterpolationConfig,
    TerrainInterpolationInputs,
    compute_coastal_sdb_support_confidence,
    compute_river_anchor_support_fields,
    compute_support_distance_density_guidance,
    interpolate_support_aware_surface,
)


def support_weighted_condition_arrays(
    *,
    candidate: np.ndarray,
    auth: np.ndarray,
    sdb_ok: np.ndarray,
    river_ok: np.ndarray,
    sdb_gw: Optional[np.ndarray],
    sdb_ti: Optional[np.ndarray],
    river_gw: Optional[np.ndarray],
    river_ti: Optional[np.ndarray],
    river_support: Optional[np.ndarray],
    river_support_depth: Optional[np.ndarray],
    estuary_transition: Optional[np.ndarray] = None,
    river_corridor_mask: Optional[np.ndarray] = None,
    river_bank_influence: Optional[np.ndarray] = None,
    river_bank_elevation: Optional[np.ndarray] = None,
    river_bank_pair_weight: Optional[np.ndarray] = None,
    river_bank_continuity_weight: Optional[np.ndarray] = None,
    river_bank_graph_confidence: Optional[np.ndarray] = None,
    river_bank_confluence_damping: Optional[np.ndarray] = None,
    river_bank_estuary_side_decay: Optional[np.ndarray] = None,
    pixel_size_m: float,
    support_decay_m: float,
    support_density_radius_m: float,
    coastal_sdb_support_transition_m: float,
    river_anchor_density_radius_m: float,
    river_scaffold_transition_m: float,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    if estuary_transition is None:
        estuary_transition = legacy_kwargs.pop("estuary_transition_mask", None)
    legacy_kwargs.pop("estuary_transition", None)
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"support_weighted_condition_arrays() got unexpected keyword argument(s): {unexpected}")
    return interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=sdb_ok,
            river_ok=river_ok,
            sdb_gw=sdb_gw,
            sdb_ti=sdb_ti,
            river_gw=river_gw,
            river_ti=river_ti,
            river_support=river_support,
            river_support_depth=river_support_depth,
            estuary_transition=estuary_transition,
            river_corridor_mask=river_corridor_mask,
            river_bank_influence=river_bank_influence,
            river_bank_elevation=river_bank_elevation,
            river_bank_pair_weight=river_bank_pair_weight,
            river_bank_continuity_weight=river_bank_continuity_weight,
            river_bank_graph_confidence=river_bank_graph_confidence,
            river_bank_confluence_damping=river_bank_confluence_damping,
            river_bank_estuary_side_decay=river_bank_estuary_side_decay,
        ),
        config=TerrainInterpolationConfig(
            pixel_size_m=pixel_size_m,
            support_decay_m=support_decay_m,
            support_density_radius_m=support_density_radius_m,
            coastal_sdb_support_transition_m=coastal_sdb_support_transition_m,
            river_anchor_density_radius_m=river_anchor_density_radius_m,
            river_scaffold_transition_m=river_scaffold_transition_m,
        ),
    )


def build_source_aware_candidate_arrays(
    *,
    legacy_candidate: Optional[np.ndarray],
    sdb_candidate: Optional[np.ndarray],
    river_candidate: Optional[np.ndarray],
    sdb_ok: np.ndarray,
    river_ok: np.ndarray,
    sdb_guidance_weight: Optional[np.ndarray],
    sdb_trusted_interior: Optional[np.ndarray],
    river_guidance_weight: Optional[np.ndarray],
    river_trusted_interior: Optional[np.ndarray],
    estuary_transition: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build a direct support-aware candidate surface before final conditioning.

    This is intentionally *not* the final DEM. It is a guidance-seeded candidate
    surface that prefers admissible river/SDB products directly and only falls
    back to a legacy fused candidate where neither guidance domain contributes.
    """

    def _shape_of(*arrays: Optional[np.ndarray]) -> tuple[int, int]:
        for arr in arrays:
            if arr is not None:
                return np.asarray(arr).shape
        raise ValueError("At least one candidate array is required")

    shp = _shape_of(legacy_candidate, sdb_candidate, river_candidate, sdb_ok, river_ok)

    def _f32(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if arr is None else np.asarray(arr, dtype=np.float32)

    legacy = _f32(legacy_candidate)
    sdb = _f32(sdb_candidate)
    river = _f32(river_candidate)
    sdb_ok = np.asarray(sdb_ok, dtype=bool)
    river_ok = np.asarray(river_ok, dtype=bool)
    estuary = np.asarray(estuary_transition, dtype=bool) if estuary_transition is not None else np.zeros(shp, dtype=bool)

    sdb_gw = np.clip(np.nan_to_num(_f32(sdb_guidance_weight), nan=0.0), 0.0, 1.0).astype(np.float32) if sdb_guidance_weight is not None else np.zeros(shp, dtype=np.float32)
    river_gw = np.clip(np.nan_to_num(_f32(river_guidance_weight), nan=0.0), 0.0, 1.0).astype(np.float32) if river_guidance_weight is not None else np.zeros(shp, dtype=np.float32)
    sdb_ti = np.asarray(sdb_trusted_interior, dtype=bool) if sdb_trusted_interior is not None else np.zeros(shp, dtype=bool)
    river_ti = np.asarray(river_trusted_interior, dtype=bool) if river_trusted_interior is not None else np.zeros(shp, dtype=bool)

    candidate = np.full(shp, np.nan, dtype=np.float32)
    provenance = np.zeros(shp, dtype=np.uint8)

    # River channels should not quietly fall back to generic coastal/SDB products except
    # inside the explicit estuary handoff zone.  This keeps the final engine centered on
    # river-corridor conditioning rather than allowing coastal products to bleed inland.
    sdb_valid = (np.isfinite(sdb) & sdb_ok & (~river_ok | estuary)) if sdb is not None else np.zeros(shp, dtype=bool)
    river_valid = (np.isfinite(river) & river_ok) if river is not None else np.zeros(shp, dtype=bool)
    overlap = sdb_valid & river_valid

    # Provenance codes for the pre-conditioning candidate raster.
    PROV_NODATA = np.uint8(0)
    PROV_SDB = np.uint8(1)
    PROV_RIVER = np.uint8(2)
    PROV_BLEND = np.uint8(3)
    PROV_LEGACY = np.uint8(4)

    sdb_signal = np.maximum(sdb_gw, 0.65 * sdb_ti.astype(np.float32))
    river_signal = np.maximum(river_gw, 0.65 * river_ti.astype(np.float32))

    sdb_only = sdb_valid & ~river_valid
    river_only = river_valid & ~sdb_valid
    if np.any(sdb_only):
        candidate[sdb_only] = sdb[sdb_only]
        provenance[sdb_only] = PROV_SDB
    if np.any(river_only):
        candidate[river_only] = river[river_only]
        provenance[river_only] = PROV_RIVER

    if np.any(overlap):
        river_exact = overlap & river_ti & ~sdb_ti
        sdb_exact = overlap & sdb_ti & ~river_ti
        if np.any(river_exact):
            candidate[river_exact] = river[river_exact]
            provenance[river_exact] = PROV_RIVER
        if np.any(sdb_exact):
            candidate[sdb_exact] = sdb[sdb_exact]
            provenance[sdb_exact] = PROV_SDB

        unresolved = overlap & np.isnan(candidate)
        if np.any(unresolved):
            total = (sdb_signal + river_signal).astype(np.float32)
            weighted = unresolved & (total > 1e-6)
            if np.any(weighted):
                candidate[weighted] = (
                    (sdb_signal[weighted] * sdb[weighted]) + (river_signal[weighted] * river[weighted])
                ) / total[weighted]
                provenance[weighted] = PROV_BLEND

            unresolved = unresolved & np.isnan(candidate)
            if np.any(unresolved):
                choose_river = unresolved & (river_signal > sdb_signal)
                choose_sdb = unresolved & ~choose_river
                if np.any(choose_river):
                    candidate[choose_river] = river[choose_river]
                    provenance[choose_river] = PROV_RIVER
                if np.any(choose_sdb):
                    candidate[choose_sdb] = sdb[choose_sdb]
                    provenance[choose_sdb] = PROV_SDB

    # Use the legacy fused candidate as a backstop only outside the fluvial corridor, plus
    # the conservative estuary handoff where coastal and river logic intentionally overlap.
    legacy_mask = (
        np.isnan(candidate) & np.isfinite(legacy) & (~river_ok | estuary)
    ) if legacy is not None else np.zeros(shp, dtype=bool)
    if np.any(legacy_mask):
        candidate[legacy_mask] = legacy[legacy_mask]
        provenance[legacy_mask] = PROV_LEGACY

    stats = {
        "candidate_pixels": int(np.isfinite(candidate).sum()),
        "sdb_pixels": int((provenance == PROV_SDB).sum()),
        "river_pixels": int((provenance == PROV_RIVER).sum()),
        "blended_pixels": int((provenance == PROV_BLEND).sum()),
        "legacy_fallback_pixels": int((provenance == PROV_LEGACY).sum()),
        "unused_pixels": int((provenance == PROV_NODATA).sum()),
    }
    return {
        "candidate": candidate,
        "provenance": provenance,
        "stats": stats,
        "provenance_codes": {
            "0": "nodata",
            "1": "sdb_direct",
            "2": "river_direct",
            "3": "sdb_river_blend",
            "4": "legacy_fused_fallback",
        },
    }


__all__ = [
    "TerrainInterpolationConfig",
    "TerrainInterpolationInputs",
    "compute_support_distance_density_guidance",
    "compute_river_anchor_support_fields",
    "compute_coastal_sdb_support_confidence",
    "build_source_aware_candidate_arrays",
    "support_weighted_condition_arrays",
]
