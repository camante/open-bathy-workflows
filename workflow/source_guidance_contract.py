from __future__ import annotations

from typing import Any, Dict

import numpy as np

from support_classes import (
    RegimeClass,
    REGIME_CLASS_CODE_TO_NAME,
    build_regime_masks,
    build_river_guidance_zones,
    regime_array_from_masks,
)
from trusted_interior import (
    build_authoritative_anchor_support,
    build_river_admissibility,
    build_soft_guidance_domain,
    build_trusted_export_region,
)


def build_sdb_source_contract(*, valid_depth: np.ndarray, guidance_weight: np.ndarray, trusted_interior: np.ndarray | None = None, admissibility: np.ndarray | None = None) -> Dict[str, Any]:
    valid = np.asarray(valid_depth, dtype=bool)
    gw = np.clip(np.nan_to_num(guidance_weight, nan=0.0), 0.0, 1.0).astype(np.float32)
    ti = np.asarray(trusted_interior, dtype=bool) if trusted_interior is not None else (gw > 0) & valid
    adm = np.asarray(admissibility, dtype=bool) if admissibility is not None else valid
    adm &= valid & ti
    gw = np.where(adm, gw, 0.0).astype(np.float32)
    regime = regime_array_from_masks(build_regime_masks(locked=np.zeros_like(valid, dtype=bool), sdb_ok=adm, river_ok=np.zeros_like(valid, dtype=bool), estuary_transition=None))
    return {
        "guidance_weight": gw,
        "trusted_interior": ti.astype(np.uint8),
        "admissibility": adm.astype(np.uint8),
        "regime": regime.astype(np.uint8),
        "regime_summary": summarize_regime_counts(regime, adm),
    }


def build_river_source_contract(*, channel: np.ndarray, valid_depth: np.ndarray, guidance_weight: np.ndarray, authoritative_support: np.ndarray | None = None, estuary_transition: np.ndarray | None = None, edge_buffer_px: int = 0) -> Dict[str, Any]:
    channel = np.asarray(channel) > 0
    valid = np.asarray(valid_depth, dtype=bool)
    est = np.asarray(estuary_transition, dtype=bool) if estuary_transition is not None else np.zeros_like(channel, dtype=bool)
    anchor = build_authoritative_anchor_support(authoritative_support=authoritative_support, channel=channel)
    trusted = build_trusted_export_region(channel=channel, estuary_transition=est, edge_buffer_px=edge_buffer_px)
    soft = build_soft_guidance_domain(trusted_export_region=trusted, valid_depth=valid)
    adm = build_river_admissibility(soft_guidance_domain=soft, authoritative_anchor_support=anchor)
    gw = np.clip(np.nan_to_num(guidance_weight, nan=0.0), 0.0, 1.0).astype(np.float32)
    gw = np.where(adm > 0, gw, 0.0).astype(np.float32)
    regime = regime_array_from_masks(build_regime_masks(locked=np.zeros_like(channel, dtype=bool), sdb_ok=np.zeros_like(channel, dtype=bool), river_ok=channel, estuary_transition=est))
    zones = build_river_guidance_zones(admissible=adm > 0, estuary_transition=est, trusted_interior=trusted > 0, authoritative_support=anchor > 0)
    return {
        "guidance_weight": gw,
        "trusted_interior": trusted.astype(np.uint8),
        "soft_guidance_domain": soft.astype(np.uint8),
        "admissibility": adm.astype(np.uint8),
        "authoritative_anchor_support": anchor.astype(np.uint8),
        "regime": regime.astype(np.uint8),
        "regime_summary": summarize_regime_counts(regime, adm > 0),
        "zones": {k: v.astype(np.uint8) for k, v in zones.items()},
    }


def summarize_regime_counts(regime: np.ndarray, active: np.ndarray | None = None) -> Dict[str, int]:
    regime = np.asarray(regime, dtype=np.uint8)
    domain = np.asarray(active, dtype=bool) if active is not None else np.ones(regime.shape, dtype=bool)
    out: Dict[str, int] = {}
    for code, name in REGIME_CLASS_CODE_TO_NAME.items():
        out[name] = int(np.sum(domain & (regime == int(code))))
    return out


__all__ = [
    "build_sdb_source_contract",
    "build_river_source_contract",
    "summarize_regime_counts",
]
