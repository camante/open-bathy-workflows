"""Deterministic source/guidance domain contract helpers."""

from __future__ import annotations

from typing import Any


def _np():
    import numpy as np
    return np


def build_river_source_contract(
    *,
    channel: Any,
    valid_depth: Any,
    guidance_weight: Any,
    authoritative_support: Any,
    estuary_transition: Any | None = None,
    edge_buffer_px: int = 0,
) -> dict[str, Any]:
    np = _np()
    channel_b = np.asarray(channel).astype(bool)
    valid_b = np.asarray(valid_depth).astype(bool)
    auth_b = np.asarray(authoritative_support).astype(bool)
    trusted = channel_b & valid_b & auth_b
    admissible = channel_b.copy()
    soft = channel_b & ~trusted
    regime = np.zeros(channel_b.shape, dtype="uint8")
    regime[channel_b] = 4  # RegimeClass.RIVER_CHANNEL
    if estuary_transition is not None:
        estuary = np.asarray(estuary_transition).astype(bool)
        regime[estuary & channel_b] = 3  # RegimeClass.ESTUARY_TRANSITION
    weight = np.asarray(guidance_weight).astype("float32", copy=True)
    weight[~soft] = 0.0
    return {
        "authoritative_anchor_support": trusted.astype("uint8"),
        "trusted_interior": trusted.astype("uint8"),
        "soft_guidance_domain": soft.astype("uint8"),
        "admissibility": admissible.astype("uint8"),
        "regime": regime,
        "guidance_weight": weight,
        "edge_buffer_px": int(edge_buffer_px),
    }


def build_sdb_source_contract(**kwargs: Any) -> dict[str, Any]:
    # Same contract keys as river, with nearshore/estuary regime left to caller
    # when a full SDB domain is available.
    return dict(kwargs)


def summarize_regime_counts(regime: Any) -> dict[str, int]:
    np = _np()
    arr = np.asarray(regime)
    return {str(int(v)): int(np.count_nonzero(arr == v)) for v in np.unique(arr)}


__all__ = [
    "build_river_source_contract",
    "build_sdb_source_contract",
    "summarize_regime_counts",
]
