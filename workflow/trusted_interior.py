"""Mask construction helpers for authoritative/trusted river interiors."""

from __future__ import annotations

from typing import Any


def _bool_array(value: Any):
    import numpy as np
    return np.asarray(value).astype(bool)


def build_authoritative_anchor_support(channel: Any, authoritative_support: Any, valid_depth: Any | None = None) -> Any:
    import numpy as np
    out = _bool_array(channel) & _bool_array(authoritative_support)
    if valid_depth is not None:
        out &= _bool_array(valid_depth)
    return out.astype("uint8")


def build_soft_guidance_domain(channel: Any, authoritative_anchor_support: Any) -> Any:
    return (_bool_array(channel) & ~_bool_array(authoritative_anchor_support)).astype("uint8")


def build_river_admissibility(channel: Any, *masks: Any) -> Any:
    out = _bool_array(channel)
    for mask in masks:
        if mask is not None:
            out &= _bool_array(mask)
    return out.astype("uint8")


def build_trusted_export_region(export_mask: Any, trusted_interior: Any | None = None) -> Any:
    out = _bool_array(export_mask)
    if trusted_interior is not None:
        out &= _bool_array(trusted_interior)
    return out.astype("uint8")


def build_river_trusted_interior(channel: Any, authoritative_support: Any, valid_depth: Any | None = None) -> Any:
    return build_authoritative_anchor_support(channel, authoritative_support, valid_depth)


__all__ = [
    "build_authoritative_anchor_support",
    "build_soft_guidance_domain",
    "build_river_admissibility",
    "build_trusted_export_region",
    "build_river_trusted_interior",
]

def restrict_river_admissibility(admissibility: Any, trusted_export_region: Any | None = None) -> Any:
    out = _bool_array(admissibility)
    if trusted_export_region is not None:
        out &= _bool_array(trusted_export_region)
    return out.astype("uint8")


def summarize_trusted_export_region(region: Any) -> dict[str, int]:
    import numpy as np
    arr = _bool_array(region)
    return {"trusted_export_pixels": int(np.count_nonzero(arr)), "total_pixels": int(arr.size)}

__all__.extend(["restrict_river_admissibility", "summarize_trusted_export_region"])
