"""Canonical support/regime class schema for final DEM products."""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class SupportClass(IntEnum):
    UNSUPPORTED = 0
    AUTHORITATIVE_LOCKED = 1
    ANCHORED_INTERPOLATION = 2
    GUIDANCE_CONDITIONED_SDB = 3
    GUIDANCE_CONDITIONED_RIVER = 4
    SCAFFOLD_INFERRED = 5
    LOW_CONFIDENCE_CONTINUOUS_FILL = 6


class RegimeClass(IntEnum):
    UNKNOWN = 0
    LAND = 1
    NEARSHORE_WATER = 2
    ESTUARY_TRANSITION = 3
    RIVER_CHANNEL = 4
    OFFSHORE_WATER = 5


SUPPORT_CLASS_CODE_TO_NAME = {
    int(SupportClass.UNSUPPORTED): "unsupported",
    int(SupportClass.AUTHORITATIVE_LOCKED): "authoritative_locked",
    int(SupportClass.ANCHORED_INTERPOLATION): "anchored_interpolation",
    int(SupportClass.GUIDANCE_CONDITIONED_SDB): "guidance_conditioned_sdb",
    int(SupportClass.GUIDANCE_CONDITIONED_RIVER): "guidance_conditioned_river",
    int(SupportClass.SCAFFOLD_INFERRED): "scaffold_inferred",
    int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL): "low_confidence_continuous_fill",
}

REGIME_CLASS_CODE_TO_NAME = {
    int(RegimeClass.UNKNOWN): "unknown",
    int(RegimeClass.LAND): "land",
    int(RegimeClass.NEARSHORE_WATER): "nearshore_water",
    int(RegimeClass.ESTUARY_TRANSITION): "estuary_transition",
    int(RegimeClass.RIVER_CHANNEL): "river_channel",
    int(RegimeClass.OFFSHORE_WATER): "offshore_water",
}

REGIME_CLASS_NAME_TO_CODE = {name: int(code) for code, name in REGIME_CLASS_CODE_TO_NAME.items()}

SUPPORT_CLASS_FAMILY = {
    int(SupportClass.UNSUPPORTED): "unsupported",
    int(SupportClass.AUTHORITATIVE_LOCKED): "authoritative",
    int(SupportClass.ANCHORED_INTERPOLATION): "interpolation",
    int(SupportClass.GUIDANCE_CONDITIONED_SDB): "guidance",
    int(SupportClass.GUIDANCE_CONDITIONED_RIVER): "guidance",
    int(SupportClass.SCAFFOLD_INFERRED): "scaffold",
    int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL): "low_confidence",
}


def support_class_code_from_name(name: str) -> int:
    lookup = {v: k for k, v in SUPPORT_CLASS_CODE_TO_NAME.items()}
    key = str(name).strip().lower()
    if key not in lookup:
        raise KeyError(f"unknown_support_class_name:{name}")
    return int(lookup[key])


def regime_class_code_from_name(name: str) -> int:
    key = str(name).strip().lower()
    if key not in REGIME_CLASS_NAME_TO_CODE:
        raise KeyError(f"unknown_regime_class_name:{name}")
    return int(REGIME_CLASS_NAME_TO_CODE[key])


def class_is_authoritative_locked(code: int) -> bool:
    return int(code) == int(SupportClass.AUTHORITATIVE_LOCKED)


def _as_bool_mask(value: Any, *, shape: tuple[int, ...] | None = None):
    """Return a boolean numpy mask without making numpy a hard import until needed."""
    import numpy as np

    if value is None:
        if shape is None:
            raise ValueError("mask shape is required when value is None")
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value)
    if arr.dtype == bool:
        out = arr
    else:
        out = np.isfinite(arr) & (arr != 0)
    if shape is not None and tuple(out.shape) != tuple(shape):
        raise ValueError(f"mask shape mismatch: got={out.shape} expected={shape}")
    return out.astype(bool, copy=False)


def build_regime_masks(
    authoritative_mask: Any = None,
    sdb_ok: Any = None,
    river_ok: Any = None,
    estuary_transition: Any = None,
    *,
    land_mask: Any = None,
    offshore_water: Any = None,
    shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build canonical boolean regime masks from workflow masks.

    This small deterministic helper centralizes the class schema expected by
    terrain/final-route code without discovering data or changing science.  It
    preserves compatibility with older calls of the form
    ``build_regime_masks(auth, sdb_ok, river_ok, estuary_transition)``.
    """
    import numpy as np

    for candidate in (authoritative_mask, sdb_ok, river_ok, estuary_transition, land_mask, offshore_water):
        if candidate is not None:
            shape = tuple(np.asarray(candidate).shape)
            break
    if shape is None:
        raise ValueError("build_regime_masks requires at least one mask or explicit shape")

    auth = _as_bool_mask(authoritative_mask, shape=shape) if authoritative_mask is not None else np.zeros(shape, dtype=bool)
    sdb = _as_bool_mask(sdb_ok, shape=shape) if sdb_ok is not None else np.zeros(shape, dtype=bool)
    river = _as_bool_mask(river_ok, shape=shape) if river_ok is not None else np.zeros(shape, dtype=bool)
    estuary = _as_bool_mask(estuary_transition, shape=shape) if estuary_transition is not None else np.zeros(shape, dtype=bool)
    land = _as_bool_mask(land_mask, shape=shape) if land_mask is not None else np.zeros(shape, dtype=bool)
    offshore = _as_bool_mask(offshore_water, shape=shape) if offshore_water is not None else np.zeros(shape, dtype=bool)

    return {
        "authoritative": auth,
        "river_channel": river,
        "estuary_transition": estuary & ~river,
        "nearshore_water": sdb & ~river & ~estuary,
        "offshore_water": offshore & ~river & ~estuary & ~sdb,
        "land": land & ~river & ~estuary & ~sdb & ~offshore,
        "unknown": ~(river | estuary | sdb | offshore | land | auth),
    }


def regime_array_from_masks(masks: dict[str, Any], *, dtype: Any = None):
    """Convert regime masks into a canonical integer class array."""
    import numpy as np

    if not isinstance(masks, dict) or not masks:
        raise ValueError("regime_array_from_masks requires a non-empty mask dictionary")
    first = next(iter(masks.values()))
    shape = tuple(np.asarray(first).shape)
    out_dtype = dtype if dtype is not None else np.uint8
    out = np.full(shape, int(RegimeClass.UNKNOWN), dtype=out_dtype)

    for name, code in (
        ("land", RegimeClass.LAND),
        ("offshore_water", RegimeClass.OFFSHORE_WATER),
        ("nearshore_water", RegimeClass.NEARSHORE_WATER),
        ("estuary_transition", RegimeClass.ESTUARY_TRANSITION),
        ("river_channel", RegimeClass.RIVER_CHANNEL),
    ):
        mask_value = masks.get(name)
        if mask_value is None:
            continue
        out[_as_bool_mask(mask_value, shape=shape)] = int(code)
    return out


def support_schema_summary() -> dict[str, object]:
    return {
        "codes": {str(k): v for k, v in SUPPORT_CLASS_CODE_TO_NAME.items()},
        "families": {str(k): v for k, v in SUPPORT_CLASS_FAMILY.items()},
        "regime_codes": {str(k): v for k, v in REGIME_CLASS_CODE_TO_NAME.items()},
    }


__all__ = [
    "REGIME_CLASS_CODE_TO_NAME",
    "REGIME_CLASS_NAME_TO_CODE",
    "SUPPORT_CLASS_CODE_TO_NAME",
    "SUPPORT_CLASS_FAMILY",
    "RegimeClass",
    "SupportClass",
    "build_regime_masks",
    "class_is_authoritative_locked",
    "regime_array_from_masks",
    "regime_class_code_from_name",
    "support_class_code_from_name",
    "support_schema_summary",
]
