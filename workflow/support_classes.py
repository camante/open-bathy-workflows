from enum import IntEnum
from typing import Dict


class SupportClass(IntEnum):
    AUTHORITATIVE_LOCKED = 1
    ANCHORED_INTERPOLATION = 2
    GUIDANCE_CONDITIONED_SDB = 3
    GUIDANCE_CONDITIONED_RIVER = 4
    SCAFFOLD_INFERRED = 5
    LOW_CONFIDENCE_CONTINUOUS_FILL = 6


class RegimeClass(IntEnum):
    UPLAND = 1
    NEARSHORE_WATER = 2
    ESTUARY_TRANSITION = 3
    RIVER_CHANNEL = 4


SUPPORT_CLASS_CODE_TO_NAME: Dict[int, str] = {
    int(SupportClass.AUTHORITATIVE_LOCKED): "authoritative_locked",
    int(SupportClass.ANCHORED_INTERPOLATION): "anchored_interpolation",
    int(SupportClass.GUIDANCE_CONDITIONED_SDB): "guidance_conditioned_sdb",
    int(SupportClass.GUIDANCE_CONDITIONED_RIVER): "guidance_conditioned_river",
    int(SupportClass.SCAFFOLD_INFERRED): "scaffold_inferred",
    int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL): "low_confidence_continuous_fill",
}
SUPPORT_CLASS_NAME_TO_CODE: Dict[str, int] = {v: k for k, v in SUPPORT_CLASS_CODE_TO_NAME.items()}

SUPPORT_CLASS_FAMILY: Dict[int, str] = {
    int(SupportClass.AUTHORITATIVE_LOCKED): "authoritative_locked",
    int(SupportClass.ANCHORED_INTERPOLATION): "anchored_interpolation",
    int(SupportClass.GUIDANCE_CONDITIONED_SDB): "guidance_conditioned",
    int(SupportClass.GUIDANCE_CONDITIONED_RIVER): "guidance_conditioned",
    int(SupportClass.SCAFFOLD_INFERRED): "scaffold_inferred",
    int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL): "low_confidence_continuous_fill",
}

REGIME_CLASS_CODE_TO_NAME: Dict[int, str] = {
    int(RegimeClass.UPLAND): "upland",
    int(RegimeClass.NEARSHORE_WATER): "nearshore_water",
    int(RegimeClass.ESTUARY_TRANSITION): "estuary_transition",
    int(RegimeClass.RIVER_CHANNEL): "river_channel",
}


def support_class_name(code: int) -> str:
    return SUPPORT_CLASS_CODE_TO_NAME[int(code)]


def support_class_code_from_name(name: str) -> int:
    try:
        return SUPPORT_CLASS_NAME_TO_CODE[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown support class name: {name}") from exc


def support_class_label_or_raise(code: int) -> str:
    return support_class_name(validate_support_class_code(code))


def support_class_family(code: int) -> str:
    return SUPPORT_CLASS_FAMILY[int(code)]


def support_class_family_or_raise(code: int) -> str:
    return support_class_family(validate_support_class_code(code))


def regime_class_name(code: int) -> str:
    return REGIME_CLASS_CODE_TO_NAME[int(code)]


def validate_support_class_code(code: int) -> int:
    code = int(code)
    if code not in SUPPORT_CLASS_CODE_TO_NAME:
        raise ValueError(f"Unknown support class code: {code}")
    return code


def class_is_authoritative_locked(code: int) -> bool:
    return validate_support_class_code(code) == int(SupportClass.AUTHORITATIVE_LOCKED)


def class_is_anchored_interpolation(code: int) -> bool:
    return validate_support_class_code(code) == int(SupportClass.ANCHORED_INTERPOLATION)


def support_class_is_guidance_conditioned(code: int) -> bool:
    return support_class_family_or_raise(code) == "guidance_conditioned"


def support_class_is_final_authoritative(code: int) -> bool:
    return class_is_authoritative_locked(code)


def support_class_allows_final_dem_guidance(code: int) -> bool:
    code = validate_support_class_code(code)
    return code in {
        int(SupportClass.GUIDANCE_CONDITIONED_SDB),
        int(SupportClass.GUIDANCE_CONDITIONED_RIVER),
        int(SupportClass.SCAFFOLD_INFERRED),
        int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL),
    }


def class_allows_sdb_guidance(code: int) -> bool:
    code = validate_support_class_code(code)
    return code in {
        int(SupportClass.GUIDANCE_CONDITIONED_SDB),
        int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL),
    }


def class_allows_river_guidance(code: int) -> bool:
    code = validate_support_class_code(code)
    return code in {
        int(SupportClass.GUIDANCE_CONDITIONED_RIVER),
        int(SupportClass.SCAFFOLD_INFERRED),
        int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL),
    }


def class_requires_low_confidence(code: int) -> bool:
    return validate_support_class_code(code) == int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)


def support_schema_summary() -> dict:
    return {
        "codes": {str(k): v for k, v in SUPPORT_CLASS_CODE_TO_NAME.items()},
        "families": {str(k): v for k, v in SUPPORT_CLASS_FAMILY.items()},
    }


def build_regime_masks(*, locked, sdb_ok, river_ok, estuary_transition=None):
    import numpy as np

    locked = np.asarray(locked, dtype=bool)
    sdb_ok = np.asarray(sdb_ok, dtype=bool)
    river_ok = np.asarray(river_ok, dtype=bool)
    est = np.asarray(estuary_transition, dtype=bool) if estuary_transition is not None else np.zeros_like(locked, dtype=bool)
    return {
        "upland": locked,
        "nearshore_water": (~locked) & sdb_ok & (~river_ok),
        "estuary_transition": (~locked) & river_ok & est,
        "river_channel": (~locked) & river_ok & (~est),
    }


def regime_array_from_masks(masks):
    import numpy as np

    shape = next(np.asarray(v).shape for v in masks.values())
    out = np.full(shape, int(RegimeClass.UPLAND), dtype=np.uint8)
    if "nearshore_water" in masks:
        out[np.asarray(masks["nearshore_water"], dtype=bool)] = int(RegimeClass.NEARSHORE_WATER)
    if "river_channel" in masks:
        out[np.asarray(masks["river_channel"], dtype=bool)] = int(RegimeClass.RIVER_CHANNEL)
    if "estuary_transition" in masks:
        out[np.asarray(masks["estuary_transition"], dtype=bool)] = int(RegimeClass.ESTUARY_TRANSITION)
    return out


def assign_support_classes(*, locked, sdb_ok, river_ok, river_scaffold_dominant=None):
    import numpy as np

    locked = np.asarray(locked, dtype=bool)
    sdb_ok = np.asarray(sdb_ok, dtype=bool)
    river_ok = np.asarray(river_ok, dtype=bool)
    scaffold = np.asarray(river_scaffold_dominant, dtype=bool) if river_scaffold_dominant is not None else np.zeros_like(locked, dtype=bool)
    out = np.full(locked.shape, int(SupportClass.ANCHORED_INTERPOLATION), dtype=np.uint8)
    out[locked] = int(SupportClass.AUTHORITATIVE_LOCKED)
    out[(~locked) & sdb_ok & (~river_ok)] = int(SupportClass.GUIDANCE_CONDITIONED_SDB)
    out[(~locked) & river_ok] = int(SupportClass.GUIDANCE_CONDITIONED_RIVER)
    out[(~locked) & scaffold] = int(SupportClass.SCAFFOLD_INFERRED)
    return out


def build_river_guidance_zones(*, admissible, estuary_transition, trusted_interior, authoritative_support):
    import numpy as np

    admissible = np.asarray(admissible, dtype=bool)
    est = np.asarray(estuary_transition, dtype=bool)
    trusted = np.asarray(trusted_interior, dtype=bool)
    support = np.asarray(authoritative_support, dtype=bool)
    fluvial_core = admissible & (~est)
    exact = trusted | support
    return {
        "estuary_transition": admissible & est,
        "fluvial_core": fluvial_core,
        "exact": exact,
    }


__all__ = [
    "SupportClass",
    "RegimeClass",
    "SUPPORT_CLASS_CODE_TO_NAME",
    "SUPPORT_CLASS_NAME_TO_CODE",
    "SUPPORT_CLASS_FAMILY",
    "REGIME_CLASS_CODE_TO_NAME",
    "support_class_name",
    "support_class_code_from_name",
    "support_class_label_or_raise",
    "support_class_family",
    "support_class_family_or_raise",
    "support_class_is_guidance_conditioned",
    "support_class_is_final_authoritative",
    "support_class_allows_final_dem_guidance",
    "support_schema_summary",
    "regime_class_name",
    "validate_support_class_code",
    "class_is_authoritative_locked",
    "class_is_anchored_interpolation",
    "class_allows_sdb_guidance",
    "class_allows_river_guidance",
    "class_requires_low_confidence",
    "build_regime_masks",
    "regime_array_from_masks",
    "assign_support_classes",
    "build_river_guidance_zones",
]
