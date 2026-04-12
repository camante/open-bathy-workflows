from enum import IntEnum
from typing import Dict


class ProvenanceClass(IntEnum):
    AUTHORITATIVE_LOCKED = 10
    ANCHORED_INTERPOLATION = 20
    SDB_CONDITIONED_FILL = 30
    RIVER_CONDITIONED_FILL = 40
    RIVER_SCAFFOLD_DOMINANT_FILL = 50
    LOW_CONFIDENCE_FILL = 60


PROVENANCE_CLASS_CODE_TO_NAME: Dict[int, str] = {
    int(ProvenanceClass.AUTHORITATIVE_LOCKED): "authoritative_locked",
    int(ProvenanceClass.ANCHORED_INTERPOLATION): "anchored_interpolation",
    int(ProvenanceClass.SDB_CONDITIONED_FILL): "sdb_conditioned_fill",
    int(ProvenanceClass.RIVER_CONDITIONED_FILL): "river_conditioned_fill",
    int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL): "river_scaffold_dominant_fill",
    int(ProvenanceClass.LOW_CONFIDENCE_FILL): "low_confidence_fill",
}
PROVENANCE_CLASS_NAME_TO_CODE: Dict[str, int] = {v: k for k, v in PROVENANCE_CLASS_CODE_TO_NAME.items()}

PROVENANCE_CLASS_FAMILY: Dict[int, str] = {
    int(ProvenanceClass.AUTHORITATIVE_LOCKED): "authoritative_locked",
    int(ProvenanceClass.ANCHORED_INTERPOLATION): "anchored_interpolation",
    int(ProvenanceClass.SDB_CONDITIONED_FILL): "guidance_conditioned",
    int(ProvenanceClass.RIVER_CONDITIONED_FILL): "guidance_conditioned",
    int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL): "scaffold_inferred",
    int(ProvenanceClass.LOW_CONFIDENCE_FILL): "low_confidence_continuous_fill",
}

PROVENANCE_TO_SUPPORT_FAMILY_HINT: Dict[int, str] = {
    int(ProvenanceClass.AUTHORITATIVE_LOCKED): "authoritative_locked",
    int(ProvenanceClass.ANCHORED_INTERPOLATION): "anchored_interpolation",
    int(ProvenanceClass.SDB_CONDITIONED_FILL): "guidance_conditioned",
    int(ProvenanceClass.RIVER_CONDITIONED_FILL): "guidance_conditioned",
    int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL): "scaffold_inferred",
    int(ProvenanceClass.LOW_CONFIDENCE_FILL): "low_confidence_continuous_fill",
}


def provenance_class_name(code: int) -> str:
    return PROVENANCE_CLASS_CODE_TO_NAME[int(code)]


def provenance_class_code_from_name(name: str) -> int:
    try:
        return PROVENANCE_CLASS_NAME_TO_CODE[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown provenance class name: {name}") from exc


def provenance_class_label_or_raise(code: int) -> str:
    return provenance_class_name(validate_provenance_class_code(code))


def provenance_class_family(code: int) -> str:
    return PROVENANCE_CLASS_FAMILY[int(code)]


def provenance_class_family_or_raise(code: int) -> str:
    return provenance_class_family(validate_provenance_class_code(code))


def validate_provenance_class_code(code: int) -> int:
    code = int(code)
    if code not in PROVENANCE_CLASS_CODE_TO_NAME:
        raise ValueError(f"Unknown provenance class code: {code}")
    return code


def provenance_class_is_guidance_conditioned(code: int) -> bool:
    return provenance_class_family_or_raise(code) == "guidance_conditioned"


def provenance_class_is_authoritative_locked(code: int) -> bool:
    return validate_provenance_class_code(code) == int(ProvenanceClass.AUTHORITATIVE_LOCKED)


def provenance_support_family_hint(code: int) -> str:
    code = validate_provenance_class_code(code)
    return PROVENANCE_TO_SUPPORT_FAMILY_HINT[code]


def provenance_schema_summary() -> dict:
    return {
        "codes": {str(k): v for k, v in PROVENANCE_CLASS_CODE_TO_NAME.items()},
        "families": {str(k): v for k, v in PROVENANCE_CLASS_FAMILY.items()},
        "support_family_hints": {str(k): v for k, v in PROVENANCE_TO_SUPPORT_FAMILY_HINT.items()},
    }


__all__ = [
    "ProvenanceClass",
    "PROVENANCE_CLASS_CODE_TO_NAME",
    "PROVENANCE_CLASS_NAME_TO_CODE",
    "PROVENANCE_CLASS_FAMILY",
    "PROVENANCE_TO_SUPPORT_FAMILY_HINT",
    "provenance_class_name",
    "provenance_class_code_from_name",
    "provenance_class_label_or_raise",
    "provenance_class_family",
    "provenance_class_family_or_raise",
    "validate_provenance_class_code",
    "provenance_class_is_guidance_conditioned",
    "provenance_class_is_authoritative_locked",
    "provenance_support_family_hint",
    "provenance_schema_summary",
]
