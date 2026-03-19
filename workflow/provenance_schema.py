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


def provenance_class_name(code: int) -> str:
    return PROVENANCE_CLASS_CODE_TO_NAME[int(code)]


def provenance_class_family(code: int) -> str:
    return PROVENANCE_CLASS_FAMILY[int(code)]


def validate_provenance_class_code(code: int) -> int:
    code = int(code)
    if code not in PROVENANCE_CLASS_CODE_TO_NAME:
        raise ValueError(f"Unknown provenance class code: {code}")
    return code


__all__ = [
    "ProvenanceClass",
    "PROVENANCE_CLASS_CODE_TO_NAME",
    "PROVENANCE_CLASS_NAME_TO_CODE",
    "PROVENANCE_CLASS_FAMILY",
    "provenance_class_name",
    "provenance_class_family",
    "validate_provenance_class_code",
]
