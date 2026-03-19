from __future__ import annotations

from typing import Any, Dict
import numpy as np
from support_classes import SupportClass


def _as_array(code_like: Any) -> np.ndarray:
    return np.asarray(code_like)


def mask_is_authoritative_locked(code_like: Any) -> np.ndarray:
    return _as_array(code_like) == int(SupportClass.AUTHORITATIVE_LOCKED)


def mask_is_anchored_interpolation(code_like: Any) -> np.ndarray:
    return _as_array(code_like) == int(SupportClass.ANCHORED_INTERPOLATION)


def mask_allows_sdb_guidance(code_like: Any) -> np.ndarray:
    arr = _as_array(code_like)
    return (arr == int(SupportClass.GUIDANCE_CONDITIONED_SDB)) | (arr == int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL))


def mask_allows_river_guidance(code_like: Any) -> np.ndarray:
    arr = _as_array(code_like)
    return (arr == int(SupportClass.GUIDANCE_CONDITIONED_RIVER)) | (arr == int(SupportClass.SCAFFOLD_INFERRED)) | (arr == int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL))


def mask_requires_low_confidence(code_like: Any) -> np.ndarray:
    return _as_array(code_like) == int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)


def build_support_class_contract() -> Dict[str, Dict[str, Any]]:
    return {
        str(int(SupportClass.AUTHORITATIVE_LOCKED)): {
            'name': 'authoritative_locked',
            'allows_sdb_guidance': False,
            'allows_river_guidance': False,
            'requires_low_confidence': False,
            'description': 'Hard authoritative control; final DEM must preserve authoritative values exactly.'
        },
        str(int(SupportClass.ANCHORED_INTERPOLATION)): {
            'name': 'anchored_interpolation',
            'allows_sdb_guidance': False,
            'allows_river_guidance': False,
            'requires_low_confidence': False,
            'description': 'Continuous interpolation anchored by nearby authoritative control.'
        },
        str(int(SupportClass.GUIDANCE_CONDITIONED_SDB)): {
            'name': 'guidance_conditioned_sdb',
            'allows_sdb_guidance': True,
            'allows_river_guidance': False,
            'requires_low_confidence': False,
            'description': 'Gap fill may be conditioned by SDB guidance products.'
        },
        str(int(SupportClass.GUIDANCE_CONDITIONED_RIVER)): {
            'name': 'guidance_conditioned_river',
            'allows_sdb_guidance': False,
            'allows_river_guidance': True,
            'requires_low_confidence': False,
            'description': 'Gap fill may be conditioned by river guidance products.'
        },
        str(int(SupportClass.SCAFFOLD_INFERRED)): {
            'name': 'scaffold_inferred',
            'allows_sdb_guidance': False,
            'allows_river_guidance': True,
            'requires_low_confidence': False,
            'description': 'Channel scaffold / skeleton guidance is allowed.'
        },
        str(int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)): {
            'name': 'low_confidence_continuous_fill',
            'allows_sdb_guidance': True,
            'allows_river_guidance': True,
            'requires_low_confidence': True,
            'description': 'Fallback continuous fill; output must carry low-confidence semantics.'
        },
    }


__all__ = [
    'mask_is_authoritative_locked',
    'mask_is_anchored_interpolation',
    'mask_allows_sdb_guidance',
    'mask_allows_river_guidance',
    'mask_requires_low_confidence',
    'build_support_class_contract',
]
