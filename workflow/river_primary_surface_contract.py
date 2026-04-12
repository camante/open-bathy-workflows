from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RIVER_PRIMARY_SURFACE_SOURCE_NAMES = {
    0: "none",
    1: "longitudinal_backbone",
    2: "centerline_backbone",
    3: "xs_local_refinement",
    4: "bank_boundary_guidance",
    5: "channel_surface_scaffold",
}

MAX_PRIMARY_SURFACE_SUPPORT_COUNT = 8


def validate_river_primary_surface_contract(*,
    primary_surface: np.ndarray,
    primary_confidence: np.ndarray,
    source_class: np.ndarray,
    support_count: np.ndarray,
    domain: np.ndarray,
    bank_influence: np.ndarray | None = None,
    xs_locality: np.ndarray | None = None,
) -> dict[str, Any]:
    primary_surface = np.asarray(primary_surface, dtype=np.float32)
    primary_confidence = np.asarray(primary_confidence, dtype=np.float32)
    source_class = np.asarray(source_class, dtype=np.uint8)
    support_count = np.asarray(support_count, dtype=np.uint8)
    domain = np.asarray(domain, dtype=bool)
    finite = domain & np.isfinite(primary_surface)
    domain_count = int(np.count_nonzero(domain))
    finite_count = int(np.count_nonzero(finite))
    coverage_fraction = float(finite_count / domain_count) if domain_count else 0.0

    failures: list[str] = []
    outside_finite = int(np.count_nonzero((~domain) & np.isfinite(primary_surface)))
    if outside_finite:
        failures.append('finite_values_outside_domain')

    nonfinite_conf = int(np.count_nonzero(finite & ~np.isfinite(primary_confidence)))
    if nonfinite_conf:
        failures.append('nonfinite_confidence_inside_domain')

    invalid_source = int(np.count_nonzero(finite & (source_class == 0)))
    if invalid_source:
        failures.append('missing_source_class_inside_domain')

    allowed_codes = np.array(sorted(RIVER_PRIMARY_SURFACE_SOURCE_NAMES), dtype=np.uint8)
    invalid_source_code = int(np.count_nonzero(finite & ~np.isin(source_class, allowed_codes)))
    if invalid_source_code:
        failures.append('invalid_source_class_code_inside_domain')

    invalid_support = int(np.count_nonzero(finite & (support_count == 0)))
    if invalid_support:
        failures.append('missing_support_count_inside_domain')

    excessive_support = int(np.count_nonzero(finite & (support_count > np.uint8(MAX_PRIMARY_SURFACE_SUPPORT_COUNT))))
    if excessive_support:
        failures.append('support_count_exceeds_expected_range')

    nonfinite_with_confidence = int(np.count_nonzero(domain & ~np.isfinite(primary_surface) & (np.nan_to_num(primary_confidence, nan=0.0) > 0.0)))
    if nonfinite_with_confidence:
        failures.append('confidence_on_nonfinite_primary_surface')

    nonfinite_with_source = int(np.count_nonzero(domain & ~np.isfinite(primary_surface) & (source_class != 0)))
    if nonfinite_with_source:
        failures.append('source_class_on_nonfinite_primary_surface')

    nonfinite_with_support = int(np.count_nonzero(domain & ~np.isfinite(primary_surface) & (support_count != 0)))
    if nonfinite_with_support:
        failures.append('support_count_on_nonfinite_primary_surface')

    outside_domain_source = int(np.count_nonzero((~domain) & (source_class != 0)))
    if outside_domain_source:
        failures.append('source_class_outside_domain')

    outside_domain_support = int(np.count_nonzero((~domain) & (support_count != 0)))
    if outside_domain_support:
        failures.append('support_count_outside_domain')

    absurd = int(np.count_nonzero(finite & ((primary_surface < -1000.0) | (primary_surface > 1000.0))))
    if absurd:
        failures.append('absurd_primary_surface_values')

    metrics: dict[str, Any] = {
        'domain_pixels': domain_count,
        'finite_pixels': finite_count,
        'coverage_fraction': coverage_fraction,
        'outside_domain_finite_pixels': outside_finite,
        'nonfinite_confidence_pixels': nonfinite_conf,
        'invalid_source_pixels': invalid_source,
        'invalid_source_code_pixels': invalid_source_code,
        'invalid_support_pixels': invalid_support,
        'excessive_support_pixels': excessive_support,
        'outside_domain_source_pixels': outside_domain_source,
        'outside_domain_support_pixels': outside_domain_support,
        'nonfinite_with_confidence_pixels': nonfinite_with_confidence,
        'nonfinite_with_source_pixels': nonfinite_with_source,
        'nonfinite_with_support_pixels': nonfinite_with_support,
        'absurd_value_pixels': absurd,
        'source_class_counts': {RIVER_PRIMARY_SURFACE_SOURCE_NAMES[int(code)]: int(np.count_nonzero(finite & (source_class == int(code)))) for code in sorted(RIVER_PRIMARY_SURFACE_SOURCE_NAMES) if int(code) != 0},
        'support_count_min': int(np.nanmin(support_count[finite])) if finite_count else 0,
        'support_count_max': int(np.nanmax(support_count[finite])) if finite_count else 0,
        'confidence_min': float(np.nanmin(primary_confidence[finite])) if finite_count else 0.0,
        'confidence_max': float(np.nanmax(primary_confidence[finite])) if finite_count else 0.0,
    }

    if bank_influence is not None:
        bank_influence = np.asarray(bank_influence, dtype=np.float32)
        bank_center = int(np.count_nonzero(finite & (source_class == 4) & (np.clip(bank_influence, 0.0, 1.0) < 0.2)))
        metrics['bank_center_dominance_pixels'] = bank_center
        if bank_center:
            failures.append('bank_boundary_guidance_dominates_channel_center')
    if xs_locality is not None:
        xs_locality = np.asarray(xs_locality, dtype=np.float32)
        xs_far = int(np.count_nonzero(finite & (source_class == 3) & (np.clip(xs_locality, 0.0, 1.0) < 0.05)))
        metrics['xs_far_field_dominance_pixels'] = xs_far
        if xs_far:
            failures.append('xs_local_refinement_dominates_far_field')

    backbone_pixels = int(np.count_nonzero(finite & np.isin(source_class, np.array([1, 2], dtype=np.uint8))))
    metrics['backbone_pixels'] = backbone_pixels
    metrics['backbone_fraction'] = float(backbone_pixels / finite_count) if finite_count else 0.0
    return {
        'ok': len(failures) == 0,
        'failures': failures,
        'metrics': metrics,
    }


def write_river_primary_surface_contract(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return path
