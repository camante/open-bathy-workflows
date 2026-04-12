from __future__ import annotations

import numpy as np

from support_classes import SupportClass
from channel_core_preservation import compute_channel_core_preserve_mask, apply_channel_core_preservation, build_channel_core_preservation_diagnostics


def test_compute_channel_core_preserve_mask_matches_expected_rule():
    mask = compute_channel_core_preserve_mask(
        prediction_support_confidence=np.array([[0.4, 0.2]], dtype=np.float32),
        measured_anchor_fraction=np.array([[0.1, 0.1]], dtype=np.float32),
        structure_only_fraction=np.array([[0.7, 0.7]], dtype=np.float32),
        low_support_caution=np.array([[0, 1]], dtype=np.uint8),
        prediction_admissibility=np.array([[1, 1]], dtype=np.uint8),
    )
    assert mask.dtype == bool
    assert mask.tolist() == [[True, False]]


def test_build_channel_core_preservation_diagnostics_reports_zone_delta_and_risk():
    primary = np.array([[-2.0, -2.0]], dtype=np.float32)
    conditioned = np.array([[-1.5, -2.0]], dtype=np.float32)
    out = build_channel_core_preservation_diagnostics(
        primary_surface=primary,
        conditioned_surface=conditioned,
        primary_domain=np.array([[1, 1]], dtype=np.uint8),
        channel_core_preserve=np.array([[1, 0]], dtype=np.uint8),
        authoritative_locked=np.array([[0, 0]], dtype=np.uint8),
        bank_influence=np.array([[1.0, 0.0]], dtype=np.float32),
        measured_anchor_fraction=np.array([[0.0, 0.0]], dtype=np.float32),
        structure_only_fraction=np.array([[1.0, 0.0]], dtype=np.float32),
        prediction_support_confidence=np.array([[0.0, 1.0]], dtype=np.float32),
    )
    zone = out["channel_core_preservation_zone"]
    delta = out["channel_core_prepost_delta"]
    risk = out["channel_core_bank_pull_risk"]
    receipt = out["receipt"]
    assert zone.tolist() == [[1, 0]]
    assert np.isclose(delta[0, 0], 0.5)
    assert np.isnan(delta[0, 1])
    assert np.isclose(risk[0, 0], 1.0)
    assert receipt["channel_core_zone_pixels"] == 1
    assert np.isclose(receipt["delta_abs_p95_m"], 0.5)


def test_apply_channel_core_preservation_pulls_conditioned_toward_primary_and_updates_support():
    out = apply_channel_core_preservation(
        conditioned_surface=np.array([[-1.0, -2.0]], dtype=np.float32),
        primary_surface=np.array([[-3.0, -2.0]], dtype=np.float32),
        primary_domain=np.array([[1, 1]], dtype=np.uint8),
        channel_core_preserve=np.array([[1, 0]], dtype=np.uint8),
        authoritative_locked=np.array([[0, 0]], dtype=np.uint8),
        bank_influence=np.array([[1.0, 0.0]], dtype=np.float32),
        measured_anchor_fraction=np.array([[0.0, 0.0]], dtype=np.float32),
        structure_only_fraction=np.array([[0.9, 0.0]], dtype=np.float32),
        prediction_support_confidence=np.array([[0.8, 0.0]], dtype=np.float32),
        guidance_influence=np.array([[0.2, 0.0]], dtype=np.float32),
        support=np.array([[int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL), int(SupportClass.ANCHORED_INTERPOLATION)]], dtype=np.uint8),
    )
    assert out["conditioned"][0, 0] < -2.5
    assert out["guidance_influence"][0, 0] >= 0.72
    assert int(out["support"][0, 0]) == int(SupportClass.SCAFFOLD_INFERRED)
    assert out["receipt"]["preservation_applied_pixels"] == 1


def test_apply_channel_core_preservation_does_not_change_locked_cells():
    out = apply_channel_core_preservation(
        conditioned_surface=np.array([[-1.0]], dtype=np.float32),
        primary_surface=np.array([[-3.0]], dtype=np.float32),
        primary_domain=np.array([[1]], dtype=np.uint8),
        channel_core_preserve=np.array([[1]], dtype=np.uint8),
        authoritative_locked=np.array([[1]], dtype=np.uint8),
        bank_influence=np.array([[1.0]], dtype=np.float32),
        measured_anchor_fraction=np.array([[0.0]], dtype=np.float32),
        structure_only_fraction=np.array([[1.0]], dtype=np.float32),
        prediction_support_confidence=np.array([[1.0]], dtype=np.float32),
        guidance_influence=np.array([[0.2]], dtype=np.float32),
        support=np.array([[int(SupportClass.AUTHORITATIVE_LOCKED)]], dtype=np.uint8),
    )
    assert np.isclose(out["conditioned"][0, 0], -1.0)
    assert np.isclose(out["guidance_influence"][0, 0], 0.2)
    assert int(out["support"][0, 0]) == int(SupportClass.AUTHORITATIVE_LOCKED)
    assert out["receipt"]["preservation_applied_pixels"] == 0
