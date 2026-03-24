from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from conditioning_audit import summarize_conditioning_audit, write_conditioning_audit
from provenance_schema import ProvenanceClass
from support_classes import SupportClass


def test_conditioning_audit_reports_locked_and_uncertainty(tmp_path: Path):
    auth = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)
    conditioned = np.array([[1.0, -3.0], [2.0, -4.0]], dtype=np.float32)
    support = np.array([
        [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
        [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_RIVER)],
    ], dtype=np.uint8)
    provenance = np.array([
        [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.SDB_CONDITIONED_FILL)],
        [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.RIVER_CONDITIONED_FILL)],
    ], dtype=np.uint8)
    guidance = np.array([[0.0, 0.8], [0.0, 0.6]], dtype=np.float32)
    sigma = np.array([[0.1, 1.5], [0.1, 2.0]], dtype=np.float32)

    audit = summarize_conditioning_audit(
        auth=auth,
        conditioned=conditioned,
        support=support,
        provenance=provenance,
        guidance_influence=guidance,
        conditioned_uncertainty=sigma,
    )
    assert audit['authoritative_lock']['lock_preserved'] is True
    assert audit['gap_fill']['continuous_fill_achieved'] is True
    assert audit['guidance_impact']['guided_gap_fraction'] == 1.0
    assert audit['uncertainty_summary']['guided']['count'] == 2
    assert audit['support_classes']['by_family']['guidance_conditioned']['count'] == 2

    out = write_conditioning_audit(tmp_path / 'conditioning_audit.json', auth=auth, conditioned=conditioned, support=support, provenance=provenance, guidance_influence=guidance, conditioned_uncertainty=sigma)
    payload = json.loads(Path(out).read_text(encoding='utf-8'))
    assert payload['uncertainty_summary']['overall']['count'] == 4
