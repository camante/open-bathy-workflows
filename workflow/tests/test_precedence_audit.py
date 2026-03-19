import json
from pathlib import Path

import numpy as np

from precedence_audit import summarize_precedence_audit, write_precedence_audit
from support_classes import SupportClass
from provenance_schema import ProvenanceClass


def test_precedence_audit_preserves_locked_cells(tmp_path: Path):
    auth = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)
    conditioned = np.array([[1.0, -3.0], [2.0, -4.0]], dtype=np.float32)
    support = np.array([
        [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
        [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)],
    ], dtype=np.uint8)
    prov = np.array([
        [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.SDB_CONDITIONED_FILL)],
        [int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.LOW_CONFIDENCE_FILL)],
    ], dtype=np.uint8)
    guidance = np.array([[0.0, 0.8], [0.0, 0.2]], dtype=np.float32)

    audit = summarize_precedence_audit(
        auth=auth,
        conditioned=conditioned,
        support=support,
        provenance=prov,
        guidance_influence=guidance,
    )
    assert audit["authoritative_lock"]["lock_preserved"] is True
    assert audit["authoritative_lock"]["guidance_nonzero_on_locked_count"] == 0
    assert audit["gap_fill"]["continuous_fill_achieved"] is True
    assert audit["support_classes"]["by_family"]["guidance_conditioned"] == 1

    out = write_precedence_audit(tmp_path / "audit.json", audit, extra={"x": 1})
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["x"] == 1
