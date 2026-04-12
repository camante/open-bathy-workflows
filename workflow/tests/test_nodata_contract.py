import numpy as np

from nodata_utils import fill_output_nodata, array_valid_mask, sanitize_for_output
from precedence_audit import summarize_precedence_audit
from support_classes import SupportClass
from provenance_schema import ProvenanceClass


def test_fill_output_nodata_normalizes_alternate_sentinels():
    arr = np.array([[1.0, -999999.0], [-99999.0, np.nan]], dtype=np.float32)
    out = fill_output_nodata(arr, nodata=-9999.0, dtype=np.float32)
    assert out[0, 0] == np.float32(1.0)
    assert out[0, 1] == np.float32(-9999.0)
    assert out[1, 0] == np.float32(-9999.0)
    assert out[1, 1] == np.float32(-9999.0)


def test_array_valid_mask_rejects_common_sentinels():
    arr = np.array([[0.0, -999999.0], [np.nan, -9999.0]], dtype=np.float32)
    valid = array_valid_mask(arr)
    assert bool(valid[0, 0]) is True
    assert bool(valid[0, 1]) is False
    assert bool(valid[1, 0]) is False
    assert bool(valid[1, 1]) is False


def test_precedence_audit_does_not_treat_sentinel_as_filled_gap():
    auth = np.array([[1.0, np.nan]], dtype=np.float32)
    conditioned = np.array([[1.0, -999999.0]], dtype=np.float32)
    support = np.array([[int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)]], dtype=np.uint8)
    provenance = np.array([[int(ProvenanceClass.AUTHORITATIVE_LOCKED), int(ProvenanceClass.LOW_CONFIDENCE_FILL)]], dtype=np.uint8)
    audit = summarize_precedence_audit(auth=auth, conditioned=conditioned, support=support, provenance=provenance)
    assert audit['gap_fill']['conditioned_finite_gap_count'] == 0
    assert audit['gap_fill']['remaining_gap_nodata_count'] == 1
    assert audit['gap_fill']['continuous_fill_achieved'] is False
