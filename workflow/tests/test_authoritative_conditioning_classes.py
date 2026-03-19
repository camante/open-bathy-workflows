import numpy as np

from authoritative_conditioning import support_weighted_condition_arrays
from support_classes import SupportClass
from provenance_schema import ProvenanceClass


def test_remaining_gap_backstop_uses_low_confidence_class():
    candidate = np.array([[np.nan, np.nan, np.nan], [np.nan, 5.0, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    auth = np.array([[1.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    sdb_ok = np.zeros_like(candidate, dtype=bool)
    river_ok = np.zeros_like(candidate, dtype=bool)

    out = support_weighted_condition_arrays(
        candidate=candidate,
        auth=auth,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_gw=None,
        sdb_ti=None,
        river_gw=None,
        river_ti=None,
        river_support=None,
        river_support_depth=None,
        pixel_size_m=1.0,
        support_decay_m=10.0,
        support_density_radius_m=10.0,
        coastal_sdb_support_transition_m=10.0,
        river_anchor_density_radius_m=10.0,
        river_scaffold_transition_m=10.0,
    )

    assert np.isfinite(out["conditioned"]).all()
    assert int(out["support"][0, 0]) == int(SupportClass.AUTHORITATIVE_LOCKED)
    assert int(out["support"][2, 2]) == int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)
    assert int(out["provenance"][2, 2]) == int(ProvenanceClass.LOW_CONFIDENCE_FILL)
