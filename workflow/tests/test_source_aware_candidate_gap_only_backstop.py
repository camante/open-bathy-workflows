import numpy as np

from authoritative_conditioning import build_source_aware_candidate_arrays


def test_source_aware_candidate_blocks_legacy_inside_river_corridor_outside_estuary():
    legacy = np.array([[11.0]], dtype=np.float32)
    out = build_source_aware_candidate_arrays(
        legacy_candidate=legacy,
        sdb_candidate=None,
        river_candidate=None,
        sdb_ok=np.array([[False]]),
        river_ok=np.array([[True]]),
        sdb_guidance_weight=None,
        sdb_trusted_interior=None,
        river_guidance_weight=None,
        river_trusted_interior=None,
        estuary_transition=np.array([[False]]),
    )
    assert np.isnan(out["candidate"][0, 0])
    assert int(out["stats"]["legacy_blocked_in_river_corridor_pixels"]) == 1
    assert out["backstop_policy"]["legacy_candidate_role"] == "gap_only_backstop"
