import numpy as np

from authoritative_conditioning import build_source_aware_candidate_arrays


def test_source_aware_candidate_prefers_direct_domains_and_estuary_handoff_blend():
    legacy = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    sdb = np.array([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    river = np.array([[np.nan, 2.0], [4.0, np.nan]], dtype=np.float32)
    sdb_ok = np.array([[True, False], [True, False]])
    river_ok = np.array([[False, True], [True, False]])

    out = build_source_aware_candidate_arrays(
        legacy_candidate=legacy,
        sdb_candidate=sdb,
        river_candidate=river,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_guidance_weight=np.array([[1.0, 0.0], [0.2, 0.0]], dtype=np.float32),
        sdb_trusted_interior=np.zeros((2, 2), dtype=np.uint8),
        river_guidance_weight=np.array([[0.0, 1.0], [0.8, 0.0]], dtype=np.float32),
        river_trusted_interior=np.zeros((2, 2), dtype=np.uint8),
        estuary_transition=np.array([[0, 0], [1, 0]], dtype=np.uint8),
    )

    cand = out["candidate"]
    prov = out["provenance"]
    np.testing.assert_allclose(cand, np.array([[1.0, 2.0], [3.8, 40.0]], dtype=np.float32))
    assert prov[0, 0] == 1  # sdb direct
    assert prov[0, 1] == 2  # river direct
    assert prov[1, 0] == 3  # weighted blend in estuary transition handoff
    assert prov[1, 1] == 4  # legacy fallback


def test_source_aware_candidate_honors_trusted_interior_in_overlap():
    sdb = np.array([[5.0]], dtype=np.float32)
    river = np.array([[9.0]], dtype=np.float32)

    out = build_source_aware_candidate_arrays(
        legacy_candidate=None,
        sdb_candidate=sdb,
        river_candidate=river,
        sdb_ok=np.array([[True]]),
        river_ok=np.array([[True]]),
        sdb_guidance_weight=np.array([[0.9]], dtype=np.float32),
        sdb_trusted_interior=np.array([[0]], dtype=np.uint8),
        river_guidance_weight=np.array([[0.1]], dtype=np.float32),
        river_trusted_interior=np.array([[1]], dtype=np.uint8),
    )

    assert float(out["candidate"][0, 0]) == 9.0
    assert int(out["provenance"][0, 0]) == 2
