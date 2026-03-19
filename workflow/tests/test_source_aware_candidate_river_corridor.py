import numpy as np

from authoritative_conditioning import build_source_aware_candidate_arrays


def test_source_aware_candidate_blocks_sdb_direct_in_fluvial_corridor_outside_estuary():
    sdb = np.array([[np.nan, -2.0, np.nan]], dtype=np.float32)
    river = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32)
    legacy = np.array([[np.nan, -9.0, np.nan]], dtype=np.float32)
    river_ok = np.array([[False, True, False]], dtype=bool)
    sdb_ok = np.array([[False, True, False]], dtype=bool)

    out = build_source_aware_candidate_arrays(
        legacy_candidate=legacy,
        sdb_candidate=sdb,
        river_candidate=river,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_guidance_weight=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        sdb_trusted_interior=np.array([[0, 1, 0]], dtype=np.uint8),
        river_guidance_weight=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        river_trusted_interior=np.array([[0, 1, 0]], dtype=np.uint8),
        estuary_transition=np.array([[0, 0, 0]], dtype=np.uint8),
    )

    assert np.isnan(out["candidate"][0, 1])
    assert out["provenance"][0, 1] == 0


def test_source_aware_candidate_allows_sdb_direct_in_estuary_transition():
    sdb = np.array([[np.nan, -2.0, np.nan]], dtype=np.float32)
    river = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32)
    legacy = np.array([[np.nan, -9.0, np.nan]], dtype=np.float32)
    river_ok = np.array([[False, True, False]], dtype=bool)
    sdb_ok = np.array([[False, True, False]], dtype=bool)

    out = build_source_aware_candidate_arrays(
        legacy_candidate=legacy,
        sdb_candidate=sdb,
        river_candidate=river,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_guidance_weight=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        sdb_trusted_interior=np.array([[0, 1, 0]], dtype=np.uint8),
        river_guidance_weight=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        river_trusted_interior=np.array([[0, 1, 0]], dtype=np.uint8),
        estuary_transition=np.array([[0, 1, 0]], dtype=np.uint8),
    )

    assert np.isfinite(out["candidate"][0, 1])
    assert float(out["candidate"][0, 1]) == -2.0
    assert out["provenance"][0, 1] == 1
