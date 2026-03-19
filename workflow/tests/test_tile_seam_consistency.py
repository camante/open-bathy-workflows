import numpy as np

from authoritative_conditioning import support_weighted_condition_arrays


def test_adjacent_tile_overlap_is_consistent_for_internal_support_context():
    full_shape = (24, 24)
    yy, xx = np.mgrid[0:full_shape[0], 0:full_shape[1]]
    candidate = (0.2 * xx + 0.15 * yy).astype(np.float32)
    auth = np.full(full_shape, np.nan, dtype=np.float32)
    auth[8:16, 11] = 2.0
    auth[8:16, 12] = 2.2
    sdb_ok = np.zeros(full_shape, dtype=bool)
    sdb_ok[7:17, 9:15] = True
    river_ok = np.zeros(full_shape, dtype=bool)

    left = np.s_[:, :18]
    right = np.s_[:, 6:]
    left_out = support_weighted_condition_arrays(
        candidate=candidate[left], auth=auth[left], sdb_ok=sdb_ok[left], river_ok=river_ok[left],
        sdb_gw=np.where(sdb_ok[left], 0.8, 0.0).astype(np.float32), sdb_ti=sdb_ok[left].astype(np.uint8),
        river_gw=None, river_ti=None, river_support=None, river_support_depth=None,
        pixel_size_m=10.0, support_decay_m=300.0, support_density_radius_m=60.0,
        coastal_sdb_support_transition_m=200.0, river_anchor_density_radius_m=200.0, river_scaffold_transition_m=800.0,
    )
    right_out = support_weighted_condition_arrays(
        candidate=candidate[right], auth=auth[right], sdb_ok=sdb_ok[right], river_ok=river_ok[right],
        sdb_gw=np.where(sdb_ok[right], 0.8, 0.0).astype(np.float32), sdb_ti=sdb_ok[right].astype(np.uint8),
        river_gw=None, river_ti=None, river_support=None, river_support_depth=None,
        pixel_size_m=10.0, support_decay_m=300.0, support_density_radius_m=60.0,
        coastal_sdb_support_transition_m=200.0, river_anchor_density_radius_m=200.0, river_scaffold_transition_m=800.0,
    )
    overlap_left = np.s_[:, 6:18]
    overlap_right = np.s_[:, :12]
    np.testing.assert_allclose(left_out["conditioned"][overlap_left], right_out["conditioned"][overlap_right], atol=1e-6)
    np.testing.assert_array_equal(left_out["support"][overlap_left], right_out["support"][overlap_right])
