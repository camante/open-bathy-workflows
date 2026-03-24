import numpy as np

from bathy_main import build_estuary_aware_sdb_domain


def test_build_estuary_aware_sdb_domain_excludes_fluvial_core_but_keeps_transition_and_estuary_water():
    water = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    river = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )
    estuary_transition = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )

    domain = build_estuary_aware_sdb_domain(
        water_mask=water,
        river_channel=river,
        estuary_transition=estuary_transition,
    )

    expected = np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(domain, expected)
