import numpy as np
from trusted_interior import build_river_trusted_interior, restrict_river_admissibility


def test_trusted_interior_is_edge_inset_not_authoritative_support():
    channel = np.ones((7, 7), dtype=np.uint8)
    estuary = np.zeros((7, 7), dtype=np.uint8)
    trusted = build_river_trusted_interior(channel=channel, estuary_transition=estuary, edge_buffer_px=1)
    assert trusted[3, 3] == 1
    assert trusted[0, 3] == 0
    assert trusted[3, 0] == 0
    assert int(trusted.sum()) == 25


def test_admissibility_is_within_trusted_interior_and_excludes_authoritative_anchors():
    trusted = np.zeros((5, 5), dtype=np.uint8)
    trusted[1:4, 1:4] = 1
    valid_depth = np.ones((5, 5), dtype=bool)
    support = np.zeros((5, 5), dtype=np.uint8)
    support[2, 2] = 1
    admissible = restrict_river_admissibility(
        trusted_interior=trusted,
        valid_depth=valid_depth,
        authoritative_support=support,
    )
    assert admissible[2, 2] == 0
    assert admissible[1, 1] == 1
    assert admissible[0, 0] == 0
