import numpy as np

from trusted_interior import (
    build_authoritative_anchor_support,
    build_soft_guidance_domain,
    build_trusted_export_region,
)


def test_trusted_export_anchor_and_soft_guidance_are_distinct_concepts():
    channel = np.ones((5, 5), dtype=np.uint8)
    estuary = np.zeros((5, 5), dtype=np.uint8)
    trusted = build_trusted_export_region(channel=channel, estuary_transition=estuary, edge_buffer_px=1)
    support = np.zeros((5, 5), dtype=np.uint8)
    support[2, 2] = 1
    anchor = build_authoritative_anchor_support(authoritative_support=support, channel=channel)
    soft = build_soft_guidance_domain(
        trusted_export_region=trusted,
        valid_depth=np.ones((5, 5), dtype=bool),
    )
    from trusted_interior import build_river_admissibility
    admissible = build_river_admissibility(soft_guidance_domain=soft, authoritative_anchor_support=anchor)
    assert trusted[2, 2] == 1
    assert anchor[2, 2] == 1
    assert soft[2, 2] == 1
    assert admissible[2, 2] == 0
    assert admissible[1, 1] == 1
    assert soft[0, 0] == 0
