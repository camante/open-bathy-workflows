from __future__ import annotations

import numpy as np

from support_classes import build_regime_masks, regime_array_from_masks, RegimeClass


def test_regime_helpers_are_importable_and_priority_ordered() -> None:
    auth = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    sdb = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    river = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    estuary = np.array([[0, 1], [0, 0]], dtype=np.uint8)

    masks = build_regime_masks(auth, sdb, river, estuary)
    regime = regime_array_from_masks(masks)

    assert int(regime[0, 1]) == int(RegimeClass.RIVER_CHANNEL)
    assert int(regime[1, 0]) == int(RegimeClass.RIVER_CHANNEL)
    assert int(regime[1, 1]) == int(RegimeClass.UNKNOWN)
