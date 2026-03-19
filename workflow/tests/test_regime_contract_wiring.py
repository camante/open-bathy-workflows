import numpy as np

from support_classes import (
    RegimeClass,
    SupportClass,
    assign_support_classes,
    build_regime_masks,
    build_river_guidance_zones,
    regime_array_from_masks,
)


def test_regime_masks_and_array_follow_shared_contracts() -> None:
    locked = np.array([[1, 0], [0, 0]], dtype=bool)
    sdb_ok = np.array([[0, 1], [1, 0]], dtype=bool)
    river_ok = np.array([[0, 0], [1, 1]], dtype=bool)
    estuary = np.array([[0, 0], [1, 0]], dtype=np.uint8)

    masks = build_regime_masks(locked=locked, sdb_ok=sdb_ok, river_ok=river_ok, estuary_transition=estuary)
    regime = regime_array_from_masks(masks)

    assert regime[0, 0] == int(RegimeClass.UPLAND)
    assert regime[0, 1] == int(RegimeClass.NEARSHORE_WATER)
    assert regime[1, 0] == int(RegimeClass.ESTUARY_TRANSITION)
    assert regime[1, 1] == int(RegimeClass.RIVER_CHANNEL)


def test_shared_support_and_river_guidance_zone_helpers_are_consistent() -> None:
    locked = np.array([[1, 0], [0, 0]], dtype=bool)
    sdb_ok = np.array([[0, 1], [0, 0]], dtype=bool)
    river_ok = np.array([[0, 0], [1, 1]], dtype=bool)
    scaffold = np.array([[0, 0], [1, 0]], dtype=bool)

    support = assign_support_classes(locked=locked, sdb_ok=sdb_ok, river_ok=river_ok, river_scaffold_dominant=scaffold)
    assert support[0, 0] == int(SupportClass.AUTHORITATIVE_LOCKED)
    assert support[0, 1] == int(SupportClass.GUIDANCE_CONDITIONED_SDB)
    assert support[1, 0] == int(SupportClass.SCAFFOLD_INFERRED)
    assert support[1, 1] == int(SupportClass.GUIDANCE_CONDITIONED_RIVER)

    zones = build_river_guidance_zones(
        admissible=np.array([[0, 1], [1, 1]], dtype=bool),
        estuary_transition=np.array([[0, 1], [0, 0]], dtype=bool),
        trusted_interior=np.array([[0, 0], [1, 0]], dtype=bool),
        authoritative_support=np.array([[0, 0], [0, 1]], dtype=bool),
    )
    assert zones["estuary_transition"][0, 1]
    assert zones["fluvial_core"][1, 0]
    assert zones["exact"][1, 0]
    assert zones["exact"][1, 1]
