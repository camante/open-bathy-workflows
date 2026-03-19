import numpy as np

from support_classes import SupportClass
from terrain_interpolator import TerrainInterpolationConfig, TerrainInterpolationInputs, interpolate_support_aware_surface


def test_sdb_guidance_is_suppressed_on_authoritative_locked_cells():
    candidate = np.full((7, 7), 5.0, dtype=np.float32)
    auth = np.full((7, 7), np.nan, dtype=np.float32)
    auth[3, 3] = 1.25
    sdb_ok = np.ones((7, 7), dtype=bool)
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=sdb_ok,
            river_ok=np.zeros((7, 7), dtype=bool),
            sdb_gw=np.ones((7, 7), dtype=np.float32),
            sdb_ti=np.ones((7, 7), dtype=np.uint8),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert out["conditioned"][3, 3] == auth[3, 3]
    assert int(out["support"][3, 3]) == int(SupportClass.AUTHORITATIVE_LOCKED)


def test_river_guidance_does_not_apply_outside_river_domain():
    candidate = np.full((9, 9), 8.0, dtype=np.float32)
    auth = np.full((9, 9), np.nan, dtype=np.float32)
    auth[4, 4] = 2.0
    river_ok = np.zeros((9, 9), dtype=bool)
    river_ok[3:6, 3:6] = True
    river_support = np.zeros((9, 9), dtype=np.uint8)
    river_support[4, 3] = 1
    river_support_depth = np.full((9, 9), np.nan, dtype=np.float32)
    river_support_depth[4, 3] = 3.0
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.zeros((9, 9), dtype=bool),
            river_ok=river_ok,
            river_support=river_support,
            river_support_depth=river_support_depth,
            river_gw=np.where(river_ok, 0.9, 0.9).astype(np.float32),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert int(out["support"][0, 0]) != int(SupportClass.GUIDANCE_CONDITIONED_RIVER)
    assert int(out["support"][4, 4]) == int(SupportClass.AUTHORITATIVE_LOCKED)
