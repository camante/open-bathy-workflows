import numpy as np

from terrain_interpolator import TerrainInterpolationConfig, TerrainInterpolationInputs, interpolate_support_aware_surface
from support_classes import SupportClass, RegimeClass


def test_estuary_transition_not_promoted_to_scaffold():
    auth = np.array([[1.0, np.nan],[np.nan, np.nan]], dtype=np.float32)
    cand = np.array([[1.0, -2.0],[-3.0, -4.0]], dtype=np.float32)
    sdb_ok = np.array([[False, False],[False, False]])
    river_ok = np.array([[False, True],[True, True]])
    est = np.array([[False, True],[False, False]])
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=cand, auth=auth, sdb_ok=sdb_ok, river_ok=river_ok,
            river_gw=np.ones((2,2), dtype=np.float32), river_ti=np.zeros((2,2), dtype=np.uint8),
            river_support=np.zeros((2,2), dtype=np.uint8), river_support_depth=None, estuary_transition=est,
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert out["regime"][0,1] == int(RegimeClass.ESTUARY_TRANSITION)
    assert out["support"][0,1] == int(SupportClass.GUIDANCE_CONDITIONED_RIVER)


def test_authoritative_locks_preserved():
    auth = np.array([[5.0, np.nan],[np.nan, np.nan]], dtype=np.float32)
    cand = np.array([[0.0, -2.0],[-3.0, -4.0]], dtype=np.float32)
    sdb_ok = np.array([[False, True],[False, False]])
    river_ok = np.array([[False, False],[True, True]])
    out = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(candidate=cand, auth=auth, sdb_ok=sdb_ok, river_ok=river_ok),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert out["conditioned"][0,0] == 5.0
    assert out["support"][0,0] == int(SupportClass.AUTHORITATIVE_LOCKED)
