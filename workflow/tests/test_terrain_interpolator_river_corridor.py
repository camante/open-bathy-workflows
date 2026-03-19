import numpy as np

from terrain_interpolator import TerrainInterpolationConfig, TerrainInterpolationInputs, interpolate_support_aware_surface
from support_classes import SupportClass


def test_river_corridor_prefers_river_support_anchor_over_bank_nearest_auth():
    candidate = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 2.0, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float32)
    auth = np.array([
        [10.0, 10.0, 10.0],
        [10.0, np.nan, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float32)
    river_ok = np.zeros((3, 3), dtype=bool)
    river_ok[1, 1] = True
    river_support = np.zeros((3, 3), dtype=np.uint8)
    river_support[1, 1] = 1
    river_support_depth = np.full((3, 3), np.nan, dtype=np.float32)
    river_support_depth[1, 1] = 1.0

    result = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_ok=river_ok,
            river_support=river_support,
            river_support_depth=river_support_depth,
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )

    center = float(result["conditioned"][1, 1])
    assert 1.0 < center < 3.0
    assert abs(center - 1.0) < abs(center - 10.0)
    assert result["support"][1, 1] == int(SupportClass.GUIDANCE_CONDITIONED_RIVER)


def test_estuary_transition_defaults_without_runtime_name_error():
    candidate = np.array([[np.nan, -2.0], [-3.0, -4.0]], dtype=np.float32)
    auth = np.array([[5.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    result = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.array([[False, True], [False, False]], dtype=bool),
            river_ok=np.array([[False, False], [True, False]], dtype=bool),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert result["regime"].shape == candidate.shape
