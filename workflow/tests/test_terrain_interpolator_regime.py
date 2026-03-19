import numpy as np

from terrain_interpolator import TerrainInterpolationConfig, TerrainInterpolationInputs, interpolate_support_aware_surface


def test_interpolator_returns_unified_regime_array():
    candidate = np.array([[np.nan, -2.0, -3.0], [5.0, -4.0, -6.0]], dtype=np.float32)
    auth = np.array([[10.0, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    sdb_ok = np.array([[False, True, False], [False, False, False]], dtype=bool)
    river_ok = np.array([[False, False, True], [True, True, False]], dtype=bool)
    estuary = np.array([[False, False, True], [False, False, False]], dtype=bool)

    result = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=sdb_ok,
            river_ok=river_ok,
            estuary_transition=estuary,
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )

    regime = result["regime"]
    assert regime[0, 0] == 1
    assert regime[0, 1] == 2
    assert regime[0, 2] == 3
    assert regime[1, 0] == 4
