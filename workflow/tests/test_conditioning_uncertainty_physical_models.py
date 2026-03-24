import numpy as np

from terrain_interpolator import TerrainInterpolationConfig, TerrainInterpolationInputs, interpolate_support_aware_surface



def test_synthesized_sdb_uncertainty_tracks_optical_quality():
    auth = np.array([[10.0, np.nan]], dtype=np.float32)
    sdb_ok = np.array([[False, True]], dtype=bool)
    river_ok = np.array([[False, False]], dtype=bool)
    cfg = TerrainInterpolationConfig(pixel_size_m=10.0, use_inverse_variance_blend_when_available=True)

    high_quality = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            sdb_ok=sdb_ok,
            river_ok=river_ok,
            sdb_depth_guidance=np.array([[np.nan, 20.0]], dtype=np.float32),
            sdb_gw=np.array([[0.0, 0.95]], dtype=np.float32),
        ),
        config=cfg,
    )
    low_quality = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            auth=auth,
            sdb_ok=sdb_ok,
            river_ok=river_ok,
            sdb_depth_guidance=np.array([[np.nan, 20.0]], dtype=np.float32),
            sdb_gw=np.array([[0.0, 0.20]], dtype=np.float32),
        ),
        config=cfg,
    )

    assert np.isfinite(high_quality["guidance_uncertainty"][0, 1])
    assert np.isfinite(low_quality["guidance_uncertainty"][0, 1])
    assert low_quality["guidance_uncertainty"][0, 1] > high_quality["guidance_uncertainty"][0, 1]



def test_synthesized_river_uncertainty_increases_with_structural_disagreement():
    auth = np.array([
        [5.0, 5.0, 5.0],
        [5.0, np.nan, 5.0],
        [5.0, 5.0, 5.0],
    ], dtype=np.float32)
    river_ok = np.array([
        [False, False, False],
        [False, True, False],
        [False, False, False],
    ], dtype=bool)
    base_kwargs = dict(
        auth=auth,
        sdb_ok=np.zeros_like(river_ok, dtype=bool),
        river_ok=river_ok,
        river_depth_guidance=np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, -4.0, np.nan],
            [np.nan, np.nan, np.nan],
        ], dtype=np.float32),
        river_gw=np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32),
        river_corridor_mask=river_ok.astype(np.uint8),
        river_centerline_elevation=np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, -4.0, np.nan],
            [np.nan, np.nan, np.nan],
        ], dtype=np.float32),
        river_centerline_influence=np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32),
    )
    cfg = TerrainInterpolationConfig(pixel_size_m=10.0, use_inverse_variance_blend_when_available=True)

    low_disagreement = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            **base_kwargs,
            river_xs_support_elevation=np.array([
                [np.nan, np.nan, np.nan],
                [np.nan, -4.1, np.nan],
                [np.nan, np.nan, np.nan],
            ], dtype=np.float32),
            river_xs_support_weight=np.array([
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32),
        ),
        config=cfg,
    )
    high_disagreement = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            **base_kwargs,
            river_xs_support_elevation=np.array([
                [np.nan, np.nan, np.nan],
                [np.nan, -8.0, np.nan],
                [np.nan, np.nan, np.nan],
            ], dtype=np.float32),
            river_xs_support_weight=np.array([
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32),
        ),
        config=cfg,
    )

    assert np.isfinite(low_disagreement["guidance_uncertainty"][1, 1])
    assert np.isfinite(high_disagreement["guidance_uncertainty"][1, 1])
    assert high_disagreement["guidance_uncertainty"][1, 1] > low_disagreement["guidance_uncertainty"][1, 1]
