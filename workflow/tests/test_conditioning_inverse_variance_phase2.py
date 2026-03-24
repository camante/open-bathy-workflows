import numpy as np

from terrain_interpolator import TerrainInterpolationConfig, TerrainInterpolationInputs, interpolate_support_aware_surface


def _base_inputs(anchor_sigma=None, guidance_sigma=None):
    auth = np.array([[10.0, np.nan]], dtype=np.float32)
    sdb_ok = np.array([[False, True]], dtype=bool)
    river_ok = np.array([[False, False]], dtype=bool)
    sdb_depth_guidance = np.array([[np.nan, 20.0]], dtype=np.float32)
    return TerrainInterpolationInputs(
        auth=auth,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_depth_guidance=sdb_depth_guidance,
        sdb_uncertainty=anchor_sigma if False else guidance_sigma,
        river_uncertainty=None,
    )


def test_inverse_variance_blending_prefers_lower_uncertainty_guidance():
    auth = np.array([[10.0, np.nan]], dtype=np.float32)
    sdb_ok = np.array([[False, True]], dtype=bool)
    river_ok = np.array([[False, False]], dtype=bool)
    inputs = TerrainInterpolationInputs(
        auth=auth,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_depth_guidance=np.array([[np.nan, 20.0]], dtype=np.float32),
        sdb_uncertainty=np.array([[np.nan, 0.01]], dtype=np.float32),
    )
    cfg = TerrainInterpolationConfig(pixel_size_m=10.0, use_inverse_variance_blend_when_available=True)
    out = interpolate_support_aware_surface(inputs=inputs, config=cfg)
    # anchor uncertainty at the gap pixel should be > guidance uncertainty, so blend should strongly favor guidance
    val = out['conditioned'][0, 1]
    assert np.isfinite(val)
    assert val > 19.5, val
    assert out['guidance_influence'][0, 1] > 0.95
    assert 'inverse_variance' in out['support_note']


def test_synthesized_sdb_uncertainty_enables_inverse_variance_when_explicit_uncertainty_missing():
    auth = np.array([[10.0, np.nan]], dtype=np.float32)
    sdb_ok = np.array([[False, True]], dtype=bool)
    river_ok = np.array([[False, False]], dtype=bool)
    inputs = TerrainInterpolationInputs(
        auth=auth,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_depth_guidance=np.array([[np.nan, 20.0]], dtype=np.float32),
        sdb_gw=np.array([[0.0, 0.95]], dtype=np.float32),
        sdb_uncertainty=None,
    )
    cfg = TerrainInterpolationConfig(pixel_size_m=10.0, use_inverse_variance_blend_when_available=True)
    out = interpolate_support_aware_surface(inputs=inputs, config=cfg)
    assert np.isfinite(out['conditioned'][0, 1])
    assert np.isfinite(out['guidance_uncertainty'][0, 1])
    assert 'inverse_variance' in out['support_note']


def test_inverse_variance_zero_zero_pair_preserves_heuristic_and_reports_fallback():
    auth = np.array([[10.0, np.nan]], dtype=np.float32)
    sdb_ok = np.array([[False, True]], dtype=bool)
    river_ok = np.array([[False, False]], dtype=bool)
    inputs = TerrainInterpolationInputs(
        auth=auth,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_depth_guidance=np.array([[np.nan, 20.0]], dtype=np.float32),
        sdb_uncertainty=np.array([[np.nan, 0.0]], dtype=np.float32),
    )
    cfg = TerrainInterpolationConfig(
        pixel_size_m=10.0,
        use_inverse_variance_blend_when_available=True,
        anchor_uncertainty_floor_m=0.0,
        anchor_uncertainty_growth_per_m=0.0,
    )
    out = interpolate_support_aware_surface(inputs=inputs, config=cfg)
    assert np.isfinite(out['conditioned'][0, 1])
    assert 'heuristic_fallback' in out['support_note']


def test_synthesized_river_uncertainty_enables_inverse_variance_when_explicit_uncertainty_missing():
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
    inputs = TerrainInterpolationInputs(
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
        river_uncertainty=None,
    )
    cfg = TerrainInterpolationConfig(pixel_size_m=10.0, use_inverse_variance_blend_when_available=True)
    out = interpolate_support_aware_surface(inputs=inputs, config=cfg)
    assert np.isfinite(out['conditioned'][1, 1])
    assert np.isfinite(out['guidance_uncertainty'][1, 1])
    assert 'inverse_variance' in out['support_note']
