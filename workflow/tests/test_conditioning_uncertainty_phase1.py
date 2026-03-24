import numpy as np

from authoritative_conditioning import support_weighted_condition_arrays


def test_conditioning_phase1_uncertainty_outputs_present_and_finite():
    auth = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    sdb_ok = np.array([[False, True], [False, False]])
    river_ok = np.array([[False, False], [True, False]])
    sdb_depth = np.array([[np.nan, 2.0], [np.nan, np.nan]], dtype=np.float32)
    river_depth = np.array([[np.nan, np.nan], [3.0, np.nan]], dtype=np.float32)
    out = support_weighted_condition_arrays(
        auth=auth,
        candidate=None,
        sdb_depth_guidance=sdb_depth,
        river_depth_guidance=river_depth,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_gw=np.array([[0.0, 0.8], [0.0, 0.0]], dtype=np.float32),
        sdb_ti=None,
        river_gw=np.array([[0.0, 0.0], [0.9, 0.0]], dtype=np.float32),
        river_ti=None,
        river_support=None,
        river_support_depth=None,
        sdb_uncertainty=np.array([[np.nan, 0.4], [np.nan, np.nan]], dtype=np.float32),
        river_uncertainty=np.array([[np.nan, np.nan], [0.6, np.nan]], dtype=np.float32),
        pixel_size_m=2.0,
        support_decay_m=10.0,
        support_density_radius_m=10.0,
        coastal_sdb_support_transition_m=10.0,
        river_anchor_density_radius_m=10.0,
        river_scaffold_transition_m=10.0,
    )

    assert "anchor_uncertainty" in out
    assert "guidance_uncertainty" in out
    assert "conditioned_uncertainty" in out
    assert out["anchor_uncertainty"][0, 0] == 0.0
    assert np.isfinite(out["guidance_uncertainty"][0, 1])
    assert np.isfinite(out["guidance_uncertainty"][1, 0])
    assert np.isfinite(out["conditioned_uncertainty"][0, 1])
    assert np.isfinite(out["conditioned_uncertainty"][1, 0])


def test_conditioning_phase1_guidance_uncertainty_falls_back_when_missing():
    auth = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    sdb_ok = np.array([[False, True], [False, False]])
    river_ok = np.zeros_like(sdb_ok)
    out = support_weighted_condition_arrays(
        auth=auth,
        candidate=None,
        sdb_depth_guidance=np.array([[np.nan, 2.0], [np.nan, np.nan]], dtype=np.float32),
        river_depth_guidance=None,
        sdb_ok=sdb_ok,
        river_ok=river_ok,
        sdb_gw=np.array([[0.0, 0.8], [0.0, 0.0]], dtype=np.float32),
        sdb_ti=None,
        river_gw=None,
        river_ti=None,
        river_support=None,
        river_support_depth=None,
        pixel_size_m=2.0,
        support_decay_m=10.0,
        support_density_radius_m=10.0,
        coastal_sdb_support_transition_m=10.0,
        river_anchor_density_radius_m=10.0,
        river_scaffold_transition_m=10.0,
    )
    assert np.isfinite(out["guidance_uncertainty"][0, 1])
    assert out["guidance_uncertainty"][0, 1] >= out["anchor_uncertainty"][0, 1]
