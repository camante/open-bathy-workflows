import pandas as pd

from train import _compute_physics_guidance_settings


def test_anchor_support_good_loosens_residuals():
    n = 1400
    df = pd.DataFrame({
        "depth_m": [-float(1 + (i % 9) * 0.6) for i in range(n)],
        "stumpf_depth": [float(1 + (i % 9) * 0.6) for i in range(n)],
        "stumpf_idx": [1.0 + (i % 50) * 0.01 for i in range(n)],
        "source_norm": ["extra_xyz:hydronos"] * n,
        "source": ["extra_xyz:hydronos"] * n,
        "longitude": [-70.9 + (i % 40) * 1e-3 for i in range(n)],
        "latitude": [42.8 + (i // 40) * 1e-3 for i in range(n)],
    })
    g = _compute_physics_guidance_settings(df)
    assert g["anchor_support_good"] is True
    assert g["low_support"] is False
    assert g["correction_alpha"] >= 0.18
    assert g["residual_clip_m"] >= 0.75
