from __future__ import annotations

import math

import pandas as pd

import xs_infer_bathy_raster as xir


def test_profile_bank_extent_fallback_uses_finite_profile_edges():
    xsp = pd.DataFrame(
        {
            "dist_m": [0.0, 5.0, 10.0, 15.0],
            "z_dem": [1.2, 0.5, 0.4, 1.4],
            "z_topo": [float("nan"), float("nan"), float("nan"), float("nan")],
        }
    )

    left, right, left_z, right_z = xir._profile_bank_extent_fallback(xsp)

    assert left == 0.0
    assert right == 15.0
    assert left_z == 1.2
    assert right_z == 1.4



def test_profile_bank_extent_fallback_requires_two_finite_samples():
    xsp = pd.DataFrame(
        {
            "dist_m": [0.0, 5.0, 10.0],
            "z_dem": [float("nan"), 0.5, float("nan")],
        }
    )

    left, right, left_z, right_z = xir._profile_bank_extent_fallback(xsp)

    assert left is None
    assert right is None
    assert left_z is None
    assert right_z is None
