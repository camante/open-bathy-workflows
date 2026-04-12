import numpy as np
import pandas as pd

from xs_infer_bathy_raster import _recover_banks_from_profile_extent, _pick_representative_bank_z


def test_recover_banks_from_profile_extent_uses_outermost_finite_bank_elevations():
    xsp = pd.DataFrame(
        {
            "dist_m": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "z_dem": [np.nan, 10.0, 9.0, 8.0, 11.0, np.nan],
            "z_topo": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    left, right, left_z, right_z = _recover_banks_from_profile_extent(xsp)
    assert left == 1.0
    assert right == 4.0
    assert left_z == 10.0
    assert right_z == 11.0


def test_pick_representative_bank_z_avoids_high_bank_outlier_near_margin():
    xsp = pd.DataFrame(
        {
            "dist_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "z_dem": [12.0, 8.0, 7.5, 7.0, 6.5],
            "z_topo": [12.5, 8.2, np.nan, np.nan, np.nan],
        }
    )
    z = _pick_representative_bank_z(xsp, side="left", bank_dist=0.5, window_m=2.0)
    assert z is not None
    assert 7.5 <= z <= 12.0
    assert z < 12.5
