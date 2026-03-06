# -*- coding: utf-8 -*-
"""Unit test for WSE-fit fallback behavior in the 1D energy solver.

Regression test: if wse_fit_m exists but is only partially populated, the solver
must fall back to wse_m for missing rows (instead of selecting wse_fit_m globally
and dropping most stations).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure workflow root is importable when running tests directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_energy_solver_uses_wse_m_when_wse_fit_missing():
    import xs_infer_bathy_raster as xir

    cfg = xir.InferConfig()
    cfg.energy_solver_enabled = True
    cfg.energy_allow_dem_proxy_wse = True
    cfg.bottom_width_frac = 0.20
    cfg.dmin_m = 0.5
    cfg.dmax_m = 15.0
    cfg.slope_min = 1e-5
    cfg.slope_max = 0.05

    df = pd.DataFrame(
        {
            "component_id": [1, 1, 1],
            "s_center_m": [0.0, 10.0, 20.0],
            "width_m": [25.0, 25.0, 25.0],
            "manning_q_cms_used": [100.0, 100.0, 100.0],
            "wse_m": [10.0, 9.9, 9.8],
            "wse_fit_m": [np.nan, 9.95, np.nan],
            "dmax_prior_m": [2.0, 2.0, 2.0],
        }
    )

    acct = {"wse_anchor_source": "dem_proxy"}
    out = xir._apply_1d_energy_solver(xs_param=df, cfg=cfg, soundings_path=None, acct=acct)

    assert int(acct.get("energy_solver_n_total", 0)) >= 2
    assert int(acct.get("energy_solver_n_applied", 0)) >= 2

    # Ensure we used both fitted and raw WSE values.
    assert int(acct.get("energy_solver_wse_fit_used_n", 0)) >= 1
    assert int(acct.get("energy_solver_wse_raw_used_n", 0)) >= 1

    assert "energy_dmax_m" in out.columns
    assert np.isfinite(pd.to_numeric(out["energy_dmax_m"], errors="coerce").values).any()
