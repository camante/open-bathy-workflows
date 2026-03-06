# -*- coding: utf-8 -*-
"""Unit tests for energy-solver safety gate and receipts.

These tests are intentionally small and deterministic. They do not require
external data or GIS libraries.
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure workflow root is importable when running tests directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestEnergySolverGate(unittest.TestCase):
    def _make_cfg(self):
        # Import locally to keep test import-time lightweight.
        import xs_infer_bathy_raster as xir

        cfg = xir.InferConfig()
        cfg.energy_solver_enabled = True
        cfg.energy_solver_only_when_no_soundings = False
        cfg.energy_allow_dem_proxy_wse = False
        cfg.manning_n = 0.03
        cfg.bottom_width_frac = 0.2
        cfg.slope_min = 1e-6
        cfg.slope_max = 1e-1
        cfg.dmin_m = 0.5
        cfg.dmax_m = 10.0
        return cfg

    def _make_xs_param(self):
        # Two XS points in one component with a simple downstream drop in WSE.
        return pd.DataFrame(
            {
                "component_id": [1, 1],
                "s_center_m": [0.0, 10.0],
                "width_m": [10.0, 10.0],
                "manning_q_cms_used": [5.0, 5.0],
                "wse_fit_m": [5.0, 4.0],
                "wse_m": [5.0, 4.0],
                "dmax_prior_m": [2.0, 2.0],
            }
        )

    def test_gate_skips_when_dem_proxy(self):
        import xs_infer_bathy_raster as xir

        cfg = self._make_cfg()
        xs = self._make_xs_param()
        acct = {"wse_anchor_source": "dem_proxy"}

        out = xir._apply_1d_energy_solver(xs_param=xs, cfg=cfg, soundings_path=None, acct=acct)
        self.assertTrue(out["dmax_prior_m"].equals(xs["dmax_prior_m"]))
        self.assertEqual(acct.get("energy_solver_reason"), "skipped_dem_proxy_wse")
        self.assertEqual(int(acct.get("energy_solver_n_applied", 0)), 0)

    def test_gate_allows_when_overridden(self):
        import xs_infer_bathy_raster as xir

        cfg = self._make_cfg()
        cfg.energy_allow_dem_proxy_wse = True
        xs = self._make_xs_param()
        acct = {"wse_anchor_source": "dem_proxy"}

        out = xir._apply_1d_energy_solver(xs_param=xs, cfg=cfg, soundings_path=None, acct=acct)
        # Should compute at least one energy_dmax value when allowed.
        self.assertIn("energy_dmax_m", out.columns)
        self.assertTrue(np.isfinite(pd.to_numeric(out["energy_dmax_m"], errors="coerce")).any())
        self.assertEqual(acct.get("energy_solver_reason"), "ok")
        self.assertGreaterEqual(int(acct.get("energy_solver_n_applied", 0)), 1)



    def test_reason_no_valid_stations_when_no_discharge(self):
        import xs_infer_bathy_raster as xir

        cfg = self._make_cfg()
        xs = self._make_xs_param()
        xs['manning_q_cms_used'] = [np.nan, np.nan]
        acct = {'wse_anchor_source': 'observed'}

        out = xir._apply_1d_energy_solver(xs_param=xs, cfg=cfg, soundings_path=None, acct=acct)
        self.assertIn('energy_dmax_m', out.columns)
        self.assertFalse(np.isfinite(pd.to_numeric(out['energy_dmax_m'], errors='coerce')).any())
        self.assertEqual(acct.get('energy_solver_reason'), 'no_valid_stations')
if __name__ == "__main__":
    raise SystemExit(unittest.main())
