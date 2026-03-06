# -*- coding: utf-8 -*-
"""Unit tests for region-specific Manning n handling.

These are lightweight, deterministic tests (no GIS, no network).
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestManningNEffective(unittest.TestCase):
    def test_region_mapping_applied(self):
        import xs_infer_bathy_raster as xir

        cfg = xir.InferConfig()
        cfg.manning_n = 0.04
        cfg.manning_region = "new_england"
        cfg.manning_n_by_region = {"new_england": 0.03}

        n_eff = xir._manning_n_effective(cfg)
        self.assertAlmostEqual(float(n_eff), 0.03, places=9)

    def test_region_mapping_fallback(self):
        import xs_infer_bathy_raster as xir

        cfg = xir.InferConfig()
        cfg.manning_n = 0.04
        cfg.manning_region = "missing"
        cfg.manning_n_by_region = {"new_england": 0.03}

        n_eff = xir._manning_n_effective(cfg)
        self.assertAlmostEqual(float(n_eff), 0.04, places=9)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
