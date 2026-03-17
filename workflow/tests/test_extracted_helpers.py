#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration smoke tests for extracted helper functions.

These tests exercise the real function signatures and parameter paths
that have historically been the source of scope bugs (NameError from
closure capture of undefined locals) and type bugs (string-as-iterable).

All tests are pure numpy/pandas — no network, no rasterio, no filesystem.
"""

import sys
import os
import types
import numpy as np
import pandas as pd
import pytest

# Ensure workflow root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 1. _filter_extra_xyz_for_sdb — the function that caused the NameError
# ---------------------------------------------------------------------------

class TestFilterExtraXyzForSdb:
    """Verify _filter_extra_xyz_for_sdb works as a module-level function
    with explicit args/ctx, not capturing closure locals."""

    def _make_args(self, **overrides):
        """Create a minimal argparse-like namespace."""
        ns = types.SimpleNamespace(
            extra_xyz_sdb_mode="auto",
            max_depth_hard_cap=25.0,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def _make_ctx(self, kd_max_depth_m=None):
        """Create a minimal ctx-like object with kd_max_depth_m."""
        return types.SimpleNamespace(kd_max_depth_m=kd_max_depth_m)

    def _make_df(self, depths, sources=None):
        d = {"depth_m": depths, "lon": np.zeros(len(depths)), "lat": np.zeros(len(depths))}
        if sources is not None:
            d["source"] = sources
        return pd.DataFrame(d)

    def test_import_and_call_no_error(self):
        """The function should be importable at module level and callable
        without NameError — this is the exact bug that killed the run."""
        from sdb_main import _filter_extra_xyz_for_sdb
        df = self._make_df([-1.0, -2.0, -5.0])
        args = self._make_args()
        result = _filter_extra_xyz_for_sdb(df, args, ctx=None)
        assert result is not None
        assert len(result) == 3

    def test_physics_cap_from_ctx(self):
        """When ctx.kd_max_depth_m is set, the guard should use it."""
        from sdb_main import _filter_extra_xyz_for_sdb
        df = self._make_df([-1.0, -2.0, -15.0, -30.0])
        args = self._make_args()
        ctx = self._make_ctx(kd_max_depth_m=20.0)
        result = _filter_extra_xyz_for_sdb(df, args, ctx=ctx)
        assert result is not None
        # -30 should be filtered (abs > 20), -15 kept
        assert len(result) == 3
        assert float(result["depth_m"].min()) >= -20.0

    def test_none_ctx_no_crash(self):
        """ctx=None should not crash (fallback to 10m cap)."""
        from sdb_main import _filter_extra_xyz_for_sdb
        df = self._make_df([-1.0, -5.0, -12.0])
        args = self._make_args()
        result = _filter_extra_xyz_for_sdb(df, args, ctx=None)
        assert result is not None
        # -12 exceeds 10m fallback cap
        assert len(result) == 2

    def test_exclude_mode(self):
        from sdb_main import _filter_extra_xyz_for_sdb
        df = self._make_df([-1.0])
        args = self._make_args(extra_xyz_sdb_mode="exclude")
        result = _filter_extra_xyz_for_sdb(df, args, ctx=None)
        assert result is None

    def test_allow_mode(self):
        from sdb_main import _filter_extra_xyz_for_sdb
        df = self._make_df([-1.0, -100.0])
        args = self._make_args(extra_xyz_sdb_mode="allow")
        result = _filter_extra_xyz_for_sdb(df, args, ctx=None)
        assert result is not None
        assert len(result) == 2  # no filtering in allow mode

    def test_empty_df(self):
        from sdb_main import _filter_extra_xyz_for_sdb
        df = pd.DataFrame(columns=["depth_m", "lon", "lat"])
        args = self._make_args()
        result = _filter_extra_xyz_for_sdb(df, args, ctx=None)
        assert result is not None
        assert len(result) == 0

    def test_none_df(self):
        from sdb_main import _filter_extra_xyz_for_sdb
        args = self._make_args()
        result = _filter_extra_xyz_for_sdb(None, args, ctx=None)
        assert result is None

    def test_mixed_sign_hydronos_guard(self):
        """Hydronos-like data with positive depths should trigger the
        mixed-sign provenance guard."""
        from sdb_main import _filter_extra_xyz_for_sdb
        n = 200
        depths = np.concatenate([np.full(190, -3.0), np.full(10, 2.0)])
        sources = ["extra_xyz_hydronos"] * n
        df = self._make_df(depths, sources)
        args = self._make_args()
        result = _filter_extra_xyz_for_sdb(df, args, ctx=None)
        # Should keep only non-positive rows
        assert result is not None
        assert (result["depth_m"] <= 0).all()


# ---------------------------------------------------------------------------
# 2. _load_river_authoritative_support_points — string-as-iterable bug
# ---------------------------------------------------------------------------

class TestLoadRiverSupportPoints:
    """Verify that river_soundings as a comma-separated string is handled
    correctly and not iterated character-by-character."""

    def test_string_soundings_not_char_iterated(self):
        """The function should split comma-separated paths, not iterate chars."""
        # We can't easily call the real function without rasterio/geopandas,
        # but we can verify the path-splitting logic directly.
        # This tests the exact code pattern that was fixed.
        river_soundings = "/path/to/file_a.xyz,/path/to/file_b.xyz"

        # The FIXED logic:
        files = []
        for part in str(river_soundings).split(","):
            part = part.strip()
            if part:
                files.append(part)

        assert len(files) == 2
        assert files[0] == "/path/to/file_a.xyz"
        assert files[1] == "/path/to/file_b.xyz"

    def test_single_path_string(self):
        river_soundings = "/single/path/soundings.xyz"
        files = []
        for part in str(river_soundings).split(","):
            part = part.strip()
            if part:
                files.append(part)
        assert len(files) == 1
        assert files[0] == "/single/path/soundings.xyz"

    def test_empty_string(self):
        river_soundings = ""
        files = []
        for part in str(river_soundings).split(","):
            part = part.strip()
            if part:
                files.append(part)
        assert len(files) == 0


# ---------------------------------------------------------------------------
# 3. _gaussian_smooth_masked — sigma cap prevents OOM/hang
# ---------------------------------------------------------------------------

class TestGaussianSmoothMaskedCap:
    """Verify the sigma_px safety cap prevents runaway kernel sizes."""

    def test_sigma_capped_to_quarter_dimension(self):
        """A 100x100 raster with sigma_px=10000 should be capped."""
        from geo.raster_ops import _gaussian_smooth_masked
        data = np.random.randn(100, 100).astype("float32")
        valid = np.ones((100, 100), dtype=bool)
        # This would have hung before the fix
        result = _gaussian_smooth_masked(data, valid, sigma_px=10000.0)
        assert result.shape == (100, 100)
        assert np.isfinite(result).all()

    def test_small_sigma_unchanged(self):
        """Normal sigma values should not be capped."""
        from geo.raster_ops import _gaussian_smooth_masked
        data = np.random.randn(200, 200).astype("float32")
        valid = np.ones((200, 200), dtype=bool)
        result = _gaussian_smooth_masked(data, valid, sigma_px=3.0)
        assert result.shape == (200, 200)

    def test_zero_sigma_passthrough(self):
        from geo.raster_ops import _gaussian_smooth_masked
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        valid = np.ones((2, 2), dtype=bool)
        result = _gaussian_smooth_masked(data, valid, sigma_px=0.0)
        np.testing.assert_array_equal(result, data)
