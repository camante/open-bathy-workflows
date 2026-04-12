#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for river_channel_template.py – normalized XS template system."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from river_channel_template import (
    ChannelTemplate,
    TemplateConfig,
    _normalize_xs_profile,
    attach_depth_fit_to_template,
    build_channel_template,
    compute_mean_template,
    evaluate_template_loo,
    extract_normalized_profiles,
    fit_width_depth_relation,
    make_fallback_template,
    synthesize_xs_from_template,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic test data
# ---------------------------------------------------------------------------

def _make_parabolic_xs(
    width_m: float,
    max_depth_m: float,
    wse_m: float,
    n_pts: int = 75,
    noise_std: float = 0.0,
    asymmetry: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a parabolic cross-section (dist_m, z_bed)."""
    dist = np.linspace(0, width_m, n_pts)
    # Normalized position [0,1]
    x = dist / width_m
    # Parabolic: depth = 4*Dmax * x * (1-x), shifted by asymmetry
    center = 0.5 + asymmetry
    depth = max_depth_m * (1.0 - ((x - center) / max(0.5, abs(center))) ** 2)
    depth = np.clip(depth, 0, max_depth_m)
    z_bed = wse_m - depth
    if noise_std > 0:
        z_bed += np.random.default_rng(42).normal(0, noise_std, size=z_bed.shape)
    return dist, z_bed


def _make_xs_dataframes(
    n_xs: int = 10,
    base_width: float = 100.0,
    base_depth: float = 3.0,
    wse: float = 10.0,
    spacing: float = 200.0,
    width_jitter: float = 20.0,
    depth_jitter: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic xs_lines and xs_points DataFrames for testing."""
    rng = np.random.default_rng(123)
    lines_rows = []
    points_rows = []

    for i in range(n_xs):
        xs_id = f"xs_{i:04d}"
        w = base_width + rng.uniform(-width_jitter, width_jitter)
        d = base_depth + rng.uniform(-depth_jitter, depth_jitter)
        asym = rng.uniform(-0.1, 0.1)
        station = i * spacing

        dist, z_bed = _make_parabolic_xs(w, d, wse, n_pts=50, asymmetry=asym)

        bank_left_z = wse + rng.uniform(0.5, 1.5)
        bank_right_z = wse + rng.uniform(0.5, 1.5)

        lines_rows.append({
            "xs_id": xs_id,
            "river_id": "river_0",
            "component_id": 0,
            "s_center_m": station,
            "bank_left_dist_m": 0.0,
            "bank_right_dist_m": w,
            "bank_left_z_m": bank_left_z,
            "bank_right_z_m": bank_right_z,
        })

        for j in range(len(dist)):
            points_rows.append({
                "xs_id": xs_id,
                "dist_m": dist[j],
                "z_dem": z_bed[j],
                "z_topo": np.nan,
                "is_bank_left": j == 0,
                "is_bank_right": j == len(dist) - 1,
            })

    xs_lines = pd.DataFrame(lines_rows)
    xs_points = pd.DataFrame(points_rows)
    return xs_lines, xs_points


# ---------------------------------------------------------------------------
# Tests: _normalize_xs_profile
# ---------------------------------------------------------------------------

class TestNormalizeXsProfile:
    def test_basic_parabolic(self):
        dist, z = _make_parabolic_xs(100, 3.0, 10.0, n_pts=100)
        result = _normalize_xs_profile(dist, z, 0.0, 100.0, 10.0, n_bins=50)
        assert result is not None
        assert result["width_m"] == 100.0
        assert abs(result["max_depth_m"] - 3.0) < 0.2
        assert result["n_valid_bins"] >= 30
        # Peak should be near center
        assert abs(result["asymmetry"]) < 0.15

    def test_asymmetric(self):
        dist, z = _make_parabolic_xs(100, 3.0, 10.0, n_pts=100, asymmetry=0.2)
        result = _normalize_xs_profile(dist, z, 0.0, 100.0, 10.0, n_bins=50)
        assert result is not None
        # Asymmetry should be positive (deeper toward right)
        assert result["asymmetry"] > 0.05

    def test_missing_banks_returns_none(self):
        dist, z = _make_parabolic_xs(100, 3.0, 10.0)
        result = _normalize_xs_profile(dist, z, np.nan, 100.0, 10.0)
        assert result is None

    def test_too_shallow_returns_none(self):
        dist = np.linspace(0, 100, 50)
        z = np.full(50, 9.9)  # only 0.1m below WSE
        result = _normalize_xs_profile(dist, z, 0.0, 100.0, 10.0, min_depth_m=0.3)
        assert result is None

    def test_norm_depth_range(self):
        dist, z = _make_parabolic_xs(80, 4.0, 10.0, n_pts=80)
        result = _normalize_xs_profile(dist, z, 0.0, 80.0, 10.0)
        assert result is not None
        nd = result["norm_depth"]
        assert np.nanmin(nd[np.isfinite(nd)]) >= 0.0
        assert np.nanmax(nd[np.isfinite(nd)]) <= 1.0


# ---------------------------------------------------------------------------
# Tests: extract_normalized_profiles
# ---------------------------------------------------------------------------

class TestExtractNormalizedProfiles:
    def test_basic_extraction(self):
        xs_lines, xs_points = _make_xs_dataframes(n_xs=8)
        profiles, summary = extract_normalized_profiles(xs_lines, xs_points)
        assert len(profiles) >= 5
        assert len(summary) == len(profiles)
        assert "width_m" in summary.columns
        assert "max_depth_m" in summary.columns

    def test_with_sounding_filter(self):
        xs_lines, xs_points = _make_xs_dataframes(n_xs=10)
        # Only mark 5 XS as having soundings
        calib = pd.DataFrame({
            "xs_id": [f"xs_{i:04d}" for i in range(5)],
            "calib_n": [10, 8, 6, 4, 2],
        })
        profiles, summary = extract_normalized_profiles(xs_lines, xs_points, sounding_calibration=calib)
        assert len(profiles) <= 5

    def test_with_numeric_point_ids_and_string_line_ids(self):
        xs_lines, xs_points = _make_xs_dataframes(n_xs=6)
        original_ids = xs_lines["xs_id"].tolist()
        id_map = {old: str(i) for i, old in enumerate(original_ids)}
        xs_lines["xs_id"] = xs_lines["xs_id"].map(id_map)
        xs_points["xs_id"] = xs_points["xs_id"].map({old: i for i, old in enumerate(original_ids)})
        calib = pd.DataFrame({
            "xs_id": [str(i) for i in range(6)],
            "calib_n": [3] * 6,
        })
        profiles, summary = extract_normalized_profiles(xs_lines, xs_points, sounding_calibration=calib)
        assert len(profiles) == 6
        assert len(summary) == 6


# ---------------------------------------------------------------------------
# Tests: compute_mean_template
# ---------------------------------------------------------------------------

class TestComputeMeanTemplate:
    def test_basic_template(self):
        xs_lines, xs_points = _make_xs_dataframes(n_xs=10)
        profiles, _ = extract_normalized_profiles(xs_lines, xs_points)
        template = compute_mean_template(profiles)
        assert template is not None
        assert template.n_profiles >= 3
        assert template.norm_pos.shape[0] == 50
        assert template.mean_depth.shape[0] == 50
        # Banks should be ~0
        assert template.mean_depth[0] == 0.0
        assert template.mean_depth[-1] == 0.0
        # Peak should be near center
        peak_idx = np.argmax(template.mean_depth)
        peak_pos = template.norm_pos[peak_idx]
        assert 0.3 < peak_pos < 0.7

    def test_too_few_profiles(self):
        cfg = TemplateConfig(min_measured_xs=5)
        profiles = [{"norm_depth": np.zeros(50)} for _ in range(2)]
        template = compute_mean_template(profiles, cfg)
        assert template is None

    def test_ignores_sparse_profiles_in_template_accounting(self):
        good = np.linspace(0.0, 1.0, 50)
        sparse = np.full(50, np.nan)
        sparse[:4] = [0.0, 0.2, 0.1, 0.0]
        profiles = [
            {"norm_depth": good, "asymmetry": 0.1},
            {"norm_depth": good * 0.9, "asymmetry": -0.1},
            {"norm_depth": good * 0.8, "asymmetry": 0.0},
            {"norm_depth": sparse, "asymmetry": 0.4},
        ]
        cfg = TemplateConfig(min_measured_xs=3)
        template = compute_mean_template(profiles, cfg)
        assert template is not None
        assert template.n_profiles == 3


# ---------------------------------------------------------------------------
# Tests: fit_width_depth_relation
# ---------------------------------------------------------------------------

class TestFitWidthDepth:
    def test_power_law_recovery(self):
        """With a clear power-law relationship, the fit should recover it."""
        rng = np.random.default_rng(99)
        true_a, true_b = 0.25, 0.45
        profiles = []
        for _ in range(20):
            w = rng.uniform(30, 200)
            d = true_a * (w ** true_b) * rng.uniform(0.9, 1.1)
            profiles.append({"width_m": w, "max_depth_m": d})
        a, b, r2, n, src = fit_width_depth_relation(profiles)
        assert src == "local_power_law"
        assert abs(b - true_b) < 0.15
        assert r2 > 0.5
        assert n == 20

    def test_fallback_on_sparse_data(self):
        cfg = TemplateConfig(fit_min_xs=10)
        profiles = [{"width_m": 100, "max_depth_m": 3}] * 3
        a, b, r2, n, src = fit_width_depth_relation(profiles, cfg)
        assert src == "fallback_leopold_maddock"
        assert a == cfg.fallback_a
        assert b == cfg.fallback_b


# ---------------------------------------------------------------------------
# Tests: ChannelTemplate
# ---------------------------------------------------------------------------

class TestChannelTemplate:
    def _make_template(self) -> ChannelTemplate:
        xs_lines, xs_points = _make_xs_dataframes(n_xs=12)
        profiles, _ = extract_normalized_profiles(xs_lines, xs_points)
        template = compute_mean_template(profiles)
        template = attach_depth_fit_to_template(template, profiles)
        return template

    def test_predict_dmax(self):
        t = self._make_template()
        d = t.predict_dmax(100.0)
        assert np.isfinite(d)
        assert d > 0

    def test_predict_dmax_nan_width(self):
        t = self._make_template()
        assert np.isnan(t.predict_dmax(np.nan))

    def test_synthesize_bed_basic(self):
        t = self._make_template()
        dist = np.linspace(0, 100, 50)
        z = t.synthesize_bed(dist, 100.0, 10.0)
        assert z.shape == dist.shape
        # All interior points should be below WSE
        inside = (dist > 5) & (dist < 95)
        assert np.all(z[inside] < 10.0)
        # Edge points should be close to WSE
        assert z[0] >= 9.5  # near bank

    def test_synthesize_bed_with_override(self):
        t = self._make_template()
        dist = np.linspace(0, 100, 50)
        z_default = t.synthesize_bed(dist, 100.0, 10.0)
        z_deep = t.synthesize_bed(dist, 100.0, 10.0, dmax_override=8.0)
        # Deeper override should produce lower bed
        inside = (dist > 20) & (dist < 80)
        assert np.nanmean(z_deep[inside]) < np.nanmean(z_default[inside])

    def test_serialization_roundtrip(self):
        t = self._make_template()
        d = t.to_dict()
        t2 = ChannelTemplate.from_dict(d)
        assert t2.n_profiles == t.n_profiles
        np.testing.assert_allclose(t2.mean_depth, t.mean_depth, atol=1e-10)
        np.testing.assert_allclose(t2.norm_pos, t.norm_pos, atol=1e-10)
        assert t2.depth_a == t.depth_a
        assert t2.depth_b == t.depth_b

    def test_json_serialization(self):
        t = self._make_template()
        s = json.dumps(t.to_dict(), default=str)
        d = json.loads(s)
        t2 = ChannelTemplate.from_dict(d)
        assert t2.n_profiles == t.n_profiles


# ---------------------------------------------------------------------------
# Tests: synthesize_xs_from_template
# ---------------------------------------------------------------------------

class TestSynthesizeXs:
    def test_basic_synthesis(self):
        xs_lines, xs_points = _make_xs_dataframes(n_xs=10)
        profiles, _ = extract_normalized_profiles(xs_lines, xs_points)
        template = compute_mean_template(profiles)
        template = attach_depth_fit_to_template(template, profiles)

        dist = np.linspace(0, 120, 60)
        z, meta = synthesize_xs_from_template(template, dist, 120.0, 10.0)
        assert z.shape == dist.shape
        assert meta["source"] == "channel_template"
        assert meta["dmax_used_m"] is not None
        # Bed should be below WSE inside channel
        inside = (dist > 10) & (dist < 110)
        assert np.all(np.isfinite(z[inside]))
        assert np.all(z[inside] < 10.0)


# ---------------------------------------------------------------------------
# Tests: LOO evaluation
# ---------------------------------------------------------------------------

class TestEvaluateLOO:
    def test_loo_runs(self):
        xs_lines, xs_points = _make_xs_dataframes(n_xs=12)
        profiles, _ = extract_normalized_profiles(xs_lines, xs_points)
        results = evaluate_template_loo(profiles)
        assert not results.empty
        assert "rmse_norm_depth" in results.columns
        assert "dmax_error_m" in results.columns
        # RMSE should be reasonable for near-identical parabolic sections
        assert results["rmse_norm_depth"].median() < 0.3

    def test_loo_too_few(self):
        cfg = TemplateConfig(min_measured_xs=5)
        profiles = [{"norm_depth": np.zeros(50), "width_m": 100, "max_depth_m": 3}] * 3
        results = evaluate_template_loo(profiles, cfg)
        assert results.empty


# ---------------------------------------------------------------------------
# Tests: build_channel_template (full pipeline)
# ---------------------------------------------------------------------------

class TestBuildChannelTemplate:
    def test_full_pipeline(self, tmp_path):
        xs_lines, xs_points = _make_xs_dataframes(n_xs=15)
        cfg = TemplateConfig(write_diagnostics=True)
        template = build_channel_template(xs_lines, xs_points, cfg=cfg, out_dir=tmp_path)
        assert template is not None
        assert template.n_profiles >= 3
        assert (tmp_path / "channel_template.json").exists()
        assert (tmp_path / "channel_template_profiles.csv").exists()
        assert (tmp_path / "channel_template_width_depth.csv").exists()

        assert (tmp_path / "channel_template_profiles.npz").exists()
        payload = json.loads((tmp_path / "channel_template.json").read_text())
        assert payload["loo_quality_gate"] in {"PASSED", "FAILED", "skipped"}

    def test_full_pipeline_with_sounding_filter(self, tmp_path):
        xs_lines, xs_points = _make_xs_dataframes(n_xs=15)
        calib = pd.DataFrame({
            "xs_id": [f"xs_{i:04d}" for i in range(8)],
            "calib_n": [5] * 8,
        })
        template = build_channel_template(xs_lines, xs_points, sounding_calibration=calib, out_dir=tmp_path)
        assert template is not None
        assert template.n_profiles <= 8

    def test_insufficient_data_returns_none(self):
        xs_lines, xs_points = _make_xs_dataframes(n_xs=2)
        template = build_channel_template(xs_lines, xs_points)
        assert template is None


# ---------------------------------------------------------------------------
# Tests: fallback template
# ---------------------------------------------------------------------------

class TestFallbackTemplate:
    def test_parabolic_shape(self):
        t = make_fallback_template()
        assert t.n_profiles == 0
        assert t.depth_fit_source == "fallback_parabolic"
        # Banks at 0
        assert t.mean_depth[0] == 0.0
        assert t.mean_depth[-1] == 0.0
        # Peak at 1.0
        assert np.nanmax(t.mean_depth) == pytest.approx(1.0, abs=0.05)
        # Should be symmetric
        mid = len(t.mean_depth) // 2
        np.testing.assert_allclose(
            t.mean_depth[:mid],
            t.mean_depth[-mid:][::-1],
            atol=0.05,
        )
        peak = int(np.nanargmax(t.mean_depth))
        assert peak not in (0, len(t.mean_depth) - 1)
        assert len(np.flatnonzero(np.isclose(t.mean_depth, np.nanmax(t.mean_depth)))) <= 2
        assert t.mean_depth[peak - 1] < t.mean_depth[peak]

    def test_fallback_synthesize_bed(self):
        t = make_fallback_template()
        t.depth_a = 0.18
        t.depth_b = 0.50
        dist = np.linspace(0, 80, 40)
        z = t.synthesize_bed(dist, 80.0, 5.0)
        assert np.all(np.isfinite(z))
        assert np.all(z <= 5.0)
        # Deepest point should be near center
        deepest_idx = np.argmin(z)
        deepest_pos = dist[deepest_idx]
        assert 25 < deepest_pos < 55
