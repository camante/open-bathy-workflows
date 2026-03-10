"""
Unit tests for core numerical functions:
- Manning inversion (manning_inversion.py)
- Refraction correction (atl.py)
- Kd-based depth limits (kd_estimation.py)
- Cross-section curvature asymmetry (xs_infer_bathy_raster.py)
"""
import sys, os, math, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Manning Inversion
# ---------------------------------------------------------------------------

class TestManningInversion(unittest.TestCase):
    """Tests for invert_manning_for_depth and related helpers."""

    def setUp(self):
        from manning_inversion import (
            invert_manning_for_depth,
            estimate_manning_n,
            compute_backwater_guard,
            estimate_q2_from_drainage_area,
            blend_manning_with_multivariate,
            ManningResult,
        )
        self.invert = invert_manning_for_depth
        self.est_n = estimate_manning_n
        self.backwater = compute_backwater_guard
        self.q2 = estimate_q2_from_drainage_area
        self.blend = blend_manning_with_multivariate
        self.ManningResult = ManningResult

    def test_basic_depth_formula(self):
        """D = (Q*n / (W*sqrt(S)))^0.6 for a textbook case."""
        Q, W, S, n = 50.0, 20.0, 0.001, 0.035
        expected = (Q * n / (W * math.sqrt(S))) ** 0.6
        result = self.invert(Q, W, S, manning_n=n)
        self.assertAlmostEqual(result.depth_m, expected, places=4)

    def test_depth_increases_with_discharge(self):
        """Higher discharge → deeper channel, all else equal."""
        r1 = self.invert(10.0, 15.0, 0.001)
        r2 = self.invert(100.0, 15.0, 0.001)
        self.assertGreater(r2.depth_m, r1.depth_m)

    def test_depth_decreases_with_wider_channel(self):
        """Wider channel → shallower for same discharge."""
        r_narrow = self.invert(30.0, 10.0, 0.001)
        r_wide = self.invert(30.0, 50.0, 0.001)
        self.assertGreater(r_narrow.depth_m, r_wide.depth_m)

    def test_depth_decreases_with_steeper_slope(self):
        """Steeper slope → faster flow → shallower depth."""
        r_flat = self.invert(30.0, 15.0, 0.0001)
        r_steep = self.invert(30.0, 15.0, 0.01)
        self.assertGreater(r_flat.depth_m, r_steep.depth_m)

    def test_zero_discharge_returns_nan(self):
        result = self.invert(0.0, 15.0, 0.001)
        self.assertTrue(math.isnan(result.depth_m))
        self.assertEqual(result.confidence, 0.0)

    def test_negative_discharge_returns_nan(self):
        result = self.invert(-5.0, 15.0, 0.001)
        self.assertTrue(math.isnan(result.depth_m))

    def test_zero_width_returns_nan(self):
        result = self.invert(30.0, 0.0, 0.001)
        self.assertTrue(math.isnan(result.depth_m))

    def test_zero_slope_returns_fallback_depth(self):
        """Zero slope triggers backwater fallback (0.18 * W^0.5)."""
        W = 25.0
        result = self.invert(30.0, W, 0.0)
        expected_fallback = 0.18 * (W ** 0.5)
        self.assertAlmostEqual(result.depth_m, expected_fallback, places=4)
        self.assertTrue(result.backwater_flag)

    def test_depth_clamped_to_min(self):
        """Very low discharge should not produce depth < min_depth_m."""
        result = self.invert(0.001, 100.0, 0.01, min_depth_m=0.3)
        self.assertGreaterEqual(result.depth_m, 0.3)

    def test_depth_clamped_to_max(self):
        """Very large discharge should not produce depth > max_depth_m."""
        result = self.invert(1e6, 10.0, 1e-6, max_depth_m=20.0)
        self.assertLessEqual(result.depth_m, 20.0)

    def test_result_has_required_fields(self):
        result = self.invert(50.0, 20.0, 0.001)
        for attr in ('depth_m', 'uncertainty_m', 'confidence', 'discharge_m3s',
                     'width_m', 'slope', 'manning_n', 'backwater_flag'):
            self.assertTrue(hasattr(result, attr), f"Missing field: {attr}")

    def test_uncertainty_positive(self):
        result = self.invert(50.0, 20.0, 0.001)
        self.assertGreater(result.uncertainty_m, 0.0)

    def test_confidence_in_range(self):
        result = self.invert(50.0, 20.0, 0.001)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_manning_n_positive(self):
        """estimate_manning_n returns a (n_value, uncertainty) tuple; both > 0."""
        for bed_type in ("sand", "gravel", "bedrock", "unknown"):
            result = self.est_n(bed_type, "national_default")
            n_val = result[0] if isinstance(result, tuple) else float(result)
            self.assertGreater(n_val, 0.0, f"n <= 0 for bed_type={bed_type!r}")

    def test_manning_n_rougher_bed_higher(self):
        """Rougher bed type → higher n."""
        n_sand = self.est_n("sand", "national_default")
        n_boulder = self.est_n("boulder", "national_default")
        n_sand_val = n_sand[0] if isinstance(n_sand, tuple) else float(n_sand)
        n_boulder_val = n_boulder[0] if isinstance(n_boulder, tuple) else float(n_boulder)
        self.assertGreater(n_boulder_val, n_sand_val)

    def test_backwater_guard_tidal_proximity(self):
        """Near-tide location should trigger backwater/tidal flag."""
        guard = self.backwater(slope=0.0001, distance_to_tide_m=100.0, elevation_m=1.0)
        self.assertTrue(guard.is_tidal or guard.is_backwater)
        self.assertLess(guard.confidence_factor, 1.0)

    def test_backwater_guard_steep_far_from_tide(self):
        """Steep, high-elevation reach should not be flagged as tidal."""
        guard = self.backwater(slope=0.005, distance_to_tide_m=50000.0, elevation_m=200.0)
        self.assertFalse(guard.is_tidal)

    def test_backwater_guard_confidence_bounded(self):
        for slope, dist, elev in [(0.0001, 50, 0.5), (0.001, 5000, 20), (0.01, 100000, 500)]:
            guard = self.backwater(slope=slope, distance_to_tide_m=dist, elevation_m=elev)
            self.assertGreater(guard.confidence_factor, 0.0)
            self.assertLessEqual(guard.confidence_factor, 1.0)

    def test_q2_positive(self):
        q2 = self.q2(drainage_area_km2=500.0, region="national_default")
        q2_val = q2.q2_m3s if hasattr(q2, 'q2_m3s') else float(q2)
        self.assertGreater(q2_val, 0.0)

    def test_q2_scales_with_drainage_area(self):
        """Larger drainage area → higher Q2 discharge."""
        q_small = self.q2(100.0, "national_default")
        q_large = self.q2(5000.0, "national_default")
        q_small_val = q_small.q2_m3s if hasattr(q_small, 'q2_m3s') else float(q_small)
        q_large_val = q_large.q2_m3s if hasattr(q_large, 'q2_m3s') else float(q_large)
        self.assertGreater(q_large_val, q_small_val)

    def test_blend_prefers_manning_at_high_confidence(self):
        """At high Manning confidence, blended depth closer to Manning than multivariate."""
        r = self.ManningResult(
            depth_m=3.5, uncertainty_m=1.0, confidence=0.9,
            discharge_m3s=50, width_m=20, slope=0.001, manning_n=0.035
        )
        result = self.blend(r, multivariate_depth_m=8.0, multivariate_confidence=0.5)
        blended_depth = result[0] if isinstance(result, tuple) else float(result)
        self.assertLess(abs(blended_depth - 3.5), abs(blended_depth - 8.0))

    def test_blend_returns_finite_depth(self):
        r = self.ManningResult(
            depth_m=2.0, uncertainty_m=0.8, confidence=0.5,
            discharge_m3s=20, width_m=10, slope=0.001, manning_n=0.035
        )
        result = self.blend(r, multivariate_depth_m=4.0, multivariate_confidence=0.6)
        blended_depth = result[0] if isinstance(result, tuple) else float(result)
        self.assertTrue(math.isfinite(blended_depth))
        self.assertGreater(blended_depth, 0.0)


# ---------------------------------------------------------------------------
# Refraction Correction
# ---------------------------------------------------------------------------

class TestRefractionCorrection(unittest.TestCase):
    """Tests for ATL03 photon refraction correction."""

    def setUp(self):
        from atl import apply_hybrid_refraction, _n_water_index
        self.refract = apply_hybrid_refraction
        self.n_water = _n_water_index

    def test_nadir_correction_matches_snells_law(self):
        """At nadir (theta=0), Z_true = Z_app / n_water (exact Snell's law result)."""
        n_w = self.n_water(temp_c=20.0, wavelength_nm=532.0)
        z_app = 7.5
        z_true = self.refract(z_app, theta_air_rad=0.0)
        self.assertAlmostEqual(z_true, z_app / n_w, places=5)

    def test_corrected_depth_less_than_apparent(self):
        """Refraction correction always makes apparent depth shorter."""
        for theta in [0.0, 0.1, 0.3, 0.5]:
            z_app = 10.0
            z_true = self.refract(z_app, theta)
            self.assertLess(z_true, z_app,
                msg=f"z_true ({z_true:.4f}) >= z_app ({z_app}) at theta={theta}")

    def test_scales_linearly_with_depth(self):
        """For a fixed angle, the depth ratio is constant (linear scaling)."""
        theta = 0.2
        z1, z2 = 5.0, 10.0
        r1 = self.refract(z1, theta) / z1
        r2 = self.refract(z2, theta) / z2
        self.assertAlmostEqual(r1, r2, places=8)

    def test_n_water_index_in_physical_range(self):
        """Refractive index should be ~1.33-1.34 for typical ocean conditions."""
        n = self.n_water(temp_c=20.0, wavelength_nm=532.0)
        self.assertGreater(n, 1.30)
        self.assertLess(n, 1.40)

    def test_n_water_decreases_with_temperature(self):
        """Warmer water has slightly lower refractive index."""
        n_cold = self.n_water(temp_c=5.0)
        n_warm = self.n_water(temp_c=30.0)
        self.assertGreater(n_cold, n_warm)

    def test_array_input_works(self):
        """Function should work on numpy arrays."""
        depths = np.array([1.0, 5.0, 10.0, 20.0])
        theta = np.full_like(depths, 0.1)
        result = self.refract(depths, theta)
        self.assertEqual(result.shape, depths.shape)
        self.assertTrue(np.all(result < depths))

    def test_zero_depth_returns_zero(self):
        """Zero apparent depth → zero true depth."""
        result = self.refract(0.0, 0.1)
        self.assertAlmostEqual(result, 0.0, places=8)

    def test_angle_changes_ratio(self):
        """Off-nadir angle changes the depth correction ratio."""
        z_app = 8.0
        ratio_nadir = self.refract(z_app, 0.01) / z_app
        ratio_offnadir = self.refract(z_app, 0.5) / z_app
        self.assertNotAlmostEqual(ratio_nadir, ratio_offnadir, places=3)


# ---------------------------------------------------------------------------
# Kd-Based Depth Limits
# ---------------------------------------------------------------------------

class TestKdDepthLimits(unittest.TestCase):
    """Tests for Kd-based SDB depth limit calculation."""

    def setUp(self):
        from kd_estimation import (
            calculate_depth_limits,
            estimate_kd_from_reflectance,
            classify_water_type,
            estimate_pixel_depth_confidence,
            DEPTH_FACTOR_CONSERVATIVE,
            DEPTH_FACTOR_MODERATE,
            DEPTH_FACTOR_OPTIMISTIC,
        )
        self.calc = calculate_depth_limits
        self.est_kd = estimate_kd_from_reflectance
        self.classify = classify_water_type
        self.confidence = estimate_pixel_depth_confidence
        self.F_cons = DEPTH_FACTOR_CONSERVATIVE
        self.F_mod = DEPTH_FACTOR_MODERATE
        self.F_opt = DEPTH_FACTOR_OPTIMISTIC

    def test_clear_water_has_deeper_limit(self):
        """Clearer water (low Kd) → deeper SDB depth limit."""
        limits_clear = self.calc(kd_median=0.05)
        limits_turbid = self.calc(kd_median=0.5)
        self.assertGreater(limits_clear["moderate"], limits_turbid["moderate"])

    def test_conservative_less_than_moderate_less_than_optimistic(self):
        limits = self.calc(kd_median=0.1)
        self.assertLess(limits["conservative"], limits["moderate"])
        self.assertLess(limits["moderate"], limits["optimistic"])

    def test_formula_conservative(self):
        kd = 0.1
        limits = self.calc(kd)
        expected = self.F_cons / kd
        self.assertAlmostEqual(limits["conservative"], round(expected, 1), places=0)

    def test_formula_moderate(self):
        kd = 0.2
        limits = self.calc(kd)
        expected = self.F_mod / kd
        self.assertAlmostEqual(limits["moderate"], round(expected, 1), places=0)

    def test_max_cap_50m(self):
        """Even in crystal-clear water, all limits cap at 50m."""
        limits = self.calc(kd_median=0.001)
        for key, val in limits.items():
            self.assertLessEqual(val, 50.0, f"{key} exceeded 50m cap: {val}")

    def test_zero_kd_doesnt_crash(self):
        """kd_median=0 should not raise ZeroDivisionError."""
        try:
            limits = self.calc(kd_median=0.0)
            for val in limits.values():
                self.assertLessEqual(val, 50.0)
        except ZeroDivisionError:
            self.fail("calculate_depth_limits raised ZeroDivisionError with kd=0")

    def test_clear_water_optional_arg_deeper(self):
        limits = self.calc(kd_median=0.1, kd_clear=0.05)
        self.assertIn("clear_water_max", limits)
        self.assertGreater(limits["clear_water_max"], limits["optimistic"])

    def test_turbid_water_optional_arg_shallower(self):
        limits = self.calc(kd_median=0.1, kd_turbid=0.3)
        self.assertIn("turbid_water_max", limits)
        self.assertLess(limits["turbid_water_max"], limits["conservative"])

    def test_all_values_positive(self):
        limits = self.calc(kd_median=0.15, kd_clear=0.08, kd_turbid=0.4)
        for key, val in limits.items():
            self.assertGreater(val, 0.0, f"{key} is not positive: {val}")

    def test_kd_from_reflectance_positive(self):
        kd = self.est_kd(rrs_blue=0.03, rrs_green=0.01, rrs_red=0.005)
        self.assertGreater(float(kd), 0.0)

    def test_kd_turbid_water_higher(self):
        """More turbid water (lower blue/green ratio) → higher Kd."""
        kd_clear = self.est_kd(rrs_blue=0.05, rrs_green=0.01, rrs_red=0.002)
        kd_turbid = self.est_kd(rrs_blue=0.01, rrs_green=0.03, rrs_red=0.02)
        self.assertGreater(float(kd_turbid), float(kd_clear))

    def test_classify_returns_nonempty_string(self):
        label = self.classify(kd_490=0.05)
        self.assertIsInstance(label, str)
        self.assertGreater(len(label), 0)

    def test_classify_clear_ocean(self):
        label = self.classify(kd_490=0.03)
        self.assertIn("clear", label.lower())

    def test_classify_turbid(self):
        label = self.classify(kd_490=1.5)
        self.assertIn("turbid", label.lower())

    def test_pixel_confidence_deeper_is_lower(self):
        """Pixel near optical limit should have lower confidence than shallow pixel."""
        kd_map = np.full((10, 10), 0.1, dtype=np.float32)
        depth_shallow = np.full((10, 10), 2.0, dtype=np.float32)
        depth_deep = np.full((10, 10), 20.0, dtype=np.float32)
        conf_shallow = self.confidence(kd_map, depth_shallow)
        conf_deep = self.confidence(kd_map, depth_deep)
        self.assertGreater(float(np.nanmean(conf_shallow)), float(np.nanmean(conf_deep)))

    def test_pixel_confidence_in_range(self):
        kd_map = np.full((5, 5), 0.15)
        depth_map = np.full((5, 5), 5.0)
        conf = self.confidence(kd_map, depth_map)
        self.assertTrue(np.all(conf >= 0.0))
        self.assertTrue(np.all(conf <= 1.0))


# ---------------------------------------------------------------------------
# Cross-Section Curvature Asymmetry
# (Pure numpy implementation, no geopandas needed)
# ---------------------------------------------------------------------------

def _local_quad_derivatives(s, v, i, half_window_m, min_pts):
    """Local quadratic fit to estimate first/second derivatives (copied from xs_infer_bathy_raster)."""
    n = int(len(s))
    if n < max(3, int(min_pts)):
        return (float("nan"), float("nan"))
    s0 = float(s[i])
    mask = np.isfinite(s) & np.isfinite(v)
    if not mask.any():
        return (float("nan"), float("nan"))
    idx = np.where(mask & (np.abs(s - s0) <= float(half_window_m)))[0]
    if idx.size < int(min_pts):
        good = np.where(mask)[0]
        if good.size < int(min_pts):
            return (float("nan"), float("nan"))
        order = np.argsort(np.abs(s[good] - s0))
        idx = good[order[: int(min_pts)]]
    ss = (s[idx] - s0).astype("float64")
    vv = v[idx].astype("float64")
    A = np.vstack([np.ones_like(ss), ss, ss ** 2]).T
    try:
        coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
    except Exception:
        return (float("nan"), float("nan"))
    return float(coef[1]), float(2.0 * coef[2])


def _compute_signed_curvature(s, x, y, half_window_m, min_pts):
    """Signed planform curvature (1/m). Positive = left turn."""
    n = int(len(s))
    kappa = np.full((n,), np.nan, dtype="float64")
    if n < max(5, int(min_pts)):
        return kappa
    order = np.argsort(s)
    s = s[order].astype("float64")
    x = x[order].astype("float64")
    y = y[order].astype("float64")
    for ii in range(n):
        dx, ddx = _local_quad_derivatives(s, x, ii, half_window_m, min_pts)
        dy, ddy = _local_quad_derivatives(s, y, ii, half_window_m, min_pts)
        if not all(np.isfinite(v) for v in [dx, dy, ddx, ddy]):
            continue
        denom = (dx * dx + dy * dy) ** 1.5
        if not np.isfinite(denom) or denom <= 0:
            continue
        kappa[ii] = (dx * ddy - dy * ddx) / denom
    out = np.full((n,), np.nan, dtype="float64")
    out[order] = kappa
    return out


class TestCurvatureAsymmetry(unittest.TestCase):
    """Tests for signed planform curvature computation."""

    def _straight_reach(self, n=50, length_m=1000.0):
        s = np.linspace(0, length_m, n)
        x = np.linspace(0, length_m, n)
        y = np.zeros(n)
        return s, x, y

    def _circular_arc(self, n=80, radius_m=200.0, arc_deg=90.0):
        angles = np.linspace(0, np.radians(arc_deg), n)
        x = radius_m * np.cos(angles)
        y = radius_m * np.sin(angles)
        s = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
        return s, x, y

    def test_straight_reach_near_zero_curvature(self):
        """Straight reach → curvature should be ~0 everywhere."""
        s, x, y = self._straight_reach()
        kappa = _compute_signed_curvature(s, x, y, half_window_m=100.0, min_pts=5)
        valid = kappa[np.isfinite(kappa)]
        self.assertGreater(len(valid), 0)
        np.testing.assert_allclose(valid, 0.0, atol=1e-6)

    def test_circular_arc_curvature_matches_1_over_r(self):
        """Circular arc → curvature should be ~1/R."""
        R = 300.0
        s, x, y = self._circular_arc(n=100, radius_m=R, arc_deg=60.0)
        kappa = _compute_signed_curvature(s, x, y, half_window_m=80.0, min_pts=5)
        valid = kappa[np.isfinite(kappa)]
        self.assertGreater(len(valid), 10)
        np.testing.assert_allclose(np.abs(valid), 1.0 / R, rtol=0.20)

    def test_curvature_sign_left_turn_positive(self):
        """Counterclockwise (left-turning) arc → positive curvature."""
        s, x, y = self._circular_arc(n=80, radius_m=200.0, arc_deg=90.0)
        kappa = _compute_signed_curvature(s, x, y, half_window_m=60.0, min_pts=5)
        valid = kappa[np.isfinite(kappa)]
        self.assertGreater(np.median(valid), 0.0)

    def test_curvature_sign_right_turn_negative(self):
        """Clockwise (right-turning) arc → negative curvature."""
        s, x, y = self._circular_arc(n=80, radius_m=200.0, arc_deg=90.0)
        kappa_ccw = _compute_signed_curvature(s, x, y, half_window_m=60.0, min_pts=5)
        kappa_cw = _compute_signed_curvature(s, -x, y, half_window_m=60.0, min_pts=5)
        self.assertGreater(np.nanmedian(kappa_ccw), 0.0)
        self.assertLess(np.nanmedian(kappa_cw), 0.0)

    def test_larger_radius_smaller_curvature(self):
        """Larger bend radius → smaller curvature magnitude."""
        s1, x1, y1 = self._circular_arc(n=100, radius_m=100.0, arc_deg=60.0)
        s2, x2, y2 = self._circular_arc(n=100, radius_m=500.0, arc_deg=60.0)
        k1 = _compute_signed_curvature(s1, x1, y1, half_window_m=30.0, min_pts=5)
        k2 = _compute_signed_curvature(s2, x2, y2, half_window_m=80.0, min_pts=5)
        self.assertGreater(np.nanmedian(np.abs(k1)), np.nanmedian(np.abs(k2)))

    def test_too_few_points_returns_all_nan(self):
        """Fewer points than min_pts → all-nan output."""
        s = np.array([0.0, 10.0, 20.0])
        x = np.array([0.0, 10.0, 20.0])
        y = np.zeros(3)
        kappa = _compute_signed_curvature(s, x, y, half_window_m=100.0, min_pts=10)
        self.assertTrue(np.all(np.isnan(kappa)))

    def test_output_length_matches_input(self):
        s, x, y = self._straight_reach(n=30)
        kappa = _compute_signed_curvature(s, x, y, half_window_m=100.0, min_pts=3)
        self.assertEqual(len(kappa), 30)

    def test_unsorted_input_same_result(self):
        """Result should be identical regardless of input ordering."""
        s, x, y = self._circular_arc(n=60, radius_m=200.0, arc_deg=60.0)
        kappa_sorted = _compute_signed_curvature(s, x, y, half_window_m=50.0, min_pts=5)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(s))
        kappa_perm = _compute_signed_curvature(s[perm], x[perm], y[perm], half_window_m=50.0, min_pts=5)
        inv_perm = np.argsort(perm)
        kappa_restored = kappa_perm[inv_perm]
        valid = np.isfinite(kappa_sorted) & np.isfinite(kappa_restored)
        if np.any(valid):
            np.testing.assert_allclose(kappa_sorted[valid], kappa_restored[valid], rtol=0.01)

    def test_sinusoidal_reach_alternates_sign(self):
        """Sinusoidal reach has alternating positive/negative curvature."""
        n = 200
        t = np.linspace(0, 4 * np.pi, n)
        x = t * 100.0
        y = 50.0 * np.sin(t)
        s = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
        kappa = _compute_signed_curvature(s, x, y, half_window_m=150.0, min_pts=5)
        valid = kappa[np.isfinite(kappa)]
        self.assertGreater(np.sum(valid > 0.001), 10)
        self.assertGreater(np.sum(valid < -0.001), 10)


if __name__ == "__main__":
    unittest.main()
