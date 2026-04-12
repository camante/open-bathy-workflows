from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from river_v2_context import RiverV2Context
from river_v2_stage_wse_proxy import (
    build_wse_pre_smooth,
    build_wse_proxy_final,
    build_wse_support,
    build_wse_trend,
)


class _Cfg:
    pass


def _build_context(tmp_path: Path) -> RiverV2Context:
    return RiverV2Context(
        cfg=_Cfg(),
        report={},
        out_dir=tmp_path,
        network_gpkg=tmp_path / "network.gpkg",
        river_dem_path=tmp_path / "river_dem.tif",
        centerline_points_gdf=None,
        vertical_reference="NAVD88",
    )


def _centerline(values: list[float | None]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "point_id": [f"p{i}" for i in range(len(values))],
            "station_m": [float(i) for i in range(len(values))],
            "component_id": ["main"] * len(values),
            "bank_wse_proxy_monotone_m": values,
        },
        geometry=[Point(float(i), 0.0) for i in range(len(values))],
        crs="EPSG:4326",
    )


def test_build_wse_support_keeps_gaps_unfilled(tmp_path: Path):
    ctx = _build_context(tmp_path)
    centerline = _centerline([1.0, None, 2.0, None])
    support_gdf, _, summary = build_wse_support(ctx, centerline, {})
    support = support_gdf["wse_support_z_m"].to_numpy(dtype=float)
    has_support = support_gdf["has_support"].to_numpy(dtype=bool)
    assert np.isfinite(support[0])
    assert not np.isfinite(support[1])
    assert np.isfinite(support[2])
    assert not np.isfinite(support[3])
    assert has_support.tolist() == [True, False, True, False]
    assert summary["supported_station_count"] == 2


def test_build_wse_trend_bridges_sparse_support_with_monotone_profile(tmp_path: Path):
    ctx = _build_context(tmp_path)
    centerline = _centerline([1.0, None, None, 1.6, 1.9])
    support_gdf, _, _ = build_wse_support(ctx, centerline, {})
    trend_gdf, _, summary = build_wse_trend(ctx, support_gdf)
    trend = trend_gdf["wse_trend_z_m"].to_numpy(dtype=float)
    assert np.isfinite(trend).all()
    assert np.all(np.diff(trend) >= -1.0e-9)
    assert summary["direction_violation_count"] == 0
    assert set(trend_gdf["station_direction"].astype(str)) == {"upstream_increasing"}


def test_build_wse_pre_smooth_only_nudges_when_support_is_near_trend(tmp_path: Path):
    ctx = _build_context(tmp_path)
    centerline = _centerline([1.0, None, 1.8, None, 2.0])
    support_gdf, _, _ = build_wse_support(ctx, centerline, {})
    trend_gdf, _, _ = build_wse_trend(ctx, support_gdf)
    near_idx = 2
    trend_gdf.loc[near_idx, "wse_support_z_m"] = trend_gdf.loc[near_idx, "wse_trend_z_m"] + 0.1
    trend_gdf.loc[near_idx, "support_type"] = "direct_local_support"
    far_idx = 4
    trend_gdf.loc[far_idx, "wse_support_z_m"] = trend_gdf.loc[far_idx, "wse_trend_z_m"] + 1.0
    trend_gdf.loc[far_idx, "support_type"] = "direct_local_support"
    pre_gdf, _, _ = build_wse_pre_smooth(ctx, trend_gdf)
    trend = trend_gdf["wse_trend_z_m"].to_numpy(dtype=float)
    pre = pre_gdf["wse_pre_smooth_z_m"].to_numpy(dtype=float)
    support = pre_gdf["wse_support_z_m"].to_numpy(dtype=float)
    assert pre[near_idx] > trend[near_idx]
    assert pre[near_idx] < support[near_idx]
    assert np.isclose(pre[near_idx] - trend[near_idx], 0.01)
    assert np.isclose(pre[far_idx], trend[far_idx])
    gap_idx = 1
    assert np.isclose(pre[gap_idx], trend[gap_idx])


def test_build_wse_proxy_final_cleans_small_reversal(tmp_path: Path):
    ctx = _build_context(tmp_path)
    centerline = _centerline([1.0, None, 1.8, None, 2.0])
    support_gdf, _, _ = build_wse_support(ctx, centerline, {})
    trend_gdf, _, _ = build_wse_trend(ctx, support_gdf)
    pre_gdf, _, _ = build_wse_pre_smooth(ctx, trend_gdf)
    pre_gdf.loc[2, "wse_pre_smooth_z_m"] = pre_gdf.loc[1, "wse_pre_smooth_z_m"] - 0.1
    final_gdf, _, summary = build_wse_proxy_final(ctx, pre_gdf)
    final = final_gdf["wse_proxy_z_m"].to_numpy(dtype=float)
    reasons = final_gdf["final_adjustment_reason"].astype(str).tolist()
    assert np.all(np.diff(final) >= -1.0e-9)
    assert summary["adjusted_station_count"] >= 1
    assert summary["direction_violation_count"] == 0
    assert summary["direction_violation_count"] == 0
    assert any(r in reasons for r in ["tiny_reversal_cleanup", "unsupported_span_bridge"])


def test_build_wse_proxy_final_bridges_long_unsupported_span_between_supports(tmp_path: Path):
    ctx = _build_context(tmp_path)
    centerline = _centerline([1.0, None, None, None, None, None, None, 1.3])
    centerline["station_m"] = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0]
    support_gdf, _, _ = build_wse_support(ctx, centerline, {})
    trend_gdf, _, _ = build_wse_trend(ctx, support_gdf)
    pre_gdf = trend_gdf.copy()
    pre_gdf["wse_pre_smooth_z_m"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3]
    pre_gdf["has_support"] = [True, False, False, False, False, False, False, True]
    final_gdf, _, summary = build_wse_proxy_final(ctx, pre_gdf)
    final = final_gdf["wse_proxy_z_m"].to_numpy(dtype=float)
    reasons = final_gdf["final_adjustment_reason"].astype(str).tolist()
    expected = np.interp([150.0], [0.0, 175.0], [1.0, 1.3])[0]
    assert np.all(np.diff(final) >= -1.0e-9)
    assert np.isclose(final[6], expected)
    assert "unsupported_span_bridge" in reasons
    assert summary["adjusted_station_count"] >= 1


def test_build_wse_trend_resolves_downstream_increasing_direction_from_support(tmp_path: Path):
    ctx = _build_context(tmp_path)
    centerline = _centerline([2.0, None, None, 1.6, 1.3])
    support_gdf, _, _ = build_wse_support(ctx, centerline, {})
    trend_gdf, _, _ = build_wse_trend(ctx, support_gdf)
    trend = trend_gdf["wse_trend_z_m"].to_numpy(dtype=float)
    assert np.isfinite(trend).all()
    assert np.all(np.diff(trend) <= 1.0e-9)
    assert set(trend_gdf["station_direction"].astype(str)) == {"downstream_increasing"}


def test_build_wse_trend_reduces_blocky_support_flats(tmp_path: Path):
    ctx = _build_context(tmp_path)
    centerline = _centerline([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
    support_gdf, _, _ = build_wse_support(ctx, centerline, {})
    trend_gdf, _, summary = build_wse_trend(ctx, support_gdf)
    trend = trend_gdf["wse_trend_z_m"].to_numpy(dtype=float)
    assert np.isfinite(trend).all()
    assert len(np.unique(np.round(trend, 6))) > 3
    assert summary["direction_violation_count"] == 0


def test_build_wse_trend_collapses_long_identical_support_runs_to_midpoint_anchors(tmp_path: Path):
    ctx = _build_context(tmp_path)
    centerline = _centerline([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    support_gdf, _, _ = build_wse_support(ctx, centerline, {})
    trend_gdf, _, summary = build_wse_trend(ctx, support_gdf)
    trend = trend_gdf["wse_trend_z_m"].to_numpy(dtype=float)
    assert np.isfinite(trend).all()
    assert summary["direction_violation_count"] == 0
    longest_flat = 1
    current = 1
    for i in range(1, len(trend)):
        if np.isclose(trend[i], trend[i-1]):
            current += 1
            longest_flat = max(longest_flat, current)
        else:
            current = 1
    assert longest_flat < 4


def test_build_wse_trend_writes_sparse_anchor_and_interp_fields(tmp_path: Path):
    ctx = _build_context(tmp_path)
    centerline = _centerline([1.0, 1.0, 1.0, None, 2.0, 2.0, 3.0])
    support_gdf, _, _ = build_wse_support(ctx, centerline, {})
    trend_gdf, _, summary = build_wse_trend(ctx, support_gdf)
    assert "wse_anchor_raw_z_m" in trend_gdf.columns
    assert "wse_interp_z_m" in trend_gdf.columns
    anchors = trend_gdf["wse_anchor_raw_z_m"].to_numpy(dtype=float)
    interp = trend_gdf["wse_interp_z_m"].to_numpy(dtype=float)
    assert np.count_nonzero(np.isfinite(anchors)) < np.count_nonzero(np.isfinite(support_gdf["wse_support_z_m"].to_numpy(dtype=float)))
    assert np.isfinite(interp).all()
    assert summary["sparse_anchor_count"] >= 3
    assert summary["collapsed_support_run_count"] >= 2
