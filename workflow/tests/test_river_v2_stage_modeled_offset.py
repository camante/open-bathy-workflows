from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from river_v2_stage_modeled_offset import build_modeled_offsets


def _joined_points() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "point_id": ["a0", "a1", "a2", "b0", "b1", "b2"],
            "component_id": ["A", "A", "A", "B", "B", "B"],
            "station_m": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            "observed_offset_m": [0.8, np.nan, 1.0, np.nan, np.nan, np.nan],
        },
        geometry=[Point(0.0, 0.0), Point(1.0, 0.0), Point(2.0, 0.0), Point(0.0, 1.0), Point(1.0, 1.0), Point(2.0, 1.0)],
        crs="EPSG:4326",
    )


def test_build_modeled_offsets_is_component_local():
    joined = _joined_points()
    modeled, summary, warnings, diagnostics = build_modeled_offsets(joined)
    assert diagnostics["group_count"] == 2
    a = modeled[modeled["component_id"] == "A"].sort_values("station_m", kind="mergesort")
    b = modeled[modeled["component_id"] == "B"].sort_values("station_m", kind="mergesort")
    a_vals = a["offset_modeled_m"].to_numpy(dtype=float)
    b_vals = b["offset_modeled_m"].to_numpy(dtype=float)
    assert np.all(a_vals < 2.0)
    assert np.isfinite(b_vals).all()
    assert set(b["offset_source"].astype(str)) == {"nhdplus_placeholder"}
    assert set(b["offset_support_class"].astype(str)) == {"inferred"}
    assert "modeled_offset_component_has_no_anchor_support" in warnings


def test_build_modeled_offsets_uses_min_fallback_only_when_component_has_no_offset_information():
    joined = _joined_points()
    joined.loc[joined["component_id"] == "B", "observed_offset_m"] = [np.nan, np.nan, np.nan]
    modeled, summary, warnings, diagnostics = build_modeled_offsets(joined)
    b = modeled[modeled["component_id"] == "B"].sort_values("station_m", kind="mergesort")
    b_vals = b["offset_modeled_m"].to_numpy(dtype=float)
    assert np.isfinite(b_vals).all()
    assert set(b["offset_source"].astype(str)) == {"nhdplus_placeholder"}
    assert set(b["offset_support_class"].astype(str)) == {"inferred"}
    assert "modeled_offset_component_has_no_anchor_support" in warnings
    assert diagnostics["record_count"] == 6


def test_build_modeled_offsets_uses_observed_only_interpolation_without_group_median_fallback():
    joined = _joined_points()
    modeled, summary, warnings, diagnostics = build_modeled_offsets(joined)
    a = modeled[modeled["component_id"] == "A"].sort_values("station_m", kind="mergesort")
    a_vals = a["offset_modeled_m"].to_numpy(dtype=float)
    assert np.all(np.isfinite(a_vals))
    assert np.all(a_vals >= 0.8) and np.all(a_vals <= 1.0)
    assert set(a["offset_support_class"].astype(str)) <= {"observed_anchor", "inferred"}
    assert all("group_observed_median_fallback" not in w for w in warnings)
    assert diagnostics["record_count"] == 6


def test_build_modeled_offsets_linearly_interpolates_between_observed_anchors():
    joined = gpd.GeoDataFrame(
        {
            "point_id": ["a0", "a1", "a2", "a3", "a4"],
            "component_id": ["A"] * 5,
            "station_m": [0.0, 10.0, 20.0, 30.0, 40.0],
            "observed_offset_m": [1.0, np.nan, np.nan, np.nan, 3.0],
        },
        geometry=[Point(float(i), 0.0) for i in range(5)],
        crs="EPSG:4326",
    )
    modeled, summary, warnings, diagnostics = build_modeled_offsets(joined)
    vals = modeled.sort_values("station_m", kind="mergesort")["offset_modeled_m"].to_numpy(dtype=float)
    assert np.allclose(vals, [1.0, 1.5, 2.0, 2.5, 3.0])
    assert diagnostics["record_count"] == 5


def test_build_modeled_offsets_continues_long_end_spans_without_constant_shelves():
    joined = gpd.GeoDataFrame(
        {
            "point_id": ["a0", "a1", "a2", "a3", "a4"],
            "component_id": ["A"] * 5,
            "station_m": [0.0, 10.0, 20.0, 30.0, 40.0],
            "observed_offset_m": [np.nan, np.nan, 2.0, 3.0, np.nan],
        },
        geometry=[Point(float(i), 0.0) for i in range(5)],
        crs="EPSG:4326",
    )
    modeled, summary, warnings, diagnostics = build_modeled_offsets(joined)
    ordered = modeled.sort_values("station_m", kind="mergesort")
    vals = ordered["offset_modeled_m"].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals))
    assert np.all(np.isfinite(vals))
    assert vals[2] == 2.0
    assert vals[3] == 3.0
    assert vals[0] != vals[1]
    assert vals[4] != vals[3]
    assert max(vals) <= 3.01
    assert min(vals) >= 0.1 - 1.0e-9
    support = ordered["offset_support_class"].astype(str).tolist()
    assert support[0] == "inferred"
    assert support[-1] == "inferred"
    assert diagnostics["record_count"] == 5


def test_build_modeled_offsets_single_anchor_hold_fills_full_component():
    joined = gpd.GeoDataFrame(
        {
            "point_id": ["a0", "a1", "a2", "a3"],
            "component_id": ["A"] * 4,
            "station_m": [0.0, 400.0, 800.0, 1200.0],
            "observed_offset_m": [1.0, np.nan, np.nan, np.nan],
        },
        geometry=[Point(float(i), 0.0) for i in range(4)],
        crs="EPSG:4326",
    )
    modeled, summary, warnings, diagnostics = build_modeled_offsets(joined)
    vals = modeled.sort_values("station_m", kind="mergesort")["offset_modeled_m"].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals))
    assert np.allclose(vals, 1.0)


def test_build_modeled_offsets_long_end_continuation_fills_full_component_without_flat_tail():
    joined = gpd.GeoDataFrame(
        {
            "point_id": ["a0", "a1", "a2", "a3", "a4"],
            "component_id": ["A"] * 5,
            "station_m": [0.0, 1000.0, 2000.0, 5000.0, 7000.0],
            "observed_offset_m": [1.0, 2.0, np.nan, np.nan, np.nan],
        },
        geometry=[Point(float(i), 0.0) for i in range(5)],
        crs="EPSG:4326",
    )
    modeled, summary, warnings, diagnostics = build_modeled_offsets(joined)
    ordered = modeled.sort_values("station_m", kind="mergesort")
    vals = ordered["offset_modeled_m"].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals))
    assert vals[0] == 1.0
    assert vals[1] == 2.0
    assert np.all(np.isfinite(vals))
    assert vals[-1] != vals[-2]
    assert vals[-1] >= 0.1 - 1.0e-9


def test_build_modeled_offsets_fills_full_centerline_grid_when_only_one_component_has_observed_anchors():
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point
    main_station = np.arange(0.0, 250.0, 25.0)
    side_station = np.arange(0.0, 100.0, 25.0)
    joined = gpd.GeoDataFrame(
        {
            "point_id": [*(f"m{i}" for i in range(len(main_station))), *(f"s{i}" for i in range(len(side_station)))],
            "component_id": [*("MAIN" for _ in main_station), *("SIDE" for _ in side_station)],
            "station_m": [*main_station.tolist(), *side_station.tolist()],
            "observed_offset_m": [np.nan]*len(main_station) + [np.nan]*len(side_station),
        },
        geometry=[*(Point(float(i), 0.0) for i in range(len(main_station))), *(Point(float(i), 1.0) for i in range(len(side_station)))],
        crs="EPSG:4326",
    )
    joined.loc[joined['point_id'].isin(['m2','m4','m6']), 'observed_offset_m'] = [2.0, 2.5, 3.0]
    modeled, summary, warnings, diagnostics = build_modeled_offsets(joined)
    assert diagnostics['record_count'] == len(joined)
    assert diagnostics['finite_modeled_count'] == len(joined)
    side = modeled[modeled['component_id']=='SIDE'].sort_values('station_m', kind='mergesort')
    assert np.isfinite(side['offset_modeled_m'].to_numpy(dtype=float)).all()
    assert set(side['offset_source'].astype(str)) == {'nhdplus_placeholder'}
    assert set(side['offset_support_class'].astype(str)) == {'inferred'}
    main = modeled[modeled['component_id']=='MAIN'].sort_values('station_m', kind='mergesort')
    main_vals = main['offset_modeled_m'].to_numpy(dtype=float)
    assert np.isfinite(main_vals).all()
    longest_flat = 1
    current = 1
    for i in range(1, len(main_vals)):
        if np.isclose(main_vals[i], main_vals[i-1]):
            current += 1
            longest_flat = max(longest_flat, current)
        else:
            current = 1
    assert longest_flat < 3


def test_validate_modeled_offsets_requires_finite_modeled_offsets_for_all_components():
    joined = _joined_points()
    modeled, summary, warnings, diagnostics = build_modeled_offsets(joined)
    from river_v2_stage_modeled_offset import validate_modeled_offsets
    validation = validate_modeled_offsets(modeled)
    assert validation["valid"] is True
    assert validation["unsupported_component_count"] == 0
    assert validation["required_record_count"] == 6
    assert validation["required_finite_modeled_offset_count"] == 6
