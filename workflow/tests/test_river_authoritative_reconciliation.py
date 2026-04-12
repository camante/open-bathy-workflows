from __future__ import annotations

import numpy as np

from river_authoritative_reconciliation import solve_authoritative_reconciliation_field


def test_reconciliation_field_carries_guidance_through_sparse_gap():
    stations = np.array([0.0, 100.0, 200.0, 300.0, 400.0], dtype=float)
    support_distance = np.array([0.0, 150.0, 450.0, 150.0, 0.0], dtype=float)
    observations = {
        "station_m": [0.0, 400.0],
        "residual_delta_m": [-0.8, -0.8],
        "observation_confidence": [1.0, 1.0],
    }
    field = solve_authoritative_reconciliation_field(
        stations=stations,
        observations=__import__('pandas').DataFrame(observations),
        support_distance_m=support_distance,
        fluvial_mask=np.ones(stations.size, dtype=bool),
        exact_anchor_mask=np.zeros(stations.size, dtype=bool),
        taper_radius_m=250.0,
        max_abs_delta_m=1.5,
    )
    mid_weight = float(field.loc[2, "authoritative_reconciliation_weight"])
    mid_delta = float(field.loc[2, "authoritative_reconciliation_delta_m"])
    assert mid_weight > 0.0
    assert mid_delta < 0.0


def test_reconciliation_field_respects_exact_anchor_zero_delta():
    stations = np.array([0.0, 100.0, 200.0], dtype=float)
    support_distance = np.array([0.0, 50.0, 0.0], dtype=float)
    observations = {
        "station_m": [100.0],
        "residual_delta_m": [-0.5],
        "observation_confidence": [1.0],
    }
    exact_anchor = np.array([False, True, False], dtype=bool)
    field = solve_authoritative_reconciliation_field(
        stations=stations,
        observations=__import__('pandas').DataFrame(observations),
        support_distance_m=support_distance,
        fluvial_mask=np.ones(stations.size, dtype=bool),
        exact_anchor_mask=exact_anchor,
        taper_radius_m=250.0,
        max_abs_delta_m=1.5,
    )
    assert float(field.loc[1, "authoritative_reconciliation_delta_m"]) == 0.0
    assert float(field.loc[1, "authoritative_reconciliation_weight"]) == 0.0


def test_reconciliation_field_keeps_more_mainstem_guidance_than_side_component_in_sparse_gap():
    stations = np.array([0.0, 100.0, 200.0, 300.0, 400.0], dtype=float)
    support_distance = np.array([0.0, 175.0, 525.0, 175.0, 0.0], dtype=float)
    observations = {
        "station_m": [0.0, 400.0],
        "residual_delta_m": [-0.8, -0.8],
        "observation_confidence": [1.0, 1.0],
    }
    import pandas as pd
    obs_df = pd.DataFrame(observations)
    mainstem_field = solve_authoritative_reconciliation_field(
        stations=stations,
        observations=obs_df,
        support_distance_m=support_distance,
        fluvial_mask=np.ones(stations.size, dtype=bool),
        exact_anchor_mask=np.zeros(stations.size, dtype=bool),
        taper_radius_m=250.0,
        max_abs_delta_m=1.5,
        component_class="unsupported_mainstem",
    )
    side_field = solve_authoritative_reconciliation_field(
        stations=stations,
        observations=obs_df,
        support_distance_m=support_distance,
        fluvial_mask=np.ones(stations.size, dtype=bool),
        exact_anchor_mask=np.zeros(stations.size, dtype=bool),
        taper_radius_m=250.0,
        max_abs_delta_m=1.5,
        component_class="unsupported_side_component",
    )
    assert float(mainstem_field.loc[2, "authoritative_reconciliation_weight"]) > float(side_field.loc[2, "authoritative_reconciliation_weight"])
    assert abs(float(mainstem_field.loc[2, "authoritative_reconciliation_delta_m"])) >= abs(float(side_field.loc[2, "authoritative_reconciliation_delta_m"]))
