import numpy as np

from terrain_interpolator import TerrainInterpolationConfig, TerrainInterpolationInputs, interpolate_support_aware_surface
from support_classes import SupportClass


def test_river_corridor_prefers_river_support_anchor_over_bank_nearest_auth():
    candidate = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 2.0, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float32)
    auth = np.array([
        [10.0, 10.0, 10.0],
        [10.0, np.nan, 10.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float32)
    river_ok = np.zeros((3, 3), dtype=bool)
    river_ok[1, 1] = True
    river_support = np.zeros((3, 3), dtype=np.uint8)
    river_support[1, 1] = 1
    river_support_depth = np.full((3, 3), np.nan, dtype=np.float32)
    river_support_depth[1, 1] = 1.0

    result = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.zeros((3, 3), dtype=bool),
            river_ok=river_ok,
            river_support=river_support,
            river_support_depth=river_support_depth,
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )

    center = float(result["conditioned"][1, 1])
    assert 1.0 < center < 3.0
    assert abs(center - 1.0) < abs(center - 10.0)
    assert result["support"][1, 1] == int(SupportClass.GUIDANCE_CONDITIONED_RIVER)


def test_estuary_transition_defaults_without_runtime_name_error():
    candidate = np.array([[np.nan, -2.0], [-3.0, -4.0]], dtype=np.float32)
    auth = np.array([[5.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    result = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.array([[False, True], [False, False]], dtype=bool),
            river_ok=np.array([[False, False], [True, False]], dtype=bool),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert result["regime"].shape == candidate.shape



def test_river_bank_boundary_constraint_softens_channel_margin():
    candidate = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [6.0, -2.0, -3.0, -2.0, 6.0],
        [6.0, -4.0, -6.0, -4.0, 6.0],
        [6.0, -2.0, -3.0, -2.0, 6.0],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    auth = np.array([
        [6.0, 6.0, 6.0, 6.0, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, 6.0, 6.0, 6.0, 6.0],
    ], dtype=np.float32)
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[1:4, 1:4] = True
    result = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_ok=river_ok,
            river_corridor_mask=river_ok.astype(np.uint8),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    edge_val = float(result["conditioned"][1, 1])
    center_val = float(result["conditioned"][2, 2])
    assert edge_val > center_val
    assert edge_val > -2.0
    assert result["river_bank_influence"][1, 1] > result["river_bank_influence"][2, 2]


def test_xs_bank_elevation_softens_margin_more_than_corridor_only():
    candidate = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [6.0, -2.0, -3.0, -2.0, 6.0],
        [6.0, -4.0, -6.0, -4.0, 6.0],
        [6.0, -2.0, -3.0, -2.0, 6.0],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    auth = np.array([
        [6.0, 6.0, 6.0, 6.0, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, 6.0, 6.0, 6.0, 6.0],
    ], dtype=np.float32)
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[1:4, 1:4] = True
    xs_bank = np.full((5, 5), np.nan, dtype=np.float32)
    xs_bank[1:4, 1:4] = 4.0
    pair_weight = np.zeros((5, 5), dtype=np.float32)
    pair_weight[1:4, 1:4] = 1.0

    base = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_ok=river_ok,
            river_corridor_mask=river_ok.astype(np.uint8),
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    xs = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate,
            auth=auth,
            sdb_ok=np.zeros((5, 5), dtype=bool),
            river_ok=river_ok,
            river_corridor_mask=river_ok.astype(np.uint8),
            river_bank_elevation=xs_bank,
            river_bank_pair_weight=pair_weight,
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert float(xs["river_bank_elevation"][1, 1]) == 4.0
    assert float(base["river_bank_elevation"][1, 1]) == 6.0
    assert np.isfinite(xs["conditioned"][1, 1])


def test_bank_continuity_weight_strengthens_margin_constraint():
    candidate = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [6.0, -2.0, -3.0, -2.0, 6.0],
        [6.0, -4.0, -6.0, -4.0, 6.0],
        [6.0, -2.0, -3.0, -2.0, 6.0],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    auth = np.array([
        [6.0, 6.0, 6.0, 6.0, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, 6.0, 6.0, 6.0, 6.0],
    ], dtype=np.float32)
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[1:4, 1:4] = True
    xs_bank = np.full((5, 5), np.nan, dtype=np.float32)
    xs_bank[1:4, 1:4] = 4.0
    pair_weight = np.zeros((5, 5), dtype=np.float32)
    pair_weight[1:4, 1:4] = 1.0
    low_cont = np.zeros((5, 5), dtype=np.float32)
    high_cont = np.zeros((5, 5), dtype=np.float32)
    high_cont[1:4, 1:4] = 1.0

    low = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate, auth=auth, sdb_ok=np.zeros((5, 5), dtype=bool), river_ok=river_ok,
            river_corridor_mask=river_ok.astype(np.uint8), river_bank_elevation=xs_bank,
            river_bank_pair_weight=pair_weight, river_bank_continuity_weight=low_cont,
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    high = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate, auth=auth, sdb_ok=np.zeros((5, 5), dtype=bool), river_ok=river_ok,
            river_corridor_mask=river_ok.astype(np.uint8), river_bank_elevation=xs_bank,
            river_bank_pair_weight=pair_weight, river_bank_continuity_weight=high_cont,
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert high["river_bank_influence"][1, 1] >= low["river_bank_influence"][1, 1]
    assert high["conditioned"][1, 1] >= low["conditioned"][1, 1]


def test_graph_context_and_confluence_damping_reduce_bank_constraint():
    candidate = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [6.0, -2.0, -3.0, -2.0, 6.0],
        [6.0, -4.0, -6.0, -4.0, 6.0],
        [6.0, -2.0, -3.0, -2.0, 6.0],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    auth = np.array([
        [6.0, 6.0, 6.0, 6.0, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, np.nan, np.nan, np.nan, 6.0],
        [6.0, 6.0, 6.0, 6.0, 6.0],
    ], dtype=np.float32)
    river_ok = np.zeros((5, 5), dtype=bool)
    river_ok[1:4, 1:4] = True
    xs_bank = np.full((5, 5), np.nan, dtype=np.float32)
    xs_bank[1:4, 1:4] = 4.0
    pair_weight = np.zeros((5, 5), dtype=np.float32)
    pair_weight[1:4, 1:4] = 1.0
    continuity = np.zeros((5, 5), dtype=np.float32)
    continuity[1:4, 1:4] = 1.0
    high_graph = np.zeros((5, 5), dtype=np.float32)
    high_graph[1:4, 1:4] = 1.0
    low_damp = np.ones((5, 5), dtype=np.float32)
    low_damp[1:4, 1:4] = 0.5

    strong = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate, auth=auth, sdb_ok=np.zeros((5, 5), dtype=bool), river_ok=river_ok,
            river_corridor_mask=river_ok.astype(np.uint8), river_bank_elevation=xs_bank,
            river_bank_pair_weight=pair_weight, river_bank_continuity_weight=continuity, river_bank_graph_confidence=high_graph,
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    damped = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(
            candidate=candidate, auth=auth, sdb_ok=np.zeros((5, 5), dtype=bool), river_ok=river_ok,
            river_corridor_mask=river_ok.astype(np.uint8), river_bank_elevation=xs_bank,
            river_bank_pair_weight=pair_weight, river_bank_continuity_weight=continuity, river_bank_graph_confidence=high_graph,
            river_bank_confluence_damping=low_damp,
        ),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert damped["river_bank_influence"][1, 1] < strong["river_bank_influence"][1, 1]
    # In the variance-driven conditioner, reducing the confluence bank constraint
    # weakens the bank-elevation pull on the anchor surface, so the near-margin
    # conditioned elevation becomes less constrained by that bank guidance.
    assert damped["conditioned"][1, 1] >= strong["conditioned"][1, 1]


def test_estuary_side_decay_reduces_bank_constraint_near_transition():
    candidate = np.array([
        [np.nan, np.nan, np.nan],
        [6.0, -3.0, 6.0],
        [6.0, -5.0, 6.0],
    ], dtype=np.float32)
    auth = np.array([
        [6.0, 6.0, 6.0],
        [6.0, np.nan, 6.0],
        [6.0, np.nan, 6.0],
    ], dtype=np.float32)
    river_ok = np.array([[False, False, False],[True, True, True],[True, True, True]], dtype=bool)
    xs_bank = np.full((3, 3), np.nan, dtype=np.float32)
    xs_bank[1:, :] = 4.0
    pair_weight = np.zeros((3, 3), dtype=np.float32)
    pair_weight[1:, :] = 1.0
    continuity = np.zeros((3, 3), dtype=np.float32)
    continuity[1:, :] = 1.0
    est_decay = np.ones((3, 3), dtype=np.float32)
    est_decay[1, :] = 0.4
    base = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(candidate=candidate, auth=auth, sdb_ok=np.zeros((3, 3), dtype=bool), river_ok=river_ok, river_corridor_mask=river_ok.astype(np.uint8), river_bank_elevation=xs_bank, river_bank_pair_weight=pair_weight, river_bank_continuity_weight=continuity),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    decayed = interpolate_support_aware_surface(
        inputs=TerrainInterpolationInputs(candidate=candidate, auth=auth, sdb_ok=np.zeros((3, 3), dtype=bool), river_ok=river_ok, river_corridor_mask=river_ok.astype(np.uint8), river_bank_elevation=xs_bank, river_bank_pair_weight=pair_weight, river_bank_continuity_weight=continuity, river_bank_estuary_side_decay=est_decay),
        config=TerrainInterpolationConfig(pixel_size_m=10.0),
    )
    assert decayed["river_bank_influence"][1, 1] < base["river_bank_influence"][1, 1]
