import sys
import types

import numpy as np
import pandas as pd

import river_bank_guidance as rbg
from river_bank_guidance import (
    build_persistent_bank_network_points,
    compute_bank_distance_influence,
    compute_bank_elevation_surface_from_authoritative,
    compute_xs_bank_guidance_surfaces,
    compute_graph_informed_bank_context_surfaces,
)


class _Pt:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class _GeomSeries(pd.Series):
    @property
    def _constructor(self):
        return _GeomSeries

    @property
    def x(self):
        return pd.Series([g.x for g in self], index=self.index, dtype=float)

    @property
    def y(self):
        return pd.Series([g.y for g in self], index=self.index, dtype=float)


class _BankPoints(pd.DataFrame):
    @property
    def _constructor(self):
        return _BankPoints

    @property
    def geometry(self):
        return _GeomSeries(self["geometry"])

    @property
    def empty(self):
        return len(self.index) == 0


class _Transform:
    a = 10.0
    e = -10.0


def _install_fake_rasterio(monkeypatch):
    pkg = types.ModuleType("rasterio")
    transform_mod = types.ModuleType("rasterio.transform")

    def xy(transform, rows, cols, offset="center"):
        xs = [5.0 + 10.0 * float(c) for c in rows * 0 + cols]
        ys = [25.0 - 10.0 * float(r) for r in rows]
        return xs, ys

    transform_mod.xy = xy
    pkg.transform = transform_mod
    monkeypatch.setitem(sys.modules, "rasterio", pkg)
    monkeypatch.setitem(sys.modules, "rasterio.transform", transform_mod)


def test_bank_influence_tracks_corridor_edge():
    corridor = np.zeros((5, 5), dtype=bool)
    corridor[1:4, 1:4] = True
    edge, dist, infl = compute_bank_distance_influence(corridor, pixel_size_m=10.0, full_influence_m=5.0, zero_influence_m=25.0)
    assert edge[1, 1] == 1
    assert edge[2, 2] == 0
    assert infl[1, 1] > infl[2, 2]
    assert dist[2, 2] > dist[1, 1]


def test_bank_elevation_uses_authoritative_outside_corridor():
    auth = np.full((5, 5), np.nan, dtype=np.float32)
    auth[0, :] = 8.0
    auth[-1, :] = 9.0
    auth[:, 0] = 7.0
    auth[:, -1] = 6.0
    corridor = np.zeros((5, 5), dtype=bool)
    corridor[1:4, 1:4] = True
    _, dist, _ = compute_bank_distance_influence(corridor, pixel_size_m=10.0)
    surf = compute_bank_elevation_surface_from_authoritative(auth, corridor, max_bank_distance_m=50.0, bank_distance_m=dist)
    assert np.isfinite(surf[1, 1])
    assert np.isfinite(surf[2, 2])


def test_persistent_bank_network_smooths_longitudinal_outlier(monkeypatch):
    fake = _BankPoints([
        {"xs_id": "xs1", "river_id": "r1", "component_id": 1, "side": "left", "side_sign": -1.0, "bank_z_m": 10.0, "bank_z_raw_m": 10.0, "s_center_m": 0.0, "geometry": _Pt(5, 0)},
        {"xs_id": "xs2", "river_id": "r1", "component_id": 1, "side": "left", "side_sign": -1.0, "bank_z_m": 30.0, "bank_z_raw_m": 30.0, "s_center_m": 10.0, "geometry": _Pt(5, 10)},
        {"xs_id": "xs3", "river_id": "r1", "component_id": 1, "side": "left", "side_sign": -1.0, "bank_z_m": 10.0, "bank_z_raw_m": 10.0, "s_center_m": 20.0, "geometry": _Pt(5, 20)},
        {"xs_id": "xs1", "river_id": "r1", "component_id": 1, "side": "right", "side_sign": 1.0, "bank_z_m": 12.0, "bank_z_raw_m": 12.0, "s_center_m": 0.0, "geometry": _Pt(15, 0)},
    ])
    monkeypatch.setattr(rbg, "load_xs_bank_points", lambda *a, **k: fake)
    pts = build_persistent_bank_network_points("unused.gpkg", target_crs="EPSG:32619", smoothing_half_window=1)
    left = pts[pts["side"] == "left"].sort_values("s_center_m")
    assert left.iloc[1]["bank_z_raw_m"] == 30.0
    assert left.iloc[1]["bank_z_m"] < 30.0
    assert 0.0 <= float(left.iloc[1]["continuity_weight"]) <= 1.0


def test_xs_bank_guidance_surfaces_return_continuity_weight(monkeypatch):
    _install_fake_rasterio(monkeypatch)
    bank_points = _BankPoints([
        {"side": "left", "bank_z_m": 10.0, "continuity_weight": 0.8, "geometry": _Pt(5, 15)},
        {"side": "right", "bank_z_m": 12.0, "continuity_weight": 0.9, "geometry": _Pt(15, 15)},
    ])
    corridor = np.ones((3, 3), dtype=bool)
    surfaces = compute_xs_bank_guidance_surfaces(
        corridor_mask=corridor,
        transform=_Transform(),
        auth=np.full((3, 3), np.nan, dtype=np.float32),
        xs_gpkg="unused.gpkg",
        bank_points_gdf=bank_points,
    )
    assert "bank_continuity_weight" in surfaces
    assert np.nanmax(surfaces["bank_continuity_weight"]) > 0.0


def test_graph_informed_bank_context_surfaces_damp_near_estuary_and_confluence(monkeypatch):
    _install_fake_rasterio(monkeypatch)
    bank_points = _BankPoints([
        {"component_id": 1, "side_sign": -1.0, "continuity_weight": 1.0, "geometry": _Pt(5, 15)},
        {"component_id": 1, "side_sign": 1.0, "continuity_weight": 1.0, "geometry": _Pt(15, 15)},
        {"component_id": 2, "side_sign": -1.0, "continuity_weight": 0.9, "geometry": _Pt(25, 15)},
        {"component_id": 2, "side_sign": 1.0, "continuity_weight": 0.9, "geometry": _Pt(25, 5)},
    ])
    corridor = np.ones((3, 3), dtype=bool)
    estuary = np.zeros((3, 3), dtype=bool)
    estuary[2, 2] = True
    ctx = compute_graph_informed_bank_context_surfaces(
        corridor_mask=corridor,
        transform=_Transform(),
        bank_points_gdf=bank_points,
        estuary_transition=estuary,
        estuary_decay_distance_m=15.0,
    )
    assert np.nanmax(ctx["bank_graph_confidence"]) > 0.0
    assert np.nanmin(ctx["bank_confluence_damping"]) < 1.0
    assert ctx["bank_estuary_side_decay"][2, 2] < ctx["bank_estuary_side_decay"][0, 0]


def test_xs_bank_guidance_prefers_lower_bank_when_sides_are_strongly_asymmetric(monkeypatch):
    _install_fake_rasterio(monkeypatch)
    bank_points = _BankPoints([
        {"side": "left", "bank_z_m": 2.0, "continuity_weight": 1.0, "geometry": _Pt(5, 15)},
        {"side": "right", "bank_z_m": 8.0, "continuity_weight": 1.0, "geometry": _Pt(15, 15)},
    ])
    corridor = np.ones((3, 3), dtype=bool)
    surfaces = compute_xs_bank_guidance_surfaces(
        corridor_mask=corridor,
        transform=_Transform(),
        auth=np.full((3, 3), np.nan, dtype=np.float32),
        xs_gpkg="unused.gpkg",
        bank_points_gdf=bank_points,
    )
    center = float(surfaces["bank_elevation"][1, 1])
    assert np.isfinite(center)
    assert center < 5.0


def test_bank_edge_guidance_from_authoritative_is_edge_limited():
    auth = np.array([
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34],
    ], dtype=np.float32)
    corridor = np.zeros((5, 5), dtype=bool)
    corridor[:, 1:4] = True
    edge, bank_distance_m, bank_influence, bank_surface = rbg.compute_bank_edge_guidance_from_authoritative(
        auth, corridor, pixel_size_m=10.0, edge_guidance_distance_m=10.0, max_bank_distance_m=80.0
    )
    assert np.isfinite(bank_surface[:, 1]).all()
    assert np.isfinite(bank_surface[:, 3]).all()
    assert np.isnan(bank_surface[:, 2]).all()
    assert np.all(bank_influence[:, 2] == 0.0)
    assert np.all(edge[:, 1:4] >= 0)
