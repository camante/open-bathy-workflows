"""conftest.py – shared mocks and fixture builders.

Imported by every test module before any pipeline module is touched.
Installs rasterio, pyproj, and other geo-library stubs into sys.modules so
the pipeline code can be imported in environments where those packages are
absent.  All raster I/O in the mocks is backed by tifffile for real array
behaviour.
"""

from __future__ import annotations

import sys
import types
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib
import tifffile
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Rasterio mock backed by tifffile
# ---------------------------------------------------------------------------

class _FakeCRS:
    def __init__(self, epsg=4326):
        self._epsg = epsg
    def to_epsg(self):
        return self._epsg
    def to_wkt(self):
        return f"EPSG:{self._epsg}"
    def __str__(self):
        return f"EPSG:{self._epsg}"


class _FakeTransform:
    def __init__(self):
        self.a, self.e, self.c, self.f = 0.01, -0.01, -81.0, 25.0


class FakeWindow:
    def __init__(self, col_off, row_off, width, height):
        self.col_off, self.row_off = col_off, row_off
        self.width, self.height = width, height


class FakeDataset:
    """Rasterio-compatible dataset backed by a tifffile-written array."""

    def __init__(self, path: str, mode: str = "r", **kw):
        self.path = str(path)
        self._mode = mode
        self.nodata = kw.get("nodata", -9999.0)
        self.crs = _FakeCRS(4326)
        self.transform = _FakeTransform()

        if mode == "r":
            raw = tifffile.imread(self.path)
            if raw.ndim == 2:
                raw = raw[np.newaxis]
            self._arr = raw.astype(np.float32)
        else:
            h = kw.get("height", 64)
            w = kw.get("width", 64)
            c = kw.get("count", 1)
            dtype = kw.get("dtype", "float32")
            self._arr = np.full((c, h, w), self.nodata, dtype=np.float32)

        self.height, self.width, self.count = (
            self._arr.shape[1], self._arr.shape[2], self._arr.shape[0]
        )
        self.dtypes = ["float32"] * self.count
        self.bounds = type("B", (), {
            "left": -81.0, "bottom": 24.0, "right": -80.0, "top": 25.0
        })()
        self.profile = {
            "driver": "GTiff", "dtype": "float32",
            "nodata": self.nodata,
            "width": self.width, "height": self.height, "count": self.count,
            "crs": self.crs, "transform": self.transform,
            "compress": "DEFLATE", "tiled": True,
            "blockxsize": 64, "blockysize": 64,
        }
        self.profile.update(kw)
        if self.profile.get("blockxsize") or self.profile.get("blockysize"):
            self.profile["tiled"] = True

    def read(self, band: int = 1, window=None, **kw) -> np.ndarray:
        d = self._arr[band - 1]
        if window is None:
            return d
        r0, c0 = int(window.row_off), int(window.col_off)
        h, w = int(window.height), int(window.width)
        return d[r0:r0+h, c0:c0+w].copy()

    def write(self, arr: np.ndarray, band: int, window=None):
        if window is None:
            self._arr[band - 1] = arr
        else:
            r0, c0 = int(window.row_off), int(window.col_off)
            h, w = int(window.height), int(window.width)
            self._arr[band - 1, r0:r0+h, c0:c0+w] = arr
        if self._mode in ("w", "r+"):
            out = self._arr[0] if self.count == 1 else self._arr
            tifffile.imwrite(self.path, out)

    def update_tags(self, **kw):
        pass

    def copy(self):
        return dict(self.profile)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def close(self):
        pass


class _WarpedVRT:
    def __init__(self, src, **kw):
        self._src = src
    def read(self, band=1, window=None, **kw):
        return self._src.read(band, window=window, **kw)
    def __enter__(self): return self
    def __exit__(self, *a): pass


def _install_mocks():
    """Install all geo-library stubs. Safe to call multiple times.

    Prefer the real rasterio package when it is available so tests that import
    submodules like ``rasterio.transform`` keep working. Fall back to the light
    tifffile-backed mock only when rasterio cannot be imported in this test
    environment.
    """
    if "rasterio" in sys.modules and hasattr(sys.modules["rasterio"], "_mocked"):
        return

    try:
        import rasterio as _real_rasterio  # noqa: F401
    except Exception:
        _real_rasterio = None

    if _real_rasterio is None:
        # --- rasterio fallback mock ---
        rio = types.ModuleType("rasterio")
        rio.float32 = "float32"
        rio.uint8 = "uint8"
        rio.open = lambda path, mode="r", **kw: FakeDataset(str(path), mode=mode, **kw)
        rio._mocked = True

        win = types.ModuleType("rasterio.windows")
        win.Window = FakeWindow
        win.from_bounds = lambda *a, **kw: FakeWindow(0, 0, 64, 64)

        vrt = types.ModuleType("rasterio.vrt")
        vrt.WarpedVRT = _WarpedVRT

        enums = types.ModuleType("rasterio.enums")
        enums.Resampling = type("Resampling", (), {"nearest": 0, "bilinear": 1})()

        sys.modules.update({
            "rasterio": rio, "rasterio.windows": win,
            "rasterio.vrt": vrt, "rasterio.enums": enums,
        })

    # --- pyproj ---
    # Use a lightweight stub by default so imports remain stable even in thin
    # environments. The stub includes CRS/Transformer/Geod/ProjError support so
    # geopandas-backed tests that do not explicitly swap in real pyproj can
    # still construct CRS-aware objects. Focused tests may replace this stub
    # with the real pyproj package by clearing sys.modules first.
    pyproj = types.ModuleType("pyproj")

    class _CRS:
        def __init__(self, value=None):
            self._value = value if value is not None else "EPSG:4326"

        @classmethod
        def from_user_input(cls, value):
            return cls(value)

        @classmethod
        def from_epsg(cls, epsg):
            return cls(f"EPSG:{int(epsg)}")

        def to_epsg(self):
            try:
                text = str(self._value).upper()
                if text.startswith("EPSG:"):
                    return int(text.split(":", 1)[1])
            except Exception:
                return None
            return None

        def to_wkt(self):
            return str(self._value)

        def __str__(self):
            return str(self._value)

        def __repr__(self):
            return f"CRS({self._value!r})"

    class _T:
        @staticmethod
        def from_crs(src, dst, always_xy=True):
            return _T()

        def transform(self, x, y, z=None):
            if z is None:
                return x, y
            return x, y, z

    class _Geod:
        def __init__(self, *a, **kw):
            pass

        def inv(self, lon1, lat1, lon2, lat2):
            arr = np.asarray(lon1, dtype=float)
            shape = arr.shape if arr.shape else np.asarray(lon2, dtype=float).shape
            zeros = np.zeros(shape, dtype=float)
            return zeros, zeros, zeros

    class ProjError(Exception):
        pass

    class CRSError(ProjError):
        pass

    class DataDirError(ProjError):
        pass

    class GeodError(ProjError):
        pass

    pyproj.CRS = _CRS
    pyproj.Transformer = _T
    pyproj.Geod = _Geod
    pyproj.Proj = _CRS
    pyproj.network = types.SimpleNamespace(set_ca_bundle_path=lambda *a, **kw: None)
    exc_mod = types.ModuleType("pyproj.exceptions")
    exc_mod.ProjError = ProjError
    exc_mod.CRSError = CRSError
    exc_mod.DataDirError = DataDirError
    exc_mod.GeodError = GeodError
    pyproj.exceptions = exc_mod
    sys.modules["pyproj"] = pyproj
    sys.modules["pyproj.exceptions"] = exc_mod

    # --- misc stubs ---
    for name in ["affine", "fiona", "shapely", "shapely.geometry",
                 "geopandas", "pyogrio"]:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)

    # --- process_utils stub ---
    if "process_utils" not in sys.modules:
        pu = types.ModuleType("process_utils")
        pu.run_cmd = lambda *a, **kw: None
        sys.modules["process_utils"] = pu


_install_mocks()
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Test-suite classification
# ---------------------------------------------------------------------------
# These markers reflect the workflow's current architecture:
# - core: authoritative-first, support-aware, guidance-conditioned end-state
# - transitional: tests protecting routes/behaviors still present during migration
# - legacy: compatibility coverage for older fusion-centered behavior
_TRANSITIONAL_TESTS = {
    "test_bathy_main.py",
    "test_bathy_main_authoritative_river_fallback.py",
    "test_fusion_atl.py",
    "test_pipeline.py",
    "test_pipeline_aoi.py",
    "test_river_guidance_controls.py",
    "test_run_summary_scientific.py",
    "test_source_aware_candidate.py",
    "test_source_aware_candidate_river_corridor.py",
    "test_tier2_path.py",
}

_LEGACY_TESTS = {
    "test_bathy_fusion_numerical.py",
}


def pytest_collection_modifyitems(config, items):
    """Classify the suite into core / transitional / legacy buckets.

    Core is the default because the repo's present architecture is centered on
    authoritative support, canonical scaffold generation, guidance artifacts,
    and terrain interpolation. Only tests that explicitly protect older or
    still-migrating routes are marked otherwise.
    """
    for item in items:
        name = Path(str(item.fspath)).name
        if name in _LEGACY_TESTS:
            item.add_marker("legacy")
        elif name in _TRANSITIONAL_TESTS:
            item.add_marker("transitional")
        else:
            item.add_marker("core")


# ---------------------------------------------------------------------------
# Scene / model fixture builders
# ---------------------------------------------------------------------------

SCENE_SIZE = 64
N_TRAIN = 400


def build_synthetic_scene(scene_dir: Path) -> Dict[str, Path]:
    rng = np.random.default_rng(0)
    s = SCENE_SIZE
    bands = {
        "B02": rng.uniform(0.02, 0.15, (s, s)).astype(np.float32),
        "B03": rng.uniform(0.015, 0.12, (s, s)).astype(np.float32),
        "B04": rng.uniform(0.01, 0.08, (s, s)).astype(np.float32),
        "B08": rng.uniform(0.005, 0.04, (s, s)).astype(np.float32),
    }
    bands["CLEAR_WATER"] = rng.uniform(0.55, 1.0, (s, s)).astype(np.float32)
    b = bands
    bands["BRIGHTNESS"] = ((b["B02"] + b["B03"] + b["B04"]) / 3).astype(np.float32)

    scene_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, arr in bands.items():
        p = scene_dir / f"{name}_10m.tif"
        tifffile.imwrite(str(p), arr)
        paths[name] = p

    lm = scene_dir / "land_mask.tif"
    tifffile.imwrite(str(lm), np.zeros((s, s), dtype=np.float32))
    paths["land_mask"] = lm
    return paths


def build_training_df(n: int = N_TRAIN, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    b02 = rng.uniform(0.02, 0.15, n).astype(np.float32)
    b03 = rng.uniform(0.015, 0.12, n).astype(np.float32)
    b04 = rng.uniform(0.01, 0.08, n).astype(np.float32)
    b08 = rng.uniform(0.005, 0.04, n).astype(np.float32)
    return pd.DataFrame({
        "longitude":    rng.uniform(-82, -81, n),
        "latitude":     rng.uniform(24, 25, n),
        "depth_m":      -rng.uniform(1.0, 15.0, n),
        "B02": b02, "B03": b03, "B04": b04, "B08": b08,
        "brightness":   (b02 + b03 + b04) / 3,
        "CLEAR_WATER":  rng.uniform(0.6, 1.0, n),
        "LAND":         rng.uniform(0.0, 0.3, n),
        "stumpf_idx":   np.log(b02 + 1e-3) / np.log(b03 + 1e-3),
        "stumpf_depth": rng.uniform(1, 15, n),
        "source":       "atl03",
        "sample_weight": np.ones(n),
    })


FEATURE_COLS = [
    "B02", "B03", "B04", "B08",
    "log_B02", "log_B03", "log_B04", "log_B08",
    "brightness", "B03_B02", "B04_B03", "nbri",
    "stumpf_idx", "stumpf_depth",
]


def build_model_artifacts(model_dir: Path) -> None:
    rng = np.random.default_rng(1)
    n = 300
    b02 = rng.uniform(0.02, 0.15, n).astype(np.float32)
    b03 = rng.uniform(0.015, 0.12, n).astype(np.float32)
    b04 = rng.uniform(0.01, 0.08, n).astype(np.float32)
    b08 = rng.uniform(0.005, 0.04, n).astype(np.float32)

    X = np.column_stack([
        b02, b03, b04, b08,
        np.log(b02+1e-3), np.log(b03+1e-3), np.log(b04+1e-3), np.log(b08+1e-3),
        (b02+b03+b04)/3,
        b03/(b02+1e-6), b04/(b03+1e-6),
        (b08-b03)/(b08+b03+1e-6),
        np.log(b02+1e-3)/np.log(b03+1e-3),
        rng.uniform(1, 15, n),
    ]).astype(np.float32)
    y = rng.uniform(1.0, 15.0, n).astype(np.float32)

    rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
    rf.fit(X, y)
    stumpf_lr = LinearRegression()
    stumpf_lr.fit(X[:, [12]], y)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_dir / "rf_model.pkl")
    joblib.dump(stumpf_lr, model_dir / "stumpf_lr.pkl")

    training_bounds = {
        col: {"min": float(X[:, i].min()), "max": float(X[:, i].max())}
        for i, col in enumerate(FEATURE_COLS)
    }
    meta = {
        "feature_columns": FEATURE_COLS,
        "max_depth_sdb_final": 20.0,
        "max_depth_sdb_final_source": "training_p95",
        "depth_stats_m": {"max": 15.0, "min": 1.0},
        "linf_enabled": False,
        "training_bounds": training_bounds,
        "doa": {
            "weights": {c: float(w) for c, w in zip(FEATURE_COLS, rf.feature_importances_)}
        },
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
