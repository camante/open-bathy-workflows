from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from presentation_figures import generate_presentation_figures
from support_classes import SupportClass


@dataclass
class _Cfg:
    out_dir: Path
    make_figs: bool = True


def _write_raster(path: Path, arr: np.ndarray, nodata: float = -9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": str(arr.dtype),
        "crs": "EPSG:4326",
        "transform": from_origin(0, 3, 1, 1),
        "nodata": nodata,
    }
    out = np.array(arr, copy=True)
    out[~np.isfinite(out)] = nodata
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(out, 1)
    return path


def test_generate_presentation_figures_core_rasters(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    baseline = _write_raster(out_dir / "baseline.tif", np.array([[1, 2], [3, 4]], dtype=np.float32))
    enhanced = _write_raster(out_dir / "enhanced.tif", np.array([[1, 1.5], [2.5, 4]], dtype=np.float32))
    support = _write_raster(
        out_dir / "support.tif",
        np.array(
            [
                [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_RIVER)],
                [int(SupportClass.SCAFFOLD_INFERRED), int(SupportClass.AUTHORITATIVE_LOCKED)],
            ],
            dtype=np.uint8,
        ),
        nodata=0,
    )
    report = {
        "outputs": {
            "baseline_comparison_navd88_all": str(baseline),
            "final_comparison_navd88_all": str(enhanced),
            "support_class": str(support),
        },
        "river": {"outputs": {}},
    }
    cfg = _Cfg(out_dir=out_dir)
    result = generate_presentation_figures(cfg, report)
    assert result["status"] == "ok"
    figs = out_dir / "figures"
    assert (figs / "support_class_map.png").exists()
    assert (figs / "baseline_vs_enhanced_hillshade.png").exists()
    assert (figs / "difference_map_locked_preserved.png").exists()
    assert report["outputs"]["presentation_support_class_map"].endswith("support_class_map.png")
    assert "river_guidance_construction" in result["skipped"]


def test_generate_presentation_figures_guidance_panel(tmp_path: Path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString, Point

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    baseline = _write_raster(out_dir / "baseline.tif", np.array([[1, 2], [3, 4]], dtype=np.float32))
    enhanced = _write_raster(out_dir / "enhanced.tif", np.array([[1, 1.5], [2.5, 4]], dtype=np.float32))
    support = _write_raster(out_dir / "support.tif", np.array([[1, 4], [5, 1]], dtype=np.uint8), nodata=0)
    guidance = _write_raster(out_dir / "guidance.tif", np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))

    retained = gpd.GeoDataFrame({"id": [1]}, geometry=[LineString([(0.5, 2.5), (1.5, 0.5)])], crs="EPSG:4326")
    banks = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(0.75, 2.25), Point(1.25, 0.75)], crs="EPSG:4326")
    center = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(0.9, 2.1), Point(1.1, 0.9)], crs="EPSG:4326")
    guide = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(1.0, 2.0), Point(1.0, 1.0)], crs="EPSG:4326")
    retained.to_file(out_dir / "retained.gpkg", driver="GPKG")
    banks.to_file(out_dir / "banks.gpkg", driver="GPKG")
    center.to_file(out_dir / "center.gpkg", driver="GPKG")
    guide.to_file(out_dir / "guide.gpkg", driver="GPKG")

    report = {
        "outputs": {
            "baseline_comparison_navd88_all": str(baseline),
            "final_comparison_navd88_all": str(enhanced),
            "support_class": str(support),
            "river_channel_surface": str(guidance),
        },
        "river": {
            "outputs": {
                "retained_network": str(out_dir / "retained.gpkg"),
                "bank_points": str(out_dir / "banks.gpkg"),
                "centerline_points": str(out_dir / "center.gpkg"),
                "guide_points": str(out_dir / "guide.gpkg"),
                "channel_surface": str(guidance),
            }
        },
    }
    cfg = _Cfg(out_dir=out_dir)
    result = generate_presentation_figures(cfg, report)
    assert result["status"] == "ok"
    assert (out_dir / "figures" / "river_guidance_construction.png").exists()


def test_generate_presentation_figures_guidance_panel_reprojects_vectors_to_raster_crs(tmp_path: Path):
    gpd = pytest.importorskip("geopandas")
    from pyproj import Transformer
    from shapely.geometry import LineString, Point

    out_dir = tmp_path / "run_reproject"
    out_dir.mkdir()
    baseline = _write_raster(out_dir / "baseline.tif", np.array([[1, 2], [3, 4]], dtype=np.float32))
    enhanced = _write_raster(out_dir / "enhanced.tif", np.array([[1, 1.5], [2.5, 4]], dtype=np.float32))
    support = _write_raster(out_dir / "support.tif", np.array([[1, 4], [5, 1]], dtype=np.uint8), nodata=0)
    guidance = _write_raster(out_dir / "guidance.tif", np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))

    tx = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    def pt(x, y):
        X, Y = tx.transform(x, y)
        return Point(X, Y)
    def ls(coords):
        return LineString([tx.transform(x, y) for x, y in coords])

    retained = gpd.GeoDataFrame({"id": [1]}, geometry=[ls([(0.5, 2.5), (1.5, 0.5)])], crs="EPSG:3857")
    banks = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[pt(0.75, 2.25), pt(1.25, 0.75)], crs="EPSG:3857")
    center = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[pt(0.9, 2.1), pt(1.1, 0.9)], crs="EPSG:3857")
    guide = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[pt(1.0, 2.0), pt(1.0, 1.0)], crs="EPSG:3857")
    retained.to_file(out_dir / "retained_3857.gpkg", driver="GPKG")
    banks.to_file(out_dir / "banks_3857.gpkg", driver="GPKG")
    center.to_file(out_dir / "center_3857.gpkg", driver="GPKG")
    guide.to_file(out_dir / "guide_3857.gpkg", driver="GPKG")

    report = {
        "outputs": {
            "baseline_comparison_navd88_all": str(baseline),
            "final_comparison_navd88_all": str(enhanced),
            "support_class": str(support),
            "river_channel_surface": str(guidance),
        },
        "river": {
            "outputs": {
                "retained_network": str(out_dir / "retained_3857.gpkg"),
                "bank_points": str(out_dir / "banks_3857.gpkg"),
                "centerline_points": str(out_dir / "center_3857.gpkg"),
                "guide_points": str(out_dir / "guide_3857.gpkg"),
                "channel_surface": str(guidance),
            }
        },
    }
    cfg = _Cfg(out_dir=out_dir)
    result = generate_presentation_figures(cfg, report)
    assert result["status"] == "ok"
    assert (out_dir / "figures" / "river_guidance_construction.png").exists()
