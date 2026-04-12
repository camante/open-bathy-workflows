from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from plot_utils import lazy_pyplot
from support_classes import SUPPORT_CLASS_CODE_TO_NAME, SupportClass

log = logging.getLogger(__name__)


def _resolve_path(value: Any, *, base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _first_existing(base_dir: Path, values: Iterable[Any]) -> Optional[Path]:
    for value in values:
        p = _resolve_path(value, base_dir=base_dir)
        if p and p.exists():
            return p
    return None


def _load_raster(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        profile = {
            "transform": ds.transform,
            "crs": (str(ds.crs) if ds.crs is not None else None),
            "bounds": ds.bounds,
            "shape": (ds.height, ds.width),
        }
    return arr, profile


def _raster_extent(profile: Dict[str, Any]) -> Tuple[float, float, float, float]:
    b = profile["bounds"]
    return (float(b.left), float(b.right), float(b.bottom), float(b.top))


def _ensure_same_shape(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for {label_a} vs {label_b}: {a.shape} != {b.shape}")


def _ensure_same_grid(profile_a: Dict[str, Any], profile_b: Dict[str, Any], label_a: str, label_b: str) -> None:
    if profile_a.get("shape") != profile_b.get("shape"):
        raise ValueError(f"Grid shape mismatch for {label_a} vs {label_b}: {profile_a.get('shape')} != {profile_b.get('shape')}")
    crs_a = str(profile_a.get("crs") or "")
    crs_b = str(profile_b.get("crs") or "")
    if crs_a != crs_b:
        raise ValueError(f"Grid CRS mismatch for {label_a} vs {label_b}: {crs_a} != {crs_b}")
    if profile_a.get("transform") != profile_b.get("transform"):
        raise ValueError(f"Grid transform mismatch for {label_a} vs {label_b}")


def _simple_hillshade(arr: np.ndarray, *, azdeg: float = 315.0, altdeg: float = 45.0) -> np.ndarray:
    if arr.size == 0:
        return arr
    a = np.asarray(arr, dtype=np.float32)
    filled = np.array(a, copy=True)
    finite = np.isfinite(filled)
    if not np.any(finite):
        return np.full_like(filled, np.nan, dtype=np.float32)
    fill_val = float(np.nanmedian(filled[finite]))
    filled[~finite] = fill_val
    dy, dx = np.gradient(filled)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(azdeg)
    alt = np.deg2rad(altdeg)
    hs = (
        np.sin(alt) * np.sin(slope)
        + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    )
    hs = np.clip(hs, 0.0, 1.0)
    hs[~finite] = np.nan
    return hs.astype(np.float32)


def _load_hillshade_or_compute(base_dir: Path, raster_value: Any, hillshade_value: Any) -> Tuple[np.ndarray, Dict[str, Any], Path]:
    hs_path = _resolve_path(hillshade_value, base_dir=base_dir)
    if hs_path and hs_path.exists():
        hs, profile = _load_raster(hs_path)
        return hs, profile, hs_path
    raster_path = _resolve_path(raster_value, base_dir=base_dir)
    if raster_path is None or not raster_path.exists():
        raise FileNotFoundError(f"Missing raster for hillshade source: {raster_value}")
    arr, profile = _load_raster(raster_path)
    return _simple_hillshade(arr), profile, raster_path


def _save_figure(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    return out_path


def _plot_support_class_map(out_path: Path, *, support_class_path: Path) -> Dict[str, Any]:
    plt = lazy_pyplot()
    arr, profile = _load_raster(support_class_path)
    extent = _raster_extent(profile)
    classes = [
        int(SupportClass.AUTHORITATIVE_LOCKED),
        int(SupportClass.ANCHORED_INTERPOLATION),
        int(SupportClass.GUIDANCE_CONDITIONED_SDB),
        int(SupportClass.GUIDANCE_CONDITIONED_RIVER),
        int(SupportClass.SCAFFOLD_INFERRED),
        int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL),
    ]
    labels = [SUPPORT_CLASS_CODE_TO_NAME[c].replace("_", " ") for c in classes]
    # Stable presentation palette.
    colors = [
        "#4c78a8",  # authoritative_locked
        "#72b7b2",  # anchored_interpolation
        "#54a24b",  # guidance_conditioned_sdb
        "#f58518",  # guidance_conditioned_river
        "#e45756",  # scaffold_inferred
        "#b279a2",  # low_confidence_continuous_fill
    ]
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(colors)
    bounds = [c - 0.5 for c in classes] + [classes[-1] + 0.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(arr, cmap=cmap, norm=norm, origin="upper", extent=extent, interpolation="nearest")
    ax.set_title("Support-class map")
    ax.set_xlabel("Longitude / X")
    ax.set_ylabel("Latitude / Y")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[i], edgecolor="none", label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_figure(fig, out_path)
    plt.close(fig)
    finite = arr[np.isfinite(arr)]
    counts = {SUPPORT_CLASS_CODE_TO_NAME[int(c)]: int(np.count_nonzero(finite == c)) for c in classes}
    return {"path": str(out_path), "support_class_counts": counts}


def _plot_baseline_vs_enhanced_hillshade(
    out_path: Path,
    *,
    base_dir: Path,
    baseline_raster: Any,
    baseline_hillshade: Any,
    enhanced_raster: Any,
    enhanced_hillshade: Any,
) -> Dict[str, Any]:
    plt = lazy_pyplot()
    lhs, lprof, lsrc = _load_hillshade_or_compute(base_dir, baseline_raster, baseline_hillshade)
    rhs, rprof, rsrc = _load_hillshade_or_compute(base_dir, enhanced_raster, enhanced_hillshade)
    _ensure_same_shape(lhs, rhs, "baseline hillshade", "enhanced hillshade")
    _ensure_same_grid(lprof, rprof, "baseline hillshade", "enhanced hillshade")
    extent = _raster_extent(lprof)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, arr, title in zip(axes, [lhs, rhs], ["Baseline hillshade", "Enhanced hillshade"]):
        ax.imshow(arr, cmap="gray", origin="upper", extent=extent)
        ax.set_title(title)
        ax.set_xlabel("Longitude / X")
        ax.set_ylabel("Latitude / Y")
    _save_figure(fig, out_path)
    plt.close(fig)
    return {"path": str(out_path), "baseline_source": str(lsrc), "enhanced_source": str(rsrc)}


def _plot_difference_map(
    out_path: Path,
    *,
    baseline_path: Path,
    enhanced_path: Path,
    support_class_path: Path,
) -> Dict[str, Any]:
    plt = lazy_pyplot()
    base, bprof = _load_raster(baseline_path)
    enh, eprof = _load_raster(enhanced_path)
    support, sprof = _load_raster(support_class_path)
    _ensure_same_shape(base, enh, "baseline", "enhanced")
    _ensure_same_shape(base, support, "baseline", "support_class")
    _ensure_same_grid(bprof, eprof, "baseline", "enhanced")
    _ensure_same_grid(bprof, sprof, "baseline", "support_class")
    diff = enh - base
    locked = support == int(SupportClass.AUTHORITATIVE_LOCKED)
    diff_nonlocked = np.array(diff, copy=True)
    diff_nonlocked[locked] = np.nan
    finite = diff_nonlocked[np.isfinite(diff_nonlocked)]
    vmax = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
    vmax = max(vmax, 0.01)
    extent = _raster_extent(bprof)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(diff_nonlocked, cmap="RdBu_r", origin="upper", extent=extent, vmin=-vmax, vmax=vmax)
    ax.contour(locked.astype(np.uint8), levels=[0.5], colors="k", linewidths=0.3, origin="upper", extent=extent)
    ax.set_title("Enhanced minus baseline (non-locked cells only)")
    ax.set_xlabel("Longitude / X")
    ax.set_ylabel("Latitude / Y")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Elevation difference (m)")
    _save_figure(fig, out_path)
    plt.close(fig)
    changed_locked = int(np.count_nonzero(np.isfinite(diff) & locked & (np.abs(diff) > 1e-6)))
    return {
        "path": str(out_path),
        "changed_locked_cell_count": changed_locked,
        "nonlocked_changed_cell_count": int(np.count_nonzero(np.isfinite(diff_nonlocked) & (np.abs(diff_nonlocked) > 1e-6))),
    }


def _read_vector(path: Path):
    import geopandas as gpd

    return gpd.read_file(path)


def _align_vector_for_plot(gdf, *, target_crs):
    if gdf is None:
        return gdf
    if target_crs is None:
        return gdf
    target_crs_norm = str(target_crs)
    try:
        if gdf.crs is None:
            return gdf
        if str(gdf.crs) == target_crs_norm:
            return gdf.set_crs(target_crs_norm, allow_override=True)
        try:
            return gdf.to_crs(target_crs_norm)
        except Exception:
            from pyproj import CRS, Transformer
            from shapely.ops import transform as shapely_transform

            src_crs = CRS.from_user_input(str(gdf.crs))
            dst_crs = CRS.from_user_input(target_crs_norm)
            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            out = gdf.copy()
            out.geometry = out.geometry.apply(lambda geom: None if geom is None else shapely_transform(transformer.transform, geom))
            return out.set_crs(target_crs_norm, allow_override=True)
    except Exception as exc:
        raise ValueError(f"Failed to align vector CRS for plotting: {exc}") from exc


def _plot_vector(ax, gdf, *, color: str, linewidth: float = 0.6, markersize: float = 1.5, alpha: float = 1.0) -> None:
    if gdf is None or gdf.empty or "geometry" not in gdf.columns:
        return
    for geom in gdf.geometry:
        if geom is None or getattr(geom, "is_empty", False):
            continue
        gtype = getattr(geom, "geom_type", "")
        if gtype == "Point":
            ax.scatter([geom.x], [geom.y], s=max(markersize, 0.1) ** 2, c=[color], alpha=alpha, linewidths=0)
        elif gtype in {"LineString", "LinearRing"}:
            xs, ys = geom.xy
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)
        elif gtype == "MultiPoint":
            xs = [pt.x for pt in geom.geoms]
            ys = [pt.y for pt in geom.geoms]
            if xs:
                ax.scatter(xs, ys, s=max(markersize, 0.1) ** 2, c=[color], alpha=alpha, linewidths=0)
        elif gtype.startswith("Multi") or gtype == "GeometryCollection":
            sub_gdf = gdf.__class__({"geometry": list(getattr(geom, "geoms", []))}, crs=gdf.crs)
            _plot_vector(ax, sub_gdf, color=color, linewidth=linewidth, markersize=markersize, alpha=alpha)
        elif hasattr(geom, "exterior") and geom.exterior is not None:
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)


def _plot_guidance_construction(
    out_path: Path,
    *,
    retained_network: Path,
    bank_points: Path,
    centerline_points: Path,
    guide_points: Path,
    river_guidance: Path,
) -> Dict[str, Any]:
    plt = lazy_pyplot()
    retained = _read_vector(retained_network)
    banks = _read_vector(bank_points)
    center = _read_vector(centerline_points)
    guide = _read_vector(guide_points)
    guidance_arr, gprof = _load_raster(river_guidance)
    target_crs = gprof.get("crs")
    retained = _align_vector_for_plot(retained, target_crs=target_crs)
    banks = _align_vector_for_plot(banks, target_crs=target_crs)
    center = _align_vector_for_plot(center, target_crs=target_crs)
    guide = _align_vector_for_plot(guide, target_crs=target_crs)
    extent = _raster_extent(gprof)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    panels = [
        (axes[0], "Retained network", retained, {"linewidth": 0.6, "color": "#1f77b4"}),
        (axes[1], "Bank points", banks, {"markersize": 1.5, "color": "#2ca02c"}),
        (axes[2], "Centerline points", center, {"markersize": 1.5, "color": "#ff7f0e"}),
        (axes[3], "Guide points", guide, {"markersize": 1.2, "color": "#d62728"}),
    ]
    for ax, title, gdf, kwargs in panels:
        if not gdf.empty:
            _plot_vector(ax, gdf, **kwargs)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title)
        ax.set_xlabel("Longitude / X")
        ax.set_ylabel("Latitude / Y")

    axes[4].imshow(guidance_arr, cmap="viridis", origin="upper", extent=extent)
    axes[4].set_title("River guidance raster")
    axes[4].set_xlabel("Longitude / X")
    axes[4].set_ylabel("Latitude / Y")

    _plot_vector(axes[5], retained, linewidth=0.4, color="#7f7f7f")
    if not center.empty:
        _plot_vector(axes[5], center, markersize=1.0, color="#ff7f0e")
    if not guide.empty:
        _plot_vector(axes[5], guide, markersize=0.8, color="#d62728", alpha=0.6)
    axes[5].set_xlim(extent[0], extent[1])
    axes[5].set_ylim(extent[2], extent[3])
    axes[5].set_title("Overlay: network + centerline + guides")
    axes[5].set_xlabel("Longitude / X")
    axes[5].set_ylabel("Latitude / Y")

    fig.suptitle("River guidance construction", y=0.98)
    _save_figure(fig, out_path)
    plt.close(fig)
    return {
        "path": str(out_path),
        "retained_network_count": int(len(retained)),
        "bank_point_count": int(len(banks)),
        "centerline_point_count": int(len(center)),
        "guide_point_count": int(len(guide)),
    }


def generate_presentation_figures(cfg: Any, report: Dict[str, Any], *, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    logger = logger or log
    out_dir = Path(getattr(cfg, "out_dir"))
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = report.get("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
    river_outputs = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}).get("outputs", {}), dict) else {}

    support_class_path = _first_existing(out_dir, [
        outputs.get("support_class"),
        outputs.get("river_channel_surface_support_class"),
    ])
    baseline_raster = _first_existing(out_dir, [
        outputs.get("baseline_comparison_navd88_all"),
        outputs.get("baseline_comparison_navd88"),
    ])
    enhanced_raster = _first_existing(out_dir, [
        outputs.get("final_comparison_navd88_all"),
        outputs.get("final_comparison_navd88"),
        outputs.get("selected_final_native"),
        outputs.get("selected_final"),
    ])
    baseline_hillshade = _first_existing(out_dir, [
        outputs.get("baseline_comparison_navd88_all_hillshade"),
        outputs.get("baseline_comparison_navd88_hillshade"),
    ])
    enhanced_hillshade = _first_existing(out_dir, [
        outputs.get("final_comparison_navd88_all_hillshade"),
        outputs.get("final_comparison_navd88_hillshade"),
        outputs.get("combined_hillshade"),
    ])

    result: Dict[str, Any] = {"status": "started", "figures_dir": str(figures_dir), "figures": {}, "skipped": {}, "failed": {}}

    def _run(name: str, fn, **kwargs):
        try:
            meta = fn(figures_dir / f"{name}.png", **kwargs)
            result["figures"][name] = meta
            report.setdefault("outputs", {})[f"figure_{name}"] = str(figures_dir / f"{name}.png")
            report.setdefault("outputs", {})[f"presentation_{name}"] = str(figures_dir / f"{name}.png")
            logger.info("[FIGURES] Wrote %s", figures_dir / f"{name}.png")
        except Exception as exc:
            logger.error("[FIGURES] Failed %s: %s", name, exc)
            result["failed"][name] = str(exc)

    if support_class_path and support_class_path.exists():
        _run("support_class_map", _plot_support_class_map, support_class_path=support_class_path)
    else:
        result["skipped"]["support_class_map"] = "support_class raster not available"

    if baseline_raster and enhanced_raster:
        _run(
            "baseline_vs_enhanced_hillshade",
            _plot_baseline_vs_enhanced_hillshade,
            base_dir=out_dir,
            baseline_raster=baseline_raster,
            baseline_hillshade=baseline_hillshade,
            enhanced_raster=enhanced_raster,
            enhanced_hillshade=enhanced_hillshade,
        )
    else:
        result["skipped"]["baseline_vs_enhanced_hillshade"] = "baseline/enhanced comparison rasters not available"

    if baseline_raster and enhanced_raster and support_class_path:
        _run(
            "difference_map_locked_preserved",
            _plot_difference_map,
            baseline_path=baseline_raster,
            enhanced_path=enhanced_raster,
            support_class_path=support_class_path,
        )
    else:
        result["skipped"]["difference_map_locked_preserved"] = "baseline/enhanced/support_class inputs not available"

    retained_network = _first_existing(out_dir, [river_outputs.get("retained_network")])
    bank_points = _first_existing(out_dir, [river_outputs.get("bank_points")])
    centerline_points = _first_existing(out_dir, [river_outputs.get("centerline_points")])
    guide_points = _first_existing(out_dir, [river_outputs.get("guide_points")])
    river_guidance = _first_existing(out_dir, [
        river_outputs.get("channel_surface"),
        outputs.get("river_channel_surface"),
    ])
    if all(p is not None and p.exists() for p in [retained_network, bank_points, centerline_points, guide_points, river_guidance]):
        _run(
            "river_guidance_construction",
            _plot_guidance_construction,
            retained_network=retained_network,
            bank_points=bank_points,
            centerline_points=centerline_points,
            guide_points=guide_points,
            river_guidance=river_guidance,
        )
    else:
        result["skipped"]["river_guidance_construction"] = "river guidance vector/raster artifacts not all available"

    if result["failed"]:
        result["status"] = "partial"
    elif result["figures"]:
        result["status"] = "ok"
    else:
        result["status"] = "skipped"
    summary_path = figures_dir / "presentation_figures_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report.setdefault("outputs", {})["presentation_figures_summary"] = str(summary_path)
    report.setdefault("presentation_figures", {}).update(result)
    return result
