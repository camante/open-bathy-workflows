#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis.py – Visual QC for ICESat-2 (ATL03) vs Training Data (with plot linkage GeoPackages)
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Refraction index helper (keep consistent with atl.py)
# -----------------------------------------------------------------------------
def _n_water_index(temp_c: float = 20.0, wavelength_nm: float = 532.0) -> float:
    """Compute water refractive index used by the ATL03 refraction model."""
    try:
        return float(-0.000001501562500 * temp_c**2
                     + 0.000000107084865 * wavelength_nm**2
                     - 0.000042759374989 * temp_c
                     - 0.000160475520686 * wavelength_nm
                     + 1.398067112092424)
    except Exception:
        return 1.333

# Optional: write GeoPackages
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

# Direct import for flat directory structure
try:
    from atl import read_atl03_basic
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from atl import read_atl03_basic

log = logging.getLogger("sdb.vis")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# -----------------------------------------------------------------------------
# Depth-of-support helper (optional)
# -----------------------------------------------------------------------------
def _resolve_max_depth_sdb(
    max_depth_arg: Optional[str],
    *,
    model_meta_path: Optional[str] = None,
    plots_dir: Optional[Path] = None,
    default: Optional[float] = None,
) -> Optional[float]:
    """Resolve max_depth_sdb for visualization.

    Accepts:
      - None -> returns default
      - numeric string -> float(value)
      - 'auto' -> reads model metadata JSON (max_depth_sdb or max_depth_sdb_auto)

    This is for *annotation* only; it does not affect training picks.
    """
    if max_depth_arg is None:
        return default

    s = str(max_depth_arg).strip().lower()
    if s == "":
        return default

    # numeric
    try:
        return float(s)
    except Exception:
        pass

    if s not in ("auto", "automatic"):
        return default

    # locate metadata
    cand_paths = []
    if model_meta_path:
        cand_paths.append(Path(model_meta_path))
    if plots_dir is not None:
        # common locations in this pipeline
        cand_paths.append(Path(plots_dir) / "model_meta.json")
        cand_paths.append(Path(plots_dir).parent / "model_meta.json")

    for p in cand_paths:
        try:
            if p.exists() and p.stat().st_size > 0:
                import json
                meta = json.loads(p.read_text())
                for k in ("max_depth_sdb", "max_depth_sdb_auto"):
                    v = meta.get(k, None)
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                        if fv > 0:
                            return fv
                    except Exception:
                        continue
        except Exception:
            continue

    return default

def _read_photons_for_plot(h5_path, laser_num, aoi_bbox, conf_min):
    """Reads ATL03 photons for a specific beam within the AOI."""
    try:
        lat, lon, h_ph, conf, _, _, _, _, _ = read_atl03_basic(h5_path, laser_num)
    except Exception as exc:
        log.debug(f"[VIS] read_atl03_basic failed for {h5_path}, beam {laser_num}: {exc}")
        return None

    W, S, E, N = aoi_bbox
    mask = (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N) & (conf >= conf_min)

    if not np.any(mask):
        return None

    return pd.DataFrame({"latitude": lat[mask], "longitude": lon[mask], "photon_height": h_ph[mask]})

def _estimate_alongtrack_dist(lon, lat):
    """Returns distance in meters from the first point in the array."""
    if lon.size == 0: 
        return np.array([], dtype=np.float32), None, (0,0)
        
    mean_lat = float(np.nanmean(lat))
    mean_lon = float(np.nanmean(lon))
    
    # Select UTM zone dynamically
    if mean_lat >= 0:
        utm_zone = int(((mean_lon + 180.0) % 360) // 6) + 1
        utm_zone = min(max(utm_zone, 1), 60)  # Fixed: handle edge cases
        utm_epsg = 32600 + utm_zone
    else:
        utm_zone = int(((mean_lon + 180.0) % 360) // 6) + 1
        utm_zone = min(max(utm_zone, 1), 60)  # Fixed: handle edge cases
        utm_epsg = 32700 + utm_zone
    
    tfm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    x, y = tfm.transform(lon, lat)
    
    x0, y0 = x[0], y[0]
    dist = np.sqrt((x - x0)**2 + (y - y0)**2).astype(np.float32)
    
    return dist, tfm, (x0, y0)

def generate_atl03_debug_plots(training_df, atl03_files, aoi_bbox, plots_dir,
                               conf_min=0, chunk_size_m=1000.0, atl24_df=None,
                               gpkg_full_path: str = None,
                               gpkg_slim_path: str = None,
                               write_gpkgs: bool = True,
                               water_temp_c: float = 20.0,      # <--- Added Arg
                               wavelength_nm: float = 532.0,    # <--- Added Arg
                               depth_mode: str = "true",        # <--- Added Arg
                               max_depth_sdb: Optional[str] = None,
                               model_meta_path: Optional[str] = None):
    """
    Generates diagnostic plots of ATL03 photons with overlaid training picks.
    """
    required_cols = {"longitude", "latitude", "depth_m", "ws_h", "granule", "beam"}
    if required_cols.difference(training_df.columns) or training_df.empty or not atl03_files:
        log.warning("[VIS] Skipping plots (missing cols/data).")
        return

    out_dir = plots_dir / "atl03_transects"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[VIS] Generating ATL03 debug plots in {out_dir} ...")

    unique_pairs = training_df[["granule", "beam"]].drop_duplicates()
    n_water = _n_water_index(float(water_temp_c), float(wavelength_nm))
    log.info(f"[VIS] Using refractive index n_water={n_water:.6f} (temp_c={water_temp_c}, wavelength_nm={wavelength_nm})")

    # Resolve max depth for depth-of-support annotation (optional)
    max_depth_val = _resolve_max_depth_sdb(
        max_depth_sdb,
        model_meta_path=model_meta_path,
        plots_dir=Path(plots_dir) if plots_dir is not None else None,
        default=None,
    )
    if max_depth_val is not None:
        log.info(f"[VIS] Depth-of-support annotation enabled: max_depth_sdb={max_depth_val:.3f} m")


    # Collect per-point metadata linking each training pick to the debug plot PNG it appears in.
    annotated_parts = []

    for _, row in unique_pairs.iterrows():
        granule = str(row["granule"])
        beam_name = str(row["beam"])
        
        matching_files = [f for f in atl03_files if granule in Path(f).stem]
        if not matching_files: continue
        h5_path = matching_files[0]
        
        if not beam_name.startswith("gt"): continue
        laser_num = beam_name[2]

        df_picks = training_df[(training_df["granule"] == granule) & (training_df["beam"] == beam_name)].copy()
        if df_picks.empty: continue

# Bottom elevation for visualization (robust to depth sign conventions)
        # Expected convention is depth_m = positive-down. If your pipeline stores negative-down,
        # auto-correct for visualization so bottom plots below the surface.
        depth_vals = pd.to_numeric(df_picks["depth_m"], errors="coerce").to_numpy(dtype=float)
        depth_med = float(np.nanmedian(depth_vals)) if np.isfinite(depth_vals).any() else float("nan")
        neg_down = np.isfinite(depth_med) and (depth_med < 0.0)
        if neg_down:
            log.warning(f"[VIS] Detected negative-down depth_m in picks (median={depth_med:.3f}). Using abs(depth_m) for visualization.")
            depth_for_vis = np.abs(depth_vals)
        else:
            depth_for_vis = depth_vals

        if str(depth_mode).lower() in ("legacy", "legacy_n_multiplier", "n_multiplier"):
            df_picks["bottom_h_vis"] = df_picks["ws_h"] - (depth_for_vis * n_water)
        else:
            df_picks["bottom_h_vis"] = df_picks["ws_h"] - depth_for_vis

        df_raw = _read_photons_for_plot(h5_path, laser_num, aoi_bbox, conf_min)
        if df_raw is None or df_raw.empty:
            continue

        dists, tfm, origin = _estimate_alongtrack_dist(df_raw["longitude"].values, df_raw["latitude"].values)
        df_raw["dist_m"] = dists
        x0, y0 = origin

        def project_to_track(lons, lats):
            if tfm is None: return np.zeros_like(lons)
            px, py = tfm.transform(lons, lats)
            return np.sqrt((px - x0)**2 + (py - y0)**2)

        df_picks["dist_m"] = project_to_track(df_picks["longitude"].values, df_picks["latitude"].values)

        df_24_local = pd.DataFrame()
        if atl24_df is not None and not atl24_df.empty:
            pad = 0.01 
            lat_min, lat_max = df_raw["latitude"].min() - pad, df_raw["latitude"].max() + pad
            lon_min, lon_max = df_raw["longitude"].min() - pad, df_raw["longitude"].max() + pad
            
            mask_24 = (
                (atl24_df["latitude"] >= lat_min) & (atl24_df["latitude"] <= lat_max) &
                (atl24_df["longitude"] >= lon_min) & (atl24_df["longitude"] <= lon_max)
            )
            subset_24 = atl24_df[mask_24].copy()
            
            if not subset_24.empty:
                subset_24["dist_m"] = project_to_track(subset_24["longitude"].values, subset_24["latitude"].values)
                df_24_local = subset_24

        max_dist = float(df_raw["dist_m"].max())
        if max_dist <= 0: continue

        for start_m in np.arange(0.0, max_dist, float(chunk_size_m)):
            end_m = start_m + chunk_size_m
            
            sub_raw = df_raw[(df_raw["dist_m"] >= start_m) & (df_raw["dist_m"] < end_m)]
            sub_picks = df_picks[(df_picks["dist_m"] >= start_m) & (df_picks["dist_m"] < end_m)]
            
            if sub_picks.empty: continue

            fig, ax = plt.subplots(figsize=(10, 5))
            
            # 1. ATL24 (Cyan Diamonds)
            if not df_24_local.empty:
                sub_24 = df_24_local[(df_24_local["dist_m"] >= start_m) & (df_24_local["dist_m"] < end_m)]
                if not sub_24.empty:
                    if len(sub_picks) > 1:
                        sorted_picks = sub_picks.sort_values("dist_m")
                        surface_h_at_24 = np.interp(
                            sub_24["dist_m"].values, 
                            sorted_picks["dist_m"].values, 
                            sorted_picks["ws_h"].values
                        )
                    else:
                        surface_h_at_24 = sub_picks["ws_h"].mean()

                    # Calculate vis elevation for ATL24 based on depth mode (robust to depth sign)
                    d24 = pd.to_numeric(sub_24["depth_m"], errors="coerce").to_numpy(dtype=float)
                    d24_med = float(np.nanmedian(d24)) if np.isfinite(d24).any() else float("nan")
                    if np.isfinite(d24_med) and d24_med < 0.0:
                        d24 = np.abs(d24)

                    if str(depth_mode).lower() in ("legacy", "legacy_n_multiplier", "n_multiplier"):
                         atl24_elev_vis = surface_h_at_24 - (d24 * n_water)
                    else:
                         atl24_elev_vis = surface_h_at_24 - d24

                    ax.scatter(sub_24["dist_m"], atl24_elev_vis, s=40, c="cyan", 
                               edgecolors="none", marker="d", label="ATL24 Depth", zorder=1, alpha=0.8)

            # 2. Raw Photons (Gray dots)
            if not sub_raw.empty:
                ax.scatter(sub_raw["dist_m"], sub_raw["photon_height"], s=1, c="gray", alpha=0.3, label="Raw Photons", zorder=2)
            
            # 3. ATL03 Picks (Red Surface / Black Bottom)
            ax.scatter(sub_picks["dist_m"], sub_picks["ws_h"], s=12, c="red", marker="_", label="ATL03 Surface", zorder=3)
            ax.scatter(sub_picks["dist_m"], sub_picks["bottom_h_vis"], s=12, c="black", marker="_", label="ATL03 Bottom", zorder=3)


            # Depth-of-support line (optional): plotted as bottom elevation at (surface - max_depth)
            if max_depth_val is not None and np.isfinite(max_depth_val) and max_depth_val > 0:
                try:
                    surface_ref = float(np.nanmedian(sub_picks["ws_h"].values))
                    if np.isfinite(surface_ref):
                        if str(depth_mode).lower() in ("legacy", "legacy_n_multiplier", "n_multiplier"):
                            y_support = surface_ref - (max_depth_val * n_water)
                        else:
                            y_support = surface_ref - max_depth_val
                        ax.axhline(
                            y=y_support,
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.9,
                            label=f"Depth support ({max_depth_val:.1f} m)",
                        )
                except Exception:
                    pass

            ax.set_title(f"{granule} – {beam_name} – {int(start_m)}-{int(end_m)} m")
            
            y_vals = np.concatenate([sub_picks["bottom_h_vis"].values, sub_picks["ws_h"].values])
            ax.set_ylim(float(np.nanmin(y_vals)) - 3.0, float(np.nanmax(y_vals)) + 3.0)
            
            ax.legend(loc="upper right", markerscale=2.0)
            ax.set_xlabel("Along-Track Distance (m)")
            ax.set_ylabel("Elevation (m)")
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            plot_png = str(out_dir / f"{granule}_{beam_name}_{int(start_m)}m.png")
            # Annotate picks in this chunk with the plot they appear in
            if not sub_picks.empty:
                sub_picks_anno = sub_picks.copy()
                sub_picks_anno["plot_png"] = plot_png
                sub_picks_anno["plot_start_m"] = float(start_m)
                sub_picks_anno["plot_end_m"] = float(end_m)
                annotated_parts.append(sub_picks_anno)

            fig.savefig(out_dir / f"{granule}_{beam_name}_{int(start_m)}m.png", dpi=150)
            plt.close(fig)

    log.info("[VIS] Plots complete.")

def main():
    parser = argparse.ArgumentParser(description="Standalone visualizer")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--atl03-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpkg-full", default=None, help="Output full GeoPackage path (includes plot_png linkage).")
    parser.add_argument("--gpkg-slim", default=None, help="Output slim GeoPackage path (only source, depth_m, geometry).")
    parser.add_argument("--no-gpkg", action="store_true", default=False, help="Do not write GeoPackages.")
    parser.add_argument("--water-temp-c", type=float, default=20.0, help="Water temperature (°C).")
    parser.add_argument("--wavelength-nm", type=float, default=532.0, help="Laser wavelength (nm).")
    parser.add_argument("--depth-mode", default="true", choices=["true","legacy_n_multiplier"], help="Depth conversion mode.")
    parser.add_argument("--max-depth-sdb", default=None,
                    help="Depth-of-support annotation (m). Use a number (e.g., 6) or 'auto' to read from model_meta.json.")
    parser.add_argument("--model-meta", default=None,
                    help="Optional path to model_meta.json (used when --max-depth-sdb auto).")
    parser.add_argument("--aoi", default=None, help="W/E/S/N override")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    
    if args.aoi:
        w, e, s, n = [float(x) for x in args.aoi.split("/")]
        bbox = [w, s, e, n]
    else:
        bbox = [df.longitude.min(), df.latitude.min(), df.longitude.max(), df.latitude.max()]

    h5_files = list(Path(args.atl03_dir).glob("*.h5"))
    h5_strs = [str(f) for f in h5_files]
    
    generate_atl03_debug_plots(
        df, h5_strs, bbox, Path(args.out_dir),
        water_temp_c=float(args.water_temp_c),
        wavelength_nm=float(args.wavelength_nm),
        depth_mode=str(args.depth_mode),
        max_depth_sdb=args.max_depth_sdb,
        model_meta_path=args.model_meta,
        gpkg_full_path=args.gpkg_full,
        gpkg_slim_path=args.gpkg_slim,
        write_gpkgs=(not args.no_gpkg)
    )


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()