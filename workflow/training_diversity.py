#!/usr/bin/env python3
"""
training_diversity.py - Analyze and improve training data spatial/spectral diversity

This module helps diagnose WHY spatial validation performs worse than random validation
and suggests strategies to improve training data collection.

Key metrics:
1. Spatial coverage - Are training points well-distributed across the AOI?
2. Depth coverage - Do we have training data across the full depth range?
3. Spectral coverage - Do training points span the range of water/bottom conditions?
4. Track diversity - How many independent ICESat-2 passes contribute data?

The goal is to collect training data that spans the full "feature space" so the
model learns generalizable relationships rather than location-specific patterns.
"""

import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class DiversityReport:
    """Summary of training data diversity analysis."""
    # Spatial metrics
    spatial_coverage_score: float  # 0-1, higher = better coverage
    n_spatial_clusters: int
    cluster_sizes: List[int]
    coverage_gaps: List[Tuple[float, float, float, float]]  # [(lon_min, lon_max, lat_min, lat_max), ...]
    
    # Depth metrics
    depth_coverage_score: float
    depth_gaps: List[Tuple[float, float]]  # [(min, max), ...] - depth ranges with no data
    depth_histogram: Dict[str, int]
    
    # Spectral metrics
    spectral_coverage_score: float
    feature_ranges: Dict[str, Tuple[float, float]]
    
    # Track diversity
    n_unique_tracks: int
    track_contribution: Dict[str, int]  # track_id -> n_points
    
    # Overall
    overall_score: float
    recommendations: List[str]


def analyze_spatial_coverage(
    df: pd.DataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    n_grid_cells: int = 25,
) -> Tuple[float, List[Tuple[float, float, float, float]]]:
    """
    Analyze how well training points cover the AOI spatially.
    
    Returns coverage score (0-1) and list of gap regions.
    """
    lons = df[lon_col].to_numpy()
    lats = df[lat_col].to_numpy()
    
    lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
    lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
    
    # Create grid
    n_x = int(np.sqrt(n_grid_cells))
    n_y = n_x
    
    lon_edges = np.linspace(lon_min, lon_max, n_x + 1)
    lat_edges = np.linspace(lat_min, lat_max, n_y + 1)
    
    # Count points per cell
    occupied_cells = 0
    gaps = []
    min_points_per_cell = 10  # Threshold for "covered"
    
    for i in range(n_x):
        for j in range(n_y):
            mask = (
                (lons >= lon_edges[i]) & (lons < lon_edges[i+1]) &
                (lats >= lat_edges[j]) & (lats < lat_edges[j+1])
            )
            n_in_cell = np.sum(mask)
            
            if n_in_cell >= min_points_per_cell:
                occupied_cells += 1
            else:
                gaps.append((
                    float(lon_edges[i]), float(lon_edges[i+1]),
                    float(lat_edges[j]), float(lat_edges[j+1])
                ))
    
    coverage_score = occupied_cells / (n_x * n_y)
    return coverage_score, gaps


def analyze_depth_coverage(
    df: pd.DataFrame,
    depth_col: str = "depth_m",
    max_expected_depth: float = 20.0,
    bin_size: float = 1.0,
) -> Tuple[float, List[Tuple[float, float]], Dict[str, int]]:
    """
    Analyze depth distribution of training data.
    
    Returns coverage score, gap ranges, and histogram.
    """
    depths = pd.to_numeric(df[depth_col], errors="coerce").to_numpy(dtype=float)
    depths = depths[np.isfinite(depths) & (depths > 0)]
    
    if len(depths) == 0:
        return 0.0, [(0, max_expected_depth)], {}
    
    # Create bins
    bins = np.arange(0, max_expected_depth + bin_size, bin_size)
    hist, edges = np.histogram(depths, bins=bins)
    
    # Find gaps (bins with < 10 points)
    gaps = []
    min_points = 10
    in_gap = False
    gap_start = 0
    
    for i, count in enumerate(hist):
        if count < min_points:
            if not in_gap:
                in_gap = True
                gap_start = edges[i]
        else:
            if in_gap:
                gaps.append((float(gap_start), float(edges[i])))
                in_gap = False
    
    if in_gap:
        gaps.append((float(gap_start), float(edges[-1])))
    
    # Coverage score = fraction of depth range with adequate data
    covered_bins = np.sum(hist >= min_points)
    max_bin = int(np.ceil(np.max(depths) / bin_size))
    coverage_score = covered_bins / max(max_bin, 1)
    
    # Create histogram dict
    histogram = {f"{edges[i]:.0f}-{edges[i+1]:.0f}m": int(hist[i]) for i in range(len(hist))}
    
    return coverage_score, gaps, histogram


def analyze_spectral_diversity(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
) -> Tuple[float, Dict[str, Tuple[float, float]]]:
    """
    Analyze spectral/feature diversity of training data.
    
    Checks if training data spans the expected range of optical conditions.
    """
    if feature_cols is None:
        feature_cols = ["B02", "B03", "B04", "log_ratio_B02_B03", "stumpf_index"]
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if not feature_cols:
        return 0.5, {}  # Neutral score if no features available
    
    ranges = {}
    coverage_scores = []
    
    for col in feature_cols:
        # Coerce to numeric to avoid object/dict columns breaking np.isfinite
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        
        if len(vals) < 10:
            continue
        
        p5, p95 = np.percentile(vals, [5, 95])
        ranges[col] = (float(p5), float(p95))
        
        # Score based on dynamic range
        # Wider range = more diversity = better
        val_range = p95 - p5
        val_std = np.std(vals)
        
        # Coefficient of variation as diversity metric
        cv = val_std / (np.mean(np.abs(vals)) + 1e-6)
        coverage_scores.append(min(cv * 2, 1.0))  # Normalize to 0-1
    
    overall_score = np.mean(coverage_scores) if coverage_scores else 0.5
    return overall_score, ranges


def analyze_track_diversity(
    df: pd.DataFrame,
    track_col: str = "gt",
    date_col: str = "datetime",
    source_col: str = "source",
) -> Tuple[int, Dict[str, int]]:
    """
    Analyze ICESat-2 track diversity.
    
    More independent tracks = more diverse viewing conditions.
    """
    # Try to identify unique acquisition events
    if track_col in df.columns and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col]).dt.date.astype(str)
            tracks = df[track_col].astype(str)
            unique_passes = tracks + "_" + dates
        except Exception:
            unique_passes = df[track_col].astype(str) if track_col in df.columns else df[source_col].astype(str)
    elif track_col in df.columns:
        unique_passes = df[track_col].astype(str)
    elif source_col in df.columns:
        unique_passes = df[source_col].astype(str)
    else:
        return 1, {"unknown": len(df)}
    
    track_counts = unique_passes.value_counts().to_dict()
    n_unique = len(track_counts)
    
    return n_unique, {str(k): int(v) for k, v in track_counts.items()}


def compute_diversity_score(
    spatial: float,
    depth: float,
    spectral: float,
    n_tracks: int,
    weights: Dict[str, float] = None,
) -> float:
    """
    Compute overall diversity score from component scores.
    """
    if weights is None:
        weights = {
            "spatial": 0.3,
            "depth": 0.3,
            "spectral": 0.2,
            "tracks": 0.2,
        }
    
    # Track score: diminishing returns after ~10 tracks
    track_score = min(n_tracks / 10, 1.0)
    
    overall = (
        weights["spatial"] * spatial +
        weights["depth"] * depth +
        weights["spectral"] * spectral +
        weights["tracks"] * track_score
    )
    
    return float(overall)


def generate_recommendations(
    spatial_score: float,
    spatial_gaps: List,
    depth_score: float,
    depth_gaps: List,
    spectral_score: float,
    n_tracks: int,
    track_contribution: Dict[str, int],
) -> List[str]:
    """
    Generate actionable recommendations for improving training data.
    """
    recommendations = []
    
    # Spatial recommendations
    if spatial_score < 0.5:
        recommendations.append(
            f"SPATIAL: Only {spatial_score*100:.0f}% of AOI has adequate training data. "
            f"Consider adding ICESat-2 tracks that cross the {len(spatial_gaps)} gap regions."
        )
    elif spatial_score < 0.8:
        recommendations.append(
            f"SPATIAL: Coverage is moderate ({spatial_score*100:.0f}%). "
            f"Adding data in {len(spatial_gaps)} gap regions would improve generalization."
        )
    
    # Depth recommendations
    if depth_gaps:
        gap_str = ", ".join([f"{g[0]:.0f}-{g[1]:.0f}m" for g in depth_gaps[:3]])
        recommendations.append(
            f"DEPTH: Gaps in training data at depths: {gap_str}. "
            f"Model predictions will be less reliable in these ranges."
        )
    
    if depth_score < 0.5:
        recommendations.append(
            "DEPTH: Training data is concentrated in a narrow depth range. "
            "Consider adding tracks over areas with greater depth variation."
        )
    
    # Track recommendations
    if n_tracks < 3:
        recommendations.append(
            f"TRACKS: Only {n_tracks} unique ICESat-2 passes. "
            "This limits the model's ability to generalize across different acquisition conditions. "
            "Aim for 5+ independent passes."
        )
    elif n_tracks < 5:
        recommendations.append(
            f"TRACKS: {n_tracks} unique passes is adequate but more would improve robustness."
        )
    
    # Check for dominant track
    if track_contribution:
        total = sum(track_contribution.values())
        max_track = max(track_contribution.values())
        if max_track / total > 0.5:
            dominant = [k for k, v in track_contribution.items() if v == max_track][0]
            recommendations.append(
                f"TRACKS: Track '{dominant}' contributes {max_track/total*100:.0f}% of data. "
                "Model may overfit to conditions of this single pass."
            )
    
    # Spectral recommendations
    if spectral_score < 0.4:
        recommendations.append(
            "SPECTRAL: Low diversity in optical features. Training data may not "
            "represent the full range of water/bottom conditions in the AOI."
        )
    
    if not recommendations:
        recommendations.append(
            "Training data diversity looks good! Spatial CV performance should be "
            "close to random CV performance."
        )
    
    return recommendations


def analyze_training_diversity(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    max_expected_depth: float = 20.0,
) -> DiversityReport:
    """
    Comprehensive analysis of training data diversity.
    
    This is the main entry point for the module.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data with coordinates, depth, and features
    feature_cols : List[str], optional
        Feature columns to analyze for spectral diversity
    max_expected_depth : float
        Maximum expected depth for gap analysis
        
    Returns
    -------
    DiversityReport
        Comprehensive diversity metrics and recommendations
    """
    log.info("Analyzing training data diversity (%s points)...", len(df))
    
    # Spatial analysis
    spatial_score, spatial_gaps = analyze_spatial_coverage(df)
    
    # Cluster analysis for spatial distribution
    from sklearn.cluster import KMeans
    coords = df[["longitude", "latitude"]].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    valid = np.all(np.isfinite(coords), axis=1)
    n_clusters = min(10, len(df) // 100)
    if n_clusters >= 2 and np.sum(valid) >= n_clusters:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = np.full(len(df), -1)
        labels[valid] = km.fit_predict(coords[valid])
        cluster_sizes = [int(np.sum(labels == k)) for k in range(n_clusters)]
    else:
        cluster_sizes = [len(df)]
        n_clusters = 1
    
    # Depth analysis
    depth_score, depth_gaps, depth_hist = analyze_depth_coverage(
        df, max_expected_depth=max_expected_depth
    )
    
    # Spectral analysis
    spectral_score, feature_ranges = analyze_spectral_diversity(df, feature_cols)
    
    # Track analysis
    n_tracks, track_contrib = analyze_track_diversity(df)
    
    # Overall score
    overall = compute_diversity_score(
        spatial_score, depth_score, spectral_score, n_tracks
    )
    
    # Recommendations
    recommendations = generate_recommendations(
        spatial_score, spatial_gaps,
        depth_score, depth_gaps,
        spectral_score,
        n_tracks, track_contrib,
    )
    
    report = DiversityReport(
        spatial_coverage_score=spatial_score,
        n_spatial_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        coverage_gaps=spatial_gaps,
        depth_coverage_score=depth_score,
        depth_gaps=depth_gaps,
        depth_histogram=depth_hist,
        spectral_coverage_score=spectral_score,
        feature_ranges=feature_ranges,
        n_unique_tracks=n_tracks,
        track_contribution=track_contrib,
        overall_score=overall,
        recommendations=recommendations,
    )
    
    # Log summary
    log.info("Spatial coverage: %.0f%%", spatial_score*100)
    log.info("Depth coverage: %.0f%%", depth_score*100)
    log.info("Spectral diversity: %.0f%%", spectral_score*100)
    log.info("Unique tracks: %s", n_tracks)
    log.info("Overall score: %.0f%%", overall*100)
    for rec in recommendations:
        log.info("→ %s", rec)
    
    return report


def plot_diversity_analysis(
    df: pd.DataFrame,
    report: DiversityReport,
    output_path: Path,
    aoi_bounds: Tuple[float, float, float, float] = None,
):
    """
    Generate visualization of training data diversity.
    """
    from plot_utils import lazy_pyplot
    plt = lazy_pyplot()
    from matplotlib.patches import Rectangle
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Spatial distribution with gaps
    ax1 = axes[0, 0]
    ax1.scatter(
        df["longitude"], df["latitude"],
        c=df["depth_m"] if "depth_m" in df.columns else "blue",
        s=5, alpha=0.5, cmap="viridis_r"
    )
    
    # Highlight gap regions
    for gap in report.coverage_gaps[:10]:  # Show first 10 gaps
        lon_min, lon_max, lat_min, lat_max = gap
        rect = Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            fill=True, facecolor="red", alpha=0.2,
            edgecolor="red", linewidth=1
        )
        ax1.add_patch(rect)
    
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title(f"Spatial Coverage ({report.spatial_coverage_score*100:.0f}%)\nRed = gaps")
    
    # 2. Depth histogram
    ax2 = axes[0, 1]
    if report.depth_histogram:
        bins = list(report.depth_histogram.keys())
        counts = list(report.depth_histogram.values())
        colors = ["red" if c < 10 else "steelblue" for c in counts]
        ax2.barh(bins, counts, color=colors)
        ax2.set_xlabel("Number of points")
        ax2.set_ylabel("Depth range")
        ax2.set_title(f"Depth Distribution ({report.depth_coverage_score*100:.0f}% coverage)")
    
    # 3. Track contributions
    ax3 = axes[1, 0]
    if report.track_contribution:
        tracks = list(report.track_contribution.keys())[:10]  # Top 10
        counts = [report.track_contribution[t] for t in tracks]
        # Truncate long track names
        tracks = [t[:20] + "..." if len(t) > 20 else t for t in tracks]
        ax3.barh(tracks, counts, color="forestgreen")
        ax3.set_xlabel("Number of points")
        ax3.set_title(f"Track Diversity ({report.n_unique_tracks} unique)")
    
    # 4. Summary scores
    ax4 = axes[1, 1]
    categories = ["Spatial", "Depth", "Spectral", "Tracks", "Overall"]
    track_score = min(report.n_unique_tracks / 10, 1.0)
    scores = [
        report.spatial_coverage_score,
        report.depth_coverage_score,
        report.spectral_coverage_score,
        track_score,
        report.overall_score,
    ]
    colors = ["green" if s > 0.7 else "orange" if s > 0.4 else "red" for s in scores]
    ax4.barh(categories, scores, color=colors)
    ax4.set_xlim(0, 1)
    ax4.set_xlabel("Score (0-1)")
    ax4.set_title("Diversity Scores")
    ax4.axvline(0.7, color="green", linestyle="--", alpha=0.5, label="Good")
    ax4.axvline(0.4, color="orange", linestyle="--", alpha=0.5, label="Moderate")
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    log.info("Saved analysis plot: %s", output_path)


def suggest_additional_tracks(
    df: pd.DataFrame,
    aoi_bounds: Tuple[float, float, float, float],
    depth_gaps: List[Tuple[float, float]] = None,
    spatial_gaps: List[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Suggest where to look for additional ICESat-2 tracks.
    
    Returns suggestions for OpenAltimetry or other data sources.
    """
    west, east, south, north = aoi_bounds
    
    suggestions = {
        "openaltimetry_url": f"https://openaltimetry.earthdatacloud.nasa.gov/data/icesat2/products?maxx={east}&maxy={north}&minx={west}&miny={south}",
        "spatial_priority_regions": [],
        "depth_priority": [],
    }
    
    # Prioritize spatial gaps
    if spatial_gaps:
        for gap in spatial_gaps[:5]:
            lon_min, lon_max, lat_min, lat_max = gap
            center_lon = (lon_min + lon_max) / 2
            center_lat = (lat_min + lat_max) / 2
            suggestions["spatial_priority_regions"].append({
                "center": (center_lon, center_lat),
                "bounds": gap,
                "note": "Look for ICESat-2 tracks crossing this region"
            })
    
    # Depth priorities
    if depth_gaps:
        suggestions["depth_priority"] = [
            f"Need data at {g[0]:.0f}-{g[1]:.0f}m depth" for g in depth_gaps
        ]
    
    suggestions["general_tips"] = [
        "Use OpenAltimetry to find ATL03/ATL24 tracks over your AOI",
        "Prioritize tracks from different seasons (varying sun angle, water clarity)",
        "Include tracks over different bottom types (sand, seagrass, coral)",
        "Aim for 5-10 independent tracks for robust training",
    ]
    
    return suggestions


# Integration with training pipeline
def add_diversity_analysis_to_training(
    df: pd.DataFrame,
    plots_dir: Path,
    feature_cols: List[str] = None,
) -> Dict[str, Any]:
    """
    Run diversity analysis and add to training metadata.
    
    Call this from train_sdb_model() to include diversity metrics.
    """
    try:
        report = analyze_training_diversity(df, feature_cols)
        
        if plots_dir:
            plot_diversity_analysis(
                df, report,
                plots_dir / "Training_Data_Diversity.png"
            )
        
        # Convert to JSON-serializable dict
        return {
            "spatial_coverage_score": report.spatial_coverage_score,
            "depth_coverage_score": report.depth_coverage_score,
            "spectral_coverage_score": report.spectral_coverage_score,
            "n_unique_tracks": report.n_unique_tracks,
            "overall_score": report.overall_score,
            "n_spatial_gaps": len(report.coverage_gaps),
            "depth_gaps": report.depth_gaps,
            "recommendations": report.recommendations,
        }
        
    except Exception as e:
        log.warning("Analysis failed: %s", e)
        return {"error": str(e)}


if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze training data diversity")
    parser.add_argument("--input-csv", required=True, help="Training data CSV")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--max-depth", type=float, default=20.0, help="Max expected depth")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    df = pd.read_csv(args.input_csv)
    report = analyze_training_diversity(df, max_expected_depth=args.max_depth)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_diversity_analysis(df, report, output_dir / "diversity_analysis.png")
    
    # Save report
    import json
    report_dict = {
        "spatial_coverage_score": report.spatial_coverage_score,
        "depth_coverage_score": report.depth_coverage_score,
        "spectral_coverage_score": report.spectral_coverage_score,
        "n_unique_tracks": report.n_unique_tracks,
        "overall_score": report.overall_score,
        "recommendations": report.recommendations,
    }
    with open(output_dir / "diversity_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    
    log.info("Recommendations:")
    for rec in report.recommendations:
        log.info("  • %s", rec)