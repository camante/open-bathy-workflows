#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_report.py — Lightweight "flight recorder" for the SDB pipeline.

Adds:
- Two auto-emitted funnel images per run (if matplotlib is available):
    1) Funnel_ATL_Points.png
    2) Funnel_SDB_Prediction_Cells.png
- Optional CLI to emit funnels from an existing run_report.json

The funnels are designed to help you see *where* depth / validity filtering happens.
They rely on keys populated across atl.py, fusion.py, train.py, predict.py and sdb_main.py.
If keys are missing, the plots will still render but will show "missing" markers.
"""


import logging
log = logging.getLogger(__name__)
import datetime
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

plt = None  # lazy-loaded matplotlib.pyplot (set via plot_utils.lazy_pyplot)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _deep_get(d: Dict[str, Any], dotted_key: str, default=None):
    cur: Any = d
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def _fmt_int(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _maybe_import_matplotlib():
    """Best-effort import of matplotlib via plot_utils.lazy_pyplot."""
    global plt
    try:
        from plot_utils import lazy_pyplot
        if plt is None:
            plt = lazy_pyplot()
        return True
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Stage timer
# -----------------------------------------------------------------------------

class _StageTimer:
    def __init__(self, rr: "RunReport", name: str):
        self.rr = rr
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        self.rr._stage_stack.append(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - (self.t0 or time.time())
        self.rr._stage_stack.pop()
        self.rr.data.setdefault("timing", {})
        self.rr.data["timing"][self.name] = float(dt)

# -----------------------------------------------------------------------------
# Public helpers often used by pipeline modules
# -----------------------------------------------------------------------------


def raster_quickstats(path: str) -> Dict[str, Any]:
    """Minimal raster stats.

    Notes:
      - Uses a writable NumPy copy (avoids 'output array is read-only' failures seen with masked arrays).
      - Treats dataset nodata (if set) as NaN for stats.
    """
    try:
        import numpy as np
        import rasterio
        p = Path(path)
        if not p.exists():
            return {"path": str(p), "exists": False}

        with rasterio.open(p) as ds:
            # Make an explicit writable copy
            a = ds.read(1).astype("float64", copy=True)
            nodata = ds.nodata

        if nodata is not None:
            # Some rasters store nodata as float or int; compare in float64
            a[a == float(nodata)] = np.nan

        finite = np.isfinite(a)
        if not finite.any():
            return {"path": str(p), "exists": True, "finite": 0, "shape": [int(a.shape[0]), int(a.shape[1])]}

        vals = a[finite]
        out = {
            "path": str(p),
            "exists": True,
            "shape": [int(a.shape[0]), int(a.shape[1])],
            "finite": int(vals.size),
            "min": float(vals.min()),
            "p50": float(np.nanpercentile(vals, 50)),
            "p95": float(np.nanpercentile(vals, 95)),
            "max": float(vals.max()),
        }
        return out
    except Exception as e:
        return {"path": str(path), "error": str(e)}


def mask_fraction(mask_path: str, true_value: Optional[int] = None) -> Dict[str, Any]:
    """Fraction of pixels that are valid/true. If true_value is None, any nonzero counts as true."""
    try:
        import numpy as np
        import rasterio
        p = Path(mask_path)
        if not p.exists():
            return {"path": str(p), "exists": False}
        with rasterio.open(p) as ds:
            m = ds.read(1)
        tot = int(m.size)
        if tot == 0:
            return {"path": str(p), "exists": True, "total": 0}
        if true_value is None:
            tv = int(np.count_nonzero(m))
        else:
            tv = int(np.count_nonzero(m == int(true_value)))
        return {"path": str(p), "exists": True, "total": tot, "true": tv, "frac": float(tv / tot)}
    except Exception as e:
        return {"path": str(mask_path), "error": str(e)}

# -----------------------------------------------------------------------------
# Funnel renderers (PNG)
# -----------------------------------------------------------------------------

def _render_funnel_panel(ax, title: str, stages: List[Tuple[str, Any]], *, hist: Optional[Tuple[List[float], List[str]]] = None):
    """Draw a funnel visualization with horizontal bars showing relative counts."""
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.axis("off")

    if not stages or (len(stages) == 1 and stages[0][1] is None):
        ax.text(0.5, 0.5, f"No {title} stages found", transform=ax.transAxes, 
                ha="center", va="center", fontsize=10, style='italic', color='gray')
        return

    # Find max value for scaling bars
    max_val = max((v for _, v in stages if isinstance(v, (int, float)) and v is not None), default=1)
    if max_val <= 0:
        max_val = 1

    # Layout parameters
    n_stages = len(stages)
    y_start = 0.92
    y_end = 0.08
    line_h = (y_start - y_end) / max(n_stages, 1)
    bar_height = line_h * 0.6
    bar_max_width = 0.45  # Max bar width as fraction of axes

    for i, (label, val) in enumerate(stages):
        y = y_start - i * line_h
        
        # Clean up label for display
        display_label = label.replace("_", " ").replace(".", " → ")
        if len(display_label) > 35:
            display_label = display_label[:32] + "..."
        
        # Draw bar if we have a numeric value
        if isinstance(val, (int, float)) and val is not None and val > 0:
            bar_width = (val / max_val) * bar_max_width
            # Color gradient: green for high retention, red for low
            retention = val / max_val
            global plt
            if plt is None:
                plt = lazy_pyplot()

            color = plt.cm.RdYlGn(retention * 0.8 + 0.1)  # Avoid extremes
            
            rect = plt.Rectangle((0.52, y - bar_height/2), bar_width, bar_height,
                                  facecolor=color, edgecolor='gray', linewidth=0.5,
                                  transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            
            # Value text at end of bar
            val_str = f"{int(val):,}" if isinstance(val, int) or val == int(val) else f"{val:.2f}"
            ax.text(0.52 + bar_width + 0.02, y, val_str, transform=ax.transAxes,
                    ha="left", va="center", fontsize=9, fontweight='bold')
        else:
            val_str = "—" if val is None else str(val)
            ax.text(0.52, y, val_str, transform=ax.transAxes,
                    ha="left", va="center", fontsize=9, color='gray')
        
        # Label text on left
        ax.text(0.50, y, display_label, transform=ax.transAxes,
                ha="right", va="center", fontsize=9)
        
        # Draw arrow if not last stage
        if i < n_stages - 1:
            ax.annotate("", xy=(0.25, y - line_h * 0.7), xytext=(0.25, y - line_h * 0.3),
                        transform=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color='gray', lw=0.8))

def _emit_funnel_pngs_from_data(data: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    """
    Emits two PNGs into out_dir/plots and returns artifact mapping.

    This version is *schema-aligned* with your current pipeline:
      - atl.py writes:   funnel.<stage>.n, funnel.<stage>.depth.*
      - fusion.py writes: funnel.<stage> = {n, finite_n, p50, ...}
      - train.py writes: funnel.train.<stage>.n (+ depth stats/hists)
      - predict.py writes: predict.fractions.{pixels_total, finite_optical, ...}

    The funnel labels are the *actual stage keys* so you can directly cross-walk
    plot ↔ run_report.json ↔ code.
    """
    artifacts: Dict[str, str] = {}
    if not _maybe_import_matplotlib():
        return artifacts

    global plt  # set by _maybe_import_matplotlib()


    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _get_n(stage_key: str) -> Optional[int]:
        """Return best-effort stage count for a stage key like 'atl03.points.concat' or 'train.start'."""
        # 1) Preferred: funnel.<stage>.n
        v = _deep_get(data, f"funnel.{stage_key}.n", None)
        if isinstance(v, (int, float)) and v is not None:
            return int(v)
        # 2) Fusion style: funnel.<stage> is a dict containing n
        v2 = _deep_get(data, f"funnel.{stage_key}", None)
        if isinstance(v2, dict) and "n" in v2 and isinstance(v2["n"], (int, float)):
            return int(v2["n"])
        # 3) Sometimes people store the stage itself as a number
        if isinstance(v2, (int, float)):
            return int(v2)
        return None

    def _pick_existing(stages: List[str]) -> List[Tuple[str, Any]]:
        out: List[Tuple[str, Any]] = []
        for st in stages:
            n = _get_n(st)
            if n is not None:
                out.append((st, n))
        return out

    # ---------------------------
    # Funnel 1: ATL-derived points through stages (ATL → Fusion → Train)
    # ---------------------------
    # Ordered to show progressive filtering flow
    atl_stage_candidates = [
        # ATL03 processing flow
        "atl03.points.concat",           # Raw concatenated photons
        "atl03.points.aoi_clip",         # Clipped to AOI
        "atl03.points.confidence_filter", # Confidence filtered
        "atl03.points.depth_filter",     # Depth range filtered
        "atl03.points.landmask",         # Land masked
        # ATL24 processing flow
        "atl24.points.concat",
        "atl24.points.aoi_clip",
        "atl24.points.pre_depth_cap",
        "atl24.points.post_depth_cap",
        "atl24.points.landmask",
        # Combined ATL mask filter stages
        "atl.mask_filter.input",
        "atl.mask_filter.output",
    ]

    fusion_stage_candidates = [
        # Inputs to fusion
        "fusion.input.atl03",
        "fusion.input.atl24",
        "fusion.input.xyz",
        # Fusion processing
        "fusion.xyz_filter.atl_combined.pre",
        "fusion.xyz_filter.atl_filtered.post",
        "fusion.atl_fuse.post",
        "fusion.combined.post",
        "fusion.schema_weighted.final",
    ]

    train_stage_candidates = [
        # Training pipeline flow
        "train.start",                    # Input to training
        "train.after_env_filter",         # After land/water mask
        "train.after_s2_features",        # After S2 band sampling
        "train.after_brightness_filter",  # After brightness QC
        "train.after_stumpf_residual_filter",  # After Stumpf outlier removal
        "train.after_finite_row_drop",    # After NaN removal
        "train.split_train",              # Training set size
        "train.split_test",               # Test set size
    ]

    atl_stages = _pick_existing(atl_stage_candidates)
    fusion_stages = _pick_existing(fusion_stage_candidates)
    train_stages = _pick_existing(train_stage_candidates)

    if atl_stages or fusion_stages or train_stages:
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        _render_funnel_panel(ax1, "ATL", atl_stages if atl_stages else [("no ATL stages found", None)])
        _render_funnel_panel(ax2, "Fusion", fusion_stages if fusion_stages else [("no Fusion stages found", None)])
        _render_funnel_panel(ax3, "Train / Split", train_stages if train_stages else [("no Train stages found", None)])

        fig.suptitle("Funnel — ATL-derived points through pipeline stages", fontsize=16, y=0.98)
        out_png = plots_dir / "Funnel_ATL_Points.png"
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        artifacts["Funnel_ATL_Points.png"] = str(out_png)

    # ---------------------------
    # Funnel 2: SDB prediction grid cell masking & filtering (Predict)
    # ---------------------------
    pred = _deep_get(data, "predict.fractions", {}) or {}
    # Backward compat: some runs may store under predict.funnel or funnel.predict.*
    if not isinstance(pred, dict) or len(pred) == 0:
        pred = _deep_get(data, "predict.funnel", {}) or {}
    if not isinstance(pred, dict) or len(pred) == 0:
        pred = _deep_get(data, "funnel.predict", {}) or {}

    # We prefer absolute counts if present (these are what your predict.py tracks).
    # If the run_report stored fractions only, we'll still show those (as floats).
    pred_order = [
        ("pixels_total", "Total grid cells"),
        ("finite_optical", "Finite optical"),
        ("cw_pass", "Clear water / SCL pass"),
        ("land_pass", "Land mask pass"),
        ("base_valid", "Base valid (after env gates)"),
        ("doa_pass", "Domain-of-applicability pass"),
        ("predicted", "Predicted"),
        ("pixels_with_uncertainty", "With uncertainty"),
        ("pixels_high_uncertainty", "High uncertainty"),
    ]

    pred_stages: List[Tuple[str, Any]] = []
    for k, label in pred_order:
        v = pred.get(k, None)
        if isinstance(v, (int, float)):
            # If it looks like a fraction (0..1), keep as float; else int
            if 0.0 <= float(v) <= 1.0 and k not in ("pixels_total",):
                pred_stages.append((f"{label} [{k}] (fraction)", float(v)))
            else:
                pred_stages.append((f"{label} [{k}]", int(v)))
        elif v is not None:
            pred_stages.append((f"{label} [{k}]", v))

    if pred_stages:
        fig, ax = plt.subplots(figsize=(10, 5))
        _render_funnel_panel(ax, "SDB prediction cells", pred_stages)
        fig.suptitle("Funnel — Prediction masking & filtering", fontsize=16, y=0.98)
        out_png = plots_dir / "Funnel_SDB_Prediction_Cells.png"
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        artifacts["Funnel_SDB_Prediction_Cells.png"] = str(out_png)

    return artifacts
@dataclass
class RunReport:
    out_dir: Path
    data: Dict[str, Any] = field(default_factory=dict)
    _stage_stack: list = field(default_factory=list)

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.data.setdefault("run", {})
        self.data["run"].setdefault("timestamp_utc", _now_utc_iso())
        self.data["run"].setdefault("cwd", os.getcwd())
        self.data.setdefault("timing", {})
        self.data.setdefault("funnel", {})
        self.data.setdefault("artifacts", {})

    def add(self, key: str, value: Any) -> None:
        _deep_set(self.data, key, value)

    def add_dict(self, key: str, d: Dict[str, Any]) -> None:
        parts = key.split(".")
        cur = self.data
        for p in parts:
            cur = cur.setdefault(p, {})
        if isinstance(cur, dict):
            _deep_merge(cur, d)
        else:
            _deep_set(self.data, key, d)

    def merge(self, d: Dict[str, Any]) -> None:
        _deep_merge(self.data, d)

    def stage(self, name: str):
        return _StageTimer(self, name)

    def record_artifact(self, name: str, path: str) -> None:
        self.data.setdefault("artifacts", {})
        self.data["artifacts"][name] = str(path)

    def write(self, status: str = "ok", error: Optional[str] = None) -> Path:
        """Write run_report.json and attempt to emit funnel PNGs."""
        self.data.setdefault("run", {})
        self.data["run"]["status"] = str(status)
        if error:
            self.data["run"]["error"] = str(error)

        out_path = Path(self.out_dir) / "run_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.data, indent=2))

        # Emit funnel PNGs (best-effort, never raises)
        try:
            artifacts = _emit_funnel_pngs_from_data(self.data, Path(self.out_dir))
            for k, v in artifacts.items():
                self.record_artifact(k, v)
            if artifacts:
                out_path.write_text(json.dumps(self.data, indent=2))
        except Exception:
            logging.getLogger(__name__).debug("Optional step failed; continuing.", exc_info=True)

        return out_path

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Emit funnel PNGs from an existing run_report.json")
    ap.add_argument("run_report_json", help="Path to run_report.json")
    ap.add_argument("--out-dir", default="", help="Override out_dir (default: run_report_json parent)")
    args = ap.parse_args()

    rr_path = Path(args.run_report_json)
    data = json.loads(rr_path.read_text())
    out_dir = Path(args.out_dir) if args.out_dir else rr_path.parent

    artifacts = _emit_funnel_pngs_from_data(data, out_dir)
    if artifacts:
        data.setdefault("artifacts", {})
        data["artifacts"].update(artifacts)
        rr_path.write_text(json.dumps(data, indent=2))
        log.info("Wrote:", ", ".join(str(Path(v)) for v in artifacts.values()))
    else:
        log.info("No funnels written (matplotlib missing or no output dir).")

if __name__ == "__main__":
    _cli()
