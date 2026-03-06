#!/usr/bin/env python3
"""compute_seam_stability_summary.py

Aggregate seam metrics + optional bank drift into seam_stability_summary.json and apply readiness gates.

Required for 'ready' (unless you intentionally omit a component):
- coastal seam CSV
- river seam CSV
- transition CSV

If bank drift CSV is provided and exists, convergence is required too.

"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np


def _read_single_row(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _f(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _gate_seam(row: Dict[str, Any], bias_max: float, rmse_max: float, p95_max: Optional[float], overlap_min: float):
    bias = abs(_f(row.get("seam_bias")))
    rmse = _f(row.get("seam_rmse"))
    p95  = _f(row.get("seam_p95_abs"))
    ov   = _f(row.get("overlap_valid_frac"))
    ok = True
    reasons = []
    if not np.isfinite(ov) or ov < overlap_min:
        ok = False; reasons.append(f"overlap_valid_frac<{overlap_min}")
    if not np.isfinite(rmse) or rmse > rmse_max:
        ok = False; reasons.append(f"seam_rmse>{rmse_max}")
    if not np.isfinite(bias) or bias > bias_max:
        ok = False; reasons.append(f"|seam_bias|>{bias_max}")
    if p95_max is not None:
        if not np.isfinite(p95) or p95 > p95_max:
            ok = False; reasons.append(f"seam_p95_abs>{p95_max}")
    return {"pass": ok, "reasons": reasons, "values": {"bias_abs": bias, "rmse": rmse, "p95_abs": p95, "overlap": ov}}


def _gate_transition(row: Dict[str, Any], bias_max: float, rmse_max: float, overlap_min: float):
    bias = abs(_f(row.get("transition_bias")))
    rmse = _f(row.get("transition_rmse"))
    ov   = _f(row.get("overlap_valid_frac"))
    ok = True
    reasons = []
    if not np.isfinite(ov) or ov < overlap_min:
        ok = False; reasons.append(f"overlap_valid_frac<{overlap_min}")
    if not np.isfinite(rmse) or rmse > rmse_max:
        ok = False; reasons.append(f"transition_rmse>{rmse_max}")
    if not np.isfinite(bias) or bias > bias_max:
        ok = False; reasons.append(f"|transition_bias|>{bias_max}")
    return {"pass": ok, "reasons": reasons, "values": {"bias_abs": bias, "rmse": rmse, "overlap": ov}}


def _gate_convergence(path: str, p95_thresh: float, n_consec: int):
    p = Path(path)
    if not p.exists():
        return {"pass": False, "reasons": ["bank_drift_missing"], "values": {}}
    df = pd.read_csv(p)
    for col in ["probe_p95_abs_delta_m", "probe_p95_abs_delta"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values
            if vals.size < n_consec:
                return {"pass": False, "reasons": [f"insufficient_updates<{n_consec}"], "values": {"n_updates": int(vals.size)}}
            last = vals[-n_consec:]
            ok = bool(np.all(last <= p95_thresh))
            reasons = [] if ok else [f"probe_p95_abs_delta_m>{p95_thresh} in last {n_consec}"]
            return {"pass": ok, "reasons": reasons, "values": {"last": last.tolist(), "threshold": p95_thresh}}
    return {"pass": False, "reasons": ["missing_metric_column"], "values": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coastal-seam", default=None)
    ap.add_argument("--river-seam", default=None)
    ap.add_argument("--fusion-seam", default=None)
    ap.add_argument("--transition", default=None)
    ap.add_argument("--bank-drift", default=None)

    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", default=None)

    # thresholds
    ap.add_argument("--coastal-bias-max", type=float, default=0.15)
    ap.add_argument("--coastal-rmse-max", type=float, default=0.50)
    ap.add_argument("--coastal-p95-max", type=float, default=1.0)
    ap.add_argument("--coastal-overlap-min", type=float, default=0.10)

    ap.add_argument("--river-bias-max", type=float, default=0.15)
    ap.add_argument("--river-rmse-max", type=float, default=0.50)
    ap.add_argument("--river-overlap-min", type=float, default=0.10)

    ap.add_argument("--transition-bias-max", type=float, default=0.25)
    ap.add_argument("--transition-rmse-max", type=float, default=0.75)
    ap.add_argument("--transition-overlap-min", type=float, default=0.05)

    ap.add_argument("--convergence-p95-thresh", type=float, default=0.25)
    ap.add_argument("--convergence-n", type=int, default=3)

    args = ap.parse_args()

    results = {}
    reasons: List[str] = []

    coastal = _read_single_row(args.coastal_seam)
    river = _read_single_row(args.river_seam)
    fusion = _read_single_row(args.fusion_seam)
    trans = _read_single_row(args.transition)

    if coastal is None:
        results["coastal_seam"] = {"pass": False, "reasons": ["missing_or_empty"], "values": {}}
        reasons.append("missing_coastal_seam_metrics")
    else:
        results["coastal_seam"] = _gate_seam(coastal, args.coastal_bias_max, args.coastal_rmse_max, args.coastal_p95_max, args.coastal_overlap_min)

    if river is None:
        results["river_seam"] = {"pass": False, "reasons": ["missing_or_empty"], "values": {}}
        reasons.append("missing_river_seam_metrics")
    else:
        results["river_seam"] = _gate_seam(river, args.river_bias_max, args.river_rmse_max, None, args.river_overlap_min)

    if trans is None:
        results["transition"] = {"pass": False, "reasons": ["missing_or_empty"], "values": {}}
        reasons.append("missing_transition_metrics")
    else:
        results["transition"] = _gate_transition(trans, args.transition_bias_max, args.transition_rmse_max, args.transition_overlap_min)

    if fusion is None:
        results["fusion_seam"] = {"pass": False, "reasons": ["missing_or_empty"], "values": {}}
    else:
        results["fusion_seam"] = _gate_seam(fusion, args.coastal_bias_max, args.coastal_rmse_max, args.coastal_p95_max, args.coastal_overlap_min)

    if args.bank_drift and Path(args.bank_drift).exists():
        results["convergence"] = _gate_convergence(args.bank_drift, args.convergence_p95_thresh, args.convergence_n)
    else:
        results["convergence"] = {"pass": False, "reasons": ["not_provided_or_missing"], "values": {}}

    # readiness logic
    ready = bool(results["coastal_seam"]["pass"] and results["river_seam"]["pass"] and results["transition"]["pass"])
    if args.bank_drift and Path(args.bank_drift).exists():
        ready = ready and results["convergence"]["pass"]

    status = "ready" if ready else "degraded"

    for k, v in results.items():
        if not v.get("pass", False):
            for r in v.get("reasons", []):
                reasons.append(f"{k}:{r}")

    summary = {"status": status, "ready": ready, "results": results, "reasons": sorted(set(reasons))}

    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    if args.out_csv:
        out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
        flat = {
            "status": status,
            "ready": ready,
            "coastal_pass": results["coastal_seam"]["pass"],
            "river_pass": results["river_seam"]["pass"],
            "transition_pass": results["transition"]["pass"],
            "fusion_pass": results["fusion_seam"]["pass"],
            "convergence_pass": results["convergence"]["pass"],
        }
        pd.DataFrame([flat]).to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()
