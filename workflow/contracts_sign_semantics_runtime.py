from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rasterio

from sign_semantics import raster_value_semantics, summarize_numeric_sign, semantics_from_soundings_mode
import logging
log = logging.getLogger(__name__)



CANONICAL_EXPECTED = {"absolute_elevation", "depth_positive_down", "depth_negative_down"}


def _read_raster_summary(path: Path) -> Dict[str, Any]:
    with rasterio.open(path) as ds:
        tags = ds.tags() or {}
        arr = ds.read(1, out_shape=(min(256, ds.height), min(256, ds.width))).astype("float64")
        m = np.isfinite(arr)
        nod = ds.nodata
        if nod is not None:
            m &= arr != float(nod)
        vals = arr[m]
        summary = summarize_numeric_sign(vals) if vals.size else None
        return {
            "path": str(path),
            "tags": {str(k): str(v) for k, v in tags.items()},
            "declared_semantics": raster_value_semantics(tags),
            "dtype": str(ds.dtypes[0]),
            "shape": [int(ds.height), int(ds.width)],
            "summary": None if summary is None else {
                "valid": int(summary.valid),
                "frac_neg": float(summary.frac_neg),
                "frac_pos": float(summary.frac_pos),
                "p01": float(summary.p01),
                "p50": float(summary.p50),
                "p99": float(summary.p99),
            },
        }


def _result(name: str, status: str, message: str, *, path: Optional[str] = None, expected: Optional[str] = None, observed: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"name": name, "status": status, "message": message}
    if path:
        out["path"] = path
    if expected:
        out["expected"] = expected
    if observed:
        out["observed"] = observed
    if details:
        out["details"] = details
    return out


def _artifact_candidates(cfg: Any, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs = report.get("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
    ab_out = report.get("authoritative_base", {}).get("outputs", {}) if isinstance(report.get("authoritative_base", {}), dict) else {}
    river_out = report.get("river", {}).get("outputs", {}) if isinstance(report.get("river", {}), dict) else {}
    items: List[Dict[str, Any]] = []
    def add(name: str, path: Optional[str], expected: str, *, required: bool = False):
        if path:
            items.append({"name": name, "path": str(path), "expected": expected, "required": required})
        elif required:
            items.append({"name": name, "path": None, "expected": expected, "required": required})
    add("authoritative_base", getattr(cfg, "authoritative_base", None), "absolute_elevation")
    add("aligned_authoritative_base", ab_out.get("aligned_authoritative_base"), "absolute_elevation")
    add("final_depth_native", outputs.get("final_depth_native") or outputs.get("selected_final_depth"), "absolute_elevation", required=True)
    add("river_bottom_native", river_out.get("bottom_elevation"), "absolute_elevation")
    add("river_depth_native", river_out.get("depth_terrain"), "depth_negative_down")
    add("river_bottom_warped", outputs.get("river_bottom_warped"), "absolute_elevation")
    add("river_warped", outputs.get("river_warped"), "depth_negative_down")
    add("sdb_warped", outputs.get("sdb_warped"), "absolute_elevation")
    add("combined_bottom_navd88", outputs.get("combined_bottom_navd88"), "absolute_elevation")
    return items


def run_sign_semantics_runtime_contracts(cfg: Any, report: Dict[str, Any], *, contracts_dir: Optional[Path] = None) -> Dict[str, Any]:
    contracts_dir = Path(contracts_dir or (Path(cfg.out_dir) / "contracts"))
    contracts_dir.mkdir(parents=True, exist_ok=True)
    suite: Dict[str, Any] = {
        "stage": "sign_semantics_runtime",
        "pass": 0,
        "warn": 0,
        "fail": 0,
        "results": [],
    }

    for item in _artifact_candidates(cfg, report):
        name = item["name"]
        expected = item["expected"]
        raw_path = item.get("path")
        required = bool(item.get("required", False))
        if raw_path is None:
            suite["warn"] += 1
            suite["results"].append(_result(name, "warning", "Expected artifact path unavailable for semantic check.", expected=expected))
            continue
        path = Path(str(raw_path))
        if not path.exists():
            if required:
                suite["fail"] += 1
                suite["results"].append(_result(name, "error", "Required artifact missing for semantic check.", path=str(path), expected=expected))
            else:
                suite["warn"] += 1
                suite["results"].append(_result(name, "warning", "Artifact missing; semantic check skipped.", path=str(path), expected=expected))
            continue
        try:
            info = _read_raster_summary(path)
        except Exception as exc:  # noqa: BLE001
            log.debug("run_sign_semantics_runtime_contracts: suppressed exception", exc_info=True)
            suite["fail"] += 1
            suite["results"].append(_result(name, "error", f"Failed to inspect raster semantics: {exc}", path=str(path), expected=expected))
            continue

        observed = info["declared_semantics"]
        if observed == "unknown":
            suite["warn"] += 1
            suite["results"].append(_result(name, "warning", "Raster has no explicit recognized semantic tags.", path=str(path), expected=expected, observed=observed, details=info))
            continue
        if observed != expected:
            suite["fail"] += 1
            suite["results"].append(_result(name, "error", "Declared raster semantics do not match expected runtime semantics.", path=str(path), expected=expected, observed=observed, details=info))
            continue

        # Secondary sanity checks: warnings only.
        stats = info.get("summary") or {}
        if expected == "absolute_elevation" and stats.get("valid", 0) > 0 and stats.get("frac_neg") is not None:
            # No sign requirement for elevation; this is just informative.
            msg = "Absolute-elevation semantics verified."
        elif expected == "depth_negative_down" and stats.get("valid", 0) > 0 and float(stats.get("frac_neg", 0.0)) < 0.2:
            suite["warn"] += 1
            suite["results"].append(_result(name, "warning", "Depth raster is tagged negative-down but sampled values are not mostly negative.", path=str(path), expected=expected, observed=observed, details=info))
            continue
        elif expected == "depth_positive_down" and stats.get("valid", 0) > 0 and float(stats.get("frac_pos", 0.0)) < 0.2:
            suite["warn"] += 1
            suite["results"].append(_result(name, "warning", "Depth raster is tagged positive-down but sampled values are not mostly positive.", path=str(path), expected=expected, observed=observed, details=info))
            continue
        suite["pass"] += 1
        suite["results"].append(_result(name, "pass", "Semantic tags match expected runtime semantics.", path=str(path), expected=expected, observed=observed, details=info))

    summary = {
        "stage": suite["stage"],
        "pass": suite["pass"],
        "warn": suite["warn"],
        "fail": suite["fail"],
        "all_ok": suite["fail"] == 0,
    }
    suite["summary"] = summary
    json_path = contracts_dir / "contracts_sign_semantics.json"
    md_path = contracts_dir / "contracts_sign_semantics.md"
    json_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    lines = [
        "# Runtime sign semantics contracts",
        "",
        f"Pass: {suite['pass']}",
        f"Warn: {suite['warn']}",
        f"Fail: {suite['fail']}",
        "",
    ]
    for r in suite["results"]:
        lines.append(f"- **{r['status'].upper()}** `{r['name']}` — {r['message']}")
        if r.get("path"):
            lines.append(f"  - path: `{r['path']}`")
        if r.get("expected") or r.get("observed"):
            lines.append(f"  - expected/observed: `{r.get('expected')}` / `{r.get('observed')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    suite["json_path"] = str(json_path)
    suite["md_path"] = str(md_path)
    report.setdefault("contracts", {})["sign_semantics"] = summary
    report.setdefault("outputs", {})["contracts_sign_semantics_json"] = str(json_path)
    report.setdefault("outputs", {})["contracts_sign_semantics_md"] = str(md_path)
    return suite


def run_sdb_sign_semantics_runtime_contracts(args: Any, *, contracts_dir: Optional[Path] = None, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Standalone SDB semantic contracts for the written SDB outputs."""
    out_root = Path(getattr(args, "out_dir", "output"))
    dir_rast = out_root / "rasters"
    report = report or {}
    outputs = report.setdefault("outputs", {})
    # Primary prediction output and optional NAVD88 conversion are both bed-elevation rasters.
    pred_candidates = sorted(dir_rast.glob("*_sdb_depth.tif"))
    navd_candidates = sorted(dir_rast.glob("*_sdb_depth_bed_navd88.tif"))
    if pred_candidates:
        outputs.setdefault("final_depth_native", str(pred_candidates[0]))
    if navd_candidates:
        outputs.setdefault("sdb_navd88_native", str(navd_candidates[0]))
    # Reuse the common runtime contracts with a tiny synthetic report.
    suite = run_sign_semantics_runtime_contracts(args, report, contracts_dir=contracts_dir)
    return suite


def _stage_result(name: str, status: str, message: str, **extra: Any) -> Dict[str, Any]:
    out = {"name": name, "status": status, "message": message}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def _write_stage_suite(suite: Dict[str, Any], contracts_dir: Path, stem: str) -> Dict[str, Any]:
    contracts_dir.mkdir(parents=True, exist_ok=True)
    json_path = contracts_dir / f"{stem}.json"
    md_path = contracts_dir / f"{stem}.md"
    summary = {
        "stage": suite["stage"],
        "pass": suite["pass"],
        "warn": suite["warn"],
        "fail": suite["fail"],
        "all_ok": suite["fail"] == 0,
    }
    suite["summary"] = summary
    json_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    lines = [f"# {suite['stage']}", "", f"Pass: {suite['pass']}", f"Warn: {suite['warn']}", f"Fail: {suite['fail']}", ""]
    for r in suite["results"]:
        lines.append(f"- **{r['status'].upper()}** `{r['name']}` — {r['message']}")
        if r.get('expected') or r.get('observed'):
            lines.append(f"  - expected/observed: `{r.get('expected')}` / `{r.get('observed')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    suite["json_path"] = str(json_path)
    suite["md_path"] = str(md_path)
    return suite


def run_sign_semantics_stage_contracts(cfg: Any, report: Dict[str, Any], *, contracts_dir: Optional[Path] = None) -> Dict[str, Any]:
    contracts_dir = Path(contracts_dir or (Path(cfg.out_dir) / "contracts"))
    suite: Dict[str, Any] = {"stage": "sign_semantics_stage", "pass": 0, "warn": 0, "fail": 0, "results": []}

    # River sounding semantics are high-risk and should be explicit where possible.
    mode = getattr(cfg, "river_soundings_mode", None)
    if mode is not None:
        observed = semantics_from_soundings_mode(str(mode))
        if observed == "auto":
            suite["warn"] += 1
            suite["results"].append(_stage_result(
                "river_soundings_mode_explicit",
                "warning",
                "River soundings remain in auto semantic inference mode; explicit bed_elev/depth_pos/depth_neg is safer for inland positive NAVD88 cases.",
                expected="explicit_semantics",
                observed=observed,
            ))
        else:
            suite["pass"] += 1
            suite["results"].append(_stage_result(
                "river_soundings_mode_explicit",
                "pass",
                "River soundings mode declares explicit runtime semantics.",
                expected="explicit_semantics",
                observed=observed,
            ))

    # Harvest alignment semantic receipts if the alignment stage returned them.
    alignment_contract = None
    for key in ("alignment", "align", "alignment_summary"):
        cand = report.get(key)
        if isinstance(cand, dict) and isinstance(cand.get("semantic_contract"), dict):
            alignment_contract = cand.get("semantic_contract")
            break
    if alignment_contract is None:
        sdb = report.get("sdb") if isinstance(report.get("sdb"), dict) else None
        if sdb and isinstance(sdb.get("alignment"), dict) and isinstance(sdb["alignment"].get("semantic_contract"), dict):
            alignment_contract = sdb["alignment"]["semantic_contract"]
    if alignment_contract is not None:
        status = str(alignment_contract.get("status", "warning"))
        norm = "pass" if status == "pass" else ("warning" if status in {"warning", "warn", "skip"} else "error")
        suite["pass" if norm == "pass" else "warn" if norm == "warning" else "fail"] += 1
        suite["results"].append(_stage_result(
            "alignment_input_semantics",
            norm,
            str(alignment_contract.get("message", "Alignment semantic contract recorded.")),
            expected=alignment_contract.get("expected"),
            observed=alignment_contract.get("observed"),
        ))

    suite = _write_stage_suite(suite, contracts_dir, 'contracts_sign_semantics_stage')
    report.setdefault("contracts", {})["sign_semantics_stage"] = suite.get("summary", {})
    report.setdefault("outputs", {})["contracts_sign_semantics_stage_json"] = suite["json_path"]
    report.setdefault("outputs", {})["contracts_sign_semantics_stage_md"] = suite["md_path"]
    return suite
