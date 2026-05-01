#!/usr/bin/env python3
"""Compare two AOI river workflow exports against the canonical-parent contract.

This is intentionally a post-run validator. It does not rebuild, rediscover, or
mutate workflow products. It only reads retained receipts plus final DEM rasters
and verifies that two AOI runs are clean exports from the same canonical parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import rasterio
from rasterio.windows import Window


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _nested_get(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _path_from_record(value: Any) -> Path | None:
    if isinstance(value, Mapping):
        value = value.get("path")
    if value in (None, ""):
        return None
    try:
        return Path(str(value))
    except (TypeError, ValueError):
        return None


def _existing_path(*candidates: Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _first_path(*candidates: Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _resolve_run(out_dir: Path) -> dict[str, Any]:
    reports = out_dir / "reports"
    final = out_dir / "final"
    identity = _read_json(reports / "aoi_export_identity.json")
    manifest = _read_json(reports / "canonical_river_solution_manifest.json")
    run_summary_json = reports / "run_summary.json"
    run_summary_text = reports / "run_summary.txt"
    run_summary = _read_json(run_summary_json)
    workflow_receipt = _read_json(final / "river_workflow_receipt.json")
    final_output_receipt = _read_json(final / "final_output_receipt.json")

    # Prefer retained review/final artifacts over historical internal paths recorded
    # in older receipts. Normal cleanup may intentionally remove river_workflow/export
    # and river_workflow/final while preserving hashes and retained final-folder review
    # products. This validator should check the retained contract, not stale paths.
    parent_record_path = _path_from_record(identity.get("parent_dem"))
    parent_receipt_path = _path_from_record(_nested_get(workflow_receipt, ("identity", "canonical_parent_dem")))
    parent_path = _existing_path(
        out_dir / "final" / "canonical_parent_dem.tif",
        parent_record_path,
        parent_receipt_path,
    ) or _first_path(out_dir / "final" / "canonical_parent_dem.tif", parent_record_path, parent_receipt_path)

    export_record_path = _path_from_record(identity.get("aoi_export_dem"))
    export_receipt_path = _path_from_record(_nested_get(workflow_receipt, ("identity", "aoi_export_dem")))
    export_path = _existing_path(
        export_record_path,
        export_receipt_path,
        out_dir / "river_workflow" / "export" / "DEM_enhanced_export.tif",
    ) or _first_path(export_record_path, export_receipt_path, out_dir / "river_workflow" / "export" / "DEM_enhanced_export.tif")

    final_path = _existing_path(
        _path_from_record(identity.get("combined_dem")),
        _path_from_record(_nested_get(workflow_receipt, ("identity", "final_dem"))),
        out_dir / "combined" / "DEM_enhanced.tif",
    ) or _first_path(
        _path_from_record(identity.get("combined_dem")),
        _path_from_record(_nested_get(workflow_receipt, ("identity", "final_dem"))),
        out_dir / "combined" / "DEM_enhanced.tif",
    )

    parent_hash = _first(
        identity.get("parent_hash"),
        _nested_get(workflow_receipt, ("identity", "canonical_parent_dem", "sha256")),
        _sha256_file(parent_path),
    )
    export_hash = _first(
        identity.get("export_hash"),
        _nested_get(workflow_receipt, ("identity", "aoi_export_dem", "sha256")),
        _sha256_file(export_path),
    )
    final_hash = _first(
        identity.get("combined_hash"),
        _nested_get(workflow_receipt, ("identity", "final_dem", "sha256")),
        _sha256_file(final_path),
    )

    system_id = _first(
        identity.get("canonical_system_id"),
        manifest.get("canonical_system_id"),
        _nested_get(workflow_receipt, ("workflow", "canonical_system_id")),
        run_summary.get("canonical_system_id"),
    )
    cache_key = _first(
        identity.get("canonical_cache_key"),
        manifest.get("canonical_cache_key"),
        manifest.get("canonical_solve_cache_key"),
        _nested_get(workflow_receipt, ("workflow", "canonical_cache_key")),
        _nested_get(run_summary, ("cache", "canonical_cache_key")),
    )

    return {
        "out_dir": str(out_dir),
        "paths": {
            "canonical_parent_dem": str(parent_path) if parent_path is not None else None,
            "aoi_export_dem": str(export_path) if export_path is not None else None,
            "final_dem": str(final_path) if final_path is not None else None,
            "aoi_export_identity": str(reports / "aoi_export_identity.json"),
            "canonical_manifest": str(reports / "canonical_river_solution_manifest.json"),
            "run_summary": str(run_summary_json if run_summary_json.is_file() else run_summary_text),
            "run_summary_json": str(run_summary_json),
            "run_summary_text": str(run_summary_text),
            "river_workflow_receipt": str(final / "river_workflow_receipt.json"),
            "final_output_receipt": str(final / "final_output_receipt.json"),
        },
        "identity": identity,
        "manifest": manifest,
        "run_summary": run_summary,
        "workflow_receipt": workflow_receipt,
        "final_output_receipt": final_output_receipt,
        "canonical_system_id": system_id,
        "canonical_cache_key": cache_key,
        "hashes": {
            "parent_hash": parent_hash,
            "export_hash": export_hash,
            "final_hash": final_hash,
        },
        "route_checks": {
            "export_vs_parent": identity.get("export_vs_parent"),
            "combined_vs_export": identity.get("combined_vs_export"),
            "identity_passed": identity.get("passed"),
            "max_abs_diff": identity.get("max_abs_diff"),
            "mismatch_pixels": identity.get("mismatch_pixels"),
            "single_writer_pass": _nested_get(workflow_receipt, ("identity", "single_writer_pass")),
            "final_source": _nested_get(workflow_receipt, ("workflow", "final_dem_source")),
        },
    }


def _path_exists(path_value: str | None) -> bool:
    return path_value not in (None, "") and Path(str(path_value)).is_file()


def _bool_pass(value: Any) -> bool:
    return value is True or str(value).upper() == "PASS"


def _compare_value(name: str, a: Any, b: Any, *, required: bool = True) -> dict[str, Any]:
    if a in (None, "") or b in (None, ""):
        return {
            "name": name,
            "passed": not required,
            "left": a,
            "right": b,
            "failure_reason": "missing_value" if required else None,
        }
    return {"name": name, "passed": a == b, "left": a, "right": b, "failure_reason": None if a == b else "mismatch"}


def _same_grid(a, b, tol: float = 1.0e-9) -> bool:
    return (
        str(a.crs) == str(b.crs)
        and abs(a.transform.a - b.transform.a) <= tol
        and abs(a.transform.e - b.transform.e) <= tol
        and abs(a.transform.b) <= tol
        and abs(a.transform.d) <= tol
        and abs(b.transform.b) <= tol
        and abs(b.transform.d) <= tol
    )


def _finite_mask(arr: np.ndarray, nodata: Any) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata
    return mask


def _array_mismatch(a: np.ndarray, b: np.ndarray, nodata_a: Any, nodata_b: Any, tolerance: float) -> tuple[int, float]:
    ma = _finite_mask(a, nodata_a)
    mb = _finite_mask(b, nodata_b)
    both = ma & mb
    nodata_mismatch = ma ^ mb
    max_abs = 0.0
    value_mismatch = np.zeros(a.shape, dtype=bool)
    if np.any(both):
        diff = np.abs(a[both].astype("float64") - b[both].astype("float64"))
        if diff.size:
            max_abs = float(np.nanmax(diff))
        value_mismatch[both] = diff > tolerance
    mismatch = nodata_mismatch | value_mismatch
    return int(np.count_nonzero(mismatch)), float(max_abs)


def _compare_export_overlap(north_final: Path, south_final: Path, tolerance: float) -> dict[str, Any]:
    with rasterio.open(north_final) as north, rasterio.open(south_final) as south:
        result: dict[str, Any] = {
            "north_dem": str(north_final),
            "south_dem": str(south_final),
            "north_shape": [int(north.height), int(north.width)],
            "south_shape": [int(south.height), int(south.width)],
            "north_bounds": [float(v) for v in north.bounds],
            "south_bounds": [float(v) for v in south.bounds],
            "same_grid_geometry": _same_grid(north, south),
            "tolerance": float(tolerance),
        }
        if not result["same_grid_geometry"]:
            result.update({"passed": False, "relation": "grid_mismatch", "failure_reason": "crs_resolution_or_rotation_mismatch"})
            return result

        nt = north.transform
        st = south.transform
        # north/south row/column offsets expressed on the north grid.
        col_south_in_north = int(round((st.c - nt.c) / nt.a)) if nt.a else 0
        row_south_in_north = int(round((st.f - nt.f) / nt.e)) if nt.e else 0
        col_delta = (st.c - nt.c) / nt.a if nt.a else float("nan")
        row_delta = (st.f - nt.f) / nt.e if nt.e else float("nan")
        grid_aligned = abs(col_delta - col_south_in_north) <= 1.0e-9 and abs(row_delta - row_south_in_north) <= 1.0e-9
        result.update({
            "grid_aligned": bool(grid_aligned),
            "south_window_on_north_grid": {
                "row_off": int(row_south_in_north),
                "col_off": int(col_south_in_north),
                "height": int(south.height),
                "width": int(south.width),
            },
        })
        if not grid_aligned:
            result.update({"passed": False, "relation": "grid_not_integer_aligned", "failure_reason": "not_parent_grid_aligned"})
            return result

        n_row0 = max(0, row_south_in_north)
        n_col0 = max(0, col_south_in_north)
        s_row0 = max(0, -row_south_in_north)
        s_col0 = max(0, -col_south_in_north)
        height = min(north.height - n_row0, south.height - s_row0)
        width = min(north.width - n_col0, south.width - s_col0)
        if height > 0 and width > 0:
            n_arr = north.read(1, window=Window(n_col0, n_row0, width, height), masked=False)
            s_arr = south.read(1, window=Window(s_col0, s_row0, width, height), masked=False)
            mismatch, max_abs = _array_mismatch(n_arr, s_arr, north.nodata, south.nodata, tolerance)
            result.update({
                "relation": "overlap",
                "overlap_window": {
                    "north": {"row_off": int(n_row0), "col_off": int(n_col0), "height": int(height), "width": int(width)},
                    "south": {"row_off": int(s_row0), "col_off": int(s_col0), "height": int(height), "width": int(width)},
                },
                "overlap_pixels": int(height * width),
                "mismatch_pixels": int(mismatch),
                "max_abs_diff": float(max_abs),
                "passed": bool(mismatch == 0 and max_abs <= tolerance),
                "failure_reason": None if mismatch == 0 and max_abs <= tolerance else "overlap_pixel_mismatch",
            })
            return result

        # Adjacent AOIs have no shared pixel to compare. Strictness here means
        # proving same grid, no gap/overlap at the shared edge, and shared x span.
        north_bottom = float(north.bounds.bottom)
        north_top = float(north.bounds.top)
        south_bottom = float(south.bounds.bottom)
        south_top = float(south.bounds.top)
        x_overlap = min(float(north.bounds.right), float(south.bounds.right)) - max(float(north.bounds.left), float(south.bounds.left))
        y_gap = max(south_top - north_bottom, north_bottom - south_top) if south_top <= north_top else max(north_top - south_bottom, south_bottom - north_top)
        touches_horizontal = abs(north_bottom - south_top) <= max(abs(float(north.res[1])) * 1.0e-6, 1.0e-9)
        touches_reverse = abs(south_bottom - north_top) <= max(abs(float(north.res[1])) * 1.0e-6, 1.0e-9)
        relation = "touching_horizontal" if touches_horizontal or touches_reverse else "separate_no_overlap"
        result.update({
            "relation": relation,
            "x_overlap_units": float(x_overlap),
            "edge_gap_units": float(abs(north_bottom - south_top) if abs(north_bottom - south_top) <= abs(south_bottom - north_top) else abs(south_bottom - north_top)),
            "overlap_pixels": 0,
            "mismatch_pixels": 0,
            "max_abs_diff": 0.0,
            "passed": bool((touches_horizontal or touches_reverse) and x_overlap > 0.0),
            "failure_reason": None if (touches_horizontal or touches_reverse) and x_overlap > 0.0 else "aois_do_not_touch_or_overlap_on_same_grid",
        })
        # Diagnostic only: adjacent rows should not be forced equal in a sloping river.
        if touches_horizontal and x_overlap > 0.0:
            width_cols = min(north.width, south.width)
            n_edge = north.read(1, window=Window(0, north.height - 1, width_cols, 1), masked=False)
            s_edge = south.read(1, window=Window(0, 0, width_cols, 1), masked=False)
            edge_mask = _finite_mask(n_edge, north.nodata) & _finite_mask(s_edge, south.nodata)
            if np.any(edge_mask):
                edge_diff = np.abs(n_edge[edge_mask].astype("float64") - s_edge[edge_mask].astype("float64"))
                result["adjacent_edge_diagnostic"] = {
                    "finite_pairs": int(np.count_nonzero(edge_mask)),
                    "mean_abs_diff": float(np.nanmean(edge_diff)),
                    "max_abs_diff": float(np.nanmax(edge_diff)),
                    "note": "diagnostic_only_adjacent_rows_are_not_same_pixels",
                }
        return result


def _build_checks(north: dict[str, Any], south: dict[str, Any], seam: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for label, run in (("north", north), ("south", south)):
        paths = run.get("paths", {})
        for role in ("final_dem", "aoi_export_identity", "canonical_manifest", "river_workflow_receipt", "final_output_receipt"):
            checks.append({
                "name": f"{label}_{role}_exists",
                "passed": _path_exists(paths.get(role)),
                "path": paths.get(role),
            })
        checks.append({
            "name": f"{label}_canonical_parent_retained_or_hashed",
            "passed": _path_exists(paths.get("canonical_parent_dem")) or run.get("hashes", {}).get("parent_hash") not in (None, ""),
            "path": paths.get("canonical_parent_dem"),
            "hash": run.get("hashes", {}).get("parent_hash"),
        })
        checks.append({
            "name": f"{label}_aoi_export_retained_or_hashed",
            "passed": _path_exists(paths.get("aoi_export_dem")) or run.get("hashes", {}).get("export_hash") not in (None, ""),
            "path": paths.get("aoi_export_dem"),
            "hash": run.get("hashes", {}).get("export_hash"),
        })
        checks.append({
            "name": f"{label}_run_summary_retained",
            "passed": _path_exists(paths.get("run_summary_json")) or _path_exists(paths.get("run_summary_text")),
            "json_path": paths.get("run_summary_json"),
            "text_path": paths.get("run_summary_text"),
        })
        route = run.get("route_checks", {})
        checks.append({"name": f"{label}_export_vs_parent_pass", "passed": _bool_pass(route.get("export_vs_parent")) or route.get("identity_passed") is True, "value": route.get("export_vs_parent"), "identity_passed": route.get("identity_passed")})
        checks.append({"name": f"{label}_combined_vs_export_pass", "passed": _bool_pass(route.get("combined_vs_export")) or run.get("hashes", {}).get("export_hash") == run.get("hashes", {}).get("final_hash"), "value": route.get("combined_vs_export")})
        checks.append({"name": f"{label}_single_writer_pass", "passed": route.get("single_writer_pass") is True, "value": route.get("single_writer_pass")})
        checks.append({
            "name": f"{label}_final_source_is_aoi_export",
            "passed": route.get("final_source") == "aoi_export_dem",
            "value": route.get("final_source"),
            "failure_reason": None if route.get("final_source") == "aoi_export_dem" else "missing_or_wrong_final_dem_source",
        })
        checks.append({"name": f"{label}_parent_hash_present", "passed": run.get("hashes", {}).get("parent_hash") not in (None, ""), "value": run.get("hashes", {}).get("parent_hash")})
        checks.append({"name": f"{label}_export_hash_present", "passed": run.get("hashes", {}).get("export_hash") not in (None, ""), "value": run.get("hashes", {}).get("export_hash")})
        checks.append({"name": f"{label}_final_hash_present", "passed": run.get("hashes", {}).get("final_hash") not in (None, ""), "value": run.get("hashes", {}).get("final_hash")})

    checks.extend([
        _compare_value("same_canonical_system_id", north.get("canonical_system_id"), south.get("canonical_system_id"), required=True),
        _compare_value("same_canonical_cache_key", north.get("canonical_cache_key"), south.get("canonical_cache_key"), required=True),
        _compare_value("same_canonical_parent_hash", north.get("hashes", {}).get("parent_hash"), south.get("hashes", {}).get("parent_hash"), required=True),
        {"name": "north_south_grid_relation", "passed": seam.get("passed") is True, "relation": seam.get("relation"), "failure_reason": seam.get("failure_reason")},
    ])
    return checks


def _run_science_lines(label: str, run: Mapping[str, Any]) -> list[str]:
    summary = run.get("run_summary") if isinstance(run.get("run_summary"), Mapping) else {}
    validation = summary.get("science_validation") if isinstance(summary.get("science_validation"), Mapping) else {}
    checks = validation.get("checks") if isinstance(validation.get("checks"), Mapping) else {}
    lines: list[str] = []
    if not checks:
        return lines
    lines.append(f"{label} read-only science/support checks:")
    for name in ("wse_monotone_downstream", "observed_offset_support", "modeled_offset_support", "backbone_downstream_rise", "support_class_counts_reported"):
        item = checks.get(name) if isinstance(checks.get(name), Mapping) else {}
        if not item:
            continue
        status = item.get("status")
        detail = {str(k): v for k, v in item.items() if k != "status" and v not in (None, "", {})}
        if name == "backbone_downstream_rise" and status == "pass":
            detail.setdefault("status_basis", "large_downstream_rise_count_gt_allowed == 0")
            detail.setdefault("note", "positive downstream step count is diagnostic; large downstream rises are the failure criterion")
        lines.append(f"- {name}: {status} {detail}")
    return lines



_WRONG_ARTIFACT_HINTS: tuple[tuple[str, str, str], ...] = (
    ("aoi_export_identity_exists", "reports/aoi_export_identity.json", "AOI export identity receipt was not retained after export."),
    ("canonical_manifest_exists", "reports/canonical_river_solution_manifest.json", "canonical parent manifest was not retained with the AOI run."),
    ("run_summary_retained", "reports/run_summary.json or reports/run_summary.txt", "run summary was not retained with the AOI run."),
    ("river_workflow_receipt_exists", "final/river_workflow_receipt.json", "final river workflow receipt was not retained."),
    ("final_output_receipt_exists", "final/final_output_receipt.json", "final output receipt was not retained."),
    ("canonical_parent_retained_or_hashed", "final/canonical_parent_dem.tif or canonical parent hash", "canonical parent review raster/hash is missing from retained outputs."),
    ("aoi_export_retained_or_hashed", "reports/aoi_export_identity.json export_hash", "AOI export raster/hash is missing from retained outputs."),
    ("final_dem_exists", "combined/DEM_enhanced.tif", "final DEM is missing or not referenced by receipts."),
    ("export_vs_parent_pass", "reports/aoi_export_identity.json", "AOI export does not match the canonical parent window."),
    ("combined_vs_export_pass", "combined/DEM_enhanced.tif", "final DEM does not match the named AOI export."),
    ("single_writer_pass", "final/river_workflow_receipt.json", "single-writer final DEM contract failed."),
    ("final_source_is_aoi_export", "final/river_workflow_receipt.json", "final DEM source is not recorded as the AOI export."),
    ("parent_hash_present", "canonical parent DEM hash", "canonical parent hash is missing from receipts."),
    ("export_hash_present", "AOI export DEM hash", "AOI export hash is missing from receipts."),
    ("final_hash_present", "final DEM hash", "final DEM hash is missing from receipts."),
    ("same_canonical_system_id", "reports/canonical_river_solution_manifest.json", "north/south runs resolved different canonical river systems."),
    ("same_canonical_cache_key", "reports/canonical_river_solution_manifest.json", "north/south runs resolved different canonical cache keys."),
    ("same_canonical_parent_hash", "canonical_solve_final_dem.tif", "north/south runs exported from different canonical parent DEMs."),
    ("north_south_grid_relation", "combined/DEM_enhanced.tif", "north/south final DEMs are not on the expected shared grid relationship."),
)


def _first_failure_summary(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a compact first-failure diagnosis without trying to repair anything."""
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    for check in checks:
        if check.get("passed"):
            continue
        name = str(check.get("name") or "unknown_check")
        artifact = None
        reason = check.get("failure_reason") or check.get("value") or check.get("path") or "failed"
        for suffix, hinted_artifact, hinted_reason in _WRONG_ARTIFACT_HINTS:
            if name.endswith(suffix) or name == suffix:
                artifact = hinted_artifact
                reason = hinted_reason if reason in (None, "", "failed") else f"{reason}; {hinted_reason}"
                break
        if artifact is None:
            artifact = str(check.get("path") or "comparison contract")
        return {
            "check": name,
            "first_wrong_artifact": artifact,
            "reason": str(reason),
            "note": "verify-only comparison; no output repair or rerouting was attempted",
        }
    return None

def _render_text(payload: Mapping[str, Any]) -> str:
    lines = [
        "AOI export comparison",
        "=====================",
        f"status: {'PASS' if payload.get('passed') else 'FAIL'}",
        f"relation: {payload.get('seam', {}).get('relation')}",
        f"canonical_system_id: {payload.get('north', {}).get('canonical_system_id')}",
        f"canonical_cache_key: {payload.get('north', {}).get('canonical_cache_key')}",
        f"parent_hash: {payload.get('north', {}).get('hashes', {}).get('parent_hash')}",
        "",
        "Checks:",
    ]
    for check in payload.get("checks", []):
        status = "PASS" if check.get("passed") else "FAIL"
        detail = check.get("failure_reason") or check.get("relation") or check.get("value") or check.get("path") or ""
        lines.append(f"- {status}: {check.get('name')} {detail}")
    first_failure = payload.get("first_failure") if isinstance(payload.get("first_failure"), Mapping) else None
    if first_failure:
        lines.extend([
            "",
            "First failing check:",
            f"- check: {first_failure.get('check')}",
            f"- first_wrong_artifact: {first_failure.get('first_wrong_artifact')}",
            f"- reason: {first_failure.get('reason')}",
            f"- note: {first_failure.get('note')}",
        ])
    seam = payload.get("seam", {}) if isinstance(payload.get("seam"), dict) else {}
    lines.extend([
        "",
        "Seam/grid relation:",
        f"- relation: {seam.get('relation')}",
        f"- overlap_pixels: {seam.get('overlap_pixels')}",
        f"- mismatch_pixels: {seam.get('mismatch_pixels')}",
        f"- max_abs_diff: {seam.get('max_abs_diff')}",
    ])
    science_lines: list[str] = []
    north = payload.get("north") if isinstance(payload.get("north"), Mapping) else {}
    south = payload.get("south") if isinstance(payload.get("south"), Mapping) else {}
    science_lines.extend(_run_science_lines("north", north))
    science_lines.extend(_run_science_lines("south", south))
    if science_lines:
        lines.extend(["", "Read-only science/support diagnostics:"])
        lines.extend(science_lines)
    diag = seam.get("adjacent_edge_diagnostic") if isinstance(seam.get("adjacent_edge_diagnostic"), dict) else None
    if diag:
        lines.extend([
            "- adjacent_edge_diagnostic: diagnostic only, adjacent rows are not the same pixels",
            f"  finite_pairs={diag.get('finite_pairs')} mean_abs_diff={diag.get('mean_abs_diff')} max_abs_diff={diag.get('max_abs_diff')}",
        ])
    failures = [c for c in payload.get("checks", []) if not c.get("passed")]
    if failures:
        lines.extend(["", "Failures:"])
        for failure in failures:
            lines.append(f"- {failure.get('name')}: {failure.get('failure_reason') or failure.get('value') or failure.get('path') or 'failed'}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("north_out_dir", type=Path)
    parser.add_argument("south_out_dir", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--text-out", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args(argv)

    north = _resolve_run(args.north_out_dir)
    south = _resolve_run(args.south_out_dir)
    north_final = Path(str(north["paths"].get("final_dem"))) if north["paths"].get("final_dem") else None
    south_final = Path(str(south["paths"].get("final_dem"))) if south["paths"].get("final_dem") else None
    if north_final is not None and south_final is not None and north_final.is_file() and south_final.is_file():
        seam = _compare_export_overlap(north_final, south_final, args.tolerance)
    else:
        seam = {"passed": False, "relation": "missing_final_dem", "failure_reason": "missing_final_dem", "overlap_pixels": 0, "mismatch_pixels": None, "max_abs_diff": None}
    checks = _build_checks(north, south, seam)
    passed = all(bool(c.get("passed")) for c in checks)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "check": "north_south_canonical_aoi_export_identity",
        "passed": bool(passed),
        "north": north,
        "south": south,
        "seam": seam,
        "checks": checks,
    }
    payload["first_failure"] = _first_failure_summary(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = _render_text(payload)
    if args.text_out is not None:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
