"""Explicit IO manifest and artifact emission helpers."""
from __future__ import annotations

import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.json_io import write_json


def is_probably_path(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    if any(ch.isspace() for ch in s):
        return False
    if "://" in s:
        return False
    if "/" in s or "\\" in s:
        return True
    return bool(re.search(r"\.[A-Za-z0-9]{2,6}$", s))


def collect_paths_from_obj(obj: Any, out: List[str]) -> None:
    try:
        if isinstance(obj, dict):
            for _, v in obj.items():
                collect_paths_from_obj(v, out)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                collect_paths_from_obj(v, out)
        elif isinstance(obj, str) and is_probably_path(obj):
            out.append(obj)
    except (RecursionError, TypeError, ValueError):
        return


def parse_paths_from_command(cmd: str) -> Dict[str, List[str]]:
    ins: List[str] = []
    outs: List[str] = []
    if not isinstance(cmd, str) or not cmd.strip():
        return {"inputs": ins, "outputs": outs}
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    out_flags = {
        "-O", "--out", "--out-dir", "--out_gpkg", "--out-gpkg", "--out_tif", "--out-tif",
        "--out-xyz", "--out-csv", "--out-json", "--out-md", "--out-mask", "--out-raster",
        "--output", "--output-dir", "--output-path", "--dem-out", "--mask-out",
    }
    in_flags = {
        "-i", "--in", "--input", "--input-path", "--dem", "--template", "--template-raster",
        "--mask", "--mask-raster", "--ocean-mask", "--water-mask", "--river-mask", "--xs",
        "--xs-gpkg", "--network", "--network-gpkg", "--soundings", "--xyz", "--extra-xyz",
    }
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in out_flags and i + 1 < len(toks):
            p = toks[i + 1]
            if is_probably_path(p):
                outs.append(p)
            i += 2
            continue
        if t in in_flags and i + 1 < len(toks):
            p = toks[i + 1]
            if is_probably_path(p):
                ins.append(p)
            i += 2
            continue
        if t.startswith("--") and "=" in t:
            flag, val = t.split("=", 1)
            if is_probably_path(val):
                if flag in out_flags:
                    outs.append(val)
                elif flag in in_flags:
                    ins.append(val)
                else:
                    ins.append(val)
        i += 1
    return {"inputs": ins, "outputs": outs}


def build_io_manifest(report: Dict[str, Any]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": report.get("run_id"),
        "out_dir": report.get("out_dir"),
        "inputs": [],
        "outputs": [],
    }
    collected: List[str] = []
    collect_paths_from_obj(report, collected)
    steps_obj = (report.get("river", {}) or {}).get("steps") if isinstance(report, dict) else None
    step_list = list(steps_obj.values()) if isinstance(steps_obj, dict) else (steps_obj if isinstance(steps_obj, list) else [])
    for step in step_list:
        cmd = step.get("command") if isinstance(step, dict) else None
        parsed = parse_paths_from_command(cmd or "")
        collected.extend(parsed.get("inputs", []))
        collected.extend(parsed.get("outputs", []))
    seen = set()
    uniq = []
    for p in collected:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    out_paths: set[str] = set()

    def _collect_outputs(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "outputs" and isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, str) and is_probably_path(vv):
                            out_paths.add(vv)
                _collect_outputs(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _collect_outputs(v)

    _collect_outputs(report)
    for step in step_list:
        cmd = step.get("command") if isinstance(step, dict) else ""
        parsed = parse_paths_from_command(cmd or "")
        for p in parsed.get("outputs", []):
            out_paths.add(p)
    manifest["inputs"] = [p for p in uniq if p not in out_paths]
    manifest["outputs"] = [p for p in uniq if p in out_paths]
    return manifest


def write_io_manifest(out_dir: Path, report: Dict[str, Any]) -> Tuple[Path, Path]:
    io = build_io_manifest(report)
    p_json = out_dir / "io_manifest.json"
    p_md = out_dir / "io_manifest.md"
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(io, f, indent=2, sort_keys=True)
    lines = ["# IO manifest", "", f"- run_id: `{io.get('run_id')}`", f"- out_dir: `{io.get('out_dir')}`", "", "## Inputs (explicit paths observed)"]
    lines.extend([f"- `{p}`" for p in io.get("inputs", [])])
    lines.append("")
    lines.append("## Outputs (explicit paths observed)")
    lines.extend([f"- `{p}`" for p in io.get("outputs", [])])
    lines.append("")
    p_md.write_text("\n".join(lines), encoding="utf-8")
    return p_json, p_md


def write_guidance_manifest(cfg: Any, report: Dict[str, Any], *, provenance_enum: Any, pipeline_version: Any) -> Path:
    river_outputs = report.get("river", {}).get("outputs", {})
    estuary_clip = report.get("river", {}).get("estuary_clip", {})
    fusion_controls = report.get("fusion", {}).get("guidance_controls", {})
    sdb_report = report.get("sdb", {})

    def _artifact(path_str: str | None, *, role: str, domain: str, authoritative: bool = False, description: str = "") -> Dict[str, Any]:
        entry = {
            "path": str(path_str) if path_str else None,
            "exists": bool(path_str and Path(path_str).exists()) if path_str else False,
            "role": role,
            "domain_class": domain,
            "authoritative": authoritative,
        }
        if description:
            entry["description"] = description
        return entry

    artifacts = []
    for key, role, desc in [
        ("guidance_weight", "guidance_weight", "0 at authoritative support and outside trusted export region; ramps upward within trusted export interior away from anchors"),
        ("trusted_interior", "trusted_interior", "Trusted export region from halo-domain river solve, inset from edges and excluding estuary transition"),
        ("soft_guidance_domain", "soft_guidance_domain", "Broader river guidance domain before removing authoritative anchors"),
        ("admissibility", "admissibility", "Anchor-excluded river guidance subset admissible for soft influence"),
        ("guide_points", "guide_points", "Spatially thinned pseudo-soundings with confidence weights"),
        ("authoritative_support", "authoritative_support", "Rasterized authoritative anchor support mask"),
        ("authoritative_support_depth", "authoritative_support_depth", "Rasterized authoritative support depth for exact overwrite"),
        ("corridor_mask", "corridor_mask", "River corridor mask delimiting fluvial guidance region"),
        ("scaffold_domains", "scaffold_domains", "Scaffold/solve/export AOI metadata for canonical river guidance"),
    ]:
        p = river_outputs.get(key)
        if p:
            artifacts.append(_artifact(p, role=role, domain="fluvial_core", authoritative=(key == "authoritative_support_depth"), description=desc))
    et_path = river_outputs.get("estuary_transition")
    if et_path:
        artifacts.append(_artifact(et_path, role="estuary_transition_mask", domain="estuary_transition", description="1 where river-to-ocean handoff zone; reduced-trust guidance"))
    ec_path = estuary_clip.get("estuary_clip_mask") or river_outputs.get("estuary_clip_mask")
    if ec_path:
        artifacts.append(_artifact(ec_path, role="estuary_clip_mask", domain="estuary_transition", description="1 where river channel was clipped (SDB may fill); river physics invalid here"))
    sdb_artifacts = sdb_report.get("artifacts", {})
    for key, role, desc in [
        ("depth_raster", "sdb_depth", "SDB dense depth raster (diagnostic_only; non-authoritative)"),
        ("guide_points", "sdb_guide_points", "Spatially thinned pseudo-soundings for guidance-first interpolation"),
        ("confidence_raster", "sdb_confidence", "Per-pixel SDB confidence"),
        ("guidance_weight_raster", "sdb_guidance_weight", "SDB guidance weight for interpolation"),
        ("trusted_interior_raster", "sdb_trusted_interior", "Trusted export region for SDB guidance"),
        ("uncertainty_raster", "sdb_uncertainty", "SDB 1-sigma uncertainty"),
        ("admissibility_raster", "sdb_admissibility", "SDB optical soft-guidance domain admissibility"),
        ("lower_bound_raster", "sdb_lower_bound", "Uncertainty-derived plausible lower bound for SDB guidance"),
        ("upper_bound_raster", "sdb_upper_bound", "Uncertainty-derived plausible upper bound for SDB guidance"),
        ("provenance_raster", "sdb_provenance", "SDB provenance (0=nodata, 1=predicted)"),
    ]:
        p = sdb_artifacts.get(key)
        if p:
            artifacts.append(_artifact(p, role=role, domain="coastal_sdb", description=desc))
    combined = report.get("outputs", {})
    for key, role, desc in [
        ("depth", "combined_depth", "Fused depth raster (river + SDB + authoritative)"),
        ("provenance", "combined_provenance", "Provenance code per pixel (see constants.Provenance)"),
    ]:
        p = combined.get(key)
        if p:
            artifacts.append(_artifact(p, role=role, domain="combined", description=desc))
    manifest = {
        "schema_version": 1,
        "pipeline_version": str(pipeline_version),
        "aoi": str(cfg.aoi),
        "run_id": str(cfg.run_id or ""),
        "domain_classes": {
            "fluvial_core": "River channel where channelized-flow assumptions are valid; river guidance is primary",
            "estuary_transition": "Handoff zone where river physics break down; reduced-trust, SDB may fill",
            "coastal_sdb": "Open water / coastal domain; SDB guidance is primary",
            "combined": "Fused output incorporating all sources with domain-aware arbitration",
        },
        "fusion_policy": {
            "authoritative_exact_overwrite": True,
            "fluvial_core_river_primary": True,
            "estuary_transition_river_excluded": True,
            "estuary_sdb_fill_enabled": True,
            "estuary_max_weight": float(fusion_controls.get("estuary_max_weight", 0.25)),
            "estuary_detection_method": estuary_clip.get("method", "width_ratio + low_slope"),
            "estuary_detection_signals": estuary_clip.get("signals", {}),
        },
        "provenance_codes": {
            str(code): desc for code, desc in [
                (provenance_enum.NODATA, "NoData"),
                (provenance_enum.SDB, "SDB guidance"),
                (provenance_enum.RIVER, "River guidance"),
                (provenance_enum.MEASURED, "Authoritative measurement"),
                (provenance_enum.BLENDED, "Blended transition"),
                (provenance_enum.ESTUARY_TRANSITION, "Estuary transition (reduced-trust)"),
            ]
        },
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }
    out_path = Path(cfg.out_dir) / "guidance_manifest.json"
    write_json(out_path, manifest)
    return out_path


def emit_artifacts_from_report(report: Dict[str, Any]) -> None:
    try:
        from flight_recorder import emit_artifact_written
    except ImportError:
        return

    def _emit_outputs(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "outputs" and isinstance(v, dict):
                    for name, path in v.items():
                        if isinstance(path, str) and is_probably_path(path):
                            emit_artifact_written(Path(path), kind="file", role=str(name))
                _emit_outputs(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _emit_outputs(v)

    _emit_outputs(report)
