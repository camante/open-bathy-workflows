from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


from core.json_io import write_json
from io_artifacts import is_probably_path


_STAGE_SECTION_KEYS: Tuple[Tuple[str, str], ...] = (
    ("run", "run"),
    ("authoritative_base", "authoritative_base"),
    ("domains", "domains"),
    ("river", "river"),
    ("sdb", "sdb"),
    ("fusion", "fusion"),
    ("gapfill", "gapfill"),
    ("benchmark", "benchmark"),
    ("final_dem_runtime", "final_dem_runtime"),
    ("outputs", "final_outputs"),
)


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _rel_or_abs(path: Path, out_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(out_dir.resolve()))
    except Exception:
        return str(path)


def _path_entry(path_str: str | Path, out_dir: Path) -> Dict[str, Any]:
    p = Path(path_str)
    if not p.is_absolute():
        p = (out_dir / p).resolve()
    exists = p.exists()
    entry: Dict[str, Any] = {
        "path": _rel_or_abs(p, out_dir),
        "absolute_path": str(p),
        "exists": exists,
        "is_dir": bool(exists and p.is_dir()),
        "is_file": bool(exists and p.is_file()),
        "is_symlink": bool(p.is_symlink()),
    }
    if exists and p.is_file():
        st = p.stat()
        entry.update({
            "size_bytes": int(st.st_size),
            "sha256": _sha256(p),
        })
    return entry


def _collect_outputs_dict(outputs: dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for key, value in sorted(outputs.items()):
        if isinstance(value, (str, Path)) and is_probably_path(str(value)):
            ent = _path_entry(value, out_dir)
            ent["key"] = key
            ap = ent["absolute_path"]
            if ap in seen:
                continue
            seen.add(ap)
            items.append(ent)
    return items


def _collect_stage_entry(section_key: str, stage_name: str, report: dict[str, Any], out_dir: Path) -> Dict[str, Any] | None:
    obj = report.get(section_key)
    if not isinstance(obj, dict):
        return None
    outputs = _collect_outputs_dict(obj.get("outputs", {}) if isinstance(obj.get("outputs"), dict) else {}, out_dir)
    entry: Dict[str, Any] = {
        "stage": stage_name,
        "section_key": section_key,
        "status": obj.get("status"),
        "outputs": outputs,
    }
    if "execution_mode" in obj:
        entry["execution_mode"] = obj.get("execution_mode")
    if "reason" in obj:
        entry["reason"] = obj.get("reason")
    if stage_name == "run":
        entry["command"] = report.get("command") or obj.get("command")
        entry["run_id"] = report.get("run_id") or obj.get("run_id")
    return entry


def _load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    try:
        if path.exists() and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _collect_io_summary(out_dir: Path) -> Dict[str, Any]:
    io_manifest = _load_json_if_exists(out_dir / "io_manifest.json") or {}
    return {
        "run_id": io_manifest.get("run_id"),
        "input_count": len(io_manifest.get("inputs", []) or []),
        "output_count": len(io_manifest.get("outputs", []) or []),
        "inputs": list(io_manifest.get("inputs", []) or []),
        "outputs": list(io_manifest.get("outputs", []) or []),
    }


def _load_identity_receipt(path_value: str | Path | None, out_dir: Path) -> Dict[str, Any] | None:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.is_absolute():
        p = (out_dir / p).resolve()
    return _load_json_if_exists(p)


def build_traceability_manifest(out_dir: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    outputs_obj = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    auth_outputs = (report.get("authoritative_base", {}) or {}).get("outputs", {}) if isinstance((report.get("authoritative_base", {}) or {}).get("outputs", {}), dict) else {}
    river_outputs = (report.get("river", {}) or {}).get("outputs", {}) if isinstance((report.get("river", {}) or {}).get("outputs", {}), dict) else {}

    final_dem = outputs_obj.get("final_dem_user_stable") or outputs_obj.get("final_depth_user_stable") or outputs_obj.get("depth") or str(out_dir / "combined" / "DEM_enhanced.tif")
    conditioned_internal = auth_outputs.get("conditioned_final_dem_internal") or str(out_dir / "combined" / "conditioned_final_dem_internal.tif")
    debug_written = auth_outputs.get("stage_debug_06_dem_enhanced_written") or str(out_dir / "combined" / "debug_final_route" / "06_DEM_enhanced_written.tif")
    touch_log = outputs_obj.get("dem_enhanced_touch_log") or str(out_dir / "combined" / "dem_enhanced_touch_log.jsonl")
    identity_receipt = outputs_obj.get("dem_enhanced_identity_receipt") or str(out_dir / "combined" / "dem_enhanced_identity_receipt.json")
    river_debug = outputs_obj.get("river_workflow_debug_report") or str(out_dir / "RIVER_WORKFLOW_DEBUG.txt")
    debug_manifest = auth_outputs.get("stage_divergence_debug_manifest") or auth_outputs.get("debug_final_route_manifest") or str(out_dir / "combined" / "debug_final_route_manifest.json")

    stages: List[Dict[str, Any]] = []
    for section_key, stage_name in _STAGE_SECTION_KEYS:
        entry = _collect_stage_entry(section_key, stage_name, report, out_dir)
        if entry is not None:
            stages.append(entry)

    workflow_trace = outputs_obj.get("workflow_input_output_trace")
    required_paths = {
        "final_dem": final_dem,
        "final_dem_touch_log": touch_log,
        "final_dem_identity_receipt": identity_receipt,
        "debug_final_route_manifest": debug_manifest,
        "conditioned_final_dem_internal": conditioned_internal,
        "debug_final_route_written": debug_written,
        "river_workflow_debug_report": river_debug,
        "workflow_input_output_trace": workflow_trace,
        "io_manifest_json": str(out_dir / "io_manifest.json") if (out_dir / "io_manifest.json").exists() else None,
        "io_manifest_md": str(out_dir / "io_manifest.md") if (out_dir / "io_manifest.md").exists() else None,
    }

    required_entries: Dict[str, Any] = {}
    missing_required: List[str] = []
    for key, value in required_paths.items():
        if isinstance(value, (str, Path)) and str(value):
            ent = _path_entry(value, out_dir)
            required_entries[key] = ent
            if not ent.get("exists"):
                missing_required.append(key)
        else:
            required_entries[key] = {"path": value, "exists": False}
            missing_required.append(key)

    touch_actions: List[Dict[str, Any]] = []
    single_writer_contract: Dict[str, Any] = {
        "final_dem": _rel_or_abs(Path(final_dem).resolve(), out_dir) if isinstance(final_dem, (str, Path)) and Path(final_dem).exists() else final_dem,
        "conditioned_internal": _rel_or_abs(Path(conditioned_internal).resolve(), out_dir) if isinstance(conditioned_internal, (str, Path)) and Path(conditioned_internal).exists() else conditioned_internal,
        "debug_written": _rel_or_abs(Path(debug_written).resolve(), out_dir) if isinstance(debug_written, (str, Path)) and Path(debug_written).exists() else debug_written,
        "writer": "final_route_outputs_stage",
        "touch_log_present": False,
        "touch_count": 0,
        "final_route_write_count": 0,
        "unexpected_actions": [],
        "ok": False,
    }
    if isinstance(touch_log, (str, Path)):
        tl_path = Path(touch_log)
        if not tl_path.is_absolute():
            tl_path = (out_dir / tl_path).resolve()
        if tl_path.exists():
            single_writer_contract["touch_log_present"] = True
            for line in tl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                touch_actions.append(payload)
            single_writer_contract["touch_count"] = len(touch_actions)
            single_writer_contract["final_route_write_count"] = sum(1 for a in touch_actions if a.get("action") == "final_route_write")
            allowed = {"final_route_write", "identity_verification_start", "identity_verification_ok"}
            unexpected = [a for a in touch_actions if a.get("action") not in allowed]
            single_writer_contract["unexpected_actions"] = [a.get("action") for a in unexpected]
            expected_final_path_abs = str((out_dir / "combined" / "DEM_enhanced.tif").resolve())
            touch_path_mismatches = []
            for a in touch_actions:
                raw_path = a.get("path")
                if not raw_path:
                    touch_path_mismatches.append(raw_path)
                    continue
                ap = Path(str(raw_path))
                if not ap.is_absolute():
                    ap = (out_dir / ap).resolve()
                else:
                    ap = ap.resolve()
                if str(ap) != expected_final_path_abs:
                    touch_path_mismatches.append(str(raw_path))
            single_writer_contract["touch_paths_ok"] = not touch_path_mismatches
            single_writer_contract["touch_path_mismatches"] = touch_path_mismatches

    final_dem_hash = required_entries.get("final_dem", {}).get("sha256")
    conditioned_internal_hash = required_entries.get("conditioned_final_dem_internal", {}).get("sha256")
    debug_written_hash = required_entries.get("debug_final_route_written", {}).get("sha256")
    final_dem_entry = required_entries.get("final_dem", {})
    identity_payload = _load_identity_receipt(identity_receipt, out_dir)
    single_writer_contract["hash_match"] = bool(final_dem_hash and conditioned_internal_hash and final_dem_hash == conditioned_internal_hash)
    single_writer_contract["debug_hash_match"] = bool(final_dem_hash and debug_written_hash and final_dem_hash == debug_written_hash)
    single_writer_contract["final_dem_is_real_file"] = bool(final_dem_entry.get("exists") and final_dem_entry.get("is_file") and not final_dem_entry.get("is_symlink"))
    single_writer_contract["final_dem_expected_path"] = "combined/DEM_enhanced.tif"
    single_writer_contract["final_dem_path_ok"] = str(final_dem_entry.get("path")) == "combined/DEM_enhanced.tif"
    single_writer_contract["identity_receipt"] = identity_payload or {}
    single_writer_contract["identity_receipt_ok"] = bool(
        isinstance(identity_payload, dict)
        and identity_payload.get("match") is True
        and identity_payload.get("writer") == "final_route_outputs_stage"
        and identity_payload.get("verification_mode") == "raster_content_no_rewrite"
        and bool((identity_payload.get("identity") or {}).get("same_values"))
        and bool((identity_payload.get("identity") or {}).get("same_shape"))
        and bool((identity_payload.get("identity") or {}).get("same_transform"))
        and bool((identity_payload.get("identity") or {}).get("same_crs"))
        and str(identity_payload.get("final_path") or "").endswith(str(Path("combined") / "DEM_enhanced.tif"))
    )
    if "touch_paths_ok" not in single_writer_contract:
        single_writer_contract["touch_paths_ok"] = False
        single_writer_contract["touch_path_mismatches"] = ["touch_log_missing"] if not single_writer_contract.get("touch_log_present") else []
    single_writer_contract["ok"] = (
        single_writer_contract["touch_log_present"]
        and single_writer_contract["final_route_write_count"] == 1
        and not single_writer_contract["unexpected_actions"]
        and single_writer_contract["touch_paths_ok"]
        and single_writer_contract["identity_receipt_ok"]
        and single_writer_contract["final_dem_is_real_file"]
        and single_writer_contract["final_dem_path_ok"]
        and single_writer_contract["identity_receipt_ok"]
    )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": report.get("run_id") or report.get("run", {}).get("run_id"),
        "out_dir": str(out_dir),
        "goal": "Complete traceability from input/context to final DEM deliverables with a single-writer final DEM contract.",
        "final_dem_path": required_entries.get("final_dem", {}).get("path"),
        "stages": stages,
        "required_artifacts": required_entries,
        "single_writer_contract": single_writer_contract,
        "touch_log_actions": touch_actions,
        "missing_required_artifacts": missing_required,
        "io_observed": _collect_io_summary(out_dir),
    }
    return manifest


def validate_traceability_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    missing = list(manifest.get("missing_required_artifacts", []) or [])
    single = manifest.get("single_writer_contract", {}) if isinstance(manifest.get("single_writer_contract"), dict) else {}
    ok = not missing and bool(single.get("ok", False))
    return {
        "ok": ok,
        "missing_required_artifacts": missing,
        "single_writer_contract_ok": bool(single.get("ok", False)),
        "final_route_write_count": int(single.get("final_route_write_count", 0) or 0),
        "unexpected_actions": list(single.get("unexpected_actions", []) or []),
        "touch_paths_ok": bool(single.get("touch_paths_ok", False)),
        "touch_path_mismatches": list(single.get("touch_path_mismatches", []) or []),
        "hash_match": bool(single.get("hash_match", False)),
        "final_dem_is_real_file": bool(single.get("final_dem_is_real_file", False)),
        "final_dem_path_ok": bool(single.get("final_dem_path_ok", False)),
        "identity_receipt_ok": bool(single.get("identity_receipt_ok", False)),
    }


def write_traceability_manifest(out_dir: Path, report: Dict[str, Any]) -> Tuple[Path, Path, Path, Dict[str, Any], Dict[str, Any]]:
    out_dir = Path(out_dir).resolve()
    manifest = build_traceability_manifest(out_dir, report)
    validation = validate_traceability_manifest(manifest)
    manifest["validation"] = validation
    json_path = out_dir / "traceability_manifest.json"
    md_path = out_dir / "traceability_manifest.md"
    contract_path = out_dir / "traceability_contract.json"
    write_json(json_path, manifest)
    write_json(contract_path, validation)
    lines = [
        "# Traceability manifest",
        "",
        f"- run_id: `{manifest.get('run_id')}`",
        f"- final_dem: `{manifest.get('final_dem_path')}`",
        f"- traceability_ok: `{validation.get('ok')}`",
        f"- single_writer_contract_ok: `{validation.get('single_writer_contract_ok')}`",
        f"- final_dem_path_ok: `{validation.get('final_dem_path_ok')}`",
        f"- touch_paths_ok: `{validation.get('touch_paths_ok')}`",
        f"- identity_receipt_ok: `{validation.get('identity_receipt_ok')}`",
        "",
        "## Required artifacts",
    ]
    for key, entry in manifest.get("required_artifacts", {}).items():
        lines.append(f"- `{key}`: `{entry.get('path')}` exists={entry.get('exists')}")
    io_obs = manifest.get("io_observed", {}) if isinstance(manifest.get("io_observed"), dict) else {}
    lines.extend(["", "## IO observed",
                  f"- inputs: `{io_obs.get('input_count')}`",
                  f"- outputs: `{io_obs.get('output_count')}`",
                  "", "## Stages"])
    for stage in manifest.get("stages", []):
        lines.append(f"- `{stage.get('stage')}` status={stage.get('status')} outputs={len(stage.get('outputs', []))}")
    lines.extend(["", "## Final DEM touch actions"])
    for action in manifest.get("touch_log_actions", []):
        lines.append(f"- `{action.get('action')}` path=`{action.get('path')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, contract_path, manifest, validation
