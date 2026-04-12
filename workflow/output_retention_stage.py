from __future__ import annotations

import fnmatch
import json
import shutil
import uuid
from pathlib import Path
from typing import Any


def is_probably_path(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    lowered = value.lower()
    return ('/' in value) or ('\\' in value) or lowered.endswith(('.tif', '.tiff', '.json', '.csv', '.md', '.gpkg', '.parquet'))


def _collect_output_paths(report_obj: Any) -> set[str]:
    out_paths: set[str] = set()
    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k == 'outputs' and isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, str) and is_probably_path(vv):
                            out_paths.add(vv)
                _walk(v)
        elif isinstance(o, list):
            for it in o:
                _walk(it)
    _walk(report_obj)
    return out_paths


def apply_output_retention_policy(cfg, log, report, final_path=None, final_for_user_path=None) -> None:
    out_dir = Path(cfg.out_dir)
    if not out_dir.exists():
        return

    keep_abs: set[Path] = set()
    report_paths = [out_dir / 'unified_bathy_report.json', out_dir / 'bathy_report.json']
    for rp in report_paths:
        if rp.exists() and rp.is_file():
            try:
                robj = json.loads(rp.read_text(encoding='utf-8'))
                for s in _collect_output_paths(robj):
                    pp = Path(s)
                    keep_abs.add(pp if pp.is_absolute() else (out_dir / pp).resolve())
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue

    io_json = out_dir / 'io_manifest.json'
    if io_json.exists() and io_json.is_file():
        try:
            io = json.loads(io_json.read_text(encoding='utf-8'))
            for s in (io.get('outputs') or []):
                if isinstance(s, str) and is_probably_path(s):
                    pp = Path(s)
                    keep_abs.add(pp if pp.is_absolute() else (out_dir / pp).resolve())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            pass

    keep_globs = [
        'run_summary*.json', 'run_summary*.md',
        'unified_bathy_report*.json', 'unified_bathy_report*.md',
        'bathy_report*.json', 'bathy_report*.md',
        'metrics_summary*.csv',
    ]

    final_ok = any(p.exists() for p in keep_abs)
    if (not final_ok) and final_for_user_path:
        final_ok = Path(final_for_user_path).exists()
    if (not final_ok) and final_path:
        final_ok = Path(final_path).exists()
    if not final_ok:
        log.info('[OUTPUT] Retention policy skipped (no final deliverable found).')
        return

    tmp = out_dir / f'keep_tmp_{uuid.uuid4().hex[:10]}'
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for ap in sorted(keep_abs, key=lambda x: str(x)):
            if (not ap.exists()) or (not ap.is_file()):
                continue
            try:
                rel = ap.resolve().relative_to(out_dir.resolve())
            except Exception:
                rel = Path('_external') / ap.name
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(ap), str(dst))

        for p in out_dir.iterdir():
            if p.is_file() and any(fnmatch.fnmatch(p.name, g) for g in keep_globs):
                dst = tmp / p.name
                shutil.copy2(str(p), str(dst))

        run_logs = out_dir / 'run_logs'
        if run_logs.exists() and run_logs.is_dir():
            shutil.copytree(str(run_logs), str(tmp / 'run_logs'), dirs_exist_ok=True)

        for ap in sorted(out_dir.rglob('xs_*_1d_solver_*.json')):
            if not ap.is_file():
                continue
            try:
                rel = ap.resolve().relative_to(out_dir.resolve())
            except Exception:
                rel = Path('_external') / ap.name
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(ap), str(dst))

        if bool(getattr(cfg, 'save_intermediates', False)):
            debug_root = out_dir / str(getattr(cfg, 'intermediates_dirname', 'debug'))
            debug_root.mkdir(parents=True, exist_ok=True)
            for child in list(out_dir.iterdir()):
                if child == tmp or child == debug_root:
                    continue
                if child.name == debug_root.name:
                    continue
                if (tmp / child.name).exists():
                    continue
                shutil.move(str(child), str(debug_root / child.name))
            for child in list(tmp.iterdir()):
                shutil.move(str(child), str(out_dir / child.name))
            shutil.rmtree(tmp, ignore_errors=True)
            return

        for child in list(out_dir.iterdir()):
            if child == tmp:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        for child in list(tmp.iterdir()):
            shutil.move(str(child), str(out_dir / child.name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
