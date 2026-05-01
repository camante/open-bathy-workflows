#!/usr/bin/env python3
"""Fast static package-integrity checks for delivered workflow zips.

This checker intentionally avoids importing workflow modules and avoids parsing
large geospatial scripts. It verifies that local module imports resolve and that
explicit local ``from module import symbol`` imports generally refer to a top
level symbol or local submodule. External dependencies are ignored.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "fresh_package_integrity.json"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "cache", "output"}
IMPORT_RE = re.compile(r"^\s*import\s+(.+)$")
FROM_RE = re.compile(r"^\s*from\s+([\w\.]+|\.+[\w\.]*)\s+import\s+(.+)$")
DEF_RE = re.compile(r"^\s*(?:def|class|async\s+def)\s+([A-Za-z_]\w*)\b|^\s*([A-Za-z_]\w*)\s*(?::[^=]+)?=")


def _python_files() -> list[Path]:
    """Return active package files, not historical legacy/test inventory."""
    manifest_path = ROOT / "active_river_modules.json"
    files: set[Path] = set()
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "entrypoint_orchestration",
            "active_river_core",
            "active_parent_export_identity",
            "active_verify_reporting_boundary_files",
            "verify_only_tools",
        ):
            for rel in manifest.get(key, []):
                path = ROOT / rel
                if path.suffix == ".py" and path.is_file():
                    files.add(path)
        for rel in manifest.get("active_transitional_utilities", []):
            path = ROOT / rel
            if path.suffix == ".py" and path.is_file():
                files.add(path)
    if files:
        return sorted(files)

    # Fallback for older packages without an active manifest.
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            if filename.endswith(".py"):
                files.add(Path(dirpath) / filename)
    return sorted(files)


def _module_file(module: str) -> Path | None:
    parts = module.split(".") if module else []
    if not parts:
        return None
    as_file = ROOT.joinpath(*parts).with_suffix(".py")
    if os.path.isfile(as_file):
        return as_file
    as_pkg = ROOT.joinpath(*parts, "__init__.py")
    if os.path.isfile(as_pkg):
        return as_pkg
    return None


def _module_or_package_exists(module: str) -> bool:
    return _module_file(module) is not None


def _local_top_modules() -> set[str]:
    tops: set[str] = set()
    for name in os.listdir(ROOT):
        path = ROOT / name
        if name.startswith("."):
            continue
        if os.path.isfile(path) and path.suffix == ".py":
            tops.add(path.stem)
        elif os.path.isdir(path) and os.path.isfile(path / "__init__.py"):
            tops.add(path.name)
    return tops


def _module_name_for_file(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative_module(current_module: str, level: int, module: str | None) -> str | None:
    parts = current_module.split(".") if current_module else []
    if parts:
        parts = parts[:-1]
    if level > len(parts) + 1:
        return None
    base = parts[: max(0, len(parts) - level + 1)]
    if module:
        base.extend(module.split("."))
    return ".".join(p for p in base if p)


def _defined_symbols(module_file: Path) -> set[str]:
    symbols: set[str] = set()
    try:
        lines = module_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return symbols
    for line in lines:
        match = DEF_RE.match(line)
        if match:
            symbols.add(match.group(1) or match.group(2))
        stripped = line.strip()
        if stripped.startswith("import "):
            for part in stripped[len("import "):].split("#", 1)[0].split(","):
                name = part.strip().split(" as ", 1)[-1].strip() or part.strip().split(".")[0]
                if name:
                    symbols.add(name.split(".")[0])
        elif stripped.startswith("from ") and " import " in stripped:
            names = stripped.split(" import ", 1)[1].split("#", 1)[0]
            for part in names.split(","):
                name = part.strip().split(" as ", 1)[-1].strip()
                if name and name != "*":
                    symbols.add(name)
    return symbols


def _top_level_imports(path: Path) -> list[tuple[str, str | None, list[str], int]]:
    """Return tuples of (kind, module, names, level)."""
    imports: list[tuple[str, str | None, list[str], int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = IMPORT_RE.match(line)
        if m:
            names = [p.strip().split(" as ", 1)[0].strip() for p in m.group(1).split("#", 1)[0].split(",")]
            imports.append(("import", None, [n for n in names if n], 0))
            continue
        m = FROM_RE.match(line)
        if m:
            raw_module = m.group(1)
            level = len(raw_module) - len(raw_module.lstrip("."))
            module = raw_module.lstrip(".") or None
            raw_names = m.group(2).split("#", 1)[0]
            names = [p.strip().strip("()").split(" as ", 1)[0].strip() for p in raw_names.split(",")]
            names = [n for n in names if n and n not in {"(", ")"}]
            # Multi-line imports that start with an opening parenthesis are
            # module-resolution checks only; symbol names are collected on
            # later lines and are intentionally skipped by this lightweight
            # scanner.
            if raw_names.strip().startswith("("):
                names = []
            imports.append(("from", module, names, level))
    return imports


def run_checks(write_report: bool = True) -> dict:
    local_tops = _local_top_modules()
    missing_local_modules: list[str] = []
    missing_local_symbols: list[str] = []
    checked_files = 0
    symbol_cache: dict[Path, set[str]] = {}

    def symbols_for(module_file: Path) -> set[str]:
        if module_file not in symbol_cache:
            symbol_cache[module_file] = _defined_symbols(module_file)
        return symbol_cache[module_file]

    for path in _python_files():
        checked_files += 1
        current_module = _module_name_for_file(path)
        for kind, module, names, level in _top_level_imports(path):
            if kind == "import":
                for name in names:
                    top = name.split(".")[0]
                    if top in local_tops and not _module_or_package_exists(name):
                        missing_local_modules.append(f"{path.relative_to(ROOT)} imports missing local module {name}")
                continue
            if level:
                resolved = _resolve_relative_module(current_module, level, module)
                is_local = True
            else:
                resolved = module
                is_local = bool(resolved and resolved.split(".")[0] in local_tops)
            if not resolved or not is_local:
                continue
            module_file = _module_file(resolved)
            if module_file is None:
                missing_local_modules.append(f"{path.relative_to(ROOT)} imports missing local module {resolved}")
                continue
            symbols = symbols_for(module_file)
            for name in names:
                if name == "*":
                    continue
                if name not in symbols and not _module_or_package_exists(f"{resolved}.{name}"):
                    missing_local_symbols.append(
                        f"{path.relative_to(ROOT)} imports missing local symbol {name} from {resolved}"
                    )

    result = {
        "checked_files": checked_files,
        "missing_local_modules": sorted(set(missing_local_modules)),
        "missing_local_symbols": sorted(set(missing_local_symbols)),
    }
    if write_report:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run_checks(write_report=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(1 if result["missing_local_modules"] or result["missing_local_symbols"] else 0)
