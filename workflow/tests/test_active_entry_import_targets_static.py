"""Static package-safety check for active workflow entry modules.

This test deliberately avoids importing geospatial dependencies. It verifies
that active entry files do not reference local modules that are absent from the
packaged workflow tree.  It uses a line-oriented import scan rather than
``ast.parse`` because the monolithic orchestrator is very large and can be slow
to parse in minimal Python runtimes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY_FILES = (
    "bathy_main.py",
    "active_pipeline.py",
    "river_runner.py",
    "river_runner_contract.py",
    "river_workflow_entry.py",
)
STDLIB_TOP_LEVEL = frozenset(
    {
        "__future__",
        "argparse",
        "collections",
        "contextlib",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "enum",
        "fnmatch",
        "glob",
        "hashlib",
        "inspect",
        "itertools",
        "json",
        "logging",
        "math",
        "multiprocessing",
        "os",
        "pathlib",
        "platform",
        "re",
        "shlex",
        "shutil",
        "statistics",
        "subprocess",
        "sys",
        "tempfile",
        "time",
        "traceback",
        "typing",
        "uuid",
        "warnings",
        "concurrent",
    }
)
GEOSPATIAL_OR_THIRD_PARTY = frozenset(
    {
        "affine",
        "fiona",
        "geopandas",
        "matplotlib",
        "numpy",
        "osgeo",
        "pandas",
        "pymt",
        "pyproj",
        "rasterio",
        "requests",
        "rioxarray",
        "scipy",
        "shapely",
        "sklearn",
        "tqdm",
        "xarray",
        "yaml",
    }
)
IMPORT_RE = re.compile(r"^\s*import\s+(.+?)\s*(?:#.*)?$")
FROM_RE = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*?)\s+import\s+.*$")


def _local_top_level_names() -> set[str]:
    file_modules = {p.stem for p in ROOT.glob("*.py")}
    package_modules = {p.parts[len(ROOT.parts)] for p in ROOT.rglob("__init__.py") if len(p.parts) > len(ROOT.parts)}
    return file_modules | package_modules


def _absolute_import_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    continuation = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = (continuation + raw_line.strip()) if continuation else raw_line.strip()
        if line.endswith("\\"):
            continuation = line[:-1] + " "
            continue
        continuation = ""
        from_match = FROM_RE.match(line)
        if from_match:
            modules.add(from_match.group(1))
            continue
        import_match = IMPORT_RE.match(line)
        if import_match:
            for item in import_match.group(1).split(","):
                name = item.strip().split(" as ")[0].strip()
                if name:
                    modules.add(name)
    return modules


def test_active_entry_import_targets_resolve() -> None:
    local = _local_top_level_names()
    missing: list[str] = []
    for rel in ENTRY_FILES:
        path = ROOT / rel
        assert path.is_file(), f"active entry file missing: {rel}"
        for module in sorted(_absolute_import_modules(path)):
            top = module.split(".")[0]
            if top in STDLIB_TOP_LEVEL or top in GEOSPATIAL_OR_THIRD_PARTY or top in local:
                continue
            missing.append(f"{rel}: {module}")
    assert not missing, "missing local import targets:\n" + "\n".join(missing)


if __name__ == "__main__":
    test_active_entry_import_targets_resolve()
    print("active_entry_import_targets_static_ok")
