from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import shutil

CACHE_DIR_NAMES = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
CACHE_FILE_SUFFIXES = (".pyc", ".pyo")


@dataclass(frozen=True)
class HygieneReport:
    bad_dirs: List[str]
    bad_files: List[str]

    @property
    def has_issues(self) -> bool:
        return bool(self.bad_dirs or self.bad_files)


def scan_repo_for_hygiene_issues(root: Path | str = ".") -> HygieneReport:
    base = Path(root)
    bad_dirs = sorted(str(p) for p in base.rglob('*') if p.is_dir() and p.name in CACHE_DIR_NAMES)
    bad_files = sorted(str(p) for p in base.rglob('*') if p.is_file() and p.suffix in CACHE_FILE_SUFFIXES)
    return HygieneReport(bad_dirs=bad_dirs, bad_files=bad_files)


def require_clean_repo_tree(root: Path | str = ".") -> HygieneReport:
    report = scan_repo_for_hygiene_issues(root)
    if report.has_issues:
        problems = []
        if report.bad_dirs:
            problems.append("cache directories: " + ", ".join(report.bad_dirs))
        if report.bad_files:
            problems.append("cache files: " + ", ".join(report.bad_files))
        raise RuntimeError("Packaging hygiene check failed; " + "; ".join(problems))
    return report


def clean_repo_hygiene(root: Path | str = ".") -> HygieneReport:
    report = scan_repo_for_hygiene_issues(root)
    for path_str in report.bad_files:
        Path(path_str).unlink(missing_ok=True)
    for path_str in report.bad_dirs:
        shutil.rmtree(path_str, ignore_errors=True)
    return report
