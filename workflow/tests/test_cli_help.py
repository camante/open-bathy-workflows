# -*- coding: utf-8 -*-
"""Smoke test: core CLIs should support `--help` without crashing.

This is intentionally lightweight: it only asserts argparse wiring works.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run_help(script: str) -> None:
    p = subprocess.run([PY, str(ROOT / script), "--help"], capture_output=True, text=True)
    assert p.returncode == 0, f"{script} --help failed\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"


def test_bathy_main_help():
    _run_help("bathy_main.py")


def test_sdb_main_help():
    _run_help("sdb_main.py")


def test_river_skeleton_help():
    _run_help("river_skeleton_bathy.py")
