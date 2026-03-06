"""Compile all workflow Python modules.

This must be a real unittest so `python -m unittest discover` cannot silently
skip it.

It catches syntax/indentation errors early.
"""

from __future__ import annotations

import pathlib
import py_compile
import unittest


class TestCompileAll(unittest.TestCase):
    def test_all_python_files_compile(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[1]
        py_files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in str(p))
        errs: list[tuple[str, str]] = []

        for p in py_files:
            try:
                py_compile.compile(str(p), doraise=True)
            except Exception as e:
                errs.append((str(p.relative_to(root)), str(e)))

        if errs:
            # Keep failure message bounded.
            head = "\n".join([f"{name}: {err}" for name, err in errs[:25]])
            self.fail(f"py_compile failed for {len(errs)} files (showing up to 25):\n{head}")
