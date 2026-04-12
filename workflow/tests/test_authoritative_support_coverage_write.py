import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class _FailPyogrioThenWrite:
    def __init__(self):
        self.calls = []

    def to_file(self, path, driver="GPKG", engine=None):
        self.calls.append((Path(path), driver, engine))
        if engine != "fiona":
            raise RuntimeError("unexpected engine")
        Path(path).write_text("gpkg", encoding="utf-8")


class _TypeErrorLegacyWriter:
    def __init__(self):
        self.calls = []

    def to_file(self, path, driver="GPKG"):
        self.calls.append((Path(path), driver))
        Path(path).write_text("gpkg", encoding="utf-8")


class TestAuthoritativeSupportCoverageWrite(unittest.TestCase):
    def test_write_support_coverage_prefers_fiona_staging_and_replace(self):
        import cudem_authoritative
        importlib.reload(cudem_authoritative)
        writer = _FailPyogrioThenWrite()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "authoritative_support_coverage.gpkg"
            cudem_authoritative._write_support_coverage(out, writer)
            self.assertTrue(out.exists())
            self.assertEqual(writer.calls[0][2], "fiona")
            self.assertFalse((Path(td) / "authoritative_support_coverage.tmp.gpkg").exists())

    def test_write_support_coverage_retries_without_engine_for_legacy_writer(self):
        import cudem_authoritative
        importlib.reload(cudem_authoritative)
        writer = _TypeErrorLegacyWriter()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "authoritative_support_coverage.gpkg"
            cudem_authoritative._write_support_coverage(out, writer)
            self.assertTrue(out.exists())
            self.assertEqual(len(writer.calls), 1)

    def test_write_support_coverage_raises_with_parent_state(self):
        import cudem_authoritative
        importlib.reload(cudem_authoritative)

        class _AlwaysFail:
            def to_file(self, path, driver="GPKG", engine=None):
                raise OSError("boom")

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "authoritative_support_coverage.gpkg"
            with self.assertRaises(RuntimeError) as ctx:
                cudem_authoritative._write_support_coverage(out, _AlwaysFail())
            self.assertIn("parent_exists=True", str(ctx.exception))
            self.assertIn("writable=True", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
