"""Tests for errors_scientific.py — scientific fallback classification."""

import unittest


class TestFallbackClass(unittest.TestCase):

    def test_enum_values(self):
        from errors_scientific import FallbackClass
        self.assertEqual(FallbackClass.SAFE.value, "safe")
        self.assertEqual(FallbackClass.DEGRADED.value, "degraded")
        self.assertEqual(FallbackClass.INVALID.value, "invalid")


class TestFallbackRegistry(unittest.TestCase):

    def setUp(self):
        from errors_scientific import FallbackRegistry, FallbackClass
        self.Registry = FallbackRegistry
        self.FC = FallbackClass

    def test_empty_registry(self):
        r = self.Registry()
        self.assertEqual(r.run_validity, "valid")
        self.assertFalse(r.has_invalid)
        self.assertFalse(r.has_degraded)
        s = r.summary()
        self.assertEqual(s["n_events"], 0)

    def test_safe_event(self):
        r = self.Registry()
        r.record("plot_generation", self.FC.SAFE, stage="visualization",
                 detail="matplotlib not available")
        self.assertEqual(r.run_validity, "valid")
        self.assertEqual(len(r.events), 1)
        self.assertEqual(r.events[0].fallback_class, self.FC.SAFE)

    def test_degraded_event(self):
        r = self.Registry()
        r.record("spatial_validation", self.FC.DEGRADED,
                 stage="training",
                 detail="fell back to random split (insufficient spatial coverage)")
        self.assertEqual(r.run_validity, "degraded")
        self.assertTrue(r.has_degraded)

    def test_invalid_event(self):
        r = self.Registry()
        r.record("training_collapse", self.FC.INVALID,
                 stage="training",
                 detail="0 training samples after QC",
                 error=ValueError("empty training set"))
        self.assertEqual(r.run_validity, "invalid")
        self.assertTrue(r.has_invalid)
        self.assertEqual(r.events[0].error_type, "ValueError")

    def test_mixed_events(self):
        r = self.Registry()
        r.record("plot", self.FC.SAFE, stage="vis")
        r.record("kd_filter", self.FC.DEGRADED, stage="pre_fusion")
        self.assertEqual(r.run_validity, "degraded")
        s = r.summary()
        self.assertEqual(s["n_safe"], 1)
        self.assertEqual(s["n_degraded"], 1)
        self.assertEqual(s["n_invalid"], 0)

    def test_invalid_overrides_degraded(self):
        r = self.Registry()
        r.record("plot", self.FC.SAFE, stage="vis")
        r.record("kd", self.FC.DEGRADED, stage="filter")
        r.record("datum", self.FC.INVALID, stage="post")
        self.assertEqual(r.run_validity, "invalid")

    def test_summary_serializable(self):
        import json
        r = self.Registry()
        r.record("test", self.FC.SAFE, stage="unit_test", detail="ok")
        s = r.summary()
        # Should be JSON-serializable
        json_str = json.dumps(s)
        self.assertIn("run_validity", json_str)

    def test_event_to_dict(self):
        from errors_scientific import FallbackEvent, FallbackClass
        evt = FallbackEvent(
            name="test", fallback_class=FallbackClass.DEGRADED,
            stage="training", detail="something")
        d = evt.to_dict()
        self.assertEqual(d["class"], "degraded")
        self.assertEqual(d["name"], "test")


class TestRecordFallbackConvenience(unittest.TestCase):

    def test_with_registry(self):
        from errors_scientific import record_fallback, FallbackRegistry, FallbackClass
        r = FallbackRegistry()
        record_fallback(r, "test", FallbackClass.SAFE, stage="s", detail="d")
        self.assertEqual(len(r.events), 1)

    def test_without_registry(self):
        """Should not raise when registry is None."""
        from errors_scientific import record_fallback, FallbackClass
        record_fallback(None, "test", FallbackClass.SAFE, stage="s", detail="d")


if __name__ == "__main__":
    unittest.main()
