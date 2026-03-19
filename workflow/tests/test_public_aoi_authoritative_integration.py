import os
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("RUN_PUBLIC_AOI_INTEGRATION") == "1", "set RUN_PUBLIC_AOI_INTEGRATION=1 to run")
class PublicAOIAuthoritativeIntegrationTest(unittest.TestCase):
    def test_public_noaa_authoritative_base_cache_reuse(self):
        from cudem_authoritative import materialize_authoritative_base_for_aoi

        # Small Merrimack-area AOI used elsewhere in this workflow.
        aoi = "-71/-70.95/42.77/42.82"
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td) / "cache"
            first = materialize_authoritative_base_for_aoi(aoi=aoi, cache_root=cache_root)
            self.assertTrue(Path(first["authoritative_base"]).exists())
            self.assertTrue(Path(first["baseline_cudem_interpolation"]).exists())
            second = materialize_authoritative_base_for_aoi(aoi=aoi, cache_root=cache_root)
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["authoritative_base"], second["authoritative_base"])
            self.assertEqual(first["baseline_cudem_interpolation"], second["baseline_cudem_interpolation"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
