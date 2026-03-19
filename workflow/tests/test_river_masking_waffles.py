import tempfile
import unittest
from pathlib import Path
from unittest import mock

import river_masking


class TestEnsureWafflesCoastlineMask(unittest.TestCase):
    def test_writes_expected_mask_path_via_shared_command_helper(self):
        with tempfile.TemporaryDirectory() as td:
            cache_masks = Path(td)

            def _fake_run_command(cmd, **kwargs):
                out_prefix = Path(cmd[cmd.index('-O') + 1])
                out_prefix.with_suffix('.tif').write_bytes(b'fake')
                return 0, '', ''

            with mock.patch.object(river_masking, 'run_command', side_effect=_fake_run_command):
                out = river_masking.ensure_waffles_coastline_mask(
                    cache_masks,
                    '-71/-70.75/42.75/43',
                    inc_arcsec=1.0,
                    want_nhd=False,
                    want_lakes=False,
                    prefix='waffles_coastline_ocean_only',
                    force=True,
                    log_enabled=False,
                )

            self.assertTrue(out.exists())
            self.assertEqual(out.suffix, '.tif')
            self.assertIn('waffles_coastline_ocean_only', out.name)


if __name__ == '__main__':
    unittest.main()
