from pathlib import Path
from unittest.mock import patch

import sdb_main


def _call_generate(tmp_path: Path, mode: str):
    cache = tmp_path / 'cache'
    cache.mkdir()
    out = tmp_path / 'aligned.tif'
    template = tmp_path / 'template.tif'
    template.write_bytes(b'not-used')
    cmds = []

    def fake_run(cmd):
        cmds.append(cmd)
        base_prefix = Path(cmd[-1])
        base_prefix.with_suffix('.tif').write_bytes(b'raw')

    with patch.object(sdb_main, '_run_safe', side_effect=fake_run), \
         patch.object(sdb_main, 'align_mask_to_target', side_effect=lambda src, target, dst: Path(dst).write_bytes(b'aligned')):
        sdb_main.generate_coastline_mask(str(template), '-71/-70/42/43', cache, str(out), mode)
    return ' '.join(cmds[0])


def test_ocean_mode_does_not_request_nhd_or_lakes(tmp_path: Path):
    cmd = _call_generate(tmp_path, 'ocean')
    assert 'want_nhd=false' in cmd
    assert 'want_lakes=false' in cmd


def test_all_sdb_mode_requests_lakes_but_not_nhd(tmp_path: Path):
    cmd = _call_generate(tmp_path, 'all_sdb')
    assert 'want_nhd=false' in cmd
    assert 'want_lakes=true' in cmd


def test_lakes_mode_requests_lakes_but_not_nhd(tmp_path: Path):
    cmd = _call_generate(tmp_path, 'lakes')
    assert 'want_nhd=false' in cmd
    assert 'want_lakes=true' in cmd
