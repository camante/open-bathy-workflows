import river_masking
from unittest import mock


def test_waffles_preflight_reports_missing_binary():
    with mock.patch('shutil.which', return_value=None):
        info = river_masking.waffles_preflight()
    assert info['available'] is False
    assert info['error'] == 'waffles_not_found_on_path'


def test_waffles_preflight_runs_help():
    with mock.patch('shutil.which', return_value='/usr/bin/waffles'):
        with mock.patch.object(river_masking, 'run_command', return_value=(0, 'ok', '')):
            info = river_masking.waffles_preflight()
    assert info['available'] is True
    assert info['help_ok'] is True
