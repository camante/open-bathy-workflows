from types import SimpleNamespace

from sdb_cli import augment_sdb_command


def test_augment_sdb_command_includes_authoritative_xyz_without_duplicates():
    cfg = SimpleNamespace(
        sdb_model_bank_enabled=False,
        sdb_model_cache_enabled=False,
        glint_correct=False,
        river_soundings='a.csv,b.csv',
        sdb_authoritative_extra_xyz='b.csv,c.csv',
        extra_xyz_crs='EPSG:4326',
        enable_adaptive_sampling=False,
        sampling_target_points=1000,
        sampling_min_threshold=10,
        sampling_max_gap_m=50,
    )
    cmd = augment_sdb_command(cfg, ['python', 'sdb_main.py'])
    idx = cmd.index('--extra-xyz')
    xyz = cmd[idx + 1: idx + 4]
    assert xyz == ['a.csv', 'b.csv', 'c.csv']
    assert '--extra-xyz-crs=EPSG:4326' in cmd
