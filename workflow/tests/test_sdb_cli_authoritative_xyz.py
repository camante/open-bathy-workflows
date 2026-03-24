from types import SimpleNamespace

from sdb_cli import augment_sdb_command, build_sdb_command


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



def test_build_sdb_command_uses_precomputed_guidance_domain_mask():
    cfg = SimpleNamespace(
        aoi='-71/-70/42/43',
        start_date='2025-01-01',
        end_date='2025-12-31',
        cloud='10',
        icesat='auto',
        sdb_mode='train',
        cache_root='cache',
        align_mode='auto',
        working_srs='EPSG:26919',
        working_vcrs_epsg=5703,
        sdb_guidance_domain_mask='cache/domains/sdb_guidance_domain_mask.tif',
    )
    cmd = build_sdb_command(cfg, out_dir='out/sdb')
    assert '--land-mask=cache/domains/sdb_guidance_domain_mask.tif' in cmd
    assert '--land-mask-type=land_binary' in cmd
    assert '--land-mask-water-val=0' in cmd
