from pathlib import Path
from types import SimpleNamespace

from authoritative_support import (
    prepare_river_support_from_authoritative,
    prepare_sdb_guidance_from_authoritative,
)


def test_prepare_authoritative_support_uses_cache(tmp_path: Path):
    auth = tmp_path / "auth.tif"
    auth.write_text("x")
    cache = tmp_path / "cache"
    out = cache / "authoritative_support" / "authoritative_sdb_support_hash.csv"
    out.parent.mkdir(parents=True)
    out.write_text("x,y,depth_m,source\n")
    cfg = SimpleNamespace(
        authoritative_base=str(auth),
        cache_root=str(cache),
        aoi='0/1/2/3',
        extra_xyz_crs='EPSG:4326',
        working_srs='EPSG:4326',
        working_vcrs_epsg=5703,
        sdb_source_vdatum='epsg:4269+5714',
    )
    report = {}
    got = prepare_sdb_guidance_from_authoritative(
        cfg=cfg,
        report=report,
        ensure_dir_fn=lambda p: p.mkdir(parents=True, exist_ok=True) or p,
        hash_key_fn=lambda *args: 'hash',
        prepare_points_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('should not build points')),
    )
    assert got == out
    assert Path(cfg.sdb_authoritative_extra_xyz) == out
    assert report['authoritative_base']['sdb_guidance']['cached'] is True


def test_prepare_authoritative_river_support_builds_points(tmp_path: Path):
    auth = tmp_path / "auth.tif"
    auth.write_text("x")
    cache = tmp_path / "cache"
    cfg = SimpleNamespace(
        authoritative_base=str(auth),
        cache_root=str(cache),
        aoi='0/1/2/3',
        extra_xyz_crs='EPSG:4326',
        working_srs='EPSG:32619',
        working_vcrs_epsg=5703,
    )
    report = {}

    def build_points(src, dst, **kwargs):
        Path(dst).write_text('x,y,depth_m,source\n1,2,-3,authoritative_base\n', encoding='utf-8')
        return {'path': str(dst), 'count': 1}

    got = prepare_river_support_from_authoritative(
        cfg=cfg,
        report=report,
        ensure_dir_fn=lambda p: p.mkdir(parents=True, exist_ok=True) or p,
        hash_key_fn=lambda *args: 'hash2',
        prepare_points_fn=build_points,
    )
    assert got is not None and Path(got).exists()
    assert Path(cfg.river_authoritative_soundings).exists()
    assert report['authoritative_base']['river_guidance']['count'] == 1
