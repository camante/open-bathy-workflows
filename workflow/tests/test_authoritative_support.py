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


def test_prepare_authoritative_river_support_applies_withheld_support_filter(tmp_path: Path):
    auth = tmp_path / "auth.tif"
    auth.write_text("x")
    cache = tmp_path / "cache"
    withheld = tmp_path / "withheld.csv"
    withheld.write_text(
        "x,y,depth_m,authoritative_role,support_point_key\n1.0000000000,2.0000000000,-3.000000,authoritative_bed_inner,1.0000000000|2.0000000000|-3.000000|authoritative_bed_inner\n",
        encoding='utf-8',
    )
    cfg = SimpleNamespace(
        authoritative_base=str(auth),
        cache_root=str(cache),
        aoi='0/1/2/3',
        extra_xyz_crs='EPSG:4326',
        working_srs='EPSG:32619',
        working_vcrs_epsg=5703,
        river_withheld_support_csv=str(withheld),
    )
    report = {}

    def build_points(src, dst, **kwargs):
        Path(dst).write_text(
            'x,y,depth_m,source,authoritative_role\n'
            '1.0000000000,2.0000000000,-3.000000,authoritative_base,authoritative_bed_inner\n'
            '3.0000000000,4.0000000000,-5.000000,authoritative_base,authoritative_bed_core\n',
            encoding='utf-8',
        )
        return {'path': str(dst), 'count': 2}

    got = prepare_river_support_from_authoritative(
        cfg=cfg,
        report=report,
        ensure_dir_fn=lambda p: p.mkdir(parents=True, exist_ok=True) or p,
        hash_key_fn=lambda *args: 'hash3',
        prepare_points_fn=build_points,
    )
    got_path = Path(got)
    assert got_path.exists()
    text = got_path.read_text(encoding='utf-8')
    assert '1.0000000000,2.0000000000,-3.000000' not in text
    assert '3.0,4.0,-5.0' in text
    receipt = report['authoritative_base']['river_guidance']['withheld_support']
    assert receipt['removed_count'] == 1
    assert receipt['remaining_count'] == 1
    assert receipt['matched_withheld_key_count'] == 1
    assert receipt['unmatched_withheld_key_count'] == 0
