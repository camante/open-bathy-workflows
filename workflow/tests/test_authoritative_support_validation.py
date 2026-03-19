from pathlib import Path
from types import SimpleNamespace

import authoritative_support


def test_prepare_sdb_guidance_requires_written_output(tmp_path: Path):
    cfg = SimpleNamespace(authoritative_base=str(tmp_path / "auth.tif"), cache_root=str(tmp_path / "cache"), aoi="0/1/0/1", extra_xyz_crs="EPSG:4326", working_srs="EPSG:4326", working_vcrs_epsg=5703, sdb_source_vdatum="epsg:4269+5714")
    Path(cfg.authoritative_base).write_text("x", encoding="utf-8")

    def _ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _hash(*_args):
        return "hash"

    def _prepare(*_args, **_kwargs):
        return {"status": "ok"}

    report = {}
    out = authoritative_support.prepare_sdb_guidance_from_authoritative(
        cfg=cfg,
        report=report,
        ensure_dir_fn=_ensure_dir,
        hash_key_fn=_hash,
        prepare_points_fn=_prepare,
    )
    assert out is None
    assert "Expected authoritative support artifact was not written" in report["authoritative_base"]["sdb_guidance"]["error"]
    assert not hasattr(cfg, "sdb_authoritative_extra_xyz")
