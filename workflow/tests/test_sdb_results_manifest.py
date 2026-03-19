
import importlib
import json
import sys
from pathlib import Path


class _Logger:
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass


def _manifest_helper(sdb_dir: Path):
    sdb_dir = Path(sdb_dir)
    manifest = sdb_dir / "artifacts_sdb.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rel = data.get("depth_raster")
    if not isinstance(rel, str) or not rel.strip():
        return None
    p = (sdb_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
    return p if p.exists() else None


def test_finalize_sdb_run_prefers_manifest_depth(tmp_path: Path):
    sys.modules["process_utils"].find_sdb_depth_raster = _manifest_helper
    import sdb_results
    importlib.reload(sdb_results)

    sdb_dir = tmp_path / "sdb"
    sdb_dir.mkdir()
    old = sdb_dir / "aaa_depth.tif"
    old.write_bytes(b"old")
    new = sdb_dir / "rasters" / "zzz_depth.tif"
    new.parent.mkdir()
    new.write_bytes(b"new")
    (sdb_dir / "artifacts_sdb.json").write_text(json.dumps({"depth_raster": "rasters/zzz_depth.tif"}), encoding="utf-8")

    report = {}
    out = sdb_results.finalize_sdb_run(sdb_dir=sdb_dir, report=report, apply_depth_metadata=lambda *_args, **_kwargs: None, logger=_Logger())
    assert out == new.resolve()
    assert report["sdb"]["depth_raster"] == str(new.resolve())
