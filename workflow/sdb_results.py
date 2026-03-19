from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    from process_utils import find_sdb_depth_raster
except ImportError:
    def find_sdb_depth_raster(sdb_dir: Path):
        sdb_dir = Path(sdb_dir)
        for p in sorted(sdb_dir.rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'} and 'depth' in p.name.lower():
                return p
        return None


def finalize_sdb_run(*, sdb_dir: Path, report: Dict[str, Any], apply_depth_metadata, logger) -> Optional[Path]:
    """Resolve the final SDB depth raster and apply lightweight metadata tagging."""
    depth = find_sdb_depth_raster(sdb_dir)
    if depth is None:
        logger.warning("[SDB] Completed but could not find a depth raster in SDB output tree.")
        report.setdefault("sdb", {})["depth_raster_found"] = False
        return None

    depth_path = Path(depth).resolve() if not isinstance(depth, Path) else depth.resolve()
    report.setdefault("sdb", {})["depth_raster_found"] = True
    report["sdb"]["depth_raster"] = str(depth_path)
    logger.info("[SDB] Depth raster found: %s", depth_path)
    try:
        apply_depth_metadata(depth_path)
    except (FileNotFoundError, OSError, ValueError):
        logger.debug("ignored", exc_info=True)
    return depth_path
