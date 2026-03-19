from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_river_domain_mask_for_fusion(cfg, report: Dict[str, Any]) -> Optional[Path]:
    river_domain_mask_for_fusion: Optional[Path] = None
    river_section = report.get("river", {})
    if isinstance(river_section, dict):
        outputs = river_section.get("outputs", {})
        if isinstance(outputs, dict):
            cand = outputs.get("river_channel_mask") or outputs.get("channel_mask") or outputs.get("river_domain_mask")
            if cand:
                river_domain_mask_for_fusion = Path(str(cand))
    if river_domain_mask_for_fusion is None or not river_domain_mask_for_fusion.exists():
        ch = getattr(cfg, "river_channel_mask", None)
        if ch:
            cand = Path(str(ch))
            if cand.exists():
                river_domain_mask_for_fusion = cand
    return river_domain_mask_for_fusion


def copy_fusion_outputs(res, out_depth: Path, out_prov: Optional[Path]):
    combined = Path(res.combined_raster)
    try:
        shutil.copy2(str(combined), str(out_depth))
    except (FileNotFoundError, OSError, shutil.Error):
        out_depth = combined

    if getattr(res, "provenance_raster", None):
        prov_src = Path(res.provenance_raster)
        if out_prov is not None:
            try:
                shutil.copy2(str(prov_src), str(out_prov))
            except (FileNotFoundError, OSError, shutil.Error):
                out_prov = prov_src
        else:
            out_prov = prov_src
    return out_depth, out_prov
