"""Vertical datum conversion utilities.

Kept in one place to prevent drift between bathy_main and sdb_main.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import logging
import shutil

from process_utils import run_cmd

def convert_sdb_msl_to_navd88(
    input_tif: Path,
    output_tif: Path,
    source_vdatum: str = "epsg:4269+5714",  # NAD83 + MSL
    target_vdatum: str = "epsg:4269+5703",  # NAD83 + NAVD88
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """Convert an SDB *elevation* raster from MSL to NAVD88 using CUDEM `dlim`.

    SCIENTIFIC NOTE:
    - SDB outputs are elevations relative to MSL (Mean Sea Level = 0).
      A pixel value of -5.0m means seabed elevation is -5.0m MSL.
    - This is *not* depth below instantaneous water surface.

    The vertical datum transform is applied as:
        elev_NAVD88 = elev_MSL + (NAVD88 - MSL separation)
    """
    log = logger or logging.getLogger(__name__)
    dlim_exe = shutil.which("dlim")
    if dlim_exe is None:
        msg = "dlim not found on PATH; cannot perform vertical datum transformation"
        log.warning("[VDATUM] %s", msg)
        return False, msg

    input_tif = Path(input_tif)
    output_tif = Path(output_tif)

    if not input_tif.exists():
        msg = f"Input raster not found: {input_tif}"
        log.error("[VDATUM] %s", msg)
        return False, msg

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        dlim_exe,
        "-i", str(input_tif),
        "-J", source_vdatum,
        "-P", target_vdatum,
        "-O", str(output_tif),
    ]

    log.info("[VDATUM] Converting SDB from MSL to NAVD88")
    res = run_cmd(cmd, timeout=600)

    if res.returncode != 0:
        msg = f"dlim failed with code {res.returncode}: {res.stderr_tail[:500]}"
        log.error("[VDATUM] %s", msg)
        return False, msg

    if not output_tif.exists():
        msg = f"dlim completed but output not found: {output_tif}"
        log.error("[VDATUM] %s", msg)
        return False, msg

    return True, f"Converted {input_tif.name} from MSL to NAVD88: {output_tif}"
