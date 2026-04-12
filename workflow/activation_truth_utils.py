from __future__ import annotations

from pathlib import Path
from typing import Optional


def count_mask_value_pixels(mask_tif: Path, *, value: int) -> Optional[int]:
    try:
        import numpy as np
        import rasterio
        with rasterio.open(mask_tif) as ds:
            arr = ds.read(1, masked=False)
        return int(np.count_nonzero(arr == value))
    except (OSError, ValueError, RuntimeError, ImportError):
        return None
