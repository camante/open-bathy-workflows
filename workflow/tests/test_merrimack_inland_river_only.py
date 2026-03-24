from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio


@pytest.mark.heavy
def test_merrimack_inland_is_river_only_and_preserves_positive_support():
    out_dir = os.environ.get("MERRIMACK_INLAND_OUTPUT_DIR")
    if not out_dir:
        pytest.skip("MERRIMACK_INLAND_OUTPUT_DIR not set")
    root = Path(out_dir)
    manifest_path = root / "guidance_domains" / "guidance_domains_manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diag = manifest.get("diagnostics", {})
    assert manifest.get("inland_only") is True or diag.get("inland_only") is True
    river_domain = Path(manifest.get("river_domain_to_check") or root / "guidance_domains" / "river_guidance_domain_mask.tif")
    sdb_domain = Path(manifest.get("sdb_domain_to_check") or root / "guidance_domains" / "sdb_guidance_domain_mask.tif")
    with rasterio.open(river_domain) as ds:
        assert int(np.count_nonzero(ds.read(1) > 0)) > 0
    with rasterio.open(sdb_domain) as ds:
        assert int(np.count_nonzero(ds.read(1) > 0)) == 0
    csv_path = os.environ.get("MERRIMACK_INLAND_RIVER_SUPPORT_CSV")
    if csv_path and Path(csv_path).exists():
        vals = []
        for line in Path(csv_path).read_text(encoding="utf-8").splitlines()[1:]:
            try:
                vals.append(float(line.split(",")[2]))
            except Exception:
                pass
        assert vals and any(v > 0.0 for v in vals)
