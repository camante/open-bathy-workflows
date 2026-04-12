import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from bathy_main import _determine_effective_methods_from_domains


def _write_mask(path: Path, arr: np.ndarray, *, nodata: int) -> None:
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',
        'transform': from_origin(-124.0, 45.0, 0.001, 0.001),
        'nodata': nodata,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr.astype('uint8'), 1)


def test_determine_effective_methods_prefers_validated_execution_masks(tmp_path: Path):
    domain_summary = tmp_path / 'domain_summary.json'
    sdb_mask = tmp_path / 'sdb_guidance_domain_mask.tif'
    river_mask = tmp_path / 'river_guidance_domain_mask.tif'

    _write_mask(sdb_mask, np.ones((4, 4), dtype=np.uint8), nodata=1)
    _write_mask(river_mask, np.ones((4, 4), dtype=np.uint8), nodata=0)

    domain_summary.write_text(json.dumps({
        'derived_activation': {
            'river_should_run': True,
            'sdb_should_run': True,
        }
    }), encoding='utf-8')

    cfg = SimpleNamespace(
        methods=['sdb', 'river', 'fuse'],
        methods_requested=['sdb', 'river', 'fuse'],
        domain_review_summary=domain_summary,
        river_guidance_domain_mask=river_mask,
        validated_sdb_guidance_domain_mask=sdb_mask,
        validated_sdb_guidance_domain_pixels=0,
    )
    report = {}

    effective, meta = _determine_effective_methods_from_domains(cfg, report)

    assert effective == ['river', 'fuse']
    assert meta['derived_activation']['river_should_run'] is True
    assert meta['derived_activation']['sdb_should_run'] is False
    assert meta['activation_mismatch']['sdb']['summary_should_run'] is True
    assert meta['activation_mismatch']['sdb']['validated_should_run'] is False
