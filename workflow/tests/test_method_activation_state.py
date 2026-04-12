import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from method_activation import determine_method_activation_truth
from workflow_execution_state import build_workflow_execution_state


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


def test_determine_method_activation_truth_records_truth_objects(tmp_path: Path):
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

    effective, meta, truths = determine_method_activation_truth(cfg, report)

    assert effective == ['river', 'fuse']
    assert truths['river'].effective_should_run is True
    assert truths['sdb'].effective_should_run is False
    assert report['method_activation_truth']['sdb']['effective_should_run'] is False
    assert meta['method_activation_truth']['river']['effective_should_run'] is True


def test_build_workflow_execution_state_uses_activation_truth():
    cfg = SimpleNamespace(out_dir='/tmp/out')
    report = {
        'guidance_domains': {
            'activation': {
                'requested': ['river', 'fuse'],
                'effective': ['river', 'fuse'],
            }
        },
        'method_activation_truth': {
            'river': {
                'method_name': 'river',
                'requested': True,
                'requested_reason': 'requested',
                'domain_summary_should_run': True,
                'validated_mask_should_run': True,
                'validated_mask_pixels': 10,
                'effective_should_run': True,
                'effective_reason': 'validated_execution_mask',
                'candidate_domain_mask': None,
                'active_domain_mask': 'river_mask.tif',
                'semantic_valid': None,
                'semantic_reason': None,
                'active_guidance_product': None,
            }
        }
    }
    state = build_workflow_execution_state(
        cfg=cfg,
        report=report,
        final_native='native.tif',
        final_for_user='user.tif',
        final_provenance='prov.tif',
    )
    payload = state.to_report_dict()
    assert payload['effective_methods'] == ['river', 'fuse']
    assert payload['method_activation']['river']['effective_should_run'] is True
    assert payload['final_outputs']['final_native'] == 'native.tif'
