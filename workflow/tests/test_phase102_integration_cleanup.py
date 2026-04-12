from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

import bathy_main
from guidance_domains import GuidanceDomainPaths, _write_domain_summary, _write_domain_validation


def _write_mask(path: Path, arr: np.ndarray, *, nodata: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        transform=from_origin(0, arr.shape[0], 1, 1),
        crs='EPSG:32619',
        nodata=nodata,
    ) as ds:
        ds.write(arr, 1)


def test_run_sdb_preserves_shared_domain_receipts(monkeypatch, tmp_path):
    mask = tmp_path / 'sdb_mask.tif'
    _write_mask(mask, np.zeros((2, 2), dtype=np.uint8), nodata=1)
    cfg = SimpleNamespace(
        out_dir=tmp_path,
        sdb_guidance_domain_mask=mask,
        validated_sdb_guidance_domain_mask=mask,
        validated_sdb_guidance_domain_pixels=4,
        aoi='0/1/0/1',
    )
    report = {'guidance_domains': {'activation': {'derived_activation': {'sdb_should_run': True}}}}

    monkeypatch.setattr(bathy_main, '_build_sdb_command', lambda *args, **kwargs: ['python', 'fake'])
    monkeypatch.setattr(bathy_main, '_augment_sdb_command', lambda cfg, cmd, **kwargs: cmd)
    monkeypatch.setattr(bathy_main, '_record_authoritative_child_passthrough', lambda *args, **kwargs: None)
    monkeypatch.setattr(bathy_main, 'run_command', lambda *args, **kwargs: (1, 'out', 'err'))

    result = bathy_main.run_sdb(cfg, report)
    assert result is None
    sdb = report['sdb']
    assert sdb['shared_domain_pixels'] == 4
    assert sdb['shared_domain_activation_should_run'] is True
    assert sdb['shared_domain_mask_used'] == str(mask.resolve())
    assert sdb['status'] == 'failed'


def _paths(tmp_path: Path) -> GuidanceDomainPaths:
    gd = tmp_path / 'guidance_domains'
    review = tmp_path / 'review'
    return GuidanceDomainPaths(
        cache_dir=gd,
        run_dir=gd,
        review_dir=review,
        manifest_json=gd / 'manifest.json',
        ocean_mask=gd / 'ocean_mask.tif',
        ocean_connectivity_mask=gd / 'ocean_connectivity_mask.tif',
        with_nhd_water_mask=gd / 'with_nhd_water_mask.tif',
        river_water_support_mask=gd / 'river_water_support_mask.tif',
        river_channel_mask=gd / 'river_channel_mask.tif',
        effective_water_mask=gd / 'effective_water_mask.tif',
        corridor_mask=gd / 'corridor_mask.tif',
        nhdarea_mask=gd / 'nhdarea_mask.tif',
        open_water_mask=gd / 'open_water_mask.tif',
        mainstem_mask=gd / 'mainstem_mask.tif',
        estuary_clip_mask=gd / 'estuary_clip_mask.tif',
        estuary_transition_mask=gd / 'estuary_transition_mask.tif',
        nhd_sea_ocean_mask=gd / 'nhd_sea_ocean_mask.tif',
        river_guidance_domain_mask=gd / 'river_guidance_domain_mask.tif',
        sdb_guidance_domain_mask=gd / 'sdb_guidance_domain_mask.tif',
        river_domain_policy_json=gd / 'river_domain_policy.json',
    )


def test_domain_validation_uses_spatial_subset_not_counts(tmp_path):
    paths = _paths(tmp_path)
    arr_ocean = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    arr_estuary = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    arr_base = np.zeros((2, 2), dtype=np.uint8)
    arr_river_candidate = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    arr_river_active = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    # same count as ocean+estuary support, but one SDB water pixel is in unsupported cell [1,1]
    arr_sdb = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    arr_transition = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    arr_sea_ocean = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    _write_mask(paths.ocean_mask, arr_ocean, nodata=1)
    _write_mask(paths.ocean_connectivity_mask, arr_ocean, nodata=1)
    _write_mask(paths.with_nhd_water_mask, arr_base, nodata=1)
    _write_mask(paths.river_water_support_mask, arr_river_candidate, nodata=0)
    _write_mask(paths.effective_water_mask, arr_base, nodata=1)
    _write_mask(paths.corridor_mask, arr_river_candidate, nodata=0)
    _write_mask(paths.nhdarea_mask, arr_river_candidate, nodata=0)
    _write_mask(paths.river_channel_mask, arr_river_candidate, nodata=0)
    _write_mask(paths.river_guidance_domain_mask, arr_river_active, nodata=0)
    _write_mask(paths.estuary_clip_mask, arr_estuary, nodata=0)
    _write_mask(paths.estuary_transition_mask, arr_transition, nodata=0)
    _write_mask(paths.sdb_guidance_domain_mask, arr_sdb, nodata=1)
    _write_mask(paths.nhd_sea_ocean_mask, arr_sea_ocean, nodata=0)

    manifest = {
        'diagnostics': {},
        'outputs': {k: str(v) for k, v in paths.as_dict().items() if k not in {'cache_dir', 'run_dir'}},
    }
    paths.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    paths.river_domain_policy_json.write_text('{}', encoding='utf-8')
    paths.manifest_json.write_text(__import__('json').dumps(manifest), encoding='utf-8')
    summary_path = tmp_path / 'domains' / 'domain_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_domain_summary(summary_path, paths=paths, manifest_path=paths.manifest_json)
    validation_path = tmp_path / 'domains' / 'domain_validation.json'
    _write_domain_validation(validation_path, paths=paths, summary_path=summary_path)
    data = __import__('json').loads(validation_path.read_text(encoding='utf-8'))
    checks = {c['name']: c for c in data['checks']}
    assert checks['sdb_candidate_supported_by_connectivity_or_handoff']['ok'] is False
    assert checks['sdb_candidate_supported_by_connectivity_or_handoff']['details']['unexpected_pixels'] == 1
