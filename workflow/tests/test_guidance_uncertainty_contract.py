from __future__ import annotations

import json
from pathlib import Path

import pytest

from final_guidance_uncertainty_contract import build_guidance_uncertainty_contract, write_guidance_uncertainty_contract


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x', encoding='utf-8')
    return str(path)


def test_guidance_uncertainty_contract_requires_core_outputs(tmp_path: Path):
    outputs = {
        'support_class': _touch(tmp_path / 'support_class.tif'),
        'support_distance': _touch(tmp_path / 'support_distance.tif'),
        'guidance_influence': _touch(tmp_path / 'guidance_influence.tif'),
        'anchor_uncertainty': _touch(tmp_path / 'anchor_uncertainty.tif'),
    }
    payload = build_guidance_uncertainty_contract(outputs=outputs)
    assert payload['ok'] is False
    assert 'conditioned_depth' in payload['missing_required']
    with pytest.raises(ValueError):
        write_guidance_uncertainty_contract(tmp_path / 'contract.json', outputs=outputs)


def test_guidance_uncertainty_contract_writes_when_complete(tmp_path: Path):
    outputs = {
        'support_class': _touch(tmp_path / 'support_class.tif'),
        'support_distance': _touch(tmp_path / 'support_distance.tif'),
        'guidance_influence': _touch(tmp_path / 'guidance_influence.tif'),
        'anchor_uncertainty': _touch(tmp_path / 'anchor_uncertainty.tif'),
        'guidance_uncertainty': _touch(tmp_path / 'guidance_uncertainty.tif'),
        'conditioned_uncertainty': _touch(tmp_path / 'conditioned_uncertainty.tif'),
        'conditioned_depth': _touch(tmp_path / 'conditioned_depth.tif'),
        'conditioned_provenance': _touch(tmp_path / 'conditioned_provenance.tif'),
    }
    out = write_guidance_uncertainty_contract(tmp_path / 'contract.json', outputs=outputs, support_note='ok')
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['ok'] is True
    assert payload['support_note'] == 'ok'
    assert payload['present_required']['conditioned_depth'].endswith('conditioned_depth.tif')
