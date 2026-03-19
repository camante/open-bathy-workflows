import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from final_support_audit import write_final_support_regime_audit


def test_write_final_support_regime_audit_reports_cross_tabs_and_checks(tmp_path: Path):
    final = tmp_path / "final.tif"
    prov = tmp_path / "prov.tif"
    support = tmp_path / "support.tif"
    regime = tmp_path / "regime.tif"

    tifffile.imwrite(final, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    tifffile.imwrite(prov, np.array([[10, 30], [40, 60]], dtype=np.uint8))
    tifffile.imwrite(support, np.array([[1, 3], [4, 6]], dtype=np.uint8))
    tifffile.imwrite(regime, np.array([[1, 2], [4, 3]], dtype=np.uint8))

    report = {
        "authoritative_base": {
            "policy": {
                "support_class_codes": {"1": "authoritative_locked", "3": "guidance_conditioned_sdb", "4": "guidance_conditioned_river", "6": "low_confidence_continuous_fill"},
                "support_class_families": {"1": "authoritative_locked", "3": "guidance_conditioned", "4": "guidance_conditioned", "6": "low_confidence_continuous_fill"},
                "provenance_class_codes": {"10": "authoritative_locked", "30": "sdb_conditioned_fill", "40": "river_conditioned_fill", "60": "low_confidence_fill"},
                "provenance_class_families": {"10": "authoritative_locked", "30": "guidance_conditioned", "40": "guidance_conditioned", "60": "low_confidence_continuous_fill"},
                "regime_class_codes": {"1": "upland", "2": "nearshore_water", "3": "estuary_transition", "4": "river_channel"},
            },
            "outputs": {"support_class": str(support), "regime_class": str(regime)},
        },
        "final_dem_runtime": {"final_generation_route": "support_aware_terrain_interpolator"},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="", tile_bbox=None)
    out = write_final_support_regime_audit(cfg, report, final_native=final, final_for_user=None, final_provenance=prov)
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["support_by_regime"]["by_class"]["3"]["by_class"]["2"]["label"] == "nearshore_water"
    assert payload["support_by_regime"]["by_class"]["4"]["by_class"]["4"]["count"] == 1
    assert payload["provenance_by_regime"]["by_class"]["40"]["by_class"]["4"]["count"] == 1
    assert payload["headline"]["river_guided_pixels"] == 1
    assert payload["late_stage_contract_checks"]["locked_pixels_with_nonlocked_provenance"] == 0
    assert report["outputs"]["final_support_regime_audit"].endswith("final_support_regime_audit.json")
