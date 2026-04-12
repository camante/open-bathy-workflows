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

    river_lock = tmp_path / "river_lock.tif"
    tifffile.imwrite(river_lock, np.array([[0, 0], [1, 0]], dtype=np.uint8))

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
        "river": {"outputs": {"channel_surface_authoritative_lock_applied": str(river_lock)}},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="", tile_bbox=None)
    out = write_final_support_regime_audit(cfg, report, final_native=final, final_for_user=None, final_provenance=prov)
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["support_by_regime"]["by_class"]["3"]["by_class"]["2"]["label"] == "nearshore_water"
    assert payload["support_by_regime"]["by_class"]["4"]["by_class"]["4"]["count"] == 1
    assert payload["provenance_by_regime"]["by_class"]["40"]["by_class"]["4"]["count"] == 1
    assert payload["headline"]["river_guided_pixels"] == 1
    assert payload["late_stage_contract_checks"]["locked_pixels_with_nonlocked_provenance"] == 0
    assert payload["late_stage_contract_checks"]["runtime_river_authoritative_lock_pixels"] == 1
    assert payload["late_stage_contract_checks"]["runtime_river_authoritative_lock_pixels_with_authoritative_support_class"] == 0
    assert payload["late_stage_contract_checks"]["runtime_river_authoritative_lock_pixels_without_authoritative_support_class"] == 1
    assert report["outputs"]["final_support_regime_audit"].endswith("final_support_regime_audit.json")


def test_write_final_support_regime_audit_reports_river_crosswalks(tmp_path: Path):
    final = tmp_path / "final.tif"
    prov = tmp_path / "prov.tif"
    support = tmp_path / "support.tif"
    regime = tmp_path / "regime.tif"
    primary = tmp_path / "river_primary_surface_source_class.tif"
    xs = tmp_path / "river_xs_participation.tif"
    auth_part = tmp_path / "river_authoritative_participation.tif"
    lock = tmp_path / "river_lock.tif"
    pred_adm = tmp_path / "river_prediction_admissibility.tif"
    caution = tmp_path / "river_low_support_caution.tif"

    tifffile.imwrite(final, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    tifffile.imwrite(prov, np.array([[10, 40], [40, 10]], dtype=np.uint8))
    tifffile.imwrite(support, np.array([[1, 4], [5, 1]], dtype=np.uint8))
    tifffile.imwrite(regime, np.array([[4, 4], [4, 1]], dtype=np.uint8))
    tifffile.imwrite(primary, np.array([[1, 3], [5, 0]], dtype=np.uint8))
    tifffile.imwrite(xs, np.array([[0, 1], [1, 0]], dtype=np.uint8))
    tifffile.imwrite(auth_part, np.array([[1, 0], [1, 0]], dtype=np.uint8))
    tifffile.imwrite(lock, np.array([[1, 0], [0, 0]], dtype=np.uint8))
    tifffile.imwrite(pred_adm, np.array([[1, 0], [0, 0]], dtype=np.uint8))
    tifffile.imwrite(caution, np.array([[0, 1], [1, 0]], dtype=np.uint8))

    report = {
        "authoritative_base": {
            "policy": {
                "support_class_codes": {"1": "authoritative_locked", "4": "guidance_conditioned_river", "5": "scaffold_inferred"},
                "support_class_families": {"1": "authoritative_locked", "4": "guidance_conditioned", "5": "scaffold_inferred"},
                "provenance_class_codes": {"10": "authoritative_locked", "40": "river_conditioned_fill"},
                "provenance_class_families": {"10": "authoritative_locked", "40": "guidance_conditioned"},
                "regime_class_codes": {"1": "upland", "4": "river_channel"},
            },
            "outputs": {
                "support_class": str(support),
                "regime_class": str(regime),
                "river_primary_surface_source_class": str(primary),
            },
        },
        "final_dem_runtime": {"final_generation_route": "support_aware_terrain_interpolator"},
        "river": {
            "outputs": {
                "channel_surface_xs_participation": str(xs),
                "channel_surface_authoritative_participation": str(auth_part),
                "channel_surface_authoritative_lock_applied": str(lock),
                "channel_surface_prediction_admissibility": str(pred_adm),
                "channel_surface_low_support_caution": str(caution),
            }
        },
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="", tile_bbox=None)
    out = write_final_support_regime_audit(cfg, report, final_native=final, final_for_user=None, final_provenance=prov)
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["river_support_crosswalk"]["domain_pixels"] == 3
    assert payload["river_support_crosswalk"]["support_by_primary_surface_source"]["available"] is True
    assert payload["river_support_crosswalk"]["support_by_primary_surface_source"]["support"]["by_class"]["1"]["label"] == "longitudinal_backbone"
    assert payload["river_support_crosswalk"]["support_by_xs_participation"]["support"]["by_class"]["1"]["label"] == "present"
    assert payload["river_support_crosswalk"]["support_by_authoritative_participation"]["support"]["by_class"]["1"]["label"] == "present"
    assert payload["river_support_crosswalk"]["support_by_runtime_authoritative_lock"]["support"]["by_class"]["1"]["label"] == "present"
    assert payload["river_support_crosswalk"]["support_by_prediction_admissibility"]["support"]["by_class"]["1"]["label"] == "present"
    assert payload["river_support_crosswalk"]["support_by_low_support_caution"]["support"]["by_class"]["1"]["label"] == "present"
