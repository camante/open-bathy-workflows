import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from provenance_reporting import write_support_provenance_summary


def test_write_support_provenance_summary_reports_classes_and_confidence(tmp_path: Path):
    final = tmp_path / "final.tif"
    prov = tmp_path / "prov.tif"
    support = tmp_path / "support.tif"
    guidance = tmp_path / "guidance.tif"
    sdb_conf = tmp_path / "sdb_conf.tif"
    river_conf = tmp_path / "river_conf.tif"
    final_regime = tmp_path / "final_regime.tif"
    sdb_regime = tmp_path / "sdb_regime.tif"
    river_regime = tmp_path / "river_regime.tif"

    tifffile.imwrite(final, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    tifffile.imwrite(prov, np.array([[10, 30], [40, 60]], dtype=np.uint8))
    tifffile.imwrite(support, np.array([[1, 3], [4, 6]], dtype=np.uint8))
    tifffile.imwrite(guidance, np.array([[0.0, 0.25], [0.75, 0.0]], dtype=np.float32))
    tifffile.imwrite(sdb_conf, np.array([[0.0, 0.2], [0.5, 0.0]], dtype=np.float32))
    tifffile.imwrite(river_conf, np.array([[0.1, 0.0], [0.8, 0.0]], dtype=np.float32))
    tifffile.imwrite(final_regime, np.array([[1, 2], [4, 3]], dtype=np.uint8))
    tifffile.imwrite(sdb_regime, np.array([[1, 2], [3, 1]], dtype=np.uint8))
    tifffile.imwrite(river_regime, np.array([[4, 4], [3, 1]], dtype=np.uint8))

    report = {
        "authoritative_base": {
            "policy": {
                "support_class_codes": {"1": "authoritative_locked", "3": "guidance_conditioned_sdb", "4": "guidance_conditioned_river", "6": "low_confidence_continuous_fill"},
                "support_class_families": {"1": "authoritative_locked", "3": "guidance_conditioned", "4": "guidance_conditioned", "6": "low_confidence_continuous_fill"},
                "provenance_class_codes": {"10": "authoritative_locked", "30": "sdb_conditioned_fill", "40": "river_conditioned_fill", "60": "low_confidence_fill"},
                "provenance_class_families": {"10": "authoritative_locked", "30": "guidance_conditioned", "40": "guidance_conditioned", "60": "low_confidence_continuous_fill"},
                "regime_class_codes": {"1": "upland", "2": "nearshore_water", "3": "estuary_transition", "4": "river_channel"},
            },
            "outputs": {
                "support_class": str(support),
                "regime_class": str(final_regime),
                "guidance_influence": str(guidance),
                "coastal_sdb_confidence": str(sdb_conf),
                "river_scaffold_confidence": str(river_conf),
            },
        },
        "sdb": {"artifacts": {"regime_class_raster": str(sdb_regime)}},
        "river": {"outputs": {"regime_class": str(river_regime)}},
        "final_dem_runtime": {},
        "outputs": {},
    }
    cfg = SimpleNamespace(out_dir=tmp_path, authoritative_base="", tile_bbox=None)
    out = write_support_provenance_summary(cfg, report, final_native=final, final_for_user=None, final_provenance=prov)
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["support_class_summary"]["by_class"]["3"]["label"] == "guidance_conditioned_sdb"
    assert payload["provenance_class_summary"]["by_family"]["guidance_conditioned"]["count"] == 2
    assert payload["guidance_influence_summary"]["guided_fraction_of_domain"] == 0.5
    assert "coastal_sdb_confidence" in payload["confidence_artifacts"]
    assert payload["regime_sources_present"] == ["final_regime_class", "sdb_regime_class", "river_regime_class"]
    assert payload["regime_class_summary"]["final_regime_class"]["by_class"]["4"]["label"] == "river_channel"
    assert payload["regime_class_summary"]["sdb_regime_class"]["by_class"]["2"]["label"] == "nearshore_water"
    assert report["outputs"]["support_provenance_summary"].endswith("support_provenance_summary.json")


def test_grid_pixel_area_uses_dx_times_dy_for_geographic_crs():
    from types import SimpleNamespace
    from provenance_reporting import _grid_pixel_size_m

    transform = SimpleNamespace(a=0.01, e=-0.02)
    crs = SimpleNamespace(is_geographic=True)
    dx_m, dy_m = _grid_pixel_size_m(transform, crs, ref_lat_deg=42.0)

    assert dx_m > 0
    assert dy_m > 0
    assert dy_m != dx_m


def test_class_summary_reports_unknown_codes():
    from provenance_reporting import _class_summary

    cls = np.array([[1, 9], [3, 3]], dtype=np.uint8)
    domain = np.ones_like(cls, dtype=bool)
    codes = {"1": "authoritative_locked", "3": "guidance_conditioned_sdb"}
    families = {"1": "authoritative_locked", "3": "guidance_conditioned"}

    summary = _class_summary(cls, codes, families, domain=domain, pixel_area_m2=4.0)

    assert summary["unknown_codes"]["count"] == 1
    assert summary["unknown_codes"]["codes"] == [9]
    assert summary["unknown_codes"]["area_m2"] == 4.0
