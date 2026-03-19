import numpy as np

import support_contract_runtime as scr
from support_classes import RegimeClass, SupportClass


def test_build_contract_target_mask_array_uses_support_and_regime_contract(monkeypatch) -> None:
    template = object()
    eligible_arr = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.uint8)
    support_arr = np.array([
        [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB), int(SupportClass.GUIDANCE_CONDITIONED_RIVER)],
        [int(SupportClass.SCAFFOLD_INFERRED), int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL), int(SupportClass.ANCHORED_INTERPOLATION)],
    ], dtype=np.uint8)
    regime_arr = np.array([
        [int(RegimeClass.UPLAND), int(RegimeClass.NEARSHORE_WATER), int(RegimeClass.RIVER_CHANNEL)],
        [int(RegimeClass.ESTUARY_TRANSITION), int(RegimeClass.UPLAND), int(RegimeClass.RIVER_CHANNEL)],
    ], dtype=np.uint8)

    def fake_align(*, template_path, src_path, dtype="uint8", nodata_value=0):
        assert template_path is template
        if src_path == "eligible":
            return eligible_arr
        if src_path == "support":
            return support_arr
        if src_path == "regime":
            return regime_arr
        return None

    monkeypatch.setattr(scr, "_align_raster_to_template", fake_align)
    mask = scr.build_contract_target_mask_array(
        template_path=template,
        eligible_fill_mask_path="eligible",
        support_class_path="support",
        regime_class_path="regime",
        allow_low_confidence_fill=True,
    )
    expected = np.array([[False, True, True], [True, False, False]], dtype=bool)
    assert np.array_equal(mask, expected)


def test_build_contract_target_mask_array_falls_back_when_only_eligible_exists(monkeypatch) -> None:
    eligible_arr = np.array([[1, 0], [1, 1]], dtype=np.uint8)

    def fake_align(*, template_path, src_path, dtype="uint8", nodata_value=0):
        return eligible_arr if src_path == "eligible" else None

    monkeypatch.setattr(scr, "_align_raster_to_template", fake_align)
    mask = scr.build_contract_target_mask_array(
        template_path=object(),
        eligible_fill_mask_path="eligible",
        support_class_path=None,
        regime_class_path=None,
    )
    assert np.array_equal(mask, eligible_arr > 0)
