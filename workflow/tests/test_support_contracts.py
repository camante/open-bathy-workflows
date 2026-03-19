from support_classes import SUPPORT_CLASS_CODE_TO_NAME
from provenance_schema import PROVENANCE_CLASS_CODE_TO_NAME
from final_dem_policy import build_final_dem_policy_dict


def test_support_contract_names_are_canonical():
    assert SUPPORT_CLASS_CODE_TO_NAME[1] == "authoritative_locked"
    assert SUPPORT_CLASS_CODE_TO_NAME[5] == "scaffold_inferred"
    assert SUPPORT_CLASS_CODE_TO_NAME[6] == "low_confidence_continuous_fill"
    assert PROVENANCE_CLASS_CODE_TO_NAME[40] == "river_conditioned_fill"
    assert PROVENANCE_CLASS_CODE_TO_NAME[60] == "low_confidence_fill"


def test_policy_builder_uses_shared_mappings():
    class Cfg:
        authoritative_support_decay_m = 300.0
        authoritative_support_density_radius_m = 250.0
        coastal_sdb_support_transition_m = 600.0
        river_anchor_density_radius_m = 200.0
        river_scaffold_transition_m = 800.0
    policy = build_final_dem_policy_dict(Cfg(), support_note="ok", guidance_masks={"x": 1})
    assert policy["support_class_codes"]["1"] == "authoritative_locked"
    assert policy["support_class_codes"]["6"] == "low_confidence_continuous_fill"
    assert policy["provenance_class_codes"]["50"] == "river_scaffold_dominant_fill"
    assert policy["provenance_class_codes"]["60"] == "low_confidence_fill"
