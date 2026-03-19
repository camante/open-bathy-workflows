import bathy_main


def test_bathy_main_has_waffles_wrapper_imports():
    required = [
        "_rm_choose_waffles_mask_for_river",
        "_rm_count_mask_water_pixels",
        "_rm_determine_effective_methods_from_waffles",
        "_rm_find_latest_waffles_mask",
        "_rm_stage_cached_waffles_mask",
        "_rm_waffles_water_fraction",
    ]
    missing = [name for name in required if not hasattr(bathy_main, name)]
    assert not missing, f"Missing imported river_masking aliases: {missing}"
