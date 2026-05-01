from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compare_script_accepts_aoi_count() -> None:
    text = (ROOT / "compare.sh").read_text(encoding="utf-8")
    assert "--aoi-count <2|3|4|5>" in text
    assert "AOI_COUNT=\"5\"" in text
    assert "AOI_LABELS_ALL=(north south contained downstream_overlap upstream_overlap)" in text
    assert "AOI_LABELS=(\"${AOI_LABELS_ALL[@]:0:${AOI_COUNT}}\")" in text
    assert "union(selected AOIs)" in text


def test_compare_script_variants_are_kept_in_sync() -> None:
    compare = (ROOT / "compare.sh").read_text(encoding="utf-8")
    final_compare = (ROOT / "final_compare.sh").read_text(encoding="utf-8")
    named_compare = (ROOT / "compare_5aois_union_final_products_parent_ref_v12.sh").read_text(encoding="utf-8")
    required = "AOI comparison count: ${AOI_COUNT}"
    for text in (compare, final_compare, named_compare):
        assert required in text
        assert "--num-aois" in text
        assert "${SELECTED_AOIS[@]}" in text
