import json
from pathlib import Path
from types import SimpleNamespace

from river_guidance import (
    build_river_guidance_manifest,
    guidance_artifact_paths,
    write_river_guidance_manifest,
)


def test_build_and_write_river_guidance_manifest(tmp_path: Path):
    river_dir = tmp_path / "river"
    river_dir.mkdir()
    paths = guidance_artifact_paths(river_dir)
    for key in (
        "guidance_weight",
        "trusted_interior",
        "soft_guidance_domain",
        "admissibility",
        "corridor_mask",
        "authoritative_support",
        "authoritative_support_depth",
        "depth_terrain",
        "bottom_elevation",
        "scaffold_domains",
        "scaffold_manifest",
        "guide_points",
    ):
        p = paths[key]
        if p.suffix in {".json", ".gpkg"}:
            p.write_text("{}", encoding="utf-8")
        else:
            p.write_bytes(b"x")

    report = {
        "river": {
            "outputs": {
                "guidance_weight": str(paths["guidance_weight"]),
                "trusted_interior": str(paths["trusted_interior"]),
                "soft_guidance_domain": str(paths["soft_guidance_domain"]),
                "admissibility": str(paths["admissibility"]),
                "guide_points": str(paths["guide_points"]),
                "depth_terrain": str(paths["depth_terrain"]),
                "bottom_elevation": str(paths["bottom_elevation"]),
            },
            "guidance": {
                "trusted_interior_definition": "trusted export interior",
                "soft_guidance_definition": "soft guidance domain",
                "admissibility_definition": "anchor-excluded admissible guidance",
            },
        }
    }
    manifest = build_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    assert manifest["guidance_only"] is True
    assert manifest["artifact_roles"]["depth_terrain"] == "diagnostic_only"
    assert manifest["artifact_roles"]["guide_points"] == "sparse_guidance_points"
    assert manifest["notes"]["trusted_interior"] == "trusted export interior"

    manifest_path = write_river_guidance_manifest(out_root=tmp_path, river_dir=river_dir, report=report)
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["artifacts"]["guide_points"].endswith("river_guide_points.gpkg")

