#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from guidance_domains import ensure_guidance_domains, make_guidance_domain_cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build shared SDB and river guidance domains early in the workflow.")
    p.add_argument("--aoi", required=True, help="W/E/S/N AOI used for WAFFLES ocean-mask generation.")
    p.add_argument("--cache-root", required=True, help="Shared cache root.")
    p.add_argument("--derived-cache-root", required=True, help="Run-scoped derived cache root where staged outputs should be written.")
    p.add_argument("--river-dem", required=True, help="Template raster used for river/SDB domain grids.")
    p.add_argument("--river-gpkg", required=True, help="River network GeoPackage containing nhdarea_clip.")
    p.add_argument("--waffles-inc-arcsec", type=float, default=1.0)
    p.add_argument("--river-channel-buffer-m", type=float, default=400.0)
    p.add_argument("--river-max-channel-width-m", type=float, default=600.0)
    p.add_argument("--river-mainstem-method", choices=["dominant_trunk", "stream_order"], default="dominant_trunk")
    p.add_argument("--river-mainstem-solve-layer", default="auto", help="Layer used for dominant trunk solving. 'auto' prefers mainstem_solve_network, then rivers_aoi, for AOI-stable trunk selection.")
    p.add_argument("--river-mainstem-min-order", type=int, default=5)
    p.add_argument("--river-max-mainstem-width-m", type=float, default=2500.0)
    p.add_argument("--river-channel-source", default="auto", choices=["auto", "nhdarea", "corridor"])
    p.add_argument("--river-use-nhdarea", action="store_true", default=False)
    p.add_argument("--river-nhdarea-layer", default="nhdarea_clip")
    p.add_argument("--river-nhdarea-allow-ftype", default="460")
    p.add_argument("--river-nhdarea-allow-fcode", default=None)
    p.add_argument("--river-ocean-keep-dist-m", type=float, default=0.0)
    p.add_argument("--estuary-width-ratio-thresh", type=float, default=3.0)
    p.add_argument("--estuary-transition-m", type=float, default=500.0)
    p.add_argument("--estuary-connect-dist-m", type=float, default=200.0)
    p.add_argument("--review-dir", default=None, help="Optional directory where review-ready guidance-domain outputs should be staged. Defaults to <derived-cache-root>/guidance_domains.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = make_guidance_domain_cfg(
        args,
        cache_root=Path(args.cache_root),
        river_channel_source=str(args.river_channel_source),
        river_nhdarea_layer=str(args.river_nhdarea_layer),
        river_nhdarea_allow_ftype=str(args.river_nhdarea_allow_ftype),
        river_nhdarea_allow_fcode=args.river_nhdarea_allow_fcode,
        river_mainstem_solve_layer=str(args.river_mainstem_solve_layer),
    )
    report = {}
    paths = ensure_guidance_domains(
        cfg,
        river_dem=Path(args.river_dem),
        river_gpkg=Path(args.river_gpkg),
        derived_cache_root=Path(args.derived_cache_root),
        review_root=Path(args.review_dir) if args.review_dir else None,
        report=report,
    )
    print(json.dumps({"manifest_json": str(paths.manifest_json), "outputs": paths.as_dict(), "report": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
