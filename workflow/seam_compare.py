#!/usr/bin/env python
"""Compute seam metrics between two tile rasters or two run manifests.

Explicit-only tool:
  - Provide --raster-a and --raster-b, OR
  - Provide --io-a and --io-b (io_manifest.json paths).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seam_metrics import compute_seam_metrics, load_primary_raster_from_io_manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adjacent-tile seam comparison metrics")
    ap.add_argument("--io-a", type=str, default=None, help="io_manifest.json for run A")
    ap.add_argument(
        "--io-b",
        type=str,
        action="append",
        default=None,
        help="io_manifest.json for run B (may be specified multiple times)",
    )
    ap.add_argument(
        "--io-b-list",
        type=str,
        default=None,
        help="Text file listing io_manifest.json paths (one per line) for run B (batch mode)",
    )
    ap.add_argument("--raster-a", type=str, default=None, help="Explicit raster path for A")
    ap.add_argument("--raster-b", type=str, default=None, help="Explicit raster path for B")
    ap.add_argument("--strip-px", type=int, default=3, help="Strip width in pixels on each side of seam")
    ap.add_argument("--out-json", type=str, default=None, help="Write JSON to this file")
    args = ap.parse_args(argv)

    def _load_io_list(path: str) -> list[str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"--io-b-list not found: {p}")
        out: list[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
        return out

    if args.raster_a and args.raster_b:
        ra = Path(args.raster_a)
        rb = Path(args.raster_b)
        m = compute_seam_metrics(ra, rb, strip_px=int(args.strip_px))
        txt = json.dumps(m, indent=2)
        if args.out_json:
            Path(args.out_json).write_text(txt, encoding="utf-8")
        else:
            print(txt)
        return 0 if m.get("status") == "ok" else 2

    if args.io_a and (args.io_b or args.io_b_list):
        ra = load_primary_raster_from_io_manifest(args.io_a)
        io_bs: list[str] = []
        if args.io_b:
            io_bs.extend(list(args.io_b))
        if args.io_b_list:
            io_bs.extend(_load_io_list(args.io_b_list))
        if not io_bs:
            ap.error("Provide at least one neighbor via --io-b or --io-b-list")

        results = []
        for io_b in io_bs:
            rb = load_primary_raster_from_io_manifest(io_b)
            m = compute_seam_metrics(ra, rb, strip_px=int(args.strip_px))
            m["neighbor_io_manifest"] = str(Path(io_b))
            results.append(m)
        payload = {
            "this_raster": str(ra),
            "strip_px": int(args.strip_px),
            "comparisons": results,
        }
        txt = json.dumps(payload, indent=2)
        if args.out_json:
            Path(args.out_json).write_text(txt, encoding="utf-8")
        else:
            print(txt)
        return 0 if all(r.get("status") == "ok" for r in results) else 2

    ap.error("Provide either --raster-a/--raster-b or --io-a with --io-b/--io-b-list")


if __name__ == "__main__":
    raise SystemExit(main())
