#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bathy_interp_cli.py - convenience CLI for v0.8.0 bathy interpolation.

This wrapper:
1) Runs `bathy_main.py` to generate SDB and/or river rasters
2) Optionally runs `gapfill_intelligent.gapfill_depth_raster()` if an authoritative raster/xyz is supplied.

It is intentionally lightweight and subprocess-based so it works in the zip as-is.

Examples
--------
Standalone SDB+river (no authoritative constraints):
  python bathy_interp_cli.py --aoi -74.5/-74.25/40.25/40.5 --start 2025-01-01 --end 2026-01-01 \
    --methods sdb,river --out-dir output/raritan

Gapfill mode (authoritative raster provided; only fill nodata gaps):
  python bathy_interp_cli.py --aoi ... --start ... --end ... --methods sdb,river --out-dir output/raritan \
    --authoritative-raster authoritative_mean.tif

"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from gapfill_intelligent import gapfill_depth_raster, GapfillConfig

def _run(cmd, verbose=True):
    if verbose:
        print('[RUN]', ' '.join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout is not None
    for line in p.stdout:
        sys.stdout.write(line)
    return int(p.wait())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aoi', required=True)
    ap.add_argument('--start', required=False)
    ap.add_argument('--end', required=False)
    ap.add_argument('--methods', default='sdb')
    ap.add_argument('--priority', default='sdb')
    ap.add_argument('--cloud', type=float, default=70.0)
    ap.add_argument('--icesat', default='all_atl')
    ap.add_argument('--sdb-mode', default='all_sdb')
    ap.add_argument('--river-dem', default=None)
    ap.add_argument('--tnm-enable', action='store_true', default=False)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--authoritative-raster', default=None)
    ap.add_argument('--auth-xyz', default=None)
    ap.add_argument('--auth-decim', type=int, default=2)
    ap.add_argument('--verbose', action='store_true', default=False)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent
    bathy_main = root / 'bathy_main.py'
    if not bathy_main.exists():
        raise SystemExit(f'Could not find bathy_main.py next to this script: {bathy_main}')

    cmd = [sys.executable, str(bathy_main),
           '--aoi', args.aoi,
           '--out-dir', str(out_dir),
           '--methods', args.methods,
           '--priority', args.priority,
           '--cloud', str(args.cloud),
           '--icesat', args.icesat,
           '--sdb-mode', args.sdb_mode]
    if args.start: cmd += ['--start', args.start]
    if args.end: cmd += ['--end', args.end]
    if args.river_dem: cmd += ['--river-dem', args.river_dem]
    if args.tnm_enable: cmd += ['--tnm-enable']

    rc = _run(cmd, verbose=args.verbose)
    if rc != 0:
        raise SystemExit(rc)

    # Standalone outputs are already produced.
    if not args.authoritative_raster:
        return 0

    auth_ras = Path(args.authoritative_raster).resolve()
    if not auth_ras.exists():
        raise SystemExit(f'authoritative raster not found: {auth_ras}')

    # Pick a prior raster from outputs
    sdb_r = next(iter(sorted(out_dir.rglob('*sdb*depth*.tif'))), None)
    river_r = next(iter(sorted(out_dir.rglob('*river*depth*.tif'))), None)
    prior = sdb_r or river_r
    if prior is None:
        raise SystemExit('No prior raster found to gapfill (expected sdb/river depth tif).')

    work = out_dir / '_gapfill'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    hq = []
    if args.auth_xyz:
        hq.append(str(Path(args.auth_xyz).resolve()))
    else:
        from osgeo import gdal
        import numpy as np
        ds = gdal.Open(str(auth_ras))
        b = ds.GetRasterBand(1)
        ndv = b.GetNoDataValue()
        gt = ds.GetGeoTransform()
        nx, ny = ds.RasterXSize, ds.RasterYSize
        def pix2geo(ix, iy):
            x = gt[0] + (ix + 0.5) * gt[1] + (iy + 0.5) * gt[2]
            y = gt[3] + (ix + 0.5) * gt[4] + (iy + 0.5) * gt[5]
            return x, y
        xyz = work / 'authoritative.xyz'
        with open(xyz, 'w') as f:
            for iy in range(0, ny, max(1, args.auth_decim)):
                arr = b.ReadAsArray(0, iy, nx, 1).reshape(-1)
                good = (arr != ndv) if ndv is not None else np.isfinite(arr)
                idx = np.where(good)[0]
                for ix in idx[::max(1, args.auth_decim)]:
                    x, y = pix2geo(int(ix), int(iy))
                    f.write(f'{x} {y} {float(arr[ix])}\n')
        hq.append(str(xyz))

    out_r = out_dir / 'bathy_gapfilled.tif'
    out_s = out_dir / 'bathy_gapfilled_sigma.tif'
    out_p = out_dir / 'bathy_gapfilled_prov.tif'

    cfg_g = GapfillConfig()
    cfg_g.rbf_smoothing = 0.0

    gapfill_depth_raster(
        prior_raster=prior,
        hq_point_files=hq,
        out_raster=out_r,
        out_sigma=out_s,
        out_provenance=out_p,
        cfg=cfg_g
    )

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
