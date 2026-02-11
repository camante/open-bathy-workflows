#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""waffles_bathy_interp.py - Waffles module wrapper for the v0.8.0 bathymetry workflow.

Goals
- Works inside CUDEM waffles as an *auxiliary* module (stack=False) so it can:
  * run standalone (no authoritative datalist) by fetching S2/ICESat/USGS/NHD as needed, OR
  * run as an intelligent gap-filler when waffles is given a datalist (authoritative stack exists).
- Produces a DEM that strictly agrees with authoritative stack where present, only filling gaps.

Integration notes
- In CUDEM, copy this file to: cudem/waffles/bathy_interp.py
- Add `from . import bathy_interp` to cudem/waffles/__init__.py
- Add WaffleFactory entry: 'bathy-interp': {'name': 'bathy_interp', 'stack': False, 'call': bathy_interp.WafflesBathyInterp}
  and import bathy_interp in WaffleFactory.

This module intentionally uses subprocess calls into the standalone pipeline scripts that ship with v0.8.0,
to avoid duplicating orchestration logic in waffles.

Author: CUDEM / CIRES SDB+River workflow team
Version: 0.8.0
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

# CUDEM / waffles imports (only available when used inside CUDEM)
from osgeo import gdal  # type: ignore
from cudem import utils  # type: ignore
from cudem import gdalfun  # type: ignore
from cudem.waffles.waffles import Waffle  # type: ignore

# Local (v0.8.0) intelligent gapfill
try:
    from gapfill_intelligent import gapfill_depth_raster, GapfillConfig
except Exception:  # pragma: no cover
    gapfill_depth_raster = None


def _run_cmd(cmd: List[str], cwd: Optional[str] = None, verbose: bool = True) -> int:
    """Run a subprocess, streaming output."""
    if verbose:
        utils.echo_msg('[BATHY-INTERP][RUN] ' + ' '.join(cmd))
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout is not None
    for line in p.stdout:
        if verbose:
            sys.stdout.write(line)
    return int(p.wait())


def _find_first_raster(out_dir: Path, patterns: Tuple[str, ...]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(out_dir.rglob(pat))
        if hits:
            return hits[0]
    return None


def _stack_to_authoritative_dem(stack_fn: str, out_fn: str, ndv: float = -9999.0) -> str:
    """Convert waffles stack (z/weight/count in bands) to an authoritative mean surface in band1."""
    ds = gdal.Open(stack_fn)
    if ds is None:
        raise RuntimeError(f'Could not open stack: {stack_fn}')
    infos = gdalfun.gdal_infos(ds, scan=False)
    # Expect: band1=z_sum, band2=count, band3=weight_sum (based on waffles stacking conventions)
    b_z = ds.GetRasterBand(1)
    b_count = ds.GetRasterBand(2) if ds.RasterCount >= 2 else None
    b_weight = ds.GetRasterBand(3) if ds.RasterCount >= 3 else None

    drv = gdal.GetDriverByName('GTiff')
    out = drv.Create(out_fn, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32,
                     options=['COMPRESS=DEFLATE', 'TILED=YES'])
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    ob = out.GetRasterBand(1)
    ob.SetNoDataValue(ndv)

    block_x, block_y = b_z.GetBlockSize()
    if block_x == 0 or block_y == 0:
        block_x, block_y = 512, 512

    for yoff in range(0, ds.RasterYSize, block_y):
        ysize = min(block_y, ds.RasterYSize - yoff)
        for xoff in range(0, ds.RasterXSize, block_x):
            xsize = min(block_x, ds.RasterXSize - xoff)
            z = b_z.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float64)
            if b_count is not None:
                c = b_count.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float64)
            else:
                c = np.ones_like(z)
            if b_weight is not None:
                w = b_weight.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float64)
            else:
                w = np.ones_like(z)

            with np.errstate(divide='ignore', invalid='ignore'):
                m = (z / w) / c
            m[~np.isfinite(m)] = ndv
            ob.WriteArray(m.astype(np.float32), xoff, yoff)

    out.FlushCache()
    out = None
    ds = None
    return out_fn


def _authoritative_xyz_from_dem(dem_fn: str, xyz_fn: str, decim: int = 1) -> str:
    """Write an XYZ point file from valid raster cells (pixel centers)."""
    ds = gdal.Open(dem_fn)
    if ds is None:
        raise RuntimeError(f'Could not open dem: {dem_fn}')
    b = ds.GetRasterBand(1)
    ndv = b.GetNoDataValue()
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize

    def pix2geo(ix: int, iy: int) -> Tuple[float, float]:
        x = gt[0] + (ix + 0.5) * gt[1] + (iy + 0.5) * gt[2]
        y = gt[3] + (ix + 0.5) * gt[4] + (iy + 0.5) * gt[5]
        return x, y

    with open(xyz_fn, 'w', encoding='utf-8') as f:
        for iy in range(0, ny, max(1, decim)):
            arr = b.ReadAsArray(0, iy, nx, 1).reshape(-1)
            if ndv is not None:
                good = arr != ndv
            else:
                good = np.isfinite(arr)
            idx = np.where(good)[0]
            for ix in idx[::max(1, decim)]:
                x, y = pix2geo(int(ix), int(iy))
                f.write(f'{x} {y} {float(arr[ix])}\n')

    ds = None
    return xyz_fn
def _authoritative_xyz_from_dem_stratified(
    dem_fn: str,
    xyz_fn: str,
    target_points: int = 5000,
    pre_decim: int = 2,
    seed: int = 42,
) -> str:
    """Sample an authoritative raster into an XYZ using the same adaptive stratified sampler
    used for extra XYZ training (depth bins + spatial coverage).
    This is a lightweight wrapper around spatial_sampling.adaptive_spatial_sample.
    """
    import numpy as np
    import pandas as pd
    from osgeo import gdal  # type: ignore

    try:
        from spatial_sampling import adaptive_spatial_sample, SamplingConfig
    except Exception:
        # Fallback to uniform decimation
        return _authoritative_xyz_from_dem(dem_fn, xyz_fn, decim=max(1, pre_decim))

    ds = gdal.Open(dem_fn)
    if ds is None:
        raise RuntimeError(f'Could not open dem: {dem_fn}')
    b = ds.GetRasterBand(1)
    ndv = b.GetNoDataValue()
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize

    # Pre-sample on a grid to avoid reading every pixel for large rasters
    xs = []
    ys = []
    zs = []
    for iy in range(0, ny, max(1, pre_decim)):
        arr = b.ReadAsArray(0, iy, nx, 1).reshape(-1)
        if ndv is not None:
            good = arr != ndv
        else:
            good = np.isfinite(arr)
        idx = np.where(good)[0]
        if idx.size == 0:
            continue
        # decimate within row
        idx = idx[::max(1, pre_decim)]
        # pixel center coords
        x = gt[0] + (idx + 0.5) * gt[1] + (iy + 0.5) * gt[2]
        y = gt[3] + (idx + 0.5) * gt[4] + (iy + 0.5) * gt[5]
        xs.append(x.astype(np.float64))
        ys.append(y.astype(np.float64))
        zs.append(arr[idx].astype(np.float64))
    ds = None

    if not xs:
        # No valid pixels
        Path(xyz_fn).write_text('', encoding='utf-8')
        return xyz_fn

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)

    # Build df expected by sampler
    df = pd.DataFrame({
        'longitude': x,
        'latitude': y,
        'depth_m': z,
        'source': 'measured',
        'sample_weight': 1.0,
    })

    # Hard cap to keep sampler fast
    rng = np.random.default_rng(int(seed))
    if len(df) > 200000:
        df = df.sample(n=200000, random_state=int(seed))

    scfg = SamplingConfig(target_total_points=int(max(200, target_points)))
    sampled, stats = adaptive_spatial_sample(df, sampling_config=scfg)

    with open(xyz_fn, 'w', encoding='utf-8') as f:
        for r in sampled.itertuples(index=False):
            f.write(f'{float(r.longitude)} {float(r.latitude)} {float(r.depth_m)}\n')
    return xyz_fn



class WafflesBathyInterp(Waffle):
    """Bathy interpolation / gapfill module.

    Usage (inside waffles):
      waffles -R... -E... -M bathy-interp:methods=sdb,river:start=2025-01-01:end=2026-01-01:cloud=70 ...

    Behavior:
      - If a datalist was provided, waffles will build a stack. We treat that stack as authoritative.
        We generate SDB/river priors, then apply physics-informed residual gapfilling to fill ONLY gaps.
      - If no datalist was provided, we operate standalone and just run the v0.8.0 pipeline to
        generate SDB and/or river bathy outputs for the region.

    Notes:
      This wrapper assumes the v0.8.0 scripts are adjacent on disk (same directory as this module)
      when used standalone (e.g., from the zip). In CUDEM integration, you can either vendor those
      scripts as a subpackage or call out to an installed "bathy_main.py" entrypoint.
    """

    def __init__(
        self,
        methods: str = 'sdb',
        priority: str = 'sdb',
        start: Optional[str] = None,
        end: Optional[str] = None,
        cloud: float = 70.0,
        icesat: str = 'all_atl',
        sdb_mode: str = 'all_sdb',
        river_dem: Optional[str] = None,
        tnm_enable: bool = True,
        gapfill: bool = True,
        auth_decim: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.methods = methods
        self.priority = priority
        self.start = start
        self.end = end
        self.cloud = utils.float_or(cloud)
        self.icesat = icesat
        self.sdb_mode = sdb_mode
        self.river_dem = river_dem
        self.tnm_enable = bool(tnm_enable)
        self.gapfill = bool(gapfill)
        self.auth_decim = max(1, utils.int_or(auth_decim, 2))

    def _pipeline_root(self) -> Path:
        # when vendored into CUDEM, keep these scripts alongside the module or adjust this path.
        return Path(__file__).resolve().parent

    def _run_standalone_pipeline(self, out_dir: Path, extra_xyz: Optional[str] = None, extra_xyz_crs: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path]]:
        """Run bathy_main.py to generate prior(s). Returns (sdb_raster, river_raster)."""
        root = self._pipeline_root()
        script = root / 'bathy_main.py'
        if not script.exists():
            raise RuntimeError(f'Expected bathy_main.py next to module, not found: {script}')

        aoi = self.region.format('str')  # waffles Region string
        cmd = [sys.executable, str(script),
               '--aoi', aoi,
               '--out-dir', str(out_dir),
               '--methods', self.methods,
               '--priority', self.priority]

        if self.start:
            cmd += ['--start', self.start]
        if self.end:
            cmd += ['--end', self.end]
        if self.cloud is not None:
            cmd += ['--cloud', str(self.cloud)]
        if self.icesat:
            cmd += ['--icesat', self.icesat]
        if self.sdb_mode:
            cmd += ['--sdb-mode', self.sdb_mode]
        if self.river_dem:
            cmd += ['--river-dem', str(self.river_dem)]
        if self.tnm_enable:
            cmd += ['--tnm-enable']
        if extra_xyz:
            cmd += ['--extra-xyz', str(extra_xyz)]
            if extra_xyz_crs:
                cmd += ['--extra-xyz-crs', str(extra_xyz_crs)]

        rc = _run_cmd(cmd, cwd=str(root), verbose=self.verbose)
        if rc != 0:
            raise RuntimeError(f'bathy_main.py failed with code {rc}')

        sdb_r = _find_first_raster(out_dir, ('**/*sdb*depth*.tif', '**/*sdb*depth*.vrt', '**/*sdb*.tif'))
        river_r = _find_first_raster(out_dir, ('**/*river*depth*.tif', '**/*river*.tif'))
        return sdb_r, river_r

    def run(self):
        work = Path(self.cache_dir or os.getcwd()) / 'bathy_interp_work' / self.name
        work.mkdir(parents=True, exist_ok=True)

        # --- Case A: waffles has an authoritative stack (datalist provided) ---
        if getattr(self, 'stack', None) and os.path.exists(self.stack) and self.gapfill:
            if gapfill_depth_raster is None:
                raise RuntimeError('gapfill_intelligent not available in this environment.')

            auth_dem = work / 'authoritative_mean.tif'
            _stack_to_authoritative_dem(self.stack, str(auth_dem), ndv=float(self.ndv))

            auth_xyz = work / 'authoritative.xyz'
            _authoritative_xyz_from_dem_stratified(str(auth_dem), str(auth_xyz), target_points=8000, pre_decim=self.auth_decim)

            # Use authoritative mean as river DEM for bank elevations if user did not provide one.
            if not self.river_dem:
                self.river_dem = str(auth_dem)

            prior_dir = work / 'priors'
            if prior_dir.exists():
                shutil.rmtree(prior_dir)
            prior_dir.mkdir(parents=True, exist_ok=True)

            # Run pipeline to generate prior rasters, using authoritative xyz as extra training.
            # IMPORTANT: This makes the prior better aligned with local conditions without violating
            # the constraint that authoritative values must be preserved (we clamp later).
            sdb_r, river_r = self._run_standalone_pipeline(
                prior_dir,
                extra_xyz=str(auth_xyz),
                extra_xyz_crs=self.dst_srs if self.dst_srs else None
            )

            # Prefer the pipeline's combined prior if available.
            combined_prior = (prior_dir / 'combined' / 'bathy_combined_depth.tif')
            prior = combined_prior if combined_prior.exists() else (sdb_r or river_r)
            if prior is None:
                raise RuntimeError('No prior rasters generated by standalone pipeline.')

            out_gap = work / f'{self.name}_gapfilled.tif'
            out_sigma = work / f'{self.name}_sigma.tif'
            out_prov = work / f'{self.name}_prov.tif'
            cfg = GapfillConfig()
            # keep smoothing low to respect local variations; exactness is enforced by the final clamp below
            cfg.rbf_smoothing = 0.0
            gapfill_depth_raster(
                prior_raster=Path(prior),
                hq_point_files=[str(auth_xyz)],
                out_raster=Path(out_gap),
                out_sigma=Path(out_sigma),
                out_provenance=Path(out_prov),
                cfg=cfg,
                logger=None
            )

            # Final clamp: ensure *strict* equality at authoritative cells.
            # (gapfill_depth_raster already does enforce_exact, but we keep this as a last guard.)
            with gdalfun.gdal_datasource(str(auth_dem)) as a_ds, gdalfun.gdal_datasource(str(out_gap), update=True) as o_ds:
                a_b = a_ds.GetRasterBand(1)
                o_b = o_ds.GetRasterBand(1)
                a_ndv = a_b.GetNoDataValue()
                o_ndv = o_b.GetNoDataValue()
                nx, ny = a_ds.RasterXSize, a_ds.RasterYSize
                block_x, block_y = a_b.GetBlockSize()
                if block_x == 0 or block_y == 0:
                    block_x, block_y = 512, 512

                for yoff in range(0, ny, block_y):
                    ysize = min(block_y, ny - yoff)
                    for xoff in range(0, nx, block_x):
                        xsize = min(block_x, nx - xoff)
                        a = a_b.ReadAsArray(xoff, yoff, xsize, ysize)
                        o = o_b.ReadAsArray(xoff, yoff, xsize, ysize)
                        if a_ndv is not None:
                            mask = a != a_ndv
                        else:
                            mask = np.isfinite(a)
                        o[mask] = a[mask]
                        o_b.WriteArray(o, xoff, yoff)

            self.fn = str(out_gap)
            if self.verbose:
                utils.echo_msg(f'[BATHY-INTERP] Gapfilled DEM -> {self.fn}')
            return self

        # --- Case B: standalone output generation (no authoritative stack) ---
        out_dir = work / 'standalone'
        out_dir.mkdir(parents=True, exist_ok=True)
        sdb_r, river_r = self._run_standalone_pipeline(out_dir)

        # Prefer combined output when both methods were requested.
        combined = out_dir / 'combined' / 'bathy_combined_depth.tif'
        chosen = combined if combined.exists() else None

        # Choose primary output based on methods/priority
        if chosen is None:
            chosen = None
        if 'sdb' in self.methods and sdb_r is not None:
            chosen = sdb_r
        if chosen is None and river_r is not None:
            chosen = river_r
        if chosen is None:
            raise RuntimeError('Standalone pipeline produced no rasters.')

        # Copy/warp into waffles expected output name/grid if needed.
        # For now, we just point waffles to the produced raster.
        self.fn = str(chosen)
        if self.verbose:
            utils.echo_msg(f'[BATHY-INTERP] Standalone raster -> {self.fn}')
        return self