# Open Bathy Workflows — Plain-Language Guide

> **Authoritative outputs:** use `io_manifest.json`, `io_manifest.md`, and `unified_bathy_report.json` from the run you just made.
> Older notes may mention filenames that are no longer guaranteed.

This workflow makes bathymetry in two different water settings and can merge them into one final result.

## The three main parts

### Coastal / nearshore bathymetry

The workflow uses Sentinel-2 imagery, ICESat-2 data, and optional external soundings to estimate depth in coastal water.

### River bathymetry

The workflow uses river hydrography, a DEM, masks, and optional soundings or hydraulic constraints to estimate the river bed.

### Fusion

If both coastal and river products are available, the workflow combines them into one final raster.

## The most important safety idea

The code keeps separate ideas of where bathymetry is allowed:

- a **coastal water domain**
- a **river/channel domain**

At the end of the run, the workflow clips deliverables so that depths do not remain outside the intended water area.
That final clipping is a safety net, not a replacement for correct upstream masking.

## What “hybrid river” means now

The current default river mode is `hybrid`.
That means the workflow does not try to force cross-sections everywhere.
Instead, it uses cross-section inference where it is most useful on the mainstem and uses the skeleton method to stabilize the rest of the river network.

## What SWOT does here

SWOT RiverSP observations are used to help stabilize water-surface elevation and slope where requested.
They are not treated as a direct bathymetry raster.

## What files should I inspect after a run?

Start here:

- `io_manifest.json` to see the exact files that went in and came out
- `bathy_report.json` to see which stages ran and whether they succeeded
- `unified_bathy_report.json` for a simpler whole-run summary
- `run_logs/` for human-readable and technical summaries

## One example command

```bash
python bathy_main.py \
  --aoi="-71.14/-71.10/42.75/42.78" \
  --start=2025-01-01 \
  --end=2026-01-01 \
  --methods=sdb,river,fuse \
  --out-dir=output/example_run \
  --cache-root=cache
```

## Common confusion points

### “Why are the filenames different from an older run?”

The workflow now treats manifests and reports as authoritative.
Do not assume an older canonical filename still applies.

### “Why is there river depth only inside the channel mask?”

That is intentional.
River deliverables are clipped to the river/channel domain.

### “Why does the combined raster not extend everywhere both methods had values?”

Fusion still respects domain policies and source priority.
The goal is not to keep every pixel at all costs, but to keep the final product physically and operationally defensible.
