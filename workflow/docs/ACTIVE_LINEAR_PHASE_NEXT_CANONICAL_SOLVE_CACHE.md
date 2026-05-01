# Active Linear Next Phase: Canonical Solved Output Reuse

This phase makes the canonical river solve outputs themselves reusable solve-domain artifacts.

## Goal

If two AOIs resolve to the same canonical river solve contract, the workflow should:

1. reuse the same canonical solve grid and solve-domain raster stack
2. reuse the same solved centerline/offset/backbone/corridor/locked-surface artifacts
3. run only export/final stages per AOI

## What is cached

Under the canonical solve outputs root:

- centerline points
- WSE proxy points
- authoritative bed points
- observed offset points
- modeled offset points
- bed backbone points
- corridor solve raster
- primary surface solve raster
- locked primary surface solve raster
- canonical solve cache manifest

## Current behavior after this phase

The inner linear pipeline still rebuilds solve-domain setup stages that are cheap and contract-defining:

- solve domain stage
- grid stage
- authoritative source stage

But if canonical solved outputs already exist, it reuses the cached solve-domain products from centerline through locked-surface and then performs AOI-specific export/final stages.
