# Active Linear Phase 9: Seam Identity Utilities

This phase adds lightweight seam-identity utilities and tests so canonical solve reuse can be inspected and validated more directly.

## Added utilities

- `load_canonical_identity(contract_path)`
  - loads a canonical solve contract and records hashes for the canonical solve grid and canonical raster stack
- `compare_touching_horizontal_border(north_raster, south_raster)`
  - compares the south edge of a north raster to the north edge of a south raster

## Purpose

These utilities do not yet run full production AOI seam checks automatically, but they provide a simple and linear contract for:

1. proving two AOIs point to the same canonical solve-state artifacts
2. comparing the actual raster border values where two AOI exports touch

## Intended next use

After north and south AOI reruns, compare:

- canonical solve cache key
- canonical solve grid hash
- canonical measured-only raster hash
- canonical locked solve surface hash
- touching export border differences

That makes seam validation explicit instead of inferred.
