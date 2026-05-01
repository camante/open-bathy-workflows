# Canonical River Solve and Export Contract

The active `shared-solve` workflow must be read as two classes of stages:

1. **solve** stages
   - resolve shared/canonical river solve inputs
   - prepare the canonical river solve contract
   - run the river solve on the canonical domain
2. **export** stages
   - warp/export AOI deliverables from solved outputs
   - register deliverables and write final output receipt
3. **finalize** stages
   - shared postrun reporting and manifest generation

Export stages must not rebuild solve-domain artifacts such as the solve network, solve grid, authoritative measured-only solve raster, or centerline/backbone/primary-surface solve products.

The canonical river solve contract is the single artifact that records the reusable solve-domain identity for a river system and its source contracts.
