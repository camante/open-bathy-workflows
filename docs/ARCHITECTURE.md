# Architecture (high-level)

This repo contains an end-to-end bathymetry workflow (SDB + river inference + fusion + intelligent gap fill).

Near-term goal:
- Maintain a runnable research workflow in `workflow/`
- Gradually extract stable primitives (gap filling, sampling, river anisotropic interpolation) into a small library that can be upstreamed into CUDEM.

Key outputs:
- SDB depth raster(s)
- River bathy raster(s)
- Combined bathy raster(s) with provenance/uncertainty
