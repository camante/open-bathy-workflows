# v0.8.0 → CUDEM Waffles integration (bathy-interp)

This zip contains the mature standalone SDB+river workflow **and** a waffles-ready module wrapper:

- `waffles_bathy_interp.py`  → copy to `cudem/waffles/bathy_interp.py` in the CUDEM repo
- `gapfill_intelligent.py`   → provides the core *prior + residual* gap-filling engine

The goal is to support **two modes** with one implementation:

## 1) Waffles mode (intelligent gap fill)
When the user runs waffles with an *authoritative* datalist (lidar DEM, sonar, etc.), waffles produces a raster stack.
The `bathy-interp` module treats that stack as authoritative and will:

1. Convert waffles stack → an authoritative mean surface (nodata where no authoritative data exists)
2. Run the v0.8.0 SDB/river workflow to generate a **prior** bathy surface (optionally trained with those authoritative points)
3. Run `gapfill_intelligent.gapfill_depth_raster()` to fill gaps using the prior + residual framework
4. **Clamp** the final DEM so that wherever authoritative values exist, they are exactly preserved
   (i.e., only gaps are filled).

Recommended waffles usage pattern:

```bash
waffles \
  -R-74.5/-74.25/40.25/40.5 \
  -E0.0000925926 \
  -M bathy-interp:methods=sdb,river:start=2025-01-01:end=2026-01-01:cloud=70 \
  my_authoritative.datalist
```

### CUDEM code edits (one-time)
In CUDEM:

1. Add module file:
   - `cudem/waffles/bathy_interp.py` (copy from this zip's `waffles_bathy_interp.py`)
2. Update `cudem/waffles/__init__.py` to import it (pattern matches existing imports, e.g. sdb)
3. Update `cudem/waffles/waffles.py`:
   - import `bathy_interp` in the `WaffleFactory` block
   - add a factory entry:

```python
'bathy-interp': {'name': 'bathy_interp', 'stack': False, 'call': bathy_interp.WafflesBathyInterp},
```

> `stack=False` is important: it lets waffles run this module **with or without** a datalist.
> If a datalist is present, waffles will still build a stack (see waffles CLI logic).

## 2) Standalone mode (no authoritative data)
If the user has **no** authoritative datalist/stack, the module runs the v0.8.0 pipeline in standalone mode,
fetching its own inputs (Sentinel-2, ICESat-2 ATL, USGS/NHD/TNM resources depending on config),
and returns the produced SDB and/or river raster.

This is the mode you want for “just give me SDB and/or river bathy outputs for this AOI + time”.

---

## Notes / assumptions
- This wrapper calls the v0.8.0 pipeline via subprocess to avoid duplicating orchestration logic in waffles.
  If you prefer a pure-library implementation, the next step is to refactor `bathy_main.py` to expose a function API.
- The “strict agreement with authoritative data” requirement is enforced twice:
  1) by using the authoritative points as HQ constraints to the interpolator, and
  2) by an explicit post-pass clamp that overwrites output cells wherever authoritative data exists.
