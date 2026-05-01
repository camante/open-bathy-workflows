# Final Output Folder Contract

The active river workflow writes the scientific AOI DEM from the canonical parent
solution first, then builds a `final/` review package around that already-written
DEM. The final-folder validator must not invalidate a successful canonical AOI
export because optional comparison products are unavailable.

Required when final output receipt is written:

- `final/DEM_enhanced.tif`
- `final/DEM_enhanced_hillshade.tif`
- `final/final_output_receipt.json`
- canonical parent DEM and hillshade when the canonical parent source is recorded
  in the receipt
- authoritative/base comparison DEM and hillshade only when the receipt records a
  resolved baseline source

When no baseline source is resolved, the receipt records a skip reason and the
active validator must not require `authoritative_base_aligned.tif` or its
hillshade. This keeps the final review package aligned with the canonical parent
architecture: the required deliverable is the AOI subset of the canonical parent;
the baseline comparison is useful but optional.
