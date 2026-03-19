# Authoritative Support Flow

The workflow is authoritative-first. `authoritative_base` is not only a final conditioning lock; it also participates upstream as structured support.

## Upstream uses
- **SDB guidance support**: raster-derived points exported from `authoritative_base` and passed as disciplined internal `extra_xyz`-style guidance.
- **River guidance support**: raster-derived points exported from `authoritative_base` and used as river sounding-style support.
- **River land/bank DEM**: projected `authoritative_base` is preferred for river-side terrain; TNM/USGS may only fill uncovered gaps.

## Final conditioning use
- finite authoritative cells are hard-locked
- only authoritative nodata gaps are eligible for conditioned fill
- subordinate guidance is allowed only in eligible gaps

## Audit expectations
Every authoritative-conditioned run should emit a precedence audit proving:
- authoritative locks were preserved
- guidance did not modify locked cells
- continuous fill class usage is explicit
