# REVIEW_NOTES_v892 — Bundle C legacy/method containment

## Main invariant

The active river workflow remains: shared solve domain -> canonical parent solution -> exact AOI export -> final DEM materialization. This pass is naming/containment only; it does not intentionally change river science, WSE modeling, gridding, authoritative lock behavior, or final DEM routing.

## Important build note

`workflow_v891.zip` was not present in `/mnt/data` when this bundle was built, so this package was produced from the latest available package, `workflow_v890.zip`, and applies Bundle C changes on top of that base. If the missing v891 stage-contract artifact needs to be preserved exactly, rebase these Bundle C changes onto the v891 zip/package in your local repo.

## What changed

- Removed active exports for old entrypoint names in `active_pipeline.py` and `river_runner.py`.
- Confined `linear_v1` compatibility names to `legacy/river/archive_root_scripts/linear_v1_runner.py`.
- Changed the active dependency/callback names from `resolve_linear_*` / `prepare_linear_*` to `resolve_river_*` / `prepare_river_*` in the bathy-main river workflow entry layer.
- Added `active_river` as the active report section while retaining an `active_river` mirror as a deprecated compatibility report section.
- Added `river_workflow_*` output keys while retaining old `river_workflow_*` output keys as deprecated compatibility keys where downstream tools may still expect them.
- Added static tests to guard that old active entrypoint aliases and old direct-runner callback names do not return to the active path.

## What intentionally remains

- The on-disk work directory is still `river_workflow/`.
- The internal implementation package is still `pipeline/river_workflow/`.
- Some old names remain inside compatibility fields, archived wrappers, and report mirrors.

Those are intentionally deferred because renaming them safely requires a full successful run and output-path migration step. This pass contains old method names rather than attempting the high-risk directory/package rename.

## Next best cleanup

After a successful local `./compare.sh merrimack 892 2`, the next pass should rename the on-disk work directory and pipeline package in a controlled way or add a compatibility symlink/copy period. That should be done only after confirming parent/export/final identity remains exact.
