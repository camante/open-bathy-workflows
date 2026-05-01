# River cleanup plan after Bundle A

1. Move implementation internals from `pipeline/river_workflow` to stage-named active modules only after parent/export/final identity tests are stable.
2. Replace report keys named `active_river` with `river_workflow`, retaining temporary compatibility mirrors only while tests and downstream reports are updated.
3. Rename on-disk `river_workflow/` products to `river_workflow/` in a dedicated path migration bundle.
4. Remove or quarantine legacy XS-heavy modules from active imports after static import tests prove they are unused.
5. Replace broad construction exception handlers with stage-specific failure classes in the WSE/diagnostics/audit cleanup bundle.
