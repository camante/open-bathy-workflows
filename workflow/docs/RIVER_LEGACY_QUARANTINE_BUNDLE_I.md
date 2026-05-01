# River legacy quarantine — Bundle I

The active river workflow is now treated as one built-in canonical workflow, not a method selected from historical route names.

Active runtime rules:

1. `--river-method` is not an active CLI option.
2. `river_method` is not an active config section or dataclass field.
3. Historical names such as `linear_v1`, `simple_v2`, and old skeleton/XS route selectors are legacy-only references.
4. Active river construction must import from `pipeline/river_workflow`, not `pipeline/river_linear` or `legacy/river`.
5. Legacy files may remain in `legacy/river` for reference, but they must not be imported by the active workflow.

This bundle intentionally does not delete all old science files. It tightens the active boundary so later deletion is safe and auditable.
