# Active Linear Phase 4 – Authoritative and Baseline Source Contracts

Phase 4 locks the active the built-in river workflow source inputs behind explicit source-contract artifacts instead of letting downstream stages re-resolve them from multiple report/config paths.

## Active contract

The active shared-source preparation stage now writes:

- `river_workflow_source_bundle/source_contracts/authoritative_source_contract.json`
- `river_workflow_source_bundle/source_contracts/baseline_source_contract.json`

The authoritative contract owns:

- `solve_authoritative_source_path`
- `export_authoritative_source_path`
- `resolution_source_path`
- `source_kind`

The baseline contract owns:

- `export_baseline_source_path`
- `source_kind`

## Active usage

The active linear authoritative stage consumes these contracts when present and uses them as the canonical source-of-truth for measured-only solve/export source rasters.

## Invariant

Downstream linear stages should consume explicit source-contract artifacts rather than re-discovering authoritative/baseline sources from multiple report/config fallback paths.
