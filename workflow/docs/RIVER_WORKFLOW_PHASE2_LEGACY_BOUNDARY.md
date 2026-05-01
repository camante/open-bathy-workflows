# River Workflow Phase 2 Legacy Boundary

Phase 2 is a cleanup and guard phase only. It does not change river science, DEM construction, cache behavior, AOI export, final materialization, or comparison behavior.

## Goal

Keep the active route simple and traceable:

```text
canonical parent DEM -> AOI export DEM -> final user DEM
```

The active route should have one visible bridge from `bathy_main.py` into the canonical parent/export pipeline. Older river workflow modules can remain in the repository for now, but the active construction path must not depend on them.

## Active entrypoint bridge

`river_runner.py` and `river_runner_contract.py` are active bridge files, not legacy workflow files. They are allowed to:

- receive the active workflow call from `bathy_main.py` / `active_pipeline.py`;
- build the `RiverWorkflowContext`;
- call `pipeline/river_shared_solve/river_pipeline.py`;
- register active products and receipts in the run report.

They are not allowed to:

- create a second river method;
- must not implement an alternate river workflow;
- run legacy v1/v2/XS/surface/backbone paths;
- repair final DEMs;
- choose a different final DEM source.

## Quarantined legacy surface

The hard-quarantine list remains focused on inactive older workflow modules such as:

- `legacy/river/archive_root_scripts/linear_v1_runner.py` and `legacy/river/archive_root_scripts/linear_v1_runner_contract.py` compatibility wrappers;
- older channel/surface/template/backbone/XS/bank modules;
- `legacy/river/` reference code.

The hard-quarantine list intentionally excludes `river_runner.py` and `river_runner_contract.py` because they are active bridge files.

## Testable outcome

The static Phase 2 guard verifies that:

- active entrypoint bridge files exist;
- active bridge files are not listed as hard-quarantine modules or paths;
- compatibility wrappers are still documented separately;
- active construction files and verify/reporting files remain disjoint from hard-quarantine modules.
