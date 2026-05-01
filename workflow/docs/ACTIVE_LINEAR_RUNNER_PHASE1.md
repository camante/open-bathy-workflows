# Active Linear Runner Phase 1 Contract

Phase 1 narrows the active `shared_solve` runner contract.

## Invariant

The direct `shared_solve` runner should:
1. run the linear pipeline,
2. return an explicit `RiverRunnerResult`, and
3. localize report registration through `register_shared_solve_runner_result(...)`.

The runner result carries:
- primary artifacts,
- registration payloads for `report`, and
- the underlying linear pipeline result for active stage-chain expansion.

## Transition rule

The active workflow may still expose the underlying linear pipeline fields through the runner result during transition, but broad ad hoc `report` mutation should remain localized in the registration helper rather than being scattered across the runner body.
