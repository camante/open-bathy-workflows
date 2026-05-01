# Legacy Workflows

This document explains how to think about older workflow paths that still exist in the repository.

## Why they still exist

The repository has gone through multiple river workflow generations. Older paths remain for reasons such as:
- regression comparison
- reference behavior
- unfinished migration
- compatibility with earlier experiments or outputs

## Legacy river method families

Examples include:
- `structured`
- `hybrid`
- `xs`
- `skeleton`
- `v1`
- `v2`
- `simple_v2`

Legacy consolidated debug helpers also now live under `legacy/debug/`. Those modules are preserved for comparison and old troubleshooting patterns, but they are not part of the active curated reports path.

These names may still appear in CLI choices, tests, helper modules, and old notes.

## How to treat them

Use the following rule:

- `shared_solve` = built-in active river workflow
- all other river methods = legacy, comparison, or compatibility paths unless explicitly required

## Important caution

Do not read old workflow notes, legacy receipts, or historical patch notes as if they define the active workflow contract.
For the current simple workflow story, start with:
- `README.md`
- `docs/ACTIVE_WORKFLOW.md`
- `docs/RUN_OUTPUTS.md`
