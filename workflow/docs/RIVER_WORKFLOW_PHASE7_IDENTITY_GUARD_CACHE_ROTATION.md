# River workflow Phase 7: canonical identity guard cache-version rotation

## Goal

Keep the canonical river identity guard strict about source/grid drift, but allow intentional canonical cache-key rotation when the workflow contract version changes for a science-stage rebuild.

## Problem fixed

The v883 run correctly produced a new canonical cache key after the backbone downstream-rise science change, but the canonical identity guard rejected the rebuild because the path-free canonical identity matched the previous run while the cache key changed.

That was too strict for contract-version changes. The same river system, solve AOI, CRS, resolution, and materialized authoritative inputs can legitimately receive a new cache key when the workflow contract version changes to force a canonical parent rebuild.

## Behavior after this phase

For the same canonical identity:

- If the cache key is unchanged and materialized canonical input hashes match, the guard reports `matched_existing`.
- If the cache key changes and materialized canonical input hashes still match, the guard reports `cache_key_rotated_for_workflow_contract` and updates the registry to the new key.
- If materialized canonical input hashes change, the guard still fails with `river_workflow_canonical_identity_drift`.

This keeps the guard aligned with the simple river architecture: it protects against accidental source/grid drift but does not block intentional contract-versioned science rebuilds.
