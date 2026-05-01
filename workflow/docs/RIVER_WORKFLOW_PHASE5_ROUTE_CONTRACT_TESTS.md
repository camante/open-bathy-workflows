# River workflow Phase 5: route contract tests

Phase 5 locks the active river route before the same framework is copied to SDB.

The contract is:

```text
canonical_parent_dem -> aoi_export_dem -> final_user_dem
```

The tests are intentionally read-only. They check retained manifests and receipts;
they do not rerun WSE, offset, backbone, surface, lock, or export construction.

Required behavior:

1. If a valid cached canonical parent manifest and parent DEM exist, the AOI run
   must export only from that parent.
2. If an AOI-export-only run has no canonical manifest, it must fail clearly.
3. If the manifest hash and parent DEM hash disagree, it must fail clearly with a
   stale-parent/hash-mismatch error.
4. The AOI export receipt must prove the export is an exact parent window and
   that no AOI construction stages ran.
5. The run summary may read parent science receipts and `canonical_science_summary`,
   but it must not recompute parent construction science during AOI export.

This is the river-side template for the later SDB cleanup: build one parent
science/guidance product, export AOI subsets from the parent, materialize the
final product from the named export artifact, and keep receipts proving the chain.
