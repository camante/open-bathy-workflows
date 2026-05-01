# v854 cleanup notes

This pass removes transition-only WSE scaffolding left over from the Phase-0 contract boundary.

Cleanups included:

- removed inactive legacy WSE repair-path markers from the active WSE stage;
- removed the unused Phase-0 contract-only receipt writer;
- removed obsolete WSE repair/flatness static tests that referred to the retired path;
- added focused static tests for the one-path WSE profile contract and optional-baseline final-folder validation.

The active river construction intent remains:

```
canonical parent solve
-> one monotone bank-edge WSE profile
-> broad canonical modeled offset profile
-> broad canonical bed backbone
-> polygon-bounded river primary surface
-> authoritative lock
-> exact AOI export
-> final review folder
```

No new science branch or fallback path is introduced in this cleanup pass.
