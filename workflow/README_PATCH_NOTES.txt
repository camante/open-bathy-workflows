Patch summary:
- Build prediction support gate from full retained support and preserve a denser authoritative support cloud.
- Relax prediction gating for dense authoritative extra_xyz runs (broader support distance, min neighbor 1, lower clear-water threshold).
- Preserve user-requested sdb_mode when extra_xyz are present; avoid ocean-only auto override.
- Preserve shallow authoritative points dropped by brightness filter.
- Raise dense-support adaptive sampling floors to 12k/20k/30k instead of tiny targets.

Recommended commit message:
Relax authoritative-anchored SDB prediction gating and preserve dense support coverage
