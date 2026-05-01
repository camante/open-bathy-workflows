# Active Linear Phase 6 — Stage Primary Artifact Contracts

This phase tightens the active linear stage chain so every successful stage reports one primary output artifact and one artifact role.

Rules:
- successful stages must have a primary_output
- skipped stages may omit primary_output
- river_stage_chain_summary.json records primary_artifact_name and primary_artifact_role
- the summary includes stages_missing_primary_output for contract verification
