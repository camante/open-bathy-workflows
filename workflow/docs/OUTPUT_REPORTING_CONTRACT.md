Output Reporting Contract
=========================

Purpose
-------
The final workflow must emit a continuous final raster together with explicit support, provenance, and confidence reporting so weakly supported terrain is not mistaken for measurement-like truth.

Required outputs
----------------
- final DEM
- final provenance raster
- support class raster
- support/provenance summary JSON
- confidence and guidance influence summaries when available

Required semantics
------------------
- support class reporting must summarize area/count by class and class family
- provenance reporting must summarize area/count by class and class family
- guidance influence reporting must describe how much of the final domain received nonzero subordinate guidance
- confidence summaries must report SDB and river confidence coverage when those artifacts exist

Interpretation goal
-------------------
A user should be able to distinguish authoritative-locked terrain, anchored interpolation, guidance-conditioned terrain, scaffold-inferred terrain, and low-confidence continuous fill from the final reporting package alone.
