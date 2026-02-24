Changes:
- river_skeleton_bathy.py: widen mainstem preserve corridor (min_corr 250m; multipliers 2.0x / 2.75x), improved corridor logging.
- regression_metrics.py: skip output/logs folders; find channel mask via bathy_report.json cached path if not present in run dir.
- debug.sh: (if included) always cleans river cache + outputs before rerun.

Recommended commit message:
  Widen mainstem preserve corridor; make regression metrics find cached channel masks and skip logs
