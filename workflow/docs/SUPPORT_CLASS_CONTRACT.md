# Support Class Contract

Canonical support classes live in `support_classes.py` and are the only allowed source of truth for support codes, names, and family mappings.

## Required rules
- scientific paths must use shared enum/code mappings, not ad hoc strings
- support class family aggregation must come from the shared mapping
- continuous backstop fill must be labeled `low_confidence_continuous_fill`
