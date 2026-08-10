---
status: accepted
---

# Sample by elapsed time when the source has a reliable clock

Timestamped sources construct six-frame RAT windows from configurable elapsed-time offsets, defaulting to the WISER schedule of approximately `[0.00, 0.55, 1.15, 1.75, 2.35, 2.95]` seconds. Sources such as VRS that expose only ordered frames use an explicitly configured ordinal step instead; frame ordinal is never presented as seconds or trajectory progress.

## Consequences

Temporal offsets, ordinal step, window-start stride, and timestamp-matching tolerance are public experiment configuration and are recorded in audits, visualizations, and checkpoints. The phase-one WISER-compatible interface still contains exactly six frames; changing that cardinality is a separate model-interface decision.
