---
status: accepted
---

# Support variable-length native-aspect latents

Source frames will enter the existing Qwen preprocessing pipeline without a dataset-wide resize, which can produce different token lengths across native aspect ratios. A training-only GWM wrapper will therefore generate positions dynamically while retaining the original learned layers instead of forcing every source into the WISER grid.

## Consequences

The learned GWM parameter shapes remain independent of sequence length, while canonical export preserves the original WISER evaluator interface. Token ceilings, fail-fast behavior, and initial batching policy are reversible experiment controls and therefore live in the phase plan rather than this ADR.
