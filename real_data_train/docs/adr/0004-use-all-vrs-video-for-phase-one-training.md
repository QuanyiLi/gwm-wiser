---
status: superseded by ADR-0010
---

# Use all VRS video for phase-one training

This historical decision is no longer current; ADR-0010 makes corpus identity provisional pending the dataset signal survey.

Phase one will train from every available VRS train and test clip across all ten embodiments to maximize real-video and embodiment diversity. Whole-robot masks are accepted whether manually annotated or propagated, and VRS is treated as the current corpus rather than a permanent architecture dependency.

## Consequences

There is no held-out VRS split in the phase-one training run. Every valid six-frame window is sampled uniformly without embodiment or clip reweighting, and WISER-dev provides the development comparison instead of VRS test metrics.
