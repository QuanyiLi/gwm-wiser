---
status: superseded by ADR-0018
---

# Treat WISER as development evaluation

This historical decision is no longer current; ADR-0018 retires WISER from the project entirely — there is no WISER evaluation of any kind.

Phase one may repeatedly inspect WISER `merged_test` MSE/cosine and run the existing closed-loop WISER evaluation to compare selected-corpus-trained checkpoints with the established setting. Because that feedback can influence development, it is named WISER-dev and is not presented as a held-out or zero-shot estimate.

## Consequences

Results for task instances originally assigned to both WISER splits and the relevant secondary metrics are reported under their original labels, but neither is described as an untouched test estimate once used for development. A later hardware checkpoint must be selected independently of WISER-dev performance; the numeric development target remains in the phase plan.
