---
status: accepted
---

# Define WISER contamination by data provenance

Out-of-domain pretraining must not use WISER RGB, masks, latents, logged trajectories, or rollouts generated from WISER environments and task configurations in its corpus or gradients. Reusing the GWM and Qwen implementation, RAT preprocessing, generic Panda URDF or meshes, and robot-only renderer machinery is allowed because those components define a shared model interface rather than target-domain outcome supervision.

## Consequences

Every candidate corpus and mixture must document its provenance and any shared assets. WISER-dev may still influence development checkpoint inspection or selection under ADR-0008, so phase-one results remain development evidence rather than a held-out estimate.
