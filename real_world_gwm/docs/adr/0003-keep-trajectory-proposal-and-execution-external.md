---
status: accepted
---

# Keep trajectory proposal and execution external

Trajectory generation, grasp-pose synthesis, rendering from robot state, geometric feasibility, collision checking, low-level control, and safety will remain external to GWM. This keeps the semantic scorer replaceable and allows mature proposal methods to supply motions beyond those observed during GWM training without attributing their physical competence to GWM.

## Consequences

The integration contract for a candidate is an executable trajectory plus its rendered future robot-only RGB sequence. During WISER-dev evaluation, the unchanged planner may use logged WISER trajectories as external proposals, but those trajectories never enter GWM pretraining. A later real system may use a different proposal module without retraining the semantic interface solely because the proposal method changed.
