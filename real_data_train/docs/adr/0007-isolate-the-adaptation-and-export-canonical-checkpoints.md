---
status: accepted
---

# Isolate the adaptation and export canonical checkpoints

All robot-video data handling and variable-length training behavior will live under `real_data_train/`, with no edits to existing `gwm_wiser/` interfaces. The training core consumes source-neutral ordered clips of aligned full RGB and robot-only RGB; a source adapter owns dataset layout, split conventions, and derivation of the robot appearance from either an observed whole-robot mask or synchronized robot state and rendering metadata. Training will reuse existing modules where possible and save canonical checkpoints that strict-load into the original fixed-1,620-token `GroundedWorldModel`, allowing the unchanged WISER evaluator to consume them.

## Consequences

Checkpoint conversion is hidden inside training. RobotSeg path conventions remain confined to the VRS adapter, while a state-backed adapter may use joint configurations, URDF geometry, calibrated camera geometry, and robot pose to render robot-only RGB. Those source fields and raw controls are not GWM inputs and do not change the trainer or planner interface.
