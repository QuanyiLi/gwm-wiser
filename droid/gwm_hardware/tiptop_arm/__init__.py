"""Baseline TiPToP on the rig -- the A/B control arm.

Original TiPToP unchanged: Gemini grounds the instruction, SAM2 segments,
M2T2 proposes grasps, cuTAMP plans, and it re-plans every trial. Nothing here
is imported by `gwm_hardware.gwm_arm`.

- `services.sh`     -- bring the perception servers up, warmed, then tiptop-run
- `inspect_plan.py` -- where a saved plan actually puts the fingers
- `viz_grasp.py`    -- the same question, drawn to scale
"""
