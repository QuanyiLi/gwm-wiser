"""GWM x TiPToP on the rig -- the method under test.

Anonymous point-cloud clusters -> 12-16 executable pick trajectories
(cuTAMP/cuRobo) -> GWM scores each against the verbatim instruction -> argmax
executes. No Gemini, no SAM2. The droid-sim implementation of the same method
is `droid/gwm_tiptop/`, which this arm CALLS but does not modify -- the sim
results stay reproducible from an unchanged tree.

Split of responsibility:

    droid/gwm_tiptop/   the METHOD (perception, proposals, scoring client,
                        grasp gate) -- shared with droid-sim, unmodified
    gwm_hardware/gwm_arm/  the RIG PLUMBING the method needs on real hardware
                        and does not need in sim: live capture, the external
                        camera's extrinsics, the renderer overlay gate, the
                        run driver, and the debug viewer

- `capture.py`       -- wrist + external observation -> the h5 files the
                        gwm_tiptop drivers read (live, or replayed from a
                        saved `tiptop_outputs/eval/<ts>` run)
- `extcam_calib.py`  -- external-camera extrinsics in the robot base frame
- `overlay_gate.py`  -- the GI-2 renderer/photo alignment gate, on real pixels
- `run_real.py`      -- capture -> propose -> score -> gate -> execute
- `viz_debug.py`     -- the Rerun debug view: clusters, every candidate
                        trajectory coloured by its GWM score, the winner
"""
