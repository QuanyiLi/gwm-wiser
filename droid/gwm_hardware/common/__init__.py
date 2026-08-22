"""The rig itself: robot model, calibration, cameras, tiptop-tree installers.

Owned by neither experiment arm. Both `gwm_hardware.tiptop_arm` and
`gwm_hardware.gwm_arm` import from here; nothing here imports from either.

- `paths.py`        -- where the shared `assets/` and `config/` live
- `build_2f140.py` / `build_2f140_cfg.py` -- generate the Panda + 2F-140 URDF
  and cuRobo config that this rig needs and cuTAMP does not ship
- `robot_2f140.py`  -- load that model with cuTAMP's API shape
- `validate_2f140.py` -- kinematics / TCP / IK / motion-planning checks on it
- `rs_preflight.py` -- put the RealSense IR pair into a state FoundationStereo
  can use (the rig ships with IR auto-exposure off)
- `warm_servers.py` -- absorb the first-call PTX JIT cost that would otherwise
  blow through tiptop's 10 s perception-server timeout
- `install_*.py`    -- versioned, idempotent patches this rig applies to the
  pristine `droid/tiptop/` worktree and to the FoundationStereo server
"""
