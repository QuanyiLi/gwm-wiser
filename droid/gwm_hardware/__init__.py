"""Hardware-experiment code for the `zhiwei` rig (Franka Panda + Robotiq 2F-140).

Everything specific to running on the real robot lives here, kept apart from
`gwm_tiptop/` (the droid-sim integration) so the sim results stay reproducible
from an unchanged tree.

- `build_2f140.py` / `build_2f140_cfg.py` -- generate the Panda + 2F-140 URDF
  and cuRobo config that this rig needs and cuTAMP does not ship.
- `robot_2f140.py`  -- load that model with cuTAMP's API shape.
- `validate_2f140.py` -- kinematics / TCP / IK / motion-planning checks on it.
- `rs_preflight.py` -- put the RealSense IR pair into a state FoundationStereo
  can use (the rig ships with IR auto-exposure off).
- `warm_servers.py` -- absorb the first-call PTX JIT cost that would otherwise
  blow through tiptop's 10 s perception-server timeout.
- `docs/` -- the bring-up procedure and the controller-machine handover brief.

Resolved in the tiptop pixi env through a site-packages symlink, same recipe
as `gwm_tiptop` and for the same reason (G-21: never a `.pth`, which shadows
the editable-installed `tiptop` and `cutamp` packages):

    ln -sfn /home/quanyi/gwm-wiser/droid/gwm_hardware \\
        "$(/home/quanyi/gwm-wiser/droid/tiptop/.pixi/envs/default/bin/python \\
           -c 'import site; print(site.getsitepackages()[0])')/gwm_hardware"
"""
