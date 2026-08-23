"""Hardware-experiment code for the `zhiwei` rig (Franka Panda + Robotiq 2F-140).

Everything specific to running on the real robot lives here, kept apart from
`gwm_tiptop/` (the droid-sim integration) so the sim results stay reproducible
from an unchanged tree.

Two experiments run on this one rig, and they are kept apart on purpose:

    common/      the RIG itself -- robot model, calibration, cameras, the
                 tiptop-tree installers, workspace. Owned by neither arm.
    tiptop_arm/  baseline TiPToP (Gemini + SAM2 + cuTAMP), the A/B control.
    gwm_arm/     GWM x TiPToP -- geometric perception + M2T2 + cuTAMP
                 proposals scored by GWM. The method under test.

`gwm_arm` imports `common`; it never imports `tiptop_arm`, and neither arm
imports the other. Anything both arms need moves down into `common` rather
than being reached across.

Shared, rig-level DATA lives at this level rather than inside a subpackage,
because both arms and the installers consume it:

    assets/   generated, gitignored -- machine-local URDF / cuRobo yml / spheres
    config/   versioned -- `tiptop.yml` (symlinked into the tiptop tree) etc.

Resolved in the tiptop pixi env through a site-packages symlink, same recipe
as `gwm_tiptop` and for the same reason (never a `.pth`, which would shadow
the editable-installed `tiptop` and `cutamp` packages):

    ln -sfn /home/quanyi/gwm-wiser/droid/gwm_hardware \\
        "$(/home/quanyi/gwm-wiser/droid/tiptop/.pixi/envs/default/bin/python \\
           -c 'import site; print(site.getsitepackages()[0])')/gwm_hardware"
"""
