"""Forward kinematics without building a motion planner.

Most consumers of the robot model here never plan: the grasp gate FKs a
closing pose, the debug viewer FKs a trajectory, the capture step FKs the
wrist camera, the framing check projects collision spheres. All of them were
calling `tiptop.motion_planning.build_curobo_solvers`, which constructs an IK
solver AND a `MotionGen` AND warms both up.

Measured on the zhiwei rig, 2026-08-19:

    build_curobo_solvers            4.55 s
    cuTAMP's container loader       0.45 s      10.2x

and the two agree exactly -- TCP to 0.0000 mm, and the 273 valid collision
spheres identical to 0.000000 mm (cuRobo pads the sphere buffer to a fixed
size, which is the only reason the raw arrays differ in length; filter on
radius > 0, as every consumer here already does).

That is ~4 s per stage, and a full turn pays it in the gate, the viewer and
the capture, so it was ~12 s of every instruction spent constructing planners
that never planned.

The embodiment comes from tiptop's config, so this honours the rig's 2F-140
redirect (`install_2f140_cutamp`) exactly as the planner path does. Anything
that actually plans -- `run_proposals`, `go_to_q`, `return_home` -- must keep
using `build_curobo_solvers`.
"""

import logging

_log = logging.getLogger(__name__)


def fk_model(tensor_args=None):
    """The robot's kinematics model, for FK / collision-sphere queries only.

    Returns an object with cuRobo's `get_state(q)` interface: `.ee_pose` and
    `.get_link_spheres()`. Falls back to the full solver build for an
    embodiment cuTAMP exposes no direct container loader for, so this is never
    the reason something stops working.
    """
    from curobo.types.base import TensorDeviceType

    from tiptop.config import tiptop_cfg

    tensor_args = tensor_args or TensorDeviceType()
    robot_type = str(tiptop_cfg().robot.type)

    try:
        import cutamp.robots as _r

        loader = {
            "panda_robotiq": "load_panda_robotiq_container",
            "fr3_robotiq": "load_fr3_robotiq_container",
        }.get(robot_type)
        if loader and hasattr(_r, loader):
            return getattr(_r, loader)(tensor_args).kin_model
    except Exception as e:      # noqa: BLE001 - falling back is always safe
        _log.debug(f"kinematics-only load unavailable ({e}); using the full solver build")

    _log.info(f"no direct kinematics loader for {robot_type!r}; building the full "
              "solver stack instead (slower, same answer)")
    from tiptop.motion_planning import build_curobo_solvers

    _, motion_gen, _ = build_curobo_solvers(num_particles=32, num_spheres=64,
                                            include_workspace=False)
    return motion_gen.kinematics
