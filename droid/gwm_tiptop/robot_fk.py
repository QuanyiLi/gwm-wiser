"""Forward kinematics without building a motion planner.

Most consumers of the robot model here never plan: the grasp gate FKs a
closing pose, the debug viewer FKs a trajectory, the capture step FKs the
wrist camera, the framing check projects collision spheres. None of them
needs `tiptop.motion_planning.build_curobo_solvers`, which constructs an IK
solver AND a `MotionGen` AND warms both up -- several seconds per stage, paid
again in every stage of a turn -- when cuTAMP's container loader yields the
same kinematics model about an order of magnitude faster.

The two agree exactly: same TCP, and identical valid collision spheres
(cuRobo pads the sphere buffer to a fixed size, which is the only reason the
raw arrays differ in length; filter on radius > 0, as every consumer here
already does).

The embodiment comes from tiptop's config, so this honours the rig's 2F-140
redirect (`install_2f140_cutamp`) exactly as the planner path does. Anything
that actually plans -- `run_proposals`, `go_to_q`, `return_home` -- must keep
using `build_curobo_solvers`.
"""

import logging

_log = logging.getLogger(__name__)

# Per-process, keyed by embodiment. The model is immutable, so a session that
# FKs in several stages loads it once.
_MODELS: dict = {}


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
    if robot_type in _MODELS:
        return _MODELS[robot_type]

    try:
        import cutamp.robots as _r

        loader = {
            "panda_robotiq": "load_panda_robotiq_container",
            "fr3_robotiq": "load_fr3_robotiq_container",
        }.get(robot_type)
        if loader and hasattr(_r, loader):
            _MODELS[robot_type] = getattr(_r, loader)(tensor_args).kin_model
            return _MODELS[robot_type]
    except Exception as e:      # noqa: BLE001 - falling back is always safe
        _log.debug(f"kinematics-only load unavailable ({e}); using the full solver build")

    _log.info(f"no direct kinematics loader for {robot_type!r}; building the full "
              "solver stack instead (slower, same answer)")
    from tiptop.motion_planning import build_curobo_solvers

    _, motion_gen, _ = build_curobo_solvers(num_particles=32, num_spheres=64,
                                            include_workspace=False)
    _MODELS[robot_type] = motion_gen.kinematics
    return _MODELS[robot_type]


_SOLVERS: dict = {}


def _build_solvers(num_particles: int, num_spheres: int, include_workspace: bool,
                   use_cuda_graph: bool):
    """`tiptop.motion_planning.build_curobo_solvers`, with the graph flag exposed.

    Identical otherwise -- same world, same helpers, same order.
    """
    from curobo.geom.types import Cuboid, WorldConfig

    from tiptop.motion_planning import get_ik_solver, get_motion_gen
    from tiptop.workspace import workspace_cuboids

    cuboids = [
        *(workspace_cuboids() if include_workspace else []),
        Cuboid(name="table", dims=[0.01, 0.01, 0.01], pose=[99.9, 99.9, 99.9, 1.0, 0.0, 0.0, 0.0]),
    ]
    world_cfg = WorldConfig(cuboid=cuboids)
    ik_solver = get_ik_solver(world_cfg, num_particles)
    motion_gen = get_motion_gen(world_cfg, collision_activation_distance=0.0,
                                num_spheres=num_spheres, use_cuda_graph=use_cuda_graph)
    return ik_solver, motion_gen, world_cfg


# CUDA graphs off on the SHARED solver, deliberately.
#
# cuRobo records the collision checker into a CUDA graph, and grows its mesh
# cache whenever a scene brings more obstacles than any scene before it. Its own
# source says what that costs (geom/sdf/world_mesh.py:96-99): "when using
# collision checker inside a recorded cuda graph, recreating the cache will
# break the graph as the reference pointer to the cache will change."
#
# A solver that only ever sees one scene never trips this. Shared across turns
# it fires as soon as one turn has more clusters than the turns before it,
# landing as
#
#     RuntimeError: CUDA error: an illegal memory access was encountered
#
# inside the IK solve, i.e. the graph replaying against freed memory. It is not
# recoverable in-process: the context is poisoned and the session dies.
#
# Pre-sizing the cache would keep the graphs, but the size has to be chosen
# before warm-up records them and there is no reliable bound on how many
# clusters a scene will have. Correctness first.
SHARED_USE_CUDA_GRAPH = False


def planning_solvers(num_particles: int, num_spheres: int, include_workspace: bool = True):
    """`build_curobo_solvers`, built once per (process, configuration).

    Constructing the IK solver + MotionGen and warming them takes several
    seconds and is the single largest fixed cost in the proposer. The result
    depends only on the arguments and the robot model, neither of which changes
    between instructions, so a session should pay it at startup and never again.

    Keyed on the arguments rather than assumed constant: a caller that asks for
    a different particle count gets a correctly-built solver, not a silently
    wrong cached one.
    """
    from tiptop.motion_planning import build_curobo_solvers

    key = (int(num_particles), int(num_spheres), bool(include_workspace))
    if key not in _SOLVERS:
        _log.info(f"building cuRobo solvers {key} (once per session, "
                  f"cuda_graph={SHARED_USE_CUDA_GRAPH})")
        _SOLVERS[key] = _build_solvers(*key[:2], include_workspace=key[2],
                                       use_cuda_graph=SHARED_USE_CUDA_GRAPH)
    return _SOLVERS[key]


def default_planning_solvers(num_particles: int = 256, include_workspace: bool = True):
    """The session's planning solvers, derived from tiptop's own config.

    Everything that needs to PLAN -- the proposer, and `go_to_capture` inside
    the capture step -- must ask for the same configuration, or the cache key
    differs and a second solver stack gets built for no reason. Deriving the
    sphere count from `build_tamp_config` the same way the proposer does is
    what keeps the keys equal.
    """
    from tiptop.config import tiptop_cfg
    from tiptop.planning import build_tamp_config

    cfg = tiptop_cfg()
    config = build_tamp_config(
        num_particles=num_particles, max_planning_time=60.0, opt_steps=500,
        robot_type=cfg.robot.type,
        time_dilation_factor=cfg.robot.time_dilation_factor, near_placement=False)
    return planning_solvers(config.num_particles, config.coll_n_spheres,
                            include_workspace=include_workspace)


def release_shared_solver(motion_gen) -> None:
    """Undo what a previous stage welded to the robot.

    cuTAMP's motion solver calls `attach_objects_to_robot` while planning a
    pick (`cutamp/motion_solver.py:226`) and detaches afterwards -- fine when
    every stage builds its own MotionGen, and not fine once they share one.
    Any path that leaves the attachment in place hands the next stage a robot
    with a phantom object welded to its gripper.

    The symptom, on the first place after a pick on a shared solver: every
    approach plan fails with INVALID_START_STATE_WORLD_COLLISION -- the
    CAPTURE pose, which the arm is physically sitting in, reported as
    colliding with the world. Nothing is wrong with the pose; the phantom is
    what collides.

    Idempotent -- cuTAMP itself calls this unconditionally at the top of its
    own solve, for the same reason.
    """
    try:
        motion_gen.detach_object_from_robot("attached_object")
    except Exception as e:      # noqa: BLE001 - nothing attached is the normal case
        _log.debug(f"nothing to detach from the shared solver ({e})")


def reset_world_to_workspace(motion_gen) -> None:
    """Put the rig's static keep-outs back into a shared solver's world.

    Both proposers call `motion_gen.update_world(...)` to plan against the
    scene they just perceived. With a shared solver, the world a later motion
    plans against is whatever the LAST proposal left there: the previous
    turn's objects, and no rig workspace at all.

    So any motion that is not part of a proposal -- going to the capture pose,
    going home -- resets the world first. Workspace-only is deliberate and is
    what tiptop's own `go_to_q` does: at the start of a turn nothing has been
    perceived yet, so the only world known to be valid is the one that does
    not change.
    """
    from curobo.geom.types import WorldConfig

    from tiptop.workspace import workspace_cuboids

    release_shared_solver(motion_gen)
    motion_gen.update_world(WorldConfig(cuboid=list(workspace_cuboids())))
