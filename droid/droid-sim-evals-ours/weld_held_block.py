"""weld_held_block: rigidly attach scene6 variant 1's `held_block` to the gripper.

Importing this module wraps `src.sim_evals.sim_utils.settle_sim` so the first
settle of a run authors a `UsdPhysics.FixedJoint` between the Robotiq
`base_link` and `held_block`, then leaves it enabled forever. The place tasks
never open the gripper -- the episode ends with the block held inside a bin --
so there is no release path and the joint is authored once and never touched.

Why this seam:
- `batch_eval_v2.main` / `capture_scene6.main` do
  `from src.sim_evals.sim_utils import settle_sim` INSIDE main, after the Isaac
  boot. `sim_utils` itself imports only torch, so this module can pre-import it
  before the boot and rebind the attribute; the later from-import picks up the
  wrapper. Same monkey-patch grain as grasp_eval/place_eval's SuccessTracker
  swap, zero upstream edits.
- Welding at first-settle time (rather than a prestartup EventTerm) keeps the
  patch inside this repo, and the moment is right: the trial reset has just
  written the robot to its home joint state and the block to its USD spawn pose
  (`reset_scene_to_default`), so the live relative pose IS the authored grasp
  pose, and every later reset restores exactly the poses the joint frames
  encode. The joint prim persists across in-process resets.

Physics facts this relies on (verified against the installed isaacsim 5.0 /
PhysX 107.3.18 by direct source read):
- This env runs `use_fabric=True`, which sets `/physics/updateToUsd=False` --
  runtime USD transforms are frozen at spawn values, so joint frames must come
  from the live tensor API, never `UsdGeom.XformCache` (that is why
  `omni.physx.scripts.utils.createJoint` is NOT used: it computes frames from
  USD and would weld the block to a stale pose).
- PhysX buffers USD authoring and applies it at the next sim step;
  `flush_changes()` forces it (omni/physx/bindings/_physx.pyi:1348). Toggling
  and authoring joints on a live stage is the pattern IsaacLab itself uses for
  fix-root-link (isaaclab/sim/schemas/schemas.py:137) and the SurfaceGripper.
- `physics:excludeFromArticulation=True` keeps the joint a maximal-coordinate
  constraint: the block stays a standalone rigid body, so the RigidObject view
  (and the SuccessTracker reading it) keeps working. The one risk of authoring
  after `sim.reset()` is tensor-view invalidation on stage-topology change;
  `SimulationManager.get_physics_sim_view().check()` guards it -- fail loudly
  rather than record garbage episodes.

The gripper stays OPEN the whole episode (the place plans carry no gripper
step): the 30 mm block sits between pads 85 mm apart with 12.5 mm clearance a
side, so there is no finger contact to fight the weld and no collision
filtering is needed. This also defuses settle_sim's lossy gripper round-trip
(observation -> binary threshold), which would re-open a closed gripper.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "droid-sim-evals"))

import src.sim_evals.sim_utils as _sim_utils

_log = logging.getLogger("weld_held_block")

JOINT_NAME = "weld_to_gripper"
GRIPPER_LINK = "base_link"  # Robotiq 2F-85 base, the wrist_cam's parent link
_MAX_FORCE = 3.4e38  # unbreakable, matches omni.physx utils' MAX_FLOAT convention


def _quat_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _quat_rot(q, v):
    # rotate v by q (wxyz)
    qv = [0.0, v[0], v[1], v[2]]
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


def _link_pose(asset, body_id):
    """Live world pose of an articulation link, (pos, quat wxyz) as lists."""
    data = asset.data
    if hasattr(data, "body_link_pose_w"):
        pose = data.body_link_pose_w[0, body_id]
        return pose[:3].tolist(), pose[3:7].tolist()
    return data.body_pos_w[0, body_id].tolist(), data.body_quat_w[0, body_id].tolist()


def _root_pose(asset):
    data = asset.data
    if hasattr(data, "root_link_pose_w"):
        pose = data.root_link_pose_w[0]
        return pose[:3].tolist(), pose[3:7].tolist()
    return data.root_pos_w[0].tolist(), data.root_quat_w[0].tolist()


def ensure_weld(env) -> None:
    """Author the fixed joint once per process; no-op on scenes without the block."""
    scene = env.unwrapped.scene
    if "held_block" not in scene.rigid_objects:
        return

    import omni.physx
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, Sdf

    obj = scene.rigid_objects["held_block"]
    body_path = obj.root_physx_view.prim_paths[0]
    stage = get_current_stage()
    joint_path = f"{body_path}/{JOINT_NAME}"
    if stage.GetPrimAtPath(joint_path).IsValid():
        return

    robot = scene["robot"]
    body_ids, _ = robot.find_bodies(GRIPPER_LINK)
    link_path = robot.root_physx_view.link_paths[0][body_ids[0]]

    # Live poses (fabric keeps USD frozen; tensors are the ground truth).
    p0, q0 = _link_pose(robot, body_ids[0])
    p1, q1 = _root_pose(obj)
    q0_inv = _quat_conj(q0)
    rel_p = _quat_rot(q0_inv, [p1[i] - p0[i] for i in range(3)])
    rel_q = _quat_mul(q0_inv, q1)

    joint = physicsUtils.add_joint_fixed(
        stage,
        joint_path,
        Sdf.Path(link_path),
        Sdf.Path(body_path),
        Gf.Vec3f(*rel_p),
        Gf.Quatf(rel_q[0], rel_q[1], rel_q[2], rel_q[3]),
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Quatf(1.0, 0.0, 0.0, 0.0),
        _MAX_FORCE,
        _MAX_FORCE,
    )
    joint.CreateExcludeFromArticulationAttr().Set(True)
    omni.physx.get_physx_simulation_interface().flush_changes()
    if not SimulationManager.get_physics_sim_view().check():
        raise RuntimeError(
            "physics tensor views invalidated by welding held_block; "
            "results would be garbage -- aborting"
        )
    _log.info(
        f"welded {body_path} to {link_path}: rel_p={[round(v, 4) for v in rel_p]} "
        f"rel_q={[round(v, 4) for v in rel_q]}"
    )


_orig_settle_sim = _sim_utils.settle_sim


def _settle_with_weld(env, obs, **kwargs):
    ensure_weld(env)
    return _orig_settle_sim(env, obs, **kwargs)


def install() -> None:
    """Idempotently rebind sim_utils.settle_sim; runs at import."""
    if _sim_utils.settle_sim is not _settle_with_weld:
        _sim_utils.settle_sim = _settle_with_weld
        _log.info("settle_sim wrapped: held_block will be welded at first settle")


install()
