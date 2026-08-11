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

RELEASE (added 2026-08-11, G-31): the weld is disabled the first time the
gripper REOPENS after having closed on the block. Without it the place
benchmark is asymmetric. The GWM place candidates end holding the block inside
the bin, but TiPToP's native place plan hovers at the bin mouth and RELEASES,
so a no-op release makes the arm carry the block home and the score measures
the weld, not the grounding — measured on the smoke run, tiptop puts the block
3-10 mm from the correct bin centre in xy (tighter than any GWM arm's 14-35 mm)
yet z_rel +0.061..+0.065 lands it outside the judge band.

Releasing on reopen costs the GWM arms nothing: their plans close the gripper
and never open it, so `_closed_seen` latches and the joint is never disabled —
already-recorded GWM results stay valid. `ensure_weld` re-enables the joint at
every settle, so a trial that released does not leak into the next one (and
settle_sim's lossy gripper round-trip stays defused).
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

# Release thresholds on the Robotiq driver joint (`finger_joint`, rad: 0 = open,
# pi/4 = fully closed; the observation term rescales by pi/4). Closing on the
# 30 mm block lands near 0.5 rad, so 0.30/0.12 sit either side of it with room.
CLOSE_RAD = 0.30
OPEN_RAD = 0.12

_closed_seen = False
_released = False


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


def _joint_prim(scene):
    """The weld joint prim if it has been authored, else None."""
    from isaacsim.core.utils.stage import get_current_stage

    body_path = scene.rigid_objects["held_block"].root_physx_view.prim_paths[0]
    prim = get_current_stage().GetPrimAtPath(f"{body_path}/{JOINT_NAME}")
    return prim if prim.IsValid() else None


def _set_joint_enabled(scene, enabled: bool) -> None:
    import omni.physx
    from pxr import UsdPhysics

    prim = _joint_prim(scene)
    if prim is None:
        return
    UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(enabled)
    omni.physx.get_physx_simulation_interface().flush_changes()


def maybe_release(scene) -> None:
    """Per-step hook: drop the weld once the gripper reopens after closing.

    Called from the success tracker's snapshot (the one per-step seam that
    already exists in batch_eval_v2's loop). Latching on `_closed_seen` is what
    keeps the GWM arms unaffected — their place plans close and never reopen,
    and the gripper starts the episode OPEN, which must not count as a release.
    """
    global _closed_seen, _released
    if _released or "held_block" not in scene.rigid_objects:
        return
    robot = scene["robot"]
    idx = [i for i, n in enumerate(robot.data.joint_names) if n == "finger_joint"]
    if not idx:
        return
    q = float(robot.data.joint_pos[0, idx[0]])
    if q > CLOSE_RAD:
        _closed_seen = True
    elif _closed_seen and q < OPEN_RAD:
        _set_joint_enabled(scene, False)
        _released = True
        _log.info(f"weld released: finger_joint {q:.3f} rad after a close — held_block is free")


def ensure_weld(env) -> None:
    """Author the fixed joint once per process; no-op on scenes without the block.

    Also the per-trial reset seam: a trial that released the weld left the joint
    disabled, so re-enable it (every reset restores exactly the poses the joint
    frames encode) and clear the release latch.
    """
    global _closed_seen, _released
    scene = env.unwrapped.scene
    if "held_block" not in scene.rigid_objects:
        return
    if _joint_prim(scene) is not None:
        if _released:
            _set_joint_enabled(scene, True)
            _log.info("weld re-enabled for the next trial")
        _closed_seen = _released = False
        return

    import omni.physx
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx.scripts import physicsUtils
    from pxr import Gf, Sdf

    obj = scene.rigid_objects["held_block"]
    body_path = obj.root_physx_view.prim_paths[0]
    stage = get_current_stage()
    joint_path = f"{body_path}/{JOINT_NAME}"   # existence handled above

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
