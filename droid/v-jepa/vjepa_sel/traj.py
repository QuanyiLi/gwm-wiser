"""Candidate plan -> V-JEPA 2-AC state / action sequence.

Timeline: the harness's control steps (15 Hz, see plan_stepper.py) sampled
every `stride` steps (4 -> 3.75 fps, the cadence DROID was subsampled to for
post-training: 15 Hz recordings, every 4th frame). The k-th sample is the
state at control step k*stride, i.e. the pose the arm is commanded to at that
step; sample 0 is q_init.

State s = [x, y, z, roll, pitch, yaw, gripper]:
  xyz / euler   panda_link8 in the base frame from fk.py (extrinsic xyz Euler)
  gripper       closedness in [0, 1]. The commanded gripper is binary; the
                measured Robotiq opening in DROID ramps over a few frames, so
                the command is ramped linearly across the 20-step gripper
                action (1.33 s), which is about what the 2F-85 takes to close.
Action a_t = poses_to_diff(s_t, s_{t+1}) as in the vjepa2 repo (delta xyz,
euler of R_{t+1} R_t^T, delta gripper).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .fk import fk_link8_batch, rot_to_euler_xyz
from .plan_stepper import GRIPPER_ACTION_STEPS, HOLD_STEPS_AFTER_PLAN, SIM_CONTROL_HZ, unroll_plan

FRAME_STRIDE = 4  # control steps per model frame
ACTION_DT = FRAME_STRIDE / SIM_CONTROL_HZ  # 0.2667 s


def ramp_gripper(grip_cmd, steps=GRIPPER_ACTION_STEPS):
    """Binary per-step gripper command -> closedness ramped over `steps` steps after each change."""
    g = np.asarray(grip_cmd, dtype=np.float64)
    out = np.empty_like(g)
    cur = g[0]
    out[0] = cur
    target, start_val, start_i = cur, cur, 0
    for i in range(1, len(g)):
        if g[i] != target:
            target, start_val, start_i = g[i], out[i - 1], i
        frac = min(1.0, (i - start_i + 1) / steps)
        out[i] = start_val + (target - start_val) * frac
    return out


def poses_to_diffs(states):
    """[T+1, 7] -> [T, 7], the repo's poses_to_diffs (droid.py) vectorised over time."""
    states = np.asarray(states, dtype=np.float64)
    xyz_diff = states[1:, :3] - states[:-1, :3]
    R = Rotation.from_euler("xyz", states[:, 3:6], degrees=False).as_matrix()
    dR = np.einsum("tij,tkj->tik", R[1:], R[:-1])  # R_{t+1} @ R_t^T
    ang = Rotation.from_matrix(dR).as_euler("xyz", degrees=False)
    g = states[1:, 6:7] - states[:-1, 6:7]
    return np.concatenate([xyz_diff, ang, g], axis=1)


def control_states(plan, hold_steps=HOLD_STEPS_AFTER_PLAN, tcp_offset=0.0):
    """Per-control-step commanded states [N+1, 7] (index 0 = q_init) plus the unroll record.

    tcp_offset: metres along panda_link8's z (the tool axis) to move the
    reported point from the flange to a tool-centre point; 0 = flange.
    """
    u = unroll_plan(plan, hold_steps=hold_steps)
    q_seq = np.concatenate([plan["q_init"][None], u["actions"][:, :7]], axis=0)  # state after each step
    # element 0 is the settled gripper state before any command: open (pick scene)
    # or already closed on the welded block (place scene); the caller sets it
    grip_seq = np.concatenate([[0.0], u["actions"][:, 7]])
    pos, rot = fk_link8_batch(q_seq)
    if tcp_offset:
        pos = pos + rot[:, :, 2] * tcp_offset
    eul = rot_to_euler_xyz(rot)
    return {"q": q_seq, "pos": pos, "rot": rot, "euler": eul, "grip_cmd": grip_seq, "unroll": u}


def plan_to_sequence(plan, stride=FRAME_STRIDE, initial_gripper=0.0, hold_steps=HOLD_STEPS_AFTER_PLAN,
                     end="hold", tcp_offset=0.0):
    """Plan -> dict(states [T+1, 7], actions [T, 7], t [T+1], step_index [T+1], close_t, ...).

    end: "hold"  include the 30-step post-plan hold (the judged end state)
         "plan"  stop at the last loop step
    """
    cs = control_states(plan, hold_steps=hold_steps if end == "hold" else 0, tcp_offset=tcp_offset)
    grip_cmd = cs["grip_cmd"].copy()
    grip_cmd[0] = initial_gripper
    grip = ramp_gripper(grip_cmd)
    n = len(cs["q"])
    idx = np.arange(0, n, stride)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)  # keep the final state
    states = np.concatenate([cs["pos"][idx], cs["euler"][idx], grip[idx, None]], axis=1)
    actions = poses_to_diffs(states)
    return {
        "states": states.astype(np.float32),
        "actions": actions.astype(np.float32),
        "t": (idx / SIM_CONTROL_HZ).astype(np.float32),
        "step_index": idx,
        "close_t": cs["unroll"]["close_t"],
        "n_loop": cs["unroll"]["n_loop"],
        "phase": cs["unroll"]["phase"],
        "q": cs["q"][idx].astype(np.float32),
        "control": cs,
    }


def action_stats(actions, maxnorm=0.075):
    a = np.asarray(actions)
    xyz = np.linalg.norm(a[:, :3], axis=1)
    return {
        "n": int(len(a)),
        "xyz_norm_mean": float(xyz.mean()),
        "xyz_norm_max": float(xyz.max()),
        "xyz_abs_max": float(np.abs(a[:, :3]).max()),
        "frac_steps_over_maxnorm_any_axis": float((np.abs(a[:, :3]) > maxnorm).any(axis=1).mean()),
        "rot_abs_max_rad": float(np.abs(a[:, 3:6]).max()),
        "grip_abs_max": float(np.abs(a[:, 6]).max()),
    }
