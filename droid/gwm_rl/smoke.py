"""Bring-up, calibration and throughput for the pick-up-bowl env.

    OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 \\
    ../droid-sim-evals/.venv/bin/python smoke.py --headless --num_envs 4
    ... smoke.py --headless --num_envs 2048 --skip-calibrate --skip-grasp --steps 300

What it checks, in order: the scene builds and the objects sit still on the
table; the torch FK agrees with Isaac's link poses (and where the pads are in
the flange frame, which pins `franka_kin.TCP_OFFSET` / `OPENING_AXIS`); a
scripted rim grasp through the same IK the executors use actually lifts the
bowl and the reward/predicates read it; the obstacle contact sensors see the
arm pressing on the table and a cabinet; and env-steps/s at the requested env
count. Hard-exits at the end (Kit's shutdown hangs after this scene).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=300, help="ticks for the throughput measurement")
parser.add_argument("--skip-calibrate", action="store_true")
parser.add_argument("--skip-grasp", action="store_true")
parser.add_argument("--capture-envs", type=int, default=0)
parser.add_argument("--env-set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides into the env config")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gwm_rl.env_cfg as E  # noqa: E402
from gwm_rl import franka_kin as K  # noqa: E402
from gwm_rl import geometry as G  # noqa: E402
from gwm_rl.mdp import task_state  # noqa: E402

np.set_printoptions(precision=4, suppress=True)


def pad_frames_from_usd(base) -> None:
    """Where the two pad meshes sit in the flange frame, read off the authored USD."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = "/World/envs/env_0/robot"
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    flange = stage.GetPrimAtPath(f"{root}/panda_link8")
    T_f = UsdGeom.Xformable(flange).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    T_f_inv = T_f.GetInverse()
    pads = []
    for side in ("left", "right"):
        prim = stage.GetPrimAtPath(f"{root}/Gripper/Robotiq_2F_85/{side}_inner_finger/Defeatured_2F_85_PAD_OPEN_fingertipsstep_01")
        if not prim.IsValid():
            print(f"[calib] pad prim for {side} not found")
            return
        c = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint()
        local = T_f_inv.Transform(Gf.Vec3d(c))
        pads.append(np.array([local[0], local[1], local[2]]))
        print(f"[calib] {side} pad centre in flange frame: {pads[-1]}")
    mid = (pads[0] + pads[1]) / 2
    sep = pads[1] - pads[0]
    print(f"[calib] pads-mid in flange frame {mid} -> TCP_OFFSET (flange z) = {mid[2]:.4f}; "
          f"opening vector {sep} -> OPENING_AXIS = {int(np.argmax(np.abs(sep[:2])))} (|sep| = {np.linalg.norm(sep):.4f} m)")
    for name in ("base_link", "left_inner_finger", "right_inner_finger"):
        prim = stage.GetPrimAtPath(f"{root}/Gripper/Robotiq_2F_85/{name}")
        T = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p = T_f_inv.Transform(T.ExtractTranslation())
        print(f"[calib] {name} origin in flange frame: {np.array([p[0], p[1], p[2]])}")


def main() -> None:
    cfg = E.make_env_cfg(num_envs=args.num_envs, capture_envs=args.capture_envs, overrides=args.env_set)
    t0 = time.perf_counter()
    env = gym.make(E.TASK_ID, cfg=cfg)
    base = env.unwrapped
    print(f"[smoke] scene build {time.perf_counter() - t0:.1f}s | {base.num_envs} envs | "
          f"max_episode_length {base.max_episode_length} | action dim {base.action_manager.total_action_dim}")
    robot = base.scene["robot"]
    print("[smoke] joint names:", robot.joint_names)
    print("[smoke] body names:", robot.body_names)
    for name, s in base.scene.sensors.items():
        fc = s.contact_physx_view.filter_count if s.cfg.filter_prim_paths_expr else 0
        print(f"[smoke] sensor {name}: bodies {s.num_bodies}, filter shapes {fc}")
    for n in ("bowl", "banana", "block"):
        print(f"[smoke] {n} mass {base.scene[n].root_physx_view.get_masses()[0].item():.4f} kg")
    if not args.skip_calibrate:
        pad_frames_from_usd(base)

    obs, _ = env.reset()
    print("[smoke] obs dim", tuple(obs["policy"].shape))
    state = task_state(base)
    N, dev = base.num_envs, base.device
    origins = base.scene.env_origins

    def act(q, close: bool):
        g = torch.full((N, 1), -1.0 if close else 1.0, device=dev)
        return torch.cat([q, g], dim=-1)

    def arm_q():
        return robot.data.joint_pos[:, state.arm_ids].clone()

    # -- settle
    q = arm_q()
    for _ in range(30):
        obs, rew, term, trunc, info = env.step(act(q, False))
    for n, rest in (("bowl", G.BOWL_POS), ("banana", G.BANANA_POS), ("block", G.BLOCK_POS)):
        d = base.scene[n].data.root_pos_w[0] - origins[0] - torch.tensor(rest, device=dev)
        print(f"[smoke] {n} drift after 30 ticks: {(1000 * d).cpu().numpy()} mm")
    pred = state.refresh(base)
    print(f"[smoke] home: tcp {pred['tcp_pos'][0].cpu().numpy()} yaw {K.yaw_of(pred['R_tool'])[0].item():.3f} "
          f"grasp_dist {pred['grasp_dist'][0].item():.4f} align {pred['grasp_align'][0].item():.3f} "
          f"reward {rew[0].item():.4f} obstacle_force {pred['obstacle_force'][0].item():.3f}")

    # -- FK calibration against Isaac's link poses
    if not args.skip_calibrate:
        torch.manual_seed(0)
        lo = K.JOINT_LOWER.to(dev) + 0.4
        hi = K.JOINT_UPPER.to(dev) - 0.4
        worst_p, worst_r = 0.0, 0.0
        for trial in range(6):
            qr = lo + torch.rand(7, device=dev) * (hi - lo)
            qr[1] = qr[1].clamp(-1.2, 0.3)   # keep the arm above the table
            qr[3] = qr[3].clamp(-2.6, -1.2)
            qt = qr.expand(N, 7).clone()
            for _ in range(60):
                env.step(act(qt, False))
            qm = arm_q()
            flange_pos = robot.data.body_link_pos_w[:, state.flange_id] - origins
            R_sim = K.quat_to_matrix(robot.data.body_link_quat_w[:, state.flange_id])
            p_fk, R_fk = K.fk_tcp(qm, tcp_offset=0.0)
            ep = (p_fk - flange_pos).norm(dim=-1)[0].item()
            er = K.rotvec_from_matrix(R_sim @ R_fk.transpose(-1, -2)).norm(dim=-1)[0].item()
            worst_p, worst_r = max(worst_p, ep), max(worst_r, er)
            print(f"[calib] trial {trial}: tracking |q_cmd-q_meas|max {(qt - qm).abs().max().item():.4f} rad | "
                  f"flange FK err {1000 * ep:.2f} mm, {math.degrees(er):.3f} deg")
        print(f"[calib] FK vs Isaac: worst {1000 * worst_p:.2f} mm / {math.degrees(worst_r):.3f} deg")
        q = K.HOME_Q.to(dev).expand(N, 7).clone()
        for _ in range(60):
            env.step(act(q, False))

    # -- macro executor, scripted
    def run_macro(p_target, yaw_target, close_after: bool, close_before: bool, n_move=30, n_hold=14):
        q0 = arm_q()
        p0, R0 = K.fk_tcp(q0)
        pt = torch.tensor(p_target, device=dev).expand(N, 3)
        yt = torch.full((N,), float(yaw_target), device=dev)
        path_p, path_yaw = K.interpolate_pose(p0, K.yaw_of(R0), pt, yt, n_move)
        rewards = []
        for k in range(n_move + n_hold):
            qm = arm_q()
            if k < n_move:
                q_cmd = K.ik_step(qm, path_p[:, k], K.target_rotation(path_yaw[:, k]))
                close = close_before
            else:
                q_cmd = K.ik_step(qm, pt, K.target_rotation(yt))
                close = close_after
            obs, rew, term, trunc, info = env.step(act(q_cmd, close))
            rewards.append(rew[0].item())
        pred = state.refresh(base)
        left, right = pred["finger_forces"]
        forces = {s: state._sensor_force(base, s)[0].item() for s in state.obstacle_sensors}
        print(f"  -> tcp {pred['tcp_pos'][0].cpu().numpy()} (target {np.array(p_target)}) yaw {K.yaw_of(pred['R_tool'])[0].item():.2f} | "
              f"grasp_dist {pred['grasp_dist'][0].item():.3f} align {pred['grasp_align'][0].item():.2f} | "
              f"finger F {left[0].item():.2f}/{right[0].item():.2f} N grasped {bool(pred['is_grasped'][0])} | "
              f"bowl h {pred['bowl_height'][0].item():.3f} lifted {bool(pred['height_reached'][0])} dwell {pred['dwell_fraction'][0].item():.2f} "
              f"success {bool(pred['success'][0])} | reward mean {np.mean(rewards):.3f} last {rewards[-1]:.3f} | "
              f"obstacle N {{{', '.join(f'{k[8:]}: {v:.1f}' for k, v in forces.items())}}}")

    if not args.skip_grasp:
        e_open = torch.zeros(3)
        e_open[K.OPENING_AXIS] = 1.0
        w0 = K.R_DOWN @ e_open  # world direction of the opening axis at yaw 0
        yaw = -math.atan2(w0[1].item(), w0[0].item())  # so the pads open along world x (the radial)
        tx, ty = G.BOWL_POS[0] - G.BOWL_RIM_RADIUS, G.BOWL_POS[1]
        z_grasp = G.TABLE_TOP_Z + 0.045
        print(f"[grasp] rim target ({tx:.3f}, {ty:.3f}, {z_grasp:.3f}), yaw {yaw:.2f}")
        print("[grasp] 1: above the rim, open")
        run_macro((tx, ty, z_grasp + 0.12), yaw, close_after=False, close_before=False)
        print("[grasp] 2: down onto the rim, then close")
        run_macro((tx, ty, z_grasp), yaw, close_after=True, close_before=False)
        print("[grasp] 3: lift, closed")
        run_macro((tx, ty, z_grasp + 0.15), yaw, close_after=True, close_before=True)
        print("[grasp] 4: hold, closed")
        run_macro((tx, ty, z_grasp + 0.15), yaw, close_after=True, close_before=True)
        print("[grasp] 5: release and retreat")
        run_macro((tx, ty, z_grasp + 0.15), yaw, close_after=False, close_before=False)
        print("[collide] push the tool into the table top")
        run_macro((0.30, -0.28, G.TABLE_TOP_Z - 0.03), 0.0, close_after=False, close_before=False)
        print("[collide] push the tool into the blue cabinet's front")
        run_macro((0.52, -0.30, 0.24), 0.0, close_after=False, close_before=False)
        run_macro((0.66, -0.30, 0.24), 0.0, close_after=False, close_before=False)
        print("[smoke] metrics window:", {k: round(v, 3) for k, v in state.drain().items()})

    # -- throughput
    q_home = K.HOME_Q.to(dev).expand(N, 7).clone()
    for _ in range(20):
        env.step(act(q_home, False))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(args.steps):
        noise = (torch.rand(N, 7, device=dev) * 2 - 1) * 0.3
        env.step(act(q_home + noise, k % 2 == 0))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[smoke] throughput: {N * args.steps / dt:.0f} env-steps/s ({args.steps} ticks x {N} envs in {dt:.1f}s)")
    print("[smoke] metrics window:", {k: round(v, 3) for k, v in state.drain().items()})
    print(f"[smoke] VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB (torch) ")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # No simulation_app.close(): Kit's shutdown hangs after this scene; the
    # process exit frees the GPU.
    os._exit(0)
