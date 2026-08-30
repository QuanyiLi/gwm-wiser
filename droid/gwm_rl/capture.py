"""The GWM capture hook: the photo, camera model and candidate format of a
`droid/server/gwm_server.py` ``/score`` request, taken from a live env.

    OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 \\
    ../droid-sim-evals/.venv/bin/python capture.py --headless --out captures/home

writes ``external_cam.png`` / ``external_cam_2.png`` (1280 x 720, the drawer
experiment's rig), ``views.json`` (intrinsics, ``world_from_cam`` in the
OpenCV convention the server consumes, camera pose) and ``state.json`` (the
arm joints, gripper and bowl pose), plus ``request_example.json``: a complete
``/score`` body for one macro action, minus the PNG payload.

Training configs mount no cameras; a run that wants the hook passes
``--capture-envs K`` and calls :func:`camera_view` / :func:`score_request`
on envs ``0..K-1`` (`train.py` does not — that is the guided-exploration
step's job).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gwm_rl import franka_kin as K  # noqa: E402
from gwm_rl import geometry as G  # noqa: E402

TICK_HZ = 15.0
#: The system instruction prefix is the server's; the task instruction is ours.
INSTRUCTION = "pick up the bowl"
INSTRUCTION_PHRASINGS = (
    "pick up the bowl",
    "lift the bowl",
    "grasp the bowl and lift it",
    "pick the bowl up from the table",
    "raise the bowl",
)


# ------------------------------------------------------------ views --


def _quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def camera_view(env, name: str, env_index: int = 0) -> dict:
    """One external camera of env ``env_index``: rgb (H, W, 3) uint8, pixel
    intrinsics, and ``world_from_cam`` (4x4, OpenCV axes, robot-base frame —
    the env origin is subtracted, the server's robot sits at the origin)."""
    sensor = env.scene.sensors[f"{name}_{env_index}"]
    sensor.update(dt=0.0, force_recompute=True)
    rgb = sensor.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    intrinsics = sensor.data.intrinsic_matrices[0].detach().cpu().numpy()
    origin = env.scene.env_origins[env_index].detach().cpu().numpy()
    pos = sensor.data.pos_w[0].detach().cpu().numpy() - origin
    quat = sensor.data.quat_w_ros[0].detach().cpu().numpy()  # (w, x, y, z), optical frame
    c2w = np.eye(4)
    c2w[:3, :3] = _quat_wxyz_to_matrix(quat)
    c2w[:3, 3] = pos
    return {"rgb": rgb, "intrinsics": intrinsics, "world_from_cam": c2w, "pos_w": pos, "quat_w_ros": quat}


def png_b64(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# -------------------------------------------------------- candidates --


def macro_candidate(q0, target_xyz, target_yaw: float, close: bool, closed_before: bool = False,
                    n_move: int = 30, n_hold: int = 14, tcp_offset: float = K.TCP_OFFSET) -> dict:
    """A macro action as a server candidate: the joint trajectory the executor
    would command (open loop from ``q0``), one sample per tick at 15 Hz, with
    the gripper switching once the arm has arrived."""
    import torch

    q = torch.as_tensor(np.asarray(q0, dtype=np.float64), dtype=torch.float32).reshape(1, 7)
    p = torch.tensor([list(map(float, target_xyz))], dtype=torch.float32)
    yaw = torch.tensor([float(target_yaw)], dtype=torch.float32)
    traj = K.plan_macro(q, p, yaw, n_move=n_move, n_hold=n_hold, tcp_offset=tcp_offset)[0].numpy()
    positions = np.concatenate([np.asarray(q0, dtype=np.float64).reshape(1, 7), traj], axis=0)
    n = positions.shape[0]
    t = np.arange(n) / TICK_HZ
    grip = np.full(n, 1.0 if closed_before else 0.0)
    grip[n_move + 1:] = 1.0 if close else 0.0
    close_t = float((n_move + 1) / TICK_HZ) if (close and not closed_before) else None
    return {"positions": positions.tolist(), "t": t.tolist(), "gripper": grip.tolist(), "grasp_close_t": close_t}


def score_request(view: dict, instruction: str, candidates: list[dict], rat_scale: float | None = 1.0,
                  task_image: str = "current") -> dict:
    """The ``/score`` body. ``rat_scale`` 1.0: a 44-tick macro-step is 2.93 s,
    the WISER schedule at unit scale (frames at 0/0.55/1.15/1.75/2.35/2.95 s)."""
    return {
        "rgb_png_b64": png_b64(view["rgb"]),
        "intrinsics": np.asarray(view["intrinsics"]).tolist(),
        "world_from_cam": np.asarray(view["world_from_cam"]).tolist(),
        "instruction": instruction,
        "candidates": candidates,
        "rat_scale": rat_scale,
        "task_image": task_image,
    }


# ------------------------------------------------------------ script --


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(HERE / "captures" / "home"))
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--settle", type=int, default=15, help="ticks to hold before the photo")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from PIL import Image

    import gwm_rl.env_cfg as E
    from gwm_rl.mdp import task_state

    cfg = E.make_env_cfg(num_envs=args.num_envs, capture_envs=1)
    env = gym.make(E.TASK_ID, cfg=cfg)
    base = env.unwrapped
    base.reset()
    state = task_state(base)
    robot = base.scene["robot"]
    q = robot.data.joint_pos[:, state.arm_ids].clone()
    action = torch.cat([q, torch.ones(base.num_envs, 1, device=base.device)], dim=-1)
    for _ in range(args.settle):
        base.step(action)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    views = {}
    for name in G.CAM_POSE:
        v = camera_view(base, name, 0)
        Image.fromarray(v["rgb"]).save(out / f"{name}.png")
        views[name] = {k: np.asarray(v[k]).tolist() for k in ("intrinsics", "world_from_cam", "pos_w", "quat_w_ros")}
        print(f"[capture] {name}: {v['rgb'].shape}, fx {v['intrinsics'][0, 0]:.1f}, pos {v['pos_w']}")
    (out / "views.json").write_text(json.dumps(views, indent=1))

    pred = state.refresh(base)
    origin = base.scene.env_origins[0]
    q0 = robot.data.joint_pos[0, state.arm_ids].cpu().numpy().tolist()
    state_json = {
        "arm_joint_pos": q0,
        "gripper": float(robot.data.joint_pos[0, state.finger_joint_ids[0]].item()) / G.GRIPPER_CLOSED,
        "tcp_pos": pred["tcp_pos"][0].cpu().numpy().tolist(),
        "bowl_pos": (base.scene["bowl"].data.root_pos_w[0] - origin).cpu().numpy().tolist(),
        "bowl_quat": base.scene["bowl"].data.root_quat_w[0].cpu().numpy().tolist(),
    }
    (out / "state.json").write_text(json.dumps(state_json, indent=1))

    # A full request for one candidate: the rim grasp the smoke test executes.
    tx, ty = G.BOWL_POS[0] - G.BOWL_RIM_RADIUS, G.BOWL_POS[1]
    cand = macro_candidate(q0, (tx, ty, G.TABLE_TOP_Z + 0.045), math.pi / 2, close=True)
    req = score_request(views and camera_view(base, "external_cam", 0), INSTRUCTION, [cand])
    req["rgb_png_b64"] = f"<{len(req['rgb_png_b64'])} base64 chars>"
    (out / "request_example.json").write_text(json.dumps(req, indent=1))
    print(f"[capture] wrote {sorted(p.name for p in out.iterdir())} to {out}")

    sys.stdout.flush()
    # No simulation_app.close(): Kit's shutdown hangs after this scene; the
    # process exit frees the GPU.
    os._exit(0)


if __name__ == "__main__":
    main()
