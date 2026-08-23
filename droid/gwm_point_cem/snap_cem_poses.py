"""snap_cem_poses: external-camera images of the arm at each CEM step, per task.

One Isaac boot. For every task in results/snap_poses.json (from prep_snap.py)
the arm is driven to each step's joint target with the gripper closed, and
both external cameras are saved to
results/cem_frames/{task}/{step:02d}_{label}_{cam}.png.

    cd /root/code/gwm/gwm-wiser/droid/gwm_point_cem && \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
    ../droid-sim-evals/.venv/bin/python -u snap_cem_poses.py
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "droid-sim-evals"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=7)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--steps-per-pose", type=int, default=35)
    args, _ = ap.parse_known_args()

    poses = json.loads((HERE / "results" / "snap_poses.json").read_text())

    from isaaclab.app import AppLauncher

    kit_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(kit_parser)
    args_cli, _ = kit_parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.sim_utils import settle_sim

    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(args.scene), args.variant)
    env = gym.make("DROID", cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()
    obs = settle_sim(env, obs, steps=60)
    device = obs["policy"]["arm_joint_pos"].device

    def drive_to(q, n):
        nonlocal obs
        target = torch.tensor([q], dtype=torch.float32, device=device)
        grip = torch.ones((1, 1), dtype=torch.float32, device=device)
        action = torch.cat([target, grip], dim=-1)
        for _ in range(n):
            obs, _, _, _, _ = env.step(action)

    scene_h = env.unwrapped.scene
    home_q = None
    for tag, rows in poses.items():
        out_dir = HERE / "results" / "cem_frames" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        for step, row in enumerate(rows):
            if home_q is None and row["label"] == "home":
                home_q = row["q"]
            drive_to(row["q"], args.steps_per_pose)
            for cam_name, short in [("external_cam", "ext"), ("external_cam_2", "ext2")]:
                cam = scene_h.sensors[cam_name]
                rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
                Image.fromarray(rgb).save(out_dir / f"{step:02d}_{row['label']}_{short}.png")
            print(f"[{tag}] {row['label']} saved", flush=True)
        if home_q is not None:
            drive_to(home_q, args.steps_per_pose)  # return home between tasks

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
