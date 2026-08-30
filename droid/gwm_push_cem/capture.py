"""capture: one Isaac boot -> scene9 layout PNGs + scoring obs H5.

The arm spawns at the push home configuration instead of the DROID reset pose,
the gripper is commanded closed and held closed, and everything is captured
only after the cubes have settled. Writes the ext/ext2/wrist PNGs,
external_obs.h5, wrist_obs.h5 and objects.json, plus home.json, which
records the commanded home configuration and the fingertip height the sim
actually settled at.

    cd /root/code/gwm/gwm-wiser/droid/gwm_push_cem && \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
    ../droid-sim-evals/.venv/bin/python -u capture.py
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "droid-sim-evals"))
sys.path.insert(0, str(HERE))

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args, _ = ap.parse_known_args()

    from config import (CAPTURE_DIR, CUBE_PRIMS, SCENE_ID, SCENE_VARIANT,
                        TABLE_TOP_Z)
    from sim_common import arm_home_qpos

    out_dir = Path(args.out_dir) if args.out_dir else CAPTURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    q_home = arm_home_qpos()

    from isaaclab.app import AppLauncher

    kit_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(kit_parser)
    args_cli, _ = kit_parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import h5py
    import numpy as np
    import torch
    from PIL import Image

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.sim_utils import settle_sim

    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(SCENE_ID), SCENE_VARIANT)
    env_cfg.scene.robot.init_state.joint_pos.update(
        {name: float(v) for name, v in zip(ARM_JOINTS, q_home)})
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # second reset for correct material loading
    obs = settle_sim(env, obs, steps=100)

    # Hold the home arm pose with the gripper commanded closed.
    hold = torch.tensor(q_home, dtype=torch.float32,
                        device=env.unwrapped.device).unsqueeze(0)
    for _ in range(60):
        action = torch.cat([hold, torch.ones((1, 1), device=env.unwrapped.device)], dim=-1)
        obs, _, _, _, _ = env.step(action)

    def np0(t):
        return t[0].cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t[0])

    scene_h = env.unwrapped.scene
    robot = scene_h["robot"]
    p = obs["policy"]
    print("gripper_pos after close:", float(np0(p["gripper_pos"])), flush=True)

    cams = {}
    for cam_name, png in [("external_cam", "ext.png"), ("external_cam_2", "ext2.png"),
                          ("wrist_cam", "wrist.png")]:
        cam = scene_h.sensors[cam_name]
        rgb = np0(cam.data.output["rgb"])[..., :3].astype(np.uint8)
        Image.fromarray(rgb).save(out_dir / png)
        cams[cam_name] = {
            "rgb": rgb,
            "K": np0(cam.data.intrinsic_matrices),
            "pos_w": np0(cam.data.pos_w),
            "quat_w_ros": np0(cam.data.quat_w_ros),
        }

    q_init = np0(p["arm_joint_pos"])
    wrist_depth = np0(scene_h.sensors["wrist_cam"].data.output["distance_to_image_plane"])
    if wrist_depth.ndim == 2:
        wrist_depth = wrist_depth[..., None]

    with h5py.File(out_dir / "wrist_obs.h5", "w") as f:
        f.create_dataset("rgb", data=cams["wrist_cam"]["rgb"])
        f.create_dataset("depth", data=wrist_depth.astype(np.float32))
        f.create_dataset("intrinsic_matrix", data=cams["wrist_cam"]["K"])
        f.create_dataset("pos_w", data=cams["wrist_cam"]["pos_w"])
        f.create_dataset("quat_w_ros", data=cams["wrist_cam"]["quat_w_ros"])
        f.create_dataset("q_init", data=q_init)

    with h5py.File(out_dir / "external_obs.h5", "w") as f:
        for cam_name in ("external_cam", "external_cam_2"):
            c = cams[cam_name]
            f.create_dataset(f"{cam_name}/rgb", data=c["rgb"])
            f.create_dataset(f"{cam_name}/intrinsic_matrix", data=c["K"])
            f.create_dataset(f"{cam_name}/pos_w", data=c["pos_w"])
            f.create_dataset(f"{cam_name}/quat_w_ros", data=c["quat_w_ros"])
        f.create_dataset("arm_joint_pos", data=q_init)
        f.create_dataset("joint_pos", data=np0(robot.data.joint_pos))
        f.create_dataset("joint_names", data=[n.encode() for n in robot.joint_names])
        f.create_dataset("gripper_pos", data=np0(p["gripper_pos"]))

    objects = {
        name: {"pos_w": np0(o.data.root_pos_w).tolist(),
               "quat_w": np0(o.data.root_quat_w).tolist()}
        for name, o in scene_h.rigid_objects.items()
    }
    (out_dir / "objects.json").write_text(json.dumps(objects, indent=2))

    # Fingertip height the articulation actually settled at.
    body_names = list(robot.body_names)
    pads = [body_names.index(n) for n in ("left_inner_finger_pad", "right_inner_finger_pad")
            if n in body_names]
    tip_z = None
    if pads:
        pos = np0(robot.data.body_pos_w)
        tip_z = float(np.mean([pos[i][2] for i in pads]))
    (out_dir / "home.json").write_text(json.dumps({
        "q_home_cmd": [float(v) for v in q_home],
        "q_home_sim": [float(v) for v in q_init.ravel()[:7]],
        "joint_err_max_rad": float(np.max(np.abs(q_init.ravel()[:7] - q_home))),
        "pad_mid_z_sim": tip_z,
        "pad_mid_clearance_m": None if tip_z is None else tip_z - TABLE_TOP_Z,
        "cube_prims": CUBE_PRIMS,
    }, indent=2))
    print("home joint error (rad):", float(np.max(np.abs(q_init.ravel()[:7] - q_home))))
    print("settled pad-mid z:", tip_z)
    for name, o in objects.items():
        print(f"  {name}: {np.round(o['pos_w'], 4).tolist()}")
    print(f"saved captures to {out_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
