"""capture_scene6: one Isaac boot -> layout PNGs + proposal-ready obs H5s.

For a scene/variant this saves, after the standard double-reset + 100-step
settle (same as capture_external_cam.py):
  - ext.png / ext2.png / wrist.png     layout views for eyeballing
  - wrist_obs.h5                       smoke_test.h5-style wrist obs
                                       (rgb, depth, intrinsic_matrix, pos_w,
                                        quat_w_ros, q_init) for propose_from_h5
  - external_obs.h5                    save_h5_obs.py-style external-cam obs +
                                       robot state for gwm score_client
  - objects.json                       settled rigid-object root poses

    cd /root/code/gwm/gwm-wiser/droid/droid-sim-evals-ours && \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
    ../droid-sim-evals/.venv/bin/python -u scenes/capture_scene6.py --scene 6 --variant 0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "droid-sim-evals"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=6)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args, _ = ap.parse_known_args()
    out_dir = Path(args.out_dir or Path(__file__).resolve().parent / "captures" / f"scene{args.scene}_{args.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    env_cfg.set_scene(str(args.scene), args.variant)
    env = gym.make("DROID", cfg=env_cfg)

    def log_objects(tag):
        sc = env.unwrapped.scene
        parts = []
        for name, o in sc.rigid_objects.items():
            pos = o.data.root_pos_w[0].cpu().numpy()
            parts.append(f"{name}=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
        print(f"[settle-trace] {tag}: " + "  ".join(parts), flush=True)

    obs, _ = env.reset()
    log_objects("after reset 1")
    obs, _ = env.reset()  # second reset for correct material loading
    log_objects("after reset 2")
    for chunk in range(10):
        obs = settle_sim(env, obs, steps=10)
        log_objects(f"settle step {(chunk + 1) * 10}")

    def np0(t):
        return t[0].cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t[0])

    scene_h = env.unwrapped.scene
    robot = scene_h["robot"]
    p = obs["policy"]

    cams = {}
    for cam_name, png in [("external_cam", "ext.png"), ("external_cam_2", "ext2.png"), ("wrist_cam", "wrist.png")]:
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
        f.create_dataset("robot_root_pos_w", data=np0(robot.data.root_pos_w))
        f.create_dataset("robot_root_quat_w", data=np0(robot.data.root_quat_w))
        f.create_dataset("body_names", data=[n.encode() for n in robot.body_names])
        f.create_dataset("body_pos_w", data=np0(robot.data.body_pos_w))
        f.create_dataset("body_quat_w", data=np0(robot.data.body_quat_w))

    objects = {
        name: {
            "pos_w": np0(o.data.root_pos_w).tolist(),
            "quat_w": np0(o.data.root_quat_w).tolist(),
        }
        for name, o in scene_h.rigid_objects.items()
    }
    (out_dir / "objects.json").write_text(json.dumps(objects, indent=2))
    print("rigid objects:", json.dumps(objects, indent=2))
    print(f"saved captures to {out_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
