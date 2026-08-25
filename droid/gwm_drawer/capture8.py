"""capture8: one Isaac boot -> scene8 layout PNGs + scoring obs H5, gripper open.

Writes the ext/ext2/wrist PNGs, external_obs.h5, wrist_obs.h5 and
objects.json at the standard home pose with the gripper open, plus a
drawer-pose check: each drawer's settled world pose is
compared against its authored spawn so a bad joint would be caught here, not
at scoring time.

    cd /root/code/gwm/gwm-wiser/droid/gwm_drawer && \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
    ../droid-sim-evals/.venv/bin/python -u capture8.py
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "droid-sim-evals"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=8)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--out-dir", default=str(HERE / "captures" / "scene8_0"))
    args, _ = ap.parse_known_args()
    out_dir = Path(args.out_dir)
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
    sys.path.insert(0, str(HERE))
    from config import apply_camera_rig

    apply_camera_rig(env_cfg.scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # second reset for correct material loading
    obs = settle_sim(env, obs, steps=100)

    def np0(t):
        return t[0].cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t[0])

    scene_h = env.unwrapped.scene
    robot = scene_h["robot"]
    p = obs["policy"]

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

    from config import DRAWERS

    print("drawer settle check (settled pos vs authored spawn):", flush=True)
    for key, cab in DRAWERS.items():
        z_lo, z_hi, _, _ = cab.front_rect()
        spawn = np.asarray(cab.to_world((cab.front_x_local, 0.0, (z_lo + z_hi) / 2)))
        drift = np.asarray(objects[f"{key}_drawer"]["pos_w"]) - spawn
        print(f"  {key}_drawer: drift {np.round(drift * 1000, 1)} mm", flush=True)
    print(f"saved captures to {out_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
