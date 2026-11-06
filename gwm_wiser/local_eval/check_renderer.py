"""
Parity check: RobotRenderer renders of the skill library's recorded joint states vs
the library's own ``image_1_robot_state`` keyframes (decoded from video), plus the
TCP forward kinematics used to calibrate candidate noise.

    python -m gwm_wiser.local_eval.check_renderer --out /path/renderer_check.json
"""

import argparse
import json
import os

import numpy as np
import torch

from gwm_wiser import PROJECT_ROOT
from gwm_wiser.local_eval.traj_tools import KeyframeRenderer, smooth_joint_noise
from gwm_wiser.utils.gwm_data import PaddedLeRobotDataset
from gwm_wiser.utils.lerobot import image_1_robot_state, obs_state_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15])
    args = ap.parse_args()

    ds = PaddedLeRobotDataset(repo_id="unused",
                              root=os.path.join(PROJECT_ROOT, "gwm_skills/config_0_train/lerobot_data"),
                              video_frame_subsample=6, num_future_frames=60)
    r = KeyframeRenderer()
    kf = ds._subsample_indices
    diffs, robot_px = [], []
    starts = []
    for ep in range(ds.num_episodes):
        info = ds.meta.episodes[ep]
        for off in (0, 20, 40, 60):
            starts.append(info["dataset_from_index"] + off)
    for idx in starts[::4]:
        s = ds[idx]
        lib = s[image_1_robot_state]  # (6, 3, H, W) in [0, 1]
        ren = r.render_qpos(s[obs_state_key][kf])
        d = (lib - ren).abs()
        mask = (lib.sum(1, keepdim=True) > 0.05) | (ren.sum(1, keepdim=True) > 0.05)
        diffs.append(float(d.mean()))
        robot_px.append(float((d * mask).sum() / (mask.sum() * 3 + 1e-6)))

    # noise calibration: mean TCP deviation over the 60-step window
    rng = np.random.default_rng(0)
    calib = {}
    windows = [ds[i][obs_state_key] for i in starts]
    clean_tcp = [r.tcp_positions(w) for w in windows]
    for sigma in args.sigmas:
        devs = []
        for w, tcp in zip(windows, clean_tcp):
            for _ in range(3):
                q = w.clone()
                q[:, :7] += torch.from_numpy(smooth_joint_noise(rng, q.shape[0], sigma))
                devs.append(np.linalg.norm(r.tcp_positions(q) - tcp, axis=1).mean())
        calib[sigma] = dict(mean_tcp_dev_m=float(np.mean(devs)), std=float(np.std(devs)))
        print(f"sigma {sigma}: mean TCP deviation {np.mean(devs) * 100:.2f} cm", flush=True)

    out = dict(mean_abs_pixel_diff=float(np.mean(diffs)),
               mean_abs_pixel_diff_robot_region=float(np.mean(robot_px)),
               n_windows=len(diffs), noise_calibration=calib)
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    # save one side-by-side example for visual inspection
    from PIL import Image

    s = ds[starts[5]]
    lib = (s[image_1_robot_state][3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ren = (r.render_qpos(s[obs_state_key][kf[3:4]])[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(np.concatenate([lib, ren], axis=1)).save(args.out.replace(".json", ".png"))


if __name__ == "__main__":
    main()
