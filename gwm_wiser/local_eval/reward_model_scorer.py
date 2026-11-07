"""
GWM-MPC on WISER with GWM's language cost replaced by a pretrained visual reward model.

Everything is the ``full`` candidate-set run of ``candidate_set_eval`` (KNN proposals
with the MSE filter, horizon 60, 6 keyframes, replan every 20 steps, stage-wise
grasp / place sub-prompts, stable argmax), except the scoring function: each
candidate's input is exactly GWM's (frame 0 = current RGB, frames 1-5 = the skill
library's robot-only renders of the candidate's future joint states on black), sent
with the stage's sub-prompt to a reward-model server (``reward_model_server.py``,
running in its own environment) that returns the model's last-frame progress.

The per-replan log keeps ``candidate_set_eval``'s field names (``grasp_cos`` etc. hold
the reward scores) so ``summarize.py`` reads it unchanged; success probabilities are
logged alongside.

    python -m gwm_wiser.local_eval.reward_model_scorer --server_url http://127.0.0.1:8950 \
        --gwm_ckpt_path ckpt.pt --result_dir out/progress --split test
"""

import argparse
import io
import json
import os
import shutil
import time

import numpy as np
import requests
import torch

from gwm_wiser import PROJECT_ROOT
from gwm_wiser.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE
from gwm_wiser.local_eval.candidate_set_eval import CandidateSetPlanner
from gwm_wiser.utils.env import build_endless_env
from gwm_wiser.utils.rollout import rollout


def _uint8_hwc(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):  # (C, H, W) float in [0, 1]
        return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)  # PIL


class RewardModelPlanner(CandidateSetPlanner):
    def __init__(self, *args, server_url: str, **kwargs):
        super().__init__(*args, variant="full", **kwargs)
        self.server_url = server_url.rstrip("/")
        self.last_success = {}

    def _frames(self, traj, cur_img) -> np.ndarray:
        """GWM's candidate input: current RGB + the library's robot-only renders (frames 1-5)."""
        frames = [_uint8_hwc(f) for f in traj.images_robot_state]
        frames[0] = _uint8_hwc(cur_img)
        return np.stack(frames, axis=0)

    def _request(self, frames: np.ndarray, task: str):
        buf = io.BytesIO()
        np.savez(buf, frames=frames, tasks=np.array([task] * len(frames)))
        r = requests.post(f"{self.server_url}/score", data=buf.getvalue(), timeout=600,
                          headers={"Content-Type": "application/octet-stream"})
        r.raise_for_status()
        out = r.json()
        return np.asarray(out["progress"], dtype=np.float64), np.asarray(out["success"], dtype=np.float64)

    def _score(self, cands, task_prompt, first_img, cur_img):
        stack = np.stack([self._frames(t, cur_img) for t in cands], axis=0)  # (N, 6, H, W, 3)
        g, g_succ = self._request(stack, self._parse_grasp_prompt(task_prompt))
        p, p_succ = self._request(stack, self._parse_place_prompt(task_prompt))
        self.last_success = dict(grasp_success=g_succ.round(6).tolist(), place_success=p_succ.round(6).tolist())
        return g, p, g, p  # the reward scores are already float32 on the server

    def _extra_record(self) -> dict:
        return dict(self.last_success)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", required=True)
    ap.add_argument("--gwm_ckpt_path", required=True,
                    help="loaded only because the planner class builds it; never used for scoring")
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--start_subset", type=int, default=0)
    ap.add_argument("--end_subset", type=int, default=24)
    ap.add_argument("--replan_horizon", type=int, default=20)
    ap.add_argument("--num_future_frames", type=int, default=60)
    ap.add_argument("--video_frame_subsample", type=int, default=6)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    planner = RewardModelPlanner(
        dataset_root=os.path.join(PROJECT_ROOT, "gwm_skills/config_0_train/lerobot_data"),
        embedder=None, gwm_checkpoint_path=args.gwm_ckpt_path, k=12,
        replan_horizon=args.replan_horizon, video_frame_subsample=args.video_frame_subsample,
        num_future_frames=args.num_future_frames, device="cuda", dtype=torch.bfloat16,
        server_url=args.server_url,
    )
    health = requests.get(f"{args.server_url.rstrip('/')}/health", timeout=60).json()
    with open(os.path.join(args.result_dir, f"args_{args.split}_{args.start_subset}_{args.end_subset}.json"), "w") as f:
        json.dump({**vars(args), "server": health}, f, indent=2)

    split_id = 0 if args.split == "train" else 1
    for i in range(args.start_subset, args.end_subset):
        sub = os.path.join(args.result_dir, f"config_{i}_{args.split}")
        if os.path.exists(os.path.join(sub, "episode_metrics.json")):
            print(f"skip config_{i}_{args.split}: done")
            continue
        if os.path.exists(sub):
            shutil.rmtree(sub)
        os.makedirs(sub)
        planner.reset()
        planner.set_context(os.path.join(sub, "replans.jsonl"), config=i, split=args.split,
                            split_id=split_id, variant="reward_model_progress")
        env_cfg = get_env_cfg(
            num_env=12, max_steps=MAX_EPISODE_STEP_WORKSPACE, obs_mode="rgb+segmentation",
            scene_cfg_to_overwrite=dict(robot_init_qpos_noise=0.0, cube_size_noise=0.0,
                                        cfg_name=f"config_{i}", mode=args.split),
            robot_uids="my_panda_wristcam",
        )
        envs = build_endless_env(env_cfg, record_video=False, data_record_dir="test")

        def policy(obs):
            a = planner.act(dict(obs))
            return a, a, dict(use_expert_action=0)

        t0 = time.perf_counter()
        perf = rollout(envs, policy, round_to_collect=1, demo_saving_dir=sub, indices_to_save=[])
        print(f"config_{i}_{args.split} {time.perf_counter() - t0:.1f}s "
              f"success_once={perf['success_once_mean']:.3f} grasp={perf['is_grasped_mean']:.3f} "
              f"reach={perf['tcp_near_goal_mean']:.3f}", flush=True)
        envs.unwrapped.close()
    print("Done.")


if __name__ == "__main__":
    main()
