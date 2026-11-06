"""
GWM-MPC on WISER with a modified KNN candidate set at every replan.

Variants (``--variant``):
    full          unchanged candidate set (the default GWM-MPC configuration)
    rerender      unchanged set, candidate videos re-rendered with RobotRenderer
                  (renderer control for the noisy variants)
    keep          ``--keep_n`` candidates, the correct one kept when present
    drop_correct  the correct candidate removed
    drop_correct_slot  every candidate whose stage-relevant slot matches the instruction
                  removed (instructed cube while grasping, instructed placard while placing),
                  so the set is all-wrong at each stage
    noise         smooth Gaussian noise (``--noise_sigma`` rad) on every candidate's
                  arm-joint trajectory; videos re-rendered, noisy actions executed

The "correct" candidate is skill-library episode ``e`` for env ``e``: the library is
config_0/train with one episode per env index, and every config shares the same cube
and placard slots, so episode ``e`` performs mode (e // 3, e % 3) = the instructed one.

Every replan is appended to ``<result_dir>/config_<i>_<split>/replans.jsonl`` with raw
cosine scores (before any softmax) for both sub-prompts.

    python -m gwm_wiser.local_eval.candidate_set_eval --variant keep --keep_n 6 \
        --gwm_ckpt_path ckpt.pt --embedder_model_path Qwen3-VL-Embedding-8B \
        --result_dir out/keep6 --split test --start_subset 0 --end_subset 4
"""

import argparse
import json
import os
import shutil
import time

import datasets
import numpy as np
import torch
import torch.nn.functional as F

from gwm_wiser import PROJECT_ROOT
from gwm_wiser.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE
from gwm_wiser.local_eval.traj_tools import (
    KeyframeRenderer,
    smooth_joint_noise,
    task_mode,
)
from gwm_wiser.planner.gwm import GWMBasedPlanner
from gwm_wiser.utils.env import build_endless_env
from gwm_wiser.utils.lerobot import obs_state_key, image_1_key
from gwm_wiser.utils.rollout import rollout

datasets.disable_progress_bar()
_job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
os.environ.setdefault(
    "HF_DATASETS_CACHE", os.path.join(os.environ.get("TMPDIR", "/tmp"), f"hf_cache_{_job_id}")
)

MSE_THRESHOLD = 0.04  # MSETrajectoryRetriever.retrieve default
PANDA_GRASP_QPOS = 0.02  # RetrievalBasedPlanner.act: grasped if finger qpos < 0.02


def to_pil(frame: torch.Tensor):
    from PIL import Image

    if frame.dim() == 3 and frame.shape[0] in [1, 3, 4]:
        frame = frame.permute(1, 2, 0)
    arr = frame.cpu().numpy()
    arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return Image.fromarray(arr)


class CandidateSetPlanner(GWMBasedPlanner):
    def __init__(
        self,
        *args,
        variant: str = "full",
        keep_n: int = 12,
        noise_sigma: float = 0.0,
        seed: int = 0,
        renderer: KeyframeRenderer = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.keep_n = keep_n
        self.noise_sigma = noise_sigma
        self.seed = seed
        self.renderer = renderer
        self.log_file = None
        self.context = {}

    def set_context(self, log_path: str, **context):
        if self.log_file is not None:
            self.log_file.close()
        self.log_file = open(log_path, "a")
        self.context = context

    # ── candidate set ────────────────────────────────────────────────────
    def _candidate_set(self, state, env_idx, rng, is_grasped=False):
        everything = self.retriever.retrieve(state, k=self.k, mse_threshold=float("inf"))
        passed = [t for t in everything if t.mse <= MSE_THRESHOLD]
        base = passed if passed else everything  # same fallback as retrieve()
        correct_ep = env_idx
        info = dict(
            correct_ep=correct_ep,
            base_eps=[t.episode_index for t in base],
            correct_in_base=any(t.episode_index == correct_ep for t in base),
            threshold_fallback=not passed,
        )

        if self.variant == "keep":
            correct = [t for t in base if t.episode_index == correct_ep]
            others = [t for t in base if t.episode_index != correct_ep]
            n_others = max(self.keep_n - len(correct), 0)
            if len(others) > n_others:
                pick = rng.choice(len(others), size=n_others, replace=False)
                others = [others[i] for i in sorted(pick)]
            cands = correct + others
        elif self.variant == "drop_correct":
            cands = [t for t in base if t.episode_index != correct_ep]
            if not cands:  # only the correct one passed the threshold
                cands = [t for t in everything if t.episode_index != correct_ep]
                info["drop_fallback"] = True
        elif self.variant == "drop_correct_slot":
            cube_slot, place_slot = task_mode(env_idx)
            wrong = (lambda ep: ep % 3 != place_slot) if is_grasped else (lambda ep: ep // 3 != cube_slot)
            cands = [t for t in base if wrong(t.episode_index)]
            if not cands:
                cands = [t for t in everything if wrong(t.episode_index)]
                info["drop_fallback"] = True
        else:
            cands = list(base)

        if self.variant in ("rerender", "noise"):
            for t in cands:
                if self.variant == "noise":
                    noise = torch.from_numpy(
                        smooth_joint_noise(rng, t.states.shape[0], self.noise_sigma)
                    )
                    t.states = t.states.clone()
                    t.actions = t.actions.clone()
                    t.states[:, :7] += noise
                    t.actions[:, :7] += noise
                kf = self.retriever.dataset._subsample_indices
                t.images_robot_state = self.renderer.render_qpos(t.states[kf])
        info["cand_eps"] = [t.episode_index for t in cands]
        info["cand_mse"] = [round(float(t.mse), 5) for t in cands]
        return cands, info

    # ── scoring with raw cosines ─────────────────────────────────────────
    def _score(self, cands, task_prompt, first_img, cur_img):
        g_emb = self._get_task_embedding(
            self._parse_grasp_prompt(task_prompt), first_img, cur_img
        )
        p_emb = self._get_task_embedding(
            self._parse_place_prompt(task_prompt), first_img, cur_img
        )
        g, p, g32, p32 = [], [], [], []
        for t in cands:
            v = self.get_video_embedding(t, cur_img)
            # selection uses the planner's own (bf16) cosines; fp32 copies are logged
            g.append(F.cosine_similarity(g_emb[None], v[None]).item())
            p.append(F.cosine_similarity(p_emb[None], v[None]).item())
            g32.append(F.cosine_similarity(g_emb[None].float(), v[None].float()).item())
            p32.append(F.cosine_similarity(p_emb[None].float(), v[None].float()).item())
        return np.array(g), np.array(p), np.array(g32), np.array(p32)

    @staticmethod
    def _top_and_margin(scores):
        # stable: ties go to the earlier candidate, like the planner's stable sort
        order = np.argsort(-scores, kind="stable")
        top = float(scores[order[0]])
        margin = float(scores[order[0]] - scores[order[1]]) if len(scores) > 1 else float("nan")
        return top, margin, int(order[0])

    def act(self, obs: dict) -> torch.Tensor:
        if "elapsed_steps" in obs and (obs["elapsed_steps"] == 0).any().item():
            if self._seen_step_zero:
                raise RuntimeError("planner.reset() missing between episodes")
            self._seen_step_zero = True

        num_envs = obs[obs_state_key].shape[0]
        actions_list = []
        for env_idx in range(num_envs):
            if env_idx not in self.steps_in_trajectory:
                self.steps_in_trajectory[env_idx] = self.replan_horizon + 1
            if self.steps_in_trajectory[env_idx] >= self.replan_horizon + 1:
                state = obs[obs_state_key][env_idx].cpu().numpy()
                task_prompt = (
                    bytes(obs["task"][env_idx].tolist()).decode("utf-8").rstrip("\x00")
                )
                first_img = to_pil(obs[image_1_key + "_first_frame"][env_idx])
                cur_img = to_pil(obs[image_1_key][env_idx])
                is_grasped = bool(state[-1] < PANDA_GRASP_QPOS)
                step = int(obs["elapsed_steps"][env_idx].item())

                rng = np.random.default_rng(
                    [self.seed, self.context.get("config", 0),
                     self.context.get("split_id", 0), env_idx, step]
                )
                cands, info = self._candidate_set(state, env_idx, rng, is_grasped)
                g, p, g32, p32 = self._score(cands, task_prompt, first_img, cur_img)
                active = p if is_grasped else g
                top, margin, best = self._top_and_margin(active)
                g_top, g_margin, _ = self._top_and_margin(g)
                p_top, p_margin, _ = self._top_and_margin(p)

                chosen = cands[best]
                self.current_trajectories[env_idx] = chosen
                self.steps_in_trajectory[env_idx] = 1

                cube_slot, place_slot = task_mode(env_idx)
                rec = dict(
                    **{k: v for k, v in self.context.items()},
                    env=env_idx, step=step, is_grasped=is_grasped,
                    stage="place" if is_grasped else "grasp",
                    task=task_prompt, target_cube_slot=cube_slot, target_place_slot=place_slot,
                    **info,
                    grasp_cos=[round(x, 6) for x in g.tolist()],
                    place_cos=[round(x, 6) for x in p.tolist()],
                    grasp_cos_fp32=[round(x, 6) for x in g32.tolist()],
                    place_cos_fp32=[round(x, 6) for x in p32.tolist()],
                    selected_ep=chosen.episode_index,
                    selected_correct=chosen.episode_index == info["correct_ep"],
                    selected_correct_cube=chosen.episode_index // 3 == cube_slot,
                    top_cos=top, margin=margin,
                    grasp_top_cos=g_top, grasp_margin=g_margin,
                    place_top_cos=p_top, place_margin=p_margin,
                )
                self.log_file.write(json.dumps(rec) + "\n")
                self.log_file.flush()

            action = self.current_trajectories[env_idx].actions[self.steps_in_trajectory[env_idx]]
            self.steps_in_trajectory[env_idx] += 1
            actions_list.append(action)

        return torch.stack(actions_list, dim=0).to(obs[obs_state_key].device).to(torch.float32)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["full", "rerender", "keep", "drop_correct", "drop_correct_slot", "noise"])
    ap.add_argument("--keep_n", type=int, default=12)
    ap.add_argument("--noise_sigma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gwm_ckpt_path", required=True)
    ap.add_argument("--embedder_model_path", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--start_subset", type=int, default=0)
    ap.add_argument("--end_subset", type=int, default=24)
    ap.add_argument("--replan_horizon", type=int, default=20)
    ap.add_argument("--num_future_frames", type=int, default=60)
    ap.add_argument("--video_frame_subsample", type=int, default=6)
    ap.add_argument("--save_videos", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLEmbedder

    embedder = Qwen3VLEmbedder(args.embedder_model_path, torch_dtype=torch.bfloat16)
    renderer = KeyframeRenderer() if args.variant in ("rerender", "noise") else None
    planner = CandidateSetPlanner(
        dataset_root=os.path.join(PROJECT_ROOT, "gwm_skills/config_0_train/lerobot_data"),
        embedder=embedder,
        gwm_checkpoint_path=args.gwm_ckpt_path,
        k=12,
        replan_horizon=args.replan_horizon,
        video_frame_subsample=args.video_frame_subsample,
        num_future_frames=args.num_future_frames,
        device="cuda",
        dtype=torch.bfloat16,
        variant=args.variant,
        keep_n=args.keep_n,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
        renderer=renderer,
    )
    with open(os.path.join(args.result_dir, f"args_{args.start_subset}_{args.end_subset}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

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
        planner.set_context(
            os.path.join(sub, "replans.jsonl"), config=i, split=args.split,
            split_id=split_id, variant=args.variant,
        )
        env_cfg = get_env_cfg(
            num_env=12,
            max_steps=MAX_EPISODE_STEP_WORKSPACE,
            obs_mode="rgb+segmentation",
            scene_cfg_to_overwrite=dict(
                robot_init_qpos_noise=0.0, cube_size_noise=0.0, cfg_name=f"config_{i}", mode=args.split
            ),
            robot_uids="my_panda_wristcam",
        )
        envs = build_endless_env(env_cfg, record_video=False, data_record_dir="test")

        def policy(obs):
            a = planner.act(dict(obs))
            return a, a, dict(use_expert_action=0)

        t0 = time.perf_counter()
        perf = rollout(
            envs, policy, round_to_collect=1, demo_saving_dir=sub,
            indices_to_save=list(range(12)) if args.save_videos else [],
        )
        print(f"config_{i}_{args.split} {time.perf_counter() - t0:.1f}s "
              f"success_once={perf['success_once_mean']:.3f} grasp={perf['is_grasped_mean']:.3f} "
              f"reach={perf['tcp_near_goal_mean']:.3f}", flush=True)
        envs.unwrapped.close()
    print("Done.")


if __name__ == "__main__":
    main()
