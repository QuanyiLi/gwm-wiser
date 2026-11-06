"""
GWM-MPC on WISER with a state-only diffusion policy as the proposal source.

Every ``replan_horizon`` steps the policy samples ``K`` action chunks from the full
low-dimensional state (no image, no language). Each chunk becomes a GWM candidate the
same way a KNN candidate does: robot-only renders of the future joint states at the 6
keyframes (frame 0 replaced by the current camera image), scored against the same
grasp / place sub-prompts. The first ``replan_horizon`` actions of the best chunk are
executed. ``--no_gwm`` executes one random sample per replan instead (policy alone).

Per replan, ``replans.jsonl`` records the mode (cube slot, placard slot) of every
sample, whether the instructed mode is among them, raw cosines, and the selection.

    python -m gwm_wiser.local_eval.state_diffusion_proposals --policy ema_060000.pt \
        --num_samples 16 --gwm_ckpt_path ckpt.pt --embedder_model_path Qwen3-VL-Embedding-8B \
        --result_dir out/k16 --split test
"""

import argparse
import json
import os
import shutil
import time
from types import SimpleNamespace

import datasets
import numpy as np
import torch
import torch.nn.functional as F

from gwm_wiser import PROJECT_ROOT
from gwm_wiser.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE, scene_cfg
from gwm_wiser.local_eval.candidate_set_eval import to_pil, PANDA_GRASP_QPOS
from gwm_wiser.local_eval.state_dp import StateDiffusionPolicy
from gwm_wiser.local_eval.state_replay import full_state, slot_orders
from gwm_wiser.local_eval.traj_tools import KeyframeRenderer, keyframe_indices, task_mode
from gwm_wiser.planner.gwm import GWMBasedPlanner
from gwm_wiser.utils.env import build_endless_env
from gwm_wiser.utils.lerobot import obs_state_key, image_1_key
from gwm_wiser.utils.rollout import rollout

datasets.disable_progress_bar()
_job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
os.environ.setdefault(
    "HF_DATASETS_CACHE", os.path.join(os.environ.get("TMPDIR", "/tmp"), f"hf_cache_{_job_id}")
)

_CX = scene_cfg["cube_spawn_center"][0]
CUBE_SLOTS_XY = np.array([[_CX + 0.05, y] for y in (-0.225, -0.075, 0.075, 0.225)])
PLACARD_SLOTS_XY = np.array([[_CX + 0.25, y] for y in (-0.2, 0.0, 0.2)])


def chunk_mode(tcp_xy: np.ndarray, gripper: np.ndarray, grasped_now: bool):
    """(cube slot or None, placard slot or None) a chunk heads for, from TCP xy and gripper actions.

    Gripper actions are normalised: < 0 closes, > 0 opens.
    """
    closed = gripper < 0
    cube, place = None, None
    if not grasped_now:
        close_steps = np.nonzero(closed)[0]
        t_close = close_steps[0] if len(close_steps) else len(gripper) - 1
        cube = int(np.argmin(np.linalg.norm(CUBE_SLOTS_XY - tcp_xy[t_close], axis=1)))
        if len(close_steps):
            after = np.nonzero(~closed[t_close:])[0]
            t_rel = t_close + after[0] if len(after) else len(gripper) - 1
            if t_rel > t_close + 5:
                place = int(np.argmin(np.linalg.norm(PLACARD_SLOTS_XY - tcp_xy[t_rel], axis=1)))
    else:
        opens = np.nonzero(~closed)[0]
        t_rel = opens[0] if len(opens) else len(gripper) - 1
        place = int(np.argmin(np.linalg.norm(PLACARD_SLOTS_XY - tcp_xy[t_rel], axis=1)))
    return cube, place


class DiffusionProposalPlanner(GWMBasedPlanner):
    def __init__(self, *args, policy=None, num_samples=16, use_gwm=True, seed=0,
                 renderer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.num_samples = num_samples
        self.use_gwm = use_gwm
        self.seed = seed
        self.renderer = renderer
        self.kf = keyframe_indices(self.num_future_frames, 6)
        self.env = None
        self.log_file = None
        self.context = {}
        self.chunks_log = []

    def set_context(self, env, log_path, **context):
        self.env = env
        if self.log_file is not None:
            self.log_file.close()
        self.log_file = open(log_path, "a")
        self.context = context
        self.chunks_log = []

    def reset(self):
        super().reset()
        self.history = None
        self.orders = None

    def _video_candidate(self, chunk: torch.Tensor, qpos_now: torch.Tensor):
        qpos = KeyframeRenderer.actions_to_qpos(chunk[[k - 1 for k in self.kf[1:]]].cpu())
        qpos = torch.cat([qpos_now[None].cpu().float(), qpos], dim=0)
        return SimpleNamespace(images_main=None, images_wrist=None,
                               images_robot_state=self.renderer.render_qpos(qpos))

    def act(self, obs: dict) -> torch.Tensor:
        env = self.env
        steps = obs["elapsed_steps"]
        if (steps == 0).any().item():
            if self._seen_step_zero:
                raise RuntimeError("planner.reset() missing between episodes")
            self._seen_step_zero = True
            self.orders = slot_orders(env)
        s_now = full_state(env, *self.orders)  # (E, D)
        self.history = torch.stack([s_now, s_now], 1) if self.history is None else \
            torch.stack([self.history[:, -1], s_now], 1)

        num_envs = s_now.shape[0]
        need = [e for e in range(num_envs)
                if self.steps_in_trajectory.get(e, self.replan_horizon) >= self.replan_horizon]
        if need:
            step = int(steps[need[0]].item())
            gen = torch.Generator(device=s_now.device).manual_seed(
                hash((self.seed, self.context.get("config", 0), self.context.get("split_id", 0), step)) % (2 ** 31))
            t0 = time.perf_counter()
            samples = self.policy.sample(self.history[need].float(), self.num_samples, generator=gen)
            sample_s = (time.perf_counter() - t0) / (len(need) * self.num_samples)
            for j, e in enumerate(need):
                self._replan(obs, e, samples[j, :, : self.num_future_frames], sample_s)

        actions = []
        for e in range(num_envs):
            actions.append(self.current_trajectories[e][self.steps_in_trajectory[e]])
            self.steps_in_trajectory[e] += 1
        return torch.stack(actions, 0).to(obs[obs_state_key].device).float()

    def _replan(self, obs, e, chunks, sample_s):
        state = obs[obs_state_key][e]
        is_grasped = bool(state[-1].item() < PANDA_GRASP_QPOS)
        step = int(obs["elapsed_steps"][e].item())
        cube_slot, place_slot = task_mode(e)
        K = chunks.shape[0]

        modes, tcp_end = [], []
        for k in range(K):
            q = KeyframeRenderer.actions_to_qpos(chunks[k, ::2].cpu())
            tcp = self.renderer.tcp_positions(q)
            modes.append(chunk_mode(tcp[:, :2], chunks[k, ::2, 7].cpu().numpy(), is_grasped))
            tcp_end.append(tcp[-1].round(4).tolist())
        target_present = any(
            (m[1] == place_slot) if is_grasped else (m[0] == cube_slot) for m in modes)
        full_present = any(m == (cube_slot, place_slot) for m in modes)

        rec = dict(**self.context, env=e, step=step, is_grasped=is_grasped,
                   stage="place" if is_grasped else "grasp",
                   target_cube_slot=cube_slot, target_place_slot=place_slot,
                   sample_modes=modes, target_stage_present=target_present,
                   target_mode_present=full_present, sample_seconds_per_chunk=sample_s)

        if self.use_gwm:
            task_prompt = bytes(obs["task"][e].tolist()).decode("utf-8").rstrip("\x00")
            first_img = to_pil(obs[image_1_key + "_first_frame"][e])
            cur_img = to_pil(obs[image_1_key][e])
            g_emb = self._get_task_embedding(self._parse_grasp_prompt(task_prompt), first_img, cur_img)
            p_emb = self._get_task_embedding(self._parse_place_prompt(task_prompt), first_img, cur_img)
            g, p, g32, p32 = [], [], [], []
            t0 = time.perf_counter()
            for k in range(K):
                v = self.get_video_embedding(self._video_candidate(chunks[k], state), cur_img)
                g.append(F.cosine_similarity(g_emb[None], v[None]).item())
                p.append(F.cosine_similarity(p_emb[None], v[None]).item())
                g32.append(F.cosine_similarity(g_emb[None].float(), v[None].float()).item())
                p32.append(F.cosine_similarity(p_emb[None].float(), v[None].float()).item())
            g, p = np.array(g), np.array(p)
            active = p if is_grasped else g
            order = np.argsort(-active, kind="stable")
            best = int(order[0])
            rec.update(grasp_cos=g.round(6).tolist(), place_cos=p.round(6).tolist(),
                       grasp_cos_fp32=np.round(g32, 6).tolist(), place_cos_fp32=np.round(p32, 6).tolist(),
                       top_cos=float(active[order[0]]),
                       margin=float(active[order[0]] - active[order[1]]) if K > 1 else None,
                       score_seconds=time.perf_counter() - t0)
        else:
            best = 0  # samples are i.i.d., index 0 is a random draw

        sel = modes[best]
        rec.update(selected=best, selected_mode=sel,
                   selected_target_stage=(sel[1] == place_slot) if is_grasped else (sel[0] == cube_slot))
        self.log_file.write(json.dumps(rec) + "\n")
        self.log_file.flush()
        self.chunks_log.append(dict(env=e, step=step, chunks=chunks.cpu().half().numpy(), selected=best))
        self.current_trajectories[e] = chunks[best]
        self.steps_in_trajectory[e] = 0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--no_gwm", action="store_true")
    ap.add_argument("--infer_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gwm_ckpt_path", required=True)
    ap.add_argument("--embedder_model_path", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--start_subset", type=int, default=0)
    ap.add_argument("--end_subset", type=int, default=24)
    ap.add_argument("--replan_horizon", type=int, default=20)
    ap.add_argument("--num_future_frames", type=int, default=60)
    ap.add_argument("--save_videos", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLEmbedder

    policy = StateDiffusionPolicy.load(args.policy)
    if args.infer_steps:
        policy.infer_steps = args.infer_steps
    embedder = None if args.no_gwm else Qwen3VLEmbedder(args.embedder_model_path, torch_dtype=torch.bfloat16)
    planner_cls = DiffusionProposalPlanner
    if args.no_gwm:  # skip GWM / embedder loading entirely
        planner = planner_cls.__new__(planner_cls)
        planner.replan_horizon = args.replan_horizon
        planner.num_future_frames = args.num_future_frames
        planner.current_trajectories, planner.steps_in_trajectory = {}, {}
        planner.query_count, planner._seen_step_zero, planner._task_embedding_cache = 0, False, {}
        planner.policy, planner.num_samples, planner.use_gwm = policy, 1, False
        planner.seed, planner.renderer = args.seed, KeyframeRenderer()
        planner.kf = keyframe_indices(args.num_future_frames, 6)
        planner.env, planner.log_file, planner.context, planner.chunks_log = None, None, {}, []
    else:
        planner = planner_cls(
            dataset_root=os.path.join(PROJECT_ROOT, "gwm_skills/config_0_train/lerobot_data"),
            embedder=embedder, gwm_checkpoint_path=args.gwm_ckpt_path, k=12,
            replan_horizon=args.replan_horizon, video_frame_subsample=6,
            num_future_frames=args.num_future_frames, device="cuda", dtype=torch.bfloat16,
            policy=policy, num_samples=args.num_samples, use_gwm=True, seed=args.seed,
            renderer=KeyframeRenderer(),
        )
    with open(os.path.join(args.result_dir, f"args_{args.split}_{args.start_subset}_{args.end_subset}.json"), "w") as f:
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
        env_cfg = get_env_cfg(
            num_env=12, max_steps=MAX_EPISODE_STEP_WORKSPACE, obs_mode="rgb+segmentation",
            scene_cfg_to_overwrite=dict(robot_init_qpos_noise=0.0, cube_size_noise=0.0,
                                        cfg_name=f"config_{i}", mode=args.split),
            robot_uids="my_panda_wristcam",
        )
        envs = build_endless_env(env_cfg, record_video=False, data_record_dir="test")
        planner.reset()
        planner.set_context(envs.unwrapped, os.path.join(sub, "replans.jsonl"),
                            config=i, split=args.split, split_id=split_id,
                            num_samples=planner.num_samples, use_gwm=planner.use_gwm)

        def policy_fn(obs):
            a = planner.act(dict(obs))
            return a, a, dict(use_expert_action=0)

        t0 = time.perf_counter()
        perf = rollout(envs, policy_fn, round_to_collect=1, demo_saving_dir=sub,
                       indices_to_save=list(range(12)) if args.save_videos else [])
        np.savez_compressed(
            os.path.join(sub, "proposals.npz"),
            env=np.array([c["env"] for c in planner.chunks_log]),
            step=np.array([c["step"] for c in planner.chunks_log]),
            selected=np.array([c["selected"] for c in planner.chunks_log]),
            chunks=np.stack([c["chunks"] for c in planner.chunks_log]),
        )
        print(f"config_{i}_{args.split} {time.perf_counter() - t0:.1f}s "
              f"success_once={perf['success_once_mean']:.3f} grasp={perf['is_grasped_mean']:.3f} "
              f"reach={perf['tcp_near_goal_mean']:.3f}", flush=True)
        envs.unwrapped.close()
    print("Done.")


if __name__ == "__main__":
    main()
