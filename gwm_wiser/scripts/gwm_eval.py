"""
Evaluate a GWM-based or retrieval-based planner across config subsets.

GWM mode (default):
    python gwm_eval.py --gwm_ckpt_path /path/to/checkpoint.pt

Retrieval (ground-truth) mode:
    python gwm_eval.py --use_gt --dataset_root /path/to/wiser_dataset/no_noise_demo_1_round

Distributed (each node handles a shard):
    CUDA_VISIBLE_DEVICES=0 python gwm_eval.py --start_subset 0 --end_subset 6 &
    ...
    wait
    python gwm_eval.py --aggregate_only --result_dir ./gwm_eval_result
"""

import argparse
import json
import os
import shutil
import time

import datasets
import torch

from gwm_wiser import PROJECT_ROOT
from gwm_wiser.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE
from gwm_wiser.utils.env import build_endless_env
from gwm_wiser.utils.rollout import rollout, calculate_averages

# Runtime setup — must come after imports but before any function calls
os.environ["SVT_LOG"] = "0"
datasets.disable_progress_bar()

# Avoid race conditions when multiple Slurm jobs load the same dataset
# concurrently: HuggingFace datasets creates .incomplete arrow files in
# a shared cache dir, causing OSError collisions.
_job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
os.environ["HF_DATASETS_CACHE"] = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), f"hf_cache_{_job_id}"
)


def parse_args():
    parser = argparse.ArgumentParser(description="GWM Evaluation")
    parser.add_argument(
        "--gwm_ckpt_path",
        type=str,
        default=None,
        help="Path to GWM checkpoint (required unless --aggregate_only)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./gwm_eval_result",
        help="Output directory for results",
    )
    parser.add_argument(
        "--start_subset", type=int, default=0, help="Start config index (inclusive)"
    )
    parser.add_argument(
        "--end_subset", type=int, default=24, help="End config index (exclusive)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Which split(s) to evaluate",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip rollouts, only compute final aggregated results",
    )
    parser.add_argument(
        "--embedder_model_path",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-8B",
        help="Path to Qwen3-VL-Embedding model",
    )
    parser.add_argument("--video_frame_subsample", type=int, default=6)
    parser.add_argument("--replan_horizon", type=int, default=20)
    parser.add_argument("--num_future_frames", type=int, default=60)
    parser.add_argument(
        "--save_videos", action="store_true", help="Save rollout videos"
    )
    parser.add_argument(
        "--num_env",
        type=int,
        default=12,
        help="Number of parallel environments per rollout",
    )
    parser.add_argument(
        "--eval_rounds", type=int, default=1, help="Number of rollout rounds per config"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--action_conditioned",
        action="store_true",
        help="Use action-conditioned GWM instead of standard GWM",
    )
    parser.add_argument(
        "--use_gt",
        action="store_true",
        help="Use retrieval-based planner (ground-truth) instead of GWM",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root dir for demo data (used by retrieval planner in --use_gt mode)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="panda",
        choices=["panda", "xarm6"],
        help="Robot embodiment (panda or xarm6)",
    )
    return parser.parse_args()


def aggregate_results(result_dir):
    """Compute final aggregated results for train and test splits."""
    for split in ["train", "test"]:
        pattern = os.path.join(result_dir, f"*{split}*")
        final_results = calculate_averages(pattern)
        if final_results:
            out_path = os.path.join(result_dir, f"final_results_{split}.json")
            with open(out_path, "w") as f:
                json.dump(final_results, f, indent=2)
            print(f"Final results saved to {out_path}")
        else:
            print(f"No results found for split '{split}'")


def main():
    args = parse_args()

    # Aggregate-only mode: just compute final results and exit
    if args.aggregate_only:
        aggregate_results(args.result_dir)
        return

    os.makedirs(args.result_dir, exist_ok=True)

    # Lazy-import heavy dependencies only when needed
    from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLEmbedder

    # Build shared embedder (used by both GWM and retrieval planners)
    embedder = Qwen3VLEmbedder(args.embedder_model_path, torch_dtype=torch.bfloat16)

    # Build planner (GWM or retrieval)
    if args.use_gt:
        # Retrieval planner: planner is rebuilt per config (different dataset_root)
        planner = None
        print(f"Using retrieval-based (ground-truth) planner for {args.robot}")
    else:
        if args.gwm_ckpt_path is None:
            raise ValueError(
                "--gwm_ckpt_path is required unless --aggregate_only or --use_gt is set"
            )
        from gwm_wiser.planner.gwm import GWMBasedPlanner, ActionConditionedGWMPlanner

        if args.robot == "panda":
            dataset_root = os.path.join(
                PROJECT_ROOT, "gwm_skills/config_0_train/lerobot_data"
            )
        else:
            dataset_root = os.path.join(
                PROJECT_ROOT, "gwm_skills/config_0_train_xarm6/lerobot_data"
            )

        if args.action_conditioned:
            planner = ActionConditionedGWMPlanner(
                dataset_root=dataset_root,
                embedder=embedder,
                verbose=args.verbose,
                replan_horizon=args.replan_horizon,
                video_frame_subsample=args.video_frame_subsample,
                num_future_frames=args.num_future_frames,
                gwm_checkpoint_path=args.gwm_ckpt_path,
                device="cuda",
                dtype=torch.bfloat16,
                robot=args.robot,
            )
        else:
            planner = GWMBasedPlanner(
                dataset_root=dataset_root,
                embedder=embedder,
                verbose=args.verbose,
                replan_horizon=args.replan_horizon,
                video_frame_subsample=args.video_frame_subsample,
                num_future_frames=args.num_future_frames,
                gwm_checkpoint_path=args.gwm_ckpt_path,
                device="cuda",
                dtype=torch.bfloat16,
                robot=args.robot,
            )

    # Determine splits to evaluate
    splits = ["train", "test"] if args.split == "both" else [args.split]

    print(
        f"Evaluating subsets {args.start_subset} to {args.end_subset - 1} "
        f"on splits: {splits}"
    )

    # Evaluation loop
    for split in splits:
        for i in range(args.start_subset, args.end_subset):
            cfg_name = f"config_{i}"

            # For retrieval planner, rebuild per config with its own dataset
            if args.use_gt:
                from gwm_wiser.planner.retrieval import RetrievalBasedPlanner

                per_cfg_dataset_root = os.path.join(
                    args.dataset_root, f"{cfg_name}_{split}", "lerobot_data"
                )
                planner = RetrievalBasedPlanner(
                    dataset_root=per_cfg_dataset_root,
                    model=embedder,
                    verbose=args.verbose,
                    k=12,
                    replan_horizon=args.replan_horizon,
                    video_frame_subsample=args.video_frame_subsample,
                    num_future_frames=args.num_future_frames,
                    robot=args.robot,
                )
            else:
                planner.reset()
            subset_result_dir = os.path.join(args.result_dir, f"config_{i}_{split}")

            # Skip if already evaluated (enables restarts & distributed writes)
            metrics_file = os.path.join(subset_result_dir, "episode_metrics.json")
            if os.path.exists(metrics_file):
                print(f"Skipping {cfg_name} ({split}) — already evaluated")
                continue

            if os.path.exists(subset_result_dir):
                shutil.rmtree(subset_result_dir)
            os.makedirs(subset_result_dir)

            # Build environment for this subset + split
            scene_cfg = dict(
                robot_init_qpos_noise=0.0,
                cube_size_noise=0.0,
                cfg_name=cfg_name,
                mode=split,
            )
            env_cfg = get_env_cfg(
                num_env=args.num_env,
                max_steps=MAX_EPISODE_STEP_WORKSPACE,
                obs_mode="rgb+segmentation",
                scene_cfg_to_overwrite=scene_cfg,
                robot_uids="my_panda_wristcam"
                if args.robot == "panda"
                else "xarm6_robotiq_wristcam",
            )
            envs = build_endless_env(
                env_cfg, record_video=False, data_record_dir="test"
            )

            # Wrap planner for rollout interface
            def wrapped_policy(obs):
                obs_subset = {key: v for key, v in obs.items()}
                action_to_take = planner.act(obs_subset)
                return action_to_take, action_to_take, dict(use_expert_action=0)

            # Rollout
            print("\n" + "=" * 60)
            print(f"Starting Rollout for {cfg_name} ({split})")
            print("=" * 60)

            start_time = time.perf_counter()
            performance = rollout(
                envs,
                wrapped_policy,
                round_to_collect=args.eval_rounds,
                demo_saving_dir=subset_result_dir,
                indices_to_save=[i for i in range(12)] if args.save_videos else [],
            )
            elapsed = time.perf_counter() - start_time

            print("\n" + "=" * 60)
            print(f"Performance for {cfg_name} ({split}) — {elapsed:.1f}s")
            print("=" * 60)
            for key, v in performance.items():
                print(f"  {key}: {v}")

            envs.unwrapped.close()

    # If running all subsets (single-machine mode), aggregate automatically
    if args.start_subset == 0 and args.end_subset == 24:
        print("\nAll subsets evaluated. Running aggregation...")
        aggregate_results(args.result_dir)

    print("Done.")


if __name__ == "__main__":
    main()
