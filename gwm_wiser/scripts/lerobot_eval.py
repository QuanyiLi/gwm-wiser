"""
Evaluate a trained LeRobot model across config subsets [start_subset, end_subset) on train/test splits.

Launched the same way as lerobot_train.py via @parser.wrap() with EvalConfig:

    python lerobot_eval.py \
        --start_subset 0 --end_subset 24 --split both \
        --result_dir ./lerobot_eval_result \
        --dataset.repo_id=unused \
        --dataset.root=<path> \
        --policy.pretrained_path=<checkpoint> \
        ...
"""

import copy
import json
import logging
import os
import shutil
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SVT_LOG"] = "0"

import datasets
import torch
from lerobot.configs import parser
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
from lerobot.policies.wall_x.processor_wall_x import make_wall_x_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins

datasets.disable_progress_bar()

from gwm_wiser.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE_EVAL  # noqa: E402
from gwm_wiser.utils.env import build_endless_env  # noqa: E402
from gwm_wiser.utils.lerobot import (  # noqa: E402
    EvalConfig,
    lerobot_policy,
    preprocess_cfg,
    load_merged_datasets,
    exclude_keys,
)
from gwm_wiser.utils.rollout import rollout, calculate_averages  # noqa: E402


@parser.wrap()
def run_eval(cfg: EvalConfig) -> None:
    result_dir = cfg.result_dir
    if cfg.aggregate_only:
        _aggregate_results(result_dir)
        return

    cfg = preprocess_cfg(cfg)
    os.makedirs(result_dir, exist_ok=True)

    # Device setup
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load dataset (needed for meta/stats even during eval)
    logging.info("Loading dataset for meta information...")
    dataset, _ = load_merged_datasets(cfg)

    # Build policy
    logging.info("Creating policy...")
    policy_ds_meta = exclude_keys(copy.deepcopy(dataset.meta))
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=policy_ds_meta,
        rename_map=cfg.rename_map,
    )

    # Create processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (
        cfg.policy.pretrained_path and not cfg.resume
    ) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
            "device_processor": {"device": device.type},
        }

    if isinstance(cfg.policy, WallXConfig):
        # Wall-X uses a custom processor factory that doesn't rely on the
        # ProcessorStepRegistry, avoiding the missing 'wall_x_task_processor' error.
        preprocessor, postprocessor = make_wall_x_pre_post_processors(
            config=cfg.policy,
            dataset_stats=dataset.meta.stats,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            **processor_kwargs,
            **postprocessor_kwargs,
        )

    # Determine splits to evaluate
    splits = ["train", "test"] if cfg.split == "both" else [cfg.split]

    logging.info(
        f"Evaluating subsets {cfg.start_subset} to {cfg.end_subset - 1} "
        f"on splits: {splits}"
    )

    # Evaluation loop
    for split in splits:
        for i in range(cfg.start_subset, cfg.end_subset):
            cfg_name = f"config_{i}"
            subset_result_dir = os.path.join(result_dir, f"config_{i}_{split}")

            # Skip if already evaluated
            metrics_file = os.path.join(subset_result_dir, "episode_metrics.json")
            if os.path.exists(metrics_file):
                logging.info(f"Skipping {cfg_name} ({split}) — already evaluated")
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
                num_env=12,
                max_steps=MAX_EPISODE_STEP_WORKSPACE_EVAL,
                obs_mode="rgb+segmentation",
                scene_cfg_to_overwrite=scene_cfg,
            )
            envs = build_endless_env(
                env_cfg, record_video=False, data_record_dir="test"
            )

            # Wrap policy for rollout
            wrapped = lerobot_policy(
                accelerator=None,
                cfg=cfg,
                ds_meta=dataset.meta,
                policy=policy,
                env_preprocessor=None,
                env_postprocessor=None,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )

            # Rollout
            print("\n" + "=" * 60)
            print(f"Starting Rollout for {cfg_name} ({split})")
            print("=" * 60)

            start_time = time.perf_counter()
            with torch.no_grad():
                performance = rollout(
                    envs,
                    wrapped,
                    round_to_collect=cfg.eval_rounds,
                    demo_saving_dir=subset_result_dir,
                    indices_to_save=[],
                )
            elapsed = time.perf_counter() - start_time

            print("\n" + "=" * 60)
            print(f"Performance for {cfg_name} ({split}) — {elapsed:.1f}s")
            print("=" * 60)
            for key, v in performance.items():
                print(f"  {key}: {v}")

            envs.unwrapped.close()

    logging.info("All subsets evaluated. Done.")


def _aggregate_results(result_dir):
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
    register_third_party_plugins()
    run_eval()


if __name__ == "__main__":
    main()
