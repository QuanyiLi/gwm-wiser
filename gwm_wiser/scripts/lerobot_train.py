import logging
import os
import copy
import os.path
import os.path as osp
import shutil
import time
from pprint import pformat

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true"
import datasets
import torch
import tqdm
from accelerate import Accelerator
from lerobot.configs import parser
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    init_logging,
)
from termcolor import colored
from gwm_wiser.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE_EVAL
from gwm_wiser.utils.env import build_endless_env
from gwm_wiser.utils.lerobot import lerobot_policy, TrainConfig, exclude_keys
from gwm_wiser.utils.lerobot import update_policy, load_merged_datasets
from gwm_wiser.utils.lerobot import (
    preprocess_cfg,
    exclude_from_obs,
    pad_images_to_match,
)
from gwm_wiser.utils.rollout import rollout


@parser.wrap()
def train(cfg: TrainConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `DaggerTrainConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg = preprocess_cfg(cfg)

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs]
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        os.environ["SVT_LOG"] = "0"
        datasets.disable_progress_bar()
        logging.info(pformat(cfg.to_dict()))
        logging.info("config keys: steps, log_freq, save_freq will be deprecated")

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(
                colored("Logs will be saved locally.", "yellow", attrs=["bold"])
            )

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset, test_dataset = load_merged_datasets(cfg)
        assert test_dataset, "No test dataset found"

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    # NOTE: all dataset stats and normalization will use the one calculated from the first BC dataset!
    if not is_main_process:
        dataset, test_dataset = load_merged_datasets(cfg)

    # all dataset will be stored in a list and do concatenation
    accelerator.wait_for_everyone()

    # env is used for evaluation
    if is_main_process:
        logging.info("Creating environment")
    train_env = None
    test_env = None
    if is_main_process:
        train_env_cfg = get_env_cfg(
            num_env=12,
            max_steps=MAX_EPISODE_STEP_WORKSPACE_EVAL,
            obs_mode="rgb+segmentation",
        )
        train_env = build_endless_env(
            env_cfg=train_env_cfg,
            data_record_dir=osp.join(cfg.output_dir, "video"),
            record_video=False,
        )

        test_env_cfg = get_env_cfg(
            num_env=12,
            max_steps=MAX_EPISODE_STEP_WORKSPACE_EVAL,
            obs_mode="rgb+segmentation",
            scene_cfg_to_overwrite=dict(mode="test"),
        )
        test_env = build_endless_env(
            env_cfg=test_env_cfg,
            data_record_dir=osp.join(cfg.output_dir, "video_test"),
            record_video=False,
        )

    # # Wait for all processes to finish policy creation before continuing
    # accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("Creating policy")
    policy_ds_meta = exclude_keys(copy.deepcopy(dataset.meta))
    policy = make_policy(
        cfg=cfg.policy, ds_meta=policy_ds_meta, rename_map=cfg.rename_map
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (
        cfg.policy.pretrained_path and not cfg.resume
    ) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
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

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    if cfg.resume:
        _, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(
            colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}"
        )
        logging.info("Creating Dummy Environment Processors for evaluation")
        env_preprocessor, env_postprocessor = None, None
        logging.info(
            f"{cfg.total_learning_epoches=} ({format_big_number(cfg.total_learning_epoches)}"
        )
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}"
        )
        logging.info(
            f"{num_learnable_params=} ({format_big_number(num_learnable_params)})"
        )
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # metrics meters
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    ckpt_metrics_to_check = ["return_mean", "success_at_end_mean", "tcp_near_goal_mean"]
    eval_iter = 0
    best_checkpoints = {}

    assert not hasattr(cfg.policy, "drop_n_last_frames")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=0,
        accelerator=accelerator,
    )

    # iterate the whole dataset for cfg.total_learning_epoches times
    for epoch in range(cfg.total_learning_epoches):
        desc = (
            f"Epoch: {epoch + 1}/{cfg.total_learning_epoches}, {effective_batch_size=}"
        )
        dl = tqdm.tqdm(dataloader, desc=desc) if is_main_process else dataloader
        policy.train()
        for batch in dl:
            batch = exclude_from_obs(batch)
            if cfg.pad_wrist_image:
                batch = pad_images_to_match(batch)
            start_time = time.perf_counter()
            batch = preprocessor(batch)
            train_tracker.dataloading_s = time.perf_counter() - start_time

            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler,
            )

            # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
            # increment `step` here.
            train_tracker.step()
            # break

        logging.info(train_tracker)
        if wandb_logger:
            wandb_log_dict = train_tracker.to_dict()
            if output_dict:
                wandb_log_dict.update(output_dict)
            wandb_logger.log_dict(wandb_log_dict, epoch)
        train_tracker.reset_averages()

        if is_main_process and (epoch + 1) % cfg.eval_interval == 0:
            # eval
            logging.info(f"Eval policy at epoch {epoch}")
            wrapped_policy = lerobot_policy(
                accelerator,
                cfg=cfg,
                ds_meta=dataset.meta,
                policy=accelerator.unwrap_model(policy),
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            eval_info_train = rollout(
                train_env, wrapped_policy, round_to_collect=cfg.eval_rounds
            )
            eval_info_train = {f"train_env/{k}": v for k, v in eval_info_train.items()}
            for k, v in eval_info_train.items():
                logging.info(f"{k}: {v}")
            if wandb_logger:
                wandb_logger.log_dict(eval_info_train, epoch, mode="eval")

            # eval on test env
            eval_info_test = rollout(
                test_env, wrapped_policy, round_to_collect=cfg.eval_rounds
            )
            eval_info_test = {f"test_env/{k}": v for k, v in eval_info_test.items()}
            for k, v in eval_info_test.items():
                logging.info(f"{k}: {v}")
            if wandb_logger:
                wandb_logger.log_dict(eval_info_test, epoch, mode="eval")

            # need checkpoint?
            eval_iter += 1
            metrics = {}
            for k in ckpt_metrics_to_check:
                metrics[f"train_{k}"] = eval_info_train[f"train_env/{k}"]
                metrics[f"test_{k}"] = eval_info_test[f"test_env/{k}"]

            save_reasons = []
            for k, v in metrics.items():
                if k not in best_checkpoints:
                    best_checkpoints[k] = {
                        "val": -float("inf"),
                        "epoch": -1,
                        "path": None,
                    }

                if v > best_checkpoints[k]["val"] - 1e-4:
                    save_reasons.append(k)

            # save checkpoint
            if cfg.save_checkpoint and len(save_reasons) > 0:
                logging.info(
                    f"Checkpoint policy after step {epoch} due to {save_reasons}"
                )
                checkpoint_dir = get_step_checkpoint_dir(
                    cfg.output_dir, cfg.steps, epoch
                )
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=epoch,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )

                # Update records and delete old checkpoints
                for reason in save_reasons:
                    old_path = best_checkpoints[reason]["path"]
                    best_checkpoints[reason]["val"] = metrics[reason]
                    best_checkpoints[reason]["epoch"] = epoch
                    best_checkpoints[reason]["path"] = checkpoint_dir

                    if old_path and old_path != checkpoint_dir:
                        # Check if old_path is still the best for any other metric
                        is_still_needed = False
                        for other_k, other_info in best_checkpoints.items():
                            if other_k != reason and other_info["path"] == old_path:
                                is_still_needed = True
                                break

                        if not is_still_needed and osp.exists(old_path):
                            logging.info(f"Deleting old checkpoint: {old_path}")
                            shutil.rmtree(old_path)

        accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("End of training")
        train_env.unwrapped.close()
        test_env.unwrapped.close()

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
