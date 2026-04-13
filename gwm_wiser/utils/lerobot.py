import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from accelerate import Accelerator
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import resolve_delta_timestamps, make_dataset
from lerobot.envs.configs import EnvConfig

try:
    from lerobot.policies.pretrained import PreTrainedPolicy
except (ImportError, ModuleNotFoundError):
    PreTrainedPolicy = None  # Not available; UniVLA eval does not need it
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, REWARD
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.logging_utils import MetricsTracker
from lerobot.utils.utils import has_method
from torch.optim import Optimizer

from gwm_wiser.utils.helpers import batch_tensor_to_string
from gwm_wiser.utils.helpers import current_time_str

# follow LeRobot 0.4.3
obs_image_prefix = OBS_IMAGES
obs_state_key = OBS_STATE
reward_key = REWARD
action_key = ACTION
image_1_key = f"{obs_image_prefix}.image_1"
image_1_robot_state = f"{obs_image_prefix}.image_1_robot_state"
image_1_segmentation_mask_key = f"{obs_image_prefix}.image_1_segmentation_mask"
image_2_key = f"{obs_image_prefix}.image_2"
wrist_image_key = f"{obs_image_prefix}.wrist_image"
task_key = "task"


class WandBLogger:
    """Custom WandB logger for GWM training."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity if hasattr(cfg.wandb, "entity") else None,
            name=cfg.job_name,
            config=self._cfg_to_dict(cfg),
            dir=cfg.output_dir,
            resume="allow",
        )

    def _cfg_to_dict(self, cfg) -> dict:
        """Convert config to a dictionary for wandb."""
        if hasattr(cfg, "__dict__"):
            result = {}
            for key, value in cfg.__dict__.items():
                if not key.startswith("_"):
                    try:
                        if hasattr(value, "__dict__"):
                            result[key] = self._cfg_to_dict(value)
                        else:
                            result[key] = value
                    except Exception:
                        result[key] = str(value)
            return result
        return {"config": str(cfg)}

    def log_dict(self, data: dict, step: int, mode: str = "train"):
        """Log a dictionary of metrics to wandb."""
        prefixed_data = {f"{mode}/{k}": v for k, v in data.items()}
        prefixed_data["epoch"] = step
        wandb.log(prefixed_data, step=step)

    def finish(self):
        """Finish the wandb run."""
        wandb.finish()


@dataclass
class TrainConfig(TrainPipelineConfig):
    total_learning_epoches: int = 10000  # epochs per Dagger iteration
    eval_rounds: int = 1
    eval_interval: int = 5
    steps = None
    pad_wrist_image: bool = False  # pad wrist_image (224x224) to match image_1 width (224x448), needed for GR00T


@dataclass
class EvalConfig(TrainConfig):
    start_subset: int = 0  # start config index (inclusive)
    end_subset: int = 24  # end config index (exclusive)
    result_dir: str = "./lerobot_eval_result"
    split: str = "both"  # "train", "test", or "both"
    aggregate_only: bool = False  # skip rollouts, only compute final aggregation


class DummyEncConfig(EnvConfig):
    """
    A placehorlder for now
    """

    @property
    def type(self):
        return {}

    def gym_kwargs(self):
        return dict()


def pad_images_to_match(batch_or_obs):
    """
    Pad wrist_image width to match image_1 width.
    Works on tensors in (B, C, H, W) format (both dataset batches and processed env obs).
    Pads with zeros on the right side of the width dimension.
    """
    if wrist_image_key not in batch_or_obs or image_1_key not in batch_or_obs:
        return batch_or_obs
    target_w = batch_or_obs[image_1_key].shape[-1]  # width is last dim in (B, C, H, W)
    current_w = batch_or_obs[wrist_image_key].shape[-1]
    if current_w < target_w:
        pad_w = target_w - current_w
        # F.pad format for 4D: (left, right, top, bottom)
        batch_or_obs[wrist_image_key] = torch.nn.functional.pad(
            batch_or_obs[wrist_image_key], (0, pad_w, 0, 0), mode="constant", value=0
        )
    return batch_or_obs


def from_env_obs_to_model(cfg, obs, env_preprocessor, preprocessor, ds_meta):
    """
    The first part of this function actually did the env_preprocessor like LiberoProcessorStep().
    Consider refactoring later.
    """
    obs = {k: obs[k] for k in obs.keys()}
    obs["task"] = batch_tensor_to_string(obs["task"])
    for k in obs.keys():
        if (
            isinstance(obs[k], torch.Tensor) and obs[k].dtype == torch.uint8
        ):  # image, norm to [0, 1]
            assert obs[k].ndim == 4
            obs[k] = obs[k].float() / 255.0
            obs[k] = obs[k].permute(0, 3, 1, 2)  # to (B, C, H, W)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    obs[ACTION] = torch.zeros(
        (len(obs["task"]), len(delta_timestamps[ACTION]), 8),
        device=obs[obs_state_key].device,
    )  # dummy action required by model...
    obs.pop(image_1_robot_state)
    obs.pop(image_1_segmentation_mask_key)

    # pad wrist image if configured (for GR00T)
    if getattr(cfg, "pad_wrist_image", False):
        obs = pad_images_to_match(obs)

    # model preprocess
    if env_preprocessor is not None:
        obs = env_preprocessor(obs)
    obs = preprocessor(obs)
    return obs


def lerobot_policy(
    accelerator,
    cfg,
    ds_meta,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
):
    """
    Adapted from LeRobot_eval.py/rollout function.
    """
    policy.eval()
    policy.reset()

    def forward(obs):
        eval_obs = from_env_obs_to_model(
            cfg, obs, env_preprocessor, preprocessor, ds_meta
        )

        # forward
        if accelerator is not None:
            with torch.inference_mode(), accelerator.autocast():
                action = policy.select_action(eval_obs)
        else:
            with torch.inference_mode():
                action = policy.select_action(eval_obs)
        action = action[:, :8]

        # post-process
        action = postprocessor(action)
        action_transition = {"action": action}
        if env_postprocessor is not None:
            action_transition = env_postprocessor(action_transition)
        action = action_transition["action"]
        action = action.to(torch.float32).to(obs.device)
        return action, action, dict()

    return forward


def _to_loggable(x):
    if torch.is_tensor(x):
        x = x.detach()
        # turn big tensors into a scalar; customize as you like
        return x.float().mean().cpu().item()
    return x


def preprocess_cfg(cfg, bypass_validate=False):
    exp_name = cfg.job_name + f"-{current_time_str()}"
    cfg.output_dir = Path(os.path.join(cfg.output_dir, exp_name))

    # lerobot check
    if not bypass_validate:
        cfg.validate()

    # ours check
    assert cfg.eval_freq > 0, "You must do evaluation by setting eval_freq > 0!"
    # assert cfg.policy.n_action_steps <= 10, "Don't use n_action_steps > 10 for the task"
    return cfg


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    output_dict = {k: _to_loggable(v) for k, v in (output_dict or {}).items()}

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def load_merged_datasets(cfg):
    """
    Load pre-merged train and test datasets from cfg.dataset.root.

    Expects cfg.dataset.root to be the parent directory (e.g., 24_dataset_merged)
    containing merged_train and merged_test subdirectories.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    original_root = cfg.dataset.root

    train_root = os.path.join(original_root, "merged_train")
    cfg.dataset.root = train_root
    train_dataset = make_dataset(cfg)

    test_root = os.path.join(original_root, "merged_test")
    cfg.dataset.root = test_root
    test_dataset = make_dataset(cfg)

    cfg.dataset.root = original_root  # restore

    return train_dataset, test_dataset


def exclude_keys(ds_meta):
    for key in [image_1_segmentation_mask_key, image_1_robot_state]:
        ds_meta.camera_keys.remove(key)
        ds_meta.features.pop(key)
    return ds_meta


def exclude_from_obs(obs):
    for key in [image_1_segmentation_mask_key, image_1_robot_state]:
        obs.pop(key)
    return obs
