import logging
import os
from dataclasses import dataclass

import torch.nn.functional as F
from lerobot.configs.train import TrainPipelineConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from lerobot.configs import parser
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import get_step_checkpoint_dir
from lerobot.utils.utils import format_big_number, init_logging
from termcolor import colored

from gwm_wiser.models.gwm import GroundedWorldModel, ActionConditionedGWM
from gwm_wiser.models.transformer import TransformerConfig
from gwm_wiser.utils.lerobot import preprocess_cfg, WandBLogger
from gwm_wiser.utils.gwm_data import (
    AugmentedPaddedDataset,
    compute_embeddings_sequentially,
    PaddedLeRobotDataset,
    ActionConditionedPaddedDataset,
    compute_embeddings_ac,
)
from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLEmbedder, Qwen3VLPreprocessor
from gwm_wiser.utils.muon import MuonWithAuxAdam
from torch.utils.data import Subset


@dataclass
class GWMTrainConfig(TrainPipelineConfig):
    total_learning_epoches: int = 10  # epochs per Dagger iteration
    eval_rounds: int = 1
    eval_interval: int = 2
    steps = None

    lr: float = 5e-5
    muon_lr: float = 0.01

    # Model architecture config
    model_dim: int = 4096
    model_ffn_dim: int = 8192
    model_head_dim: int = 128
    model_n_layer: int = 5
    model_n_head: int = 32
    model_n_kv_head: int = 8
    model_token_dropout_p: float = 0.0
    model_attn_dropout_p: float = 0.0
    model_ffn_dropout_p: float = 0.0

    # GWM model settings
    output_dim: int = 4096

    # model
    embedder_model_path: str = "Qwen/Qwen3-VL-Embedding-8B"

    # dataset
    dataset_subsample_ratio: int = 1
    flip_prob: float = 0.5

    use_compile: bool = False

    # Action-conditioned baseline
    action_conditioned: bool = False


@parser.wrap()
def train(cfg: GWMTrainConfig):
    """
    Main function to train the Grounded World Model with on-the-fly embedding.
    """
    cfg = preprocess_cfg(cfg, bypass_validate=True)

    # Initialize DDP
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = rank == 0

    init_logging()

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
        set_seed(cfg.seed + rank)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # -------------------------------------------------------------------------
    # load embedder per process
    # -------------------------------------------------------------------------
    embedder_model_path = getattr(
        cfg, "embedder_model_path", "Qwen/Qwen3-VL-Embedding-8B"
    )  # Default if not in cfg
    if is_main_process:
        logging.info(f"Loading Qwen3VL Embedder from {embedder_model_path}")
    embedder = Qwen3VLEmbedder(embedder_model_path, torch_dtype=torch.bfloat16)
    embedder.model.eval()
    embedder.model.requires_grad_(False)

    preprocessor = Qwen3VLPreprocessor(embedder_model_path)

    # -------------------------------------------------------------------------
    # Load Datasets
    # -------------------------------------------------------------------------
    if is_main_process:
        logging.info("Loading Datasets...")

    # Define paths
    train_root = os.path.join(cfg.dataset.root, "merged_train")
    test_root = os.path.join(cfg.dataset.root, "merged_test")

    # Create datasets
    if cfg.action_conditioned:
        dataset = ActionConditionedPaddedDataset(
            repo_id="unused",
            root=train_root,
            video_frame_subsample=6,
            num_future_frames=60,
            preprocess_qwen=True,
            preprocessor=preprocessor,
            flip_prob=cfg.flip_prob,
        )
        test_dataset = ActionConditionedPaddedDataset(
            repo_id="unused",
            root=test_root,
            video_frame_subsample=6,
            num_future_frames=60,
            preprocess_qwen=True,
            preprocessor=preprocessor,
            flip_prob=0.0,  # no augmentation for test
        )
    else:
        dataset = AugmentedPaddedDataset(
            repo_id="unused",
            root=train_root,
            video_frame_subsample=6,  # Default 6 frames for video
            num_future_frames=60,  # As per usage
            preprocess_qwen=True,
            preprocessor=preprocessor,
            flip_prob=cfg.flip_prob,
        )
        test_dataset = PaddedLeRobotDataset(
            repo_id="unused",
            root=test_root,
            video_frame_subsample=6,
            num_future_frames=60,
            preprocess_qwen=True,
            preprocessor=preprocessor,
        )

    # subsample
    if cfg.dataset_subsample_ratio > 1:
        indices_dataset = list(range(0, len(dataset), cfg.dataset_subsample_ratio))
        indices_test_dataset = list(
            range(0, len(test_dataset), cfg.dataset_subsample_ratio)
        )
        dataset = Subset(dataset, indices_dataset)
        test_dataset = Subset(test_dataset, indices_test_dataset)

    if is_main_process:
        logging.info(f"Train dataset: {len(dataset)} samples")
        logging.info(f"Test dataset: {len(test_dataset)} samples")

    dist.barrier()

    # Create GWM model
    if is_main_process:
        logging.info("Creating Grounded World Model")

    model_config = TransformerConfig(
        dim=cfg.model_dim,
        ffn_dim=cfg.model_ffn_dim,
        head_dim=cfg.model_head_dim,
        n_layer=cfg.model_n_layer,
        n_head=cfg.model_n_head,
        n_kv_head=cfg.model_n_kv_head,
        token_dropout_p=cfg.model_token_dropout_p,
        attn_dropout_p=cfg.model_attn_dropout_p,
        ffn_dropout_p=cfg.model_ffn_dropout_p,
    )
    if cfg.action_conditioned:
        model = ActionConditionedGWM(
            config=model_config,
            output_dim=cfg.output_dim,
        )
    else:
        model = GroundedWorldModel(
            config=model_config,
            output_dim=cfg.output_dim,
        )

    model = model.to(device).to(torch.bfloat16)
    if cfg.use_compile:
        if is_main_process:
            logging.info("Compiling model...")
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Create optimizer
    raw_model = model.module

    hidden_weights = [p for p in raw_model.backbone.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in raw_model.backbone.parameters() if p.ndim < 2]

    # Input and output projections are efficiently the "embed" and "head"
    nonhidden_params = [
        *raw_model.input_proj.parameters(),
        *raw_model.output_proj.parameters(),
    ]
    if cfg.action_conditioned:
        nonhidden_params += [*raw_model.action_proj.parameters(), raw_model.pad_tokens]

    param_groups = [
        dict(params=hidden_weights, use_muon=True, lr=cfg.muon_lr, weight_decay=0.01),
        dict(
            params=hidden_gains_biases + nonhidden_params,
            use_muon=False,
            lr=cfg.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        ),
    ]

    optimizer = MuonWithAuxAdam(param_groups)

    # Cosine scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.total_learning_epoches,
        eta_min=1e-6,
    )

    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())

    if is_main_process:
        logging.info(
            colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}"
        )
        logging.info(
            f"{cfg.total_learning_epoches=} ({format_big_number(cfg.total_learning_epoches)})"
        )
        num_processes = world_size
        effective_bs = cfg.batch_size * num_processes
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}"
        )
        logging.info(
            f"{num_learnable_params=} ({format_big_number(num_learnable_params)})"
        )
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Dataloaders
    train_sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    effective_batch_size = cfg.batch_size * world_size

    # Training loop
    for epoch in range(cfg.total_learning_epoches):
        train_sampler.set_epoch(epoch)
        desc = (
            f"Epoch: {epoch + 1}/{cfg.total_learning_epoches}, {effective_batch_size=}"
        )
        dl = tqdm.tqdm(dataloader, desc=desc) if is_main_process else dataloader
        model.train()

        for batch in dl:
            optimizer.zero_grad()

            # Move batch to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            # Compute embeddings on the fly
            if cfg.action_conditioned:
                current_image_emb, gt_trajectory_emb = compute_embeddings_ac(
                    embedder, batch
                )
            else:
                current_image_emb, gt_trajectory_emb = compute_embeddings_sequentially(
                    embedder, batch
                )

            # Direct Transformer training
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Forward pass
                if cfg.action_conditioned:
                    pred_emb = model(current_image_emb, batch["action"])
                else:
                    pred_emb = model(current_image_emb)

                # mse loss for logging
                mse_loss = F.mse_loss(pred_emb, gt_trajectory_emb)
                # cosine similarity loss
                cos_sim = F.cosine_similarity(
                    pred_emb, gt_trajectory_emb, dim=-1
                ).mean()
                loss = mse_loss

            # Backward
            loss.backward()

            # Gradient clipping
            grad_clip_norm = (
                cfg.optimizer.grad_clip_norm
                if hasattr(cfg.optimizer, "grad_clip_norm")
                else 1.0
            )
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        # Step scheduler
        lr_scheduler.step()

        if is_main_process:
            logging.info(
                f"Epoch {epoch} finished. Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, CosSim: {cos_sim.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            if wandb_logger:
                wandb_logger.log_dict(
                    {
                        "loss": loss.item(),
                        "mse": mse_loss.item(),
                        "cos_sim": cos_sim.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    epoch,
                )

        # Evaluate on test dataset
        if epoch % cfg.eval_interval == 0 and epoch != 0:
            model.eval()
            total_mse_loss = 0.0
            total_cos_sim = 0.0
            num_batches = 0

            with torch.no_grad():
                test_dl = (
                    tqdm.tqdm(test_dataloader, desc="Evaluating")
                    if is_main_process
                    else test_dataloader
                )
                for test_batch in test_dl:
                    # Move batch to device
                    for k in test_batch:
                        if isinstance(test_batch[k], torch.Tensor):
                            test_batch[k] = test_batch[k].to(device)

                    # Compute embeddings on the fly for test batch too
                    if cfg.action_conditioned:
                        current_image_emb, gt_trajectory_emb = compute_embeddings_ac(
                            embedder, test_batch
                        )
                    else:
                        current_image_emb, gt_trajectory_emb = (
                            compute_embeddings_sequentially(embedder, test_batch)
                        )

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        if cfg.action_conditioned:
                            predicted_emb = model(
                                current_image_emb, test_batch["action"]
                            )  # (B, 1620, 4096)
                        else:
                            predicted_emb = model(current_image_emb)  # (B, 1620, 4096)
                        mse_loss = F.mse_loss(predicted_emb, gt_trajectory_emb)
                        cos_sim = F.cosine_similarity(
                            predicted_emb, gt_trajectory_emb, dim=-1
                        ).mean()

                    total_mse_loss += mse_loss.item()
                    total_cos_sim += cos_sim.item()

                    num_batches += 1
                    # break

            avg_mse_loss = (
                total_mse_loss / num_batches if num_batches > 0 else float("inf")
            )
            avg_cos_sim = total_cos_sim / num_batches if num_batches > 0 else 0.0

            # Gather metrics from all processes
            metrics_tensor = torch.tensor([avg_mse_loss, avg_cos_sim], device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            avg_mse_loss = metrics_tensor[0].item() / world_size
            avg_cos_sim = metrics_tensor[1].item() / world_size

            if is_main_process:
                logging.info(
                    f"Epoch {epoch}: Test MSE = {avg_mse_loss:.4f}, Test CosSim = {avg_cos_sim:.4f}"
                )
                if wandb_logger:
                    wandb_logger.log_dict(
                        {
                            "test_mse": avg_mse_loss,
                            "test_cos_sim": avg_cos_sim,
                        },
                        epoch,
                        mode="eval",
                    )

                # Save checkpoint always
                logging.info(
                    f"Saving checkpoint for epoch {epoch} (MSE: {avg_mse_loss:.4f}, CosSim: {avg_cos_sim:.4f})"
                )
                checkpoint_dir = get_step_checkpoint_dir(
                    cfg.output_dir, cfg.steps, epoch
                )

                # Save model state
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": lr_scheduler.state_dict(),
                        "test_mse": avg_mse_loss,
                        "test_cos_sim": avg_cos_sim,
                        "config": model_config,
                        "action_conditioned": cfg.action_conditioned,
                    },
                    os.path.join(checkpoint_dir, "checkpoint.pt"),
                )

        dist.barrier()

    if is_main_process:
        logging.info("End of training")

    dist.barrier()
    dist.destroy_process_group()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
