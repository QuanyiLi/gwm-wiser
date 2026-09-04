"""
z-direct training (advisor question C012): predict the pooled 4096-d clip
embedding z_t = B(p_t) directly instead of the 1620 x 4096 token latent.

Same corpus (merged_train / merged_test), condition clip, augmentation,
747M transformer, Muon+Adam optimiser, cosine LR schedule, epochs and global
batch (32 x 12 = 384, here 32 x world_size x grad_accum_steps) as
gwm_train.py. Only the head (mean-pool + linear, see PooledGWM), the target
(frozen readout of the real future clip, L2-normalised) and the loss change:

    --zdirect_loss mse : MSE between the un-normalised head output and the unit target
    --zdirect_loss cos : 1 - cos between the normalised head output and the unit target

Held-out cosine(predicted, true pooled vector) is logged to W&B (and
<output_dir>/metrics.jsonl) every --eval_every_steps optimiser steps on a fixed
subset of merged_test, and on the full merged_test at the end.

Launch (per node, 4 GPUs):
    torchrun --nproc_per_node=4 gwm_wiser/scripts/zdirect_train.py \
        --dataset.repo_id=unused --dataset.root=wiser_dataset \
        --output_dir=/work/.../gwm_zdirect --job_name=zdirect_mse \
        --zdirect_loss=mse --grad_accum_steps=3 --total_learning_epoches=3 ...
"""

import json
import logging
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import format_big_number, init_logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from gwm_wiser.models.gwm import PooledGWM
from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLEmbedder, Qwen3VLPreprocessor
from gwm_wiser.models.transformer import TransformerConfig
from gwm_wiser.utils.gwm_data import (
    AugmentedPaddedDataset,
    PaddedLeRobotDataset,
    compute_pooled_targets,
)
from gwm_wiser.utils.lerobot import WandBLogger
from gwm_wiser.utils.muon import MuonWithAuxAdam


@dataclass
class ZDirectTrainConfig(TrainPipelineConfig):
    total_learning_epoches: int = 3
    steps = None

    lr: float = 5e-5
    muon_lr: float = 0.01

    # Model architecture config (identical to gwm_train.py defaults)
    model_dim: int = 4096
    model_ffn_dim: int = 8192
    model_head_dim: int = 128
    model_n_layer: int = 5
    model_n_head: int = 32
    model_n_kv_head: int = 8
    model_token_dropout_p: float = 0.0
    model_attn_dropout_p: float = 0.0
    model_ffn_dropout_p: float = 0.0
    output_dim: int = 4096

    embedder_model_path: str = "Qwen/Qwen3-VL-Embedding-8B"
    dataset_subsample_ratio: int = 1
    flip_prob: float = 0.5
    use_compile: bool = False

    # z-direct specifics
    zdirect_loss: str = "mse"  # "mse" | "cos"
    grad_accum_steps: int = 3  # 32 x 4 GPUs x 3 = 384 = the GWM global batch
    log_every_steps: int = 20
    eval_every_steps: int = 100
    eval_subset_size: int = 1024
    full_eval_at_end: bool = True
    max_steps: int = 0  # smoke tests: stop after this many optimiser steps


def zdirect_loss_and_stats(pred: torch.Tensor, target: torch.Tensor, kind: str):
    pred = pred.float()
    target = target.float()
    cos = F.cosine_similarity(pred, target, dim=-1)
    mse_raw = F.mse_loss(pred, target)
    mse_unit = F.mse_loss(F.normalize(pred, dim=-1), target)
    if kind == "mse":
        loss = mse_raw
    elif kind == "cos":
        loss = (1.0 - cos).mean()
    else:
        raise ValueError(f"unknown zdirect_loss {kind!r}")
    stats = {
        "cos": cos.mean().item(),
        "mse_raw": mse_raw.item(),
        "mse_unit": mse_unit.item(),
        "pred_norm": pred.norm(dim=-1).mean().item(),
    }
    return loss, stats


@torch.no_grad()
def evaluate(model, embedder, loader, device, kind, desc, is_main):
    """Per-sample sums all-reduced across ranks -> exact means over the split."""
    model.eval()
    sums = torch.zeros(4, device=device)  # cos, mse_raw, mse_unit, count
    it = tqdm.tqdm(loader, desc=desc, leave=False) if is_main else loader
    for batch in it:
        cur, z = compute_pooled_targets(embedder, batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(cur)
        pred = pred.float()
        z = z.float()
        n = pred.shape[0]
        sums[0] += F.cosine_similarity(pred, z, dim=-1).sum()
        sums[1] += ((pred - z) ** 2).mean(dim=-1).sum()
        sums[2] += ((F.normalize(pred, dim=-1) - z) ** 2).mean(dim=-1).sum()
        sums[3] += n
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    model.train()
    count = max(sums[3].item(), 1.0)
    return {
        "cos": sums[0].item() / count,
        "mse_raw": sums[1].item() / count,
        "mse_unit": sums[2].item() / count,
        "n": int(sums[3].item()),
    }


def git_commit(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


@parser.wrap()
def train(cfg: ZDirectTrainConfig):
    assert cfg.zdirect_loss in ("mse", "cos"), cfg.zdirect_loss
    cfg.output_dir = Path(cfg.output_dir) / cfg.job_name

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    init_logging()
    if is_main:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = None
    if cfg.wandb.enable and cfg.wandb.project and is_main:
        wandb_logger = WandBLogger(cfg)
    if cfg.seed is not None:
        set_seed(cfg.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- frozen embedder (condition encoder + target readout) ----
    if is_main:
        logging.info(f"Loading Qwen3VL Embedder from {cfg.embedder_model_path}")
    embedder = Qwen3VLEmbedder(cfg.embedder_model_path, torch_dtype=torch.bfloat16)
    embedder.model.eval()
    embedder.model.requires_grad_(False)
    preprocessor = Qwen3VLPreprocessor(cfg.embedder_model_path)

    # ---- datasets (identical to gwm_train.py) ----
    train_root = os.path.join(cfg.dataset.root, "merged_train")
    test_root = os.path.join(cfg.dataset.root, "merged_test")
    dataset = AugmentedPaddedDataset(
        repo_id="unused",
        root=train_root,
        video_frame_subsample=6,
        num_future_frames=60,
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
    if cfg.dataset_subsample_ratio > 1:
        dataset = Subset(dataset, list(range(0, len(dataset), cfg.dataset_subsample_ratio)))
        test_dataset = Subset(
            test_dataset, list(range(0, len(test_dataset), cfg.dataset_subsample_ratio))
        )
    # fixed held-out subset for the periodic curve
    subset_idx = np.linspace(
        0, len(test_dataset) - 1, min(cfg.eval_subset_size, len(test_dataset))
    ).astype(int).tolist()
    eval_subset = Subset(test_dataset, subset_idx)
    if is_main:
        logging.info(
            f"Train dataset: {len(dataset)} samples, test dataset: {len(test_dataset)} "
            f"samples, periodic-eval subset: {len(eval_subset)} samples"
        )
    dist.barrier()

    # ---- model ----
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
    model = PooledGWM(config=model_config, output_dim=cfg.output_dim)
    model = model.to(device).to(torch.bfloat16)
    if cfg.use_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module

    hidden_weights = [p for p in raw_model.backbone.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in raw_model.backbone.parameters() if p.ndim < 2]
    nonhidden_params = [
        *raw_model.input_proj.parameters(),
        *raw_model.output_proj.parameters(),
    ]
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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.total_learning_epoches, eta_min=1e-6
    )
    grad_clip_norm = (
        cfg.optimizer.grad_clip_norm
        if cfg.optimizer is not None and hasattr(cfg.optimizer, "grad_clip_norm")
        else 1.0
    )

    num_params = sum(p.numel() for p in model.parameters())
    effective_bs = cfg.batch_size * world_size * cfg.grad_accum_steps
    if is_main:
        logging.info(f"Output dir: {cfg.output_dir}")
        logging.info(f"PooledGWM params: {num_params} ({format_big_number(num_params)})")
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {world_size} ranks x "
            f"{cfg.grad_accum_steps} accum = {effective_bs}; loss = {cfg.zdirect_loss}"
        )
        run_config = {
            "cfg": {k: str(v) for k, v in asdict(cfg).items()},
            "world_size": world_size,
            "effective_batch_size": effective_bs,
            "num_params": num_params,
            "git_commit": git_commit(os.path.dirname(os.path.abspath(__file__))),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        json.dump(run_config, open(cfg.output_dir / "run_config.json", "w"), indent=1)

    # ---- loaders ----
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    def make_eval_loader(ds):
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
        return torch.utils.data.DataLoader(
            ds,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

    subset_loader = make_eval_loader(eval_subset)
    metrics_path = cfg.output_dir / "metrics.jsonl"

    def log_point(data: dict, step: int, epoch: int):
        if not is_main:
            return
        data = dict(data, step=step, epoch=epoch, time=time.time())
        with open(metrics_path, "a") as f:
            f.write(json.dumps(data) + "\n")
        if wandb_logger is not None:
            import wandb

            wandb.log(data, step=step)

    def run_subset_eval(step, epoch):
        m = evaluate(model, embedder, subset_loader, device, cfg.zdirect_loss,
                     f"eval subset @ step {step}", is_main)
        if is_main:
            logging.info(
                f"[eval subset] step {step}: cos={m['cos']:.4f} mse_raw={m['mse_raw']:.5f} "
                f"mse_unit={m['mse_unit']:.5f} (n={m['n']})"
            )
        log_point({f"eval/subset_{k}": v for k, v in m.items()}, step, epoch)
        return m

    def save_checkpoint(epoch, step, extra):
        if not is_main:
            return
        ckpt_dir = cfg.output_dir / f"epoch_{epoch + 1:02d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "config": model_config,
                "zdirect": True,
                "zdirect_loss": cfg.zdirect_loss,
                "effective_batch_size": effective_bs,
                **extra,
            },
            ckpt_dir / "checkpoint.pt",
        )
        last = cfg.output_dir / "last"
        if last.is_symlink() or last.exists():
            last.unlink()
        last.symlink_to(ckpt_dir.name)
        logging.info(f"Saved checkpoint -> {ckpt_dir} (last -> {ckpt_dir.name})")

    # ---- training ----
    global_step = 0
    stop = False
    run_subset_eval(0, 0)  # random-init baseline point of the curve
    t_log, s_log = time.time(), 0
    stats = {}
    for epoch in range(cfg.total_learning_epoches):
        train_sampler.set_epoch(epoch)
        model.train()
        n_micro = len(dataloader)
        dl = (
            tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.total_learning_epoches}")
            if is_main
            else dataloader
        )
        optimizer.zero_grad(set_to_none=True)
        for micro_idx, batch in enumerate(dl):
            is_last_micro = (micro_idx + 1) % cfg.grad_accum_steps == 0 or (
                micro_idx + 1 == n_micro
            )
            cur_emb, target = compute_pooled_targets(embedder, batch)
            ctx = nullcontext() if is_last_micro else model.no_sync()
            with ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(cur_emb)
                loss, stats = zdirect_loss_and_stats(pred, target, cfg.zdirect_loss)
                (loss / cfg.grad_accum_steps).backward()
            if not is_last_micro:
                continue

            if grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.tensor(0.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % cfg.log_every_steps == 0:
                dt = time.time() - t_log
                sps = (global_step - s_log) / dt if dt > 0 else 0.0
                t_log, s_log = time.time(), global_step
                point = {
                    "train/loss": loss.item(),
                    "train/cos": stats["cos"],
                    "train/mse_raw": stats["mse_raw"],
                    "train/mse_unit": stats["mse_unit"],
                    "train/pred_norm": stats["pred_norm"],
                    "train/grad_norm": float(grad_norm),
                    "train/lr_muon": optimizer.param_groups[0]["lr"],
                    "train/lr_adam": optimizer.param_groups[1]["lr"],
                    "train/steps_per_s": sps,
                }
                if is_main:
                    logging.info(
                        f"step {global_step} (epoch {epoch + 1}, micro {micro_idx + 1}/{n_micro}): "
                        f"loss={loss.item():.5f} cos={stats['cos']:.4f} "
                        f"mse_raw={stats['mse_raw']:.5f} norm={stats['pred_norm']:.3f} "
                        f"steps/s={sps:.3f}"
                    )
                log_point(point, global_step, epoch)
            if cfg.eval_every_steps > 0 and global_step % cfg.eval_every_steps == 0:
                run_subset_eval(global_step, epoch)
            if cfg.max_steps and global_step >= cfg.max_steps:
                stop = True
                break

        lr_scheduler.step()
        m = run_subset_eval(global_step, epoch)
        if is_main:
            logging.info(
                f"Epoch {epoch + 1} finished at step {global_step}: last loss={loss.item():.5f}, "
                f"subset cos={m['cos']:.4f}"
            )
        save_checkpoint(epoch, global_step, {"subset_cos": m["cos"], "subset_mse_unit": m["mse_unit"]})
        dist.barrier()
        if stop:
            break

    if cfg.full_eval_at_end:
        full = evaluate(model, embedder, make_eval_loader(test_dataset), device,
                        cfg.zdirect_loss, "eval full merged_test", is_main)
        if is_main:
            logging.info(
                f"[eval FULL merged_test] step {global_step}: cos={full['cos']:.4f} "
                f"mse_raw={full['mse_raw']:.5f} mse_unit={full['mse_unit']:.5f} (n={full['n']})"
            )
            json.dump(full, open(cfg.output_dir / "final_heldout.json", "w"), indent=1)
        log_point({f"eval/full_{k}": v for k, v in full.items()}, global_step, cfg.total_learning_epoches)
    if is_main:
        logging.info("End of training")
        if wandb_logger is not None:
            wandb_logger.finish()
    dist.barrier()
    dist.destroy_process_group()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
