"""Train the GWM on out-of-domain robot video (phase one, VRS adapter).

Mirrors gwm_wiser/scripts/gwm_train.py — frozen Qwen online embedding, MSE
objective with cosine logging, Muon+AuxAdam, bf16 — but consumes the
source-neutral RAT samples produced by a real_world_gwm source adapter,
schedules at optimizer-step granularity, and exports canonical fixed-1620
checkpoints (ADR-0007). The WISER trainer and its configuration are untouched.

Local smoke example (RTX 3090; reduced GWM because the full 4096-dim training
state plus the frozen embedder exceeds 24 GB):

    python -m real_world_gwm.train \\
        --dataset_roots /root/data/vrs/test \\
        --manifest audit_manifest.json \\
        --output_dir runs/smoke \\
        --limit_videos 4 --total_steps 20 --save_every 10 \\
        --model_dim 512 --model_ffn_dim 1024 --model_n_layer 2

Cluster full-size run: keep the model_* defaults (identical to gwm_train.py)
and launch with torchrun for DDP.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from gwm_wiser.models.transformer import TransformerConfig
from gwm_wiser.utils.gwm_data import compute_embeddings_sequentially
from real_world_gwm.gwm_model import VariableLenGWM, export_canonical
from real_world_gwm.sampling import epoch_permutation, sample_position


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    # data
    p.add_argument("--dataset_adapter", default="vrs", choices=["vrs", "wiser"],
                   help="'wiser' trains on <root>/merged_train through the "
                        "unchanged WISER dataset class — a pipeline-debug "
                        "path to reproduce gwm_train.py behavior; its "
                        "checkpoints are WISER-contaminated by definition")
    p.add_argument("--dataset_subsample_ratio", type=int, default=1,
                   help="keep every N-th sample (same semantics as gwm_train.py)")
    p.add_argument("--dataset_roots", nargs="+", required=True)
    p.add_argument("--manifest", default=None,
                   help="audit manifest JSON; omitted -> the audit runs "
                        "automatically at startup (motion stats skipped) and "
                        "is written into output_dir, so a single command works "
                        "on slurm")
    p.add_argument("--frame_step", type=int, default=1)
    p.add_argument("--window_stride", type=int, default=1)
    p.add_argument("--flip_prob", type=float, default=0.5)
    p.add_argument("--jitter_prob", type=float, default=0.5)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--token_ceiling", type=int, default=2048,
                   help="fail-fast ceiling on concatenated visual tokens; 0 disables")
    p.add_argument("--limit_videos", type=int, default=None)
    p.add_argument("--limit_windows", type=int, default=None)
    # schedule
    p.add_argument("--total_steps", type=int, required=True)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="path to a train_state.pt")
    p.add_argument("--overfit_one_batch", action="store_true")
    # optimization (defaults identical to gwm_train.py)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--muon_lr", type=float, default=0.01)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    # model (defaults identical to gwm_train.py)
    p.add_argument("--model_dim", type=int, default=4096)
    p.add_argument("--model_ffn_dim", type=int, default=8192)
    p.add_argument("--model_head_dim", type=int, default=128)
    p.add_argument("--model_n_layer", type=int, default=5)
    p.add_argument("--model_n_head", type=int, default=32)
    p.add_argument("--model_n_kv_head", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=4096)
    p.add_argument("--use_compile", action="store_true")
    p.add_argument("--embedder_model_path", default="Qwen/Qwen3-VL-Embedding-8B")
    # logging
    p.add_argument("--wandb_enable", action="store_true")
    p.add_argument("--wandb_project", default="gwm_vrs")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None,
                   help="defaults to the output_dir name")
    # output / optional development evaluation
    p.add_argument("--output_dir", required=True)
    p.add_argument("--wiser_dev_dataset_root", default=None,
                   help="optional WISER merged_test root for open-loop dev metrics")
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_batches", type=int, default=None)
    return p.parse_args(argv)


def init_distributed():
    """Single-process group when not under torchrun (Muon requires one)."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def qwen_collate(samples):
    """Collate only what training needs; fail clearly on mixed grids."""
    keys = ("qwen_current_inputs", "qwen_trajectory_gt")
    out = {}
    for key in keys:
        tensors = {}
        for name in samples[0][key]:
            values = [s[key][name] for s in samples]
            if isinstance(values[0], torch.Tensor):
                shapes = {tuple(v.shape) for v in values}
                if len(shapes) > 1:
                    raise RuntimeError(
                        f"mixed Qwen grids in one batch ({key}/{name}: {shapes}); "
                        "the audit found multiple batch shapes, so use "
                        "--batch_size 1 (padding/bucketing is deliberately "
                        "deferred by the plan)"
                    )
                tensors[name] = torch.stack(values)
        out[key] = tensors
    out["video_id"] = [s.get("video_id", "") for s in samples]
    out["tokens"] = [s["tokens"] for s in samples]
    return out


class TokenCeilingDataset(torch.utils.data.Dataset):
    """Wraps the adapter dataset with the fail-fast token-ceiling check."""

    def __init__(self, dataset, token_ceiling, pixel_budget):
        self.dataset = dataset
        self.token_ceiling = token_ceiling
        self.pixel_budget = pixel_budget

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        from real_world_gwm.qwen_rat import count_visual_tokens

        sample = self.dataset[i]
        tokens = count_visual_tokens(sample["qwen_trajectory_gt"])
        if self.token_ceiling and tokens > self.token_ceiling:
            grid = sample["qwen_trajectory_gt"]["video_grid_thw"].reshape(-1).tolist()
            raise RuntimeError(
                f"token ceiling exceeded: video={sample.get('video_id', f'<sample {i}>')} "
                f"pixel_budget={self.pixel_budget} grid={grid} tokens={tokens} "
                f"> ceiling={self.token_ceiling}; raise --token_ceiling to accept"
            )
        sample["tokens"] = tokens
        return sample


def save_train_state(path, model, optimizer, scheduler, step, args):
    raw = getattr(model, "_orig_mod", model)
    raw = getattr(raw, "module", raw)
    torch.save(
        {
            "step": step,
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "args": vars(args),
        },
        path,
    )


def restore_rng(rng):
    torch.set_rng_state(rng["torch"].cpu() if isinstance(rng["torch"], torch.Tensor) else rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def build_wiser_dev_loader(args, preprocessor):
    """Optional WISER-dev open-loop data (requires lerobot; cluster only)."""
    from gwm_wiser.utils.gwm_data import PaddedLeRobotDataset  # lazy: needs lerobot

    dataset = PaddedLeRobotDataset(
        repo_id="unused",
        root=os.path.join(args.wiser_dev_dataset_root, "merged_test"),
        video_frame_subsample=6,
        num_future_frames=60,
        preprocess_qwen=True,
        preprocessor=preprocessor,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True,
    )


@torch.no_grad()
def evaluate_wiser_dev(model, embedder, loader, device, max_batches=None):
    from gwm_wiser.utils.gwm_data import compute_embeddings_sequentially

    model.eval()
    mses, coss = [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        cur, traj = compute_embeddings_sequentially(embedder, batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(cur)
            mses.append(F.mse_loss(pred, traj).item())
            coss.append(F.cosine_similarity(pred, traj, dim=-1).mean().item())
    model.train()
    return float(np.mean(mses)), float(np.mean(coss))


def main(argv=None):
    args = parse_args(argv)
    # force=True: importing lerobot (via gwm_data) installs its own logging
    # config, which would otherwise silently drop these INFO lines
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", force=True
    )

    init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- frozen embedder + preprocessor ----
    from gwm_wiser.models.qwen3_vl_embedding import (
        Qwen3VLEmbedder,
        Qwen3VLPreprocessor,
    )
    from real_world_gwm.qwen_rat import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS

    if is_main:
        logging.info(f"Loading Qwen3VL Embedder from {args.embedder_model_path}")
    embedder = Qwen3VLEmbedder(args.embedder_model_path, torch_dtype=torch.bfloat16)
    embedder.model.eval()
    embedder.model.requires_grad_(False)
    preprocessor = Qwen3VLPreprocessor(args.embedder_model_path)

    # ---- audit + dataset (adapter-specific) ----
    min_pixels = DEFAULT_MIN_PIXELS if args.min_pixels is None else args.min_pixels
    max_pixels = DEFAULT_MAX_PIXELS if args.max_pixels is None else args.max_pixels
    output_dir = Path(args.output_dir)

    if args.dataset_adapter == "vrs":
        if args.manifest:
            manifest = json.loads(Path(args.manifest).read_text())
        else:
            from real_world_gwm.audit import build_manifest, qwen_token_counter

            if is_main:
                logging.info("no --manifest given; running the audit now")
            manifest = build_manifest(
                roots=args.dataset_roots,
                frame_step=args.frame_step,
                window_stride=args.window_stride,
                candidate_steps=(args.frame_step,),
                token_counter=qwen_token_counter(
                    preprocessor, min_pixels, max_pixels
                ),
                token_ceiling=args.token_ceiling,
                motion_sample_windows=0,
                pixel_budget={"min_pixels": min_pixels, "max_pixels": max_pixels},
                limit_videos=args.limit_videos,
            )
            if is_main:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "audit_manifest.json").write_text(
                    json.dumps(manifest, indent=1)
                )
                logging.info(
                    f"audit: {manifest['totals']} shapes={manifest['batch_shapes']}"
                )
        if manifest["token_ceiling_violations"]:
            raise SystemExit(
                "token ceiling exceeded (raise --token_ceiling to accept):\n"
                + json.dumps(manifest["token_ceiling_violations"][:5], indent=1)
            )

        from real_world_gwm.adapters.vrs.dataset import VRSWindowDataset

        base_dataset = VRSWindowDataset(
            args.dataset_roots,
            frame_step=args.frame_step,
            window_stride=args.window_stride,
            flip_prob=args.flip_prob,
            jitter_prob=args.jitter_prob,
            preprocessor=preprocessor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            limit_videos=args.limit_videos,
            limit_windows=args.limit_windows,
        )
        if is_main:
            logging.info(
                f"dataset: {len(base_dataset)} windows from "
                f"{len(base_dataset.clips)} clips "
                f"({len(base_dataset.excluded)} excluded)"
            )
    else:  # wiser: pipeline-debug adapter over the unchanged WISER data path
        import hashlib

        from gwm_wiser.utils.gwm_data import AugmentedPaddedDataset

        assert len(args.dataset_roots) == 1, "wiser adapter takes one root"
        train_root = os.path.join(args.dataset_roots[0], "merged_train")
        base_dataset = AugmentedPaddedDataset(
            repo_id="unused",
            root=train_root,
            video_frame_subsample=6,
            num_future_frames=60,
            preprocess_qwen=True,
            preprocessor=preprocessor,
            flip_prob=args.flip_prob,
        )
        # WISER frames share one grid; no VRS-style audit applies here.
        manifest = {
            "source": "wiser (pipeline-debug adapter)",
            "roots": [train_root],
            "batch_shapes": [[3, 18, 30]],
            "token_ceiling_violations": [],
        }
        manifest["manifest_hash"] = "wiser:" + hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        if is_main:
            logging.warning(
                "WISER adapter is for debugging the training pipeline against "
                "gwm_train.py; its checkpoints are WISER-contaminated and are "
                "not phase-one candidates"
            )
            logging.info(f"dataset: {len(base_dataset)} samples from {train_root}")

    manifest_hash = manifest["manifest_hash"]
    if args.batch_size > 1 and len(manifest["batch_shapes"]) > 1:
        raise SystemExit(
            f"audit found {len(manifest['batch_shapes'])} distinct Qwen grids "
            f"{manifest['batch_shapes']}; per the batch-shape policy use "
            "--batch_size 1 (padding/bucketing is deliberately deferred)"
        )

    dataset = TokenCeilingDataset(
        base_dataset,
        token_ceiling=args.token_ceiling,
        pixel_budget={"min_pixels": min_pixels, "max_pixels": max_pixels},
    )
    if args.dataset_subsample_ratio > 1:
        dataset = torch.utils.data.Subset(
            dataset, list(range(0, len(dataset), args.dataset_subsample_ratio))
        )
        if is_main:
            logging.info(f"subsampled to {len(dataset)} samples")
    if len(dataset) == 0:
        raise SystemExit("no valid samples under the configured sampling")

    # ---- model / optimizer / scheduler (as gwm_train.py) ----
    model_config = TransformerConfig(
        dim=args.model_dim,
        ffn_dim=args.model_ffn_dim,
        head_dim=args.model_head_dim,
        n_layer=args.model_n_layer,
        n_head=args.model_n_head,
        n_kv_head=args.model_n_kv_head,
    )
    model = VariableLenGWM(config=model_config, output_dim=args.output_dim)
    model = model.to(device).to(torch.bfloat16)
    if args.use_compile:
        model = torch.compile(model)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
    raw_model = getattr(model, "module", model)
    raw_model = getattr(raw_model, "_orig_mod", raw_model)

    from gwm_wiser.utils.muon import MuonWithAuxAdam

    hidden_weights = [p for p in raw_model.backbone.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in raw_model.backbone.parameters() if p.ndim < 2]
    nonhidden_params = [
        *raw_model.input_proj.parameters(),
        *raw_model.output_proj.parameters(),
    ]
    optimizer = MuonWithAuxAdam(
        [
            dict(params=hidden_weights, use_muon=True, lr=args.muon_lr,
                 weight_decay=0.01),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                 lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01),
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=1e-6
    )

    # ---- resume ----
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng(state["rng"])
        start_step = state["step"]
        if is_main:
            logging.info(f"resumed from {args.resume} at step {start_step}")

    # ---- optional WISER-dev loader ----
    dev_loader = None
    if args.wiser_dev_dataset_root:
        dev_loader = build_wiser_dev_loader(args, preprocessor)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(
                {**vars(args), "manifest_hash": manifest_hash,
                 "world_size": world_size,
                 "pixel_budget": {"min_pixels": min_pixels,
                                  "max_pixels": max_pixels}},
                indent=1,
            )
        )

    metadata = {
        "manifest_hash": manifest_hash,
        "resolved_config": vars(args),
    }

    # ---- wandb (reuses the WISER WandBLogger; rank 0 only) ----
    wandb_logger = None
    if args.wandb_enable and is_main:
        from types import SimpleNamespace

        from gwm_wiser.utils.lerobot import WandBLogger

        wandb_logger = WandBLogger(
            SimpleNamespace(
                **{
                    **vars(args),
                    "wandb": SimpleNamespace(
                        project=args.wandb_project, entity=args.wandb_entity
                    ),
                    "job_name": args.wandb_run_name or output_dir.name,
                    "output_dir": str(output_dir),
                    "manifest_hash": manifest_hash,
                }
            )
        )

    def loader_for_epoch(epoch, offset):
        perm = epoch_permutation(len(dataset), args.seed, epoch)
        # shard the epoch permutation across ranks after applying the offset;
        # truncate so every rank gets the same number of batches (unequal
        # shards would desynchronize DDP allreduce at the epoch boundary)
        indices = perm[offset:]
        if world_size > 1:
            per_rank = len(indices) // world_size
            indices = indices[rank * per_rank : (rank + 1) * per_rank]
        subset = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(
            subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=qwen_collate,
            pin_memory=True,
            drop_last=True,
        )

    def checkpoint(step):
        step_dir = output_dir / f"step_{step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        export_canonical(
            raw_model, step_dir / "checkpoint.pt", config=model_config,
            step=step, metadata=metadata,
        )
        save_train_state(
            step_dir / "train_state.pt", model, optimizer, scheduler, step, args
        )
        if is_main:
            logging.info(f"saved checkpoint at step {step} -> {step_dir}")

    # ---- training loop (step-granular) ----
    model.train()
    step = start_step
    epoch, offset = sample_position(step, len(dataset),
                                    args.batch_size * world_size)
    overfit_batch = None
    t_last, s_last = time.time(), step

    while step < args.total_steps:
        loader = loader_for_epoch(epoch, offset)
        offset = 0
        batches = iter(loader)
        for batch in batches:
            if step >= args.total_steps:
                break
            if args.overfit_one_batch:
                if overfit_batch is None:
                    overfit_batch = batch
                batch = overfit_batch

            optimizer.zero_grad()
            cur_emb, traj_emb = compute_embeddings_sequentially(embedder, batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(cur_emb)
                mse_loss = F.mse_loss(pred, traj_emb)
                cos_sim = F.cosine_similarity(pred, traj_emb, dim=-1).mean()
                loss = mse_loss  # cosine is logged, never optimized
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm
                )
            optimizer.step()
            scheduler.step()
            step += 1

            if is_main and step % args.log_every == 0:
                dt = time.time() - t_last
                sps = (step - s_last) / dt if dt > 0 else 0.0
                logging.info(
                    f"step {step}/{args.total_steps} "
                    f"mse={mse_loss.item():.5f} cos={cos_sim.item():.4f} "
                    f"lr={optimizer.param_groups[-1]['lr']:.2e} "
                    f"tokens={batch['tokens'][0]} "
                    f"steps/s={sps:.2f}"
                )
                if wandb_logger:
                    wandb_logger.log_dict(
                        {
                            "mse": mse_loss.item(),
                            "cos_sim": cos_sim.item(),
                            "lr": optimizer.param_groups[-1]["lr"],
                            "steps_per_s": sps,
                            "tokens": batch["tokens"][0],
                        },
                        step,
                    )
                t_last, s_last = time.time(), step

            if dev_loader is not None and step % args.eval_every == 0:
                dev_mse, dev_cos = evaluate_wiser_dev(
                    model, embedder, dev_loader, device, args.eval_batches
                )
                if is_main:
                    logging.info(
                        f"[wiser-dev open-loop] step {step} "
                        f"mse={dev_mse:.5f} cos={dev_cos:.4f}"
                    )
                    if wandb_logger:
                        wandb_logger.log_dict(
                            {"mse": dev_mse, "cos_sim": dev_cos},
                            step,
                            mode="eval",
                        )

            if is_main and step % args.save_every == 0:
                checkpoint(step)
        epoch += 1

    if is_main:
        checkpoint(step)
        logging.info("End of training")
        if wandb_logger:
            wandb_logger.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
