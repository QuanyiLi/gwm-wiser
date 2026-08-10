"""Train the GWM on the Molmo corpora (Stage 1, plan of record).

Consumes the normalized rendered tree (rendered.py) — the single training-side
data contract for MolmoAct2-DROID and MolmoBot (decision D-18) — with frozen
Qwen online embedding, MSE objective with cosine logging, Muon+AuxAdam, bf16,
step-granular checkpoint/resume, and canonical fixed-1620 exports (ADR-0007).
Open-loop development metrics run on the deterministic episode-level held-out
split (decision D-10); WISER and VRS are fully retired.

Local smoke example (RTX 3090; reduced GWM because the full 4096-dim training
state plus the frozen embedder exceeds 24 GB):

    python -m real_data_train.train \\
        --data_root real_data_train/data \\
        --output_dir runs/smoke \\
        --total_steps 200 --save_every 100 \\
        --model_dim 512 --model_ffn_dim 1024 --model_n_layer 2

Cluster full-size run: keep the model_* defaults (identical to gwm_train.py)
and launch with torchrun for DDP (slurm/submit_gwm_molmo.run).
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
from real_data_train.gwm_model import VariableLenGWM, export_canonical
from real_data_train.sampling import epoch_permutation, sample_position
from real_data_train.windows import DEFAULT_TOLERANCE_S


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    # data
    p.add_argument("--data_root", required=True,
                   help="folder holding rendered/ plus the source trees")
    p.add_argument("--sources", nargs="+", default=None,
                   help="restrict to these sources (default: all rendered)")
    p.add_argument("--manifest", default=None,
                   help="audit manifest JSON; omitted -> the audit runs "
                        "automatically at startup and is written into "
                        "output_dir, so a single command works on slurm")
    p.add_argument("--discovery_cache", default=None,
                   help="clip discovery cache JSON: loaded when the file "
                        "exists, else written after the scan (rank 0). "
                        "Regenerate after any re-render.")
    p.add_argument("--stride_s", type=json.loads, default=None,
                   help='JSON per-source anchor stride override, e.g. '
                        '\'{"molmobot": 3.0}\'')
    p.add_argument("--tolerance_s", type=float, default=None)
    p.add_argument("--jitter_prob", type=float, default=0.5)
    p.add_argument("--time_scale", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="global schedule time-scale range, sampled "
                        "log-uniformly per training sample (D-30); omitted = "
                        "the per-source defaults (D-33: DROID 0.5-1.5, "
                        "molmobot 1-3); '1 1' disables")
    p.add_argument("--eval_scale_sweep", type=float, nargs="*",
                   default=[0.5, 1.5],
                   help="extra fixed-scale held-out evals (robustness "
                        "dashboard); pass nothing to disable")
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--token_ceiling", type=int, default=2048,
                   help="fail-fast ceiling on concatenated visual tokens; 0 disables")
    p.add_argument("--holdout_permille", type=int, default=20)
    p.add_argument("--dataset_subsample_ratio", type=int, default=1,
                   help="keep every N-th window")
    p.add_argument("--limit_clips", type=int, default=None)
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
    p.add_argument("--wandb_project", default="gwm_molmo")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None,
                   help="defaults to the output_dir name")
    # output / held-out development evaluation
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_every", type=int, default=1000,
                   help="0 disables the held-out open-loop evaluation")
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
                        f"mixed Qwen grids in one batch ({key}/{name}: "
                        f"{shapes}); the exact-grid policy (D-2) admits one "
                        "operating grid — run the audit"
                    )
                tensors[name] = torch.stack(values)
        out[key] = tensors
    out["video_id"] = [s.get("video_id", "") for s in samples]
    out["tokens"] = [s["tokens"] for s in samples]
    return out


class TokenCeilingDataset(torch.utils.data.Dataset):
    """Wraps the window dataset with the fail-fast token-ceiling check."""

    def __init__(self, dataset, token_ceiling, pixel_budget):
        self.dataset = dataset
        self.token_ceiling = token_ceiling
        self.pixel_budget = pixel_budget

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        from real_data_train.qwen_rat import count_visual_tokens

        sample = self.dataset[i]
        tokens = count_visual_tokens(sample["qwen_trajectory_gt"])
        if self.token_ceiling and tokens > self.token_ceiling:
            grid = sample["qwen_trajectory_gt"]["video_grid_thw"].reshape(-1).tolist()
            raise RuntimeError(
                f"token ceiling exceeded: video={sample.get('video_id', f'<sample {i}>')} "
                f"pixel_budget={self.pixel_budget} grid={grid} tokens={tokens} "
                f"> ceiling={self.token_ceiling}; raise --token_ceiling to accept"
            )
        # Ship only what training consumes. The raw RAT tensors (rgb,
        # robot_only, condition, target — ~63 MB/sample) otherwise ride the
        # worker->main dataloader queues and OOM the node: 4 ranks x
        # 8 workers x 2 prefetched batches of 32 hit the 200 GB cgroup cap
        # (observed MaxRSS 209.7e6 K on job 4047278).
        return {
            "qwen_current_inputs": sample["qwen_current_inputs"],
            "qwen_trajectory_gt": sample["qwen_trajectory_gt"],
            "video_id": sample.get("video_id", ""),
            "tokens": tokens,
        }


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


@torch.no_grad()
def evaluate_heldout(model, embedder, loader, max_batches=None):
    """Open-loop MSE/cosine on the episode-level held-out split."""
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
    from real_data_train.qwen_rat import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS

    if is_main:
        logging.info(f"Loading Qwen3VL Embedder from {args.embedder_model_path}")
    embedder = Qwen3VLEmbedder(args.embedder_model_path, torch_dtype=torch.bfloat16)
    embedder.model.eval()
    embedder.model.requires_grad_(False)
    preprocessor = Qwen3VLPreprocessor(args.embedder_model_path)

    # ---- audit (source-agnostic, over the rendered tree) ----
    min_pixels = DEFAULT_MIN_PIXELS if args.min_pixels is None else args.min_pixels
    max_pixels = DEFAULT_MAX_PIXELS if args.max_pixels is None else args.max_pixels
    output_dir = Path(args.output_dir)

    # One discovery scan per process, shared by the audit and every dataset
    # split — the scan reads every clip's meta.json, O(all clips) on GPFS.
    from real_data_train.rendered import (
        DEFAULT_SCALE_RANGES,
        RenderedWindowDataset,
        discover_rendered_clips,
        load_discovery_cache,
        save_discovery_cache,
    )

    t_scan = time.time()
    cache = Path(args.discovery_cache) if args.discovery_cache else None
    all_clips, how = None, "scan"
    if cache and cache.is_file():
        all_clips = load_discovery_cache(cache, args.data_root, args.sources)
        if all_clips is None:
            if is_main:
                logging.info("discovery cache is stale; rescanning")
        else:
            how = f"cache {cache.name}"
    if all_clips is None:
        # The cache must hold the UNFILTERED tree so a later run with a
        # different --sources selection stays correct.
        unfiltered = discover_rendered_clips(args.data_root, None)
        if cache and is_main:
            save_discovery_cache(cache, unfiltered, args.data_root)
        all_clips = ([c for c in unfiltered if c.source in args.sources]
                     if args.sources else unfiltered)
    if is_main:
        logging.info(
            f"discovered {len(all_clips)} rendered clips via {how} "
            f"in {time.time() - t_scan:.0f}s"
        )

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
    else:
        from real_data_train.audit import build_manifest, qwen_token_counter

        if is_main:
            logging.info("no --manifest given; running the audit now")
        manifest = build_manifest(
            data_root=args.data_root,
            sources=args.sources,
            token_counter=qwen_token_counter(preprocessor, min_pixels, max_pixels),
            stride_s=args.stride_s,
            tolerance_s=(DEFAULT_TOLERANCE_S if args.tolerance_s is None
                         else args.tolerance_s),
            token_ceiling=args.token_ceiling,
            holdout_permille=args.holdout_permille,
            pixel_budget={"min_pixels": min_pixels, "max_pixels": max_pixels},
            limit_clips=args.limit_clips,
            clips=all_clips,
        )
        if is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "audit_manifest.json").write_text(
                json.dumps(manifest, indent=1)
            )
            logging.info(
                f"audit: {manifest['totals']} shapes={manifest['batch_shapes']}"
            )
    if manifest["off_grid_violations"]:
        raise SystemExit(
            "clips off the operating grid (exact-grid policy D-2):\n"
            + json.dumps(manifest["off_grid_violations"][:5], indent=1)
        )
    if manifest["token_ceiling_violations"]:
        raise SystemExit(
            "token ceiling exceeded (raise --token_ceiling to accept):\n"
            + json.dumps(manifest["token_ceiling_violations"][:5], indent=1)
        )
    manifest_hash = manifest["manifest_hash"]

    # ---- dataset ----
    def build_split(split, jitter, scale_range=None, anchor_jitter_s=None):
        return RenderedWindowDataset(
            args.data_root,
            sources=args.sources,
            split=split,
            stride_s=args.stride_s,
            tolerance_s=args.tolerance_s,
            jitter_prob=jitter,
            scale_range=scale_range,
            anchor_jitter_s=anchor_jitter_s,
            preprocessor=preprocessor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            holdout_permille=args.holdout_permille,
            limit_clips=args.limit_clips,
            limit_windows=args.limit_windows,
            clips=all_clips,
        )

    if args.time_scale is None:
        train_scale = DEFAULT_SCALE_RANGES     # per-source (D-33)
    elif tuple(args.time_scale) == (1.0, 1.0):
        train_scale = None
    else:
        train_scale = tuple(args.time_scale)   # global override
    base_dataset = build_split(
        "train", args.jitter_prob,
        scale_range=train_scale,
    )
    if is_main:
        logging.info(
            f"dataset: {len(base_dataset)} train anchors from "
            f"{len(base_dataset.clips)} clips (time-scale "
            f"{'off' if train_scale is None else train_scale})"
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

    # ---- held-out open-loop loaders (episode-level split, jitter off) ----
    # Canonical scale 1 is THE comparable metric; the fixed-scale sweep
    # loaders are the time-scale robustness dashboard (D-30).
    def heldout_loader(scale_range=None, anchor_jitter_s=None):
        ds = TokenCeilingDataset(
            build_split("heldout", 0.0, scale_range=scale_range,
                        anchor_jitter_s=anchor_jitter_s),
            token_ceiling=args.token_ceiling,
            pixel_budget={"min_pixels": min_pixels, "max_pixels": max_pixels},
        )
        if len(ds) == 0:
            return None
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size,
            num_workers=args.num_workers, collate_fn=qwen_collate,
            shuffle=False, pin_memory=True,
        )

    dev_loader, sweep_loaders = None, {}
    if args.eval_every:
        dev_loader = heldout_loader()
        if dev_loader is None:
            if is_main:
                logging.warning("held-out split is empty; evaluation disabled")
        else:
            if is_main:
                logging.info(
                    f"held-out: {len(dev_loader.dataset)} windows")
            for s in args.eval_scale_sweep or []:
                if s == 1.0:
                    continue
                loader = heldout_loader(scale_range=(s, s), anchor_jitter_s={})
                if loader is not None:
                    sweep_loaders[s] = loader

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(
                {**vars(args), "stride_s": args.stride_s,
                 "manifest_hash": manifest_hash,
                 "world_size": world_size,
                 "pixel_budget": {"min_pixels": min_pixels,
                                  "max_pixels": max_pixels}},
                indent=1, default=str,
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
                dev_mse, dev_cos = evaluate_heldout(
                    model, embedder, dev_loader, args.eval_batches
                )
                sweep = {
                    s: evaluate_heldout(model, embedder, loader,
                                        args.eval_batches)
                    for s, loader in sweep_loaders.items()
                }
                if is_main:
                    extra = "".join(
                        f" | s={s:g} mse={m:.5f} cos={c:.4f}"
                        for s, (m, c) in sorted(sweep.items())
                    )
                    logging.info(
                        f"[held-out open-loop] step {step} "
                        f"mse={dev_mse:.5f} cos={dev_cos:.4f}{extra}"
                    )
                    if wandb_logger:
                        wandb_logger.log_dict(
                            {"mse": dev_mse, "cos_sim": dev_cos,
                             **{f"mse_s{s:g}": m
                                for s, (m, c) in sweep.items()},
                             **{f"cos_sim_s{s:g}": c
                                for s, (m, c) in sweep.items()}},
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
