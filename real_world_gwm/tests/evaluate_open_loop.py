"""WISER-dev open-loop evaluation of a saved canonical checkpoint.

Development evaluation only (never a held-out estimate): reads
``<wiser_dev_dataset_root>/merged_test`` through the unchanged WISER data
path, computes token-level MSE and cosine similarity, and never backpropagates.
Requires lerobot (cluster environment).

The same metrics are computed online during training via
``train.py --wiser_dev_dataset_root``; this standalone script exists to
inspect an already-saved canonical checkpoint.

Usage:
    python -m real_world_gwm.tests.evaluate_open_loop \\
        --checkpoint runs/vrs1/step_0001000/checkpoint.pt \\
        --wiser_dev_dataset_root /path/to/wiser_dataset
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wiser_dev_dataset_root", required=True)
    parser.add_argument("--embedder_model_path", default="Qwen/Qwen3-VL-Embedding-8B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    from gwm_wiser.models.qwen3_vl_embedding import (
        Qwen3VLEmbedder,
        Qwen3VLPreprocessor,
    )
    from gwm_wiser.utils.gwm_data import compute_embeddings_sequentially
    from real_world_gwm.gwm_model import load_canonical_like_planner
    from real_world_gwm.train import build_wiser_dev_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_canonical_like_planner(args.checkpoint)
    model = model.to(device)
    model.eval()

    embedder = Qwen3VLEmbedder(args.embedder_model_path, torch_dtype=torch.bfloat16)
    embedder.model.eval()
    embedder.model.requires_grad_(False)
    preprocessor = Qwen3VLPreprocessor(args.embedder_model_path)

    args.wiser_dev_dataset_root = args.wiser_dev_dataset_root  # used by builder
    loader = build_wiser_dev_loader(args, preprocessor)

    mses, coss = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            cur, traj = compute_embeddings_sequentially(embedder, batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(cur)
            mses.append(F.mse_loss(pred, traj).item())
            coss.append(F.cosine_similarity(pred, traj, dim=-1).mean().item())

    result = {
        "checkpoint": args.checkpoint,
        "step": checkpoint.get("step"),
        "open_loop_mse": float(np.mean(mses)),
        "open_loop_cosine": float(np.mean(coss)),
        "batches": len(mses),
    }
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
