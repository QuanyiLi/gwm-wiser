"""
Reference points for the z-direct held-out cosine: how well does a *constant*
predictor (the mean pooled target) or a random other window's target already
match a held-out target? Uses the same fixed merged_test subset as the training
script (np.linspace over the dataset, --eval_subset_size windows).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLEmbedder, Qwen3VLPreprocessor
from gwm_wiser.utils.gwm_data import PaddedLeRobotDataset, compute_pooled_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--embedder_model_path", required=True)
    ap.add_argument("--eval_subset_size", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    embedder = Qwen3VLEmbedder(args.embedder_model_path, torch_dtype=torch.bfloat16)
    preprocessor = Qwen3VLPreprocessor(args.embedder_model_path)
    ds = PaddedLeRobotDataset(
        repo_id="unused", root=os.path.join(args.dataset_root, "merged_test"),
        video_frame_subsample=6, num_future_frames=60, preprocess_qwen=True, preprocessor=preprocessor,
    )
    idx = np.linspace(0, len(ds) - 1, min(args.eval_subset_size, len(ds))).astype(int).tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=args.batch_size, num_workers=8)
    targets, conds = [], []
    for batch in loader:
        cur, z = compute_pooled_targets(embedder, batch)
        targets.append(z.float().cpu())
        # pooled readout of the *condition* clip (segmentation video) for reference
        c_in = {k: v for k, v in batch["qwen_current_inputs"].items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            zc = []
            for i in range(cur.shape[0]):
                ci = {k: v[i].unsqueeze(0).to(embedder.model.device) for k, v in c_in.items()}
                vl, dsk = embedder.encode_video_to_latent(**ci)
                zc.append(embedder.pooling_video_latent(embedder.embed_video_latent(vl, dsk, **ci))[0])
        conds.append(torch.stack(zc).float().cpu())
    Z = torch.cat(targets)  # (n, 4096) unit vectors
    C = torch.cat(conds)
    n = Z.shape[0]
    mean = Z.mean(0, keepdim=True)
    loo_mean = (mean * n - Z) / (n - 1)  # leave-one-out mean
    cos_const = F.cosine_similarity(Z, loo_mean.expand_as(Z), dim=-1)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    cos_rand = F.cosine_similarity(Z, Z[perm], dim=-1)
    cos_cond = F.cosine_similarity(Z, C, dim=-1)
    sims = Z @ Z.T
    sims.fill_diagonal_(-1)
    cos_nn = sims.max(1).values
    out = {
        "n": n,
        "cos_constant_mean_predictor": cos_const.mean().item(),
        "cos_random_other_window": cos_rand.mean().item(),
        "cos_target_vs_condition_clip_readout": cos_cond.mean().item(),
        "cos_nearest_other_window": cos_nn.mean().item(),
        "mean_vector_norm": mean.norm().item(),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
