"""Standalone open-loop MSE/cosine for a saved canonical checkpoint.

Evaluates against the episode-level held-out split of the rendered tree (the
same deterministic hash split training uses), or --split all for diagnostics.

    python -m real_data_train.tests.evaluate_open_loop \\
        --checkpoint runs/x/step_0000100/checkpoint.pt \\
        --data_root real_data_train/data [--split heldout] [--max_batches 64]
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--split", default="heldout",
                   choices=["heldout", "train", "all"])
    p.add_argument("--holdout_permille", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--embedder_model_path",
                   default="Qwen/Qwen3-VL-Embedding-8B")
    args = p.parse_args()

    from gwm_wiser.models.qwen3_vl_embedding import (
        Qwen3VLEmbedder,
        Qwen3VLPreprocessor,
    )
    from gwm_wiser.utils.gwm_data import compute_embeddings_sequentially
    from real_data_train.gwm_model import load_canonical_like_planner
    from real_data_train.rendered import RenderedWindowDataset
    from real_data_train.train import qwen_collate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_canonical_like_planner(args.checkpoint)
    model = model.to(device).eval()
    print(f"checkpoint step={checkpoint.get('step')} "
          f"manifest={checkpoint.get('metadata', {}).get('manifest_hash', '?')[:12]}")

    embedder = Qwen3VLEmbedder(args.embedder_model_path,
                               torch_dtype=torch.bfloat16)
    embedder.model.eval()
    preprocessor = Qwen3VLPreprocessor(args.embedder_model_path)

    dataset = RenderedWindowDataset(
        args.data_root, sources=args.sources, split=args.split,
        jitter_prob=0.0, preprocessor=preprocessor,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        holdout_permille=args.holdout_permille,
    )
    print(f"{args.split}: {len(dataset)} windows from {len(dataset.clips)} clips")
    if len(dataset) == 0:
        raise SystemExit("empty split")

    class WithTokens(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, i):
            from real_data_train.qwen_rat import count_visual_tokens

            sample = dataset[i]
            sample["tokens"] = count_visual_tokens(sample["qwen_trajectory_gt"])
            return sample

    loader = torch.utils.data.DataLoader(
        WithTokens(), batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=qwen_collate, shuffle=False,
    )
    mses, coss = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            cur, traj = compute_embeddings_sequentially(embedder, batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(cur)
            mses.append(F.mse_loss(pred.float(), traj.float()).item())
            coss.append(F.cosine_similarity(pred.float(), traj.float(),
                                            dim=-1).mean().item())
    print(f"open-loop {args.split}: mse={np.mean(mses):.5f} "
          f"cos={np.mean(coss):.4f} over {len(mses)} batches")


if __name__ == "__main__":
    main()
