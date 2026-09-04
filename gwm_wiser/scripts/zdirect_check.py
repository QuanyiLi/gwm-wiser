"""
z-direct sanity checks (run inside the smoke job, one GPU):
  1. compute_pooled_targets == the planner's inference path
     (encode_video_to_latent -> embed_video_latent -> pooling_video_latent on
     raw PIL frames) == embedder.process(video) for the same windows.
  2. A z-direct checkpoint loads strictly into ZDirectPlanner and
     get_video_embedding returns a unit 4096-d vector for a retrieved trajectory.
"""

import argparse
import os

import torch
import torch.nn.functional as F

from gwm_wiser import PROJECT_ROOT
from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLEmbedder, Qwen3VLPreprocessor
from gwm_wiser.utils.gwm_data import (
    PaddedLeRobotDataset,
    compute_pooled_targets,
    tensor_images_to_pil,
)
from gwm_wiser.utils.lerobot import image_1_key, obs_state_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help=".../wiser_dataset (merged_test is used)")
    ap.add_argument("--embedder_model_path", required=True)
    ap.add_argument("--ckpt", default=None, help="z-direct checkpoint to load (optional)")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    embedder = Qwen3VLEmbedder(args.embedder_model_path, torch_dtype=torch.bfloat16)
    preprocessor = Qwen3VLPreprocessor(args.embedder_model_path)
    ds = PaddedLeRobotDataset(
        repo_id="unused",
        root=os.path.join(args.dataset_root, "merged_test"),
        video_frame_subsample=6,
        num_future_frames=60,
        preprocess_qwen=True,
        preprocessor=preprocessor,
    )
    idxs = [int(i) for i in torch.linspace(0, len(ds) - 1, args.n)]
    samples = [ds[i] for i in idxs]
    batch = torch.utils.data.default_collate(samples)
    cur, z_train = compute_pooled_targets(embedder, batch)
    print(f"condition latent {tuple(cur.shape)} {cur.dtype}, target {tuple(z_train.shape)} {z_train.dtype}")
    print("target norms:", [round(float(v), 4) for v in z_train.float().norm(dim=-1)])

    worst_cos, worst_abs = 1.0, 0.0
    for j, i in enumerate(idxs):
        raw = ds._getitem(i)
        pil = tensor_images_to_pil(raw[image_1_key])
        inputs = [{"video": pil}]
        with torch.no_grad():
            vl, dsk = embedder.encode_video_to_latent(inputs)
            out = embedder.embed_video_latent(vl, dsk, inputs)
            z_planner = embedder.pooling_video_latent(out)[0]
            z_process = embedder.process(inputs)[0]
        for name, z_ref in (("planner-path", z_planner), ("process()", z_process)):
            c = F.cosine_similarity(z_train[j].float(), z_ref.float(), dim=0).item()
            a = (z_train[j].float() - z_ref.float()).abs().max().item()
            worst_cos, worst_abs = min(worst_cos, c), max(worst_abs, a)
            print(f"window {i}: target vs {name}: cos={c:.6f} max|diff|={a:.2e}")
    print(f"CHECK1 worst cos={worst_cos:.6f} worst max|diff|={worst_abs:.2e}")
    assert worst_cos > 0.999, "training target does not match the planner path"

    if args.ckpt:
        from gwm_wiser.planner.zdirect import ZDirectPlanner

        planner = ZDirectPlanner(
            dataset_root=os.path.join(PROJECT_ROOT, "gwm_skills/config_0_train/lerobot_data"),
            embedder=embedder,
            gwm_checkpoint_path=args.ckpt,
            k=12,
            num_future_frames=60,
            replan_horizon=20,
            video_frame_subsample=6,
            device="cuda",
            dtype=torch.bfloat16,
        )
        state = planner.retriever.get_state_at_index(0)
        trajs = planner.retriever.retrieve(state, k=2)
        raw = ds._getitem(idxs[0])
        current_frame = tensor_images_to_pil(raw[image_1_key][:1])[0]
        z_hat = planner.get_video_embedding(trajs[0], current_frame)
        print(f"CHECK2 z_hat {tuple(z_hat.shape)} {z_hat.dtype} norm={z_hat.float().norm():.4f} "
              f"cos(z_hat, target[0])={F.cosine_similarity(z_hat.float(), z_train[0].float(), dim=0).item():.4f}")
        assert z_hat.shape == (4096,)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
