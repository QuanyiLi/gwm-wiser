"""Human-inspection renderings of the exact RAT samples consumed by training.

Uses the same adapter, window selection, masking, and configured transforms
as training (no separate preprocessing path). For each sampled window, writes
one PNG contact sheet with rows:

  1. full RGB (six selected frames, ordinal indices in the title row)
  2. full RGB with the whole-robot mask overlaid (provenance in the title)
  3. robot-only RGB on black
  4. full RGB after photometric augmentation (robot-only row is unchanged)
  5. RAT condition (current full frame + future robot-only frames)
  6. full-RGB target

Usage:
    python -m real_world_gwm.adapters.vrs.visualize \\
        --roots /root/data/vrs/test --out viz/ --num_windows 8 \\
        [--frame_step 1] [--window_stride 1] [--jitter_prob 1.0] [--seed 0]
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from real_world_gwm.adapters.vrs.dataset import (
    VRSWindowDataset,
    build_rat_pair,
)
from real_world_gwm.augment import augment_window

ROW_LABELS = [
    "full RGB",
    "mask overlay (002 whole robot)",
    "robot-only on black",
    "full RGB after color jitter (robot-only unchanged)",
    "RAT condition",
    "full-RGB target",
]


def _to_pil(frame: torch.Tensor) -> Image.Image:
    arr = (frame.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np.transpose(arr, (1, 2, 0)))


def _overlay(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    color = torch.tensor([1.0, 0.2, 0.2]).view(3, 1, 1)
    return rgb * (1 - 0.5 * mask) + color * 0.5 * mask


def render_window(ds: VRSWindowDataset, index: int, jitter_prob: float) -> Image.Image:
    ci, indices = ds.index[index]
    clip = ds.clips[ci]
    from real_world_gwm.adapters.vrs.dataset import load_window

    raw = load_window(clip, indices)
    aug = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
    aug = augment_window(aug, flip_prob=0.0, jitter_prob=jitter_prob)
    condition, target = build_rat_pair(aug["rgb"], aug["robot_only"])

    rows = [
        raw["rgb"],
        _overlay(raw["rgb"], raw["mask"]),
        raw["robot_only"],
        aug["rgb"],
        condition,
        target,
    ]

    _, _, h, w = raw["rgb"].shape
    pad, header = 4, 44
    sheet = Image.new(
        "RGB", (6 * (w + pad) + pad, len(rows) * (h + pad) + pad + header), "white"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 2),
        f"{clip.video_id}\n"
        f"frames(ordinal)={indices}  mask_provenance={clip.mask_provenance}  "
        f"main_camera=image (no auxiliary views)  temporal=ordinal (no clock)",
        fill="black",
    )
    for r, frames in enumerate(rows):
        y = header + pad + r * (h + pad)
        for t in range(6):
            sheet.paste(_to_pil(frames[t]), (pad + t * (w + pad), y))
    footer = "  |  ".join(f"row{r + 1}: {label}" for r, label in enumerate(ROW_LABELS))
    return sheet, footer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_windows", type=int, default=8)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--jitter_prob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = VRSWindowDataset(
        args.roots, frame_step=args.frame_step, window_stride=args.window_stride
    )
    print(f"{len(ds)} windows from {len(ds.clips)} clips; excluded: {len(ds.excluded)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(ds)), min(args.num_windows, len(ds)))
    for i in picks:
        sheet, footer = render_window(ds, i, args.jitter_prob)
        ci, indices = ds.index[i]
        name = f"{ds.clips[ci].video_id}__w{indices[0]:05d}.png"
        sheet.save(out / name)
        print(f"wrote {out / name}")
    print(footer)


if __name__ == "__main__":
    main()
