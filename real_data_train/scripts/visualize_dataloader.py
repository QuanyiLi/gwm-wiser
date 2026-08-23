"""Contact sheets from EXACT training samples, for human data-supply checks.

Draws windows from the same RenderedWindowDataset training consumes (same
window enumeration, same split logic, augmentation switchable) and writes one
PNG per sample to <out>/:

    row 1  full RGB target frames t0..t5      (the semantic outcome)
    row 2  robot-only condition frames        (state-rendered appearance)
    row 3  0.55 * RGB + 0.45 * robot-only     (alignment check: the rendered
                                              robot must sit on the observed
                                              robot in every frame)

Timestamps and the clip id are stamped in the strip header.

    python -m real_data_train.scripts.visualize_dataloader \\
        --data_root real_data_train/data --out viz/ [--num 12] [--split all]
"""

import argparse
import random
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "viz")
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--split", default="all",
                   choices=["train", "heldout", "all"])
    p.add_argument("--num", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter_prob", type=float, default=0.0,
                   help="set to the training value to preview augmentation")
    p.add_argument("--time_scale", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="preview the schedule time-scale augmentation "
                        "e.g. 0.5 1.5")
    return p.parse_args(argv)


def to_u8(t):
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def main(argv=None):
    from PIL import Image, ImageDraw

    from real_data_train.rendered import RenderedWindowDataset

    args = parse_args(argv)
    ds = RenderedWindowDataset(
        args.data_root, sources=args.sources, split=args.split,
        jitter_prob=args.jitter_prob,
        scale_range=tuple(args.time_scale) if args.time_scale else None,
    )
    if len(ds) == 0:
        raise SystemExit("no windows — run setup_data / render_actions first")
    print(f"{len(ds)} windows from {len(ds.clips)} clips ({args.split})")

    rng = random.Random(args.seed)
    picks = rng.sample(range(len(ds)), min(args.num, len(ds)))
    args.out.mkdir(parents=True, exist_ok=True)

    for i in picks:
        ci, _ = ds.index[i]
        clip = ds.clips[ci]
        sample = ds[i]
        indices = sample["frame_indices"]   # actual drawn window
        rgb = [to_u8(f) for f in sample["target"]]
        robot = [to_u8(f) for f in sample["robot_only"]]
        overlay = [
            (0.55 * a + 0.45 * b).astype(np.uint8) for a, b in zip(rgb, robot)
        ]
        rows = [np.concatenate(r, axis=1) for r in (rgb, robot, overlay)]
        sheet = np.concatenate(rows, axis=0)

        header = 22
        img = Image.new("RGB", (sheet.shape[1], sheet.shape[0] + header),
                        (16, 16, 16))
        img.paste(Image.fromarray(sheet), (0, header))
        ts = [clip.timestamps[k] for k in indices]
        from real_data_train.rendered import is_heldout

        ImageDraw.Draw(img).text(
            (4, 4),
            f"{clip.source}/{clip.clip_id}  frames={indices}  "
            f"t={['%.2f' % t for t in ts]}  "
            f"scale={sample.get('time_scale', 1.0):.2f}  "
            f"heldout={is_heldout(clip.episode_uid)}",
            fill=(240, 240, 90),
        )
        name = f"{clip.source}__{clip.clip_id}__w{i:05d}.png"
        img.save(args.out / name)
        print("wrote", args.out / name)


if __name__ == "__main__":
    main()
