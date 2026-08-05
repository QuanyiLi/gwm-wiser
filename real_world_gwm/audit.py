"""Preprocessing audit: the mandatory machine-readable gate before training.

Usage (exact production token counting requires the Qwen processor, not the
model weights):

    python -m real_world_gwm.audit --roots /root/data/vrs/test \\
        --out audit_manifest.json [--frame_step 1] [--window_stride 1] \\
        [--min_pixels 131072] [--max_pixels 138240] [--token_ceiling 2048]
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from real_world_gwm.adapters.vrs.dataset import (
    discover_clips,
    enumerate_windows,
    load_window,
)

MAIN_CAMERA_NOTE = (
    "VRS releases a single RGB stream per clip; it is the main camera and "
    "no auxiliary views exist to ignore."
)
TEMPORAL_NOTE = (
    "VRS publishes no timestamps or FPS; windows are ordinal and are never "
    "labeled with seconds."
)


def _frame_sizes(clip):
    """(h, w) per frame from image headers, without decoding pixels."""
    sizes = []
    for p in clip.rgb_paths:
        with Image.open(p) as im:
            w, h = im.size
        sizes.append((h, w))
    return sizes


def _motion_stats(clip, candidate_steps, sample_windows):
    """Per-window robot-motion statistics for candidate frame steps."""
    stats = []
    for step in candidate_steps:
        windows = enumerate_windows(clip.n_frames, step, window_stride=1)
        if not windows:
            stats.append(
                {"frame_step": step, "windows": 0, "mask_centroid_disp": None,
                 "frame_abs_diff": None}
            )
            continue
        take = windows[:: max(1, len(windows) // sample_windows)][:sample_windows]
        disps, diffs = [], []
        for indices in take:
            sample = load_window(clip, indices)
            mask = sample["mask"][:, 0].numpy()  # (6, H, W)
            rgb = sample["rgb"].numpy()
            h, w = mask.shape[-2:]
            diag = float(np.hypot(h, w))
            for t in range(5):
                a, b = mask[t], mask[t + 1]
                if a.sum() > 0 and b.sum() > 0:
                    ca = np.array(np.nonzero(a)).mean(axis=1)
                    cb = np.array(np.nonzero(b)).mean(axis=1)
                    disps.append(float(np.linalg.norm(cb - ca)) / diag)
                diffs.append(float(np.abs(rgb[t + 1] - rgb[t]).mean()))
        stats.append(
            {
                "frame_step": step,
                "windows": len(windows),
                "mask_centroid_disp": float(np.mean(disps)) if disps else None,
                "frame_abs_diff": float(np.mean(diffs)) if diffs else None,
            }
        )
    return stats


def qwen_token_counter(preprocessor, min_pixels=None, max_pixels=None):
    """Exact-production-path token counter, cached per distinct frame size."""
    from real_world_gwm.qwen_rat import count_visual_tokens, rat_to_qwen_inputs

    cache = {}

    def count(clip):
        sizes = _frame_sizes(clip)
        key = sizes[0]
        if key not in cache:
            indices = list(range(6))
            sample = load_window(clip, indices)
            out = rat_to_qwen_inputs(
                sample["rgb"], sample["rgb"], preprocessor,
                min_pixels=min_pixels, max_pixels=max_pixels,
            )
            inputs = out["qwen_trajectory_gt"]
            grid = inputs["video_grid_thw"].reshape(-1).tolist()
            cache[key] = {"grid": grid, "tokens": count_visual_tokens(inputs)}
        return cache[key]

    return count


def build_manifest(
    roots,
    frame_step,
    window_stride,
    token_counter,
    candidate_steps=(1, 2, 3),
    token_ceiling=2048,
    motion_sample_windows=3,
    pixel_budget=None,
    limit_videos=None,
):
    """motion_sample_windows=0 skips motion statistics (train.py auto-audit)."""
    clip_entries, exclusions = [], []
    token_hist = Counter()
    batch_shapes = []
    violations = []

    for root in roots:
        root = Path(root)
        clips, excluded = discover_clips(root)
        if limit_videos is not None:
            clips = clips[:limit_videos]
        for e in excluded:
            exclusions.append({"root": str(root), **e})
        for clip in clips:
            sizes = _frame_sizes(clip)
            size_set = sorted(set(sizes))
            inconsistent = len(size_set) > 1
            valid_windows = len(
                enumerate_windows(clip.n_frames, frame_step, window_stride)
            )
            entry = {
                "root": str(root),
                "split": root.name,
                "video_id": clip.video_id,
                "embodiment": clip.embodiment,
                "frame_count": clip.n_frames,
                "frame_size": list(size_set[0]),
                "inconsistent_frame_sizes": (
                    [list(s) for s in size_set] if inconsistent else []
                ),
                "mask_provenance": clip.mask_provenance,
                "valid_windows": valid_windows,
                "motion_stats": (
                    _motion_stats(clip, candidate_steps, motion_sample_windows)
                    if motion_sample_windows > 0
                    else []
                ),
            }
            if inconsistent:
                exclusions.append(
                    {
                        "root": str(root),
                        "video_id": clip.video_id,
                        "reason": "inconsistent_frame_sizes",
                        "sizes": [list(s) for s in size_set],
                    }
                )
                continue
            if valid_windows > 0:
                tk = token_counter(clip)
                entry["qwen_grid"] = list(tk["grid"])
                entry["qwen_tokens"] = int(tk["tokens"])
                token_hist[str(tk["tokens"])] += 1
                if list(tk["grid"]) not in batch_shapes:
                    batch_shapes.append(list(tk["grid"]))
                if token_ceiling and tk["tokens"] > token_ceiling:
                    violations.append(
                        {
                            "video_id": clip.video_id,
                            "frame_size": list(size_set[0]),
                            "pixel_budget": pixel_budget,
                            "grid": list(tk["grid"]),
                            "tokens": int(tk["tokens"]),
                        }
                    )
            clip_entries.append(entry)

    manifest = {
        "source": "vrs",
        "roots": [str(r) for r in roots],
        "main_camera": {"key": "image", "note": MAIN_CAMERA_NOTE},
        "temporal_sampling": {
            "kind": "ordinal",
            "frame_step": frame_step,
            "window_stride": window_stride,
        },
        "temporal_note": TEMPORAL_NOTE,
        "pixel_budget": pixel_budget,
        "token_ceiling": token_ceiling,
        "clips": clip_entries,
        "exclusions": exclusions,
        "token_histogram": dict(sorted(token_hist.items())),
        "batch_shapes": sorted(batch_shapes),
        "token_ceiling_violations": violations,
        "totals": {
            "clips": len(clip_entries),
            "frames": sum(c["frame_count"] for c in clip_entries),
            "valid_windows": sum(c["valid_windows"] for c in clip_entries),
            "excluded": len(exclusions),
        },
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", default="audit_manifest.json")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--candidate_steps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--token_ceiling", type=int, default=2048,
                        help="0 disables the ceiling check")
    parser.add_argument("--motion_sample_windows", type=int, default=3)
    parser.add_argument(
        "--embedder_model_path", default="Qwen/Qwen3-VL-Embedding-8B"
    )
    args = parser.parse_args()

    from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLPreprocessor
    from real_world_gwm.qwen_rat import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS

    preprocessor = Qwen3VLPreprocessor(args.embedder_model_path)
    min_pixels = args.min_pixels if args.min_pixels is not None else DEFAULT_MIN_PIXELS
    max_pixels = args.max_pixels if args.max_pixels is not None else DEFAULT_MAX_PIXELS

    manifest = build_manifest(
        roots=args.roots,
        frame_step=args.frame_step,
        window_stride=args.window_stride,
        candidate_steps=tuple(args.candidate_steps),
        token_counter=qwen_token_counter(preprocessor, min_pixels, max_pixels),
        token_ceiling=args.token_ceiling,
        motion_sample_windows=args.motion_sample_windows,
        pixel_budget={"min_pixels": min_pixels, "max_pixels": max_pixels},
    )
    Path(args.out).write_text(json.dumps(manifest, indent=1))
    t = manifest["totals"]
    print(
        f"clips={t['clips']} frames={t['frames']} valid_windows={t['valid_windows']} "
        f"excluded={t['excluded']} shapes={manifest['batch_shapes']} "
        f"hash={manifest['manifest_hash'][:12]}"
    )
    if manifest["token_ceiling_violations"]:
        raise SystemExit(
            "token ceiling exceeded (fail-fast, raise --token_ceiling to accept):\n"
            + json.dumps(manifest["token_ceiling_violations"], indent=1)
        )


if __name__ == "__main__":
    main()
