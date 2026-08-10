"""Preprocessing audit over the rendered tree: the mandatory gate before training.

Source-agnostic: it walks the normalized rendered clips, enumerates the
timestamped windows exactly as training will, counts Qwen visual tokens
through the production preprocessing path, and enforces the exact-grid policy
(decision D-2): every clip must land on the operating grid — off-grid clips
are violations, not curiosities.

    python -m real_data_train.audit --data_root real_data_train/data \\
        --out audit_manifest.json [--min_pixels ...] [--max_pixels ...]
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from real_data_train.rendered import (
    OPERATING_ANCHOR_WH,
    DEFAULT_STRIDE_S,
    HOLDOUT_PERMILLE,
    discover_rendered_clips,
    is_heldout,
)
from real_data_train.windows import DEFAULT_TOLERANCE_S, enumerate_timed_windows

OPERATING_GRID = [3, 18, 30]   # ADR-0019: 1620 tokens


def qwen_token_counter(preprocessor, min_pixels=None, max_pixels=None):
    """Exact-production-path token counter, cached per frame size."""
    from real_data_train.qwen_rat import count_visual_tokens, rat_to_qwen_inputs

    cache = {}

    def count(height, width):
        key = (height, width)
        if key not in cache:
            frames = torch.zeros(6, 3, height, width)
            out = rat_to_qwen_inputs(frames, frames, preprocessor,
                                     min_pixels=min_pixels,
                                     max_pixels=max_pixels)
            inputs = out["qwen_trajectory_gt"]
            grid = inputs["video_grid_thw"].reshape(-1).tolist()
            cache[key] = {"grid": grid, "tokens": count_visual_tokens(inputs)}
        return cache[key]

    return count


def build_manifest(
    data_root,
    token_counter,
    sources=None,
    stride_s: dict = None,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    token_ceiling: int = 2048,
    expected_grid=OPERATING_GRID,
    holdout_permille: int = HOLDOUT_PERMILLE,
    pixel_budget: dict = None,
    limit_clips: int = None,
    clips: list = None,
):
    stride_s = {**DEFAULT_STRIDE_S, **(stride_s or {})}
    if clips is None:
        clips = discover_rendered_clips(data_root, sources)
    elif sources:
        clips = [c for c in clips if c.source in sources]
    if limit_clips is not None:
        clips = clips[:limit_clips]

    entries, token_hist, batch_shapes = [], Counter(), []
    off_grid, ceiling_violations = [], []
    split_windows = {"train": 0, "heldout": 0}
    for clip in clips:
        windows = enumerate_timed_windows(
            clip.timestamps, stride_s[clip.source], tolerance_s
        )
        max_err = 0.0
        ts = clip.timestamps
        from real_data_train.windows import SCHEDULE

        for w in windows:
            t0 = ts[w[0]]
            max_err = max(
                max_err,
                max(abs(ts[k] - (t0 + off)) for k, off in zip(w, SCHEDULE)),
            )
        h, w_px = clip.meta["height"], clip.meta["width"]
        # Windows are anchor-resized before preprocessing (decision D-29), so
        # the token grid is a property of the anchor, not the native size.
        aw, ah = OPERATING_ANCHOR_WH
        tk = token_counter(ah, aw)
        entry = {
            "source": clip.source,
            "clip_id": clip.clip_id,
            "episode_uid": clip.episode_uid,
            "camera": clip.meta.get("camera"),
            "heldout": is_heldout(clip.episode_uid, holdout_permille),
            "n_frames": clip.n_frames,
            "frame_size": [h, w_px],
            "valid_windows": len(windows),
            "max_schedule_error_s": round(max_err, 4),
            "qwen_grid": list(tk["grid"]),
            "qwen_tokens": int(tk["tokens"]),
        }
        entries.append(entry)
        token_hist[str(tk["tokens"])] += 1
        if list(tk["grid"]) not in batch_shapes:
            batch_shapes.append(list(tk["grid"]))
        if expected_grid and list(tk["grid"]) != list(expected_grid):
            off_grid.append({"clip_id": clip.clip_id,
                             "frame_size": [h, w_px],
                             "grid": list(tk["grid"])})
        if token_ceiling and tk["tokens"] > token_ceiling:
            ceiling_violations.append({"clip_id": clip.clip_id,
                                       "tokens": int(tk["tokens"])})
        split = "heldout" if entry["heldout"] else "train"
        split_windows[split] += len(windows)

    manifest = {
        "data_root": str(data_root),
        "sources": sorted({c.source for c in clips}),
        "temporal_sampling": {
            "kind": "timestamped",
            "stride_s": stride_s,
            "tolerance_s": tolerance_s,
        },
        "pixel_budget": pixel_budget,
        "anchor_wh": list(OPERATING_ANCHOR_WH),
        "token_ceiling": token_ceiling,
        "expected_grid": list(expected_grid) if expected_grid else None,
        "holdout_permille": holdout_permille,
        "clips": entries,
        "token_histogram": dict(sorted(token_hist.items())),
        "batch_shapes": sorted(batch_shapes),
        "off_grid_violations": off_grid,
        "token_ceiling_violations": ceiling_violations,
        "totals": {
            "clips": len(entries),
            "frames": sum(e["n_frames"] for e in entries),
            "valid_windows": sum(e["valid_windows"] for e in entries),
            "train_windows": split_windows["train"],
            "heldout_windows": split_windows["heldout"],
        },
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--out", default="audit_manifest.json")
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--token_ceiling", type=int, default=2048)
    parser.add_argument("--no_grid_check", action="store_true",
                        help="report grids without enforcing the operating grid")
    parser.add_argument("--limit_clips", type=int, default=None)
    parser.add_argument("--embedder_model_path",
                        default="Qwen/Qwen3-VL-Embedding-8B")
    args = parser.parse_args()

    from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLPreprocessor
    from real_data_train.qwen_rat import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS

    preprocessor = Qwen3VLPreprocessor(args.embedder_model_path)
    min_pixels = DEFAULT_MIN_PIXELS if args.min_pixels is None else args.min_pixels
    max_pixels = DEFAULT_MAX_PIXELS if args.max_pixels is None else args.max_pixels

    manifest = build_manifest(
        data_root=args.data_root,
        sources=args.sources,
        token_counter=qwen_token_counter(preprocessor, min_pixels, max_pixels),
        token_ceiling=args.token_ceiling,
        expected_grid=None if args.no_grid_check else OPERATING_GRID,
        pixel_budget={"min_pixels": min_pixels, "max_pixels": max_pixels},
        limit_clips=args.limit_clips,
    )
    Path(args.out).write_text(json.dumps(manifest, indent=1))
    t = manifest["totals"]
    print(
        f"clips={t['clips']} frames={t['frames']} windows={t['valid_windows']} "
        f"(train={t['train_windows']} heldout={t['heldout_windows']}) "
        f"shapes={manifest['batch_shapes']} hash={manifest['manifest_hash'][:12]}"
    )
    problems = []
    if manifest["off_grid_violations"]:
        problems.append(
            f"{len(manifest['off_grid_violations'])} clips off the operating "
            f"grid {manifest['expected_grid']}"
        )
    if manifest["token_ceiling_violations"]:
        problems.append(
            f"{len(manifest['token_ceiling_violations'])} clips above the "
            "token ceiling"
        )
    if problems:
        raise SystemExit("audit FAILED: " + "; ".join(problems))


if __name__ == "__main__":
    main()
