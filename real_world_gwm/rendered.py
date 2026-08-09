"""The normalized rendered tree: discovery, held-out split, window dataset.

Training consumes ONE on-disk contract regardless of source (decision D-18):

    <data_root>/rendered/<source>/<clip_id>/
        robot_only.mkv               state-rendered robot-only RGB, native res
                                     (real sources: FFV1 lossless bit-exact,
                                     D-27; molmobot: near-lossless VP9, D-32
                                     — verified at write time either way)
        meta.json                    written last by render_actions (completion
                                     mark); pairs the stream with its source
                                     RGB video and timestamps

Full-scene RGB stays in the source videos and is decoded on the fly
(torchcodec); robot-only frames are the offline renders. Windows are
timestamped (windows.enumerate_timed_windows) with per-source stride.

Held-out policy (decision D-10): deterministic episode-level hash split on
episode_uid — camera-independent, so all streams of one episode land on the
same side; machine-independent, and newly provisioned episodes fall on a
stable side.
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from real_world_gwm.windows import (
    build_rat_pair,
    enumerate_timed_windows,
    nearest_index,
    resolve_scaled_window,
)

# Per-source anchor stride in seconds (decision D-6): dense for long real
# episodes, one-to-two windows for short sim episodes.
DEFAULT_STRIDE_S = {"molmoact2_droid": 0.5, "molmobot": 3.0}
HOLDOUT_PERMILLE = 20   # 2% of episodes

# Per-source time-scale ranges (decision D-33). MolmoBot PnP trajectories are
# scripted, smooth, and slow — at s < 1 their windows are near-static, so the
# sim range stretches (1-3x canonical) instead of compressing; DROID keeps
# D-30's symmetric range. Sources not listed stay canonical (no augmentation).
DEFAULT_SCALE_RANGES = {
    "molmoact2_droid": (0.5, 1.5),
    "molmobot": (1.0, 3.0),
}


def scale_range_for(source: str, scale_range):
    """Resolve one item's (lo, hi): a tuple applies globally, a dict maps
    source -> range (unlisted sources stay canonical), None = off."""
    if scale_range is None:
        return None
    if isinstance(scale_range, dict):
        rng = tuple(scale_range.get(source, (1.0, 1.0)))
    else:
        rng = tuple(scale_range)
    return None if rng == (1.0, 1.0) else rng

# Operating-grid anchor resolution (decision D-29): every window is brought
# to this resolution before Qwen preprocessing, because the pixel-budget
# mechanism alone cannot land every native resolution on the exact (3,18,30)
# grid (320x180's reachable grids jump straight from 16x28 to 20x32).
# MolmoBot clips are natively 624x352 (no-op); DROID's 320x180 shares the
# aspect ratio to 0.3%, so the anchor resize adds no meaningful distortion.
OPERATING_ANCHOR_WH = (624, 352)


def anchor_resize(frames: torch.Tensor) -> torch.Tensor:
    """(T, 3, H, W) float -> the operating anchor resolution (bilinear)."""
    w, h = OPERATING_ANCHOR_WH
    if frames.shape[-2:] == (h, w):
        return frames
    return torch.nn.functional.interpolate(
        frames, size=(h, w), mode="bilinear", align_corners=False,
        antialias=True,
    )


@dataclass
class RenderedClip:
    clip_dir: Path
    meta: dict

    @property
    def source(self) -> str:
        return self.meta["source"]

    @property
    def clip_id(self) -> str:
        return self.meta["clip_id"]

    @property
    def episode_uid(self) -> str:
        return self.meta["episode_uid"]

    @property
    def n_frames(self) -> int:
        return self.meta["n_frames"]

    @property
    def timestamps(self) -> list:
        return self.meta["timestamps"]

    @property
    def robot_only_video(self) -> Path:
        return self.clip_dir / self.meta["robot_only_video"]


def discover_rendered_clips(data_root, sources=None) -> list:
    """All complete rendered clips under data_root/rendered."""
    root = Path(data_root) / "rendered"
    clips = []
    if not root.is_dir():
        return clips
    for meta_path in sorted(root.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if sources and meta["source"] not in sources:
            continue
        clips.append(RenderedClip(clip_dir=meta_path.parent, meta=meta))
    return clips


def tree_signature(data_root) -> dict:
    """Cheap staleness signal for the discovery cache: per-source counts and
    a hash of the sorted clip-directory names. Costs one readdir per source
    (seconds), not one file read per clip (~21 min for 147k clips on GPFS).
    Catches clips added, removed, or renamed; an in-place re-render into the
    SAME directories is invisible to it — delete the cache file then."""
    root = Path(data_root) / "rendered"
    sig = {}
    if not root.is_dir():
        return sig
    for src_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        names = sorted(p.name for p in src_dir.iterdir())
        sig[src_dir.name] = {
            "n": len(names),
            "sha": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        }
    return sig


def save_discovery_cache(path, clips, data_root) -> None:
    """Serialize a discovery result (clip dirs stored data_root-relative);
    stamped with tree_signature so a changed tree invalidates it."""
    root = Path(data_root)
    payload = {
        "version": 2,
        "signature": tree_signature(root),
        "clips": [
            {"clip_dir": str(c.clip_dir.relative_to(root)), "meta": c.meta}
            for c in clips
        ],
    }
    Path(path).write_text(json.dumps(payload))


def load_discovery_cache(path, data_root, sources=None):
    """Clips from a cache file, or None when the cache is stale (the tree
    signature changed — e.g. new shards were rendered) or incompatible."""
    root = Path(data_root)
    payload = json.loads(Path(path).read_text())
    if payload.get("version") != 2:
        return None
    if payload["signature"] != tree_signature(root):
        return None
    clips = [RenderedClip(clip_dir=root / e["clip_dir"], meta=e["meta"])
             for e in payload["clips"]]
    if sources:
        clips = [c for c in clips if c.source in sources]
    return clips


def is_heldout(episode_uid: str, permille: int = HOLDOUT_PERMILLE) -> bool:
    digest = hashlib.sha256(episode_uid.encode()).digest()
    return int.from_bytes(digest[:8], "big") % 1000 < permille


def split_clips(clips, split: str, permille: int = HOLDOUT_PERMILLE) -> list:
    if split == "all":
        return list(clips)
    if split == "train":
        return [c for c in clips if not is_heldout(c.episode_uid, permille)]
    if split == "heldout":
        return [c for c in clips if is_heldout(c.episode_uid, permille)]
    raise ValueError(f"unknown split {split!r}")


class RenderedWindowDataset(torch.utils.data.Dataset):
    """All accepted timestamped six-frame windows across rendered clips.

    Yields raw RAT samples (condition/target); with a Qwen preprocessor also
    the preprocessed ``qwen_current_inputs`` / ``qwen_trajectory_gt``.
    Horizontal flip is banned (render homology, decision D-13); color jitter
    applies to full RGB only.

    Time-scale augmentation (decisions D-30 / D-33): the index stays
    anchor-level — one entry per canonical (scale = 1) window, so epoch size,
    source mixture, and the audit are untouched — and with ``scale_range``
    set, __getitem__ re-resolves the schedule at a per-sample scale drawn
    log-uniformly, with a small anchor jitter. ``scale_range`` may be a
    (lo, hi) tuple applied to every source, or a {source: (lo, hi)} dict
    (D-33 — see DEFAULT_SCALE_RANGES; unlisted sources stay canonical).
    Draws that do not fit the clip retry, then fall back to the stored
    canonical window. A degenerate tuple (s, s) with zero jitter instead
    re-resolves the index ONCE at that fixed scale, dropping what does not
    fit — the deterministic, fallback-free form the fixed-scale held-out
    sweeps use.
    """

    RESAMPLE_TRIES = 8

    def __init__(
        self,
        data_root,
        sources=None,
        split: str = "train",
        stride_s: dict = None,
        tolerance_s: float = None,
        jitter_prob: float = 0.5,
        scale_range: tuple = None,
        anchor_jitter_s: dict = None,
        preprocessor=None,
        min_pixels: int = None,
        max_pixels: int = None,
        holdout_permille: int = HOLDOUT_PERMILLE,
        limit_clips: int = None,
        limit_windows: int = None,
        clips: list = None,
    ):
        from real_world_gwm.windows import DEFAULT_TOLERANCE_S

        self.data_root = Path(data_root)
        self.jitter_prob = jitter_prob
        self.preprocessor = preprocessor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        stride_s = {**DEFAULT_STRIDE_S, **(stride_s or {})}
        self.tolerance_s = (DEFAULT_TOLERANCE_S if tolerance_s is None
                            else tolerance_s)
        tolerance_s = self.tolerance_s
        if isinstance(scale_range, dict):
            self.scale_range = dict(scale_range)
        else:
            self.scale_range = tuple(scale_range) if scale_range else None
        # Anchor jitter defaults to half the per-source stride; {} disables.
        self.anchor_jitter_s = ({s: v / 2 for s, v in stride_s.items()}
                                if anchor_jitter_s is None else anchor_jitter_s)

        # ``clips`` lets one process share a single discovery scan across the
        # train/held-out/sweep datasets (the scan is O(all clips) on GPFS).
        if clips is None:
            clips = discover_rendered_clips(self.data_root, sources)
        elif sources:
            clips = [c for c in clips if c.source in sources]
        self.clips = split_clips(clips, split, holdout_permille)
        if limit_clips is not None:
            self.clips = self.clips[:limit_clips]

        self.index = []   # (clip_idx, [six frame indices at scale 1])
        for ci, clip in enumerate(self.clips):
            windows = enumerate_timed_windows(
                clip.timestamps, stride_s[clip.source], tolerance_s
            )
            self.index.extend((ci, w) for w in windows)

        self._fixed_scale = 1.0
        fixed = (isinstance(self.scale_range, tuple)
                 and self.scale_range[0] == self.scale_range[1]
                 and not any((self.anchor_jitter_s or {}).values()))
        if fixed:
            s = self.scale_range[0]
            self._fixed_scale = s
            self.scale_range = None   # materialized below, no per-item draw
            if s != 1.0:
                reresolved = []
                for ci, w in self.index:
                    got = resolve_scaled_window(
                        self.clips[ci].timestamps, w[0], s, tolerance_s)
                    if got is not None:
                        reresolved.append((ci, got))
                self.index = reresolved
        if limit_windows is not None:
            self.index = self.index[:limit_windows]

        self._decoders = {}   # per-process video decoder cache

    def __len__(self):
        return len(self.index)

    def _decoder(self, video_path: Path):
        key = str(video_path)
        if key not in self._decoders:
            from torchcodec.decoders import VideoDecoder

            if len(self._decoders) >= 16:   # rgb + robot-only per clip
                self._decoders.clear()
            self._decoders[key] = VideoDecoder(key, num_ffmpeg_threads=1)
        return self._decoders[key]

    def load_window(self, clip: RenderedClip, indices) -> dict:
        video_path = self.data_root / clip.meta["rgb_video"]
        start = clip.meta["rgb_frame_start"]
        dec = self._decoder(video_path)
        rgb = dec.get_frames_at([start + i for i in indices]).data
        rgb = rgb.float() / 255.0                       # (6, 3, H, W)
        robot_only = self._decoder(clip.robot_only_video).get_frames_at(
            list(indices)).data.float() / 255.0         # per-clip FFV1 (D-27)
        return {
            "rgb": rgb,
            "robot_only": robot_only,
            "frame_indices": list(indices),
            "video_id": clip.clip_id,
            "source": clip.source,
            "episode_uid": clip.episode_uid,
        }

    def _sample_indices(self, ci, canonical) -> tuple:
        """(indices, scale) for one draw: scaled schedule at a jittered
        anchor, falling back to the stored canonical window."""
        if self.scale_range is None:
            return canonical, self._fixed_scale
        clip = self.clips[ci]
        rng = scale_range_for(clip.source, self.scale_range)
        if rng is None:
            return canonical, 1.0
        ts = clip.timestamps
        lo, hi = rng
        jitter = (self.anchor_jitter_s or {}).get(clip.source, 0.0)
        for _ in range(self.RESAMPLE_TRIES):
            scale = math.exp(random.uniform(math.log(lo), math.log(hi)))
            anchor = canonical[0]
            if jitter:
                anchor = nearest_index(
                    ts, ts[canonical[0]] + random.uniform(-jitter, jitter))
            got = resolve_scaled_window(ts, anchor, scale, self.tolerance_s)
            if got is not None:
                return got, scale
        return canonical, 1.0

    def __getitem__(self, i):
        from real_world_gwm.augment import jitter_window

        ci, canonical = self.index[i]
        indices, scale = self._sample_indices(ci, canonical)
        sample = self.load_window(self.clips[ci], indices)
        sample["time_scale"] = scale
        sample["rgb"] = anchor_resize(sample["rgb"])
        sample["robot_only"] = anchor_resize(sample["robot_only"])
        sample = jitter_window(sample, self.jitter_prob)
        condition, target = build_rat_pair(sample["rgb"], sample["robot_only"])
        sample["condition"] = condition
        sample["target"] = target
        if self.preprocessor is not None:
            from real_world_gwm.qwen_rat import rat_to_qwen_inputs

            sample.update(rat_to_qwen_inputs(
                condition, target, self.preprocessor,
                min_pixels=self.min_pixels, max_pixels=self.max_pixels,
            ))
        return sample
