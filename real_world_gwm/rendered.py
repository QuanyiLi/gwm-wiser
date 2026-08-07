"""The normalized rendered tree: discovery, held-out split, window dataset.

Training consumes ONE on-disk contract regardless of source (decision D-18):

    <data_root>/rendered/<source>/<clip_id>/
        robot_only.mkv               state-rendered robot-only RGB, native res
                                     (FFV1 lossless, bit-exact-verified at
                                     write time — D-27)
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from real_world_gwm.windows import build_rat_pair, enumerate_timed_windows

# Per-source anchor stride in seconds (decision D-6): dense for long real
# episodes, one-to-two windows for short sim episodes.
DEFAULT_STRIDE_S = {"molmoact2_droid": 0.5, "molmobot": 3.0}
HOLDOUT_PERMILLE = 20   # 2% of episodes


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
    """

    def __init__(
        self,
        data_root,
        sources=None,
        split: str = "train",
        stride_s: dict = None,
        tolerance_s: float = None,
        jitter_prob: float = 0.5,
        preprocessor=None,
        min_pixels: int = None,
        max_pixels: int = None,
        holdout_permille: int = HOLDOUT_PERMILLE,
        limit_clips: int = None,
        limit_windows: int = None,
    ):
        from real_world_gwm.windows import DEFAULT_TOLERANCE_S

        self.data_root = Path(data_root)
        self.jitter_prob = jitter_prob
        self.preprocessor = preprocessor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        stride_s = {**DEFAULT_STRIDE_S, **(stride_s or {})}
        tolerance_s = DEFAULT_TOLERANCE_S if tolerance_s is None else tolerance_s

        clips = discover_rendered_clips(self.data_root, sources)
        self.clips = split_clips(clips, split, holdout_permille)
        if limit_clips is not None:
            self.clips = self.clips[:limit_clips]

        self.index = []   # (clip_idx, [six frame indices])
        for ci, clip in enumerate(self.clips):
            windows = enumerate_timed_windows(
                clip.timestamps, stride_s[clip.source], tolerance_s
            )
            self.index.extend((ci, w) for w in windows)
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

    def __getitem__(self, i):
        from real_world_gwm.augment import jitter_window

        ci, indices = self.index[i]
        sample = self.load_window(self.clips[ci], indices)
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
