"""VRS (RobotSeg) source adapter: clip discovery, ordinal windows, RAT samples.

Layout of a released VRS tree (train or test):
    <root>/image/<video_id>/00001.jpg ...
    <root>/mask_gt/<video_id>/{000,001,002}/00001.png          (test: human-annotated)
    <root>/mask_gt_dinov3/<video_id>/{000,001,002}/00001.png   (train: DINOv3 pseudo)

The adapter uses the whole-robot mask category ``002`` only, prefers ``mask_gt``
over ``mask_gt_dinov3``, and records which one a clip used (provenance).
"""


from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WHOLE_ROBOT_CATEGORY = "002"
# Precedence: human/densely annotated masks first, then DINOv3 pseudo masks.
MASK_DIRS = ("mask_gt", "mask_gt_dinov3")


@dataclass
class Clip:
    """One VRS video normalized to the corpus contract (single main camera)."""

    video_id: str
    rgb_paths: list
    mask_paths: list
    mask_provenance: str  # which mask tree supplied the whole-robot masks

    @property
    def embodiment(self) -> str:
        parts = self.video_id.split("___")
        return parts[1] if len(parts) > 1 else "unknown"

    @property
    def n_frames(self) -> int:
        return len(self.rgb_paths)


def discover_clips(root) -> tuple:
    """Discover clips under one released VRS tree (train or test).

    Returns (clips, excluded) where excluded entries carry a machine-readable
    reason. A clip is usable only if every RGB frame has a whole-robot (002)
    mask from a single mask tree.
    """
    root = Path(root)
    clips, excluded = [], []
    image_root = root / "image"
    if not image_root.is_dir():
        raise FileNotFoundError(f"not a VRS tree (no image/ dir): {root}")

    for video_dir in sorted(image_root.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        rgb_paths = sorted(video_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not rgb_paths:
            excluded.append({"video_id": video_id, "reason": "no_rgb_frames"})
            continue

        chosen = None
        missing_report = []
        for mask_dir in MASK_DIRS:
            cat_dir = root / mask_dir / video_id / WHOLE_ROBOT_CATEGORY
            if not cat_dir.is_dir():
                continue
            mask_paths = [cat_dir / f"{p.stem}.png" for p in rgb_paths]
            missing = [p.name for p in mask_paths if not p.is_file()]
            if missing:
                missing_report.append((mask_dir, missing))
                continue
            chosen = (mask_dir, mask_paths)
            break

        if chosen is None:
            if missing_report:
                mask_dir, missing = missing_report[0]
                excluded.append(
                    {
                        "video_id": video_id,
                        "reason": f"missing_mask_frames:{mask_dir}",
                        "missing": missing,
                    }
                )
            else:
                excluded.append(
                    {"video_id": video_id, "reason": "no_whole_robot_mask"}
                )
            continue

        provenance, mask_paths = chosen
        clips.append(
            Clip(
                video_id=video_id,
                rgb_paths=rgb_paths,
                mask_paths=mask_paths,
                mask_provenance=provenance,
            )
        )
    return clips, excluded


def enumerate_windows(n_frames: int, frame_step: int, window_stride: int) -> list:
    """Every complete six-frame ordinal window [i + k*frame_step for k in 0..5].

    Incomplete windows are rejected, never padded or tail-repeated.
    """
    span = 5 * frame_step
    windows = []
    for start in range(0, n_frames, window_stride):
        if start + span >= n_frames:
            break
        windows.append([start + k * frame_step for k in range(6)])
    return windows


def derive_robot_only(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """robot_only[t] = rgb[t] where mask is robot, black elsewhere."""
    return rgb * mask


def build_rat_pair(rgb: torch.Tensor, robot_only: torch.Tensor) -> tuple:
    """RAT condition = [rgb[0], robot_only[1:6]]; target = rgb[0:6]."""
    condition = torch.cat([rgb[:1], robot_only[1:]], dim=0)
    return condition, rgb


def _load_rgb(path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def _load_mask(path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (torch.from_numpy(arr) > 127).float().unsqueeze(0)


class VRSWindowDataset(torch.utils.data.Dataset):
    """All complete six-frame windows across one or more released VRS trees.

    Yields raw RAT samples (condition/target tensors); when a Qwen
    preprocessor is supplied, additionally yields the preprocessed
    ``qwen_current_inputs`` / ``qwen_trajectory_gt`` tensors used by training.
    Windows are sampled uniformly by the dataloader — no balancing by
    embodiment, clip, or window count.
    """

    def __init__(
        self,
        roots,
        frame_step: int = 1,
        window_stride: int = 1,
        flip_prob: float = 0.5,
        jitter_prob: float = 0.5,
        preprocessor=None,
        min_pixels: int = None,
        max_pixels: int = None,
        limit_videos: int = None,
        limit_windows: int = None,
    ):
        self.frame_step = frame_step
        self.window_stride = window_stride
        self.flip_prob = flip_prob
        self.jitter_prob = jitter_prob
        self.preprocessor = preprocessor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.clips = []
        self.excluded = []
        for root in roots:
            clips, excluded = discover_clips(root)
            self.clips.extend(clips)
            self.excluded.extend(excluded)
        if limit_videos is not None:
            self.clips = self.clips[:limit_videos]

        self.index = []  # (clip_idx, [frame indices])
        for ci, clip in enumerate(self.clips):
            for w in enumerate_windows(clip.n_frames, frame_step, window_stride):
                self.index.append((ci, w))
        if limit_windows is not None:
            self.index = self.index[:limit_windows]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        from real_world_gwm.augment import augment_window

        ci, indices = self.index[i]
        sample = load_window(self.clips[ci], indices)
        sample = augment_window(sample, self.flip_prob, self.jitter_prob)
        condition, target = build_rat_pair(sample["rgb"], sample["robot_only"])
        sample["condition"] = condition
        sample["target"] = target
        if self.preprocessor is not None:
            from real_world_gwm.qwen_rat import rat_to_qwen_inputs

            sample.update(
                rat_to_qwen_inputs(
                    condition,
                    target,
                    self.preprocessor,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
            )
        return sample


def load_window(clip: Clip, indices: list) -> dict:
    """Load one six-frame window as float tensors in [0, 1]."""
    rgb = torch.stack([_load_rgb(clip.rgb_paths[i]) for i in indices])
    mask = torch.stack([_load_mask(clip.mask_paths[i]) for i in indices])
    return {
        "rgb": rgb,  # (6, 3, H, W)
        "mask": mask,  # (6, 1, H, W)
        "robot_only": derive_robot_only(rgb, mask),
        "frame_indices": list(indices),
        "video_id": clip.video_id,
        "mask_provenance": clip.mask_provenance,
    }
