"""Image preprocessing that reproduces what V-JEPA 2-AC saw in training.

What the upstream code does (vjepa2/configs/train/vitg16/droid-256px-8f.yaml,
app/vjepa_droid/transforms.py): `random_resized_crop` with
random_resize_scale=(1.777, 1.777), random_resize_aspect_ratio=(0.75, 1.35),
crop_size=256, no flip; then ImageNet mean/std. A crop of 1.777x the image
area can never fit, so every draw fails and `_get_param_spatial_crop` falls
back to its deterministic central crop:

  in_ratio = W/H
  in_ratio > 1.35  -> full height, width = 1.35*H, centred
  in_ratio < 0.75  -> full width,  height = W/0.75, centred
  otherwise        -> whole image

followed by a bilinear `F.interpolate` (align_corners=False, no antialias) to
256x256. The paper (Sec. 3.1) states the DROID clips were stored at 256x256
and 4 fps, and the repo's own robot example (notebooks/franka_example_traj.npz)
is a full 16:9 camera frame squashed to 256x256, so on the stored square
frames the transform was the identity and the model saw whole frames
anisotropically resized to 256x256. Modes:

  full_aa  (faithful; default) whole frame -> antialiased bicubic resize to
           256x256, as a video re-encode to 256x256 does; then the training
           normalisation
  train    the transform's fallback applied to the raw 16:9 frame (1.35:1
           centre crop, bilinear, no antialias) -- what one gets by feeding
           the raw frame through the training transform; kept as an ablation
  full     whole frame, bilinear without antialias
  square   central square crop
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CROP_SIZE = 256
RATIO_MIN, RATIO_MAX = 0.75, 1.35
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32) * 255.0
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32) * 255.0


def aspect_clamp_box(height, width, ratio_min=RATIO_MIN, ratio_max=RATIO_MAX):
    """(i, j, h, w) of the central crop that clamps W/H into [ratio_min, ratio_max]."""
    in_ratio = float(width) / float(height)
    if in_ratio < ratio_min:
        w = width
        h = int(round(w / ratio_min))
    elif in_ratio > ratio_max:
        h = height
        w = int(round(h * ratio_max))
    else:
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def crop_frames(frames, mode="full_aa"):
    """Crop uint8 frames [T, H, W, 3] per `mode` (see module docstring); returns a uint8 array."""
    frames = np.asarray(frames)
    T, H, W, _ = frames.shape
    if mode == "train":
        i, j, h, w = aspect_clamp_box(H, W)
    elif mode in ("full", "full_aa"):
        i, j, h, w = 0, 0, H, W
    elif mode == "square":
        s = min(H, W)
        i, j, h, w = (H - s) // 2, (W - s) // 2, s, s
    else:
        raise ValueError(f"unknown crop mode {mode!r}")
    return frames[:, i : i + h, j : j + w]


def _resize(cropped, mode, crop_size):
    """uint8 [T, h, w, 3] -> float [T, 3, crop, crop] in 0..255."""
    if mode == "full_aa":
        out = np.stack([np.asarray(Image.fromarray(f).resize((crop_size, crop_size), Image.BICUBIC)) for f in cropped])
        return torch.from_numpy(out).to(torch.float32).permute(0, 3, 1, 2)
    x = torch.from_numpy(np.ascontiguousarray(cropped)).to(torch.float32).permute(0, 3, 1, 2)  # T C H W
    return F.interpolate(x, size=(crop_size, crop_size), mode="bilinear", align_corners=False)


def to_model_input(frames, mode="full_aa", crop_size=CROP_SIZE):
    """uint8 frames [T, H, W, 3] -> normalised float tensor [T, 3, crop, crop]."""
    x = _resize(crop_frames(frames, mode), mode, crop_size)
    return (x - MEAN.view(1, 3, 1, 1)) / STD.view(1, 3, 1, 1)


def model_view_uint8(frames, mode="full_aa", crop_size=CROP_SIZE):
    """What the model sees, as uint8 [T, crop, crop, 3] (for saving / inspection)."""
    x = _resize(crop_frames(frames, mode), mode, crop_size)
    return x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()
