import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_jpg(path, arr):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=95)


def _write_png(path, arr):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def synthetic_vrs_root(tmp_path):
    """A tiny synthetic tree mimicking the released VRS layout.

    <root>/image/<video>/00001.jpg ...
    <root>/mask_gt/<video>/002/00001.png ...          (test-tree convention)
    <root>/mask_gt_dinov3/<video>/002/00001.png ...   (train-tree convention)

    Videos:
      video_0001___01_franka___a   12 frames, mask_gt (human)     32x48
      video_0002___06_ur5___b      7 frames, mask_gt_dinov3 only  32x48
      video_0003___01_franka___c   6 frames, NO masks at all      32x48
      video_0004___01_franka___d   6 frames, mask missing frame 3 32x48
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "vrs_test"

    def make_video(name, n_frames, mask_dir, skip_mask_frames=()):
        for i in range(1, n_frames + 1):
            rgb = rng.integers(0, 255, size=(32, 48, 3), dtype=np.uint8)
            _write_jpg(root / "image" / name / f"{i:05d}.jpg", rgb)
            if mask_dir is not None and i not in skip_mask_frames:
                mask = np.zeros((32, 48), dtype=np.uint8)
                # robot occupies a moving column block
                mask[:, (i * 3) % 40 : (i * 3) % 40 + 8] = 255
                for cat in ("000", "001", "002"):
                    _write_png(root / mask_dir / name / cat / f"{i:05d}.png", mask)

    make_video("video_0001___01_franka___a", 12, "mask_gt")
    make_video("video_0002___06_ur5___b", 7, "mask_gt_dinov3")
    make_video("video_0003___01_franka___c", 6, None)
    make_video("video_0004___01_franka___d", 6, "mask_gt", skip_mask_frames=(3,))
    return root
