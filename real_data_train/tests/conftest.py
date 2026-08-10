import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _robot_frames(n, h, w):
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        frames[i, 10:30, 20 + i % 20 : 40 + i % 20] = 200
    return frames


def _base_meta(data_root, video_path, n, h, w):
    return {
        "source": "molmobot",
        "clip_id": "houseX__ep0__camA",
        "episode_uid": "molmobot/testcfg/houseX/batch_1_of_1_ep0",
        "camera": "camA",
        "n_frames": n,
        "width": w,
        "height": h,
        "timestamps": [i / 15.0 for i in range(n)],
        "rgb_video": str(video_path.relative_to(data_root)),
        "rgb_frame_start": 0,
        "intrinsics": [[100.0, 0, 48.0], [0, 100.0, 32.0], [0, 0, 1.0]],
        "renderer_provenance": {"arm": "fr3", "test": True},
    }


def _write_rgb_mp4(data_root, n, h, w):
    import imageio.v2 as imageio

    rng = np.random.default_rng(0)
    frames = (rng.random((n, h, w, 3)) * 255).astype(np.uint8)
    video_path = data_root / "molmobot" / "houseX" / "ep0_camA.mp4"
    video_path.parent.mkdir(parents=True)
    with imageio.get_writer(video_path, fps=15, macro_block_size=1) as writer:
        for f in frames:
            writer.append_data(f)
    return video_path


@pytest.fixture
def synthetic_rendered_root(tmp_path):
    """A tiny data root with one schema-v2 rendered clip + a real RGB mp4.

    Mirrors what render_actions.py writes: 15 fps, 60 frames, 96x64 — enough
    for one accepted 3-second window; robot-only as an FFV1 lossless video.
    """
    from real_data_train.lossless_video import write_lossless_video

    n, h, w = 60, 64, 96
    data_root = tmp_path / "data_v2"
    clip_dir = data_root / "rendered" / "molmobot" / "houseX__ep0__camA"
    clip_dir.mkdir(parents=True)
    video_path = _write_rgb_mp4(data_root, n, h, w)
    write_lossless_video(_robot_frames(n, h, w), clip_dir / "robot_only.mkv")
    meta = {
        "schema_version": 2,
        **_base_meta(data_root, video_path, n, h, w),
        "robot_only_video": "robot_only.mkv",
        "robot_only_codec": "ffv1/bgr0",
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta))
    return data_root


