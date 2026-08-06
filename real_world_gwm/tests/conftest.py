import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def synthetic_rendered_root(tmp_path):
    """A tiny data root with one rendered clip + a real mp4 (imageio-encoded).

    Mirrors the normalized rendered tree render_actions.py writes: 15 fps,
    60 frames, 96x64 — enough for one accepted 3-second window.
    """
    import imageio.v2 as imageio
    from PIL import Image

    rng = np.random.default_rng(0)
    n, h, w = 60, 64, 96
    data_root = tmp_path / "data"
    clip_dir = data_root / "rendered" / "molmobot" / "houseX__ep0__camA"
    frame_dir = clip_dir / "robot_only"
    frame_dir.mkdir(parents=True)

    frames = (rng.random((n, h, w, 3)) * 255).astype(np.uint8)
    video_path = data_root / "molmobot" / "houseX" / "ep0_camA.mp4"
    video_path.parent.mkdir(parents=True)
    with imageio.get_writer(video_path, fps=15, macro_block_size=1) as writer:
        for f in frames:
            writer.append_data(f)

    for i in range(n):
        robot = np.zeros((h, w, 3), dtype=np.uint8)
        robot[10:30, 20 + i % 20 : 40 + i % 20] = 200
        Image.fromarray(robot).save(frame_dir / f"{i:05d}.png")

    meta = {
        "schema_version": 1,
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
    (clip_dir / "meta.json").write_text(json.dumps(meta))
    return data_root
