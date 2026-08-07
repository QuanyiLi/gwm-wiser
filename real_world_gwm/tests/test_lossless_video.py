"""FFV1 per-clip video: bit-exact roundtrip is a hard requirement (D-27)."""

import numpy as np
import pytest

from real_world_gwm.lossless_video import (
    read_video_frames,
    verify_lossless,
    write_lossless_video,
)


def _frames(t=8, h=32, w=48, seed=0):
    rng = np.random.default_rng(seed)
    # robot-only-like content: mostly black with a moving bright blob
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        frames[i, 8:20, 4 + i : 16 + i] = rng.integers(
            0, 256, (12, 12, 3), dtype=np.uint8
        )
    return frames


def test_roundtrip_is_bit_exact(tmp_path):
    frames = _frames()
    path = write_lossless_video(frames, tmp_path / "clip.mkv")
    got = read_video_frames(path)
    assert got.shape == frames.shape
    assert np.array_equal(got, frames)
    verify_lossless(path, frames)  # must not raise


def test_indexed_reads_match_full_decode(tmp_path):
    frames = _frames(t=12)
    path = write_lossless_video(frames, tmp_path / "clip.mkv")
    got = read_video_frames(path, indices=[0, 5, 11])
    assert np.array_equal(got, frames[[0, 5, 11]])


def test_verify_rejects_corruption(tmp_path):
    frames = _frames()
    path = write_lossless_video(frames, tmp_path / "clip.mkv")
    wrong = frames.copy()
    wrong[0, 0, 0, 0] ^= 1
    with pytest.raises(RuntimeError, match="lossless verification FAILED"):
        verify_lossless(path, wrong)
