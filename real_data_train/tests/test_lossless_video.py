"""Per-clip video: FFV1 bit-exact roundtrip is a hard requirement for real
sources (D-27); the sim tree's near-lossless VP9 must stay inside its
calibrated tolerance gates (D-32)."""

import numpy as np
import pytest

from real_data_train.lossless_video import (
    read_video_frames,
    verify_lossless,
    verify_near_lossless,
    write_lossless_video,
    write_near_lossless_video,
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


def _smooth_frames(t=8, h=32, w=48):
    # render-like content: smooth gray gradient blob moving over black —
    # matches what the sim tree actually stores (no sensor noise)
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    ramp = np.linspace(60, 220, 12, dtype=np.uint8)[:, None]
    for i in range(t):
        frames[i, 8:20, 4 + i : 16 + i] = np.broadcast_to(
            ramp[:, :, None], (12, 12, 3)
        )
    return frames


def test_near_lossless_roundtrip_within_tolerance(tmp_path):
    frames = _smooth_frames()
    path = write_near_lossless_video(frames, tmp_path / "clip.mkv")
    got = read_video_frames(path)
    assert got.shape == frames.shape
    verify_near_lossless(path, frames)  # must not raise


def test_near_lossless_verify_rejects_gross_error(tmp_path):
    frames = _smooth_frames()
    path = write_near_lossless_video(frames, tmp_path / "clip.mkv")
    wrong = np.full_like(frames, 255)
    with pytest.raises(RuntimeError,
                       match="near-lossless verification FAILED"):
        verify_near_lossless(path, wrong)
