"""VRSWindowDataset: clips x windows -> augmented RAT samples."""

import torch

from real_world_gwm.adapters.vrs.dataset import VRSWindowDataset


def test_length_is_total_complete_windows_across_clips(synthetic_vrs_root):
    # video_0001: 12 frames -> 7 windows; video_0002: 7 frames -> 2 windows
    ds = VRSWindowDataset([synthetic_vrs_root], frame_step=1, window_stride=1)
    assert len(ds) == 9


def test_frame_step_reduces_window_count(synthetic_vrs_root):
    # step=2 needs an 11-frame span -> only video_0001 (12 frames) qualifies,
    # with two start indices (0 and 1)
    ds = VRSWindowDataset([synthetic_vrs_root], frame_step=2, window_stride=1)
    assert len(ds) == 2


def test_sample_contains_rat_pair_and_identifiers(synthetic_vrs_root):
    ds = VRSWindowDataset(
        [synthetic_vrs_root], frame_step=1, window_stride=1, flip_prob=0.0, jitter_prob=0.0
    )
    s = ds[7]  # first window of video_0002 (after video_0001's 7)
    assert s["video_id"] == "video_0002___06_ur5___b"
    assert s["frame_indices"] == [0, 1, 2, 3, 4, 5]
    assert s["condition"].shape == (6, 3, 32, 48)
    assert s["target"].shape == (6, 3, 32, 48)
    # condition[0] is the full current frame; futures are robot-only (masked)
    assert torch.equal(s["condition"][0], s["target"][0])
    mask = s["mask"][1].bool().expand(3, 32, 48)
    assert torch.all(s["condition"][1][~mask] == 0)


def test_limit_videos_caps_clip_count(synthetic_vrs_root):
    ds = VRSWindowDataset(
        [synthetic_vrs_root], frame_step=1, window_stride=1, limit_videos=1
    )
    assert len(ds) == 7  # only video_0001


def test_multiple_roots_concatenate(synthetic_vrs_root):
    ds = VRSWindowDataset(
        [synthetic_vrs_root, synthetic_vrs_root], frame_step=1, window_stride=1
    )
    assert len(ds) == 18
