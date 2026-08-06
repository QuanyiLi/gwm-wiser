"""Rendered-tree discovery, held-out hashing, and the window dataset."""

import torch

from real_world_gwm.rendered import (
    RenderedWindowDataset,
    discover_rendered_clips,
    is_heldout,
    split_clips,
)


def test_discovery_finds_complete_clip(synthetic_rendered_root):
    clips = discover_rendered_clips(synthetic_rendered_root)
    assert len(clips) == 1
    assert clips[0].source == "molmobot"
    assert clips[0].n_frames == 60


def test_incomplete_clip_without_meta_is_ignored(synthetic_rendered_root):
    broken = (synthetic_rendered_root / "rendered" / "molmobot"
              / "houseY__ep1__camA" / "robot_only")
    broken.mkdir(parents=True)
    clips = discover_rendered_clips(synthetic_rendered_root)
    assert len(clips) == 1


def test_holdout_hash_is_deterministic_and_camera_independent():
    uid = "molmoact2_droid/ep000123"
    assert is_heldout(uid, 20) == is_heldout(uid, 20)
    # ~2% at 20 permille over a large population
    frac = sum(is_heldout(f"molmobot/x/h{i}/ep0", 20)
               for i in range(10000)) / 10000
    assert 0.01 < frac < 0.03


def test_split_partitions_exactly(synthetic_rendered_root):
    clips = discover_rendered_clips(synthetic_rendered_root)
    train = split_clips(clips, "train")
    heldout = split_clips(clips, "heldout")
    assert len(train) + len(heldout) == len(clips)
    assert split_clips(clips, "all") == clips


def test_window_dataset_yields_rat_samples(synthetic_rendered_root):
    ds = RenderedWindowDataset(synthetic_rendered_root, split="all",
                               jitter_prob=0.0)
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["rgb"].shape == (6, 3, 64, 96)
    assert sample["robot_only"].shape == (6, 3, 64, 96)
    assert torch.equal(sample["condition"][0], sample["rgb"][0])
    assert torch.equal(sample["condition"][1:], sample["robot_only"][1:])
    assert torch.equal(sample["target"], sample["rgb"])
    assert sample["video_id"] == "houseX__ep0__camA"
