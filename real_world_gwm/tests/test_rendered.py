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
    from real_world_gwm.rendered import OPERATING_ANCHOR_WH

    aw, ah = OPERATING_ANCHOR_WH
    ds = RenderedWindowDataset(synthetic_rendered_root, split="all",
                               jitter_prob=0.0)
    assert len(ds) >= 1
    sample = ds[0]
    # every sample is anchor-resized regardless of native clip size (D-29)
    assert sample["rgb"].shape == (6, 3, ah, aw)
    assert sample["robot_only"].shape == (6, 3, ah, aw)
    assert torch.equal(sample["condition"][0], sample["rgb"][0])
    assert torch.equal(sample["condition"][1:], sample["robot_only"][1:])
    assert torch.equal(sample["target"], sample["rgb"])
    assert sample["video_id"] == "houseX__ep0__camA"


def test_robot_only_decodes_bit_exact_from_clip_video(synthetic_rendered_root):
    """The decode path must yield the exact frames the renderer wrote
    (anchor resizing happens later, in __getitem__)."""
    from real_world_gwm.lossless_video import read_video_frames

    ds = RenderedWindowDataset(synthetic_rendered_root, split="all",
                               jitter_prob=0.0)
    ci, indices = ds.index[0]
    raw = read_video_frames(ds.clips[ci].robot_only_video, indices)
    win = ds.load_window(ds.clips[ci], indices)
    got = (win["robot_only"].permute(0, 2, 3, 1) * 255).round().byte().numpy()
    assert (raw == got).all()


def test_fixed_scale_dataset_is_deterministic(synthetic_rendered_root):
    """(s, s) with no anchor jitter re-resolves the index once (D-30)."""
    ds = RenderedWindowDataset(synthetic_rendered_root, split="all",
                               jitter_prob=0.0, scale_range=(0.5, 0.5),
                               anchor_jitter_s={})
    assert len(ds) == 1
    _, indices = ds.index[0]
    assert indices == [0, 4, 9, 13, 18, 22]
    assert ds[0]["time_scale"] == 0.5
    # 1.5x span does not fit the 60-frame clip: everything is dropped
    ds_big = RenderedWindowDataset(synthetic_rendered_root, split="all",
                                   jitter_prob=0.0, scale_range=(1.5, 1.5),
                                   anchor_jitter_s={})
    assert len(ds_big) == 0


def test_random_scale_draws_vary_and_fall_back(synthetic_rendered_root):
    import random

    ds = RenderedWindowDataset(synthetic_rendered_root, split="all",
                               jitter_prob=0.0, scale_range=(0.5, 1.0))
    assert len(ds) == 1   # anchor-level index is unchanged by augmentation
    canonical = ds.index[0][1]
    random.seed(3)
    seen = set()
    for _ in range(30):
        sample = ds[0]
        idx = tuple(sample["frame_indices"])
        assert len(idx) == 6 and list(idx) == sorted(set(idx))
        assert 0.5 <= sample["time_scale"] <= 1.0 or idx == tuple(canonical)
        seen.add(idx)
    assert len(seen) > 1   # augmentation actually varies the window


def test_prediscovered_clips_are_reused(synthetic_rendered_root):
    """A caller-supplied clips list must behave exactly like discovery
    (train.py shares one scan across the audit and every split)."""
    clips = discover_rendered_clips(synthetic_rendered_root)
    ref = RenderedWindowDataset(synthetic_rendered_root, split="all",
                                jitter_prob=0.0)
    ds = RenderedWindowDataset(synthetic_rendered_root, split="all",
                               jitter_prob=0.0, clips=clips)
    assert [c.clip_id for c in ds.clips] == [c.clip_id for c in ref.clips]
    assert ds.index == ref.index
    # the sources filter still applies to a pre-discovered list
    empty = RenderedWindowDataset(synthetic_rendered_root, split="all",
                                  jitter_prob=0.0, clips=clips,
                                  sources=["molmoact2_droid"])
    assert len(empty.clips) == 0


def test_per_source_scale_ranges(synthetic_rendered_root):
    import random

    from real_world_gwm.rendered import DEFAULT_SCALE_RANGES, scale_range_for

    # resolver semantics (D-33)
    assert scale_range_for("molmobot", DEFAULT_SCALE_RANGES) == (1.0, 3.0)
    assert scale_range_for("molmoact2_droid",
                           DEFAULT_SCALE_RANGES) == (0.5, 1.5)
    assert scale_range_for("unlisted", DEFAULT_SCALE_RANGES) is None
    assert scale_range_for("any", (2.0, 2.0)) == (2.0, 2.0)
    assert scale_range_for("any", None) is None

    # a dict range drives this source exactly like the tuple form
    probe = RenderedWindowDataset(synthetic_rendered_root, split="all",
                                  jitter_prob=0.0)
    src = probe.clips[0].source
    ds = RenderedWindowDataset(synthetic_rendered_root, split="all",
                               jitter_prob=0.0,
                               scale_range={src: (0.5, 1.0)})
    canonical = tuple(ds.index[0][1])
    random.seed(3)
    seen = set()
    for _ in range(30):
        sample = ds[0]
        idx = tuple(sample["frame_indices"])
        assert 0.5 <= sample["time_scale"] <= 1.0 or idx == canonical
        seen.add(idx)
    assert len(seen) > 1

    # a source missing from the dict stays canonical
    ds_other = RenderedWindowDataset(synthetic_rendered_root, split="all",
                                     jitter_prob=0.0,
                                     scale_range={"someone_else": (0.5, 1.0)})
    sample = ds_other[0]
    assert sample["time_scale"] == 1.0
    assert tuple(sample["frame_indices"]) == tuple(ds_other.index[0][1])
