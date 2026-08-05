"""Augmentation: flip consistency; photometric jitter never touches robot-only."""

import torch

from real_world_gwm.augment import augment_window


def _sample():
    torch.manual_seed(0)
    rgb = torch.rand(6, 3, 8, 10)
    mask = (torch.rand(6, 1, 8, 10) > 0.5).float()
    return {"rgb": rgb, "mask": mask, "robot_only": rgb * mask}


def test_jitter_changes_full_rgb_but_robot_only_is_byte_identical():
    sample = _sample()
    rgb0 = sample["rgb"].clone()
    robot0 = sample["robot_only"].clone()
    mask0 = sample["mask"].clone()
    out = augment_window(sample, flip_prob=0.0, jitter_prob=1.0)
    assert not torch.equal(out["rgb"], rgb0)
    assert torch.equal(out["robot_only"], robot0)
    assert torch.equal(out["mask"], mask0)


def test_flip_applies_to_all_streams_consistently():
    sample = _sample()
    rgb0 = sample["rgb"].clone()
    robot0 = sample["robot_only"].clone()
    mask0 = sample["mask"].clone()
    out = augment_window(sample, flip_prob=1.0, jitter_prob=0.0)
    assert torch.equal(out["rgb"], torch.flip(rgb0, dims=[-1]))
    assert torch.equal(out["robot_only"], torch.flip(robot0, dims=[-1]))
    assert torch.equal(out["mask"], torch.flip(mask0, dims=[-1]))


def test_no_augmentation_is_identity():
    sample = _sample()
    rgb0 = sample["rgb"].clone()
    out = augment_window(sample, flip_prob=0.0, jitter_prob=0.0)
    assert torch.equal(out["rgb"], rgb0)
