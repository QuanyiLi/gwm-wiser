"""Augmentation: jitter touches full RGB only; flip does not exist."""

import torch

import real_data_train.augment as augment
from real_data_train.augment import jitter_window


def _sample():
    torch.manual_seed(0)
    rgb = torch.rand(6, 3, 8, 10)
    robot = torch.rand(6, 3, 8, 10)
    return {"rgb": rgb, "robot_only": robot}


def test_jitter_changes_full_rgb_but_robot_only_is_byte_identical():
    sample = _sample()
    rgb0 = sample["rgb"].clone()
    robot0 = sample["robot_only"].clone()
    out = jitter_window(sample, jitter_prob=1.0)
    assert not torch.equal(out["rgb"], rgb0)
    assert torch.equal(out["robot_only"], robot0)


def test_zero_probability_is_identity():
    sample = _sample()
    rgb0 = sample["rgb"].clone()
    out = jitter_window(sample, jitter_prob=0.0)
    assert torch.equal(out["rgb"], rgb0)


def test_flip_is_gone_for_good():
    # Render homology: no horizontal flip may exist here.
    assert not hasattr(augment, "augment_window")
    source = open(augment.__file__).read()
    assert "hflip" not in source
