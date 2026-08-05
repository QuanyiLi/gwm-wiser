"""Robot-only RGB derivation and RAT condition/target assembly."""

import torch

from real_world_gwm.adapters.vrs.dataset import (
    build_rat_pair,
    derive_robot_only,
    discover_clips,
    load_window,
)


def test_robot_only_keeps_robot_pixels_and_blacks_background():
    rgb = torch.rand(6, 3, 4, 5)
    mask = torch.zeros(6, 1, 4, 5)
    mask[:, :, :2, :] = 1.0  # top half is robot
    robot_only = derive_robot_only(rgb, mask)
    assert torch.equal(robot_only[:, :, :2, :], rgb[:, :, :2, :])
    assert torch.all(robot_only[:, :, 2:, :] == 0)


def test_rat_condition_is_first_full_frame_then_robot_only_futures():
    rgb = torch.rand(6, 3, 4, 5)
    robot_only = torch.rand(6, 3, 4, 5)
    condition, target = build_rat_pair(rgb, robot_only)
    assert torch.equal(condition[0], rgb[0])
    for t in range(1, 6):
        assert torch.equal(condition[t], robot_only[t])
    assert torch.equal(target, rgb)


def test_load_window_reads_selected_indices_and_masks(synthetic_vrs_root):
    clips, _ = discover_clips(synthetic_vrs_root)
    clip = clips[0]  # 8 frames
    indices = [1, 2, 3, 4, 5, 6]
    sample = load_window(clip, indices)
    assert sample["rgb"].shape == (6, 3, 32, 48)
    assert sample["robot_only"].shape == (6, 3, 32, 48)
    assert sample["mask"].shape == (6, 1, 32, 48)
    assert sample["frame_indices"] == indices
    # spec: background black, robot pixels preserved from the decoded RGB
    m = sample["mask"].bool().expand_as(sample["rgb"])
    assert torch.all(sample["robot_only"][~m] == 0)
    assert torch.equal(sample["robot_only"][m], sample["rgb"][m])
    # masks are binary in the released tree
    assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}
