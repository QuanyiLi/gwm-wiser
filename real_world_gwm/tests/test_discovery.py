"""Clip discovery: frame ordering, whole-robot mask 002 pairing, provenance."""

from real_world_gwm.adapters.vrs.dataset import discover_clips


def test_discovers_only_clips_with_complete_whole_robot_masks(synthetic_vrs_root):
    clips, excluded = discover_clips(synthetic_vrs_root)
    names = [c.video_id for c in clips]
    assert names == [
        "video_0001___01_franka___a",
        "video_0002___06_ur5___b",
    ]
    reasons = {e["video_id"]: e["reason"] for e in excluded}
    assert "no_whole_robot_mask" in reasons["video_0003___01_franka___c"]
    assert "missing_mask_frames" in reasons["video_0004___01_franka___d"]


def test_frames_are_ordered_by_released_ordinal_index(synthetic_vrs_root):
    clips, _ = discover_clips(synthetic_vrs_root)
    clip = clips[0]
    assert len(clip.rgb_paths) == 12
    assert [p.stem for p in clip.rgb_paths] == [f"{i:05d}" for i in range(1, 13)]
    assert [p.stem for p in clip.mask_paths] == [f"{i:05d}" for i in range(1, 13)]
    assert all(p.parent.name == "002" for p in clip.mask_paths)


def test_mask_provenance_prefers_human_mask_gt(synthetic_vrs_root):
    clips, _ = discover_clips(synthetic_vrs_root)
    by_id = {c.video_id: c for c in clips}
    assert by_id["video_0001___01_franka___a"].mask_provenance == "mask_gt"
    assert by_id["video_0002___06_ur5___b"].mask_provenance == "mask_gt_dinov3"


def test_embodiment_parsed_from_video_id(synthetic_vrs_root):
    clips, _ = discover_clips(synthetic_vrs_root)
    assert clips[0].embodiment == "01_franka"
    assert clips[1].embodiment == "06_ur5"
