"""Audit manifest: coverage, exclusions, motion stats, tokens, stable hash."""

from real_world_gwm.audit import build_manifest


def fake_token_counter(clip):
    # injected stand-in for the exact-production-path counter (E2E uses real)
    return {"grid": [3, 16, 32], "tokens": 1536}


def _manifest(root, **kw):
    defaults = dict(
        roots=[root],
        frame_step=1,
        window_stride=1,
        candidate_steps=(1, 2),
        token_counter=fake_token_counter,
        token_ceiling=2048,
        motion_sample_windows=2,
    )
    defaults.update(kw)
    return build_manifest(**defaults)


def test_manifest_covers_clips_and_exclusions(synthetic_vrs_root):
    m = _manifest(synthetic_vrs_root)
    assert m["source"] == "vrs"
    assert m["temporal_sampling"] == {
        "kind": "ordinal",
        "frame_step": 1,
        "window_stride": 1,
    }
    by_id = {c["video_id"]: c for c in m["clips"]}
    assert by_id["video_0001___01_franka___a"]["frame_count"] == 12
    assert by_id["video_0001___01_franka___a"]["valid_windows"] == 7
    assert by_id["video_0001___01_franka___a"]["mask_provenance"] == "mask_gt"
    assert by_id["video_0001___01_franka___a"]["embodiment"] == "01_franka"
    assert by_id["video_0001___01_franka___a"]["frame_size"] == [32, 48]  # h, w
    assert by_id["video_0002___06_ur5___b"]["mask_provenance"] == "mask_gt_dinov3"
    reasons = {e["video_id"]: e["reason"] for e in m["exclusions"]}
    assert "video_0003___01_franka___c" in reasons
    assert "video_0004___01_franka___d" in reasons


def test_manifest_reports_tokens_histogram_and_batch_shapes(synthetic_vrs_root):
    m = _manifest(synthetic_vrs_root)
    clip = m["clips"][0]
    assert clip["qwen_grid"] == [3, 16, 32]
    assert clip["qwen_tokens"] == 1536
    assert m["token_histogram"] == {"1536": 2}
    assert m["batch_shapes"] == [[3, 16, 32]]
    assert m["token_ceiling"] == 2048
    assert m["token_ceiling_violations"] == []


def test_ceiling_violations_are_reported_with_details(synthetic_vrs_root):
    m = _manifest(synthetic_vrs_root, token_ceiling=1000)
    assert len(m["token_ceiling_violations"]) == 2
    v = m["token_ceiling_violations"][0]
    assert v["video_id"] == "video_0001___01_franka___a"
    assert v["tokens"] == 1536
    assert v["grid"] == [3, 16, 32]
    assert v["frame_size"] == [32, 48]


def test_motion_stats_grow_with_frame_step(synthetic_vrs_root):
    m = _manifest(synthetic_vrs_root)
    stats = {
        s["frame_step"]: s
        for s in m["clips"][0]["motion_stats"]  # video_0001, 12 frames
    }
    # synthetic mask moves 3 px/frame, so step-2 displacement must exceed step-1
    assert stats[2]["mask_centroid_disp"] > stats[1]["mask_centroid_disp"] > 0
    assert stats[1]["frame_abs_diff"] > 0


def test_manifest_hash_is_stable_and_config_sensitive(synthetic_vrs_root):
    m1 = _manifest(synthetic_vrs_root)
    m2 = _manifest(synthetic_vrs_root)
    assert m1["manifest_hash"] == m2["manifest_hash"]
    m3 = _manifest(synthetic_vrs_root, frame_step=2)
    assert m3["manifest_hash"] != m1["manifest_hash"]
