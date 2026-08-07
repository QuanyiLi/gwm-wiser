"""Timestamped six-frame window enumeration (reject beyond tolerance, no padding)."""

from real_world_gwm.windows import (
    SCHEDULE,
    build_rat_pair,
    enumerate_timed_windows,
    nearest_index,
    resolve_scaled_window,
)


def ts(fps, n):
    return [i / fps for i in range(n)]


def test_15fps_schedule_lands_on_known_indices():
    windows = enumerate_timed_windows(ts(15, 60), stride_s=10.0)
    assert windows == [[0, 8, 17, 26, 35, 44]]


def test_stride_advances_anchor():
    windows = enumerate_timed_windows(ts(15, 75), stride_s=1.0)
    starts = [w[0] for w in windows]
    assert starts == [0, 15, 30]


def test_short_clip_yields_no_window():
    assert enumerate_timed_windows(ts(15, 40), stride_s=1.0) == []


def test_tolerance_rejects_sparse_clock():
    # 2 fps: nearest frame to the 0.55 s offset is 0.25 s away
    assert enumerate_timed_windows(ts(2, 20), stride_s=1.0) == []


def test_molmobot_66ms_clock_is_within_tolerance():
    # policy_dt_ms = 66: worst schedule offset lands 32 ms off (accepted)
    t = [i * 0.066 for i in range(69)]
    windows = enumerate_timed_windows(t, stride_s=3.0)
    assert len(windows) == 1
    w = windows[0]
    t0 = t[w[0]]
    errs = [abs(t[k] - (t0 + off)) for k, off in zip(w, SCHEDULE)]
    assert max(errs) <= 0.033


def test_duplicate_anchor_windows_are_deduped():
    windows = enumerate_timed_windows(ts(15, 60), stride_s=0.01)
    assert len(windows) == len({tuple(w) for w in windows})


def test_nearest_index_boundaries():
    t = [0.0, 1.0, 2.0]
    assert nearest_index(t, -5.0) == 0
    assert nearest_index(t, 0.6) == 1
    assert nearest_index(t, 99.0) == 2


def test_rat_pair_shapes_and_content():
    import torch

    rgb = torch.rand(6, 3, 4, 4)
    robot = torch.rand(6, 3, 4, 4)
    condition, target = build_rat_pair(rgb, robot)
    assert torch.equal(condition[0], rgb[0])
    assert torch.equal(condition[1:], robot[1:])
    assert torch.equal(target, rgb)


def test_resolve_scaled_window_matches_enumeration_at_scale_one():
    t15 = ts(15, 90)
    canonical = enumerate_timed_windows(t15, stride_s=100.0)[0]
    assert resolve_scaled_window(t15, canonical[0], 1.0) == canonical


def test_resolve_scaled_window_compresses_and_rejects():
    t15 = ts(15, 60)
    half = resolve_scaled_window(t15, 0, 0.5)
    assert half == [0, 4, 9, 13, 18, 22]
    # 1.5x span (4.43 s) does not fit a 4 s clip
    assert resolve_scaled_window(t15, 0, 1.5) is None
    # neither does scale 1 anchored too late
    assert resolve_scaled_window(t15, 30, 1.0) is None


def test_resolve_scaled_window_rejects_collapsed_frames():
    # 2 fps: at scale 0.5 the 0.275 s offset collapses onto frame 0
    t2 = ts(2, 40)
    assert resolve_scaled_window(t2, 0, 0.5, tolerance_s=0.3) is None
