"""Ordinal six-frame window enumeration (plan: reject incomplete, no tail repeat)."""

from real_world_gwm.adapters.vrs.dataset import enumerate_windows


def test_exact_six_frames_step1_yields_single_window():
    assert enumerate_windows(6, frame_step=1, window_stride=1) == [[0, 1, 2, 3, 4, 5]]


def test_step1_stride1_slides_by_one():
    ws = enumerate_windows(8, frame_step=1, window_stride=1)
    assert ws == [
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7],
    ]


def test_step2_requires_span_of_eleven_frames():
    # span = 5 * step + 1 = 11
    assert enumerate_windows(11, frame_step=2, window_stride=1) == [
        [0, 2, 4, 6, 8, 10]
    ]


def test_incomplete_window_is_rejected_not_padded():
    assert enumerate_windows(10, frame_step=2, window_stride=1) == []
    assert enumerate_windows(5, frame_step=1, window_stride=1) == []


def test_window_stride_skips_start_indices():
    ws = enumerate_windows(12, frame_step=1, window_stride=3)
    assert ws == [
        [0, 1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7, 8],
        [6, 7, 8, 9, 10, 11],
    ]
