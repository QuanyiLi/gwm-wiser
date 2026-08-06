"""Timestamped six-frame RAT windows (plan of record, ADR-0012).

Both selected sources have reliable clocks, so windows are enumerated against
elapsed-time offsets — the legacy WISER 3-second schedule — with a hard
tolerance: a window is rejected outright when any scheduled offset has no
frame within the tolerance. There is no ordinal fallback on the training path.
"""

import bisect

import torch

# Legacy 3-second schedule (seconds from the window anchor), decision D-6.
SCHEDULE = (0.0, 0.55, 1.15, 1.75, 2.35, 2.95)
# Half a frame at 15 fps, decision D-6: reject beyond, never snap silently.
DEFAULT_TOLERANCE_S = 0.033


def nearest_index(timestamps, t):
    """Index of the timestamp closest to t (timestamps ascending)."""
    j = bisect.bisect_left(timestamps, t)
    if j == 0:
        return 0
    if j >= len(timestamps):
        return len(timestamps) - 1
    return j if timestamps[j] - t < t - timestamps[j - 1] else j - 1


def enumerate_timed_windows(
    timestamps,
    stride_s: float,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    schedule=SCHEDULE,
):
    """All accepted windows as lists of six frame indices.

    Anchors advance by stride_s seconds from the first frame; each anchor
    snaps to the nearest actual frame. Incomplete or out-of-tolerance
    windows are rejected, never padded.
    """
    if len(timestamps) == 0:
        return []
    windows = []
    span = schedule[-1]
    t_anchor = timestamps[0]
    last_start = timestamps[-1] - span
    while t_anchor <= last_start + tolerance_s:
        i0 = nearest_index(timestamps, t_anchor)
        t0 = timestamps[i0]
        indices, ok = [], True
        for off in schedule:
            k = nearest_index(timestamps, t0 + off)
            if abs(timestamps[k] - (t0 + off)) > tolerance_s:
                ok = False
                break
            indices.append(k)
        if ok and indices[0] <= indices[-1]:
            windows.append(indices)
        t_anchor += stride_s
    # dedupe anchors that snapped to the same frame set
    seen, unique = set(), []
    for w in windows:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return unique


def build_rat_pair(rgb: torch.Tensor, robot_only: torch.Tensor) -> tuple:
    """RAT condition = [rgb[0], robot_only[1:6]]; target = rgb[0:6]."""
    condition = torch.cat([rgb[:1], robot_only[1:]], dim=0)
    return condition, rgb
