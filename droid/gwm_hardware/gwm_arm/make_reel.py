"""Stitch several runs' execution videos into one reel, captioned with the prompt.

A session's videos are one file per turn, in one directory per turn, named by a
truncated tag. Watching a session back means opening seven files and
remembering which instruction each belonged to. This puts them end to end with
the instruction burnt in, read from each run's `turn.json` -- so the caption is
the prompt that was actually issued, not the directory name, which is
truncated to 40 characters and digest-suffixed.

Speed-up is done by DROPPING frames rather than restamping the container's
frame rate: the file then plays at the right speed in anything, including
players that ignore the header.

    python -m gwm_hardware.gwm_arm.make_reel --last 7 --speed 2 --out reel.mp4
    python -m gwm_hardware.gwm_arm.make_reel --runs A B C --speed 1
"""

import argparse
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np

RUNS = Path("droid/gwm_hardware/runs/session")
BAR_H = 96
FONT = cv2.FONT_HERSHEY_SIMPLEX


def turn_of(run: Path) -> dict:
    p = run / "turn.json"
    return json.loads(p.read_text()) if p.exists() else {}


def videos(run: Path):
    return sorted(run.glob("exec_*.mp4"))


def caption(frame, text: str, sub: str, idx: str):
    """Prompt on a bar above the frame, so it never covers the robot."""
    h, w = frame.shape[:2]
    # Wrap to the frame width at this font scale, then grow the bar if needed.
    scale, thick = 0.85, 2
    per_line = max(20, int(w / (scale * 19)))
    lines = textwrap.wrap(text, per_line)[:3]
    bar_h = max(BAR_H, 30 + 34 * len(lines))
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:] = frame
    for i, line in enumerate(lines):
        cv2.putText(out, line, (18, 34 + i * 34), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.putText(out, idx, (w - 150, 30), FONT, 0.7, (140, 200, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (w - 150, 58), FONT, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run dir names; default: --last N")
    ap.add_argument("--last", type=int, default=7, help="most recent N runs that have a video")
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed multiplier")
    ap.add_argument("--fps", type=float, default=15.0, help="output frame rate")
    ap.add_argument("--out", type=Path, default=Path("reel.mp4"))
    ap.add_argument("--runs-root", type=Path, default=RUNS)
    args = ap.parse_args()

    if args.runs:
        picked = [args.runs_root / r for r in args.runs]
    else:
        have = [d for d in sorted(args.runs_root.iterdir()) if d.is_dir() and videos(d)]
        picked = have[-args.last:]
    if not picked:
        raise SystemExit(f"no runs with an exec_*.mp4 under {args.runs_root}")

    step = max(1, int(round(args.speed)))
    if abs(step - args.speed) > 1e-6:
        print(f"speed {args.speed} rounded to {step}x (frames are dropped, not resampled)")

    writer, size = None, None
    total_in = total_out = 0
    print(f"{len(picked)} runs, {step}x:")
    for n, run in enumerate(picked, 1):
        t = turn_of(run)
        instruction = t.get("instruction") or "(instruction not recorded)"
        sub = t.get("outcome", "")
        vids = videos(run)
        if not vids:
            print(f"  {run.name}: no video, skipped"); continue
        cap = cv2.VideoCapture(str(vids[0]))
        kept = read = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read += 1
            if (read - 1) % step:
                continue
            f = caption(frame, instruction, sub, f"{n}/{len(picked)}")
            if writer is None:
                size = (f.shape[1], f.shape[0])
                writer = cv2.VideoWriter(str(args.out),
                                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
                if not writer.isOpened():
                    raise SystemExit(f"cv2 could not open {args.out} for writing")
            if (f.shape[1], f.shape[0]) != size:
                f = cv2.resize(f, size)
            writer.write(f)
            kept += 1
        cap.release()
        total_in += read
        total_out += kept
        print(f"  {n}/{len(picked)}  {run.name}  {read:4d} -> {kept:4d} frames  "
              f"{sub:12s} | {instruction[:64]}")
    if writer is None:
        raise SystemExit("nothing was written -- no readable frames")
    writer.release()
    print(f"\n{total_in} frames in, {total_out} out ({step}x), "
          f"{total_out / args.fps:.1f} s at {args.fps:.0f} fps -> {args.out}")


if __name__ == "__main__":
    main()
