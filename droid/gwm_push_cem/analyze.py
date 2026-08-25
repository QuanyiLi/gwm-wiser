"""analyze: turn the executed rollouts into the numbers the study is about.

For each prompt: where the CEM endpoints landed, how far and in which direction
the named cube actually moved, how often a cube other than the named one was
disturbed, and how well the mean displacement lines up with the direction the
prompt asked for.

    /root/code/gwm/gwm-wiser/.venv/bin/python analyze.py \
        --exec exec_winner.json --out summary_winner.json
"""

import argparse
import json

import numpy as np

from config import CUBES, DIRECTIONS, HOME_XY, PROMPTS, RESULTS

# Two thresholds, because the outcomes are bimodal. The closed hand is 2.4 cm
# wide at the fingertips but its knuckles sit at table + 4.7 cm, exactly the
# height of a cube top, and are much wider -- so a sweep that passes 6-8 cm
# clear of a cube can still clip it and nudge it a centimetre. Those grazes are
# not what the prompt asked for, and counting them as successes inflates the
# short cube distances, where the crowded cubes are brushed constantly.
MOVED_M = 0.01          # touched at all, grazes included
PUSHED_M = 0.03         # carried by the blade, which is what the prompt asked for
ON_TARGET_DEG = 45.0    # displacement within this of the asked-for direction


def summarize(tag, recs):
    d = np.array([r["disp"][tag][:2] for r in recs])
    finals = np.array([r["final"][tag][:2] for r in recs])
    ends = np.array([r["endpoint"] for r in recs])
    want = np.array(DIRECTIONS[tag])
    mag = np.linalg.norm(d, axis=1)
    moved = mag >= MOVED_M
    pushed = mag >= PUSHED_M
    proj = d @ want
    ang = np.full(len(d), np.nan)
    if moved.any():
        u = d[moved] / mag[moved, None]
        ang[moved] = np.degrees(np.arccos(np.clip(u @ want, -1, 1)))
    others = [t for t in CUBES if t != tag]
    other_mag = np.array([[np.linalg.norm(r["disp"][t][:2]) for t in others]
                          for r in recs])
    return {
        "n": len(recs),
        "instruction": PROMPTS[tag],
        "cube_spawn": list(CUBES[tag]),
        "asked_direction": list(DIRECTIONS[tag]),
        "endpoint_mean": ends.mean(axis=0).round(4).tolist(),
        "endpoint_sd": ends.std(axis=0).round(4).tolist(),
        "endpoint_unique": len({tuple(e) for e in ends.tolist()}),
        "mean_disp_m": d.mean(axis=0).round(4).tolist(),
        "mean_disp_norm_m": round(float(np.linalg.norm(d.mean(axis=0))), 4),
        "mean_along_asked_m": round(float(proj.mean()), 4),
        "median_disp_m": np.median(d, axis=0).round(4).tolist(),
        "final_mean": finals.mean(axis=0).round(4).tolist(),
        "final_sd": finals.std(axis=0).round(4).tolist(),
        "frac_moved": round(float(moved.mean()), 3),
        "frac_pushed": round(float(pushed.mean()), 3),
        "mean_along_asked_pushed_m": (round(float(proj[pushed].mean()), 4)
                                      if pushed.any() else 0.0),
        "median_disp_pushed_m": (round(float(np.median(mag[pushed])), 4)
                                 if pushed.any() else 0.0),
        "frac_on_target_dir_pushed": (round(float(np.nanmean(ang[pushed] <= ON_TARGET_DEG)), 3)
                                      if pushed.any() else 0.0),
        "frac_moved_forward": round(float((proj >= MOVED_M).mean()), 3),
        "frac_on_target_dir": round(float(np.nanmean(ang[moved] <= ON_TARGET_DEG))
                                    if moved.any() else 0.0, 3),
        "median_angle_err_deg": round(float(np.nanmedian(ang[moved])), 1)
        if moved.any() else None,
        "displacement_m": {"mean_abs": round(float(mag.mean()), 4),
                           "max": round(float(mag.max()), 4)},
        "other_cubes_moved_frac": round(float((other_mag >= MOVED_M).any(axis=1).mean()), 3),
        "other_cubes_max_disp_m": round(float(other_mag.max()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", dest="exec_file", default="exec_winner.json")
    ap.add_argument("--out", default="summary_winner.json")
    args = ap.parse_args()

    ex = json.loads((RESULTS / args.exec_file).read_text())
    out = {"home_xy": list(HOME_XY), "moved_threshold_m": MOVED_M,
           "pushed_threshold_m": PUSHED_M,
           "on_target_deg": ON_TARGET_DEG, "prompts": {}}
    for tag, recs in ex.items():
        s = summarize(tag, recs)
        out["prompts"][tag] = s
        print(f"[{tag}] {s['instruction']!r}")
        print(f"  endpoints: mean=({s['endpoint_mean'][0]:.3f},{s['endpoint_mean'][1]:.3f}) "
              f"sd=({s['endpoint_sd'][0]:.3f},{s['endpoint_sd'][1]:.3f}) "
              f"{s['endpoint_unique']} distinct of {s['n']}")
        print(f"  named cube: mean displacement "
              f"({s['mean_disp_m'][0]:+.3f},{s['mean_disp_m'][1]:+.3f}) m, "
              f"{s['mean_along_asked_m']:+.3f} m along the asked direction")
        print(f"  pushed >= {PUSHED_M * 100:.0f} cm in {s['frac_pushed'] * 100:.0f}% "
              f"(touched >= {MOVED_M * 100:.0f} cm in {s['frac_moved'] * 100:.0f}%)")
        print(f"  moved >= {MOVED_M * 100:.0f} cm in {s['frac_moved'] * 100:.0f}% of rollouts; "
              f"{s['frac_on_target_dir'] * 100:.0f}% of those within "
              f"{ON_TARGET_DEG:.0f} deg of the asked direction "
              f"(median error {s['median_angle_err_deg']} deg)")
        print(f"  another cube moved in {s['other_cubes_moved_frac'] * 100:.0f}% of rollouts "
              f"(max {s['other_cubes_max_disp_m'] * 100:.1f} cm)")
    (RESULTS / args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
