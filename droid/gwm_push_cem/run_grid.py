"""run_grid: uniform endpoint-lattice sweep -> score map data.

Scores every feasible lattice endpoint under every prompt, both cameras, and
writes results/<out>. Scores go through the shared ScoreCache, so a coarse
sweep followed by a fine one only pays for the points the coarse sweep missed,
and a later CEM run that snaps to the same lattice pays nothing at all.
"""

import argparse
import json

import numpy as np

from config import GRID_STEP, PROMPTS, REGION, RESULTS
from pushing import PushKinematics, ScoreCache, load_views, score_points, snap


def grid_points(step):
    xs = np.round(np.arange(REGION[0], REGION[1] + 1e-9, step), 4)
    ys = np.round(np.arange(REGION[2], REGION[3] + 1e-9, step), 4)
    return ([snap(float(x), float(y)) for x in xs for y in ys],
            list(map(float, xs)), list(map(float, ys)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated prompt tags")
    ap.add_argument("--step", type=float, default=GRID_STEP)
    ap.add_argument("--cache", default="cache_main.json")
    ap.add_argument("--out", default="grid.json")
    ap.add_argument("--server-url", default=None)
    args = ap.parse_args()

    prompts = dict(PROMPTS)
    if args.only:
        keep = {t.strip() for t in args.only.split(",")}
        prompts = {k: v for k, v in prompts.items() if k in keep}

    views, _ = load_views()
    kin = PushKinematics()
    cache = ScoreCache(RESULTS / args.cache)
    pts, xs, ys = grid_points(args.step)
    feasible = [p for p in pts if kin.candidate(*p) is not None]
    print(f"grid {len(xs)}x{len(ys)} = {len(pts)} points "
          f"({len(feasible)} feasible), prompts: {list(prompts)}")

    kw = {"server_url": args.server_url} if args.server_url else {}
    out = {"xs": xs, "ys": ys, "grid_step": args.step, "prompts": {}}
    for tag, instruction in prompts.items():
        print(f"[{tag}] {instruction!r}")
        res = score_points(views, kin, pts, instruction, cache, **kw)
        out["prompts"][tag] = {
            "instruction": instruction,
            "points": {f"{p[0]},{p[1]}": e for p, e in res.items()},
        }
        vals = [e["score"] for e in res.values() if e is not None]
        print(f"[{tag}] scored {len(vals)}/{len(pts)} feasible points, "
              f"score range {min(vals):+.4f}..{max(vals):+.4f}")
    (RESULTS / args.out).write_text(json.dumps(out))
    print(f"wrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
