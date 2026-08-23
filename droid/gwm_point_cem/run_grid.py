"""run_grid: uniform hover-lattice sweep -> score map data.

Scores every IK-feasible grid point under every prompt in prompts.json, both
cameras, and writes results/<out>. Scores go through the shared ScoreCache;
raw scores and priors are both stored per point.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from config import GRID_STEP, REGION, RESULTS
from pointing import PointerKinematics, ScoreCache, load_views, score_points, snap


def grid_points():
    xs = np.round(np.arange(REGION[0], REGION[1] + 1e-9, GRID_STEP), 4)
    ys = np.round(np.arange(REGION[2], REGION[3] + 1e-9, GRID_STEP), 4)
    return [snap(float(x), float(y)) for x in xs for y in ys], list(map(float, xs)), list(map(float, ys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", default=str(RESULTS / "prompts.json"))
    ap.add_argument("--only", default=None, help="comma-separated prompt tags")
    ap.add_argument("--cache", default="cache_main.json")
    ap.add_argument("--out", default="grid.json")
    ap.add_argument("--server-url", default=None)
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts_file).read_text())
    if args.only:
        keep = {t.strip() for t in args.only.split(",")}
        prompts = {k: v for k, v in prompts.items() if k in keep}

    views, q_init = load_views()
    kin = PointerKinematics(q_init)
    cache = ScoreCache(RESULTS / args.cache)
    pts, xs, ys = grid_points()
    print(f"grid {len(xs)}x{len(ys)} = {len(pts)} points, prompts: {list(prompts)}")

    kw = {}
    if args.server_url:
        kw["server_url"] = args.server_url
    out = {"xs": xs, "ys": ys, "prompts": {}, "grid_step": GRID_STEP}
    for tag, instruction in prompts.items():
        print(f"[{tag}] {instruction!r}")
        res = score_points(views, kin, pts, instruction, cache, **kw)
        out["prompts"][tag] = {
            "instruction": instruction,
            "points": {f"{p[0]},{p[1]}": e for p, e in res.items()},
        }
        done = sum(1 for e in res.values() if e is not None)
        print(f"[{tag}] scored {done}/{len(pts)} feasible points")
    (RESULTS / args.out).write_text(json.dumps(out))
    print(f"wrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
