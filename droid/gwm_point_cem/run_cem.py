"""run_cem: CEM over hover (x, y), one run per prompt.

Vanilla cross-entropy method: sample a Gaussian, keep the elite fraction,
refit mean/sigma, repeat. Samples snap to the 1 cm lattice and share the
score cache with the grid sweep. The full iteration history -- every sample,
its objective, the fitted mean/sigma -- is saved verbatim.

Objective: the fused GWM score (--objective raw, default); --objective lang
uses score minus the empty-instruction prior instead. Infeasible (IK-fail)
samples score -inf; the Gaussian is clipped to the search REGION.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from config import CELLS, IMG_SIZE, REGION, RESULTS
from pointing import PointerKinematics, ScoreCache, load_views, score_points, snap

POP = 30
ITERS = 5
ELITE = 8
SIGMA0 = 0.10
SIGMA_FLOOR = 0.008
SEEDS = {tag: i for i, tag in enumerate(CELLS)}  # fixed per-prompt seeds


def objective_of(entry, objective):
    if entry is None:
        return -np.inf
    return entry["score"] if objective == "raw" else entry["score"] - entry["prior"]


def cem_run(tag, instruction, views, kin, cache, rng, server_url=None,
            objective="raw", pop=POP, iters=ITERS, elite_n=ELITE):
    mean = np.array([(REGION[0] + REGION[1]) / 2, (REGION[2] + REGION[3]) / 2])
    sigma = np.array([SIGMA0, SIGMA0])
    kw = {"server_url": server_url} if server_url else {}
    history = []
    for it in range(iters):
        raw = rng.normal(mean, sigma, size=(pop, 2))
        raw[:, 0] = np.clip(raw[:, 0], REGION[0], REGION[1])
        raw[:, 1] = np.clip(raw[:, 1], REGION[2], REGION[3])
        pts = [snap(float(x), float(y)) for x, y in raw]
        res = score_points(views, kin, pts, instruction, cache,
                           log=lambda s: None, **kw)
        objs = np.array([objective_of(res[p], objective) for p in pts])
        order = np.argsort(-objs)
        elite_idx = [i for i in order if np.isfinite(objs[i])][:elite_n]
        elite = np.array([pts[i] for i in elite_idx], dtype=float)
        history.append({
            "iter": it,
            "samples": [list(p) for p in pts],
            "objectives": [None if not np.isfinite(o) else float(o) for o in objs],
            "raw_scores": [None if res[p] is None else res[p]["score"] for p in pts],
            "priors": [None if res[p] is None else res[p]["prior"] for p in pts],
            "mean": mean.tolist(), "sigma": sigma.tolist(),
            "elite": elite.tolist(),
        })
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), SIGMA_FLOOR)
        print(f"  [{tag}] iter {it}: best {np.nanmax(objs[np.isfinite(objs)]):+.4f} "
              f"mean=({mean[0]:.3f},{mean[1]:.3f}) sigma=({sigma[0]:.3f},{sigma[1]:.3f})")

    all_pts, all_objs = [], []
    for h in history:
        for p, o in zip(h["samples"], h["objectives"]):
            if o is not None:
                all_pts.append(p)
                all_objs.append(o)
    winner = all_pts[int(np.argmax(all_objs))]
    cx, cy = CELLS.get(tag, (np.nan, np.nan))
    half = IMG_SIZE / 2

    def in_cell(p, cx, cy):
        return abs(p[0] - cx) <= half and abs(p[1] - cy) <= half

    return {
        "instruction": instruction,
        "objective": objective,
        "pop": pop, "iters": iters, "elite": elite_n,
        "sigma0": SIGMA0, "sigma_floor": SIGMA_FLOOR, "seed": SEEDS.get(tag),
        "history": history,
        "final_mean": mean.tolist(),
        "winner": winner,
        "winner_obj": float(np.max(all_objs)),
        "target_cell": None if tag not in CELLS else [cx, cy],
        "hit_final_mean": None if tag not in CELLS else bool(in_cell(mean, cx, cy)),
        "hit_winner": None if tag not in CELLS else bool(in_cell(winner, cx, cy)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", default=str(RESULTS / "prompts.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--cache", default="cache_main.json")
    ap.add_argument("--out", default="cem.json")
    ap.add_argument("--objective", default="raw", choices=["raw", "lang"])
    ap.add_argument("--pop", type=int, default=POP)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--elite", type=int, default=ELITE)
    ap.add_argument("--server-url", default=None)
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts_file).read_text())
    if args.only:
        keep = {t.strip() for t in args.only.split(",")}
        prompts = {k: v for k, v in prompts.items() if k in keep}

    views, q_init = load_views()
    kin = PointerKinematics(q_init)
    cache = ScoreCache(RESULTS / args.cache)
    out = {}
    for tag, instruction in prompts.items():
        print(f"[{tag}] CEM on {instruction!r}")
        rng = np.random.default_rng(SEEDS.get(tag, 99))
        out[tag] = cem_run(tag, instruction, views, kin, cache, rng,
                           server_url=args.server_url, objective=args.objective,
                           pop=args.pop, iters=args.iters, elite_n=args.elite)
        hit = out[tag]["hit_final_mean"]
        print(f"[{tag}] final mean {out[tag]['final_mean']} "
              f"target {out[tag]['target_cell']} hit={hit}")
    (RESULTS / args.out).write_text(json.dumps(out))
    print(f"wrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
