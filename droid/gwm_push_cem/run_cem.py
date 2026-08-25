"""run_cem: independent CEM searches over the slide endpoint, many per prompt.

Vanilla cross-entropy method over the endpoint (x, y): sample a Gaussian
centred on the home position, keep the elite fraction, refit mean/sigma,
repeat. Samples snap to the lattice the score map uses, so a run costs the
server nothing once that lattice is cached and the whole rollout set can be
searched independently rather than sharing one solution.

Each run is one rollout. Two standard ways of reading off "the trajectory CEM
found" are recorded for every run, and both are written out as executable plan
sets:

  winner  the best-scoring sample the run saw (an argmax over everything it
          scored, so the executed trajectory is always one CEM actually rated)
  sample  one draw from the run's converged Gaussian (CEM's answer is a
          distribution over solutions; this executes a draw from it)

Runs differ only in their random seed, so in both cases the spread of the
endpoints is the spread of what the search itself produces -- not noise
injected at execution time.

Objective: the fused GWM score minus the empty-instruction prior (--objective
lang, default); --objective raw uses the fused score directly. Infeasible
samples score -inf; the Gaussian is clipped to the search REGION.
"""

import argparse
import json

import numpy as np

from config import HOME_XY, LATTICE_STEP, PROMPTS, REGION, RESULTS
from pushing import PushKinematics, ScoreCache, load_views, score_points, snap

POP = 24
ITERS = 4
ELITE = 6
SIGMA0 = 0.10
# One lattice cell: a floor finer than the lattice would collapse the
# population onto a handful of cells and kill the spread.
SIGMA_FLOOR = max(LATTICE_STEP, 0.01)
ROLLOUTS = 100


def objective_of(entry, objective):
    if entry is None:
        return -np.inf
    return entry["score"] if objective == "raw" else entry["score"] - entry["prior"]


def cem_run(instruction, views, kin, cache, rng, objective, pop, iters, elite_n,
            server_url=None):
    mean = np.array(HOME_XY, dtype=float)
    sigma = np.array([SIGMA0, SIGMA0])
    kw = {"server_url": server_url} if server_url else {}
    history, seen = [], {}
    for it in range(iters):
        raw = rng.normal(mean, sigma, size=(pop, 2))
        raw[:, 0] = np.clip(raw[:, 0], REGION[0], REGION[1])
        raw[:, 1] = np.clip(raw[:, 1], REGION[2], REGION[3])
        pts = [snap(float(x), float(y)) for x, y in raw]
        res = score_points(views, kin, pts, instruction, cache,
                           log=lambda s: None, **kw)
        objs = np.array([objective_of(res[p], objective) for p in pts])
        for p, o in zip(pts, objs):
            if np.isfinite(o):
                seen[p] = float(o)
        order = np.argsort(-objs)
        elite_idx = [i for i in order if np.isfinite(objs[i])][:elite_n]
        if not elite_idx:                      # degenerate draw: keep the prior
            history.append({"iter": it, "mean": mean.tolist(),
                            "sigma": sigma.tolist(), "n_feasible": 0})
            continue
        elite = np.array([pts[i] for i in elite_idx], dtype=float)
        history.append({
            "iter": it,
            "mean": mean.tolist(), "sigma": sigma.tolist(),
            "n_feasible": int(np.isfinite(objs).sum()),
            "best": float(objs[elite_idx[0]]),
            "elite": elite.tolist(),
        })
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), SIGMA_FLOOR)

    if not seen:
        return None
    winner = max(seen, key=seen.get)

    # A draw from the converged Gaussian, resampled until it lands on a
    # feasible endpoint; the fitted mean is the fallback.
    draw = None
    for _ in range(50):
        p = rng.normal(mean, sigma)
        p = snap(float(np.clip(p[0], REGION[0], REGION[1])),
                 float(np.clip(p[1], REGION[2], REGION[3])))
        if kin.candidate(*p) is not None:
            draw = p
            break
    if draw is None:
        draw = snap(*mean)
        if kin.candidate(*draw) is None:
            draw = winner

    return {
        "winner": list(winner),
        "winner_obj": seen[winner],
        "sample": list(draw),
        "sample_obj": seen.get(tuple(draw)),
        "final_mean": mean.tolist(),
        "final_sigma": sigma.tolist(),
        "n_scored": len(seen),
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--rollouts", type=int, default=ROLLOUTS)
    ap.add_argument("--cache", default="cache_main.json")
    ap.add_argument("--out", default="cem.json")
    ap.add_argument("--plans", default="plans.json")
    ap.add_argument("--objective", default="lang", choices=["lang", "raw"])
    ap.add_argument("--pop", type=int, default=POP)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--elite", type=int, default=ELITE)
    ap.add_argument("--server-url", default=None)
    args = ap.parse_args()

    prompts = dict(PROMPTS)
    if args.only:
        keep = {t.strip() for t in args.only.split(",")}
        prompts = {k: v for k, v in prompts.items() if k in keep}

    views, _ = load_views()
    kin = PushKinematics()
    cache = ScoreCache(RESULTS / args.cache)

    cem = {"objective": args.objective, "pop": args.pop, "iters": args.iters,
           "elite": args.elite, "sigma0": SIGMA0, "sigma_floor": SIGMA_FLOOR,
           "rollouts": args.rollouts, "prompts": {}}
    rules = ("winner", "sample")
    rollouts = {rule: {} for rule in rules}
    for tag, instruction in prompts.items():
        runs = []
        for i in range(args.rollouts):
            rng = np.random.default_rng(i)     # paired seeds across prompts
            r = cem_run(instruction, views, kin, cache, rng, args.objective,
                        args.pop, args.iters, args.elite, args.server_url)
            if r is None:
                continue
            r["seed"] = i
            runs.append(r)
            if (i + 1) % 20 == 0:
                print(f"  [{tag}] {i + 1}/{args.rollouts} runs")
        cem["prompts"][tag] = {"instruction": instruction, "runs": runs}
        for rule in rules:
            rollouts[rule][tag] = [
                {"i": k, "seed": r["seed"], "endpoint": r[rule],
                 "obj": r.get(f"{rule}_obj")} for k, r in enumerate(runs)]
            e = np.array([r[rule] for r in runs])
            print(f"[{tag}/{rule}] {len(runs)} runs; endpoint "
                  f"mean=({e[:, 0].mean():.3f},{e[:, 1].mean():.3f}) "
                  f"sd=({e[:, 0].std():.3f},{e[:, 1].std():.3f}) "
                  f"unique={len({tuple(p) for p in e})}")

    (RESULTS / args.out).write_text(json.dumps(cem))

    stem = args.plans.replace(".json", "")
    for rule in rules:
        trajectories = {}
        for rs in rollouts[rule].values():
            for r in rs:
                key = f"{r['endpoint'][0]:.4f},{r['endpoint'][1]:.4f}"
                if key not in trajectories:
                    trajectories[key] = kin.candidate(*r["endpoint"])
        path = RESULTS / f"{stem}_{rule}.json"
        path.write_text(json.dumps({
            "objective": args.objective, "rule": rule,
            "trajectories": trajectories, "rollouts": rollouts[rule],
        }))
        print(f"wrote {path} ({len(trajectories)} distinct trajectories)")
    print(f"wrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
