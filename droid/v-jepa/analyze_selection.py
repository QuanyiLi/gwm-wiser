"""analyze_selection: trajectory-selection accuracy of V-JEPA 2-AC from the
energy tensors written by score_vjepa.py.

For every task (target cluster k) the goal image comes from a candidate g of
cluster k that succeeded in the replay; the selector must pick a candidate from
energies alone. Two selection rules, both reported:

  argmin     candidate = argmin_c E(c, g)
  two-stage  object = argmin over clusters of the MEAN energy of its candidates
             (what GWM does with its score), then the candidate with the
             highest M2T2 confidence inside that object, re-picked through the
             grasp gate where GWM applied one (pick pool, gate.json)

Goal banks: `final` (goal = the executed end frame after the hold) with
time-aggregations `final` (predicted end frame vs goal, headline), `min`
(best frame along the rollout), `close` (the frame ~1.3 s after the gripper
close); `lift` (pick only; goal = the frame 3 s after the goal candidate's
close command) with `own_lift` (the candidate's own frame at its lift time)
and `min`; `h1.5` / `h3` / `h6` (goal = the goal candidate's frame H seconds
in; the rollout is only used up to H) with `at_h` (the candidate's predicted
frame at H), `min` and `mean` over t <= H.

Arms: `pred` uses the world model's open-loop rollout; `oracle` uses the
encoder on the actually executed frames.

Pools: `excl` (the goal-producing candidate is removed from the pool -- it
would trivially win the oracle arm) and `incl` (kept; the handover asks for
both). Goal protocols: `loo` (every successful candidate of the target serves
as the goal once, accuracy pooled), `single` (one designated goal per task:
the successful target candidate with the highest M2T2 confidence, first in
file order on ties). Outputs: selection.json, summary.md,
csv/results_vjepa_*.csv (harness CSV schema; trial = goal index).

    .venv/bin/python analyze_selection.py --family pick --energy-dir runs/vjepa_pick/w32_s4 \
        --plans-dir ../gwm_integrate_doc/proposals/scene6_rev2
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from vjepa_sel.tasks import candidate_success, clusters, tasks

ARMS = ("pred", "oracle")
RULES = ("argmin", "two_stage")
POOLS = ("excl", "incl")
CLOSE_DELAY_FRAMES = 5  # 20-step gripper action = 5 frames at stride 4
HEADLINE = "pred/final:final/two_stage"


def aggregate(E, n_frames, close_frame, lift_frame, agg, stride_frames=1, t_end=None):
    """E [C, G, T] -> S [C, G] per time aggregation; t_end fixes the last usable frame (horizon banks)."""
    C, G, T = E.shape
    out = np.full((C, G), np.nan, np.float32)
    for c in range(C):
        n = int(n_frames[c]) if t_end is None else int(t_end) + 1
        if agg in ("final", "at_h"):
            out[c] = E[c, :, n - 1]
        elif agg == "min":
            out[c] = np.nanmin(E[c, :, :n], axis=1)
        elif agg == "mean":
            out[c] = np.nanmean(E[c, :, :n], axis=1)
        elif agg == "close":
            t = n - 1 if close_frame[c] < 0 else min(n - 1, int(close_frame[c]) + max(1, CLOSE_DELAY_FRAMES // stride_frames))
            out[c] = E[c, :, t]
        elif agg == "own_lift":
            out[c] = E[c, :, int(lift_frame[c])]
        else:
            raise ValueError(agg)
    return out


def select(scores, pool, cands, gate, rule):
    """scores [C] (energy vs one goal, lower is better); returns the chosen index from `pool`."""
    if rule == "argmin":
        return min(pool, key=lambda c: scores[c])
    by_obj = {}
    for c in pool:
        by_obj.setdefault(cands[c]["target"], []).append(c)
    obj = min(by_obj, key=lambda o: float(np.mean([scores[c] for c in by_obj[o]])))
    members = by_obj[obj]
    if gate is not None:
        passing = [c for c in members if gate.get(cands[c]["file"], False)]
        if passing:
            members = passing
    # highest M2T2 confidence; energy (lower) as tie-break
    return min(members, key=lambda c: (-(cands[c]["conf"] if cands[c]["conf"] is not None else -np.inf), scores[c]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["pick", "place"], required=True)
    ap.add_argument("--energy-dir", type=Path, required=True)
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--tag", default=None, help="config tag for the report (default: energy-dir name)")
    args = ap.parse_args()
    tag = args.tag or args.energy_dir.name
    fam = args.family
    ez = np.load(args.energy_dir / "energies.npz")
    cfg = json.loads((args.energy_dir / "config.json").read_text())
    names = [str(n) for n in ez["names"]]
    targets = [str(t) for t in ez["targets"]]
    confs = ez["confidences"]
    C = len(names)
    judges = cfg["judges"]
    cands = []
    for i, n in enumerate(names):
        stem = Path(n).stem
        cands.append({"file": n, "stem": stem, "target": targets[i],
                      "conf": None if np.isnan(confs[i]) else float(confs[i]),
                      "own_success": candidate_success(fam, judges[stem], targets[i]),
                      "judge": judges[stem]})
    gate = None
    if fam == "pick" and (args.plans_dir / "gate.json").exists():
        g = json.loads((args.plans_dir / "gate.json").read_text())
        gate = {k: bool(v.get("pass")) for k, v in g["results"].items()}
    stride_frames = cfg["stride"] // 4
    n_frames, close_frame = ez["n_frames"], ez["close_frame"]
    lift_frame = ez["lift_frame"] if "lift_frame" in ez else np.full(C, -1)
    has_lift = bool((lift_frame >= 0).all()) and "E_pred_lift" in ez
    banks = {"final": {"pred": ez["E_pred"], "oracle": ez["E_obs"], "aggs": ("final", "min", "close"), "t_end": None}}
    if has_lift:
        banks["lift"] = {"pred": ez["E_pred_lift"], "oracle": ez["E_obs_lift"], "aggs": ("own_lift", "min"), "t_end": None}
    for H in (ez["horizons"].tolist() if "horizons" in ez else []):
        hk = f"h{H:g}"
        banks[hk] = {"pred": ez[f"E_pred_{hk}"], "oracle": ez[f"E_obs_{hk}"], "aggs": ("at_h", "min", "mean"),
                     "t_end": int(ez[f"idx_{hk}"])}

    # ---- per-task selection ----------------------------------------------------
    results = {}  # key "arm/bank:agg/rule" -> {pool: {task: {...}}}
    for arm in ARMS:
        for bank, B in banks.items():
            for agg in B["aggs"]:
                S = aggregate(B[arm], n_frames, close_frame, lift_frame, agg, stride_frames, B["t_end"])
                for rule in RULES:
                    per_pool = {}
                    for pool_mode in POOLS:
                        per_task = {}
                        for tg, instr, k, obj in tasks(fam):
                            goals = [g for g in range(C) if targets[g] == k and cands[g]["own_success"]]
                            rows = []
                            for g in goals:
                                pool = [c for c in range(C) if pool_mode == "incl" or c != g]
                                scores = S[:, g]
                                ch = select(scores, pool, cands, gate, rule)
                                rows.append({
                                    "goal": names[g], "chosen": names[ch], "chosen_target": targets[ch],
                                    "object_correct": targets[ch] == k,
                                    "success": candidate_success(fam, judges[cands[ch]["stem"]], k),
                                    "energies": {names[c]: float(scores[c]) for c in range(C)},
                                })
                            single = None
                            if goals:
                                gstar = max(goals, key=lambda g: ((cands[g]["conf"] if cands[g]["conf"] is not None else -np.inf), -g))
                                single = [r for r in rows if r["goal"] == names[gstar]][0]
                            per_task[tg] = {"target": k, "instruction": instr, "n_goals": len(goals),
                                            "loo": rows, "single": single}
                        per_pool[pool_mode] = per_task
                    results[f"{arm}/{bank}:{agg}/{rule}"] = per_pool

    # ---- summaries ---------------------------------------------------------------
    def tally(per_task, protocol, key):
        num = den = 0
        for tg, d in per_task.items():
            rows = d["loo"] if protocol == "loo" else ([d["single"]] if d["single"] else [])
            num += sum(1 for r in rows if r[key])
            den += len(rows)
        return num, den

    cl = clusters(fam)
    lines = [f"# V-JEPA 2-AC selection — {fam} — config `{tag}`", "",
             f"window={cfg['window']} frames, stride={cfg['stride']} control steps, cam={cfg['cam']}, crop={cfg['crop_mode']}", "",
             "## Candidate pool (replayed once each)", "",
             "| candidate | cluster | object | M2T2 conf | gate | own success |", "|---|---|---|---|---|---|"]
    for c in cands:
        j = c["judge"]
        if fam == "pick":
            moved = {k: v["z_rel"] for k, v in j["lifted"].items() if abs(v["z_rel"]) > 0.02}
            own = f"{c['own_success']} (z_rel {moved})"
        else:
            own = f"{c['own_success']} (landed_in {j['place']['landed_in']})"
        lines.append(f"| {c['file']} | {c['target']} | {cl.get(c['target'])} | {c['conf']} | "
                     f"{'' if gate is None else gate.get(c['file'])} | {own} |")
    n_goals = {tg: d["n_goals"] for tg, d in results[HEADLINE]["excl"].items()}
    lines += ["", f"Goal candidates per task (successful candidates of the target): {n_goals}", "",
              "## Selection accuracy — object correct / execution success",
              "", "LOO = every successful target candidate serves as the goal once (pooled over tasks); "
              "single = one designated goal per task. `excl` removes the goal-producing candidate from the pool, `incl` keeps it.", "",
              "| arm | goal bank : energy | rule | LOO obj excl | LOO succ excl | single obj excl | single succ excl | LOO obj incl | LOO succ incl | single obj incl | single succ incl |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for key, per_pool in results.items():
        arm, bankagg, rule = key.split("/")
        cells, s = [], {}
        for pool_mode in POOLS:
            pt = per_pool[pool_mode]
            for proto in ("loo", "single"):
                for m in ("object_correct", "success"):
                    num, den = tally(pt, proto, m)
                    s[f"{pool_mode}_{proto}_{m}"] = [num, den]
                    cells.append(f"{num}/{den}")
        summary[key] = s
        lines.append(f"| {arm} | {bankagg} | {rule} | " + " | ".join(cells) + " |")

    per_task_keys = [HEADLINE, "pred/final:final/argmin", "oracle/final:final/two_stage", "oracle/final:final/argmin"]
    if has_lift:
        per_task_keys += ["pred/lift:own_lift/two_stage", "pred/lift:min/argmin"]
    per_task_keys += [k for k in results if k.startswith("pred/h") and k.endswith(("at_h/argmin", "at_h/two_stage"))]
    for key in per_task_keys:
        per_task = results[key]["excl"]
        lines += ["", f"## Per task — {key} (goal excluded from the pool)", "",
                  "| task | target | single goal | chosen | obj ok | success | LOO obj | LOO success |",
                  "|---|---|---|---|---|---|---|---|"]
        for tg, d in per_task.items():
            s = d["single"]
            lo = sum(r["object_correct"] for r in d["loo"])
            ls = sum(r["success"] for r in d["loo"])
            n = len(d["loo"])
            if s is None:
                lines.append(f"| {tg} | {d['target']} | (no successful goal candidate) | | | | {lo}/{n} | {ls}/{n} |")
            else:
                lines.append(f"| {tg} | {d['target']} | {s['goal']} | {s['chosen']} ({s['chosen_target']}) | "
                             f"{s['object_correct']} | {s['success']} | {lo}/{n} | {ls}/{n} |")

    # energy matrices for both arms and banks: rows candidates, cols goals
    for bank, B in banks.items():
        agg = {"final": "final", "lift": "own_lift"}.get(bank, "at_h")
        for arm in ARMS:
            S = aggregate(B[arm], n_frames, close_frame, lift_frame, agg, stride_frames, B["t_end"])
            lines += ["", f"## Energy matrix — goal bank `{bank}`, energy `{agg}`, arm `{arm}` (rows: candidate rolled out / observed; cols: goal candidate)", "",
                      "| cand \\ goal | " + " | ".join(Path(n).stem.replace("plan_", "") for n in names) + " |",
                      "|---|" + "---|" * C]
            for c in range(C):
                lines.append(f"| {Path(names[c]).stem.replace('plan_', '')} | " + " | ".join(f"{S[c, g]:.3f}" for g in range(C)) + " |")
            # structure diagnostics: how much of the variance is row (candidate drift) vs column (goal) vs interaction
            M = S.copy()
            np.fill_diagonal(M, np.nan)
            mu = np.nanmean(M)
            row = np.nanmean(M, axis=1, keepdims=True) - mu
            col = np.nanmean(M, axis=0, keepdims=True) - mu
            inter = M - mu - row - col
            tot = np.nanvar(M)
            lines += ["", f"variance decomposition (diagonal excluded): candidate-row effect {np.nanvar(np.broadcast_to(row, M.shape)[~np.isnan(M)])/tot:.0%}, "
                      f"goal-column effect {np.nanvar(np.broadcast_to(col, M.shape)[~np.isnan(M)])/tot:.0%}, interaction (the only part that can select) {np.nanvar(inter[~np.isnan(M)])/tot:.0%}; "
                      f"mean same-object energy {np.nanmean([M[c, g] for c in range(C) for g in range(C) if c != g and targets[c] == targets[g]]):.3f} "
                      f"vs different-object {np.nanmean([M[c, g] for c in range(C) for g in range(C) if targets[c] != targets[g]]):.3f}"]

    # predictor-vs-reality diagnostic
    Et, Es = ez["E_track"], ez["E_still"]
    lines += ["", "## Predictor vs reality along each candidate's own trajectory", "",
              "E_track = |z_pred(t) - z_obs(t)|, E_still = |z_obs(0) - z_obs(t)| (the 'nothing moved' baseline). "
              "If E_track is not below E_still the rollout is not tracking what the simulator shows.", "",
              "| candidate | frames | E_track t=1 | E_still t=1 | E_track final | E_still final | E_track mean | E_still mean | E_cur (current vs own goal) | E_pred own goal (final) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in range(C):
        n = int(n_frames[c])
        lines.append(f"| {Path(names[c]).stem} | {n} | {Et[c, 1]:.4f} | {Es[c, 1]:.4f} | {Et[c, n-1]:.4f} | {Es[c, n-1]:.4f} | {np.nanmean(Et[c, :n]):.4f} | "
                     f"{np.nanmean(Es[c, :n]):.4f} | {ez['E_cur'][c]:.4f} | {ez['E_pred'][c, c, n-1]:.4f} |")

    # oracle goal-bank discriminability: nearest-neighbour object agreement among end frames
    S = aggregate(banks["final"]["oracle"], n_frames, close_frame, lift_frame, "final", stride_frames)
    nn_ok = 0
    for g in range(C):
        pool = [c for c in range(C) if c != g]
        nn = min(pool, key=lambda c: S[c, g])
        nn_ok += targets[nn] == targets[g]
    lines += ["", f"End-frame nearest-neighbour object agreement among the {C} executed candidates: {nn_ok}/{C} "
              "(does the image-goal cost separate objects at all?)"]

    (args.energy_dir / "selection.json").write_text(json.dumps({
        "tag": tag, "family": fam, "config": {k: v for k, v in cfg.items() if k not in ("judges", "sequences")},
        "candidates": [{k: v for k, v in c.items() if k != "judge"} for c in cands],
        "summary": summary, "results": results, "nn_object_agreement": [nn_ok, C],
    }, indent=1))
    (args.energy_dir / "summary.md").write_text("\n".join(lines) + "\n")

    # harness-style CSVs (goal excluded; LOO rows as trials)
    csv_dir = args.energy_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    for key in (HEADLINE, "pred/final:final/argmin", "oracle/final:final/two_stage", "oracle/final:final/argmin"):
        arm, bankagg, rule = key.split("/")
        for tg, d in results[key]["excl"].items():
            task_id = f"{'refer6' if fam == 'pick' else 'place'}_{tg}"
            with open(csv_dir / f"results_vjepa_{arm}_{rule}_{tg}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["task", "trial", "success", "plan_failed", "detail"])
                w.writeheader()
                for i, r in enumerate(d["loo"]):
                    w.writerow({"task": task_id, "trial": i, "success": bool(r["success"]), "plan_failed": False,
                                "detail": json.dumps({"goal": r["goal"], "chosen": r["chosen"],
                                                      "chosen_target": r["chosen_target"],
                                                      "object_correct": r["object_correct"], "arm": arm,
                                                      "energy": bankagg, "rule": rule})})
    print("\n".join(lines[:70]))
    print(f"... -> {args.energy_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
