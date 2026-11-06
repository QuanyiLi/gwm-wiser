"""
Tables for the candidate-set ablation and the state-diffusion-proposal runs.

    python -m gwm_wiser.local_eval.summarize candidate_set --root /path/candidate_set_ablation \
        --runs full keep6 keep3 drop_correct noise_2cm noise_5cm --reference full --nocorrect drop_correct
    python -m gwm_wiser.local_eval.summarize proposals --root /path/state_diffusion_proposals/eval
"""

import argparse
import glob
import json
import os

import numpy as np


def episode_table(run_dir, split):
    files = sorted(glob.glob(os.path.join(run_dir, f"config_*_{split}", "episode_metrics.json")))
    eps = []
    for f in files:
        eps += json.load(open(f))["episodes"]
    if not eps:
        return None
    m = lambda k: float(np.mean([float(e[k]) for e in eps]))
    return dict(n_configs=len(files), n_episodes=len(eps), success=m("success_once"),
                grasp=m("is_grasped"), reach=m("tcp_near_goal"), success_at_end=m("success_at_end"))


def _derive(r):
    """fp32 scores of the selected candidate and its margin over the best other-mode candidate.

    Candidates that share the stage-relevant slot (same cube while grasping, same placard
    while placing) execute near-identical motions, so the plain top-1 vs top-2 margin is
    often exactly 0; the other-mode margin compares against a genuinely different choice.
    """
    if "cand_eps" not in r or "grasp_cos_fp32" not in r:
        return r
    act = np.array(r["place_cos_fp32"] if r["is_grasped"] else r["grasp_cos_fp32"])
    slot = (lambda ep: ep % 3) if r["is_grasped"] else (lambda ep: ep // 3)
    i = r["cand_eps"].index(r["selected_ep"])
    others = [act[j] for j, ep in enumerate(r["cand_eps"]) if slot(ep) != slot(r["selected_ep"])]
    r["top_cos32"] = float(act[i])
    srt = np.sort(act)[::-1]
    r["margin32"] = float(srt[0] - srt[1]) if len(srt) > 1 else float("nan")
    r["margin32_other_mode"] = float(act[i] - max(others)) if others else float("nan")
    return r


def replans(run_dir, split):
    recs = []
    for f in sorted(glob.glob(os.path.join(run_dir, f"config_*_{split}", "replans.jsonl"))):
        recs += [_derive(json.loads(l)) for l in open(f)]
    return recs


def auroc(pos, neg):
    """P(score_pos > score_neg) with ties counted half (pos = should be accepted)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        idx = allv == v
        ranks[idx] = ranks[idx].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def detection(ref, bad, key, stage=None, first_only=False):
    sel = lambda rs: [r for r in rs if (stage is None or r["stage"] == stage)
                      and (not first_only or r["step"] == 0)]
    a = np.array([r[key] for r in sel(ref) if key in r and np.isfinite(r[key])])
    b = np.array([r[key] for r in sel(bad) if key in r and np.isfinite(r[key])])
    if len(a) == 0 or len(b) == 0:
        return None
    thr = float(np.quantile(a, 0.05))  # keeps 95 % of the reference selections
    return dict(n_ref=len(a), n_bad=len(b), auroc=auroc(a, b), threshold_95=thr,
                rejected_bad=float((b < thr).mean()), ref_mean=float(a.mean()), bad_mean=float(b.mean()))


def candidate_set(args):
    out = {}
    rows = []
    for run in args.runs:
        d = os.path.join(args.root, run)
        t = episode_table(d, args.split)
        rs = replans(d, args.split)
        if t is None:
            continue
        first = [r for r in rs if r["step"] == 0]
        t.update(
            n_replans=len(rs),
            first_replan_correct=float(np.mean([r["selected_correct"] for r in first])) if first else None,
            first_replan_correct_cube=float(np.mean([r["selected_correct_cube"] for r in first])) if first else None,
            all_replan_correct=float(np.mean([r["selected_correct"] for r in rs])) if rs else None,
            mean_candidates=float(np.mean([len(r["cand_eps"]) for r in rs])) if rs else None,
            correct_in_base=float(np.mean([r["correct_in_base"] for r in rs])) if rs else None,
        )
        out[run] = t
        rows.append((run, t))

    det = {}
    if args.nocorrect:
        ref = replans(os.path.join(args.root, args.reference), args.split)
        bad = replans(os.path.join(args.root, args.nocorrect), args.split)
        for name, kw in {
            "fp32_top_cos_all_replans": dict(key="top_cos32"),
            "fp32_top_cos_first_replan": dict(key="top_cos32", first_only=True),
            "fp32_margin_other_mode_all_replans": dict(key="margin32_other_mode"),
            "fp32_margin_other_mode_first_replan": dict(key="margin32_other_mode", first_only=True),
            "fp32_top_cos_grasp_stage": dict(key="top_cos32", stage="grasp"),
            "fp32_top_cos_place_stage": dict(key="top_cos32", stage="place"),
            "fp32_margin_other_mode_grasp_stage": dict(key="margin32_other_mode", stage="grasp"),
            "fp32_margin_other_mode_place_stage": dict(key="margin32_other_mode", stage="place"),
            "active_top_cos_all_replans": dict(key="top_cos"),
            "active_margin_all_replans": dict(key="margin"),
            "active_top_cos_first_replan": dict(key="top_cos", first_only=True),
            "active_margin_first_replan": dict(key="margin", first_only=True),
            "grasp_stage_top_cos": dict(key="top_cos", stage="grasp"),
            "grasp_stage_margin": dict(key="margin", stage="grasp"),
            "place_stage_top_cos": dict(key="top_cos", stage="place"),
            "place_stage_margin": dict(key="margin", stage="place"),
        }.items():
            det[name] = detection(ref, bad, **kw)
    out["_detection"] = det

    lines = ["| run | success | grasp | reach | first-replan correct (episode) | first-replan correct cube | "
             "all-replan correct | mean #cands | correct in KNN set |", "|---" * 9 + "|"]
    f = lambda x: "–" if x is None else f"{x:.3f}"
    for run, t in rows:
        lines.append(f"| {run} | {f(t['success'])} | {f(t['grasp'])} | {f(t['reach'])} | "
                     f"{f(t['first_replan_correct'])} | {f(t['first_replan_correct_cube'])} | "
                     f"{f(t['all_replan_correct'])} | {f(t['mean_candidates'])} | {f(t['correct_in_base'])} |")
    if det:
        lines += ["", f"Detection ({args.reference} vs {args.nocorrect}); threshold keeps 95 % of {args.reference}",
                  "| score | n ref | n no-correct | AUROC | threshold | no-correct rejected | ref mean | no-correct mean |",
                  "|---" * 8 + "|"]
        for k, v in det.items():
            if v:
                lines.append(f"| {k} | {v['n_ref']} | {v['n_bad']} | {v['auroc']:.3f} | {v['threshold_95']:.4f} | "
                             f"{v['rejected_bad']:.3f} | {v['ref_mean']:.4f} | {v['bad_mean']:.4f} |")
    return out, "\n".join(lines)


def proposals(args):
    out, rows = {}, []
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        t = episode_table(d, args.split)
        if t is None:
            continue
        rs = replans(d, args.split)
        first = [r for r in rs if r["step"] == 0]
        grasp_rs = [r for r in rs if r["stage"] == "grasp"]
        place_rs = [r for r in rs if r["stage"] == "place"]

        def cube_hist(rs_):
            h = np.zeros(4)
            for r in rs_:
                for m in r["sample_modes"]:
                    if m[0] is not None:
                        h[m[0]] += 1
            return (h / max(h.sum(), 1)).round(3).tolist()

        def mode_hist(rs_):
            h = np.zeros(12)
            for r in rs_:
                for m in r["sample_modes"]:
                    if m[0] is not None and m[1] is not None:
                        h[m[0] * 3 + m[1]] += 1
            return (h / max(h.sum(), 1)).round(3).tolist()

        t.update(
            name=os.path.basename(d),
            first_replan_target_cube_present=float(np.mean([r["target_stage_present"] for r in first])) if first else None,
            first_replan_full_mode_present=float(np.mean([r["target_mode_present"] for r in first])) if first else None,
            first_replan_selected_target_cube=float(np.mean([r["selected_target_stage"] for r in first])) if first else None,
            grasp_stage_target_present=float(np.mean([r["target_stage_present"] for r in grasp_rs])) if grasp_rs else None,
            place_stage_target_present=float(np.mean([r["target_stage_present"] for r in place_rs])) if place_rs else None,
            place_stage_selected_target=float(np.mean([r["selected_target_stage"] for r in place_rs])) if place_rs else None,
            first_replan_cube_slot_hist=cube_hist(first),
            first_replan_mode_hist=mode_hist(first),
            all_replans_cube_slot_hist=cube_hist(grasp_rs),
            sample_seconds_per_chunk=float(np.mean([r["sample_seconds_per_chunk"] for r in rs])) if rs else None,
        )
        out[t["name"]] = t
        rows.append(t)
    f = lambda x: "–" if x is None else f"{x:.3f}"
    lines = ["| run | success | grasp | reach | 1st replan: target cube among samples | 1st replan: full target mode among samples | "
             "1st replan: selected target cube | place stage: target placard among samples | place stage: selected target placard |",
             "|---" * 10 + "|"]
    for t in rows:
        lines.append(f"| {t['name']} | {f(t['success'])} | {f(t['grasp'])} | {f(t['reach'])} | "
                     f"{f(t['first_replan_target_cube_present'])} | {f(t['first_replan_full_mode_present'])} | "
                     f"{f(t['first_replan_selected_target_cube'])} | {f(t['place_stage_target_present'])} | "
                     f"{f(t['place_stage_selected_target'])} |")
    return out, "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["candidate_set", "proposals"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--reference", default="full")
    ap.add_argument("--nocorrect", default=None)
    a = ap.parse_args()
    res, md = candidate_set(a) if a.kind == "candidate_set" else proposals(a)
    print(md)
    tag = f"{a.kind}_{a.split}"
    with open(os.path.join(a.root, f"summary_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    with open(os.path.join(a.root, f"summary_{tag}.md"), "w") as fh:
        fh.write(md + "\n")
