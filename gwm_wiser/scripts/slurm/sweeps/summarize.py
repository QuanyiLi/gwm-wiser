"""Summarise <ret_root>/<exp>/<value>/{config_*_<split>/episode_metrics.json,
final_results_<split>.json} into <ret_root>/summary.md (+ summary.json). Stdlib only."""
import glob
import json
import os
import sys

ret_root = sys.argv[1] if len(sys.argv) > 1 else "gwm_wiser_exp_ret"
METRICS = ["is_grasped_mean", "tcp_near_goal_mean", "success_at_end_mean"]
EXPS = [("gt_based_replanning", "replan N"), ("gwm_planning_horizon", "horizon H"),
        ("gwm_subsample_exp", "keyframes K")]
out = {}
lines = ["# GT-MPC sweeps (RetrievalBasedPlanner, Qwen3-VL-Embedding-8B)", ""]
for exp, label in EXPS:
    out[exp] = {}
    lines += [f"## {exp} ({label})", "", "| value | split | n_cfg | grasp | reach | success | min | max |",
              "|---|---|---|---|---|---|---|---|"]
    for vdir in sorted(glob.glob(os.path.join(ret_root, exp, "*")), key=lambda p: float(os.path.basename(p))):
        value = os.path.basename(vdir)
        out[exp][value] = {}
        for split in ["train", "test"]:
            cfgs = sorted(glob.glob(os.path.join(vdir, f"config_*_{split}", "episode_metrics.json")))
            if not cfgs:
                continue
            per = [json.load(open(c))["summary"] for c in cfgs]
            row = {m: sum(p[m] for p in per) / len(per) for m in METRICS}
            row["n_cfg"] = len(per)
            row["success_min"] = min(p["success_at_end_mean"] for p in per)
            row["success_max"] = max(p["success_at_end_mean"] for p in per)
            fr = os.path.join(vdir, f"final_results_{split}.json")
            if os.path.exists(fr):
                row["final_results_success"] = json.load(open(fr))["success_at_end_mean"]["avg"]
            out[exp][value][split] = row
            flag = "" if len(per) == 24 else " (INCOMPLETE)"
            lines.append(f"| {value} | {split} | {len(per)}{flag} | {row['is_grasped_mean']:.3f} | "
                         f"{row['tcp_near_goal_mean']:.3f} | {row['success_at_end_mean']:.3f} | "
                         f"{row['success_min']:.3f} | {row['success_max']:.3f} |")
    lines.append("")
json.dump(out, open(os.path.join(ret_root, "summary.json"), "w"), indent=1)
open(os.path.join(ret_root, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
