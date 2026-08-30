"""Read finished runs (sim-free): sample efficiency, cost of learning, curves.

    python report.py 'experiments/pickbowl-*'            # table
    python report.py 'experiments/pickbowl-*' --plot curves.png

Per run: the first sweep at or above each success level (env steps = ticks
summed over envs, and wall seconds), the best sweep, the held-out final
evaluation if the run did one, and the exploration cost totals; then the
mean +/- sd over seeds of the steps to each level (n = seeds that got there).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

LEVELS = ("0.5", "0.9", "0.95", "0.99", "1")


def load_runs(patterns):
    runs = []
    for pattern in patterns:
        for d in sorted(glob.glob(pattern)):
            f = Path(d) / "evals.json"
            if not f.exists():
                continue
            data = json.loads(f.read_text())
            cfg = data.get("config", {})
            runs.append({
                "name": Path(d).name, "action": "macro", "seed": cfg.get("seed"),
                "data": data,
            })
    return runs


def fmt_steps(v):
    return "-" if v is None else f"{v / 1e6:.2f}M"


def run_row(run):
    s = run["data"]["summary"]
    ms = s.get("milestones", {})
    best = s.get("best", {})
    ce = s.get("cost_explore", {})
    final = s.get("final_eval") or {}
    row = {
        "run": run["name"], "action": run["action"], "seed": run["seed"],
        "ticks/s": s.get("env_steps_per_s"), "ticks": s.get("env_steps"),
        "best": best.get("value"), "best_at": best.get("env_steps"),
        "final": final.get("success_at_end", {}).get("mean"),
        "final_sd": final.get("success_at_end", {}).get("sd"),
        "episodes": ce.get("episodes"), "impulse": ce.get("obstacle_impulse"),
        "contact_ticks": ce.get("contact_ticks"),
        "contact_frac": (ce.get("contact_episodes", 0.0) / ce["episodes"]) if ce.get("episodes") else None,
        "block_disp": (ce.get("block_disp", 0.0) / ce["episodes"]) if ce.get("episodes") else None,
    }
    for level in LEVELS:
        m = ms.get(level)
        row[f"steps@{level}"] = m["env_steps"] if m else None
        row[f"wall@{level}"] = m["wall_time_s"] if m else None
    return row


def mean_sd(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return m, sd, len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--plot", default=None, help="write success / cost curves to this png")
    args = ap.parse_args()
    runs = load_runs(args.patterns)
    if not runs:
        raise SystemExit("no runs with evals.json matched")
    rows = [run_row(r) for r in runs]

    print("| run | action | seed | ticks/s | " + " | ".join(f"steps@{l}" for l in LEVELS)
          + " | best (at) | held-out | episodes | impulse N.s | contact ticks | contact ep % | block disp m |")
    print("|---" * (10 + len(LEVELS)) + "|")
    for r in rows:
        print(f"| {r['run']} | {r['action']} | {r['seed']} | {r['ticks/s']} | "
              + " | ".join(fmt_steps(r[f'steps@{l}']) for l in LEVELS)
              + f" | {r['best']:.4f} ({fmt_steps(r['best_at'])}) | "
              + (f"{r['final']:.4f}+/-{r['final_sd']:.4f}" if r["final"] is not None else "-")
              + f" | {r['episodes']:.0f} | {r['impulse']:.0f} | {r['contact_ticks']:.0f} | "
              + (f"{100 * r['contact_frac']:.1f}" if r["contact_frac"] is not None else "-")
              + f" | {r['block_disp']:.3f} |" if r["block_disp"] is not None else " | - |")

    print("\nper action space, over seeds (mean +/- sd, n reached):")
    for action in sorted({r["action"] for r in rows}):
        sub = [r for r in rows if r["action"] == action]
        parts = []
        for level in LEVELS:
            m, sd, n = mean_sd([r[f"steps@{level}"] for r in sub])
            parts.append(f"@{level}: " + (f"{m / 1e6:.2f}+/-{sd / 1e6:.2f}M (n={n}/{len(sub)})" if m else f"- (0/{len(sub)})"))
        mi, si, _ = mean_sd([r["impulse"] for r in sub])
        mc, sc, _ = mean_sd([r["contact_frac"] for r in sub])
        print(f"  {action:5s} " + "  ".join(parts))
        print(f"        exploration cost: impulse {mi:.0f}+/-{si:.0f} N.s, "
              f"{100 * mc:.1f}+/-{100 * sc:.1f}% episodes with obstacle contact")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        colors = {"macro": "tab:blue", "tick": "tab:orange"}
        for run in runs:
            hist = run["data"]["history"]
            c = colors.get(run["action"], "gray")
            xs = [h["env_steps"] / 1e6 for h in hist if "eval" in h]
            ys = [h["eval"]["success_at_end"] for h in hist if "eval" in h]
            axes[0].plot(xs, ys, marker="o", ms=3, color=c, alpha=0.8, label=f"{run['action']} s{run['seed']}")
            xs = [h["env_steps"] / 1e6 for h in hist]
            axes[1].plot(xs, [h["rollout"].get("obstacle_impulse", float("nan")) for h in hist], color=c, alpha=0.8)
            axes[2].plot(xs, [h["cumulative"]["explore"]["obstacle_impulse"] for h in hist], color=c, alpha=0.8)
        axes[0].set(xlabel="env steps (M ticks)", ylabel="success_at_end (deterministic sweep)", ylim=(0, 1.02))
        axes[1].set(xlabel="env steps (M ticks)", ylabel="obstacle impulse per exploration episode (N.s)")
        axes[2].set(xlabel="env steps (M ticks)", ylabel="cumulative exploration impulse (N.s)")
        axes[0].legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
