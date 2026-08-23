"""make_headline: the adopted V-JEPA 2-AC number in the scene-6 table's format.

Headline = config w32_s4 (faithful preprocessing, 32-frame context, 0.267 s
actions, external_cam_2, flange state), final-frame energy, GWM's two-stage
rule (object by mean energy -> M2T2 confidence -> grasp gate), one designated
goal per task, goal-producing candidate excluded from the pool. Selection and
fixed-plan replay are both deterministic, so each task's outcome is written
as 5 identical trials to match the 14 x 5 layout of droid_sim_ret
(parse_results.py then prints the same per-task / subtotal / total rows).

    .venv/bin/python make_headline.py      # -> vjepa_ret/headline/results_vjepa_<tag>.csv + headline.json
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "vjepa_ret" / "headline"
KEY = "pred/final:final/two_stage"
CONFIG = "w32_s4"
TRIALS = 5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for fam, prefix in (("pick", "refer6"), ("place", "place")):
        sel = json.loads((HERE / "runs" / f"vjepa_{fam}" / CONFIG / "selection.json").read_text())
        per_task = sel["results"][KEY]["excl"]
        for tg, d in per_task.items():
            s = d["single"]
            task_id = f"{prefix}_{tg}"
            detail = {"arm": "vjepa2ac_predicted", "config": CONFIG, "energy": "final", "rule": "two_stage",
                      "goal": s["goal"], "chosen": s["chosen"], "chosen_target": s["chosen_target"],
                      "target": d["target"], "object_correct": s["object_correct"],
                      "replicated_trials": "deterministic selection + deterministic replay"}
            with open(OUT / f"results_vjepa_{tg}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["task", "trial", "success", "plan_failed", "detail"])
                w.writeheader()
                for i in range(TRIALS):
                    w.writerow({"task": task_id, "trial": i, "success": bool(s["success"]), "plan_failed": False,
                                "detail": json.dumps(detail)})
            summary[task_id] = {"target": d["target"], "instruction": d["instruction"], "goal": s["goal"],
                                "chosen": s["chosen"], "chosen_target": s["chosen_target"],
                                "object_correct": s["object_correct"], "success": s["success"],
                                "trials": f"{TRIALS if s['success'] else 0}/{TRIALS}"}
    pick = sum(TRIALS for k, v in summary.items() if k.startswith("refer6") and v["success"])
    place = sum(TRIALS for k, v in summary.items() if k.startswith("place") and v["success"])
    (OUT / "headline.json").write_text(json.dumps({
        "arm": "V-JEPA 2-AC (predicted)", "config": CONFIG, "selection": KEY, "protocol": "single goal, goal excluded",
        "pick": f"{pick}/50", "place": f"{place}/20", "total": f"{pick + place}/70", "tasks": summary}, indent=1))
    print(f"pick {pick}/50  place {place}/20  total {pick + place}/70")


if __name__ == "__main__":
    main()
