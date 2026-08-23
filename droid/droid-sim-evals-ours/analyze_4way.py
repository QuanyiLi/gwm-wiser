"""analyze_4way: comparison table for the four-system scene-6 evaluation.

Reads one results directory holding results_<arm>_<tag>.csv for both families
(pick task ids refer6_*, place ids place_*) and writes a markdown report:
success per task per arm, the place confusion columns (which bin the block
actually reached), and the offline selection provenance of each GWM arm (which
object/plan its viewpoint chose). Partial runs are fine — missing cells show as
"—" and the header records how complete each arm is, so the same script serves
the running summaries and the final report.

    python3 analyze_4way.py --runs runs/eval_4way [--out runs/eval_4way/comparison.md]
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROP_PICK = Path("/root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene6_rev2")
PROP_PLACE = Path("/root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene6_place_v2")

ARMS = [("gwmfusion", "_fusion", "GWM fused (cam1+cam2)"),
        ("gwmcam1", "_cam1", "GWM cam1"),
        ("gwmcam2", "", "GWM cam2"),
        ("tiptop", None, "tiptop (baseline)")]

PICK_TAGS = ["fruit", "yellow", "eat", "negation", "puzzle", "colorful", "nearbowl",
             "eatfrom", "between", "container"]
PICK_TARGET = {"fruit": "banana", "yellow": "banana", "eat": "banana", "negation": "banana",
               "puzzle": "cube", "colorful": "cube", "nearbowl": "cube",
               "eatfrom": "bowl", "between": "bowl", "container": "bowl"}
PLACE_TAGS = ["red", "green", "tomato", "grass"]
PLACE_TARGET = {"red": "red_bin", "green": "green_bin", "tomato": "red_bin", "grass": "green_bin"}
PICK_NAME = {"object_0": "green_bin", "object_1": "bowl", "object_2": "red_bin",
             "object_3": "cube", "object_4": "banana"}
PLACE_NAME = {"object_0": "green_bin", "object_1": "red_bin", "object_2": "cube",
              "object_3": "banana", "object_4": "bowl", "object_5": "bowl"}


def rows(runs: Path, arm: str, tag: str, task_id: str):
    f = runs / f"results_{arm}_{tag}.csv"
    if not f.exists():
        return []
    with open(f) as fh:
        return [r for r in csv.DictReader(fh) if r["task"] == task_id]


def cell(rs, trials):
    if not rs:
        return "—", 0, 0
    s = sum(r["success"] == "True" for r in rs)
    pf = sum(r["plan_failed"] == "True" for r in rs)
    mark = "" if len(rs) >= trials else f" *({len(rs)} run)*"
    return f"{s}/{len(rs)}{mark}", s, len(rs)


def served(prop: Path, prefix: str, tag: str, wsuf, name_map):
    """Which plan/object the arm's offline selection served (None for tiptop)."""
    if wsuf is None:
        return "online", "—"
    sf = prop / f"scores_{prefix}_{tag}{wsuf}.json"
    wf = prop / f"winner_{prefix}_{tag}{wsuf}.json"
    if not sf.exists() or not wf.exists():
        return "—", "—"
    s = json.loads(sf.read_text())
    wh = hashlib.md5(wf.read_bytes()).hexdigest()
    plan = s.get("winner_file", "?")
    for p in prop.glob("plan_*.json"):          # confirm the copy matches the recorded file
        if hashlib.md5(p.read_bytes()).hexdigest() == wh:
            plan = p.name
            break
    return name_map.get(s.get("selected_target"), s.get("selected_target", "?")), plan[:-5]


NOTES = """
## Protocol

Scene 6, 5 trials per task per arm, default speed tier (1 Hz cameras +
per-trial videos, **not** `--fast`). One judge for all four arms: pick =
target lifted >= 0.15 m at episode end; place = block's mesh centre within
0.05 m of the named bin's centre and inside z_rel [-0.03, +0.03].

The three GWM arms share one pipeline and differ only in which scoring
viewpoint chose the plan; they replay a fixed plan, so no planner runs at
trial time. TiPToP replans from scratch every trial (perception + Gemini
grounding + M2T2 + cuTAMP) and on the place family runs its native plan —
pick, carry, open, go home — with the welded block released the moment the
gripper is commanded open (`weld_held_block.maybe_release`). An alternative
protocol (truncate tiptop at its last Place step and judge the held block
exactly like the GWM arms; `TRUNCATE=Place`) is implemented but unused:
tiptop hovers above the bin mouth before releasing, so that protocol would
need a wider z band and would judge "the block hovers over the bin" rather
than "the block lands in the bin".
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE / "runs" / "eval_4way")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    L = []

    L.append("# Scene-6 four-system evaluation — 10 pick + 4 place, 5 trials each\n")
    L.append(f"Results dir `{args.runs}`, default speed tier (videos on, not `--fast`).\n")
    L.append("Arms: three GWM arms replaying the SAME pipeline and differing only in the "
             "scoring viewpoint that chose the plan, plus upstream TiPToP replanning every "
             "trial.\n")

    totals = {a: [0, 0] for a, _, _ in ARMS}

    L.append("\n## Pick (scene 6 variant 0) — success = target lifted ≥ 0.15 m\n")
    L.append("| task | target | " + " | ".join(t for _, _, t in ARMS) + " |")
    L.append("|---|---|" + "---|" * len(ARMS))
    for tag in PICK_TAGS:
        cells = []
        for arm, _, _ in ARMS:
            c, s, n = cell(rows(args.runs, arm, tag, f"refer6_{tag}"), args.trials)
            totals[arm][0] += s
            totals[arm][1] += n
            cells.append(c)
        L.append(f"| {tag} | {PICK_TARGET[tag]} | " + " | ".join(cells) + " |")

    L.append("\n## Place (scene 6 variant 1) — success = block inside the named bin\n")
    L.append("| task | target | " + " | ".join(t for _, _, t in ARMS) + " |")
    L.append("|---|---|" + "---|" * len(ARMS))
    for tag in PLACE_TAGS:
        cells = []
        for arm, _, _ in ARMS:
            rs = rows(args.runs, arm, tag, f"place_{tag}")
            c, s, n = cell(rs, args.trials)
            totals[arm][0] += s
            totals[arm][1] += n
            landed = {}
            for r in rs:
                try:
                    landed[str(json.loads(r["detail"]).get("_landed_in"))] = \
                        landed.get(str(json.loads(r["detail"]).get("_landed_in")), 0) + 1
                except (json.JSONDecodeError, KeyError):
                    pass
            where = ", ".join(f"{k}×{v}" for k, v in landed.items() if k != "None")
            cells.append(c + (f" <br><sub>{where}</sub>" if where else ""))
        L.append(f"| {tag} | {PLACE_TARGET[tag]} | " + " | ".join(cells) + " |")

    L.append("\n## Totals\n")
    L.append("| arm | success | trials recorded | of 70 planned |")
    L.append("|---|---|---|---|")
    for arm, _, title in ARMS:
        s, n = totals[arm]
        rate = f"{100*s/n:.0f}%" if n else "—"
        L.append(f"| {title} | {s}/{n} ({rate}) | {n} | {100*n/70:.0f}% |")

    L.append("\n## Offline selection provenance (GWM arms)\n")
    L.append("| task | target | " + " | ".join(t for a, w, t in ARMS if w is not None) + " |")
    L.append("|---|---|" + "---|" * (len(ARMS) - 1))
    for tag in PICK_TAGS:
        cells = []
        for arm, wsuf, _ in ARMS:
            if wsuf is None:
                continue
            obj, plan = served(PROP_PICK, "refer6", tag, wsuf, PICK_NAME)
            ok = "" if obj != PICK_TARGET[tag] else "✓ "
            cells.append(f"{ok}{obj}<br><sub>{plan}</sub>")
        L.append(f"| {tag} | {PICK_TARGET[tag]} | " + " | ".join(cells) + " |")
    for tag in PLACE_TAGS:
        cells = []
        for arm, wsuf, _ in ARMS:
            if wsuf is None:
                continue
            obj, plan = served(PROP_PLACE, "place", tag, wsuf, PLACE_NAME)
            ok = "" if obj != PLACE_TARGET[tag] else "✓ "
            cells.append(f"{ok}{obj}<br><sub>{plan}</sub>")
        L.append(f"| place_{tag} | {PLACE_TARGET[tag]} | " + " | ".join(cells) + " |")

    anomalies = []
    for fam, tags, pref in (("pick", PICK_TAGS, "refer6"), ("place", PLACE_TAGS, "place")):
        for tag in tags:
            for arm, _, _ in ARMS:
                for r in rows(args.runs, arm, tag, f"{pref}_{tag}"):
                    if r["plan_failed"] == "True" or "scoring_error" in r["detail"]:
                        anomalies.append(f"- {arm}/{pref}_{tag} trial {r['trial']}: {r['detail'][:200]}")
    if anomalies:
        L.append("\n## Plan failures / scoring errors\n")
        L.extend(anomalies)

    L.append(NOTES)
    text = "\n".join(L) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"written: {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
