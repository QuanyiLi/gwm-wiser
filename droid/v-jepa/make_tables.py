"""make_tables: markdown summary tables from runs/vjepa_<family>/<tag>/selection.json.

    .venv/bin/python make_tables.py > runs/tables.md
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = ["w32_s4", "w8_s4", "w32_s8", "w32_s16", "w32_s4_cam1", "w32_s4_tcp", "w32_s4_crop135"]


def load(fam, tag):
    p = HERE / "runs" / f"vjepa_{fam}" / tag / "selection.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell(summary, key, pool, proto, metric):
    if key not in summary:
        return "–"
    v = summary[key][f"{pool}_{proto}_{metric}"]
    return f"{v[0]}/{v[1]}"


def decomp(fam, tag):
    p = HERE / "runs" / f"vjepa_{fam}" / tag / "summary.md"
    if not p.exists():
        return {}
    txt = p.read_text()
    out = {}
    for m in re.finditer(r"## Energy matrix — goal bank `(\w+(?:\.\d)?)`, energy `(\w+)`, arm `(\w+)`.*?candidate-row effect (\d+)%, goal-column effect (\d+)%, interaction \(the only part that can select\) (\d+)%; mean same-object energy ([\d.]+) vs different-object ([\d.]+)", txt, re.S):
        out[(m.group(1), m.group(3))] = (m.group(4), m.group(5), m.group(6), m.group(7), m.group(8))
    return out


def main():
    for fam in ("pick", "place"):
        print(f"\n### {fam}\n")
        sels = {tag: load(fam, tag) for tag in CONFIGS}
        sels = {k: v for k, v in sels.items() if v}
        # headline table for w32_s4
        s = sels["w32_s4"]["summary"]
        print("headline (w32_s4), goal excluded:\n")
        print("| arm | goal bank : energy | rule | LOO object | LOO success | single object | single success |")
        print("|---|---|---|---|---|---|---|")
        for key in ["pred/final:final/argmin", "pred/final:final/two_stage", "pred/final:min/argmin",
                    "pred/h1.5:at_h/argmin", "pred/h3:at_h/argmin", "pred/h3:min/argmin", "pred/h3:mean/argmin", "pred/h3:at_h/two_stage",
                    "pred/h6:at_h/argmin", "pred/h6:min/argmin", "pred/h6:mean/argmin", "pred/h6:at_h/two_stage",
                    "pred/lift:own_lift/two_stage",
                    "oracle/final:final/argmin", "oracle/h1.5:at_h/argmin", "oracle/h3:at_h/argmin", "oracle/h6:at_h/argmin", "oracle/lift:own_lift/argmin"]:
            if key not in s:
                continue
            arm, bankagg, rule = key.split("/")
            print(f"| {arm} | {bankagg} | {rule} | {cell(s, key, 'excl', 'loo', 'object_correct')} | {cell(s, key, 'excl', 'loo', 'success')} | "
                  f"{cell(s, key, 'excl', 'single', 'object_correct')} | {cell(s, key, 'excl', 'single', 'success')} |")
        # sweep: final bank
        print("\nsweep, final goal bank (LOO object correct, goal excluded):\n")
        print("| config | argmin, final | two-stage, final | argmin, min | argmin, at close | two-stage, at close | oracle argmin, final (single) | two-stage single-goal success |")
        print("|---|---|---|---|---|---|---|---|")
        for tag, sel in sels.items():
            s = sel["summary"]
            print(f"| `{tag}` | {cell(s,'pred/final:final/argmin','excl','loo','object_correct')} | {cell(s,'pred/final:final/two_stage','excl','loo','object_correct')} | "
                  f"{cell(s,'pred/final:min/argmin','excl','loo','object_correct')} | {cell(s,'pred/final:close/argmin','excl','loo','object_correct')} | "
                  f"{cell(s,'pred/final:close/two_stage','excl','loo','object_correct')} | {cell(s,'oracle/final:final/argmin','excl','loo','object_correct')} "
                  f"({cell(s,'oracle/final:final/argmin','excl','single','object_correct')}) | {cell(s,'pred/final:final/two_stage','excl','single','success')} |")
        # sweep: horizon banks
        print("\nsweep, short-horizon goal banks (LOO object correct, goal excluded; pred arm unless noted):\n")
        print("| config | h1.5 at_h argmin | h1.5 min | h1.5 two-stage | h3 at_h argmin | h3 min | h3 mean | h3 two-stage | h6 at_h argmin | h6 min | h6 mean | h6 two-stage | oracle h3 at_h | oracle h6 at_h |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for tag, sel in sels.items():
            s = sel["summary"]
            if "pred/h3:at_h/argmin" not in s:
                continue
            print(f"| `{tag}` | {cell(s,'pred/h1.5:at_h/argmin','excl','loo','object_correct')} | {cell(s,'pred/h1.5:min/argmin','excl','loo','object_correct')} | {cell(s,'pred/h1.5:at_h/two_stage','excl','loo','object_correct')} | "
                  f"{cell(s,'pred/h3:at_h/argmin','excl','loo','object_correct')} | {cell(s,'pred/h3:min/argmin','excl','loo','object_correct')} | {cell(s,'pred/h3:mean/argmin','excl','loo','object_correct')} | {cell(s,'pred/h3:at_h/two_stage','excl','loo','object_correct')} | "
                  f"{cell(s,'pred/h6:at_h/argmin','excl','loo','object_correct')} | {cell(s,'pred/h6:min/argmin','excl','loo','object_correct')} | {cell(s,'pred/h6:mean/argmin','excl','loo','object_correct')} | {cell(s,'pred/h6:at_h/two_stage','excl','loo','object_correct')} | "
                  f"{cell(s,'oracle/h3:at_h/argmin','excl','loo','object_correct')} | {cell(s,'oracle/h6:at_h/argmin','excl','loo','object_correct')} |")
        # decompositions
        print("\nvariance decomposition (row / column / interaction %, same- vs different-object energy):\n")
        print("| config | final pred | final oracle | h3 pred | h3 oracle | h6 pred | h6 oracle |")
        print("|---|---|---|---|---|---|---|")
        for tag in sels:
            d = decomp(fam, tag)
            def f(bank, arm):
                v = d.get((bank, arm))
                return "–" if v is None else f"{v[0]}/{v[1]}/{v[2]} %, {v[3]} vs {v[4]}"
            print(f"| `{tag}` | {f('final','pred')} | {f('final','oracle')} | {f('h3','pred')} | {f('h3','oracle')} | {f('h6','pred')} | {f('h6','oracle')} |")
        # per-task for headline and the h6 argmin
        for key in ("pred/final:final/two_stage", "pred/final:final/argmin", "pred/h6:at_h/argmin", "pred/h3:at_h/argmin", "oracle/final:final/argmin"):
            if key not in sels["w32_s4"]["results"]:
                continue
            r = sels["w32_s4"]["results"][key]["excl"]
            print(f"\nper task, w32_s4, {key}, goal excluded:\n")
            print("| task | target | single goal | chosen | object ok | success | LOO object | LOO success |")
            print("|---|---|---|---|---|---|---|---|")
            for tg, d in r.items():
                sgl = d["single"]
                lo = sum(x["object_correct"] for x in d["loo"]); ls = sum(x["success"] for x in d["loo"]); n = len(d["loo"])
                if sgl is None:
                    print(f"| {tg} | {d['target']} | – | – | – | – | {lo}/{n} | {ls}/{n} |")
                else:
                    print(f"| {tg} | {d['target']} | {sgl['goal'].replace('.json','')} | {sgl['chosen'].replace('.json','')} ({sgl['chosen_target']}) | "
                          f"{'✓' if sgl['object_correct'] else '✗'} | {'✓' if sgl['success'] else '✗'} | {lo}/{n} | {ls}/{n} |")


if __name__ == "__main__":
    main()
