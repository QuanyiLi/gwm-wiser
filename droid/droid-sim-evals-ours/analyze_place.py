"""analyze_place: aggregate runs/place_v1/results_gwm_*.csv into a markdown table.

Reads the PlaceTracker detail JSON (place_eval.py) per row, so besides the
success rate it reports which bin the block actually ended up in (_landed_in),
separating wrong-bin groundings from drops and plan failures.

    python3 analyze_place.py [--runs runs/place_v1]
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

TAGS = ("red", "green", "tomato", "grass")
TARGET = {"red": "red_bin", "green": "green_bin", "tomato": "red_bin", "grass": "green_bin"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path(__file__).resolve().parent / "runs" / "place_v1")
    ap.add_argument("--arm", default="gwm")
    args = ap.parse_args()

    print(f"# scene-6 place eval ({args.arm} arm)\n")
    print("| task | instruction target | trials | success | landed red | landed green | nowhere | plan_failed |")
    print("|---|---|---|---|---|---|---|---|")
    tot = Counter()
    for tag in TAGS:
        path = args.runs / f"results_{args.arm}_{tag}.csv"
        if not path.exists():
            print(f"| place_{tag} | {TARGET[tag]} | — | — | — | — | — | — |")
            continue
        rows = [r for r in csv.DictReader(open(path)) if r["task"] == f"place_{tag}"]
        n = len(rows)
        succ = sum(r["success"] == "True" for r in rows)
        pf = sum(r["plan_failed"] == "True" for r in rows)
        landed = Counter()
        for r in rows:
            try:
                d = json.loads(r["detail"])
            except json.JSONDecodeError:
                continue
            where = d.get("_landed_in")
            landed[where if isinstance(where, str) else "nowhere"] += 1
        print(f"| place_{tag} | {TARGET[tag]} | {n} | {succ}/{n} | {landed['red_bin']} "
              f"| {landed['green_bin']} | {landed['nowhere']} | {pf} |")
        tot.update({"n": n, "succ": succ, "pf": pf})
    if tot["n"]:
        print(f"| **total** | | **{tot['n']}** | **{tot['succ']}/{tot['n']}** | | | | {tot['pf']} |")

    # per-trial detail lines
    print()
    for tag in TAGS:
        path = args.runs / f"results_{args.arm}_{tag}.csv"
        if not path.exists():
            continue
        for r in csv.DictReader(open(path)):
            if r["task"] != f"place_{tag}":
                continue
            try:
                d = json.loads(r["detail"])
            except json.JSONDecodeError:
                d = {"raw": r["detail"][:80]}
            blk = d.get("held_block", {})
            print(f"place_{tag} t{r['trial']}: success={r['success']} landed={d.get('_landed_in')} "
                  f"target={d.get('_target')} xy={blk.get('xy')} z_rel={blk.get('z_rel')} "
                  f"cand={json.dumps(d.get('_candidates', {}))}")


if __name__ == "__main__":
    main()
