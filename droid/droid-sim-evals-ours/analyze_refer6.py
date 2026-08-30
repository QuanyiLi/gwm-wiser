"""Aggregate the scene-6 referral eval: success table, wallclock stats, anomalies.

    python3 analyze_refer6.py   # reads runs/refer6/, prints markdown
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "runs/refer6"
TAGS = ["fruit", "yellow", "eat", "negation", "puzzle", "colorful", "nearbowl", "eatfrom", "between", "container"]
TARGET = {"fruit": "banana", "yellow": "banana", "eat": "banana", "negation": "banana",
          "puzzle": "cube", "colorful": "cube", "nearbowl": "cube",
          "eatfrom": "bowl", "between": "bowl", "container": "bowl"}


def load(arm: str, tag: str):
    f = OUT / f"results_{arm}_{tag}.csv"
    if not f.exists():
        return []
    with open(f) as fh:
        return list(csv.DictReader(fh))


def trial_times():
    """Per-trial wallclock from watcher deltas (epoch csv rowcount)."""
    log = OUT / "trial_timing.log"
    if not log.exists():
        return {}
    hist = defaultdict(list)  # csv -> [(epoch, rows)]
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        epoch, name, rows = int(parts[0]), parts[1], int(parts[2])
        hist[name].append((epoch, rows))
    deltas = defaultdict(list)  # csv -> [seconds per new row]
    for name, seq in hist.items():
        for (t0, r0), (t1, r1) in zip(seq, seq[1:]):
            if r1 > r0:
                per = (t1 - t0) / (r1 - r0)
                deltas[name].extend([per] * (r1 - r0))
    return deltas


def main():
    times = trial_times()
    print("| task | target | tiptop | gwm | tiptop plan_fail | gwm plan_fail |")
    print("|---|---|---|---|---|---|")
    tot = {"tiptop": [0, 0], "gwm": [0, 0]}
    anomalies = []
    for tag in TAGS:
        cells = {}
        for arm in ("tiptop", "gwm"):
            rows = load(arm, tag)
            n = len(rows)
            s = sum(r["success"] == "True" for r in rows)
            pf = sum(r["plan_failed"] == "True" for r in rows)
            tot[arm][0] += s
            tot[arm][1] += n
            cells[arm] = (s, n, pf)
            for r in rows:
                d = json.loads(r["detail"]) if r["detail"] else {}
                if "scoring_error" in d or (r["plan_failed"] == "True"):
                    anomalies.append(f"{arm}/{tag} trial {r['trial']}: {r['detail'][:160]}")
        print(f"| {tag} | {TARGET[tag]} | {cells['tiptop'][0]}/{cells['tiptop'][1]} | {cells['gwm'][0]}/{cells['gwm'][1]} | {cells['tiptop'][2]} | {cells['gwm'][2]} |")
    print(f"| **total** | | **{tot['tiptop'][0]}/{tot['tiptop'][1]}** | **{tot['gwm'][0]}/{tot['gwm'][1]}** | | |")

    print("\n## per-trial wallclock (watcher deltas, includes boot in first row)")
    over = []
    for name, ds in sorted(times.items()):
        body = ds[1:] if len(ds) > 1 else ds  # first delta includes Isaac boot
        if not body:
            continue
        mx = max(body)
        print(f"- {name}: n={len(ds)} median={sorted(body)[len(body)//2]:.0f}s max={mx:.0f}s (first-row incl. boot: {ds[0]:.0f}s)")
        if mx > 200:
            over.append((name, mx))
    if over:
        print("\nTRIALS OVER 200s BUDGET:", over)
    else:
        print("\nAll steady-state trials within the 200 s budget.")

    if anomalies:
        print("\n## plan failures / scoring errors")
        for a in anomalies:
            print("-", a)


if __name__ == "__main__":
    main()
