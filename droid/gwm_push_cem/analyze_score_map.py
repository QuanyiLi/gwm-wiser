"""analyze_score_map: what the score map says, before any search is run.

Reads the cached score map and answers three questions per prompt:

  * where the objective's argmax over the whole search region is, and how far
    that is from the cube the prompt names;
  * whether the endpoint that serves this prompt outranks the endpoints that
    serve the other two, i.e. whether the map alone knows which cube was asked
    for;
  * whether that holds per camera as well as after fusing the two, which
    separates "no signal" from "two cameras with opposite signals".

    /root/code/gwm/gwm-wiser/.venv/bin/python analyze_score_map.py
"""

import argparse
import json

import numpy as np

from config import (CAMS, CUBES, DIRECTIONS, HOME_XY, LATTICE_STEP, PROMPTS,
                    REGION, RESULTS)

TAGS = ("front", "left", "right")
OVERSHOOT = 0.04      # how far past the cube the probe endpoint reaches


def probe_point(tag):
    """The lattice endpoint that serves `tag`: its cube plus an overshoot.

    This is the endpoint a search would have to find to push that cube, so
    comparing the three probes under one instruction asks whether the score map
    alone knows which cube was named. Clipped to the region and snapped to the
    lattice so the probe is an endpoint the search could actually have chosen.
    """
    (cx, cy), (dx, dy) = CUBES[tag], DIRECTIONS[tag]
    x = min(max(cx + dx * OVERSHOOT, REGION[0]), REGION[1])
    y = min(max(cy + dy * OVERSHOOT, REGION[2]), REGION[3])
    return (round(round(x / LATTICE_STEP) * LATTICE_STEP, 4),
            round(round(y / LATTICE_STEP) * LATTICE_STEP, 4))


def value(entry, objective, cam):
    if entry is None:
        return np.nan
    if cam is None:
        s, p = entry["score"], entry["prior"]
    else:
        s, p = entry["per_cam"][cam]["score"], entry["per_cam"][cam]["prior"]
    return s if objective == "raw" else s - p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="grid.json")
    ap.add_argument("--out", default="score_map_report.json")
    ap.add_argument("--objective", default=None, choices=["lang", "raw"],
                    help="report only this one; both are always written out")
    args = ap.parse_args()
    objectives = ("lang", "raw") if args.objective is None else (args.objective,)

    g = json.loads((RESULTS / args.grid).read_text())
    probes = {t: probe_point(t) for t in TAGS}
    report = {"overshoot": OVERSHOOT, "probes": {t: list(p) for t, p in probes.items()},
              "objectives": {}}

    for objective in objectives:
        obj = {"argmax": {}, "which_cube": {}}
        for view in (None,) + CAMS:
            hits = 0
            per_prompt = {}
            for tag in TAGS:
                pts = g["prompts"][tag]["points"]
                sub = np.array([value(pts.get(f"{probes[t][0]},{probes[t][1]}"),
                                      objective, view) for t in TAGS])
                if np.all(np.isnan(sub)):
                    per_prompt[tag] = {"picked": None, "margin": None,
                                       "scores": {t: None for t in TAGS}}
                    continue
                pick = TAGS[int(np.nanargmax(sub))]
                margin = float(sub[TAGS.index(tag)] - np.nanmax(np.delete(sub, TAGS.index(tag))))
                per_prompt[tag] = {"picked": pick, "margin": round(margin, 5),
                                   "scores": {t: round(float(v), 5) for t, v in zip(TAGS, sub)}}
                hits += pick == tag
            obj["which_cube"]["fused" if view is None else view] = {
                "hits_of_3": hits, "per_prompt": per_prompt}

        for tag in TAGS:
            pts = g["prompts"][tag]["points"]
            best, bestv = None, -np.inf
            for k, e in pts.items():
                v = value(e, objective, None)
                if np.isfinite(v) and v > bestv:
                    best, bestv = k, v
            bx, by = map(float, best.split(","))
            cx, cy = CUBES[tag]
            obj["argmax"][tag] = {
                "endpoint": [bx, by], "value": round(bestv, 5),
                "dist_to_named_cube_m": round(float(np.hypot(bx - cx, by - cy)), 4),
                "offset_from_home_m": [round(bx - HOME_XY[0], 4), round(by - HOME_XY[1], 4)],
            }
        report["objectives"][objective] = obj

        print(f"=== objective: {objective}")
        for view, d in obj["which_cube"].items():
            detail = ", ".join(f"{t}->{d['per_prompt'][t]['picked']}"
                               f"({d['per_prompt'][t]['margin']:+.4f})" for t in TAGS)
            print(f"  which cube, {view:<16} {d['hits_of_3']}/3   {detail}")
        for tag in TAGS:
            a = obj["argmax"][tag]
            print(f"  argmax [{tag:<5}] {a['endpoint']} "
                  f"= home + ({a['offset_from_home_m'][0]:+.2f},{a['offset_from_home_m'][1]:+.2f}), "
                  f"{a['dist_to_named_cube_m'] * 100:.0f} cm from the {tag} cube")

    (RESULTS / args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()
