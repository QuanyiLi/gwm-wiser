"""place_eval: batch evaluation for place-into-bin tasks on scene 6 variant 1.

Reuses batch_eval_v2 wholesale, like grasp_eval.py does, but keeps the STOCK
placement judge -- `batch_eval_v2.SuccessTracker` already scores
`{"objects", "container", "xy_tol", "z_rel"}` with the mesh-centre and
last-pre-reset-snapshot corrections. The only addition is bookkeeping:
PlaceTracker also measures every *candidate* container named in the rule, so a
confidently-wrong placement (block delivered to the green bin when the
instruction said red) is distinguishable in the CSV from a plan failure or a
drop. Without that, both are just `success=False` and the referring-expression
confusion matrix is unrecoverable.

The block is never released -- the episode ends once it is inside a bin -- so
"placed" here means the block's mesh centre sits within `xy_tol` of the bin
centre and inside the `z_rel` band, still gripped.

    ../droid-sim-evals/.venv/bin/python -u place_eval.py \
        --task-id place_red --scene 6 --variant 1 \
        --instruction "put the block into the red box" \
        --success-rule '{"objects":["held_block"],"container":"red_bin",
                         "candidates":["red_bin","green_bin"],
                         "xy_tol":0.05,"z_rel":[-0.03,0.03]}' \
        --trials 10 --results-csv runs/place_v1/results_place_red.csv
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "droid-sim-evals"))

import weld_held_block  # noqa: F401  -- wraps settle_sim; welds held_block at first settle

import batch_eval_v2


class PlaceTracker(batch_eval_v2.SuccessTracker):
    """Stock placement judge plus a per-candidate-container record.

    The parent computes a mesh-centre offset for everything it is given in
    `objects` + `container`, so the cheapest way to get offsets for the other
    candidate bins is to hand it an extended `objects` list and then split
    `self.bodies` back into targets vs candidates. `self.rule` is restored to
    the caller's rule afterwards because the parent's `judge` derives its
    default `min_count` from `len(self.rule["objects"])`.
    """

    def __init__(self, scene, rule: dict):
        cands = list(rule.get("candidates", []))
        extended = list(rule["objects"]) + [c for c in cands if c not in rule["objects"]]
        super().__init__(scene, {**rule, "objects": extended})
        self.candidates = {c: self.bodies[c] for c in cands}
        self.bodies = {p: self.bodies[p] for p in rule["objects"]}
        self.rule = rule

    def judge(self) -> tuple[bool, dict]:
        ok, detail = super().judge()
        if not self.candidates:
            return ok, detail
        obj = self.center(self.bodies[self.rule["objects"][0]])
        per, nearest, best_xy = {}, None, None
        for pat, body in self.candidates.items():
            c = self.center(body)
            xy = float(np.linalg.norm(obj[:2] - c[:2]))
            zr = float(obj[2] - c[2])
            per[pat] = {
                "xy": round(xy, 3),
                "z_rel": round(zr, 3),
                "inside": xy <= self.rule["xy_tol"]
                and self.rule["z_rel"][0] <= zr <= self.rule["z_rel"][1],
            }
            if best_xy is None or xy < best_xy:
                nearest, best_xy = pat, xy
        inside = [p for p, d in per.items() if d["inside"]]
        detail["_candidates"] = per
        detail["_nearest"] = nearest
        detail["_landed_in"] = inside[0] if len(inside) == 1 else (inside or None)
        detail["_target"] = self.rule["container"]
        return ok, detail


batch_eval_v2.SuccessTracker = PlaceTracker

if __name__ == "__main__":
    batch_eval_v2.main()
