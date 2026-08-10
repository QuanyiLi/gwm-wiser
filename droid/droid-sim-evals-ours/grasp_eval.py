"""grasp_eval: batch evaluation for grasp-and-hold tasks on the DROID sim scenes.

Reuses batch_eval_v2 from droid-sim-evals wholesale (IsaacLab boot, tiptop
websocket client, CSV schema + resume, --fast render gating); only the success
judge differs — GraspTracker replaces the placement SuccessTracker.

    ../droid-sim-evals/.venv/bin/python -u grasp_eval.py --task-id grasp_cube --scene 1 \
        --instruction "pick up the cube" --success-rule '{"objects":["cube"],"lift":0.15}' \
        --trials 5 --results-csv runs/grasp_v1/results_grasp_cube.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "droid-sim-evals"))

import batch_eval_v2


class GraspTracker(batch_eval_v2.SuccessTracker):
    """Judge grasp-and-hold from the last pre-reset snapshot: each target's mesh
    center must sit `lift` meters above its settled height. A dropped object falls
    back before the post-plan hold ends, so sustained elevation implies a stable grasp.
    """

    def __init__(self, scene, rule: dict):
        # parent __init__ requires a container; alias the first target
        super().__init__(scene, {**rule, "container": rule["objects"][0]})
        self.z0 = {body: self.center(body)[2] for body in self.bodies.values()}

    def judge(self) -> tuple[bool, dict]:
        lifted, detail = 0, {}
        for pat, body in self.bodies.items():
            pos = self.center(body)
            z_rel = float(pos[2] - self.z0[body])
            ok = z_rel >= self.rule["lift"]
            lifted += ok
            detail[pat] = {"z_rel": round(z_rel, 3), "ok": ok, "pos": [round(float(v), 3) for v in pos]}
        return lifted >= self.rule.get("min_count", len(self.bodies)), detail


batch_eval_v2.SuccessTracker = GraspTracker

if __name__ == "__main__":
    batch_eval_v2.main()
