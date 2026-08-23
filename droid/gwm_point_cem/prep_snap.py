"""prep_snap: CEM iteration means -> joint targets for snap_cem_poses.py.

Reads results/cem.json, IKs each per-iteration mean (and the final mean) for
the four tasks, and writes results/snap_poses.json:
    {task: [{"label", "xy", "q"}...]}
Run in the repo venv (SAPIEN):  .venv/bin/python prep_snap.py
"""

import json

from config import RESULTS
from pointing import PointerKinematics, load_views


def main() -> None:
    cem = json.loads((RESULTS / "cem.json").read_text())
    views, q_init = load_views()
    kin = PointerKinematics(q_init)
    out = {}
    for tag, run in cem.items():
        seq = [("home", None)]
        means = [h["mean"] for h in run["history"][1:]] + [run["final_mean"]]
        for k, m in enumerate(means, start=1):
            seq.append((f"iter{k}", m))
        # landing point = the best-scoring sample of the whole run
        seq.append(("final", run["winner"]))
        rows = []
        for label, xy in seq:
            if xy is None:
                rows.append({"label": label, "xy": None,
                             "q": [float(v) for v in q_init]})
            else:
                q = kin.ik(float(xy[0]), float(xy[1]))
                rows.append({"label": label, "xy": [float(xy[0]), float(xy[1])],
                             "q": [float(v) for v in q]})
        out[tag] = rows
        print(tag, "->", [r["label"] for r in rows])
    (RESULTS / "snap_poses.json").write_text(json.dumps(out))
    print("wrote", RESULTS / "snap_poses.json")


if __name__ == "__main__":
    main()
