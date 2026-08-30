"""run_select: score the six candidates under the six drawer tasks.

Every (task, phrasing) instruction is scored against all six candidates
(three drawer pulls, three object grasps) on both external cameras through
gwm-server; per-candidate scores are fused by the camera mean and averaged
over each task's five phrasings (prompt ensemble) into a 6x6 matrix S
(rows = tasks, columns = candidates). The selection for a task is the
argmax of its row. Writes results/selection.json.

    /root/code/gwm/gwm-wiser/.venv/bin/python run_select.py [--dump]
"""

import argparse
import json
import sys

import numpy as np

from config import (CAMS, CAPTURE_DIR, DRAWERS, RAT_SCALE, REPO, RESULTS,
                    SERVER_URL, TASK_IMAGE, TASKS)

sys.path.insert(0, str(REPO / "droid"))

from gwm_tiptop.score_client import score_candidates  # noqa: E402


def load_views():
    import h5py
    from scipy.spatial.transform import Rotation

    views = {}
    with h5py.File(CAPTURE_DIR / "external_obs.h5") as f:
        for cam in CAMS:
            pos = np.asarray(f[f"{cam}/pos_w"])
            w, x, y, z = np.asarray(f[f"{cam}/quat_w_ros"])
            c2w = np.eye(4)
            c2w[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            c2w[:3, 3] = pos
            views[cam] = (np.asarray(f[f"{cam}/rgb"])[..., :3],
                          np.asarray(f[f"{cam}/intrinsic_matrix"]), c2w)
    return views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default=SERVER_URL)
    ap.add_argument("--dump", action="store_true",
                    help="save the RAT strips of the canonical phrasings")
    args = ap.parse_args()

    cands = json.loads((RESULTS / "candidates.json").read_text())
    names = list(cands)
    cand_list = [cands[n]["candidate"] for n in names]
    views = load_views()
    tasks = list(TASKS)

    detail = {}
    for task in tasks:
        detail[task] = []
        for pi, text in enumerate(TASKS[task]["phrases"]):
            per_cam = {}
            for cam in CAMS:
                rgb, K, c2w = views[cam]
                sampling = {"rat_scale": RAT_SCALE, "task_image": TASK_IMAGE}
                if args.dump and pi == 0:
                    sampling["dump_dir"] = str(RESULTS / "strips" / task / cam)
                r = score_candidates(args.server_url, rgb, K, c2w, text,
                                     cand_list, sampling)
                per_cam[cam] = {"scores": r["scores"]}
            fused = np.mean([per_cam[c]["scores"] for c in CAMS], axis=0)
            detail[task].append({"instruction": text,
                                 "fused": [float(v) for v in fused],
                                 "per_cam": per_cam})
            print(f"  scored [{task}] {text!r}")

    S = np.array([np.mean([p["fused"] for p in detail[t]], axis=0) for t in tasks])
    picks = {t: names[int(np.argmax(S[i]))] for i, t in enumerate(tasks)}
    margins = {t: float(np.sort(S[i])[-1] - np.sort(S[i])[-2])
               for i, t in enumerate(tasks)}
    n_correct = sum(picks[t] == TASKS[t]["target"] for t in tasks)
    n_drawer = sum(picks[t] in DRAWERS for t in tasks)

    out = {
        "candidates": names,
        "tasks": {t: TASKS[t]["target"] for t in tasks},
        "matrix": {t: {n: float(S[i, j]) for j, n in enumerate(names)}
                   for i, t in enumerate(tasks)},
        "argmax": picks,
        "margins": margins,
        "n_correct": int(n_correct),
        "n_picked_a_drawer": int(n_drawer),
        "detail": detail,
    }
    (RESULTS / "selection.json").write_text(json.dumps(out, indent=2))

    print("\nensembled fused matrix (rows = task, cols = " + " ".join(names) + "):")
    for i, t in enumerate(tasks):
        print(f"  {t:16s} " + " ".join(f"{v:+.4f}" for v in S[i]))
    print(f"argmax {n_correct}/{len(tasks)} correct "
          f"({n_drawer}/{len(tasks)} picked a drawer trajectory)")
    for t in tasks:
        flag = "OK  " if picks[t] == TASKS[t]["target"] else "MISS"
        print(f"  {flag} {t:16s} -> {picks[t]:7s} margin {margins[t]:+.4f}")
    print(f"wrote {RESULTS / 'selection.json'}")


if __name__ == "__main__":
    main()
