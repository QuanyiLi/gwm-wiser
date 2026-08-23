"""diag_onestep: one-step energy-landscape sanity checks, in-domain vs in the sim.

For a (frame_0, frame_k) pair with known EEF delta, predict frame k from
frame_0 under a 5x5x5 grid of translation actions (the goal-reaching test in
the repo's energy_landscape_example notebook) and ask whether the energy to
the true frame k is lowest near the true action. Reported per pair:
  E_best / a_best     the grid minimum and where it is
  E_gt                energy at the grid action nearest the true delta
  E_zero              energy of the zero action
  E_nochange          |z_0 - z_k| (predicting 'nothing moves')
  spearman            rank correlation of energy with |a - a_true| (should be > 0)
Pairs: the repo's franka_example_traj.npz (real DROID, 1 step) and sim
candidates at 2-4 s (several steps of motion, so the frames differ).

    .venv/bin/python diag_onestep.py --replay-dir runs/replay_pick --out runs/diag_onestep.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from score_vjepa import observed_frames
from vjepa_sel.model import VJEPA2AC, l1_energy
from vjepa_sel.plan_stepper import load_plan
from vjepa_sel.traj import plan_to_sequence, poses_to_diffs

SIM_PAIRS = [("plan_13_object_4", 8), ("plan_13_object_4", 16), ("plan_10_object_3", 12),
             ("plan_04_object_1", 12), ("plan_07_object_2", 12), ("plan_00_object_0", 12)]


def landscape(model, z0, zk, s0, gt, include_rotation):
    r = max(0.075, float(np.abs(gt[:3]).max()))
    grid = np.linspace(-r, r, 5)
    rot = gt[3:6] if include_rotation else np.zeros(3)
    acts = np.array([[dx, dy, dz, *rot, gt[6] if include_rotation else 0.0] for dx in grid for dy in grid for dz in grid], np.float32)
    B = len(acts)
    with torch.no_grad():
        zp = model.predict_next(z0[None].expand(B, -1, -1)[:, None], torch.from_numpy(acts)[:, None],
                                torch.from_numpy(np.asarray(s0)[None]).float().expand(B, -1)[:, None])
    E = l1_energy(zp, zk[None].expand(B, -1, -1)).cpu().numpy()
    d = np.linalg.norm(acts[:, :3] - gt[:3], axis=1)
    i = int(np.argmin(E))
    zero = int(np.argmin(np.abs(acts[:, :3]).sum(1)))
    return {
        "grid_half_width": r, "E_best": float(E[i]), "a_best_xyz": acts[i, :3].round(3).tolist(),
        "E_gt": float(E[int(np.argmin(d))]), "E_zero": float(E[zero]), "E_max": float(E.max()),
        "E_nochange": float(l1_energy(z0, zk).item()), "spearman_E_vs_dist_to_gt": float(spearmanr(E, d).correlation),
        "gt_xyz": gt[:3].round(3).tolist(), "gt_rot_abs_max": float(np.abs(gt[3:6]).max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", type=Path, default=Path("runs/replay_pick"))
    ap.add_argument("--plans-dir", type=Path, default=Path("../gwm_integrate_doc/proposals/scene6_rev2"))
    ap.add_argument("--out", type=Path, default=Path("runs/diag_onestep.json"))
    ap.add_argument("--crop-mode", default="full_aa")
    args = ap.parse_args()
    model = VJEPA2AC()
    out = {}
    tr = np.load("vjepa2/notebooks/franka_example_traj.npz")
    obs, st = tr["observations"][0], tr["states"][0]
    z = model.encode(obs, crop_mode="full")
    gt = poses_to_diffs(st)[0]
    out["droid_example"] = {"with_rotation": landscape(model, z[0], z[1], st[0], gt, True),
                            "translation_only": landscape(model, z[0], z[1], st[0], gt, False)}
    for stem, k in SIM_PAIRS:
        frames, steps, judge = observed_frames(args.replay_dir, stem, "external_cam_2", 4)
        seq = plan_to_sequence(load_plan(args.plans_dir / f"{stem}.json"), stride=4)
        z = model.encode(frames[[0, k]], crop_mode=args.crop_mode)
        gt = poses_to_diffs(np.stack([seq["states"][0], seq["states"][k]]))[0]
        out[f"sim:{stem}:0->{k}"] = {"t_s": float(seq["t"][k]),
                                     "with_rotation": landscape(model, z[0], z[1], seq["states"][0], gt, True),
                                     "translation_only": landscape(model, z[0], z[1], seq["states"][0], gt, False)}
    args.out.write_text(json.dumps(out, indent=1))
    print("| pair | rot in action | E_best (at) | E_gt | E_zero | E_nochange | spearman(E, dist to gt) |")
    print("|---|---|---|---|---|---|---|")
    for k, v in out.items():
        for mode in ("with_rotation", "translation_only"):
            r = v[mode]
            print(f"| {k} | {mode} | {r['E_best']:.3f} ({r['a_best_xyz']}) | {r['E_gt']:.3f} | {r['E_zero']:.3f} | {r['E_nochange']:.3f} | {r['spearman_E_vs_dist_to_gt']:+.2f} |")


if __name__ == "__main__":
    main()
