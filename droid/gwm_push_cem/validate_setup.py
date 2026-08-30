"""validate_setup: solve the home configuration and check the search region.

Writes results/home_q.json (the handover to the Isaac-side scripts) and reports
how much of the endpoint lattice is reachable as a full straight-line slide.

    /root/code/gwm/gwm-wiser/.venv/bin/python validate_setup.py
"""

import json

import numpy as np

from config import (CUBES, GRID_STEP, HOME_XY, PAD_DROP, REGION, RESULTS,
                    TABLE_TOP_Z, TIP_CLEAR, TRAJ_STEPS, Z_PUSH)
from pushing import PushKinematics, snap


def lattice():
    xs = np.round(np.arange(REGION[0], REGION[1] + 1e-9, GRID_STEP), 4)
    ys = np.round(np.arange(REGION[2], REGION[3] + 1e-9, GRID_STEP), 4)
    return [snap(float(x), float(y)) for x in xs for y in ys], list(map(float, xs)), \
        list(map(float, ys))


def main() -> None:
    kin = PushKinematics()
    kin.pm.compute_forward_kinematics(kin.full_q(kin.q_home))
    pads = np.mean([kin.pm.get_link_pose(i).p for i in kin.i_pads], axis=0)
    print(f"home fingertip target ({HOME_XY[0]}, {HOME_XY[1]}, {Z_PUSH:.4f}) "
          f"-> FK {np.round(pads, 4).tolist()}")
    print(f"q_home = {np.round(kin.q_home, 5).tolist()}")
    print(f"table clearance of the lowest closed-gripper point: "
          f"{Z_PUSH - PAD_DROP - TABLE_TOP_Z:.4f} m")

    pts, xs, ys = lattice()
    feasible = [p for p in pts if kin.candidate(*p) is not None]
    print(f"endpoint lattice {len(xs)}x{len(ys)} = {len(pts)}; "
          f"{len(feasible)} have a fully feasible {TRAJ_STEPS}-waypoint slide")
    infeasible_x = sorted({p[0] for p in pts if p not in set(feasible)})
    if infeasible_x:
        print(f"  infeasible endpoints only at x in {infeasible_x} (behind the gripper)")

    for tag, (cx, cy) in CUBES.items():
        c = kin.candidate(cx, cy)
        print(f"  slide to the {tag} cube ({cx:.2f}, {cy:.2f}): "
              f"{'feasible' if c else 'INFEASIBLE'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "home_q.json").write_text(json.dumps({
        "q_home": [float(v) for v in kin.q_home],
        "home_xy": list(HOME_XY),
        "z_pad_mid": Z_PUSH,
        "tip_clearance": TIP_CLEAR,
        "d_tip": kin.d_tip,
        "lattice": {"xs": xs, "ys": ys, "feasible": len(feasible), "total": len(pts)},
    }, indent=2))
    print(f"wrote {RESULTS / 'home_q.json'}")


if __name__ == "__main__":
    main()
