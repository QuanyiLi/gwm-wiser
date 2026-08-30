"""validate_fk: compare the numpy Panda FK (vjepa_sel/fk.py) against the Isaac
articulation's recorded `panda_link8` world pose from a replay, and the
offline plan unroll against the executed timeline.

    .venv/bin/python sim/validate_fk.py runs/replay_pick/plan_13_object_4
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vjepa_sel.fk import fk_link8_batch, quat_wxyz_to_rot  # noqa: E402
from vjepa_sel.plan_stepper import load_plan, unroll_plan  # noqa: E402


def main(run_dir, plan_path=None):
    run_dir = Path(run_dir)
    tr = np.load(run_dir / "traj.npz")
    q_meas, q_cmd = tr["q_meas"], tr["q_cmd"]
    p_sim, q_sim = tr["link8_pos"], tr["link8_quat"]
    # FK of the MEASURED joints must match the sim's link8 pose (same instant)
    p_fk, R_fk = fk_link8_batch(q_meas)
    dp = np.linalg.norm(p_fk - p_sim, axis=1)
    dang = []
    for R, q in zip(R_fk, q_sim):
        Rs = quat_wxyz_to_rot(q)
        dang.append(np.degrees(np.linalg.norm(Rotation.from_matrix(R.T @ Rs).as_rotvec())))
    dang = np.asarray(dang)
    print(f"FK(q_meas) vs sim link8: |dpos| mean {dp.mean()*1000:.2f} mm max {dp.max()*1000:.2f} mm; "
          f"angle mean {dang.mean():.3f} deg max {dang.max():.3f} deg  (N={len(dp)})")
    # tracking error: commanded vs measured joints / poses
    p_cmd, _ = fk_link8_batch(q_cmd)
    lag = np.linalg.norm(p_cmd - p_sim, axis=1)
    print(f"command-vs-measured link8 position (controller lag): mean {lag.mean()*1000:.1f} mm, max {lag.max()*1000:.1f} mm")
    print(f"|q_cmd - q_meas| max {np.abs(q_cmd - q_meas).max():.4f} rad")
    if plan_path is not None:
        u = unroll_plan(load_plan(plan_path))
        n = min(len(u["actions"]), len(q_cmd))
        d = np.abs(u["actions"][:n, :7] - q_cmd[:n]).max()
        print(f"offline unroll vs executed commands: {len(u['actions'])} vs {len(q_cmd)} steps, max|dq| {d:.2e} rad")
        g = np.abs(u["actions"][:n, 7] - tr["grip_cmd"][:n]).max()
        print(f"gripper command match: max diff {g:.1e}")


if __name__ == "__main__":
    main(*sys.argv[1:])
