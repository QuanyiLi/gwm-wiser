"""Check where a saved TiPToP plan actually puts the fingers, before executing it.

`tiptop-run` executes the moment cuTAMP returns a plan -- there is no
confirmation prompt between planning and motion (`tiptop_run.py:680`). Running
with `--no-execute-plan` first and then this script is the way to see a grasp
before the gripper commits to it.

What it reports, all in world coordinates:

  * the target object's extent, and the table plane fitted from the run's own
    point cloud (an independent check on tiptop's RANSAC table)
  * the grasp TCP, and how far it sits from the object centre
  * where the two finger PADS actually land -- the number that decides whether
    the gripper closes on the object or above it. The TCP alone does not tell
    you this: `grasp_frame` is a convention, the pads are geometry.

Why the pads and not the TCP: `grasp_frame` is a convention, the pads are
geometry, and a tilt lying in the closing plane turns straight into height
difference between them.

The pads must be swept through the CLOSE, not read at the open width. The
2F-140 is a four-bar linkage: closing swings each finger inward along an arc
that also drops it, by up to **44 mm** from open to closed. Judging the open
state alone -- the one configuration where the pads sit highest -- can call a
grasp ungraspable when a pad is a little above a rim, even though by the time
the fingers meet the pad is well BELOW it. A check that blocks good grasps is
worse than no check.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.tiptop_arm.inspect_plan \
        droid/tiptop/tiptop_outputs/eval/<timestamp> --object tomato
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from gwm_hardware.common.paths import ASSETS
CFG = ASSETS / "panda_robotiq_2f_140.yml"


def table_plane(pcd_path: Path):
    """Fit the dominant horizontal plane in the run's own cloud."""
    import open3d as o3d

    P = np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points)
    Q = P[(P[:, 2] > -0.15) & (P[:, 2] < 0.25)]
    hist, edges = np.histogram(Q[:, 2], bins=200)
    zpk = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    band = Q[np.abs(Q[:, 2] - zpk) < 0.01]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(band))
    (a, b, c, d), _ = pcd.segment_plane(0.004, 3, 1000)
    n = np.array([a, b, c])
    if c < 0:
        n, d = -n, -d
    tilt = np.degrees(np.arccos(abs(n[2])))
    return (lambda x, y: (-d - n[0] * x - n[1] * y) / n[2]), tilt


DRIVER_CLOSED = 0.70


def pad_sweep(q_grasp, steps=15):
    """Each pad's world z-EXTENT and centre across the whole close.

    The pad's own geometry, not its link origin: the pad is a 30 x 70 mm plate
    and the grasp is usually tilted, so its lower edge sits tens of millimetres
    below the origin. Using the origin would put the pad higher than it is.

    cuRobo's kinematics model locks the gripper, so the sweep goes through the
    URDF, whose mimic chain drives the four-bar properly.
    """
    from yourdfpy import URDF

    robot = URDF.load(str(ASSETS / "panda_robotiq_2f_140.urdf"),
                      load_meshes=False, build_scene_graph=True)
    kin = yaml.safe_load(CFG.read_text())["robot_cfg"]["kinematics"]
    local = {n: np.array([[*sp["center"], sp["radius"]] for sp in kin["collision_spheres"][n]])
             for n in ("left_inner_finger_pad", "right_inner_finger_pad")}
    q = np.asarray(q_grasp)
    out = {}
    for drv in np.linspace(0.0, DRIVER_CLOSED, steps):
        robot.update_cfg({**{f"panda_joint{i + 1}": q[i] for i in range(7)},
                          "finger_joint": float(drv)})
        for ln, sp in local.items():
            T = robot.get_transform(ln, "panda_link0")
            w = (T[:3, :3] @ sp[:, :3].T).T + T[:3, 3]
            out.setdefault(ln, []).append(dict(
                drv=float(drv), c=w.mean(0),
                z_lo=float((w[:, 2] - sp[:, 3]).min()),
                z_hi=float((w[:, 2] + sp[:, 3]).max())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--object", help="target label; default = the one the plan names")
    a = ap.parse_args()

    plan = json.loads((a.run_dir / "tiptop_plan.json").read_text())
    env = pickle.loads((a.run_dir / "perception/cutamp_env.pkl").read_bytes())

    # The grasp configuration is the end of the last trajectory before the close.
    close_at = next(i for i, s in enumerate(plan["steps"]) if s.get("type") == "gripper")
    q_grasp = np.asarray(plan["steps"][close_at - 1]["positions"][-1])
    label = a.object or plan["steps"][close_at]["label"].split("(")[1].split(",")[0]

    target = next(m for m in env.movables if m.name == label)
    v = np.asarray(target.get_mesh().vertices) + np.asarray(target.pose)[:3]
    lo, hi = v.min(0), v.max(0)
    ctr = 0.5 * (lo + hi)

    z_at, tilt = table_plane(a.run_dir / "perception/pointcloud.ply")
    z_tab = z_at(ctr[0], ctr[1])

    from curobo.types.base import TensorDeviceType
    from cutamp.robots import load_panda_robotiq_container
    ta = TensorDeviceType()
    T = (load_panda_robotiq_container(ta)
         .kin_model.get_state(ta.to_device(q_grasp[None])).ee_pose.get_numpy_matrix()[0])
    tcp = T[:3, 3]
    tilt_grasp = np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))

    print(f"\ntarget: {label}")
    print(f"  extent   x[{lo[0]:+.3f},{hi[0]:+.3f}]  y[{lo[1]:+.3f},{hi[1]:+.3f}]  z[{lo[2]:+.4f},{hi[2]:+.4f}]")
    print(f"  centre   ({ctr[0]:+.3f}, {ctr[1]:+.3f})   size {(hi-lo)[0]*1000:.0f} x {(hi-lo)[1]*1000:.0f} x {(hi-lo)[2]*1000:.0f} mm")
    print(f"  table plane under it  z = {z_tab:+.4f}   (fit tilt {tilt:.2f} deg over the whole surface)")
    print(f"  sits {(lo[2]-z_tab)*1000:+.1f} mm relative to that plane")

    print(f"\ngrasp")
    print(f"  TCP      ({tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.4f})   {tilt_grasp:.1f} deg from vertical")
    print(f"  TCP is {np.linalg.norm(tcp[:2]-ctr[:2])*1000:.0f} mm from the object centre in xy")

    sweep = pad_sweep(q_grasp)
    (ln_l, tl), (ln_r, tr) = sweep.items()

    # Contact is where the pads have converged to the object's width along the
    # closing axis; that, not the open width, is where the grasp is decided.
    width = float(np.linalg.norm((hi - lo)[:2]) / np.sqrt(2))
    seps = np.array([np.linalg.norm(a["c"] - b["c"]) for a, b in zip(tl, tr)])
    kc = int(np.argmin(np.abs(seps - width)))

    print(f"\nfinger pads through the close (the 2F-140 swings them inward AND down)")
    print(f"  {'driver':>7s}  {'separation':>10s}   {ln_l:>26s}   {ln_r:>26s}")
    for k in (0, len(tl) // 2, kc, len(tl) - 1):
        tag = {0: "open", len(tl) - 1: "closed", kc: "<- contact width"}.get(k, "")
        print(f"  {tl[k]['drv']:7.2f}  {seps[k] * 1000:8.0f} mm   "
              f"z[{tl[k]['z_lo']:+.4f},{tl[k]['z_hi']:+.4f}]           "
              f"z[{tr[k]['z_lo']:+.4f},{tr[k]['z_hi']:+.4f}]        {tag}")

    print(f"\nvertical overlap with the object from contact width to closed "
          f"(contact at {seps[kc] * 1000:.0f} mm, object {width * 1000:.0f} mm)")
    ok = True
    for name, tr_ in ((ln_l, tl), (ln_r, tr)):
        # best overlap anywhere from the contact width to fully closed -- the
        # fingers keep descending after first contact
        best = max(min(st["z_hi"], hi[2]) - max(st["z_lo"], lo[2]) for st in tr_[kc:])
        drop = (tr_[0]["z_lo"] - tr_[-1]["z_lo"]) * 1000
        flag = "" if best > 0.002 else "   <-- misses the object through the whole close"
        if best <= 0.002:
            ok = False
        print(f"  {name:24s} best overlap {best * 1000:+.0f} mm   "
              f"(lower edge descends {drop:+.0f} mm from open to closed){flag}")

    print(f"\n{'looks graspable' if ok else 'SUSPECT -- a pad misses the object through the whole close'}")


if __name__ == "__main__":
    main()
