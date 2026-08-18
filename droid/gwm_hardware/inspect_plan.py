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

Why the pads and not the TCP: on the first tomato/bowl runs the TCP looked
plausible (49 mm from the centre of a 100 mm bowl, i.e. on the rim) while the
pad over the bowl's opening sat 9 mm ABOVE the rim and could never enter it.
The plan would have pushed the bowl across the table.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.inspect_plan \
        droid/tiptop/tiptop_outputs/eval/<timestamp> --object tomato
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

ASSETS = Path(__file__).resolve().parent / "assets"
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


def pad_bands(q_grasp):
    """World z-band and xy centre of each finger pad at this configuration."""
    from curobo.types.base import TensorDeviceType
    from cutamp.robots import load_panda_robotiq_container

    kin = yaml.safe_load(CFG.read_text())["robot_cfg"]["kinematics"]
    names, spheres = kin["collision_link_names"], kin["collision_spheres"]
    ta = TensorDeviceType()
    S = (load_panda_robotiq_container(ta)
         .kin_model.get_state(ta.to_device(np.asarray(q_grasp)[None]))
         .link_spheres_tensor[0].cpu().numpy())
    out, i = {}, 0
    for n in names:
        k = len(spheres.get(n, []))
        if k and "pad" in n:
            s = S[i:i + k]
            out[n] = dict(z_lo=float((s[:, 2] - s[:, 3]).min()),
                          z_hi=float((s[:, 2] + s[:, 3]).max()),
                          x=float(s[:, 0].mean()), y=float(s[:, 1].mean()))
        i += k
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

    bands = pad_bands(q_grasp)
    if len(bands) == 2:
        (l, r) = bands.values()
        dz = abs((l["z_hi"] + l["z_lo"]) / 2 - (r["z_hi"] + r["z_lo"]) / 2)
        dxy = np.hypot(l["x"] - r["x"], l["y"] - r["y"])
        close_tilt = np.degrees(np.arctan2(dz, dxy))
        print(f"  closing axis is {close_tilt:.1f} deg off horizontal  ->  {dz*1000:.0f} mm between the pads")
        print(f"    this, not the TCP tilt, is what decides whether both pads reach: a tilt")
        print(f"    lying in the closing plane turns straight into pad height difference,")
        print(f"    scaled by the 136 mm open span (a 2F-85 would scale it by 85 mm).")

    print(f"\nfinger pads at this configuration (gripper OPEN)")
    ok = True
    for n, p in bands.items():
        r = np.linalg.norm([p["x"] - ctr[0], p["y"] - ctr[1]]) * 1000
        overlap = min(p["z_hi"], hi[2]) - max(p["z_lo"], lo[2])
        flag = "" if overlap > 0.005 else "   <-- NO vertical overlap with the object"
        if overlap <= 0.005:
            ok = False
        print(f"  {n:24s} z[{p['z_lo']:+.4f},{p['z_hi']:+.4f}]  xy({p['x']:+.3f},{p['y']:+.3f})  "
              f"{r:5.0f} mm from centre  overlap {overlap*1000:+.0f} mm{flag}")

    print(f"\n{'looks graspable' if ok else 'SUSPECT -- a pad cannot reach the object'}")


if __name__ == "__main__":
    main()
