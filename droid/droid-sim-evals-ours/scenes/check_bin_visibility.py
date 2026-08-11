"""check_bin_visibility: is every bin corner inside the home wrist frame?

The home wrist RGB-D is the ONLY view M2T2, Gemini and perception_geometric
get (plan.md G-23), so a bin clipped by that frustum yields a truncated point
cluster and therefore a wrong fitted placement surface. This projects the 8
world-space bbox corners of each bin into the captured wrist camera and reports
the pixel margin to each image edge, plus the clearance to every other object.

Camera convention matches tiptop_websocket._get_camera_params: world_from_cam
is built from wrist_cam_pos_w + wrist_cam_quat_w (wxyz, ROS optical frame:
x right, y down, z forward), so p_cam = R.T @ (p_world - t).

    /root/code/gwm/gwm-wiser/.venv/bin/python check_bin_visibility.py \
        [--bin-x 0.37 --red-y -0.06 --green-y -0.365 --bin-scale 0.6]
"""

import argparse
import itertools
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
CAP = HERE / "captures" / "scene6_0" / "wrist_obs.h5"

TABLE_TOP_Z = 0.045141201291582375
# small_KLT_visual_collision asset-local outer bbox, origin at the bbox centre
KLT_X, KLT_Y, KLT_Z = 0.19784, 0.29663, 0.14636
BIN_DROP = 0.005
# Settled AABBs of the stock scene-6 objects (captures/scene6_0/objects.json + USD bboxes)
STOCK = {
    "rubiks_cube": ((0.3328, 0.4049), (0.1542, 0.2263)),
    "_24_bowl": ((0.4217, 0.5828), (0.0337, 0.1952)),
    "_11_banana": ((0.5120, 0.5820), (-0.3400, -0.1460)),
}


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def bin_aabb(x, y, scale, yaw90=True):
    """World AABB of a bin. yaw90 = scene3's orient (asset Y -> world X, long axis X)."""
    ex, ey = (KLT_Y, KLT_X) if yaw90 else (KLT_X, KLT_Y)
    hx, hy, hz = ex * scale / 2, ey * scale / 2, KLT_Z * scale / 2
    z = TABLE_TOP_Z + KLT_Z * scale / 2 + BIN_DROP
    return (x - hx, x + hx), (y - hy, y + hy), (z - hz, z + hz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-x", type=float, default=0.37)
    ap.add_argument("--red-y", type=float, default=-0.06)
    ap.add_argument("--green-y", type=float, default=-0.365)
    ap.add_argument("--bin-scale", type=float, default=0.6)
    args = ap.parse_args()

    with h5py.File(CAP, "r") as f:
        K = f["intrinsic_matrix"][()]
        t = f["pos_w"][()].reshape(3)
        q = f["quat_w_ros"][()].reshape(4)
        H, W = f["rgb"].shape[:2]
    R = quat_to_mat(q)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def project(p):
        c = R.T @ (np.asarray(p) - t)
        return np.array([fx * c[0] / c[2] + cx, fy * c[1] / c[2] + cy]), c[2]

    print(f"wrist cam: pos={t.round(4).tolist()} K=[fx={fx:.1f} cx={cx:.0f} cy={cy:.0f}] image={W}x{H}\n")

    bins = {"red_bin": args.red_y, "green_bin": args.green_y}
    boxes = {}
    for name, y in bins.items():
        (x0, x1), (y0, y1), (z0, z1) = bin_aabb(args.bin_x, y, args.bin_scale)
        boxes[name] = ((x0, x1), (y0, y1))
        uvs = [project(p)[0] for p in itertools.product((x0, x1), (y0, y1), (z0, z1))]
        u = np.array([p[0] for p in uvs])
        v = np.array([p[1] for p in uvs])
        margins = {"left": u.min(), "right": W - u.max(), "top": v.min(), "bottom": H - v.max()}
        worst = min(margins, key=margins.get)
        ok = "OK " if margins[worst] > 0 else "CLIP"
        print(f"{ok} {name:10s} X=[{x0:.3f},{x1:.3f}] Y=[{y0:.3f},{y1:.3f}] top_z={z1:.3f}")
        print(f"     u=[{u.min():7.1f},{u.max():7.1f}] v=[{v.min():7.1f},{v.max():7.1f}]  "
              f"tightest edge: {worst} {margins[worst]:+.0f} px")

    print()
    all_boxes = {**boxes, **{k: v for k, v in STOCK.items()}}
    names = list(all_boxes)
    for a, b in itertools.combinations(names, 2):
        (ax0, ax1), (ay0, ay1) = all_boxes[a]
        (bx0, bx1), (by0, by1) = all_boxes[b]
        gx = max(bx0 - ax1, ax0 - bx1)
        gy = max(by0 - ay1, ay0 - by1)
        gap = max(gx, gy)
        flag = "OVERLAP" if gap < 0 else ("tight  " if gap < 0.02 else "       ")
        print(f"{flag} {a:12s} <-> {b:12s} AABB gap = {gap * 1000:6.1f} mm")

    sep = abs(args.red_y - args.green_y)
    print(f"\nbin centre separation: {sep:.3f} m")


if __name__ == "__main__":
    main()
