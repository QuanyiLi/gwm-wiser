"""optimize_bin_layout: search bin placements for the scene-6 place tasks.

Feasible layouts are ranked by how far apart the two bins are IN EXTERNAL-
CAMERA PIXELS rather than in world space: two bins 0.3 m apart along the
external camera's viewing ray project nearly on top of each other, whereas the
same 0.3 m perpendicular to that ray keeps the two destinations visually
distinct from the static viewpoint.

Constraints, all evaluated against the scene captures:
  wrist    every bin bbox corner inside the home wrist frame with margin, AND
           the bin's opening not shadowed by the gripper -- M2T2/Gemini/
           perception_geometric see only that view, and a clipped or shadowed
           cluster fits a truncated (wrong) placement surface. The gripper
           mask is built by unprojecting the no-bins capture's wrist depth and
           keeping world z > GRIPPER_Z (every table object tops out at
           0.118 m, the fingers start around 0.32 m).
  external every bin fully in frame; bins must not occlude each other, and must
           not occlude the banana (four refer6 pick tasks target it).
  physical AABB clearance to the three stock objects, and a planar reach band
           matching what the stock scenes actually exercise (0.415 .. 0.660 m).

    /root/code/gwm/gwm-wiser/.venv/bin/python optimize_bin_layout.py [--top 10]
"""

import argparse
import itertools
from pathlib import Path

import h5py
import numpy as np

from bin_geom import BIN_DROP, DEFAULT_HEIGHT, DEFAULT_SIZE, STOCK, TABLE_TOP_Z, bin_report

HERE = Path(__file__).resolve().parent
CAPTURES = HERE / "captures" / "scene6_0"
# The gripper mask must come from a capture with no bins in it, else the bins
# occlude the very region we are testing.
CLEAN_WRIST = HERE / "captures" / "scene6_0_rev1_nobins" / "wrist_obs.h5"
BANANA = STOCK["_11_banana"]
GRIPPER_Z = 0.20  # world z above every table object, below the finger tips


def quat_to_mat(q):
    w, x, y, z = np.asarray(q).reshape(4)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class Cam:
    """ROS optical frame (x right, y down, z forward), matching tiptop_websocket."""

    def __init__(self, K, pos, quat, w, h):
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.t = np.asarray(pos).reshape(3)
        self.R = quat_to_mat(quat)
        self.W, self.H = w, h

    def boxes(self, corners):
        """corners (N,8,3) -> image AABB (N,4) as [u0,u1,v0,v1], depth range (N,2)."""
        c = (corners - self.t) @ self.R
        u = self.fx * c[..., 0] / c[..., 2] + self.cx
        v = self.fy * c[..., 1] / c[..., 2] + self.cy
        img = np.stack([u.min(1), u.max(1), v.min(1), v.max(1)], 1)
        return img, np.stack([c[..., 2].min(1), c[..., 2].max(1)], 1)

    def margins(self, img):
        return np.minimum.reduce([img[:, 0], self.W - img[:, 1], img[:, 2], self.H - img[:, 3]])

    def centres(self, corners):
        c = (corners.mean(1) - self.t) @ self.R
        return np.stack([self.fx * c[:, 0] / c[:, 2] + self.cx, self.fy * c[:, 1] / c[:, 2] + self.cy], 1)

    def uv(self, pts):
        """(N,M,3) world points -> integer pixel coords (N,M,2), clipped to the frame."""
        c = (pts - self.t) @ self.R
        u = np.clip(self.fx * c[..., 0] / c[..., 2] + self.cx, 0, self.W - 1)
        v = np.clip(self.fy * c[..., 1] / c[..., 2] + self.cy, 0, self.H - 1)
        return np.stack([u, v], -1).astype(int)


def gripper_mask(cam, depth):
    """Pixels whose unprojected world z exceeds GRIPPER_Z, i.e. the gripper itself."""
    h, w = depth.shape[:2]
    v, u = np.mgrid[0:h, 0:w]
    z = depth.reshape(h, w)
    cpts = np.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], -1)
    world_z = cpts @ cam.R.T[:, 2] + cam.t[2]
    return (world_z > GRIPPER_Z) & np.isfinite(world_z)


def shadow_fraction(cam, mask, xy, size, height, n=13):
    """Fraction of each bin's top-face samples that land on gripper pixels."""
    h, z = size / 2, TABLE_TOP_Z + height + BIN_DROP
    g = np.linspace(-h, h, n)
    off = np.array([[a, b, 0.0] for a in g for b in g])
    pts = np.zeros((len(xy), len(off), 3))
    pts[:, :, :2] = xy[:, None, :] + off[None, :, :2]
    pts[:, :, 2] = z
    px = cam.uv(pts)
    return mask[px[..., 1], px[..., 0]].mean(1)


def corners_of(xy, size, height):
    """(N,2) centres -> (N,8,3) world bbox corners of a square bin."""
    h, z = size / 2, TABLE_TOP_Z + height / 2 + BIN_DROP
    off = np.array(list(itertools.product((-h, h), (-h, h), (-height / 2, height / 2))))
    p = np.zeros((len(xy), 8, 3))
    p[:, :, :2] = xy[:, None, :] + off[None, :, :2]
    p[:, :, 2] = z + off[None, :, 2]
    return p


def gap_to(xy, size, box):
    """Signed AABB gap from square bins at xy to a fixed ((x0,x1),(y0,y1)) box."""
    h = size / 2
    gx = np.maximum(box[0][0] - (xy[:, 0] + h), (xy[:, 0] - h) - box[0][1])
    gy = np.maximum(box[1][0] - (xy[:, 1] + h), (xy[:, 1] - h) - box[1][1])
    return np.maximum(gx, gy)


def img_overlap(a, b):
    return ~((a[:, 1] < b[:, 0]) | (b[:, 1] < a[:, 0]) | (a[:, 3] < b[:, 2]) | (b[:, 3] < a[:, 2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--bin-size", type=float, default=DEFAULT_SIZE)
    ap.add_argument("--bin-height", type=float, default=DEFAULT_HEIGHT)
    ap.add_argument("--wrist-px", type=float, default=50.0)
    ap.add_argument("--obj-gap", type=float, default=0.030)
    ap.add_argument("--shadow", type=float, default=0.02,
                    help="max fraction of a bin opening allowed to fall in the gripper shadow")
    ap.add_argument("--bin-gap", type=float, default=0.050)
    ap.add_argument("--reach", type=float, nargs=2, default=(0.36, 0.66))
    ap.add_argument("--step", type=float, default=0.005)
    args = ap.parse_args()

    with h5py.File(CLEAN_WRIST, "r") as f:
        h, w = f["rgb"].shape[:2]
        wrist = Cam(f["intrinsic_matrix"][()], f["pos_w"][()], f["quat_w_ros"][()], w, h)
        gmask = gripper_mask(wrist, f["depth"][()])
    with h5py.File(CAPTURES / "external_obs.h5", "r") as f:
        g = f["external_cam"]
        h, w = g["rgb"].shape[:2]
        ext = Cam(g["intrinsic_matrix"][()], g["pos_w"][()], g["quat_w_ros"][()], w, h)

    xs = np.arange(0.28, 0.64 + 1e-9, args.step)
    ys = np.arange(-0.44, 0.16 + 1e-9, args.step)
    xy = np.array([[x, y] for x in xs for y in ys])

    corners = corners_of(xy, args.bin_size, args.bin_height)
    wimg, _ = wrist.boxes(corners)
    eimg, edep = ext.boxes(corners)
    r = np.hypot(xy[:, 0], xy[:, 1])

    shadow = shadow_fraction(wrist, gmask, xy, args.bin_size, args.bin_height)
    ok = (wrist.margins(wimg) >= args.wrist_px) & (ext.margins(eimg) >= 0)
    ok &= shadow <= args.shadow
    ok &= (r >= args.reach[0]) & (r <= args.reach[1])
    for box in STOCK.values():
        ok &= gap_to(xy, args.bin_size, box) >= args.obj_gap
    # a bin must not hide the banana: image overlap while strictly nearer to the cam
    ban_c = corners_of(np.array([[(BANANA[0][0] + BANANA[0][1]) / 2, (BANANA[1][0] + BANANA[1][1]) / 2]]),
                       BANANA[0][1] - BANANA[0][0], BANANA[2][1] - BANANA[2][0])
    bimg, bdep = ext.boxes(ban_c)
    ok &= ~(img_overlap(eimg, np.repeat(bimg, len(xy), 0)) & (edep[:, 1] < bdep[0, 0]))

    xy, eimg, edep, shadow = xy[ok], eimg[ok], edep[ok], shadow[ok]
    ecen = ext.centres(corners[ok])
    n = len(xy)
    print(f"bin geometry: {bin_report(args.bin_size, args.bin_height)}")
    print(f"\n{n} single-bin positions pass wrist/external/reach/clearance "
          f"(grid {args.step * 1000:.0f} mm, wrist margin >= {args.wrist_px:.0f} px, "
          f"obj gap >= {args.obj_gap * 1000:.0f} mm, gripper shadow <= {args.shadow:.0%}, "
          f"reach {args.reach[0]}-{args.reach[1]} m)")
    if n < 2:
        return

    i, j = np.triu_indices(n, 1)
    world_gap = np.maximum(
        np.abs(xy[i, 0] - xy[j, 0]), np.abs(xy[i, 1] - xy[j, 1])
    ) - args.bin_size
    keep = world_gap >= args.bin_gap
    # neither bin may hide the other in the external view
    keep &= ~(img_overlap(eimg[i], eimg[j]) & ((edep[i, 1] < edep[j, 0]) | (edep[j, 1] < edep[i, 0])))
    i, j = i[keep], j[keep]
    dpx = np.linalg.norm(ecen[i] - ecen[j], axis=1)
    order = np.argsort(-dpx)

    print(f"{len(i)} feasible pairs. Top {args.top} by external-camera separation:\n")
    print(f"{'ext_px':>7} {'world_m':>8} {'red_x':>6} {'red_y':>7} {'grn_x':>6} {'grn_y':>7} "
          f"{'r_red':>6} {'r_grn':>6} {'gap_mm':>7} {'shadow':>7}")
    shown = set()
    for k in order:
        a, b = i[k], j[k]
        # name the +Y bin red so the pair is reported consistently
        flip = xy[a][1] <= xy[b][1]
        (rx, ry), (gx, gy) = (xy[b], xy[a]) if flip else (xy[a], xy[b])
        shd = max(shadow[a], shadow[b])
        key = (round(rx, 2), round(ry, 2), round(gx, 2), round(gy, 2))
        if key in shown:
            continue
        shown.add(key)
        og = min(gap_to(np.array([[rx, ry], [gx, gy]]), args.bin_size, s).min() for s in STOCK.values())
        print(f"{dpx[k]:7.0f} {np.hypot(rx - gx, ry - gy):8.3f} {rx:6.3f} {ry:+7.3f} {gx:6.3f} {gy:+7.3f} "
              f"{np.hypot(rx, ry):6.3f} {np.hypot(gx, gy):6.3f} {og * 1000:7.0f} {shd:7.1%}")
        if len(shown) >= args.top:
            break


if __name__ == "__main__":
    main()
