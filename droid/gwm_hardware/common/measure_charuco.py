"""Measure the Charuco board's checker size with the wrist camera itself.

Tape readings of a laminated board are ambiguous at the sub-millimetre level:
spans over different numbers of squares can disagree by close to a percent,
depending on which edge was read. That error lands straight in the hand-eye
solve and then in every grasp, and no amount of re-reading a ruler resolves it.

The depth camera does resolve it. Detect the interior Charuco corners, read
each one's metric depth, unproject to 3D in the camera frame, and measure the
distance between adjacent corners. That distance *is* the checker size, in
metres, from an instrument that is already metrically calibrated -- no ruler,
no reading of where the laminate ends.

Depth comes from FoundationStereo (the same estimator tiptop plans on), with
the RealSense ASIC depth reported alongside as an independent second opinion.

Hold the board flat and square-on in the wrist camera's view, 30-60 cm away,
then:

    python -m gwm_hardware.common.measure_charuco
"""

import argparse
import asyncio

import numpy as np

SQUARES_X, SQUARES_Y = 11, 8
DICT = "DICT_5X5_100"
# Nominal, only used to build a detector; the answer does not depend on it.
NOMINAL_CHECKER_M = 0.034


def _plane_fit(pts):
    cen = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - cen, full_matrices=False)
    n = vt[-1]
    return n / np.linalg.norm(n), cen


def measure(rgb, depth, K, label):
    import cv2
    from cv2 import aruco

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    d = aruco.getPredefinedDictionary(getattr(aruco, DICT))
    board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), NOMINAL_CHECKER_M,
                               NOMINAL_CHECKER_M * 0.7117, d)
    cc, ci, _, _ = aruco.CharucoDetector(board).detectBoard(gray)
    if cc is None or len(cc) < 12:
        print(f"  {label}: only {0 if cc is None else len(cc)} corners -- "
              "hold the board flatter / closer / better lit")
        return None
    cc = cc.reshape(-1, 2)
    ci = ci.ravel()

    # Unproject each corner with its depth, then measure neighbour distances.
    pts, keep = {}, 0
    for uv, i in zip(cc, ci):
        u, v = uv
        z = depth[int(round(v)), int(round(u))]
        if not np.isfinite(z) or z <= 0.05 or z > 3.0:
            continue
        pts[int(i)] = np.array([(u - K[0, 2]) * z / K[0, 0],
                                (v - K[1, 2]) * z / K[1, 1], z])
        keep += 1
    if keep < 12:
        print(f"  {label}: only {keep} corners had usable depth")
        return None

    # Corners lie on a plane; projecting onto the fitted plane suppresses
    # per-pixel depth noise, which is the dominant error here.
    ids = sorted(pts)
    P = np.array([pts[i] for i in ids])
    n, cen = _plane_fit(P)
    P = P - np.outer((P - cen) @ n, n)
    proj = {i: p for i, p in zip(ids, P)}

    dists = []
    for i, p in proj.items():
        r, c = divmod(i, SQUARES_X - 1)
        if c < SQUARES_X - 2 and (i + 1) in proj:
            dists.append(np.linalg.norm(proj[i + 1] - p))
        if (i + SQUARES_X - 1) in proj:
            dists.append(np.linalg.norm(proj[i + SQUARES_X - 1] - p))
    dists = np.array(dists)
    med = float(np.median(dists))
    print(f"  {label}: {keep} corners with depth, {len(dists)} neighbour pairs, "
          f"range {np.median(P[:, 2]):.3f} m")
    print(f"    checker = {med*1000:.2f} mm   "
          f"(iqr {np.percentile(dists,75)*1000-np.percentile(dists,25)*1000:.2f} mm)")
    return med


def main() -> None:
    import aiohttp
    from tiptop.config import tiptop_cfg
    from tiptop.perception.cameras.rs_camera import RealsenseCamera, rs_infer_depth_async

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=5)
    args = ap.parse_args()

    cfg = tiptop_cfg()
    cam = RealsenseCamera(str(cfg.cameras.hand.serial), enable_depth=True, enable_ir=True)
    results = {"FoundationStereo": [], "RealSense ASIC": []}
    try:
        async def run():
            async with aiohttp.ClientSession() as s:
                for k in range(args.shots):
                    for _ in range(8):
                        f = cam.read_camera()
                    fs = await rs_infer_depth_async(s, f, cam.get_intrinsics())
                    K = cam.get_intrinsics().K_color
                    print(f"shot {k+1}/{args.shots}")
                    a = measure(f.rgb, fs, K, "FoundationStereo")
                    b = measure(f.rgb, f.depth, K, "RealSense ASIC ")
                    if a: results["FoundationStereo"].append(a)
                    if b: results["RealSense ASIC"].append(b)
        asyncio.run(run())
    finally:
        cam.close()

    print("\nsummary:")
    for k, v in results.items():
        if v:
            print(f"  {k:18s} median {np.median(v)*1000:.2f} mm  "
                  f"over {len(v)} shots  spread {(max(v)-min(v))*1000:.2f} mm")
    print("\n  compare against the tape reading (span the whole grid and divide)")
    print("  pass the value you trust to install_charuco_params --checker-mm")


if __name__ == "__main__":
    main()
