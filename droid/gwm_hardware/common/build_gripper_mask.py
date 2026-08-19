"""Generate the wrist-camera gripper mask by DEPTH, cross-checked against geometry.

tiptop zeroes the point cloud wherever this mask is True
(`perception_wrapper.py:91`, `xyz_map[gripper_mask] = 0.0`). Upstream builds it
with `compute-gripper-mask`: Gemini finds the gripper, SAM segments it, a human
approves. Two reasons that does not apply here.

**The shipped mask is actively harmful.** It is DROID's 2F-85 + ZED silhouette
and covers 20.6 % of the frame, reaching to y = 403. On this rig that region is
clean tabletop, so it silently deletes a fifth of the scene.

**The 2F-140 is barely in frame at all**, so Gemini has almost nothing to find.

What works is depth. At the capture pose the table sits at 0.55-0.65 m and the
gripper at 0.15-0.25 m -- a 3x separation, so a threshold cuts them apart from
direct observation, with no dependence on the URDF, the hand-eye extrinsic, or
a detector.

That independence matters, because geometry and depth disagree here. Projecting
the URDF's collision meshes through the calibrated extrinsic puts the fingers at
image x 400-470 and 800-1000; depth puts the real near-field blobs at x 299-397
and x 1246-1280. The left one is ~90 px off and the right one lands on nothing.
Some of that is hand-eye residual, and some is likely that the physical 2F-140
does not carry the stock fingertips the URDF models. Either way the measurement
wins over the model, and `--from geometry` is kept only so the two can be
compared.

    python -m gwm_hardware.common.build_gripper_mask                 # dry run
    python -m gwm_hardware.common.build_gripper_mask --install
"""

import argparse
import asyncio
import shutil
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 1280, 720
# Well below the table (0.55 m) and well above the gripper (0.25 m).
NEAR_THRESHOLD_M = 0.40
DILATE_PX = 15          # margin for depth noise at the silhouette edge
MIN_BLOB_PX = 300
FRAMES = 7
from gwm_hardware.common.paths import PKG_ROOT as HERE


def depth_mask(frames_depth):
    """Union of near-field pixels across frames, de-speckled and dilated."""
    import cv2

    acc = np.zeros((HEIGHT, WIDTH), np.uint16)
    for d in frames_depth:
        acc += (np.isfinite(d) & (d > 0.02) & (d < NEAR_THRESHOLD_M)).astype(np.uint16)
    # Require the pixel to be near in most frames: a single noisy frame should
    # not carve a hole in the tabletop.
    m = (acc >= max(1, len(frames_depth) // 2)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    blobs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_PX:
            keep[lab == i] = 1
            blobs.append((int(stats[i, cv2.CC_STAT_AREA]),
                          int(stats[i, cv2.CC_STAT_LEFT]),
                          int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH])))
    keep = cv2.dilate(keep, np.ones((DILATE_PX, DILATE_PX), np.uint8))
    return keep.astype(bool), blobs


def main() -> None:
    global NEAR_THRESHOLD_M
    import aiohttp
    import cv2
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--threshold", type=float, default=NEAR_THRESHOLD_M)
    ap.add_argument("--frames", type=int, default=FRAMES)
    args = ap.parse_args()

    from tiptop.config import tiptop_cfg
    from tiptop.perception.cameras.rs_camera import RealsenseCamera, rs_infer_depth_async
    from tiptop.utils import gripper_mask_path

    NEAR_THRESHOLD_M = args.threshold

    cfg = tiptop_cfg()
    cam = RealsenseCamera(str(cfg.cameras.hand.serial), enable_depth=True, enable_ir=True)

    async def grab():
        out = []
        async with aiohttp.ClientSession() as s:
            for _ in range(args.frames):
                for _ in range(4):
                    f = cam.read_camera()
                out.append(await rs_infer_depth_async(s, f, cam.get_intrinsics()))
        return out
    try:
        depths = asyncio.run(grab())
    finally:
        cam.close()

    table = np.concatenate([d[np.isfinite(d) & (d > 0.45)] for d in depths])
    print(f"{args.frames} frames; table sits at {np.median(table):.3f} m, "
          f"threshold {NEAR_THRESHOLD_M:.2f} m")

    mask, blobs = depth_mask(depths)
    print(f"mask covers {mask.sum()} px ({mask.mean()*100:.2f} % of the frame) "
          f"after {DILATE_PX} px dilation")
    for a, l, r in sorted(blobs, key=lambda t: -t[0]):
        print(f"  blob {a:6d} px, image x {l} to {r}")
    if mask.any():
        ys, xs = np.nonzero(mask)
        print(f"  bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")

    old = Path(gripper_mask_path)
    if old.exists():
        prev = np.array(Image.open(old)).astype(bool)
        print(f"currently installed: {prev.mean()*100:.2f} %")

    out = HERE / "assets/gripper_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(out)
    print(f"wrote {out}")

    if args.install:
        backup = HERE / "config/gripper_mask.png.upstream"
        if old.exists() and not backup.exists():
            shutil.copy2(old, backup)
            print(f"saved the upstream mask to {backup}")
        shutil.copy2(out, old)
        print(f"installed to {old}")
    else:
        print("(dry run -- pass --install)")


if __name__ == "__main__":
    main()
