"""Live preview with framing guides, for aiming the external (scoring) camera.

GWM scores a candidate trajectory from ONE third-person RGB frame plus five
robot-only renders of that trajectory. So this camera decides what the scorer
can and cannot see, and on the sim side that turned out to be first-order:
switching viewpoint moved object accuracy 9/10 -> 10/10 (G-29), and the task it
fixed had failed purely because the target sat small, distant and inside the
gripper's shadow.

It does NOT have to reproduce DROID's extrinsics. What it has to do:

  1. the WHOLE ARM, from base to fingertips, inside the frame at every pose the
     robot will reach -- the renders are of the arm, and a cropped arm cannot be
     aligned against them;
  2. every candidate object visible and not hidden behind the gripper;
  3. no strong backlight -- a window behind the workspace blows out the RGB and
     kills the IR pattern FoundationStereo needs;
  4. the table surface filling a decent part of the frame, not the background.

Keys: q quit, s save a still to ~/Desktop/rig_check/.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.aim_camera
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path.home() / "Desktop/rig_check"


def _overlay(img, stats):
    h, w = img.shape[:2]
    out = img.copy()

    # Rule-of-thirds guides and a centre box marking where the workspace should sit.
    for f in (1 / 3, 2 / 3):
        cv2.line(out, (int(w * f), 0), (int(w * f), h), (70, 70, 70), 1)
        cv2.line(out, (0, int(h * f)), (w, int(h * f)), (70, 70, 70), 1)
    x0, y0, x1, y1 = int(w * 0.12), int(h * 0.15), int(w * 0.88), int(h * 0.92)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 2)
    cv2.putText(out, "keep the whole arm + all objects inside this box",
                (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    panel = out[:120, :430].astype(np.float32) * 0.35
    out[:120, :430] = panel.astype(np.uint8)
    for i, (k, v, good) in enumerate(stats):
        colour = (80, 255, 80) if good else (80, 80, 255)
        cv2.putText(out, f"{k}: {v}", (10, 26 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tiptop"))
    from tiptop.config import tiptop_cfg
    from tiptop.perception.cameras.rs_camera import RealsenseCamera

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=None,
                    help="default: cameras.external from tiptop.yml")
    args = ap.parse_args()

    serial = args.serial or str(tiptop_cfg().cameras.external.serial)
    print(f"previewing external camera {serial} -- q to quit, s to save")
    OUT.mkdir(parents=True, exist_ok=True)
    cam = RealsenseCamera(serial, enable_depth=True, enable_ir=True)
    try:
        import pyrealsense2 as rs
        ds = cam._profile.get_device().first_depth_sensor()
        if ds.supports(rs.option.enable_auto_exposure):
            ds.set_option(rs.option.enable_auto_exposure, 1)

        while True:
            f = cam.read_camera()
            rgb = f.rgb
            d = f.depth
            valid = float(((d > 0) & (d < 5)).mean()) if d is not None else 0.0
            ir = f.ir_left
            sat = float((ir >= 250).mean())
            blown = float((rgb.max(axis=2) >= 250).mean())
            near = float(np.median(d[(d > 0) & (d < 5)])) if valid > 0.01 else float("nan")

            stats = [
                ("depth valid", f"{valid*100:5.1f} %", valid > 0.60),
                ("IR saturated", f"{sat*100:5.1f} %", sat < 0.10),
                ("RGB blown out", f"{blown*100:5.1f} %", blown < 0.15),
                ("median range", f"{near:.2f} m", 0.5 < near < 2.0),
            ]
            view = _overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), stats)
            cv2.imshow("external camera -- aim me", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                p = OUT / f"external_{serial}_{int(time.time())}.png"
                cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                print(f"saved {p}")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
