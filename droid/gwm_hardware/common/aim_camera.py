"""Live preview with framing guides, for aiming the external (scoring) camera.

GWM scores a candidate trajectory from ONE third-person RGB frame plus five
robot-only renders of that trajectory. So this camera decides what the scorer
can and cannot see, and on the sim side that turned out to be first-order:
switching viewpoint moved object accuracy 9/10 -> 10/10 (G-29), and the task it
fixed had failed purely because the target sat small, distant and inside the
gripper's shadow.

It does NOT have to reproduce DROID's extrinsics, and it does NOT have to fit
the whole arm in frame. An earlier version of this file claimed it did, on the
reasoning that "a cropped arm cannot be aligned against a render" -- which is
simply wrong: the render is produced with THIS camera's intrinsics and
extrinsics, so it is cropped in exactly the same place as the photo. The
overlay gate passes at 8.04 % robot coverage with the base out of frame
(2026-08-19). What it has to do:

  1. enough of the arm visible to distinguish one trajectory from another --
     the gripper end above all, since that is what differs between candidates;
  2. every candidate object visible and not hidden behind the gripper;
  3. no strong backlight -- a window behind the workspace blows out the RGB,
     and the scorer only ever sees RGB;
  4. the table surface filling a decent part of the frame, not the background.

What a scoring camera does NOT have to do is produce usable depth. Only the
WRIST camera feeds FoundationStereo; the third-person views hand over RGB, K
and a pose and nothing else. So the depth-valid and IR-saturation rows are
shown for information and are NOT graded by default -- a camera can fail
`rs_preflight` outright and still be a perfectly good scoring viewpoint. Pass
`--role depth` to grade them, e.g. when aiming the wrist camera.

Aim it with the arm AT THE CAPTURE POSE (`go-to-capture`): that is the pose
every scene photo is taken in, and where the gripper actually sits.

`--with-robot` draws the ROBOT ITSELF into the preview, live, instead of
leaving you to judge a generic box by eye. It works because the Charuco board
on the table gives a pose every frame: the wrist camera fixes the board in the
base frame once (the arm must not move afterwards), then each preview frame
re-detects the board in THIS camera and re-solves where this camera is. So the
projection follows the camera as you move it, and you can see the arm slide
into frame as you aim. A live read-out says how much of the robot is inside and
which edge is clipping it.

That mode needs: the robot reachable (joint angles only -- it never commands
motion), the board visible to the wrist camera AND to this one, and the arm
held still.

Keys: q quit, s save a still to ~/Desktop/rig_check/.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.aim_camera
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.aim_camera \
        --with-robot
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path.home() / "Desktop/rig_check"


def _overlay(img, stats, clipped_edges=()):
    """Stats panel, plus a marker on any frame edge the robot is spilling over.

    There is deliberately no inset "keep everything in here" box. The usable
    area is the whole frame, and an arbitrary margin drawn inside it only makes
    the camera look worse than it is. The in-frame percentage is a read-out,
    not a target: the arm does not have to be wholly inside (see the module
    docstring), so a red edge marker means "this edge is cutting the arm", not
    "this is wrong".
    """
    h, w = img.shape[:2]
    out = img.copy()
    for f in (1 / 3, 2 / 3):
        cv2.line(out, (int(w * f), 0), (int(w * f), h), (70, 70, 70), 1)
        cv2.line(out, (0, int(h * f)), (w, int(h * f)), (70, 70, 70), 1)
    for edge in clipped_edges:
        p0, p1 = {"top": ((0, 0), (w, 10)), "bottom": ((0, h - 10), (w, h)),
                  "left": ((0, 0), (10, h)), "right": ((w - 10, 0), (w, h))}[edge]
        cv2.rectangle(out, p0, p1, (60, 60, 255), -1)

    n = max(len(stats), 1)
    panel = out[:26 * n + 12, :470].astype(np.float32) * 0.35
    out[:26 * n + 12, :470] = panel.astype(np.uint8)
    for i, (k, v, good) in enumerate(stats):
        colour = (80, 255, 80) if good else (80, 80, 255)
        cv2.putText(out, f"{k}: {v}", (10, 26 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


def _robot_anchor():
    """(spheres (N,4) in the base frame, TCP (3,), base_from_board (4,4)).

    Read once: the arm must stand still while aiming, so its spheres are fixed,
    and so is where the board sits in the base frame. Only the CAMERA moves,
    and that is re-solved per frame from the board.
    """
    import numpy as np
    from curobo.types.base import TensorDeviceType

    from tiptop.config import load_calibration, tiptop_cfg
    from tiptop.perception.cameras.rs_camera import RealsenseCamera
    from tiptop.utils import get_robot_client

    from gwm_hardware.gwm_arm.extcam_calib import board_pose

    cfg = tiptop_cfg()
    tensor_args = TensorDeviceType()
    from gwm_tiptop.robot_fk import fk_model

    q = np.asarray(get_robot_client().get_joint_positions(), dtype=np.float64)
    state = fk_model(tensor_args).get_state(tensor_args.to_device(q))
    spheres = state.get_link_spheres()[0].cpu().numpy().astype(np.float64)
    spheres = spheres[spheres[:, 3] > 0.0]
    tcp = state.ee_pose.position[0].cpu().numpy().astype(np.float64)
    world_from_ee = state.ee_pose.get_numpy_matrix()[0]

    from gwm_hardware.common.rs_open import open_with_retry
    hand = open_with_retry(lambda: RealsenseCamera(str(cfg.cameras.hand.serial)),
                           cfg.cameras.hand.serial)
    try:
        hf, hi = hand.read_camera(), hand.get_intrinsics()
        wb = board_pose(hf.rgb, np.asarray(hi.K_color), np.asarray(hi.distortion_color))
    finally:
        hand.close()
    if not wb:
        raise SystemExit(
            "the WRIST camera cannot see the Charuco board, and it is what puts the "
            "board in the base frame. Move the board into its view (`viz-gripper-cam` "
            "shows what it sees), or drop --with-robot."
        )
    base_from_board = world_from_ee @ np.asarray(load_calibration(hand.serial)) @ wb[0]
    print(f"anchored: wrist sees {wb[1]}/70 corners ({wb[2]:.2f} px reprojection); "
          f"board at {np.round(base_from_board[:3, 3], 3).tolist()} in the base frame")
    print("KEEP THE ARM STILL from here on -- the overlay assumes it has not moved.")
    return spheres, tcp, base_from_board


def _draw_robot(img, spheres, tcp, base_from_board, K, dist):
    """Project the robot through a per-frame board solve. Returns a stats row."""
    from gwm_hardware.gwm_arm.extcam_calib import board_pose

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bp = board_pose(rgb, K, dist)
    if not bp:
        cv2.putText(img, "board not visible -- cannot place the robot",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2)
        return ("robot in frame", "board lost", False), []

    world_from_cam = base_from_board @ np.linalg.inv(bp[0])
    w2c = np.linalg.inv(world_from_cam)
    pc = spheres[:, :3] @ w2c[:3, :3].T + w2c[:3, 3]
    front = pc[:, 2] > 0.05
    pc, rad = pc[front], spheres[front, 3]
    uv = (K @ pc.T)
    uv = (uv[:2] / uv[2]).T
    r_px = rad * K[0, 0] / pc[:, 2]

    inside = ((uv[:, 0] - r_px > 0) & (uv[:, 0] + r_px < w)
              & (uv[:, 1] - r_px > 0) & (uv[:, 1] + r_px < h))
    for (u, v), rr, ins in zip(uv, r_px, inside):
        if -2000 < u < w + 2000 and -2000 < v < h + 2000:
            cv2.circle(img, (int(u), int(v)), max(int(rr), 1),
                       (120, 255, 140) if ins else (60, 60, 255), 1, cv2.LINE_AA)

    tp = (K @ (tcp @ w2c[:3, :3].T + w2c[:3, 3]))
    if tp[2] > 0.05:
        cv2.drawMarker(img, (int(tp[0] / tp[2]), int(tp[1] / tp[2])), (0, 220, 255),
                       cv2.MARKER_TILTED_CROSS, 44, 3)

    # Name the edge that is costing the most, so aiming has a direction.
    over = {"top": float(max(0.0, -(uv[:, 1] - r_px).min())),
            "bottom": float(max(0.0, (uv[:, 1] + r_px).max() - h)),
            "left": float(max(0.0, -(uv[:, 0] - r_px).min())),
            "right": float(max(0.0, (uv[:, 0] + r_px).max() - w))}
    note = f"{inside.mean()*100:4.0f} %"
    spilling = [e for e, px in over.items() if px > 0]
    if spilling:
        note += "  (" + ", ".join(f"{over[e]:.0f} px past {e}" for e in spilling) + ")"
    return ("robot in frame", note, bool(inside.all())), spilling


def main() -> None:
    import sys
    from gwm_hardware.common.paths import TIPTOP_ROOT
    sys.path.insert(0, str(TIPTOP_ROOT))
    from tiptop.config import tiptop_cfg
    from tiptop.perception.cameras.rs_camera import RealsenseCamera

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=None,
                    help="default: cameras.external from tiptop.yml")
    ap.add_argument("--role", default="scoring", choices=["scoring", "depth"],
                    help="scoring (default): grade RGB and framing only, since a "
                         "third-person view never produces depth for the pipeline. "
                         "depth: also grade depth validity and IR saturation")
    ap.add_argument("--with-robot", action="store_true",
                    help="draw the robot into the preview, re-solved from the Charuco "
                         "board every frame. Needs the robot reachable (reads joint "
                         "angles only) and the arm held still")
    args = ap.parse_args()

    serial = args.serial or str(tiptop_cfg().cameras.external.serial)
    anchor = _robot_anchor() if args.with_robot else None
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

            grade_depth = args.role == "depth"
            stats = [
                ("RGB blown out", f"{blown*100:5.1f} %", blown < 0.15),
                ("median range", f"{near:.2f} m", 0.5 < near < 2.0),
                # Informational for a scoring camera: nothing downstream asks a
                # third-person view for depth, so a red row here is not a reason
                # to move it.
                ("depth valid", f"{valid*100:5.1f} %" + ("" if grade_depth else "  (not graded)"),
                 valid > 0.60 if grade_depth else True),
                ("IR saturated", f"{sat*100:5.1f} %" + ("" if grade_depth else "  (not graded)"),
                 sat < 0.10 if grade_depth else True),
            ]
            view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            clipped = []
            if anchor is not None:
                intr = cam.get_intrinsics()
                row, clipped = _draw_robot(view, *anchor, np.asarray(intr.K_color),
                                           np.asarray(intr.distortion_color))
                stats.insert(0, row)
            view = _overlay(view, stats, clipped)
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
