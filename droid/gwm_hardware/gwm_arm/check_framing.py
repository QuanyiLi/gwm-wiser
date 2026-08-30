"""Is the scoring camera aimed well enough? Answered in numbers, not by eye.

The third-person camera decides what GWM can see, and that is first-order: a
target that sits small or gripper-shadowed in the scoring view cannot be
scored well, whatever the candidates are. So "looks about right" is not good
enough, and this is the check that replaces it.

It answers three questions the naked eye cannot:

  1. **Is any of the robot outside the frame, and how much?** The arm is
     rendered onto a canvas deliberately larger than the real image, so the
     parts that fall outside can be measured rather than merely missed.
  2. **Can it be fixed by tilting, or does the camera have to move back?**
     If the robot's total span already exceeds the frame, tilting only trades
     one clipped end for the other, and the tool says so with the distance
     factor needed instead.
  3. **What would tilting cost?** Which end gets clipped, and by how much.

If something must be clipped, clip the BASE, not the gripper. The base is in
the same place in every candidate's render and carries no signal to
discriminate them; the gripper is where the entire decision lives.

No SAPIEN and no meshes: the robot's footprint comes from cuRobo's own
collision spheres, projected. That is coarser than a mesh render by roughly a
sphere radius, which is far below the precision any aiming decision needs, and
it keeps the whole thing inside the tiptop pixi env where the robot lives.

Needs the Charuco board on the table, visible to the WRIST camera and to the
camera being checked -- it borrows one board placement for a provisional
extrinsic. That is enough to aim with; it is NOT a calibration (no error bar),
and `extcam_calib` is still what produces the pose the scorer uses.

READ-ONLY on the robot: joint angles only, never a motion command.

    python -m gwm_hardware.gwm_arm.check_framing
    python -m gwm_hardware.gwm_arm.check_framing --cam external_cam_2
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from gwm_hardware.common.paths import PKG_ROOT
from gwm_hardware.gwm_arm.capture import EXTERNAL_CAM, external_camera_specs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_arm.check_framing")

MARGIN_PX = 60          # breathing room we ask for on every edge
PAD_PX = 900            # canvas grown beyond the real frame, to see what falls off


def provisional_extrinsic(cam_name: str, serial: str):
    """(q, K, world_from_cam, external rgb) from one shared board placement."""
    from curobo.types.base import TensorDeviceType

    from tiptop.config import load_calibration, tiptop_cfg
    from tiptop.perception.cameras.rs_camera import RealsenseCamera
    from gwm_tiptop.robot_fk import fk_model
    from tiptop.utils import get_robot_client

    from gwm_hardware.gwm_arm.extcam_calib import board_pose

    cfg = tiptop_cfg()
    tensor_args = TensorDeviceType()
    kin = fk_model(tensor_args)
    q = np.asarray(get_robot_client().get_joint_positions(), dtype=np.float64)
    world_from_ee = kin.get_state(tensor_args.to_device(q)).ee_pose.get_numpy_matrix()[0]

    hand = RealsenseCamera(str(cfg.cameras.hand.serial))
    try:
        hf, hi = hand.read_camera(), hand.get_intrinsics()
        wb = board_pose(hf.rgb, np.asarray(hi.K_color), np.asarray(hi.distortion_color))
    finally:
        hand.close()
    if not wb:
        raise SystemExit(
            "the WRIST camera cannot see the Charuco board. It is what carries the base "
            "frame to the board, so nothing can be measured without it -- move the board "
            "into the wrist camera's view (viz-gripper-cam shows what it sees)."
        )
    base_from_board = world_from_ee @ np.asarray(load_calibration(hand.serial)) @ wb[0]

    ext = RealsenseCamera(serial)
    try:
        ef, ei = ext.read_camera(), ext.get_intrinsics()
        eb = board_pose(ef.rgb, np.asarray(ei.K_color), np.asarray(ei.distortion_color))
        rgb, K = ef.rgb, np.asarray(ei.K_color, dtype=np.float64)
    finally:
        ext.close()
    if not eb:
        raise SystemExit(
            f"{cam_name} ({serial}) cannot see the Charuco board. Either the board is "
            "outside its view, or the camera is not pointed at the workspace at all -- "
            "which is itself the answer to the framing question."
        )
    _log.info(f"wrist sees {wb[1]}/70 corners ({wb[2]:.2f} px), "
              f"{cam_name} sees {eb[1]}/70 ({eb[2]:.2f} px)")
    return q, K, base_from_board @ np.linalg.inv(eb[0]), rgb


def robot_pixels(q: np.ndarray, K: np.ndarray, world_from_cam: np.ndarray,
                 width: int, height: int, pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Project the robot's collision spheres -> (uv (N,2), radius in px)."""
    from curobo.types.base import TensorDeviceType

    from gwm_tiptop.robot_fk import fk_model

    tensor_args = TensorDeviceType()
    state = fk_model(tensor_args).get_state(tensor_args.to_device(q))
    sph = state.get_link_spheres()[0].cpu().numpy().astype(np.float64)
    sph = sph[sph[:, 3] > 0.0]        # cuRobo pads the buffer with negative radii

    w2c = np.linalg.inv(world_from_cam)
    pc = sph[:, :3] @ w2c[:3, :3].T + w2c[:3, 3]
    front = pc[:, 2] > 0.05
    pc, r = pc[front], sph[front, 3]
    uv = (K @ pc.T)
    uv = (uv[:2] / uv[2]).T
    r_px = r * K[0, 0] / pc[:, 2]
    return uv, r_px


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", default=EXTERNAL_CAM)
    ap.add_argument("--out-dir", type=Path, default=PKG_ROOT / "runs/framing")
    args = ap.parse_args()

    import cv2

    specs = dict(external_camera_specs())
    if args.cam not in specs:
        raise SystemExit(f"unknown camera {args.cam!r}; known: {sorted(specs)}")
    q, K, c2w, rgb = provisional_extrinsic(args.cam, specs[args.cam])
    h, w = rgb.shape[:2]
    uv, r_px = robot_pixels(q, K, c2w, w, h, PAD_PX)

    top = float((uv[:, 1] - r_px).min())
    bottom = float((uv[:, 1] + r_px).max())
    left = float((uv[:, 0] - r_px).min())
    right = float((uv[:, 0] + r_px).max())
    span_v, span_h = bottom - top, right - left

    _log.info(f"camera at {np.round(c2w[:3, 3], 3).tolist()} in the base frame")
    _log.info(f"frame {w}x{h}; robot occupies x {left:.0f}..{right:.0f}, y {top:.0f}..{bottom:.0f}")
    _log.info(f"robot needs {span_h:.0f} x {span_v:.0f} px; frame gives {w} x {h}")

    clip = {"top": max(0.0, -top), "bottom": max(0.0, bottom - h),
            "left": max(0.0, -left), "right": max(0.0, right - w)}
    for edge, px in clip.items():
        if px > 0:
            _log.warning(f"{px:.0f} px of the robot are past the {edge} edge")

    need_v = span_v + 2 * MARGIN_PX
    need_h = span_h + 2 * MARGIN_PX
    factor = max(need_v / h, need_h / w)
    if factor > 1.0:
        _log.error(
            f"the robot cannot fit at this distance: it spans {span_v:.0f} px vertically "
            f"in a {h} px frame. Tilting only swaps which end is clipped. "
            f"MOVE THE CAMERA BACK by a factor of {factor:.2f} "
            f"(currently {np.linalg.norm(c2w[:3, 3]):.2f} m from the robot base, "
            f"so roughly +{np.linalg.norm(c2w[:3, 3]) * (factor - 1):.2f} m)."
        )
    elif any(v > 0 for v in clip.values()):
        dy = (MARGIN_PX - top) if clip["top"] else -(bottom - h + MARGIN_PX)
        _log.warning(
            f"it FITS at this distance but is not centred: tilt the camera "
            f"{'UP' if dy > 0 else 'DOWN'} by {abs(np.degrees(np.arctan2(dy, K[1, 1]))):.1f} deg "
            f"({abs(dy):.0f} px) and it will be inside the frame."
        )
    else:
        _log.info(f"FITS with {min(top, h - bottom, left, w - right):.0f} px to spare on "
                  "the tightest edge")

    # A picture of the same numbers: the robot's spheres over the photo, and a
    # box showing where they would have to end up.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    for (u, v), rr in zip(uv, r_px):
        if -PAD_PX < v < h + PAD_PX and -PAD_PX < u < w + PAD_PX:
            cv2.circle(img, (int(u), int(v)), max(int(rr), 1), (90, 255, 130), 1, cv2.LINE_AA)
    # No inset "keep it in here" rectangle: the usable area is the whole frame,
    # and drawing a smaller one only makes a fine camera look clipped. Edges the
    # robot actually spills over get marked red instead.
    for edge, px in clip.items():
        if px <= 0:
            continue
        pts = {"top": ((0, 0), (w, 8)), "bottom": ((0, h - 8), (w, h)),
               "left": ((0, 0), (8, h)), "right": ((w - 8, 0), (w, h))}[edge]
        cv2.rectangle(img, pts[0], pts[1], (60, 60, 255), -1)
    cv2.putText(img, f"{args.cam}: robot needs {span_v:.0f} px of {h}"
                     f"   ({100 * (1 - sum(clip.values()) / max(span_v, 1)):.0f}% in frame)",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    out = args.out_dir / f"framing_{args.cam}.png"
    cv2.imwrite(str(out), img)
    _log.info(f"-> {out}")


if __name__ == "__main__":
    main()
