"""The hard gate: does the rendered robot land on the real one?

GWM scores a candidate by looking at `[the external photo, five robot-only
renders of the candidate]`. If the render and the photo disagree about where
the robot is, every RAT frame is a lie and the scores are noise dressed as
semantics. In droid-sim this was GI-2, and it passed on ground truth the sim
handed over: FK matched `body_pos_w` to 0.0 mm on every arm link. Hardware has
no such oracle -- the extrinsics come from `extcam_calib`, the kinematics from
a generated URDF, the flange standoff from a tape measure -- so the check has
to be made against pixels.

Two numbers, from `real_data_train.renderer.edge_gate` (the same gate that
admits DROID streams into the training tree, D-28), because either alone can
be fooled:

  lift    oriented-edge agreement above chance. The rendered silhouette's
          contour is scored against strong image edges of MATCHING orientation,
          minus the hit rate a silhouette dropped at random would get. Being
          above chance is contrast-invariant, which raw edge-hit fractions are
          not.
  margin  lift minus the lift of the SAME render under a camera deliberately
          rotated by PERTURB_DEG. This is the one that catches a plausible but
          wrong calibration: a busy scene gives any silhouette a decent lift,
          and only a correct pose beats its own perturbation.

A failure is not ambiguous about where to look, and the order matters:

  1. the extrinsics      -- re-run `extcam_calib`, check its placement spread
  2. the flange standoff -- `common/build_2f140.py --flange-offset`; this rig
                            assumes 0 (gripper bolted straight to the flange)
  3. the URDF choice     -- `render_model.py` must be building the 2F-140

    python -m gwm_hardware.gwm_arm.overlay_gate \
        --external-h5 runs/gwm/scene01/external_obs.h5 \
        --out-dir runs/gwm/scene01/overlay
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_arm.overlay_gate")

# droid-sim's validated overlay sat at 13.3-13.5 % robot pixel coverage from a
# comparable third-person distance (G-29). Coverage is not a correctness test
# -- it catches the gross failures (camera pointing elsewhere, robot off frame)
# that would make the edge numbers meaningless to interpret.
MIN_COVERAGE = 0.02
MAX_COVERAGE = 0.45


def render_at(urdf: Path, q: np.ndarray, gripper: float, K: np.ndarray,
              c2w: np.ndarray, w: int, h: int) -> np.ndarray:
    from real_data_train.renderer.franka_renderer import FrankaRobotRenderer

    r = FrankaRobotRenderer(str(urdf), arm="panda")
    return r.render(np.asarray(q)[None], np.array([gripper]), K, c2w, w, h)[0]


def perturbed(c2w: np.ndarray, deg: float) -> np.ndarray:
    """Rotate the camera about its own y axis -- the edge gate's null model."""
    from scipy.spatial.transform import Rotation

    out = c2w.copy()
    out[:3, :3] = c2w[:3, :3] @ Rotation.from_euler("y", deg, degrees=True).as_matrix()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external-h5", required=True, type=Path)
    ap.add_argument("--cam", default=None,
                    help="comma-separated; default = every camera in the h5. "
                         "Each has its own extrinsics, so each is gated separately")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--gripper", type=float, default=0.0,
                    help="gripper state in the photo, 0 open .. 1 closed")
    ap.add_argument("--urdf", type=Path, default=None)
    ap.add_argument("--min-lift", type=float, default=None)
    ap.add_argument("--min-margin", type=float, default=None)
    args = ap.parse_args()

    import h5py
    from PIL import Image
    from scipy.spatial.transform import Rotation

    from real_data_train.renderer.edge_gate import (
        DEFAULT_MIN_MARGIN,
        DEFAULT_MIN_SCORE,
        PERTURB_DEG,
        silhouette_edge_score,
    )

    from gwm_hardware.gwm_arm.render_model import ensure_render_urdf

    min_lift = DEFAULT_MIN_SCORE if args.min_lift is None else args.min_lift
    min_margin = DEFAULT_MIN_MARGIN if args.min_margin is None else args.min_margin
    urdf = args.urdf or ensure_render_urdf()

    with h5py.File(args.external_h5) as f:
        if "q_init" not in f:
            raise SystemExit(
                f"{args.external_h5} has no q_init. The gate renders the robot at the "
                "configuration the photo was taken in; without it there is nothing to "
                "compare. Re-capture with gwm_hardware.gwm_arm.capture live."
            )
        q = np.asarray(f["q_init"], dtype=np.float64)
        cams = ([c.strip() for c in args.cam.split(",") if c.strip()] if args.cam
                else [k for k in f if isinstance(f[k], h5py.Group)])
        views = {}
        for cam in cams:
            if cam not in f:
                raise SystemExit(f"{args.external_h5} has no group {cam!r}")
            c2w = np.eye(4)
            w_, x_, y_, z_ = np.asarray(f[f"{cam}/quat_w_ros"])
            c2w[:3, :3] = Rotation.from_quat([x_, y_, z_, w_]).as_matrix()
            c2w[:3, 3] = np.asarray(f[f"{cam}/pos_w"])
            views[cam] = (np.asarray(f[f"{cam}/rgb"])[..., :3],
                          np.asarray(f[f"{cam}/intrinsic_matrix"], dtype=np.float64), c2w)

    # One renderer per camera would be wasteful; the resolution differs per view
    # though, so the render call is what varies, not the model.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report, failed = {}, []
    for cam, (rgb, K, c2w) in views.items():
        h, w = rgb.shape[:2]
        render = render_at(urdf, q, args.gripper, K, c2w, w, h)
        render_bad = render_at(urdf, q, args.gripper, K, perturbed(c2w, PERTURB_DEG), w, h)

        mask = render.max(axis=2) > 8
        coverage = float(mask.mean())
        lift = float(silhouette_edge_score(render, rgb))
        lift_bad = float(silhouette_edge_score(render_bad, rgb))
        margin = lift - lift_bad

        checks = {
            "coverage": (MIN_COVERAGE <= coverage <= MAX_COVERAGE,
                         f"{coverage:.2%} (want {MIN_COVERAGE:.0%}..{MAX_COVERAGE:.0%})"),
            "edge_lift": (lift >= min_lift, f"{lift:+.4f} (want >= {min_lift})"),
            "perturb_margin": (margin >= min_margin,
                               f"{margin:+.4f} = {lift:+.4f} - {lift_bad:+.4f} at "
                               f"{PERTURB_DEG} deg (want >= {min_margin})"),
        }
        verdict = "PASS" if all(ok for ok, _ in checks.values()) else "FAIL"
        if verdict == "FAIL":
            failed.append(cam)

        blend = rgb.copy()
        blend[mask] = (0.45 * rgb[mask] + 0.55 * np.array([80, 255, 120])).astype(np.uint8)
        Image.fromarray(np.concatenate([rgb, render, blend], axis=1)).save(
            args.out_dir / f"overlay_{cam}.png")

        report[cam] = {
            "verdict": verdict, "coverage": coverage, "edge_lift": lift,
            "edge_lift_perturbed": lift_bad, "margin": margin,
            "checks": {k: {"pass": ok, "detail": d} for k, (ok, d) in checks.items()},
        }
        _log.info(f"--- {cam} ---")
        for name, (ok, detail) in checks.items():
            _log.info(f"  {'PASS' if ok else 'FAIL'}  {name:<16}{detail}")
        _log.info(f"  {verdict}")

    (args.out_dir / "overlay_gate.json").write_text(json.dumps({
        "verdict": "FAIL" if failed else "PASS",
        "external_h5": str(args.external_h5), "urdf": str(urdf),
        "q": q.tolist(), "gripper": args.gripper, "perturb_deg": PERTURB_DEG,
        "thresholds": {"min_lift": min_lift, "min_margin": min_margin,
                       "min_coverage": MIN_COVERAGE, "max_coverage": MAX_COVERAGE},
        "cameras": report,
    }, indent=2))
    _log.info(f"images in {args.out_dir}")

    if failed:
        _log.error(
            f"{failed} did not pass. Do not score from them. In order: re-run "
            "extcam_calib for that camera (check its placement spread), then the "
            "flange standoff (build_2f140 --flange-offset, this rig assumes 0), "
            "then whether render_model is really building the 2F-140. A camera "
            "that passes is still usable on its own -- drop the failing one from "
            "score_client --cam."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
