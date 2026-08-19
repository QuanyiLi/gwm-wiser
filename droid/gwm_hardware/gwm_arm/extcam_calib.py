"""External-camera extrinsics in the robot base frame.

GWM scores from the third-person camera: it renders the robot into that view
and compares the render against the photo, so it needs `world_from_cam` for a
camera tiptop never calibrates. tiptop calibrates the WRIST camera only
(`calibrate-wrist-cam`), and droid-sim simply reports both camera poses in the
observation. On hardware this is the missing link, and it is the one piece of
the GWM arm that has no counterpart anywhere else in the stack.

Method -- no new hardware, and no second calibration standard. The wrist camera
is already hand-eye calibrated, so it can carry the base frame to a Charuco
board sitting on the table, and the external camera reads the same board:

    T_base_from_board  = FK(q) @ ee_from_cam @ T_wristcam_from_board
    T_base_from_extcam = T_base_from_board @ inv(T_extcam_from_board)

Repeat with the board in several places and average; the spread across
placements is the error bar, and it is the number to trust rather than any
single solve. The board, dictionary and checker size come from
`tiptop.scripts.calibrate_wrist_cam` -- the copy this rig's
`install_charuco_params` already corrected (11x8, DICT_5X5_100, 34.31 mm) --
so there is exactly one board definition on the rig.

Split into three commands on purpose: `shoot` is the only one that touches the
robot, and it only READS joint angles (never commands motion), so the whole
solve can be re-run and re-argued offline afterwards.

    # 1. park the arm at the capture pose, put the board on the table.
    #    Then, moving the board between shots:
    python -m gwm_hardware.gwm_arm.extcam_calib shoot --shots-dir runs/extcal --n 6

    # 2. solve, offline, as often as you like
    python -m gwm_hardware.gwm_arm.extcam_calib solve --shots-dir runs/extcal

    # 3. what it looks like
    python -m gwm_hardware.gwm_arm.extcam_calib check --shots-dir runs/extcal
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from gwm_hardware.common.paths import CONFIG
from gwm_hardware.gwm_arm.capture import EXTCAM_CALIB, EXTERNAL_CAM, external_camera_specs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_arm.extcam_calib")

MIN_CORNERS = 20          # of the board's 70 interior corners

# Acceptance is on the POOLED reprojection residual, in pixels, because pixels
# are what the overlay gate and the scorer care about. A per-placement pose
# spread in millimetres was the first gate here and it was the wrong quantity:
# it is dominated by the wrist camera's ~3 deg hand-eye residual, which no
# amount of re-shooting this camera can reduce. It is still reported, as a
# diagnostic of the WRIST calibration.
MAX_REPROJ_PX = 2.0


def _board():
    """The rig's Charuco board, from the single definition tiptop carries."""
    from tiptop.scripts.calibrate_wrist_cam import (
        CHARUCO_BOARD,
        charuco_params,
        detector_params,
    )

    return CHARUCO_BOARD, charuco_params, detector_params


def board_detect(rgb: np.ndarray, K: np.ndarray, dist: np.ndarray):
    """RGB -> (corners (N,2) px, ids (N,), T_cam_from_board, n, reproj rms px) or None.

    Same detection `board_pose` reports, but keeping the raw corner
    correspondences, which is what the pooled solve in `solve()` fits.
    """
    import cv2
    from cv2 import aruco

    board, charuco_params, detector_params = _board()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = aruco.CharucoDetector(board, charuco_params, detector_params)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or len(charuco_corners) < MIN_CORNERS:
        return None

    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2).sum(1).mean()))

    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return (img_pts.reshape(-1, 2), charuco_ids.ravel(), T, int(len(charuco_corners)), rms)


def board_corners_3d() -> np.ndarray:
    """(70, 3) interior corner positions in the board frame."""
    return np.asarray(_board()[0].getChessboardCorners(), dtype=np.float64)


def board_pose(rgb: np.ndarray, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, int, float] | None:
    """RGB + intrinsics -> (T_cam_from_board 4x4, n_corners, reprojection rms px).

    Returns None when the board is not seen well enough to be worth using.
    """
    d = board_detect(rgb, K, dist)
    return None if d is None else (d[2], d[3], d[4])


def average_pose(mats: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    """Mean of several 4x4 poses -> (mean, translation spread m, rotation spread deg).

    Rotation is averaged by the SVD projection of the summed matrices (the
    chordal L2 mean on SO(3)), which needs no reference pose and no quaternion
    sign bookkeeping.
    """
    from scipy.spatial.transform import Rotation

    R_sum = np.sum([m[:3, :3] for m in mats], axis=0)
    u, _, vt = np.linalg.svd(R_sum)
    R_mean = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    t = np.array([m[:3, 3] for m in mats])
    mean = np.eye(4)
    mean[:3, :3] = R_mean
    mean[:3, 3] = t.mean(axis=0)

    t_spread = float(np.linalg.norm(t - t.mean(axis=0), axis=1).max())
    r_spread = float(max(
        Rotation.from_matrix(R_mean.T @ m[:3, :3]).magnitude() for m in mats
    ) * 180.0 / np.pi) if len(mats) > 1 else 0.0
    return mean, t_spread, r_spread


# ---------------------------------------------------------------------- shoot


def shoot(shots_dir: Path, n: int) -> None:
    """Grab n (wrist rgb, every external rgb, q) samples. READ-ONLY on the robot.

    All cameras are read at the same shot, so one board placement calibrates
    every third-person view at once and they all inherit the same wrist solve.
    A shot the board is missing from in ONE view still calibrates the others --
    the per-camera solves are independent.
    """
    import cv2
    from curobo.types.base import TensorDeviceType

    from tiptop.config import load_calibration, tiptop_cfg
    from tiptop.perception.cameras import get_hand_camera

    from gwm_hardware.common.rs_open import open_with_retry
    from gwm_tiptop.robot_fk import fk_model
    from tiptop.perception.cameras.rs_camera import RealsenseCamera
    from tiptop.utils import get_robot_client

    shots_dir.mkdir(parents=True, exist_ok=True)
    client = get_robot_client()
    hand = open_with_retry(get_hand_camera, tiptop_cfg().cameras.hand.serial)
    ee_from_cam = load_calibration(hand.serial)
    kin = fk_model()            # FK only: it never commands motion
    tensor_args = TensorDeviceType()

    specs = external_camera_specs()
    externals = {name: RealsenseCamera(serial) for name, serial in specs}
    hi = hand.get_intrinsics()
    ei = {name: c.get_intrinsics() for name, c in externals.items()}
    print(f"\nCalibrating: {', '.join(n for n, _ in specs)}")
    print("Park the arm where the wrist camera sees the board (the capture pose is\n"
          "the natural choice). This command never moves the robot.\n")
    try:
        for i in range(n):
            input(f"  shot {i + 1}/{n}: place the board where the wrist camera AND as many "
                  "third-person cameras as possible see it, then press Enter ")
            q = np.asarray(client.get_joint_positions(), dtype=np.float64)
            world_from_ee = kin.get_state(
                tensor_args.to_device(q)).ee_pose.get_numpy_matrix()[0]
            hf = hand.read_camera()
            ef = {name: c.read_camera() for name, c in externals.items()}

            d = shots_dir / f"shot_{i:02d}"
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / "wrist.png"), cv2.cvtColor(hf.rgb, cv2.COLOR_RGB2BGR))
            for name, fr in ef.items():
                cv2.imwrite(str(d / f"{name}.png"), cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR))
            (d / "shot.json").write_text(json.dumps({
                "q": q.tolist(),
                "world_from_ee": world_from_ee.tolist(),
                "ee_from_cam": np.asarray(ee_from_cam).tolist(),
                "wrist": {"serial": hand.serial, "K": np.asarray(hi.K_color).tolist(),
                          "dist": np.asarray(hi.distortion_color).tolist()},
                "externals": {name: {"serial": externals[name].serial,
                                     "K": np.asarray(ei[name].K_color).tolist(),
                                     "dist": np.asarray(ei[name].distortion_color).tolist()}
                              for name in ef},
            }, indent=2))

            wp = board_pose(hf.rgb, np.asarray(hi.K_color), np.asarray(hi.distortion_color))
            print(f"    wrist        : {'%d corners, %.2f px reproj' % wp[1:] if wp else 'BOARD NOT SEEN'}")
            seen = 0
            for name, fr in ef.items():
                ep = board_pose(fr.rgb, np.asarray(ei[name].K_color),
                                np.asarray(ei[name].distortion_color))
                seen += bool(ep)
                print(f"    {name:<13}: {'%d corners, %.2f px reproj' % ep[1:] if ep else 'BOARD NOT SEEN'}")
            if not wp:
                print("    -> unusable for every camera (the WRIST must see the board); re-take it")
            elif seen == 0:
                print("    -> no third-person camera saw it; move the board and re-take")
    finally:
        # Same rule as gwm_arm.capture: a RealSense admits one process at a
        # time, so every camera opened here is released here.
        hand.close()
        for c in externals.values():
            c.close()
    print(f"\n{n} shots in {shots_dir}. Now: extcam_calib solve --shots-dir {shots_dir}")


# ---------------------------------------------------------------------- solve


def solve(shots_dir: Path, out: Path, install: bool) -> None:
    """Fit ONE camera pose per camera to every board corner of every placement.

    Not an average of per-placement poses. Averaging discards the corner
    correspondences and inherits, in full, the error in `base_from_board` --
    which on this rig is dominated by the wrist camera's ~3 deg hand-eye
    rotational residual (docs/tiptop-modifications.md). That residual moves the
    board's estimated base-frame position by up to 35 mm at a 0.66 m standoff,
    differently for each placement, so a spread of per-placement poses measures
    the WRIST calibration far more than it measures this camera.

    Pooling instead fits the one unknown -- where this camera is -- to all the
    2D corners at once, and reports the residual in PIXELS. That is the
    quantity that decides whether a rendered robot lands on the real one, it is
    directly comparable to the overlay gate, and it does not pretend the
    hand-eye residual is not there: the residual absorbs it and shows its size.

    The per-placement spread is still computed and reported, as the diagnostic
    it actually is.
    """
    import cv2

    shots = sorted(p for p in shots_dir.glob("shot_*") if (p / "shot.json").exists())
    if not shots:
        raise SystemExit(f"no shots in {shots_dir}")
    corners_board = board_corners_3d()

    pooled: dict[str, dict] = {}
    per_pose: dict[str, list[np.ndarray]] = {}
    rows: list[dict] = []
    for d in shots:
        meta = json.loads((d / "shot.json").read_text())
        wrist = cv2.cvtColor(cv2.imread(str(d / "wrist.png")), cv2.COLOR_BGR2RGB)
        wp = board_pose(wrist, np.asarray(meta["wrist"]["K"]), np.asarray(meta["wrist"]["dist"]))
        if not wp:
            _log.warning(f"{d.name}: the WRIST does not see the board -- the whole shot "
                         "is unusable (it is what carries the base frame)")
            continue
        T_wc_b, nw, rw = wp
        base_from_wristcam = (np.asarray(meta["world_from_ee"], dtype=np.float64)
                              @ np.asarray(meta["ee_from_cam"], dtype=np.float64))
        base_from_board = base_from_wristcam @ T_wc_b
        _log.info(f"{d.name}: board at {np.round(base_from_board[:3, 3], 3).tolist()} "
                  f"(wrist {nw} corners, {rw:.2f} px)")

        externals = meta.get("externals") or {EXTERNAL_CAM: meta["external"]}
        for name, spec in externals.items():
            img_path = d / f"{name}.png"
            if not img_path.exists():
                img_path = d / "external.png"
            if not img_path.exists():
                continue
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            det = board_detect(img, np.asarray(spec["K"]), np.asarray(spec["dist"]))
            if det is None:
                _log.info(f"  {name}: board not seen -- skipped for this placement")
                continue
            img_pts, ids, T_ec_b, ne, re = det
            pts_base = (corners_board[ids] @ base_from_board[:3, :3].T
                        + base_from_board[:3, 3])
            e = pooled.setdefault(name, {"obj": [], "img": [], "K": np.asarray(spec["K"]),
                                         "dist": np.asarray(spec["dist"]), "shots": []})
            e["obj"].append(pts_base)
            e["img"].append(img_pts)
            e["shots"].append(d.name)
            per_pose.setdefault(name, []).append(base_from_board @ np.linalg.inv(T_ec_b))
            rows.append({"shot": d.name, "cam": name, "wrist_corners": nw,
                         "wrist_rms_px": round(rw, 3), "external_corners": ne,
                         "external_rms_px": round(re, 3),
                         "board_in_base": np.round(base_from_board[:3, 3], 4).tolist()})
            _log.info(f"  {name}: {ne} corners, single-view reproj {re:.2f} px")

    if not pooled:
        raise SystemExit("no camera saw the board in any usable shot")

    cameras, quality, verdicts = {}, {}, []
    for name, e in sorted(pooled.items()):
        obj = np.concatenate(e["obj"]).astype(np.float64)
        img = np.concatenate(e["img"]).astype(np.float64)
        if len(e["shots"]) < 2:
            _log.error(f"{name}: only {len(e['shots'])} usable placement -- one placement "
                       "cannot separate a camera-pose error from a board-pose error")
            quality[name] = {"n_placements": len(e["shots"]), "verdict": "FAIL",
                             "reason": "fewer than 2 usable placements"}
            verdicts.append("FAIL")
            continue

        ok, rvec, tvec = cv2.solvePnP(obj[:, None, :], img[:, None, :], e["K"], e["dist"],
                                      flags=cv2.SOLVEPNP_SQPNP)
        rvec, tvec = cv2.solvePnPRefineLM(obj[:, None, :], img[:, None, :], e["K"], e["dist"],
                                          rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, e["K"], e["dist"]).__getitem__(0), None
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        rms, p95 = float(np.sqrt((err ** 2).mean())), float(np.percentile(err, 95))

        cam_from_base = np.eye(4)
        cam_from_base[:3, :3] = cv2.Rodrigues(rvec)[0]
        cam_from_base[:3, 3] = tvec.ravel()
        world_from_cam = np.linalg.inv(cam_from_base)

        _, t_spread, r_spread = average_pose(per_pose[name])
        ok_gate = rms <= MAX_REPROJ_PX
        verdicts.append("PASS" if ok_gate else "FAIL")
        _log.info(
            f"{name}: pooled over {len(e['shots'])} placements / {len(obj)} corners -> "
            f"reprojection rms {rms:.2f} px (p95 {p95:.2f}); camera at "
            f"{np.round(world_from_cam[:3, 3], 3).tolist()}")
        _log.info(
            f"  diagnostic: per-placement pose spread {t_spread * 1000:.1f} mm / "
            f"{r_spread:.2f} deg -- this tracks the WRIST hand-eye residual, not this camera")
        if not ok_gate:
            _log.error(f"  {rms:.2f} px > {MAX_REPROJ_PX} px: the corners do not fit ONE "
                       "camera pose. Something moved between shots (the camera, or the "
                       "board within a shot), or a placement is badly conditioned.")

        cameras[name] = {
            "world_from_cam": world_from_cam.tolist(),
            "position": world_from_cam[:3, 3].tolist(),
            "convention": "CV-axis cam2world (x right, y down, z forward), robot base frame",
        }
        quality[name] = {
            "n_placements": len(e["shots"]), "n_corners": int(len(obj)),
            "reproj_rms_px": rms, "reproj_p95_px": p95,
            "pose_spread_m": t_spread, "pose_spread_deg": r_spread,
            "verdict": "PASS" if ok_gate else "FAIL", "placements": e["shots"],
        }

    verdict = "PASS" if verdicts and all(v == "PASS" for v in verdicts) else "FAIL"
    payload = {"cameras": cameras, "quality": quality, "verdict": verdict,
               "max_reproj_px": MAX_REPROJ_PX, "shots_dir": str(shots_dir),
               "per_shot": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    _log.info(f"{verdict}: wrote {out} ({len(cameras)} camera(s))")

    if not install:
        return
    good = {k: v for k, v in cameras.items() if quality[k]["verdict"] == "PASS"}
    if not good:
        _log.error("no camera passed; not installing")
        return
    EXTCAM_CALIB.write_text(json.dumps({**payload, "cameras": good}, indent=2))
    _log.info(f"installed {sorted(good)} as the rig's extrinsics: {EXTCAM_CALIB}")


# ---------------------------------------------------------------------- check


def check(shots_dir: Path, calib: Path) -> None:
    """Reproject the board's base-frame position through the solved extrinsics."""
    import cv2

    cams = json.loads(calib.read_text())["cameras"]
    errs: dict[str, list[float]] = {}
    for d in sorted(p for p in shots_dir.glob("shot_*") if (p / "shot.json").exists()):
        meta = json.loads((d / "shot.json").read_text())
        wrist = cv2.cvtColor(cv2.imread(str(d / "wrist.png")), cv2.COLOR_BGR2RGB)
        wp = board_pose(wrist, np.asarray(meta["wrist"]["K"]), np.asarray(meta["wrist"]["dist"]))
        if not wp:
            continue
        base_from_board = (np.asarray(meta["world_from_ee"]) @ np.asarray(meta["ee_from_cam"])
                           @ wp[0])
        externals = meta.get("externals") or {EXTERNAL_CAM: meta["external"]}
        for name, spec in externals.items():
            if name not in cams:
                continue
            img_path = d / f"{name}.png"
            if not img_path.exists():
                img_path = d / "external.png"
            if not img_path.exists():
                continue
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            ep = board_pose(img, np.asarray(spec["K"]), np.asarray(spec["dist"]))
            if not ep:
                continue
            w2c = np.linalg.inv(np.asarray(cams[name]["world_from_cam"], dtype=np.float64))
            pred = (w2c @ np.append(base_from_board[:3, 3], 1.0))[:3]
            e = float(np.linalg.norm(pred - ep[0][:3, 3]))
            errs.setdefault(name, []).append(e)
            print(f"  {d.name}  {name:<15} board origin {e * 1000:6.1f} mm off")
    for name, e in sorted(errs.items()):
        print(f"\n{name}: mean {np.mean(e) * 1000:.1f} mm, worst {np.max(e) * 1000:.1f} mm "
              f"over {len(e)} placements")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("shoot", help="grab wrist+external+q samples (reads the robot, never moves it)")
    s.add_argument("--shots-dir", required=True, type=Path)
    s.add_argument("--n", type=int, default=6)

    v = sub.add_parser("solve", help="solve the extrinsics from saved shots")
    v.add_argument("--shots-dir", required=True, type=Path)
    v.add_argument("--out", type=Path, default=None)
    v.add_argument("--no-install", dest="install", action="store_false",
                   help="write the result next to the shots but do not make it the rig's")

    c = sub.add_parser("check", help="reproject the board through the solved extrinsics")
    c.add_argument("--shots-dir", required=True, type=Path)
    c.add_argument("--calib", type=Path, default=EXTCAM_CALIB)

    args = ap.parse_args()
    if args.cmd == "shoot":
        shoot(args.shots_dir, args.n)
    elif args.cmd == "solve":
        solve(args.shots_dir, args.out or args.shots_dir / "extcam_calib.json",
              args.install)
    else:
        check(args.shots_dir, args.calib)


if __name__ == "__main__":
    main()
