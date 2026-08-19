"""Rig observation -> the two h5 files the `gwm_tiptop` drivers read.

`gwm_tiptop` was written against droid-sim, where the harness hands the
proposer an h5 and the scorer another one. On hardware nothing produces those
files, so this module does -- from a live capture, or by replaying a
`tiptop_outputs/eval/<timestamp>/` run the baseline arm already recorded.

Two files, because the method looks at the scene through two different cameras
and they are not interchangeable (G-9):

    wrist_obs.h5      PLANNING geometry -- wrist RGB + FoundationStereo depth +
                      K + the camera pose FK puts it at + q_init.
                      Read by `gwm_tiptop.propose_from_h5` and `grasp_gate`.
    external_obs.h5   SCORING viewpoint -- third-person RGB + K + pose, one
                      group per camera. Read by `gwm_tiptop.score_client`.
                      GWM never sees the wrist camera: its training corpus has
                      no wrist views and RAT assumes a fixed one.

The extrinsics correction, and why the hardware files carry a 0
--------------------------------------------------------------
`load_h5_observation` subtracts 15 mm from the camera height by default,
matching what droid-sim's websocket client does to the observation before
tiptop sees it (magic_numbers.md #F). That is a property of THAT client, not of
the pipeline: here `world_from_cam` is FK x hand-eye, already correct, so these
files set `extrinsics_z_correction = 0.0` and say so out loud. Getting this
wrong drops the entire cloud 15 mm -- through the table plane, and straight
into every grasp height.

    # replay what the baseline arm already captured (no robot needed)
    python -m gwm_hardware.gwm_arm.capture replay \
        droid/tiptop/tiptop_outputs/eval/2026-08-18_21-23-50 \
        --out-dir runs/gwm/scene01

    # live, from the rig (moves the arm to q_capture unless --no-move)
    python -m gwm_hardware.gwm_arm.capture live --out-dir runs/gwm/scene01
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from gwm_hardware.common.paths import CONFIG

_log = logging.getLogger("gwm_arm.capture")

# External-camera extrinsics, written by `gwm_hardware.gwm_arm.extcam_calib`.
EXTCAM_CALIB = CONFIG / "extcam_calib.json"

# Names the third-person views carry inside external_obs.h5 and on
# `score_client --cam`. Deliberately the same names droid-sim used, because the
# rig now has the same shape it did: two third-person cameras on opposite sides
# plus the wrist. That makes `--cam external_cam,external_cam_2` -- the sim's
# best-measured configuration (G-30 two-camera fusion) -- the same string here.
#
# `cameras.external` -> external_cam, `cameras.external_2` -> external_cam_2,
# read from tiptop.yml so the serials have exactly one home.
EXTERNAL_CAM = "external_cam"
EXTERNAL_CAM_2 = "external_cam_2"
_CFG_KEY_TO_CAM = {"external": EXTERNAL_CAM, "external_2": EXTERNAL_CAM_2}


def connected_serials() -> set[str]:
    """Serials librealsense can actually see right now."""
    import pyrealsense2 as rs

    return {d.get_info(rs.camera_info.serial_number) for d in rs.context().devices}


def external_camera_specs(connected_only: bool = True) -> list[tuple[str, str]]:
    """[(cam_name, serial)] for the rig's third-person cameras.

    `connected_only` (the default) drops any the config declares but that is not
    plugged in. Cameras get unplugged and moved between sessions -- the config
    records the rig's intent, and a tool that dies with "No device connected"
    because a SECOND camera is absent is a tool that fails for the wrong reason.
    Set it False when you want the declared list regardless.
    """
    from tiptop.config import tiptop_cfg

    cams = tiptop_cfg().cameras
    declared = [(name, str(cams[key].serial)) for key, name in _CFG_KEY_TO_CAM.items()
                if key in cams and cams[key].get("serial")]
    if not declared:
        raise SystemExit("tiptop.yml declares no third-person camera; GWM has nothing to score from")
    if not connected_only:
        return declared

    live = connected_serials()
    out = [(n, s) for n, s in declared if s in live]
    for n, s in declared:
        if s not in live:
            _log.warning(f"{n} ({s}) is declared in tiptop.yml but not connected -- skipping it")
    if not out:
        raise SystemExit(
            "none of the third-person cameras in tiptop.yml are connected: declared "
            f"{[s for _, s in declared]}, present {sorted(live)}. Check the USB cables."
        )
    return out


def mat_to_pos_quat(world_from_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 CV-axis cam2world -> (pos (3,), quat wxyz (4,)), the h5 convention."""
    from scipy.spatial.transform import Rotation

    m = np.asarray(world_from_cam, dtype=np.float64)
    x, y, z, w = Rotation.from_matrix(m[:3, :3]).as_quat()
    return m[:3, 3].copy(), np.array([w, x, y, z])


def write_wrist_h5(
    path: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    world_from_cam: np.ndarray,
    q_init: np.ndarray,
    extrinsics_z_correction: float = 0.0,
) -> Path:
    import h5py

    pos, quat = mat_to_pos_quat(world_from_cam)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["rgb"] = np.asarray(rgb, dtype=np.uint8)
        f["depth"] = np.asarray(depth, dtype=np.float32)
        f["intrinsic_matrix"] = np.asarray(K, dtype=np.float64)
        f["pos_w"] = pos
        f["quat_w_ros"] = quat
        f["q_init"] = np.asarray(q_init, dtype=np.float64)
        f["extrinsics_z_correction"] = float(extrinsics_z_correction)
        f.attrs["source"] = "gwm_hardware.gwm_arm.capture"
    _log.info(f"wrist observation -> {path}  (cam at {np.round(pos, 4).tolist()}, "
              f"z correction {extrinsics_z_correction:+.4f} m)")
    return path


def write_external_h5(path: Path, views: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                      q: np.ndarray | None = None) -> Path:
    """views: {cam_name: (rgb, K, world_from_cam)}.

    `q` is the arm configuration at the moment the frames were taken. The
    scorer does not need it -- but the renderer overlay gate does, because the
    only way to check these extrinsics against reality is to render the robot
    into this exact frame and see whether it lands on the robot in the photo.
    """
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for cam, (rgb, K, c2w) in views.items():
            pos, quat = mat_to_pos_quat(c2w)
            g = f.create_group(cam)
            g["rgb"] = np.asarray(rgb, dtype=np.uint8)
            g["intrinsic_matrix"] = np.asarray(K, dtype=np.float64)
            g["pos_w"] = pos
            g["quat_w_ros"] = quat
        if q is not None:
            f["q_init"] = np.asarray(q, dtype=np.float64)
        f.attrs["source"] = "gwm_hardware.gwm_arm.capture"
    _log.info(f"external observation -> {path}  (views: {sorted(views)}"
              f"{'' if q is not None else ', NO q -- the overlay gate cannot run on it'})")
    return path


def load_extcam_calibration(path: Path = EXTCAM_CALIB) -> dict:
    """{cam_name: 4x4 world_from_cam} produced by extcam_calib."""
    if not path.exists():
        raise SystemExit(
            f"no external-camera extrinsics at {path}.\n"
            "Run `python -m gwm_hardware.gwm_arm.extcam_calib` first -- GWM scores from\n"
            "the third-person view and cannot render the robot into it without a pose."
        )
    data = json.loads(path.read_text())
    return {cam: np.asarray(v["world_from_cam"], dtype=np.float64)
            for cam, v in data["cameras"].items()}


# --------------------------------------------------------------------- replay


def replay_run(run_dir: Path, out_dir: Path) -> Path:
    """A baseline-arm run directory -> wrist_obs.h5.

    `tiptop_run` already saves everything the proposer needs, in a different
    shape: rgb.png, perception/depth.png (uint16 millimetres, so zero means
    "no return" and must become NaN rather than a point at the camera),
    perception/intrinsics.json, and metadata.json's `world_from_cam` and
    `q_at_capture`. Replaying those lets the whole GWM arm be exercised on real
    rig data with the robot switched off.
    """
    import cv2

    meta = json.loads((run_dir / "metadata.json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(run_dir / "rgb.png")), cv2.COLOR_BGR2RGB)
    depth_mm = cv2.imread(str(run_dir / "perception/depth.png"), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise SystemExit(f"{run_dir} has no perception/depth.png")
    depth = depth_mm.astype(np.float32) / 1000.0
    depth[depth_mm == 0] = np.nan
    K = np.asarray(json.loads((run_dir / "perception/intrinsics.json").read_text())["intrinsics"])
    world_from_cam = np.asarray(meta["observation"]["world_from_cam"], dtype=np.float64)
    q_init = np.asarray(meta["observation"]["q_at_capture"], dtype=np.float64)

    out = write_wrist_h5(out_dir / "wrist_obs.h5", rgb, depth, K, world_from_cam, q_init)
    (out_dir / "capture_provenance.json").write_text(json.dumps({
        "mode": "replay",
        "run_dir": str(run_dir),
        "baseline_instruction": meta.get("task_instruction"),
        "timestamp": meta.get("timestamp"),
        "valid_depth_fraction": float(np.isfinite(depth).mean()),
        "gripper_mask_fraction": masked_frac,
    }, indent=2))
    _log.info(f"replayed {run_dir.name}: instruction {meta.get('task_instruction')!r}, "
              f"{np.isfinite(depth).mean():.1%} valid depth")
    _log.warning("this run has NO external view -- the baseline arm does not record one. "
                 "Scoring needs `capture live --external-only` on the same scene.")
    return out


# ------------------------------------------------------------ pipeline selftest


def wrist_as_external(wrist_h5: Path, out: Path, cam: str = EXTERNAL_CAM) -> Path:
    """Rewrite a wrist capture as an external_obs.h5. **MECHANICS TEST ONLY.**

    The scoring half of the pipeline cannot run at all until the external
    camera has extrinsics, and those need the robot. This lets the whole
    chain -- score_client, gwm-server, the renderer seam, the gate, the
    viewer -- be exercised before that, using a camera pose that is genuinely
    correct (the wrist camera's, from FK x hand-eye) rather than an invented
    one. So the plumbing is tested honestly; nothing here is fabricated.

    What it is NOT is a scoring viewpoint. GWM's training corpus contains no
    wrist views and RAT assumes a fixed camera (G-9), and at the capture pose
    the wrist camera is INSIDE the robot it is supposed to be looking at, so
    the robot-only renders come out as a close-up of the gripper. Scores from
    this file mean nothing. The file says so in its own attributes.
    """
    import h5py

    with h5py.File(wrist_h5) as f:
        rgb = np.asarray(f["rgb"])
        K = np.asarray(f["intrinsic_matrix"])
        pos = np.asarray(f["pos_w"])
        quat = np.asarray(f["quat_w_ros"])
        q = np.asarray(f["q_init"]) if "q_init" in f else None

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        g = f.create_group(cam)
        g["rgb"] = rgb.astype(np.uint8)
        g["intrinsic_matrix"] = K.astype(np.float64)
        g["pos_w"] = pos
        g["quat_w_ros"] = quat
        if q is not None:
            f["q_init"] = q
        f.attrs["source"] = "gwm_hardware.gwm_arm.capture wrist-as-external"
        f.attrs["selftest_only"] = ("wrist camera masquerading as the scoring view; "
                                    "pose is real, viewpoint is wrong. Scores are "
                                    "mechanics only, never evidence.")
    _log.warning("wrote a SELFTEST external h5 from the wrist view. The camera pose is "
                 "real, the viewpoint is not a scoring viewpoint -- any score computed "
                 "from this file is a plumbing check, not a result.")
    return out


# ----------------------------------------------------------------------- live


def capture_live(out_dir: Path, move: bool, external_only: bool,
                 apply_gripper_mask: bool = True) -> None:
    """Capture from the rig itself.

    Reuses tiptop's own camera, hand-eye and depth code rather than
    reimplementing it, so the geometry the GWM arm plans on is the same
    geometry the baseline arm plans on -- which is the point of an A/B.
    """
    import aiohttp
    from curobo.types.base import TensorDeviceType

    from tiptop.config import load_calibration, tiptop_cfg
    from tiptop.motion_planning import go_to_capture
    from gwm_tiptop.robot_fk import fk_model
    from tiptop.perception.cameras import get_depth_estimator, get_hand_camera

    from gwm_hardware.common.rs_open import open_with_retry
    from tiptop.perception.cameras.rs_camera import RealsenseCamera
    from tiptop.utils import get_robot_client, load_gripper_mask

    cfg = tiptop_cfg()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = get_robot_client()
    kin = fk_model()            # FK only: where the wrist camera is
    if move and not external_only:
        _log.info("moving to q_capture -- hand on the E-stop")
        # go_to_capture PLANS, so it needs a MotionGen, not the kinematics
        # model. Taking it from the shared cache means the session's existing
        # solver is reused rather than a second one built. It is also
        # workspace-aware, where the previous local build was not -- the
        # capture motion is now collision-checked against the rig's keep-outs
        # like every other motion.
        from gwm_tiptop.robot_fk import default_planning_solvers, reset_world_to_workspace

        _, motion_gen, _ = default_planning_solvers()
        # The proposers overwrite this shared solver's world with the scene
        # they perceived. Reset before moving, or the capture motion plans
        # against LAST turn's objects and without the rig's keep-outs.
        reset_world_to_workspace(motion_gen)
        go_to_capture(time_dilation_factor=cfg.robot.time_dilation_factor,
                      motion_gen=motion_gen)
        # tiptop opens the gripper here because its flow only ever picks. Doing
        # that unconditionally DROPS a held object at the capture pose, before
        # the place it was asked to do -- observed 2026-08-19. This is an
        # invariant rather than a caller flag on purpose: a flag is something a
        # future caller forgets, and the cost of forgetting is on the floor.
        st = client.get_gripper_state().get("state", {})
        if st.get("is_grasped"):
            _log.info(f"gripper is holding something (width {st.get('width', 0) * 1000:.1f} mm) "
                      "-- NOT opening it")
        else:
            client.open_gripper()
    elif not external_only:
        _log.warning("--no-move: capturing wherever the arm currently stands. "
                     "The cloud is still correct (FK gives the true camera pose), "
                     "but the footprint is whatever this pose happens to see.")

    # One joint reading for BOTH captures, taken before either frame, so the
    # external frame and the wrist frame describe the same robot.
    tensor_args = TensorDeviceType()
    q = np.asarray(client.get_joint_positions(), dtype=np.float64)

    # Every third-person camera the config declares, in one shot, so all of
    # them describe the same instant and the same robot pose.
    calib = load_extcam_calibration()
    views, ext_serials = {}, {}
    for name, serial in external_camera_specs():
        if name not in calib:
            _log.warning(f"{name} ({serial}) has no extrinsics in "
                         f"{EXTCAM_CALIB.name} -- skipping it. Scoring will fall back "
                         "to whichever views ARE calibrated.")
            continue
        c = RealsenseCamera(serial)
        try:
            fr = c.read_camera()
            views[name] = (fr.rgb, fr.intrinsics, calib[name])
            ext_serials[name] = c.serial
        finally:
            c.close()
    if not views:
        raise SystemExit("no calibrated third-person camera; run extcam_calib first")
    write_external_h5(out_dir / "external_obs.h5", views, q=q)
    if external_only:
        _log.info("--external-only: skipping the wrist capture (the arm was not moved)")
        return

    # Closed in a finally, and that matters more than it looks: a RealSense
    # admits ONE process at a time, and since the stages moved in-process the
    # session outlives the turn. A handle left open here made the NEXT
    # instruction fail with "Device or resource busy" -- the pick worked, the
    # place could not even capture. Every camera in this module is opened and
    # released within the call that needs it.
    cam = open_with_retry(get_hand_camera, cfg.cameras.hand.serial)
    try:
        ee_from_cam = load_calibration(cam.serial)
        world_from_ee = kin.get_state(tensor_args.to_device(q)).ee_pose.get_numpy_matrix()[0]
        world_from_cam = world_from_ee @ ee_from_cam
        hand_serial = cam.serial

        frame = cam.read_camera()
        estimator = get_depth_estimator(cam)

        async def _depth():
            async with aiohttp.ClientSession() as session:
                return await estimator(session, frame)

        depth = asyncio.run(_depth())
    finally:
        cam.close()

    # The gripper is in the wrist frame and its own surface is not scene
    # geometry. tiptop's live path zeroes the cloud wherever the mask is True
    # (`perception_wrapper.py:91`); doing it here keeps the two A/B arms
    # consuming the SAME geometry, which is the whole point of an A/B.
    #
    # NaN rather than upstream's 0.0: a zeroed point lands at the base origin,
    # below the table, where the above-table cut discards it anyway -- the
    # effect is identical and NaN says what is meant. This rig's mask covers
    # 0.74 % of the frame (the fingers grazing the bottom edge), not the 20.6 %
    # of DROID's shipped 2F-85 + ZED mask, which on this bench would delete a
    # fifth of the tabletop; see docs/tiptop-modifications.md section 6.
    gripper_mask = load_gripper_mask() if apply_gripper_mask else None
    masked_frac = 0.0
    if gripper_mask is not None:
        m = np.asarray(gripper_mask)
        if m.shape != depth.shape:
            _log.warning(f"gripper mask {m.shape} does not match depth {depth.shape} "
                         "-- NOT applying it")
        else:
            masked_frac = float(m.mean())
            depth = depth.astype(np.float32).copy()
            depth[m] = np.nan
            _log.info(f"gripper mask applied: {masked_frac:.2%} of the frame removed")

    write_wrist_h5(out_dir / "wrist_obs.h5", frame.rgb, depth, frame.intrinsics,
                   world_from_cam, q)
    (out_dir / "capture_provenance.json").write_text(json.dumps({
        "mode": "live",
        "moved_to_capture": bool(move),
        "q_init": q.tolist(),
        "hand_serial": hand_serial,
        "external_serials": ext_serials,
        "valid_depth_fraction": float(np.isfinite(depth).mean()),
        "gripper_mask_fraction": masked_frac,
    }, indent=2))
    _log.info(f"live capture done: {np.isfinite(depth).mean():.1%} valid depth")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("replay", help="rebuild a wrist h5 from a saved baseline run")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--out-dir", required=True, type=Path)

    l = sub.add_parser("live", help="capture from the rig")
    l.add_argument("--out-dir", required=True, type=Path)
    l.add_argument("--no-move", dest="move", action="store_false",
                   help="do not drive to q_capture; capture from wherever the arm is")
    l.add_argument("--external-only", action="store_true",
                   help="only grab the third-person frame; never talks to the robot")
    l.add_argument("--no-gripper-mask", dest="gripper_mask", action="store_false",
                   help="keep the pixels the gripper occupies. Needed for PLACING: the "
                        "held object sits exactly where the mask cuts, and place_propose "
                        "separates it from the gripper by the robot's own collision "
                        "spheres, which is a better discriminator but needs the pixels")

    t = sub.add_parser("wrist-as-external",
                       help="MECHANICS TEST ONLY: rewrite a wrist h5 as an external one "
                            "so the scoring chain can run before the extrinsics exist")
    t.add_argument("--wrist-h5", required=True, type=Path)
    t.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    if args.mode == "replay":
        replay_run(args.run_dir, args.out_dir)
    elif args.mode == "wrist-as-external":
        wrist_as_external(args.wrist_h5, args.out)
    else:
        capture_live(args.out_dir, args.move, args.external_only, args.gripper_mask)


if __name__ == "__main__":
    main()
