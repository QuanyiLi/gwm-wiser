"""Find `q_capture` empirically: plan -> move -> look -> score -> repeat.

TiPToP plans an entire episode from ONE wrist-camera frame taken at
`q_capture`, so that pose decides what the system can see at all. The upstream
default is MIT's bench and on this rig covers far too little table.

Analysis alone cannot settle it. Fitting the table plane to the wrist depth at
a known pose recovers the camera's height and tilt relative to the flange (on
this rig the camera sits **136 mm behind the TCP** along the approach axis,
its optical axis a few degrees off the gripper's), but a single plane gives no
lateral offset and no roll, so where the footprint actually lands and which
way it is oriented can only be seen by looking. Hence the loop.

Safety, because this commands real motion:

- every candidate is IK-solved and collision-checked against
  `gwm_hardware.common.rig_workspace` before anything moves, and the cuRobo plan is
  checked again as a trajectory;
- the clearance between the arm's outermost sphere and the +y table edge
  (`LEFT_WALL_Y`) is printed for each candidate, and candidates under
  `MIN_WALL_CLEARANCE_M` are dropped;
- `--dry-run` (the default) plans and reports without sending anything.

    # look first
    python -m gwm_hardware.common.find_capture_pose
    # then, with a hand on the e-stop
    python -m gwm_hardware.common.find_capture_pose --execute
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import torch

MIN_WALL_CLEARANCE_M = 0.040
LEFT_WALL_Y = 0.500     # +y table edge
TABLE_TOP_Z = 0.055
# Camera height above the TCP along the approach axis, from a table-plane fit
# at a known pose with the table top at z = +0.055 (the robot base sits 55 mm
# below the working surface).
CAMERA_AXIAL_OFFSET_M = 0.136
# Wrist D435 colour FOV, from its own intrinsics.
FOOTPRINT_W_PER_M = 1.380
FOOTPRINT_H_PER_M = 0.782
OUT = Path.home() / "Desktop/rig_check"


def candidates():
    """TCP targets over the work area, best-first.

    The table is 1 m wide and CENTRED on the base, so the view should centre
    on y = 0.

    `yaw` rotates the gripper about the vertical -- the redundant DOF for a
    straight-down TCP -- and it matters because the wrist camera is **not on
    the TCP axis**: yaw swings where the footprint lands. At yaw 0 the robot's
    own base fills part of the frame; at -90 the base is out of frame entirely
    and the table fills the most of it, so -90 leads, and the remaining freedom
    is where to centre it.

    WHICH IMAGE AXIS IS WHICH WORLD AXIS (from forward kinematics at q_capture
    through the hand-eye extrinsic -- the opposite of the intuitive reading):

        image width  (1.380 per m) -> world  y
        image height (0.782 per m) -> world -x

    At TCP [0.45, 0, 0.55] the camera sits 0.633 m above the table, so:

        y in [-0.40, +0.47]   already WIDER than M2T2's crop of y in [-0.30, +0.30]
        x in [ 0.27,  0.77]

    That settles where to put objects: **spread them along y, not along x**.
    y is capped at +-0.30 by M2T2 regardless of what the camera sees, and x is
    the axis the frame actually runs out of.

    Raising the camera does not help. It buys nothing in y (M2T2's crop binds
    first) and IK runs out before it buys much in x: with
    the 2F-140's TCP 212 mm past the flange and panda_joint4 unable to
    straighten, a straight-down TCP solves only up to z ~ 0.65-0.67, and only
    for tx <= 0.37. That pose sees x in [0.14, 0.73] -- it trades the far end
    of the table for the near end rather than covering more of it.
    """
    for tz in (0.55,):
        for tx in (0.50, 0.45, 0.55):
            for ty in (0.0, -0.10):
                yield tx, ty, tz, np.radians(-90)


def evaluate(rgb, depth, K):
    """How much of the frame is usable table, and how flat is it?"""
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:3, 0:w:3]
    d = depth[::3, ::3]
    m = np.isfinite(d) & (d > 0.1) & (d < 3.0)
    if m.sum() < 1000:
        return dict(table_frac=0.0, dist=float("nan"), tilt=float("nan"))
    z = d[m]
    P = np.stack([(uu[m] - K[0, 2]) * z / K[0, 0],
                  (vv[m] - K[1, 2]) * z / K[1, 1], z], 1)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        inl = np.abs(P @ n + (-n @ a)) < 0.006
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    cnt, inl = best
    Pi = P[inl]
    cen = Pi.mean(0)
    _, _, vt = np.linalg.svd(Pi - cen, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    return dict(table_frac=cnt / len(P), dist=abs(n @ cen),
                tilt=float(np.degrees(np.arccos(np.clip(abs(n[2]), -1, 1)))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true",
                    help="actually move the robot (default: plan and report only)")
    ap.add_argument("--max-poses", type=int, default=3)
    ap.add_argument("--speed", type=float, default=0.10,
                    help="time_dilation_factor for the moves, 1.0 = full speed")
    args = ap.parse_args()

    import aiohttp
    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    from tiptop.config import tiptop_cfg
    from tiptop.motion_planning import build_curobo_solvers, go_to_q
    from tiptop.perception.cameras.rs_camera import RealsenseCamera, rs_infer_depth_async
    from tiptop.utils import get_robot_client
    from tiptop.workspace import workspace_cuboids
    from gwm_hardware.common import robot_2f140

    cfg = tiptop_cfg()
    dev = TensorDeviceType().device
    OUT.mkdir(parents=True, exist_ok=True)

    client = get_robot_client()
    q_now = np.array(client.get_joint_states()["qpos"])
    print(f"robot is at {np.round(q_now, 4).tolist()}")

    _, mg, _ = build_curobo_solvers(num_particles=32, num_spheres=32)
    world = WorldConfig(cuboid=list(workspace_cuboids()))
    ik = robot_2f140.get_ik_solver(world, num_seeds=24)
    names = mg.kinematics.joint_names

    targets = list(candidates())
    # R_z(yaw) . R_x(pi), wxyz -> (0, cos(yaw/2), sin(yaw/2), 0): approach
    # straight down, rotated by yaw about the vertical.
    quats = [[0.0, float(np.cos(t[3] / 2)), float(np.sin(t[3] / 2)), 0.0] for t in targets]
    res = ik.solve_batch(Pose(
        position=torch.tensor([[t[0], t[1], t[2]] for t in targets],
                              dtype=torch.float32, device=dev),
        quaternion=torch.tensor(quats, dtype=torch.float32, device=dev)))
    succ = res.success.view(-1).cpu().numpy()
    sol = res.solution.view(len(targets), -1).cpu().numpy()

    print(f"\ncandidate capture poses (camera {CAMERA_AXIAL_OFFSET_M*1000:.0f} mm "
          f"above the TCP, table at z={TABLE_TOP_Z:+.3f}):")
    viable = []
    for i, (tx, ty, tz, yaw) in enumerate(targets):
        cam_h = tz + CAMERA_AXIAL_OFFSET_M - TABLE_TOP_Z
        fw, fh = FOOTPRINT_W_PER_M * cam_h, FOOTPRINT_H_PER_M * cam_h
        if not succ[i]:
            print(f"  TCP [{tx:.2f} {ty:+.2f} {tz:.2f}] yaw {np.degrees(yaw):+4.0f}  IK failed")
            continue
        q = sol[i]
        js = JointState.from_position(
            torch.tensor(np.array([q]), dtype=torch.float32, device=dev), joint_names=names)
        ok, _ = mg.check_start_state(js)
        sph = mg.kinematics.get_state(js.position).link_spheres_tensor.view(-1, 4).cpu().numpy()
        sph = sph[sph[:, 3] > 0]
        clear = LEFT_WALL_Y - (sph[:, 1] + sph[:, 3]).max()
        good = ok and clear >= MIN_WALL_CLEARANCE_M
        print(f"  TCP [{tx:.2f} {ty:+.2f} {tz:.2f}] yaw {np.degrees(yaw):+4.0f}  "
              f"{'free' if ok else 'COLLIDES':>8s}  wall {clear*1000:5.0f} mm  "
              f"view {fw:.2f} x {fh:.2f} m  {'OK' if good else 'rejected'}")
        if good:
            viable.append((tx, ty, tz, yaw, q))

    if not viable:
        sys.exit("no viable capture pose among the candidates")
    print(f"\n{len(viable)} viable; will try the top {min(args.max_poses, len(viable))}")

    if not args.execute:
        print("\ndry run -- nothing sent to the robot. Re-run with --execute "
              "(hand on the e-stop) to move and measure the real coverage.")
        client.close()
        return

    cam = RealsenseCamera(str(cfg.cameras.hand.serial), enable_depth=True, enable_ir=True)
    try:
        for tx, ty, tz, yaw, q in viable[:args.max_poses]:
            print(f"\n  moving to TCP [{tx:.2f} {ty:+.2f} {tz:.2f}] "
                  f"yaw {np.degrees(yaw):+.0f} deg, speed {args.speed}")
            # tiptop's own mover -- it is what feeds time_dilation_factor into
            # MotionGenPlanConfig; hand-rolled execution here would bypass it
            # and run at full speed.
            go_to_q(q_target=[float(v) for v in q],
                    time_dilation_factor=args.speed, motion_gen=mg)
            time.sleep(1.0)

            async def grab():
                for _ in range(10):
                    fr = cam.read_camera()
                async with aiohttp.ClientSession() as sess:
                    return fr, await rs_infer_depth_async(sess, fr, cam.get_intrinsics())
            fr, depth = asyncio.run(grab())
            st = evaluate(fr.rgb, depth, cam.get_intrinsics().K_color)
            tag = f"cap_x{tx:.2f}_y{ty:+.2f}_z{tz:.2f}_yaw{int(np.degrees(yaw)):+d}"
            import imageio.v3 as iio
            iio.imwrite(OUT / f"{tag}.png", fr.rgb)
            print(f"    table fills {st['table_frac']*100:5.1f} % of the frame, "
                  f"camera {st['dist']*1000:.0f} mm up, tilt {st['tilt']:.1f} deg")
            print(f"    real footprint {FOOTPRINT_W_PER_M*st['dist']:.2f} x "
                  f"{FOOTPRINT_H_PER_M*st['dist']:.2f} m -> {tag}.png")
            print(f"    q = [{', '.join(f'{v:.4f}' for v in q)}]")
    finally:
        cam.close()
        client.close()


if __name__ == "__main__":
    main()
