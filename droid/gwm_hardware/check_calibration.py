"""Is the table tilted, or is the hand-eye calibration tilted?

Reconstructing the tabletop through FK -> ee_from_cam -> depth at q_capture
gives a plane that is flat to 0.84 mm rms but sits 2.56 deg off the base
frame's vertical, 5 mm high. Those two explanations have opposite consequences:

- **the table really is tilted** relative to the robot base -- harmless.
  tiptop does not assume a table height; `find_table_plane` RANSACs it out of
  the observed cloud every capture, so a tilt is simply measured.
- **the calibration is tilted** -- the whole cloud is rotated, every grasp
  inherits the error, and the RANSAC cannot help because it fits a plane to
  already-wrong points. 2.56 deg is 18 mm of height error at 0.4 m out, which
  presents as a gripper that intermittently scuffs the table or closes on air.

They separate cleanly: measure the plane's normal **in the base frame** from
several arm poses. A physical tilt is the same from every pose; a calibration
error rotates with the wrist.

    python -m gwm_hardware.check_calibration            # look only
    python -m gwm_hardware.check_calibration --execute  # move and measure
"""

import argparse
import asyncio

import numpy as np

TABLE_Z = 0.055
SPEED = 0.10
# A pose whose plane fit rejects many points is not seeing a clean tabletop --
# usually the camera's 75 mm lateral offset has swung part of the frame past
# the table edge onto the floor. Including such poses is what made the
# six-pose solve diverge to a nonsense 98 deg on 2026-08-18; they are dropped
# rather than averaged in.
MIN_INLIER_FRAC = 0.95


def fit_plane_base(Pw, iters=6, tol=0.010):
    keep = np.ones(len(Pw), bool)
    for _ in range(iters):
        cen = Pw[keep].mean(0)
        _, _, vt = np.linalg.svd(Pw[keep] - cen, full_matrices=False)
        n = vt[-1]
        n = n if n[2] > 0 else -n
        keep = np.abs((Pw - cen) @ n) < tol
    r = (Pw[keep] - cen) @ n
    return n, cen, float(r.std()), int(keep.sum()), int(len(Pw))


def main() -> None:
    import aiohttp
    import torch
    from scipy.spatial.transform import Rotation as R

    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.geom.types import WorldConfig

    from tiptop.config import load_calibration, tiptop_cfg
    from tiptop.motion_planning import build_curobo_solvers, go_to_q
    from tiptop.perception.cameras.rs_camera import RealsenseCamera, rs_infer_depth_async
    from tiptop.utils import get_robot_client, load_gripper_mask
    from tiptop.workspace import workspace_cuboids
    from gwm_hardware import robot_2f140

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--fix", action="store_true",
                    help="if the error is a constant rotation, solve for the "
                         "correction and write a corrected calibration")
    ap.add_argument("--speed", type=float, default=SPEED)
    args = ap.parse_args()

    cfg = tiptop_cfg()
    dev = TensorDeviceType().device
    E = load_calibration(str(cfg.cameras.hand.serial))
    _, mg, _ = build_curobo_solvers(num_particles=32, num_spheres=32)
    names = mg.kinematics.joint_names

    # Poses that view the same table from noticeably different wrist
    # orientations -- that difference is what separates the two hypotheses.
    # Yaw diversity is what conditions the solve: a fixed rotation error in
    # ee_from_cam is constant in the EE frame and rotates in the base frame,
    # while a physically tilted table does exactly the reverse. Poses that
    # differ only in position cannot tell them apart.
    targets = [(0.45, 0.00, 0.55, -90), (0.45, 0.00, 0.55, 0),
               (0.45, 0.00, 0.55, 90), (0.45, 0.00, 0.55, 180),
               (0.50, -0.12, 0.52, -45), (0.50, -0.12, 0.52, 135)]
    ik = robot_2f140.get_ik_solver(WorldConfig(cuboid=list(workspace_cuboids())), num_seeds=24)
    quats = [[0.0, float(np.cos(np.radians(y) / 2)), float(np.sin(np.radians(y) / 2)), 0.0]
             for *_, y in targets]
    res = ik.solve_batch(Pose(
        position=torch.tensor([[t[0], t[1], t[2]] for t in targets],
                              dtype=torch.float32, device=dev),
        quaternion=torch.tensor(quats, dtype=torch.float32, device=dev)))
    succ = res.success.view(-1).cpu().numpy()
    sol = res.solution.view(len(targets), -1).cpu().numpy()

    poses = []
    for i, (tx, ty, tz, yaw) in enumerate(targets):
        if not succ[i]:
            print(f"  TCP [{tx:.2f} {ty:+.2f} {tz:.2f}] yaw {yaw:+4d}: IK failed")
            continue
        js = JointState.from_position(
            torch.tensor(np.array([sol[i]]), dtype=torch.float32, device=dev),
            joint_names=names)
        ok, _ = mg.check_start_state(js)
        print(f"  TCP [{tx:.2f} {ty:+.2f} {tz:.2f}] yaw {yaw:+4d}: "
              f"{'collision-free' if ok else 'COLLIDES, skipping'}")
        if ok:
            poses.append((tx, ty, tz, yaw, sol[i]))

    if not args.execute:
        print(f"\ndry run: {len(poses)} poses would be visited. "
              "Re-run with --execute (hand on the e-stop).")
        return

    client = get_robot_client()
    cam = RealsenseCamera(str(cfg.cameras.hand.serial), enable_depth=True, enable_ir=True)
    gm = load_gripper_mask()
    normals = []
    ee_normals = []
    base_R = []
    try:
        for tx, ty, tz, yaw, q in poses:
            print(f"\n  moving to TCP [{tx:.2f} {ty:+.2f} {tz:.2f}] yaw {yaw:+d}")
            go_to_q(q_target=[float(v) for v in q],
                    time_dilation_factor=args.speed, motion_gen=mg)
            import time
            time.sleep(1.0)

            qn = np.array(client.get_joint_states()["qpos"])
            js = JointState.from_position(
                torch.tensor(np.array([qn]), dtype=torch.float32, device=dev),
                joint_names=names)
            st = mg.kinematics.get_state(js.position)
            qt = st.ee_quaternion[0].cpu().numpy()
            B = np.eye(4)
            B[:3, :3] = R.from_quat([qt[1], qt[2], qt[3], qt[0]]).as_matrix()
            B[:3, 3] = st.ee_position[0].cpu().numpy()
            Wm = B @ E

            async def grab():
                for _ in range(10):
                    fr = cam.read_camera()
                async with aiohttp.ClientSession() as s:
                    return fr, await rs_infer_depth_async(s, fr, cam.get_intrinsics())
            fr, d = asyncio.run(grab())
            K = cam.get_intrinsics().K_color
            hh, ww = d.shape
            vv, uu = np.mgrid[0:hh:2, 0:ww:2]
            dd = d[::2, ::2]
            m = np.isfinite(dd) & (dd > 0.3) & (dd < 1.2) & (~gm[::2, ::2])
            z = dd[m]
            P = np.stack([(uu[m] - K[0, 2]) * z / K[0, 0],
                          (vv[m] - K[1, 2]) * z / K[1, 1], z], 1)
            Pw = (Wm[:3, :3] @ P.T).T + Wm[:3, 3]
            n, cen, rms, ninl, ntot = fit_plane_base(Pw)
            tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
            frac = ninl / max(ntot, 1)
            if frac < MIN_INLIER_FRAC:
                print(f"    DROPPED: only {frac*100:.0f} % of points fit one plane "
                      f"-- the view is not all tabletop")
                continue
            normals.append((yaw, n, cen[2], tilt, rms))
            ee_normals.append(np.linalg.inv(B[:3, :3]) @ n)
            base_R.append(B[:3, :3].copy())
            print(f"    plane normal {np.round(n,4).tolist()}  tilt {tilt:.2f} deg  "
                  f"height {cen[2]*1000:+.2f} mm  rms {rms*1000:.2f} mm  "
                  f"({ninl}/{ntot} inliers, {ninl/max(ntot,1)*100:.0f} %)")
    finally:
        cam.close()
        client.close()

    # Joint solve: a fixed rotation error `R_corr` in ee_from_cam AND a real
    # table normal `n_table` in the base frame both produce a tilt, but they
    # are separable -- the first is constant in the EE frame, the second in the
    # base frame. Fit both and see how much of the 2.95 deg each explains.
    if len(ee_normals) >= 3:
        from scipy.optimize import least_squares

        Rb = np.array(base_R)                       # base_from_ee per pose
        obs = np.array([nb for _, nb, _, _, _ in normals])   # base-frame normals

        def unpack(x):
            rc = R.from_rotvec(x[:3])
            nt = np.array([x[3], x[4], np.sqrt(max(1e-9, 1 - x[3]**2 - x[4]**2))])
            return rc, nt

        def resid(x):
            rc, nt = unpack(x)
            out = []
            for Rbi, o in zip(Rb, obs):
                # what we WOULD observe: the true table normal, seen through
                # the erroneous extrinsic
                pred = Rbi @ rc.as_matrix() @ Rbi.T @ nt
                pred /= np.linalg.norm(pred)
                out.extend(pred - o)
            return out

        sol = least_squares(resid, np.zeros(5), xtol=1e-12, ftol=1e-12)
        rc, nt = unpack(sol.x)
        ang = np.degrees(np.linalg.norm(sol.x[:3]))
        tilt_tab = np.degrees(np.arccos(np.clip(nt[2], -1, 1)))
        rms = np.degrees(np.sqrt(np.mean(np.square(sol.fun))))
        print(f"\n  joint solve over {len(obs)} poses:")
        print(f"    extrinsic rotation error : {ang:.2f} deg")
        print(f"    physical table tilt      : {tilt_tab:.2f} deg "
              f"(normal {np.round(nt,4).tolist()})")
        print(f"    fit residual             : {rms:.2f} deg")
        if args.fix and ang > 0.3:
            E_new = E.copy()
            E_new[:3, :3] = rc.as_matrix() @ E[:3, :3]
            import json
            from tiptop.config import calib_info_path
            d = json.load(open(calib_info_path))
            key = str(cfg.cameras.hand.serial)
            d.setdefault(key + "_uncorrected", dict(d[key]))
            d[key]["pose"] = [*E_new[:3, 3].tolist(),
                              *R.from_matrix(E_new[:3, :3]).as_euler("xyz").tolist()]
            json.dump(d, open(calib_info_path, "w"), indent=2)
            print(f"    applied the {ang:.2f} deg correction to ee_from_cam; "
                  f"the original is kept as '{key}_uncorrected'")
            print(f"    re-run without --fix to confirm the tilt dropped")
        elif not args.fix:
            print("    pass --fix to apply the extrinsic correction")

    # Express each observed table normal back in the EE frame. A CONSTANT
    # rotational error in ee_from_cam shows up as a base-frame tilt whose
    # direction rotates with the wrist but whose EE-frame direction is the
    # same every time. A genuinely tilted table does the opposite. This is the
    # test that says whether the error can simply be solved for.
    if ee_normals:
        M = np.array(ee_normals)
        sp = max(np.degrees(np.arccos(np.clip(abs(M[i] @ M[j]), -1, 1)))
                 for i in range(len(M)) for j in range(i + 1, len(M)))
        mean_ee = M.mean(0); mean_ee /= np.linalg.norm(mean_ee)
        print(f"\n  the same normals expressed in the EE frame disagree by {sp:.2f} deg")
        print(f"  mean EE-frame table normal {np.round(mean_ee,4).tolist()}")
        if sp < 1.0:
            ang = np.degrees(np.arccos(np.clip(mean_ee[2], -1, 1)))
            print(f"  => CONSISTENT in the EE frame: a single fixed rotation of "
                  f"{ang:.2f} deg in ee_from_cam explains all of it.")
            if args.fix:
                axis = np.cross(mean_ee, np.array([0.0, 0.0, 1.0]))
                axis /= np.linalg.norm(axis)
                R_corr = R.from_rotvec(axis * np.radians(ang))
                E_new = E.copy()
                E_new[:3, :3] = R_corr.as_matrix() @ E[:3, :3]
                import json
                from tiptop.config import calib_info_path
                d = json.load(open(calib_info_path))
                key = str(cfg.cameras.hand.serial)
                d[key + "_uncorrected"] = dict(d[key])
                rpy = R.from_matrix(E_new[:3, :3]).as_euler("xyz")
                d[key]["pose"] = [*E_new[:3, 3].tolist(), *rpy.tolist()]
                json.dump(d, open(calib_info_path, "w"), indent=2)
                print(f"  applied a {ang:.2f} deg correction about "
                      f"{np.round(axis,3).tolist()} and rewrote the calibration")
                print(f"  (the uncorrected pose is kept as '{key}_uncorrected')")
            else:
                print("  pass --fix to apply it")
        else:
            print("  => NOT a single fixed rotation; something else is going on")

    if len(normals) < 2:
        print("\nneed at least two poses to separate the hypotheses")
        return
    print("\nverdict:")
    N = np.array([n for _, n, _, _, _ in normals])
    spread = max(np.degrees(np.arccos(np.clip(abs(N[i] @ N[j]), -1, 1)))
                 for i in range(len(N)) for j in range(i + 1, len(N)))
    mean_tilt = float(np.mean([t for *_, t, _ in normals]))
    hs = [h for _, _, h, _, _ in normals]
    print(f"  normals disagree between poses by at most {spread:.2f} deg")
    print(f"  mean tilt off base vertical            {mean_tilt:.2f} deg")
    print(f"  reconstructed table height             "
          f"{np.mean(hs)*1000:+.2f} mm  (spread {(max(hs)-min(hs))*1000:.2f} mm)")
    if spread < 1.0:
        print(f"\n  => the normal is POSE-INVARIANT: the table really is tilted "
              f"{mean_tilt:.2f} deg relative to the robot base.")
        print("     Harmless for tiptop, which RANSACs the table out of each capture.")
    else:
        print(f"\n  => the normal MOVES WITH THE WRIST: this is hand-eye error, "
              "not a tilted table.")
        print("     Re-run calibrate-wrist-cam with the board presented at several "
              "orientations, not flat on the table.")


if __name__ == "__main__":
    main()
