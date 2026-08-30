"""RealSense pre-flight: put the IR stereo pair into a state FoundationStereo
can actually use, and report the rig's camera inventory.

Why this exists: the rig's D435s can come up with IR **auto-exposure OFF and
exposure pinned at 40000 us** (the maximum), which saturates most of the IR
pair at 255. tiptop does not use the RealSense ASIC's own depth --
``rs_camera.get_depth_estimator`` sends the **IR pair** to FoundationStereo
(``rs_infer_depth_async``) -- and a saturated pair has no projector dot pattern
left, which on a white tabletop is the only texture there is. Every downstream
stage (world point cloud, RANSAC table plane, DBSCAN clusters, M2T2 grasps,
cuTAMP collision meshes) is built on that depth. With auto-exposure back on,
saturation drops to a few percent and most of the frame gets valid depth.

``tiptop.perception.cameras.rs_camera.RealsenseCamera`` never touches exposure
-- it only enables streams -- so the device's persisted state carries straight
into a run, silently. ``tiptop/`` is kept pristine, so the fix lives here as a
pre-flight step both arms run, rather than as a patch to the camera class.

Run before every session, from the gwm-wiser repo root:

    pixi run --manifest-path droid/tiptop/pixi.toml \
        python -m gwm_hardware.common.rs_preflight
    pixi run --manifest-path droid/tiptop/pixi.toml \
        python -m gwm_hardware.common.rs_preflight --check   # report only, change nothing
"""

import argparse
import sys
import time

import numpy as np

SATURATION_LEVEL = 250      # uint8 IR value treated as saturated
MAX_SATURATED_FRAC = 0.10   # fail the check above this fraction of the pair
MIN_DEPTH_VALID_FRAC = 0.50 # fail below this fraction of finite in-range depth
DEPTH_MAX_M = 10.0          # beyond this is the uint16 no-return marker


def _ir_stats(frame) -> dict:
    out = {}
    for side in ("left", "right"):
        ir = getattr(frame, f"ir_{side}")
        out[side] = (float(ir.mean()),
                     float((ir >= SATURATION_LEVEL).mean()))
    return out


def _depth_valid_frac(frame) -> float:
    d = frame.depth
    if d is None:
        return float("nan")
    return float(((d > 0) & (d < DEPTH_MAX_M)).mean())


def preflight(serial: str, label: str, apply: bool = True) -> bool:
    import pyrealsense2 as rs
    from tiptop.perception.cameras.rs_camera import RealsenseCamera

    cam = RealsenseCamera(serial, enable_depth=True, enable_ir=True)
    try:
        sensor = cam._profile.get_device().first_depth_sensor()
        device = cam._profile.get_device()

        for _ in range(5):
            frame = cam.read_camera()
        before_ir = _ir_stats(frame)
        before_depth = _depth_valid_frac(frame)

        ae = sensor.get_option(rs.option.enable_auto_exposure)
        print(f"\n=== {label}  s/n {cam.serial} ===")
        print(f"  firmware        {device.get_info(rs.camera_info.firmware_version)} "
              f"(recommended {device.get_info(rs.camera_info.recommended_firmware_version)})")
        print(f"  usb             {device.get_info(rs.camera_info.usb_type_descriptor)}")
        print(f"  ir auto-exposure {'ON' if ae else 'OFF'}, "
              f"exposure {sensor.get_option(rs.option.exposure):.0f} us")
        print(f"  emitter         {sensor.get_option(rs.option.emitter_enabled):.0f}, "
              f"laser power {sensor.get_option(rs.option.laser_power):.0f}")
        for side, (mean, sat) in before_ir.items():
            print(f"  ir_{side:<5s} mean {mean:6.1f}  saturated {sat * 100:5.1f} %")
        print(f"  depth valid     {before_depth * 100:5.1f} %")

        needs_fix = (not ae) or any(s > MAX_SATURATED_FRAC for _, s in before_ir.values())
        if needs_fix and apply:
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            if sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, 1)
            time.sleep(1.5)
            for _ in range(30):          # let AE converge
                frame = cam.read_camera()
            after_ir = _ir_stats(frame)
            after_depth = _depth_valid_frac(frame)
            print("  --- after enabling IR auto-exposure ---")
            for side, (mean, sat) in after_ir.items():
                print(f"  ir_{side:<5s} mean {mean:6.1f}  saturated {sat * 100:5.1f} %")
            print(f"  depth valid     {after_depth * 100:5.1f} %")
            ir_stats, depth_valid = after_ir, after_depth
        else:
            ir_stats, depth_valid = before_ir, before_depth

        intr = cam.get_intrinsics()
        print(f"  K_color         fx={intr.K_color[0, 0]:.1f} fy={intr.K_color[1, 1]:.1f} "
              f"cx={intr.K_color[0, 2]:.1f} cy={intr.K_color[1, 2]:.1f}")
        print(f"  ir baseline     {intr.baseline_ir * 1000:.2f} mm")

        usb = device.get_info(rs.camera_info.usb_type_descriptor)
        usb_ok = usb.startswith("3")
        if not usb_ok:
            print(f"  !! USB {usb} -- a D400 cannot stream 1280x720 colour plus "
                  f"both IR channels on USB 2. Symptom is a bare "
                  f"`RuntimeError: Couldn't resolve requests` at pipeline start, "
                  f"which does not mention USB at all. Move it to a USB 3 port.")

        ok = (usb_ok
              and all(s <= MAX_SATURATED_FRAC for _, s in ir_stats.values())
              and depth_valid >= MIN_DEPTH_VALID_FRAC)
        print(f"  => {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("     saturated IR or sparse depth: check scene lighting "
                  "(direct sun / windows blow out the projector pattern), "
                  "lens covers, and that the emitter is on")
        return ok
    finally:
        cam.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report only, do not change camera settings")
    ap.add_argument("--serial", action="append", metavar="SERIAL[=LABEL]",
                    help="camera to check; repeatable. Default: every device found")
    args = ap.parse_args()

    import pyrealsense2 as rs

    if args.serial:
        targets = []
        for spec in args.serial:
            sn, _, label = spec.partition("=")
            targets.append((sn, label or sn))
    else:
        ctx = rs.context()
        targets = [(d.get_info(rs.camera_info.serial_number),
                    d.get_info(rs.camera_info.name))
                   for d in ctx.query_devices()]
        if not targets:
            sys.exit("no RealSense devices found")
        print(f"found {len(targets)} RealSense device(s)")

    results = []
    for sn, label in targets:
        try:
            results.append(preflight(sn, label, apply=not args.check))
        except RuntimeError as exc:
            # Most often USB 2: the stream request cannot be satisfied and
            # librealsense reports it without naming the cause.
            print(f"\n=== {label}  s/n {sn} ===")
            print(f"  FAILED TO OPEN: {exc}")
            print("  check the USB port -- a D400 on USB 2 cannot serve "
                  "1280x720 colour + both IR channels")
            results.append(False)
    print(f"\n{sum(results)}/{len(results)} cameras PASS")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
