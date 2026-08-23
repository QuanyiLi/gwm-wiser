"""Warm the perception microservices before a tiptop run, and report health.

Why: on the RTX 5090, the M2T2 and FoundationStereo pixi envs both pin
**torch 2.4.1 / CUDA 12.0**, whose builds carry no ``sm_120`` cubins -- only
``compute_90`` PTX. Blackwell runs that PTX fine, but the driver has to JIT
every kernel on first use. On FoundationStereo with a real 1280x720 RealSense
IR pair:

    first /infer after server start   ~30 s
    every subsequent /infer            ~1 s

``tiptop.perception.foundation_stereo.infer_depth_async`` sets a hard-coded
``aiohttp.ClientTimeout(total=10.0)``, so **the first capture of every session
raises TimeoutError** -- once per server restart, then never again. ``tiptop/``
is kept pristine, so the fix is operational: send one throwaway request per
server before handing the rig to tiptop.

Run after starting the servers, before ``tiptop-run``:

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.warm_servers
"""

import argparse
import io
import sys
import time

import numpy as np
import requests

M2T2_DEFAULT = "http://localhost:8123"
FS_DEFAULT = "http://localhost:1234"
# A cold JIT pass is ~35 s; leave generous headroom, this runs once.
WARM_TIMEOUT_S = 1200


def _health(url: str, name: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        r.raise_for_status()
        print(f"  {name:<18s} healthy  {r.json()}")
        return True
    except Exception as e:
        print(f"  {name:<18s} UNREACHABLE at {url} ({type(e).__name__})")
        return False


def warm_m2t2(url: str) -> bool:
    """One tiny whole-scene grasp request: exercises pointnet2_ops' CUDA kernels."""
    rng = np.random.default_rng(0)
    table = np.stack([rng.uniform(0.3, 0.8, 4000),
                      rng.uniform(-0.25, 0.25, 4000),
                      np.zeros(4000)], axis=1)
    box = rng.uniform(-0.03, 0.03, (1500, 3)) + np.array([0.55, 0.0, 0.03])
    pts = np.concatenate([table, box])
    payload = {"pointcloud": {"points": pts.tolist(),
                              "rgb": [[0.6, 0.6, 0.6]] * len(pts)},
               "num_points": 16384, "num_runs": 1, "mask_thresh": 0.2,
               "apply_bounds": True}
    t0 = time.perf_counter()
    r = requests.post(f"{url}/predict", json=payload, timeout=WARM_TIMEOUT_S)
    r.raise_for_status()
    print(f"  M2T2 warm-up       {time.perf_counter() - t0:6.1f} s  "
          f"({r.json()['num_grasps']} grasps on the throwaway scene)")
    return True


def warm_foundation_stereo(url: str, serial: str | None) -> bool:
    """One /infer call. Uses a live IR pair when a camera is present, else noise
    at the same resolution -- either way the JIT cache gets populated."""
    from PIL import Image

    if serial:
        from tiptop.perception.cameras.rs_camera import RealsenseCamera
        cam = RealsenseCamera(serial, enable_depth=False, enable_ir=True)
        try:
            for _ in range(10):
                frame = cam.read_camera()
            left, right = frame.ir_left, frame.ir_right
            intr = cam.get_intrinsics()
            fx, fy = float(intr.K_ir[0, 0]), float(intr.K_ir[1, 1])
            cx, cy = float(intr.K_ir[0, 2]), float(intr.K_ir[1, 2])
            baseline = float(intr.baseline_ir)
        finally:
            cam.close()
    else:
        rng = np.random.default_rng(0)
        left = rng.integers(0, 255, (720, 1280), dtype=np.uint8)
        right = np.roll(left, -8, axis=1)
        fx = fy = 640.0
        cx, cy, baseline = 640.0, 360.0, 0.05

    def png(a):
        buf = io.BytesIO()
        Image.fromarray(np.stack([a] * 3, -1)).save(buf, "PNG")
        return buf.getvalue()

    t0 = time.perf_counter()
    r = requests.post(
        f"{url}/infer",
        files={"left_image": ("l.png", png(left), "image/png"),
               "right_image": ("r.png", png(right), "image/png")},
        data={"fx": fx, "fy": fy, "cx": cx, "cy": cy,
              "baseline": baseline, "valid_iters": 32},
        timeout=WARM_TIMEOUT_S,
    )
    r.raise_for_status()
    npz = np.load(io.BytesIO(r.content))
    depth = npz[list(npz.keys())[0]]
    valid = float((np.isfinite(depth) & (depth > 0) & (depth < 5)).mean())
    dt = time.perf_counter() - t0
    print(f"  FoundationStereo   {dt:6.1f} s  depth {depth.shape} "
          f"valid {valid * 100:.1f} %"
          + ("   <- cold JIT, subsequent calls ~1 s" if dt > 10 else ""))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m2t2-url", default=M2T2_DEFAULT)
    ap.add_argument("--fs-url", default=FS_DEFAULT)
    ap.add_argument("--hand-serial", default=None,
                    help="RealSense serial to source a real IR pair from; "
                         "omit to warm with synthetic images")
    ap.add_argument("--skip", choices=["m2t2", "fs"], action="append", default=[])
    args = ap.parse_args()

    print("health:")
    ok = True
    if "m2t2" not in args.skip:
        ok &= _health(args.m2t2_url, "M2T2")
    if "fs" not in args.skip:
        ok &= _health(args.fs_url, "FoundationStereo")
    if not ok:
        sys.exit("start the M2T2 / FoundationStereo servers first")

    print("warming (first call JITs compute_90 PTX for sm_120):")
    if "m2t2" not in args.skip:
        warm_m2t2(args.m2t2_url)
    if "fs" not in args.skip:
        warm_foundation_stereo(args.fs_url, args.hand_serial)
    print("servers warm -- tiptop's 10 s client timeout is safe now")


if __name__ == "__main__":
    main()
