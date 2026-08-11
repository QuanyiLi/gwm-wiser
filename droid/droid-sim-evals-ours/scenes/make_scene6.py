"""make_scene6: author scene6_*.usd = stock scene1 + a YCB 011_banana.

Scene 6 ("scene1 + banana") supports richer referring-expression grasp tasks
(three objects: rubiks_cube, _24_bowl, _11_banana). The banana prim spec is
copied verbatim from scene3_0.usd (same payload URL, physics APIs, convex
decomposition) with only its pose replaced; everything else is scene1_0.usd
unchanged. All file asset paths are absolutized so the output can live here
and be reached through a symlink in droid-sim-evals/assets/ (the loader
formats `assets/scene{name}_{variant}.usd`).

Placement default (0.547, -0.243, long-axis yaw 100°), user-specified
geometry: |banana - bowl| >= |bowl - cube| (0.298 vs 0.212 m), bowl in the
middle, banana on the OPPOSITE side of the bowl from the cube. Solved offline
against the scene-1 home-pose captures under: (a) wrist visibility — M2T2 and
Gemini both perceive from the home wrist rgbd, and the exact opposite-side ray
exits the wrist frame, so this is the most-collinear wrist-visible point
(cube-bowl-banana angle 132°, 9/9 body samples visible on clean table);
(b) no spawn interpenetration — clearance is measured SEGMENT-to-center (a
19 cm banana tip pointed at the bowl caused a PhysX depenetration blast in an
earlier rev), enforced against BOTH the bowl's stock spawn and its rest pose,
so the stock bowl and its ~6 cm settle-wobble stay untouched and clear.
Banana base orientation is its MEASURED resting attitude (an x90 spawn drops
3.3 cm and rolls/yaws ~80°); spawning already-settled plus a compensating
world-Z yaw makes spawn == rest. scene1's three prims are byte-identical to
stock. Run with the gwm-wiser .venv python (has usd-core):

    /root/code/gwm/gwm-wiser/.venv/bin/python make_scene6.py [--variant 0]
        [--banana-x 0.547 --banana-y -0.243 --banana-yaw-deg -31.6]
"""

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf

ASSETS = Path("/root/code/gwm/gwm-wiser/droid/droid-sim-evals/assets")
HERE = Path(__file__).resolve().parent
BANANA_SPAWN_Z = 0.065  # measured rest height 0.063 + 2 mm
# Resting orientation measured after settling the scene3-style x90 spawn
# (capture scene6_0 rev1); localX (long axis) at in-plane yaw 131.6 deg.
BASE_QUAT = (0.2829093635082245, 0.29658013582229614, 0.679715633392334, 0.6082672476768494)


def banana_quat(yaw_deg: float, base=BASE_QUAT) -> Gf.Quatf:
    """Extra yaw about world Z composed onto the resting base orientation."""
    qx = np.asarray(base, dtype=float)
    a = np.radians(yaw_deg) / 2
    qy = np.array([np.cos(a), 0.0, 0.0, np.sin(a)])
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qx
    q = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    return Gf.Quatf(q[0], q[1], q[2], q[3])


def absolutize(layer: Sdf.Layer, prim_path: str) -> None:
    spec = layer.GetPrimAtPath(prim_path)
    items = spec.payloadList.prependedItems
    if items:
        spec.payloadList.prependedItems = [
            Sdf.Payload(str((ASSETS / p.assetPath).resolve()) if not p.assetPath.startswith(("http", "/")) else p.assetPath)
            for p in items
        ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--banana-x", type=float, default=0.547)
    ap.add_argument("--banana-y", type=float, default=-0.243)
    ap.add_argument("--banana-yaw-deg", type=float, default=-31.6,
                    help="extra world-Z yaw on top of the resting base quat (default puts the long axis at 100 deg)")
    args = ap.parse_args()

    out_path = HERE / f"scene6_{args.variant}.usd"
    out_path.unlink(missing_ok=True)
    out = Sdf.Layer.CreateNew(str(out_path), args={"format": "usda"})
    out.TransferContent(Sdf.Layer.FindOrOpen(str(ASSETS / "scene1_0.usd")))

    for prim in ("/World/table", "/World/robot"):
        absolutize(out, prim)
    dome_tex = out.GetAttributeAtPath("/World/DomeLight.inputs:texture:file")
    dome_tex.default = Sdf.AssetPath(str((ASSETS / dome_tex.default.path).resolve()))

    s3 = Sdf.Layer.FindOrOpen(str(ASSETS / "scene3_0.usd"))
    Sdf.CopySpec(s3, "/World/_11_banana", out, "/World/_11_banana")
    out.GetAttributeAtPath("/World/_11_banana.xformOp:translate").default = Gf.Vec3d(
        args.banana_x, args.banana_y, BANANA_SPAWN_Z
    )
    out.GetAttributeAtPath("/World/_11_banana.xformOp:orient").default = banana_quat(args.banana_yaw_deg)

    out.Save()

    link = ASSETS / out_path.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(out_path)
    print(f"wrote {out_path} and symlinked {link}")
    print(f"banana pose: ({args.banana_x}, {args.banana_y}, {BANANA_SPAWN_Z}) yaw={args.banana_yaw_deg}")


if __name__ == "__main__":
    main()
