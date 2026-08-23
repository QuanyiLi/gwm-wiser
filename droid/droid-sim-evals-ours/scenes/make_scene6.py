"""make_scene6: author scene6_*.usd = stock scene1 + a YCB 011_banana + two colored bins.

Scene 6 ("scene1 + banana") supports richer referring-expression grasp tasks
(three objects: rubiks_cube, _24_bowl, _11_banana). The banana prim spec is
copied verbatim from scene3_0.usd (same payload URL, physics APIs, convex
decomposition) with only its pose replaced; everything else is scene1_0.usd
unchanged. All file asset paths are absolutized so the output can live here
and be reached through a symlink in droid-sim-evals/assets/ (the loader
formats `assets/scene{name}_{variant}.usd`).

Placement default (0.547, -0.243, long-axis yaw 100°) satisfies the layout
constraints |banana - bowl| >= |bowl - cube| (0.298 vs 0.212 m), bowl in the
middle, banana on the OPPOSITE side of the bowl from the cube. Solved offline
against the scene-1 home-pose captures under: (a) wrist visibility — M2T2 and
Gemini both perceive from the home wrist rgbd, and the exact opposite-side ray
exits the wrist frame, so this is the most-collinear wrist-visible point
(cube-bowl-banana angle 132°, every body sample visible on a clean table);
(b) no spawn interpenetration — clearance is measured SEGMENT-to-center, so a
banana tip pointed at the bowl cannot overlap it and trigger a PhysX
depenetration impulse; this is enforced against BOTH the bowl's stock spawn
and its rest pose, so the stock bowl and its ~6 cm settle-wobble stay
untouched and clear. Banana base orientation is its MEASURED resting attitude
(an x90 spawn drops 3.3 cm and rolls/yaws ~80°); spawning already-settled plus
a compensating world-Z yaw makes spawn == rest. scene1's three prims are
byte-identical to stock.

BINS (for the placement/referring eval). Two `small_KLT_visual_collision`
prims copied from scene3_0.usd, squared off by a non-uniform scale (see
bin_geom) and recolored via a UsdPreviewSurface bound on the payload's
`Visuals` scope with `strongerThanDescendants` -- the stock magenta binding
sits on the descendant meshes, so a weaker binding is ignored.

The positions come from optimize_bin_layout.py, which searches the table for
bin placements under these layout constraints:

  - Wrist visibility and gripper shadow. The home wrist RGB-D is the only view
    M2T2/Gemini/perception_geometric get, and the fingers black out a wide
    band across the near-centre of it; every bin corner must be inside the
    wrist frame with margin, and at most 2% of a bin opening may fall in the
    gripper shadow, which rules out the x ~ 0.55 slot between the bowl and
    the banana and pushes both bins to lower x.
  - External-camera visibility. Both bins fully in frame, not occluding each
    other or the banana, and as far apart in the image as the other
    constraints allow, so the two destinations are visually distinct from the
    static viewpoint.
  - Square footprint. The stock 0.198 x 0.297 bin cannot fit the corridor
    between the bowl and the banana at any orientation. Squaring it to 0.115
    frees the whole -Y half of the free block; the price is 0.43 mm walls in
    the squeezed direction (stock 1.10/1.20 mm), which still settle within a
    few steps and hold their pose. Smaller bins also unlock slots a larger
    bin cannot enter.
  - Clearance and reach. AABB clearance to the three stock objects, and a
    planar reach band matching what the stock scenes exercise.

Result: 0.115 m square bins 0.068 m tall, 0.102 x 0.105 m opening, red at
(0.395, -0.055), green at (0.305, -0.250) -- 0.215 m apart in world. Spawn z
gives a 5 mm drop (stock KLT uses 7.1 mm).

VARIANTS. `--variant 0` is the pick scene (no held block) so the refer6 pick
tasks keep working; `--variant 1` adds `held_block` inside the gripper and is
the place scene. Splitting by variant costs nothing -- `--variant` is already a
batch_eval_v2 CLI arg -- whereas putting the block in a single scene would break
all ten refer6 pick tasks, which assume an empty hand.

Prim names are `red_bin` / `green_bin` / `held_block`, none containing "KLT" or
"cube": SuccessTracker resolves rule patterns by unique case-insensitive
substring over scene.rigid_objects, and a 2-way match raises. "bin" alone
matches both bins by design -- rules must say "red_bin"/"green_bin".

Run with the gwm-wiser .venv python (has usd-core):

    /root/code/gwm/gwm-wiser/.venv/bin/python make_scene6.py [--variant 0]
        [--banana-x 0.547 --banana-y -0.243 --banana-yaw-deg -31.6]
        [--bin-size 0.115 --bin-height 0.068]
        [--red-x 0.395 --red-y -0.055 --green-x 0.305 --green-y -0.250]
        [--no-bins]
    /root/code/gwm/gwm-wiser/.venv/bin/python make_scene6.py --variant 1 \
        --held-block [--block-edge 0.030]
"""

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdShade

import bin_geom
from bin_geom import DEFAULT_HEIGHT, DEFAULT_SIZE, bin_report, bin_scale, bin_spawn_z

ASSETS = Path("/root/code/gwm/gwm-wiser/droid/droid-sim-evals/assets")
HERE = Path(__file__).resolve().parent
BANANA_SPAWN_Z = 0.065  # measured rest height 0.063 + 2 mm
# Resting orientation measured after settling the scene3-style x90 spawn;
# localX (long axis) at in-plane yaw 131.6 deg.
BASE_QUAT = (0.2829093635082245, 0.29658013582229614, 0.679715633392334, 0.6082672476768494)

KLT_SRC = "/World/small_KLT_visual_collision"
KLT_YAW90 = Gf.Quatd(0.7071067811865476, 0, 0, 0.7071067811865475)  # scene3 orient, kept for asset-axis parity
# Saturated hues so the color survives the 624x352 anchor resize GWM scores on,
# and so "the color of a tomato" / "the color of grass" are unambiguous.
BIN_COLORS = {"red_bin": (0.85, 0.06, 0.06), "green_bin": (0.08, 0.60, 0.12)}

BLOCK_SRC = "/World/basic_block"  # scene5's 47 mm Isaac 5.1 Props/Blocks cube
BLOCK_STOCK_EDGE = 0.047
# basic_block's prim origin is NOT its mesh centre: the payload's `Cube` child
# carries xformOp:translate z = -0.0392149. Confirmed on scene5, where the prim
# sits at z 0.10765 and the measured mesh centre is 0.06844. The offset lives on
# a descendant, so a prim-level uniform scale scales it too.
BLOCK_MESH_OFFSET_Z = -0.039214913
# The block is welded to the gripper rather than grasped, so grip robustness
# is irrelevant and the ONLY thing edge length controls is whether
# the gripper fits through the bin opening. Gripper y-width profile, measured by
# unprojecting the home wrist depth, fully open:
#     0-20 mm above the tips  0.0969   (pads only, 6.0 mm thick a side)
#    20-30 mm                 0.1225
#    40-50 mm                 0.1438   <- widest, the knuckles
# Closing on an edge-e block narrows everything by (0.085 - e), so the widest
# section is 0.0588 + e. Against the 0.105 m bin opening a 47 mm block gives
# 0.106 m -- 1 mm too wide, every plan would fail on collision. 30 mm gives
# 0.089 m, i.e. 16 mm of clearance.
BLOCK_EDGE = 0.030
GRIPPER_TIP_Z = 0.3226  # measured finger-tip height at the home pose
BLOCK_HELD_XY = (0.355, 0.0)  # pads are centred here; inner faces at y = +-0.0425
# Blue: unmistakable against the red/green bins, so a colour referring
# expression can never be read as pointing at the block itself.
BLOCK_COLOR = (0.05, 0.15, 0.85)


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


def add_bin(out: Sdf.Layer, src: Sdf.Layer, name: str, x: float, y: float, size: float, height: float) -> dict:
    """Copy the scene3 KLT prim spec (payload + convex-decomposition overs) under a new name.

    The scale is non-uniform (see bin_geom) so the 3:2 stock footprint becomes a
    square; it is applied in the asset's local frame, before xformOp:orient.
    """
    path = f"/World/{name}"
    Sdf.CopySpec(src, KLT_SRC, out, path)
    sx, sy, sz = bin_scale(size, height)
    z = bin_spawn_z(height)
    out.GetAttributeAtPath(f"{path}.xformOp:translate").default = Gf.Vec3d(x, y, z)
    out.GetAttributeAtPath(f"{path}.xformOp:scale").default = Gf.Vec3f(sx, sy, sz)
    out.GetAttributeAtPath(f"{path}.xformOp:orient").default = KLT_YAW90
    return {"name": name, "pos": (x, y, z), "size": size, "height": height}


def add_held_block(out: Sdf.Layer, src: Sdf.Layer, name: str, edge: float) -> dict:
    """Copy scene5's `basic_block`, scale it to `edge`, and spawn it in the gripper.

    Sits centred between the pads with its bottom 1 mm above the finger tips.
    The mesh-centre offset rides on a descendant prim, so it scales with the
    prim-level scale and has to be un-applied at the scaled size.
    """
    path = f"/World/{name}"
    Sdf.CopySpec(src, BLOCK_SRC, out, path)
    s = edge / BLOCK_STOCK_EDGE
    x, y = BLOCK_HELD_XY
    z = GRIPPER_TIP_Z + edge / 2 + 0.001
    out.GetAttributeAtPath(f"{path}.xformOp:translate").default = Gf.Vec3d(x, y, z - BLOCK_MESH_OFFSET_Z * s)
    out.GetAttributeAtPath(f"{path}.xformOp:orient").default = Gf.Quatd(1, 0, 0, 0)
    out.GetAttributeAtPath(f"{path}.xformOp:scale").default = Gf.Vec3d(s, s, s)
    return {"name": name, "mesh_centre": (x, y, z), "edge": edge, "scale": s}


def paint(out_path: Path, targets) -> None:
    """Author UsdPreviewSurface materials and bind them over each payload's mesh scope.

    `targets` maps prim name -> (rgb, subpath), where subpath is the prim under
    the payload that carries the stock binding. Opened with LoadNone: these
    payloads are remote Isaac URLs that only resolve inside the Isaac runtime,
    and we only need to author local prims. `strongerThanDescendants` is
    required -- the stock bindings live on the descendant meshes (the KLT's
    FOF_Mesh_Magenta_Box plus two label decals, the block's Cube).
    """
    stage = Usd.Stage.Open(str(out_path), Usd.Stage.LoadNone)
    looks = stage.OverridePrim("/World/Looks")
    looks.SetSpecifier(Sdf.SpecifierDef)
    looks.SetTypeName("Scope")
    for name, (rgb, subpath) in targets.items():
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        target = stage.OverridePrim(f"/World/{name}/{subpath}")
        UsdShade.MaterialBindingAPI.Apply(target).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
    stage.Save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--banana-x", type=float, default=0.547)
    ap.add_argument("--banana-y", type=float, default=-0.243)
    ap.add_argument("--banana-yaw-deg", type=float, default=-31.6,
                    help="extra world-Z yaw on top of the resting base quat (default puts the long axis at 100 deg)")
    ap.add_argument("--bin-size", type=float, default=DEFAULT_SIZE, help="square world footprint edge (m)")
    ap.add_argument("--bin-height", type=float, default=DEFAULT_HEIGHT)
    ap.add_argument("--red-x", type=float, default=0.395)
    ap.add_argument("--red-y", type=float, default=-0.055)
    ap.add_argument("--green-x", type=float, default=0.305)
    ap.add_argument("--green-y", type=float, default=-0.250)
    ap.add_argument("--no-bins", action="store_true", help="author the scene without bins")
    ap.add_argument("--block-edge", type=float, default=BLOCK_EDGE,
                    help="held block edge (m); gripper width when closed on it is 0.0588+edge")
    ap.add_argument("--held-block", action="store_true",
                    help="spawn a 47 mm block inside the gripper (the place-task variant)")
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

    bins = []
    if not args.no_bins:
        bins.append(add_bin(out, s3, "red_bin", args.red_x, args.red_y, args.bin_size, args.bin_height))
        bins.append(add_bin(out, s3, "green_bin", args.green_x, args.green_y, args.bin_size, args.bin_height))

    held = None
    if args.held_block:
        s5 = Sdf.Layer.FindOrOpen(str(ASSETS / "scene5_0.usd"))
        held = add_held_block(out, s5, "held_block", args.block_edge)

    out.Save()
    targets = {b["name"]: (BIN_COLORS[b["name"]], "Visuals") for b in bins}
    if held:
        targets["held_block"] = (BLOCK_COLOR, "Cube")
    if targets:
        paint(out_path, targets)

    link = ASSETS / out_path.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(out_path)
    print(f"wrote {out_path} and symlinked {link}")
    print(f"banana pose: ({args.banana_x}, {args.banana_y}, {BANANA_SPAWN_Z}) yaw={args.banana_yaw_deg}")
    if bins:
        print("bin geometry:", bin_report(args.bin_size, args.bin_height))
    for b in bins:
        h = b["size"] / 2
        x, y, z = b["pos"]
        print(f"{b['name']}: pos=({x:.3f}, {y:.3f}, {z:.4f}) "
              f"X=[{x - h:.3f},{x + h:.3f}] Y=[{y - h:.3f},{y + h:.3f}] top_z={z + b['height'] / 2:.3f} "
              f"reach={np.hypot(x, y):.3f}")
    if len(bins) == 2:
        (rx, ry, _), (gx, gy, _) = bins[0]["pos"], bins[1]["pos"]
        print(f"bin centre separation: {np.hypot(rx - gx, ry - gy):.3f} m")
    if held:
        x, y, z = held["mesh_centre"]
        e = held["edge"]
        print(f"held_block: mesh centre=({x:.3f}, {y:.3f}, {z:.3f}) edge={e:.3f} scale={held['scale']:.4f} "
              f"spans z=[{z - e / 2:.4f},{z + e / 2:.4f}] (tips at {GRIPPER_TIP_Z})")
        print(f"    gripper width closed on it: {0.0588 + e:.3f} m vs bin opening "
              f"{bin_geom.KLT_INNER_X * bin_geom.bin_scale(args.bin_size, args.bin_height)[1]:.3f} m "
              f"-> {(bin_geom.KLT_INNER_X * bin_geom.bin_scale(args.bin_size, args.bin_height)[1] - 0.0588 - e) * 500:.1f} mm clearance a side")


if __name__ == "__main__":
    main()
