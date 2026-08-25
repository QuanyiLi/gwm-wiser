"""make_scene8: author scene8_0.usd = scene1 + banana + three cabinets.

Copies scene1_0.usd wholesale, absolutizes the table/robot payloads and the
dome HDRI, drops the rubiks_cube, re-places the stock _24_bowl, copies the
basic_block (painted a neutral gray) and the _11_banana prim spec (resting
attitude, yaw for a 100 deg long axis), then authors the cabinets from Cube
prims.

Each cabinet is authored in its own local frame (origin at the footprint
centre, x toward the back) under a parent Xform that carries the cabinet's
world position and yaw. Carcasses are static colliders: Xforms with no
RigidBodyAPI and no payload, so the scene loader leaves them unregistered
scenery. Each drawer is one rigid body (RigidBodyAPI + MassAPI on the Xform,
box colliders underneath) held by a prismatic joint to the world frame whose
axis is the cabinet's local x, limits [-pull-1 cm, 0]: it spawns floating with
3 mm reveals, touches nothing, and can only slide out toward the robot.

Run with the repo venv (has usd-core):

    /root/code/gwm/gwm-wiser/.venv/bin/python make_scene8.py
"""

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from config import ASSETS_DIR, BLOCK_COLOR, COLORS, DRAWERS, KNOB, OBJECTS

HERE = Path(__file__).resolve().parent
SHRINK = 0.0004  # inset between touching static slabs (kills coplanar faces)
FRONT_T = 0.018  # drawer front panel thickness
DRAWER_MASS = 0.35

BANANA_SPAWN_Z = 0.065
# Resting attitude of the YCB banana (scene3 spawn settled), long axis at
# in-plane yaw 131.6 deg; the extra world-Z yaw below puts it at 100 deg.
BANANA_BASE_QUAT = (0.2829093635082245, 0.29658013582229614,
                    0.679715633392334, 0.6082672476768494)
BANANA_YAW_DEG = -31.6


def banana_quat(yaw_deg, base=BANANA_BASE_QUAT):
    a = np.radians(yaw_deg) / 2
    w1, x1, y1, z1 = np.cos(a), 0.0, 0.0, np.sin(a)
    w2, x2, y2, z2 = base
    return Gf.Quatf(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def absolutize(layer: Sdf.Layer, prim_path: str) -> None:
    spec = layer.GetPrimAtPath(prim_path)
    items = spec.payloadList.prependedItems
    if items:
        spec.payloadList.prependedItems = [
            Sdf.Payload(str((ASSETS_DIR / p.assetPath).resolve())
                        if not p.assetPath.startswith(("http", "/")) else p.assetPath)
            for p in items
        ]


def make_material(stage, name, rgb, roughness=0.55, metallic=0.0):
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def yaw_quat(deg):
    a = np.radians(deg) / 2
    return Gf.Quatf(float(np.cos(a)), 0.0, 0.0, float(np.sin(a)))


def box(stage, path, center, size, mat, phys_mat=None):
    """Axis-aligned box collider in its parent's frame."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cube.AddScaleOp().Set(Gf.Vec3f(size[0] / 2, size[1] / 2, size[2] / 2))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    if phys_mat is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            phys_mat, materialPurpose="physics")
    return cube


def carcass(stage, cab, mats):
    """Static shell in the cabinet's local frame: side walls, top, back, base."""
    _, center = cab.frame()
    root = UsdGeom.Xform.Define(stage, f"/World/{cab.name}")
    root.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    root.AddOrientOp().Set(yaw_quat(cab.yaw))
    p = f"/World/{cab.name}"
    z0, h, w, d, t = cab.z0, cab.height, cab.width, cab.depth, cab.t
    body = mats[f"{cab.name}_body"]
    for tag, sy in (("left", -1), ("right", 1)):
        box(stage, f"{p}/wall_{tag}", (0, sy * (w - t) / 2, z0 + (h - t) / 2),
            (d, t, h - t), body)
    box(stage, f"{p}/top", (0, 0, z0 + h - t / 2), (d - SHRINK, w - SHRINK, t), body)
    box(stage, f"{p}/back", (d / 2 - t / 2, 0, z0 + (h - t) / 2),
        (t, w - 2 * t - SHRINK, h - t - SHRINK), body)
    box(stage, f"{p}/plinth", (0.002, 0, z0 + cab.plinth / 2),
        (d - 0.004, w - 2 * t - SHRINK, cab.plinth - SHRINK), body)


def drawer(stage, key, cab, mats, knob_phys):
    """One rigid-body drawer: front panel + block knob + open tray, on a world
    prismatic joint along the cabinet's local x."""
    z_lo, z_hi, y_lo, y_hi = cab.front_rect()
    z_fc, fw, fh = (z_lo + z_hi) / 2, y_hi - y_lo, z_hi - z_lo
    bay_lo, _ = cab.bay_z()
    origin = cab.to_world((cab.front_x_local, 0.0, z_fc))
    name = f"{key}_drawer"
    p = f"/World/{name}"

    x = UsdGeom.Xform.Define(stage, p)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in origin]))
    x.AddOrientOp().Set(yaw_quat(cab.yaw))
    prim = x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(DRAWER_MASS)
    prim.CreateAttribute("physxRigidBody:linearDamping",
                         Sdf.ValueTypeNames.Float).Set(3.0)
    prim.CreateAttribute("physxRigidBody:angularDamping",
                         Sdf.ValueTypeNames.Float).Set(3.0)

    front = mats[f"{cab.name}_front"]
    box(stage, f"{p}/front", (FRONT_T / 2, 0, 0), (FRONT_T, fw, fh), front)
    box(stage, f"{p}/knob", (-KNOB["depth"] / 2, 0, 0),
        (KNOB["depth"], KNOB["face"], KNOB["face"]), mats["knob"], knob_phys)

    td = cab.depth - cab.t - FRONT_T - 0.006  # tray depth behind the front
    tw = cab.width - 2 * cab.t - 0.006        # tray outer width (3 mm play/side)
    zb = (bay_lo + 0.008) - z_fc              # tray bottom centre, local z
    tray = mats["tray"]
    box(stage, f"{p}/tray_bottom", (FRONT_T + td / 2, 0, zb), (td, tw, 0.010), tray)
    wh = cab.tray_wall_h
    zw = (bay_lo + 0.013 + wh / 2) - z_fc
    for tag, sy in (("l", -1), ("r", 1)):
        box(stage, f"{p}/tray_{tag}", (FRONT_T + td / 2, sy * (tw - 0.010) / 2, zw),
            (td, 0.010, wh), tray)
    box(stage, f"{p}/tray_back", (FRONT_T + td - 0.005, 0, zw),
        (0.010, tw - 0.020, wh), tray)

    j = UsdPhysics.PrismaticJoint.Define(stage, f"/World/{name}_joint")
    j.CreateAxisAttr("X")
    j.CreateBody1Rel().SetTargets([Sdf.Path(p)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in origin]))
    j.CreateLocalRot0Attr(yaw_quat(cab.yaw))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-(cab.pull + 0.010))
    j.CreateUpperLimitAttr(0.0)
    return name


def main() -> None:
    out_path = HERE / "scene8_0.usd"
    out_path.unlink(missing_ok=True)
    out = Sdf.Layer.CreateNew(str(out_path), args={"format": "usda"})
    out.TransferContent(Sdf.Layer.FindOrOpen(str(ASSETS_DIR / "scene1_0.usd")))

    for prim in ("/World/table", "/World/robot"):
        absolutize(out, prim)
    dome_tex = out.GetAttributeAtPath("/World/DomeLight.inputs:texture:file")
    dome_tex.default = Sdf.AssetPath(str((ASSETS_DIR / dome_tex.default.path).resolve()))

    # stock bowl re-placed, rubiks cube dropped; block from scene5 and banana
    # from scene3 copied in
    world = out.GetPrimAtPath("/World")
    del world.nameChildren["rubiks_cube"]
    tr = out.GetAttributeAtPath("/World/_24_bowl.xformOp:translate")
    bx, by = OBJECTS["bowl"]["xy"]
    tr.default = Gf.Vec3d(bx, by, tr.default[2])
    s5 = Sdf.Layer.FindOrOpen(str(ASSETS_DIR / "scene5_0.usd"))
    Sdf.CopySpec(s5, "/World/basic_block", out, "/World/basic_block")
    tr = out.GetAttributeAtPath("/World/basic_block.xformOp:translate")
    bx, by = OBJECTS["block"]["xy"]
    tr.default = Gf.Vec3d(bx, by, tr.default[2])
    out.GetAttributeAtPath("/World/basic_block.xformOp:orient").default = \
        Gf.Quatd(1, 0, 0, 0)
    s3 = Sdf.Layer.FindOrOpen(str(ASSETS_DIR / "scene3_0.usd"))
    Sdf.CopySpec(s3, "/World/_11_banana", out, "/World/_11_banana")
    bx, by = OBJECTS["banana"]["xy"]
    out.GetAttributeAtPath("/World/_11_banana.xformOp:translate").default = \
        Gf.Vec3d(bx, by, BANANA_SPAWN_Z)
    out.GetAttributeAtPath("/World/_11_banana.xformOp:orient").default = \
        banana_quat(BANANA_YAW_DEG)
    out.Save()

    stage = Usd.Stage.Open(str(out_path), Usd.Stage.LoadNone)
    mats = {name: make_material(stage, name, rgb) for name, rgb in COLORS.items()}
    pm = UsdShade.Material.Define(stage, "/World/Looks/knob_phys")
    papi = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    papi.CreateStaticFrictionAttr(1.3)
    papi.CreateDynamicFrictionAttr(1.2)
    papi.CreateRestitutionAttr(0.0)

    for key, cab in DRAWERS.items():
        carcass(stage, cab, mats)
        drawer(stage, key, cab, mats, pm)
    # neutral paint over the block payload's stock binding (lives on its
    # descendant `Cube` mesh, so the override must be stronger)
    block_mat = make_material(stage, "block_paint", BLOCK_COLOR, roughness=0.6)
    target = stage.OverridePrim("/World/basic_block/Cube")
    UsdShade.MaterialBindingAPI.Apply(target).Bind(
        block_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    stage.Save()

    link = ASSETS_DIR / out_path.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(out_path)
    print(f"wrote {out_path} and symlinked {link}")
    for key, cab in DRAWERS.items():
        kx, ky, kz = cab.knob_center()
        print(f"  {key}_drawer: knob=({kx:.3f}, {ky:.3f}, {kz:.3f}) yaw={cab.yaw} "
              f"pull={cab.pull} size={cab.width}x{cab.height}x{cab.depth}")
    for key, spec in OBJECTS.items():
        print(f"  {key} ({spec['prim']}): xy={spec['xy']}")


if __name__ == "__main__":
    main()
