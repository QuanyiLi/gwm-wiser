"""make_scene: author the push scene = scene1 minus objects, plus three cubes.

Writes scene9_0.usd: a clean table carrying three cubes, one in front of the
gripper's home position and one to each side. Copies scene1_0.usd wholesale,
absolutizes the table/robot payloads and the dome HDRI, drops the stock
rubiks_cube and _24_bowl, then authors the cubes.

Each cube is a direct child of /World carrying RigidBodyAPI and MassAPI with a
single box collider underneath, which is what the scene loader looks for
(droid_environment.py: a /World child with a rigid body or a payload becomes a
RigidObjectCfg initialised from its translate/orient).

Run with the repo venv (has usd-core):

    /root/code/gwm/gwm-wiser/.venv/bin/python make_scene.py
"""

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from config import (ASSETS_DIR, CUBE_FRICTION, CUBE_MASS, CUBE_PRIMS, CUBE_RGB,
                    CUBE_SIZE, CUBE_SPAWN_Z, CUBES, SCENE_ID, SCENE_VARIANT)

HERE = Path(__file__).resolve().parent
REMOVE_PRIMS = ("rubiks_cube", "_24_bowl")

LINEAR_DAMPING = 0.05
ANGULAR_DAMPING = 0.05


def absolutize(layer: Sdf.Layer, prim_path: str) -> None:
    spec = layer.GetPrimAtPath(prim_path)
    items = spec.payloadList.prependedItems
    if items:
        spec.payloadList.prependedItems = [
            Sdf.Payload(str((ASSETS_DIR / p.assetPath).resolve())
                        if not p.assetPath.startswith(("http", "/")) else p.assetPath)
            for p in items
        ]


def make_material(stage, name, rgb, roughness=0.5):
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def add_cube(stage, name, x, y, mat, phys_mat):
    """One free rigid cube resting on the table at (x, y)."""
    root = UsdGeom.Xform.Define(stage, f"/World/{name}")
    root.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(CUBE_SPAWN_Z)))
    root.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(CUBE_MASS)
    prim.CreateAttribute("physxRigidBody:linearDamping",
                         Sdf.ValueTypeNames.Float).Set(LINEAR_DAMPING)
    prim.CreateAttribute("physxRigidBody:angularDamping",
                         Sdf.ValueTypeNames.Float).Set(ANGULAR_DAMPING)

    geom = UsdGeom.Cube.Define(stage, f"/World/{name}/geom")
    geom.CreateSizeAttr(2.0)
    geom.AddScaleOp().Set(Gf.Vec3f(*([CUBE_SIZE / 2] * 3)))
    geom.CreateExtentAttr([Gf.Vec3f(-1, -1, -1), Gf.Vec3f(1, 1, 1)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    binding = UsdShade.MaterialBindingAPI.Apply(geom.GetPrim())
    binding.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    binding.Bind(phys_mat, materialPurpose="physics")


def main() -> None:
    out_path = HERE / f"scene{SCENE_ID}_{SCENE_VARIANT}.usd"
    out_path.unlink(missing_ok=True)
    out = Sdf.Layer.CreateNew(str(out_path), args={"format": "usda"})
    out.TransferContent(Sdf.Layer.FindOrOpen(str(ASSETS_DIR / "scene1_0.usd")))

    for prim in ("/World/table", "/World/robot"):
        absolutize(out, prim)
    dome_tex = out.GetAttributeAtPath("/World/DomeLight.inputs:texture:file")
    dome_tex.default = Sdf.AssetPath(str((ASSETS_DIR / dome_tex.default.path).resolve()))

    world = out.GetPrimAtPath("/World")
    for name in REMOVE_PRIMS:
        if name in world.nameChildren:
            del world.nameChildren[name]
    out.Save()

    stage = Usd.Stage.Open(str(out_path), Usd.Stage.LoadNone)
    mats = {tag: make_material(stage, f"cube_paint_{tag}", rgb)
            for tag, rgb in CUBE_RGB.items()}
    pm = UsdShade.Material.Define(stage, "/World/Looks/cube_phys")
    papi = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    papi.CreateStaticFrictionAttr(CUBE_FRICTION)
    papi.CreateDynamicFrictionAttr(CUBE_FRICTION)
    papi.CreateRestitutionAttr(0.0)
    for tag, (x, y) in CUBES.items():
        add_cube(stage, CUBE_PRIMS[tag], x, y, mats[tag], pm)
    stage.Save()

    link = ASSETS_DIR / out_path.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(out_path)
    print(f"wrote {out_path} and symlinked {link}")
    for tag, (x, y) in CUBES.items():
        print(f"  {CUBE_PRIMS[tag]}: centre=({x:.3f}, {y:.3f}, {CUBE_SPAWN_Z:.4f}) "
              f"size={CUBE_SIZE} mass={CUBE_MASS} rgb={CUBE_RGB[tag]}")


if __name__ == "__main__":
    main()
