"""make_scene7: author scene7_0.usd = scene1 minus objects, plus a 2x2 grid of photos.

Scene 7 is a clean table with four 15 cm "printed photos" (textured static
quads) for the pointing/CEM showcase. Built exactly like make_scene6: copy
scene1_0.usd wholesale, absolutize the table/robot payloads and the dome HDRI,
then author the new prims. The rubiks_cube and _24_bowl prim specs are removed
so the table carries nothing but the photos.

The quads are plain UsdGeom.Mesh prims with no payload and no RigidBodyAPI;
SceneCfg.dynamic_scene leaves them unregistered (droid_environment.py:115) and
they stay static scenery. Each quad is rotated -90 deg about Z: the texture's
"up" points toward world +x.

Run with the repo venv (has usd-core):

    /root/code/gwm/gwm-wiser/.venv/bin/python make_scene7.py
"""

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from config import ASSETS_DIR, CELLS, IMG_FILES, IMG_LIFT, IMG_SIZE, TABLE_TOP_Z

HERE = Path(__file__).resolve().parent
REMOVE_PRIMS = ("rubiks_cube", "_24_bowl")


def absolutize(layer: Sdf.Layer, prim_path: str) -> None:
    spec = layer.GetPrimAtPath(prim_path)
    items = spec.payloadList.prependedItems
    if items:
        spec.payloadList.prependedItems = [
            Sdf.Payload(str((ASSETS_DIR / p.assetPath).resolve())
                        if not p.assetPath.startswith(("http", "/")) else p.assetPath)
            for p in items
        ]


def add_photo(stage: Usd.Stage, name: str, x: float, y: float, png: Path) -> None:
    h = IMG_SIZE / 2
    mesh = UsdGeom.Mesh.Define(stage, f"/World/img_{name}")
    mesh.CreatePointsAttr(Vt.Vec3fArray([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([(0, 0, 1)] * 4))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr(Vt.Vec3fArray([(-h, -h, 0), (h, h, 0)]))
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([(0, 0), (1, 0), (1, 1), (0, 1)]))
    mesh.AddTranslateOp().Set(Gf.Vec3d(x, y, TABLE_TOP_Z + IMG_LIFT))
    mesh.AddRotateZOp().Set(-90.0)  # texture up -> world +x

    mat = UsdShade.Material.Define(stage, f"/World/Looks/img_{name}_mat")
    surf = UsdShade.Shader.Define(stage, f"/World/Looks/img_{name}_mat/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, f"/World/Looks/img_{name}_mat/StReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"/World/Looks/img_{name}_mat/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(png.resolve()))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def main() -> None:
    out_path = HERE / "scene7_0.usd"
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
    for name, (x, y) in CELLS.items():
        add_photo(stage, name, x, y, IMG_FILES[name])
    stage.Save()

    link = ASSETS_DIR / out_path.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(out_path)
    print(f"wrote {out_path} and symlinked {link}")
    for name, (x, y) in CELLS.items():
        print(f"  img_{name}: centre=({x:.3f}, {y:.3f}, {TABLE_TOP_Z + IMG_LIFT:.4f}) "
              f"span x=[{x - IMG_SIZE/2:.3f},{x + IMG_SIZE/2:.3f}] y=[{y - IMG_SIZE/2:.3f},{y + IMG_SIZE/2:.3f}]")


if __name__ == "__main__":
    main()
