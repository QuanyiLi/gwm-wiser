"""Write the scene's USD assets as text (.usda) — no USD library needed.

    python assets/make_assets.py          # from droid/gwm_rl/

Produces, next to this script:

- ``cab_red.usda`` / ``cab_yellow.usda`` / ``cab_blue.usda`` — one cabinet each:
  carcass slabs, the closed drawer front, knob and tray as box colliders under
  a single kinematic rigid body, with the scene8 colours — a body, not static
  scenery, so a contact sensor can be attached to it.
- ``table.usda`` — payloads DROID's ``table.usd`` with the same overrides
  scene1/scene8 apply (payload offset zeroed, oak top), as a kinematic body.
- ``bowl.usda`` / ``banana.usda`` — payload the YCB meshes fetched by
  ``fetch_assets.sh`` with the physics schemas scene8 puts on them (rigid
  body, convex-decomposition collider).

Paths inside the files are relative to this directory, so the folder can be
copied to another machine together with ``fetch_assets.sh``'s output.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from gwm_rl.geometry import CABINETS, COLORS, DROID_SIM_EVALS_ASSETS  # noqa: E402

HEADER = """#usda 1.0
(
    defaultPrim = "{default}"
    metersPerUnit = 1
    upAxis = "Z"
)
"""


def material(name: str, rgb, roughness=0.55, metallic=0.0, indent=8) -> str:
    pad = " " * indent
    r, g, b = rgb
    return (
        f'{pad}def Material "{name}"\n{pad}{{\n'
        f'{pad}    token outputs:surface.connect = </{{root}}/Looks/{name}/Shader.outputs:surface>\n\n'
        f'{pad}    def Shader "Shader"\n{pad}    {{\n'
        f'{pad}        uniform token info:id = "UsdPreviewSurface"\n'
        f'{pad}        color3f inputs:diffuseColor = ({r}, {g}, {b})\n'
        f'{pad}        float inputs:metallic = {metallic}\n'
        f'{pad}        float inputs:roughness = {roughness}\n'
        f'{pad}        token outputs:surface\n'
        f'{pad}    }}\n{pad}}}\n'
    )


def cube(name: str, center, size, mat: str, root: str, phys_mat: str | None = None) -> str:
    cx, cy, cz = center
    sx, sy, sz = (s / 2 for s in size)  # Cube size 2 -> half extents are the scale
    phys = f"        rel material:binding:physics = </{root}/Looks/{phys_mat}>\n" if phys_mat else ""
    return (
        f'    def Cube "{name}" (\n'
        f'        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]\n'
        f'    )\n    {{\n'
        f'        rel material:binding = </{root}/Looks/{mat}>\n'
        f'{phys}'
        f'        float3 xformOp:scale = ({sx}, {sy}, {sz})\n'
        f'        double3 xformOp:translate = ({cx}, {cy}, {cz})\n'
        f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n'
        f'    }}\n'
    )


def cabinet_usda(cab) -> str:
    root = cab.name
    mats = [f"{root}_body", f"{root}_front", "knob", "tray"]
    out = HEADER.format(default=root)
    out += (
        f'\ndef Xform "{root}" (\n'
        f'    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]\n'
        f')\n{{\n'
        f'    bool physics:kinematicEnabled = 1\n'
        f'    bool physics:rigidBodyEnabled = 1\n\n'
    )
    out += '    def Scope "Looks"\n    {\n'
    for m in mats:
        out += material(m, COLORS[m]).replace("{root}", root)
    out += (
        '        def Material "knob_phys" (\n'
        '            prepend apiSchemas = ["PhysicsMaterialAPI"]\n'
        '        )\n        {\n'
        '            float physics:dynamicFriction = 1.2\n'
        '            float physics:restitution = 0\n'
        '            float physics:staticFriction = 1.3\n'
        '        }\n'
    )
    out += "    }\n\n"
    for box in cab.boxes():
        out += cube(box.name, box.center, box.size, box.material, root,
                    phys_mat="knob_phys" if box.name == "knob" else None) + "\n"
    out += "}\n"
    return out


def table_usda() -> str:
    rel = Path("..") / ".." / "droid-sim-evals" / "assets" / "table.usd"
    assert (DROID_SIM_EVALS_ASSETS / "table.usd").exists(), "droid-sim-evals/assets/table.usd missing"
    return HEADER.format(default="table") + f'''
def Xform "table" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
    prepend payload = @{rel.as_posix()}@
)
{{
    bool physics:kinematicEnabled = 1
    bool physics:rigidBodyEnabled = 1

    over "table_01"
    {{
        double3 xformOp:translate = (0, 0, 0)

        over "top" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            rel material:binding = </table/Looks/Oak> (
                bindMaterialAs = "weakerThanDescendants"
            )
        }}
    }}
}}
'''


def ycb_usda(name: str, mesh_prim: str, file: str, friction: tuple[float, float] | None = None) -> str:
    """``friction`` = (static, dynamic) binds a physics material to the mesh."""
    mat = ""
    bind = ""
    if friction is not None:
        mat = f'''
    def Material "grip_phys" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:dynamicFriction = {friction[1]}
        float physics:restitution = 0
        float physics:staticFriction = {friction[0]}
    }}
'''
        bind = f'''        rel material:binding:physics = </{name}/grip_phys>
'''
    return HEADER.format(default=name) + f'''
def Xform "{name}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
    prepend payload = @./ycb/{file}@
)
{{
    bool physics:kinematicEnabled = 0
    bool physics:rigidBodyEnabled = 1
{mat}
    over "{mesh_prim}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysxConvexHullCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI", "MaterialBindingAPI"]
    )
    {{
{bind}        uniform token physics:approximation = "convexDecomposition"
        bool physics:collisionEnabled = 1
    }}
}}
'''


def main() -> None:
    for cab in CABINETS:
        (HERE / f"{cab.name}.usda").write_text(cabinet_usda(cab))
    (HERE / "table.usda").write_text(table_usda())
    # The bowl gets the drawer experiment's knob friction (1.3 / 1.2): a pinch
    # on its thin rim must not slip on the first lift.
    (HERE / "bowl.usda").write_text(ycb_usda("bowl", "_24_bowl", "024_bowl.usd", friction=(1.3, 1.2)))
    (HERE / "banana.usda").write_text(ycb_usda("banana", "_11_banana", "011_banana.usd"))
    for f in sorted(HERE.glob("*.usda")):
        print(f"wrote {f.name} ({f.stat().st_size} B)")
    missing = [f for f in ("024_bowl.usd", "011_banana.usd") if not (HERE / "ycb" / f).exists()]
    if missing:
        print(f"NOTE: run fetch_assets.sh first; missing {missing}")


if __name__ == "__main__":
    main()
