"""Build the Panda + Robotiq **2F-140** robot model (URDF + cuRobo config).

Why this exists: the rig at `zhiwei` carries a Robotiq **2F-140**, but every
model in the stack is a **2F-85** -- cuTAMP ships only
`panda_robotiq_2f_85.urdf/.yml` and `get_robotiq_2f_85_gripper_spheres`, and
tiptop's `panda_robotiq` embodiment resolves to those. Planning a 2F-140 with a
2F-85 model is not cosmetic: the 2F-140 reaches substantially further past the
flange, so every grasp pose would drive the real fingers that much deeper than
the planner believes, and the collision spheres would clear volume the hardware
actually occupies. This generator produces the missing embodiment.

Sources (all already vendored inside the cuTAMP clone, nothing downloaded):

- arm half + DROID wrist-camera mount: `panda_robotiq_2f_85.urdf` -- reused
  verbatim down to `tool0`, so Panda kinematics stay single-sourced.
- gripper half: `robotiq_description/urdf/robotiq_2f_140{,_macro}.xacro`
  expanded by hand into the constants below (xacro is not a build dependency;
  every number is quoted from those files and cited inline).
- meshes: `robotiq_description/meshes/{visual,collision}/2f_140/*.stl`.

cuTAMP stays unforked (G-4/G-7) and `tiptop/` stays pristine (G-18): this
writes into our own tree and is consumed by `gwm_hardware.robot_2f140`.

The cuTAMP clone is gitignored and rebuilt by `install-cutamp.sh`, and the
emitted URDF carries absolute mesh paths, so the OUTPUT is machine-local (same
convention as `real_data_train/renderer/assets.py`) and this GENERATOR is the
versioned artifact. Re-run it after any cuTAMP reinstall:

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.build_2f140
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

PI = math.pi

# --- Provenance: cuTAMP asset locations -------------------------------------
CUTAMP_ASSETS = Path(__file__).resolve().parents[1] / "tiptop/cutamp/cutamp/robots/assets"
ARM_URDF = CUTAMP_ASSETS / "panda_robotiq_2f_85.urdf"
ARM_YML = CUTAMP_ASSETS / "panda_robotiq_2f_85.yml"
MESH_ROOT = CUTAMP_ASSETS / "robotiq_description/meshes"

# --- 2F-140 kinematic constants ---------------------------------------------
# Verbatim from robotiq_2f_140.xacro / robotiq_2f_140_macro.urdf.xacro.
# The macro's own comment calls robotiq_base_link "a temporary link to rotate
# the 2f-140 gripper to match the 2f-85" -- that pi/2 yaw is what puts the
# closing axis where the rest of the stack expects it, so it is load-bearing.
BASE_YAW = PI / 2                       # robotiq_base_joint rpy="0 0 ${pi/2}"
KNUCKLE_TILT = PI / 2 + 0.725           # ${pi / 2 + .725}
OUTER_KNUCKLE_XYZ = (0.0, 0.030601, 0.054905)   # sign flipped per side
INNER_KNUCKLE_Y = 0.0127
INNER_KNUCKLE_Z = 0.06142
OUTER_FINGER_XYZ = (0.0, 0.01821998610742, 0.0260018192872234)
INNER_FINGER_XYZ = (0.0, 0.0817554015893473, -0.0282203446692936)
INNER_FINGER_RPY = (-0.725, 0.0, 0.0)
PAD_XYZ = (0.0, 0.0457554015893473, -0.0272203446692936)
PAD_VISUAL_BOX = (0.027, 0.065, 0.0075)
PAD_COLLISION_BOX = (0.03, 0.07, 0.0075)

DRIVER_JOINT = "finger_joint"
DRIVER_LIMIT = (0.0, 0.7)               # 0 = fully open, 0.7 = fully closed
MIMIC_LIMIT = (-0.8757, 0.8757)
RIGHT_KNUCKLE_LIMIT = (-0.725, 0.725)

# Inertials, verbatim from the xacro (mass kg, inertia kg m^2)
INERTIALS = {
    "robotiq_140_base_link": dict(
        origin=(8.625e-08, -4.6583e-06, 0.03145), mass=0.22652,
        i=dict(ixx="0.00020005", ixy="-4.2442E-10", ixz="-2.9069E-10",
               iyy="0.00017832", iyz="-3.4402E-08", izz="0.00013478")),
    "outer_knuckle": dict(
        origin=(-0.000200000000003065, 0.0199435877845359, 0.0292245259211331),
        mass=0.00853198276973456,
        i=dict(ixx="2.89328108496468E-06", ixy="-1.57935047237397E-19",
               ixz="-1.93980378593255E-19", iyy="1.86719750325683E-06",
               iyz="-1.21858577871576E-06", izz="1.21905238907251E-06")),
    "outer_finger": dict(
        origin=(0.00030115855001899, 0.0373907951953854, -0.0208027427000385),
        mass=0.022614240507152,
        i=dict(ixx="1.52518312458174E-05", ixy="9.76583423954399E-10",
               ixz="-5.43838577022588E-10", iyy="6.17694243867776E-06",
               iyz="6.78636130740228E-06", izz="1.16494917907219E-05")),
    "inner_knuckle": dict(
        origin=(0.000123011831763771, 0.0507850843201817, 0.00103968640075166),
        mass=0.0271177346495152,
        i=dict(ixx="2.61910379223783E-05", ixy="-2.43616858946494E-07",
               ixz="-6.37789906117123E-09", iyy="2.8270243746167E-06",
               iyz="-5.37200748039765E-07", izz="2.83695868220296E-05")),
    "inner_finger": dict(
        origin=(0.000299999999999317, 0.0160078233491243, -0.0136945669206257),
        mass=0.0104003125914103,
        i=dict(ixx="2.71909453810972E-06", ixy="1.35402465472579E-21",
               ixz="-7.1817349065269E-22", iyy="7.69100314106116E-07",
               iyz="6.74715432769696E-07", izz="2.30315190420171E-06")),
}

# Links kept from the 2F-85 URDF: the arm and the flange chain.
#
# The 2F-85 model's `hand_camera_mount_part_1/2` are deliberately NOT carried
# over. They are DROID's ZED bracket, and expressed in grasp_frame they sit
# 318 mm and 279 mm off the approach axis -- a third of a metre, which cannot be
# any wrist bracket. This rig's camera is a RealSense on a different mount, and
# the hand-eye calibration puts its optical centre 75 mm off the axis. Keeping
# meshes that are wrong by 240 mm would only invite someone to trust them; the
# camera's collision volume is carried by calibration-derived spheres in
# `build_2f140_cfg.camera_mount_spheres` instead.
ARM_LINKS = ["base_link", *[f"panda_link{i}" for i in range(9)],
             "flange", "tool0"]
ARM_JOINTS = ["panda_fixed", *[f"panda_joint{i}" for i in range(1, 9)],
              "panda_link8-flange", "flange-tool0"]


def _resolve_arm_mesh(rel: str) -> str:
    """Absolutise a mesh path from the stock 2F-85 URDF.

    Its arm meshes are written as bare `meshes/visual/linkN.dae`, which resolve
    against neither the assets dir nor `.../robots/` -- they dangle upstream
    too (yourdfpy warns on the stock file as well). They actually ship inside
    cuRobo, so look there first and fall back to the cuTAMP tree for the
    camera-mount and gripper meshes, which do resolve.
    """
    from curobo.util_file import get_assets_path

    candidates = [
        Path(get_assets_path()) / "robot/franka_description" / rel,
        CUTAMP_ASSETS.parent / rel,
        CUTAMP_ASSETS / rel,
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    raise FileNotFoundError(f"cannot resolve mesh {rel!r}; tried {candidates}")


def _fmt(v):
    return " ".join(f"{x:.12g}" for x in v)


def _sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def _add_inertial(link, spec):
    if spec is None:
        # cuRobo/yourdfpy tolerate massless links, but a token inertial keeps
        # the URDF valid for any downstream consumer (e.g. physics sims).
        inertial = _sub(link, "inertial")
        _sub(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        _sub(inertial, "mass", value="0.001")
        _sub(inertial, "inertia", ixx="1e-7", ixy="0", ixz="0",
             iyy="1e-7", iyz="0", izz="1e-7")
        return
    inertial = _sub(link, "inertial")
    _sub(inertial, "origin", xyz=_fmt(spec["origin"]), rpy="0 0 0")
    _sub(inertial, "mass", value=spec["mass"])
    _sub(inertial, "inertia", **spec["i"])


def _add_mesh_link(root, name, stl, inertial_key, color="0.1 0.1 0.1 1"):
    link = _sub(root, "link", name=name)
    _add_inertial(link, INERTIALS.get(inertial_key))
    for kind in ("visual", "collision"):
        el = _sub(link, kind)
        _sub(el, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = _sub(el, "geometry")
        _sub(geom, "mesh", filename=str(MESH_ROOT / kind / "2f_140" / stl))
        if kind == "visual":
            mat = _sub(el, "material", name="")
            _sub(mat, "color", rgba=color)
    return link


def _add_box_link(root, name):
    link = _sub(root, "link", name=name)
    _add_inertial(link, None)
    for kind, box in (("visual", PAD_VISUAL_BOX), ("collision", PAD_COLLISION_BOX)):
        el = _sub(link, kind)
        _sub(el, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = _sub(el, "geometry")
        _sub(geom, "box", size=_fmt(box))
        if kind == "visual":
            mat = _sub(el, "material", name="")
            _sub(mat, "color", rgba="0.9 0.9 0.9 1")


def _add_joint(root, name, jtype, parent, child, xyz, rpy,
               axis=None, limit=None, mimic=None):
    j = _sub(root, "joint", name=name, type=jtype)
    _sub(j, "parent", link=parent)
    _sub(j, "child", link=child)
    _sub(j, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
    if axis:
        _sub(j, "axis", xyz=_fmt(axis))
    if limit:
        _sub(j, "limit", lower=limit[0], upper=limit[1], velocity=2.0, effort=1000)
    if mimic:
        _sub(j, "mimic", joint=mimic[0], multiplier=mimic[1], offset=0)
    return j


def build_urdf(flange_offset_m: float, out_path: Path) -> Path:
    src = ET.parse(ARM_URDF).getroot()
    keep_links = {l.get("name"): l for l in src.findall("link") if l.get("name") in ARM_LINKS}
    keep_joints = {j.get("name"): j for j in src.findall("joint") if j.get("name") in ARM_JOINTS}
    missing = (set(ARM_LINKS) - keep_links.keys()) | (set(ARM_JOINTS) - keep_joints.keys())
    if missing:
        raise RuntimeError(f"2F-85 source URDF is missing expected arm elements: {sorted(missing)}")

    root = ET.Element("robot", {"name": "panda_robotiq_2f_140"})
    for mat in src.findall("material"):
        root.append(mat)
    for name in ARM_LINKS:
        link = keep_links[name]
        for mesh in link.iter("mesh"):          # relative -> absolute
            fn = mesh.get("filename")
            if fn and not fn.startswith("/"):
                mesh.set("filename", _resolve_arm_mesh(fn))
        root.append(link)
    for name in ARM_JOINTS:
        root.append(keep_joints[name])

    # ---- 2F-140 gripper ----
    _sub(root, "link", name="robotiq_base_link")
    _add_joint(root, "robotiq_base_joint", "fixed", "tool0", "robotiq_base_link",
               (0, 0, 0), (0, 0, BASE_YAW))
    _add_mesh_link(root, "robotiq_140_base_link",
                   "robotiq_2f_140_base_link.stl", "robotiq_140_base_link")
    # The <*origin> the macro takes from its caller. Direct flange mount on this
    # rig => 0 by default; --flange-offset carries any coupling plate.
    _add_joint(root, "robotiq_140_base_joint", "fixed",
               "robotiq_base_link", "robotiq_140_base_link",
               (0, 0, flange_offset_m), (0, 0, 0))

    for side, reflect in (("left", 1.0), ("right", -1.0)):
        _add_mesh_link(root, f"{side}_outer_knuckle",
                       "robotiq_2f_140_outer_knuckle.stl", "outer_knuckle",
                       color="0.792156862745098 0.819607843137255 0.933333333333333 1")
        _add_mesh_link(root, f"{side}_outer_finger",
                       "robotiq_2f_140_outer_finger.stl", "outer_finger")
        _add_mesh_link(root, f"{side}_inner_knuckle",
                       "robotiq_2f_140_inner_knuckle.stl", "inner_knuckle")
        _add_mesh_link(root, f"{side}_inner_finger",
                       "robotiq_2f_140_inner_finger.stl", "inner_finger")
        _add_box_link(root, f"{side}_inner_finger_pad")

    # Driver joint (left) and its mimicking twin (right)
    _add_joint(root, DRIVER_JOINT, "revolute", "robotiq_140_base_link",
               "left_outer_knuckle",
               (OUTER_KNUCKLE_XYZ[0], -OUTER_KNUCKLE_XYZ[1], OUTER_KNUCKLE_XYZ[2]),
               (KNUCKLE_TILT, 0, 0), axis=(-1, 0, 0), limit=DRIVER_LIMIT)
    _add_joint(root, "right_outer_knuckle_joint", "revolute", "robotiq_140_base_link",
               "right_outer_knuckle", OUTER_KNUCKLE_XYZ, (KNUCKLE_TILT, 0, PI),
               axis=(1, 0, 0), limit=RIGHT_KNUCKLE_LIMIT,
               mimic=(DRIVER_JOINT, -1))

    for side, reflect in (("left", 1.0), ("right", -1.0)):
        _add_joint(root, f"{side}_outer_finger_joint", "fixed",
                   f"{side}_outer_knuckle", f"{side}_outer_finger",
                   OUTER_FINGER_XYZ, (0, 0, 0), axis=(1, 0, 0))
        _add_joint(root, f"{side}_inner_knuckle_joint", "revolute",
                   "robotiq_140_base_link", f"{side}_inner_knuckle",
                   (0, reflect * -INNER_KNUCKLE_Y, INNER_KNUCKLE_Z),
                   (KNUCKLE_TILT, 0, (reflect - 1) * PI / 2),
                   axis=(1, 0, 0), limit=MIMIC_LIMIT, mimic=(DRIVER_JOINT, -1))
        _add_joint(root, f"{side}_inner_finger_joint", "revolute",
                   f"{side}_outer_finger", f"{side}_inner_finger",
                   INNER_FINGER_XYZ, INNER_FINGER_RPY,
                   axis=(1, 0, 0), limit=MIMIC_LIMIT, mimic=(DRIVER_JOINT, 1))
        _add_joint(root, f"{side}_inner_finger_pad_joint", "fixed",
                   f"{side}_inner_finger", f"{side}_inner_finger_pad",
                   PAD_XYZ, (0, 0, 0))

    # TCP frames. z is filled in by calibrate_tcp() below; 0 is a placeholder.
    #
    # The -90 deg yaw is NOT cosmetic. `grasp_frame` is cuRobo's ee_link -- the
    # frame the planner is told to place at an M2T2 grasp pose -- and
    # `m2t2_to_tiptop_transform()` is gripper-agnostic, so this frame's
    # ORIENTATION convention has to match the 2F-85 the pipeline was tuned on.
    # Measured at equal joint angles, without this yaw the two grasp_frames
    # differ by exactly 90 deg about the approach axis:
    #
    #   2F-85  grasp_frame in panda_link8:  x=[0,1,0]   y=[-1,0,0]  z=[0,0,1]
    #   2F-140 without this yaw:            x=[-1,0,0]  y=[0,-1,0]  z=[0,0,1]
    #
    # The closing axes still coincide at equal q, so the models look fine
    # side by side -- the error only shows up once the planner is asked to
    # satisfy a grasp pose, where it rotates the wrist 90 deg and closes the
    # fingers across the wrong axis of the object. Same class of failure as the
    # stock Franka ready pose's q7 = pi/4, which exists to cancel the Franka
    # Hand's flange offset and becomes the error once a Robotiq is bolted on
    # square (found on the controller side, 2026-08-18).
    _sub(root, "link", name="gripper_frame")
    _add_joint(root, "gripper_joint", "fixed", "robotiq_140_base_link",
               "gripper_frame", (0, 0, 0), (0, 0, -PI / 2))
    _sub(root, "link", name="grasp_frame")
    _add_joint(root, "grasp_joint", "fixed", "gripper_frame", "grasp_frame",
               (0, 0, 0), (0, 0, 0))

    ET.indent(root, "  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def calibrate_tcp(urdf_path: Path, driver_angle: float) -> float:
    """Measure the fingertip plane by FK+mesh and write it into grasp_joint.

    tiptop and M2T2 express every grasp relative to `grasp_frame`, so its
    offset IS the grasp convention -- get it wrong and every grasp is
    systematically too deep or too shallow. Rather than guess, the convention
    is read off the 2F-85 that the whole stack was tuned against:

        2F-85, fully open:  fingertip plane 149.3 mm, grasp_frame 150.0 mm

    i.e. **grasp_frame sits at the fingertip plane with the gripper open**.
    Applying the same rule to the 2F-140 measures 212.0 mm -- 62 mm further out
    than the 2F-85's 150 mm, which is exactly the error a 2F-85 model would
    have injected into every 2F-140 grasp.

    The plane is taken from collision-mesh vertices, not link origins: on both
    grippers the `*_finger_tip`/`*_inner_finger_pad` origins sit well short of
    the physical tip.
    """
    import numpy as np
    import trimesh
    from yourdfpy import URDF

    robot = URDF.load(str(urdf_path), load_meshes=True, build_scene_graph=True)
    robot.update_cfg({DRIVER_JOINT: driver_angle})

    finger_links = [f"{s}_{p}" for s in ("left", "right")
                    for p in ("outer_finger", "inner_finger", "inner_finger_pad")]
    z_tip = -np.inf
    pads = []
    for name in finger_links:
        T = robot.get_transform(name, "robotiq_140_base_link")
        for col in robot.link_map[name].collisions:
            geom = col.geometry
            if geom.mesh is not None:
                mesh = trimesh.load(robot._filename_handler(geom.mesh.filename),
                                    force="mesh")
                if geom.mesh.scale is not None:
                    mesh.apply_scale(geom.mesh.scale)
                verts = mesh.vertices
            elif geom.box is not None:
                verts = trimesh.creation.box(extents=geom.box.size).vertices
            else:
                continue
            origin = col.origin if col.origin is not None else np.eye(4)
            world = (T @ origin @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
            z_tip = max(z_tip, float(world[:, 2].max()))
        if name.endswith("inner_finger_pad"):
            pads.append(T[:3, 3])
    span = float(abs(pads[0][1] - pads[1][1])) if len(pads) == 2 else float("nan")

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for joint in root.findall("joint"):
        if joint.get("name") == "grasp_joint":
            joint.find("origin").set("xyz", f"0 0 {z_tip:.9g}")
    ET.indent(root, "  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"  TCP: fingertip plane z = {z_tip * 1000:.1f} mm from "
          f"robotiq_140_base_link at driver={driver_angle} rad "
          f"(2F-85 reference: 150.0 mm); pad separation {span * 1000:.1f} mm")
    return z_tip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flange-offset", type=float, default=0.0,
                    help="metres between the flange face and the gripper base "
                         "(the 'standoff'). 0 = gripper bolted straight to the "
                         "flange, which is this rig. Recover it from the "
                         "renderer overlay gate if the model sits proud.")
    ap.add_argument("--tcp-driver-angle", type=float, default=0.0,
                    help="driver angle the TCP is measured at; 0 = fully open")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "assets")
    args = ap.parse_args()

    if not ARM_URDF.exists():
        raise SystemExit(f"cuTAMP assets not found at {CUTAMP_ASSETS}. "
                         "Run `pixi run install-cutamp` in droid/tiptop first.")

    urdf = args.out_dir / "panda_robotiq_2f_140.urdf"
    print(f"building {urdf}")
    build_urdf(args.flange_offset, urdf)
    calibrate_tcp(urdf, args.tcp_driver_angle)
    print("  done")


if __name__ == "__main__":
    main()
