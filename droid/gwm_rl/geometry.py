"""Scene geometry for the pick-up-bowl environment — scene8 of `gwm_drawer`
rebuilt as static obstacles plus three free objects.

Pure python (no Isaac, no torch), shared by the asset generator
(`assets/make_assets.py`), the env config (`scene.py`) and the GWM capture
hook (`capture.py`). World frame: robot base at the origin, +x away from the
base, +z up; every number below is in metres in that frame, and the cabinet
dimensions, colours and camera rig are copied from `gwm_drawer/config.py` so
the photo this scene produces matches the one the drawer experiment scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DROID_DIR = HERE.parent
DROID_SIM_EVALS_ASSETS = DROID_DIR / "droid-sim-evals" / "assets"
ASSETS = HERE / "assets"

#: Franka + Robotiq 2F-85 as DROID ships it (13 DoF, 8 actuated). The same
#: gripper the GWM renderer's URDF carries; the M3 2F-140 is not usable here.
ROBOT_USD = DROID_SIM_EVALS_ASSETS / "franka_robotiq_2f_85_flattened.usd"
HDRI = DROID_SIM_EVALS_ASSETS / "backgrounds" / "brown_photostudio_01_4k.hdr"

# ------------------------------------------------------------------ table --

TABLE_TOP_Z = 0.045141201291582375
#: `/World/table` translate in scene1/scene8; with the payload's own offset
#: zeroed the slab's top face lands at z = TABLE_TOP_Z.
TABLE_POS = (0.19858086399571107, 0.022072189459978908, TABLE_TOP_Z)
#: Slab footprint in world x/y (0.70 x 1.00 m, 3 cm thick).
TABLE_X = (0.199, 0.899)
TABLE_Y = (-0.478, 0.522)
#: The table's legs are 0.67 m long under a 3 cm slab; the floor's top face
#: sits 2 cm below their ends (a kinematic leg overlapping the static floor
#: would give a free object wedged in the corner two infinite-mass pushers).
FLOOR_Z = TABLE_TOP_Z - 0.03 - 0.67 - 0.02

# --------------------------------------------------------------- cabinets --

KNOB = {"depth": 0.030, "face": 0.026}
SHRINK = 0.0004  # inset between touching slabs (kills coplanar faces)
FRONT_T = 0.018  # drawer front panel thickness


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in the cabinet's local frame (centre, full size)."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    material: str


class Cabinet:
    """Box-built cabinet: carcass + one inset drawer front, all static here.

    Local frame: origin at the footprint centre on the world ground plane
    (z is world z), x toward the cabinet's back; the cabinet is yawed about
    the vertical through that origin. `front_x`/`yc` are the world x of the
    closed drawer-front plane and the world y of the centre before the yaw.
    """

    def __init__(self, name, front_x, yc, width, height, depth, t, plinth,
                 tray_wall_h, pull, yaw=0.0):
        self.name = name
        self.front_x = front_x
        self.yc = yc
        self.width = width
        self.height = height
        self.depth = depth
        self.t = t
        self.plinth = plinth
        self.tray_wall_h = tray_wall_h
        self.pull = pull
        self.yaw = yaw
        self.z0 = TABLE_TOP_Z
        self.bay_h = height - plinth - t

    # -- frame
    @property
    def world_pos(self) -> tuple[float, float, float]:
        return (self.front_x + self.depth / 2, self.yc, 0.0)

    @property
    def world_quat(self) -> tuple[float, float, float, float]:
        a = math.radians(self.yaw) / 2
        return (math.cos(a), 0.0, 0.0, math.sin(a))

    @property
    def front_x_local(self) -> float:
        return -self.depth / 2

    def bay_z(self) -> tuple[float, float]:
        lo = self.z0 + self.plinth
        return lo, lo + self.bay_h

    def front_rect(self) -> tuple[float, float, float, float]:
        """Inset drawer front, local (z_lo, z_hi, y_lo, y_hi), 3 mm reveal."""
        z_lo, z_hi = self.bay_z()
        g = 0.003
        return (z_lo + g, z_hi - g,
                -self.width / 2 + self.t + g, self.width / 2 - self.t - g)

    def knob_center_world(self) -> tuple[float, float, float]:
        z_lo, z_hi, y_lo, y_hi = self.front_rect()
        local = (self.front_x_local - KNOB["depth"] / 2, (y_lo + y_hi) / 2, (z_lo + z_hi) / 2)
        a = math.radians(self.yaw)
        cx, cy, _ = self.world_pos
        return (cx + math.cos(a) * local[0] - math.sin(a) * local[1],
                cy + math.sin(a) * local[0] + math.cos(a) * local[1], local[2])

    # -- geometry
    def boxes(self) -> list[Box]:
        """Every collider of the cabinet in its local frame: the carcass
        (`make_scene8.carcass`) and the drawer parts (`make_scene8.drawer`)
        re-expressed in the carcass frame with the drawer closed."""
        n = self.name
        z0, h, w, d, t = self.z0, self.height, self.width, self.depth, self.t
        body = f"{n}_body"
        out = [
            Box("wall_left", (0, -(w - t) / 2, z0 + (h - t) / 2), (d, t, h - t), body),
            Box("wall_right", (0, (w - t) / 2, z0 + (h - t) / 2), (d, t, h - t), body),
            Box("top", (0, 0, z0 + h - t / 2), (d - SHRINK, w - SHRINK, t), body),
            Box("back", (d / 2 - t / 2, 0, z0 + (h - t) / 2), (t, w - 2 * t - SHRINK, h - t - SHRINK), body),
            Box("plinth", (0.002, 0, z0 + self.plinth / 2), (d - 0.004, w - 2 * t - SHRINK, self.plinth - SHRINK), body),
        ]
        # drawer, closed, in the carcass frame
        z_lo, z_hi, y_lo, y_hi = self.front_rect()
        z_fc, fw, fh = (z_lo + z_hi) / 2, y_hi - y_lo, z_hi - z_lo
        bay_lo, _ = self.bay_z()
        ox = self.front_x_local
        td = d - t - FRONT_T - 0.006
        tw = w - 2 * t - 0.006
        zb = bay_lo + 0.008
        wh = self.tray_wall_h
        zw = bay_lo + 0.013 + wh / 2
        out += [
            Box("front", (ox + FRONT_T / 2, 0, z_fc), (FRONT_T, fw, fh), f"{n}_front"),
            Box("knob", (ox - KNOB["depth"] / 2, 0, z_fc), (KNOB["depth"], KNOB["face"], KNOB["face"]), "knob"),
            Box("tray_bottom", (ox + FRONT_T + td / 2, 0, zb), (td, tw, 0.010), "tray"),
            Box("tray_l", (ox + FRONT_T + td / 2, -(tw - 0.010) / 2, zw), (td, 0.010, wh), "tray"),
            Box("tray_r", (ox + FRONT_T + td / 2, (tw - 0.010) / 2, zw), (td, 0.010, wh), "tray"),
            Box("tray_back", (ox + FRONT_T + td - 0.005, 0, zw), (0.010, tw - 0.020, wh), "tray"),
        ]
        return out


# Knob heights 0.40 / 0.31 / 0.24 m; sizes large / medium / small.
CAB_RED = Cabinet("cab_red", front_x=0.66, yc=0.33, width=0.30, height=0.45,
                  depth=0.18, t=0.016, plinth=0.275, tray_wall_h=0.08, pull=0.14)
CAB_YELLOW = Cabinet("cab_yellow", front_x=0.70, yc=-0.02, width=0.24,
                     height=0.35, depth=0.14, t=0.014, plinth=0.190,
                     tray_wall_h=0.07, pull=0.10, yaw=-25.0)
CAB_BLUE = Cabinet("cab_blue", front_x=0.62, yc=-0.30, width=0.20, height=0.27,
                   depth=0.14, t=0.014, plinth=0.130, tray_wall_h=0.06, pull=0.09)
CABINETS = (CAB_RED, CAB_YELLOW, CAB_BLUE)

# Big cabinet muted, small cabinet saturated (gwm_drawer/config.py).
COLORS = {
    "cab_red_body": (0.50, 0.17, 0.14), "cab_red_front": (0.62, 0.24, 0.20),
    "cab_yellow_body": (0.78, 0.58, 0.08), "cab_yellow_front": (0.92, 0.74, 0.16),
    "cab_blue_body": (0.07, 0.19, 0.62), "cab_blue_front": (0.12, 0.29, 0.80),
    "knob": (0.10, 0.10, 0.11),
    "tray": (0.55, 0.44, 0.30),
}

# ---------------------------------------------------------------- objects --

#: Settled poses from `gwm_drawer/captures/scene8_0/objects.json` (100 settle
#: steps after spawn), so the objects are reset onto the table rather than
#: dropped onto it.
BOWL_POS = (0.47, 0.05, 0.07386092096567154)
BOWL_QUAT = (0.7071078419685364, -0.7071057558059692, 0.0, 0.0)  # -90 deg about x
BANANA_POS = (0.32999786734580994, -0.09000014513731003, 0.06295853108167648)
BANANA_QUAT = (0.43785274028778076, 0.4704599976539612, 0.5732700824737549, 0.5082458257675171)
#: The Isaac `basic_block` is a 4.7 cm cube whose physics root sits 3.92 cm
#: above the mesh centre; here it is a plain cube, so the root *is* the centre.
BLOCK_SIZE = 0.047
BLOCK_POS = (0.369, 0.19, 0.10765181481838226 - 0.039214913)
BLOCK_MASS = 0.02
BLOCK_COLOR = (0.72, 0.72, 0.70)

#: YCB 024_bowl, measured off the mesh (`assets/ycb/024_bowl.usd`, cm -> m):
#: a body of revolution about the asset's y axis, 16.1 cm across, 5.5 cm
#: tall; the wide end (the rim, mean vertex radius 0.077) is at y = -0.0275
#: and the base (radius 0.034) at y = +0.0275. Under BOWL_QUAT the asset's
#: -y points at world +z, i.e. the rim is on top.
BOWL_AXIS_BODY = (0.0, -1.0, 0.0)  # body-frame unit vector from centre to the rim plane
BOWL_HALF_HEIGHT = 0.0275
BOWL_RIM_RADIUS = 0.077
BOWL_OUTER_RADIUS = 0.0806

# --------------------------------------------------------------- cameras --

#: The drawer experiment's external rig: pulled in, aimed at the cabinet
#: block, focal 2.6 mm (stock 2.1), apertures 5.376 x 3.024, 1280 x 720.
CAM_FOCAL = 2.6
CAM_POSE = {
    "external_cam": {"pos": (0.08, 0.56, 0.44), "lookat": (0.55, -0.04, 0.32)},
    "external_cam_2": {"pos": (0.08, -0.56, 0.44), "lookat": (0.55, 0.04, 0.32)},
}
CAM_RES = (1280, 720)


def cam_offset_quat(pos, lookat):
    """(w, x, y, z) OpenGL-convention camera rotation: -z toward lookat, world +z up."""
    fwd = [lookat[i] - pos[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in fwd))
    fwd = [v / n for v in fwd]
    up = (0.0, 0.0, 1.0)
    right = [fwd[1] * up[2] - fwd[2] * up[1], fwd[2] * up[0] - fwd[0] * up[2], fwd[0] * up[1] - fwd[1] * up[0]]
    n = math.sqrt(sum(v * v for v in right))
    right = [v / n for v in right]
    true_up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2], right[0] * fwd[1] - right[1] * fwd[0]]
    # columns: right, true_up, -fwd
    R = [[right[0], true_up[0], -fwd[0]],
         [right[1], true_up[1], -fwd[1]],
         [right[2], true_up[2], -fwd[2]]]
    w = math.sqrt(max(0.0, 1.0 + R[0][0] + R[1][1] + R[2][2])) / 2
    x = (R[2][1] - R[1][2]) / (4 * w)
    y = (R[0][2] - R[2][0]) / (4 * w)
    z = (R[1][0] - R[0][1]) / (4 * w)
    return (float(w), float(x), float(y), float(z))


# ------------------------------------------------------------- workspace --

#: The box the policy's absolute target pose (macro action) is mapped into,
#: base frame. x reaches the cabinet fronts (0.62-0.70) but not behind them;
#: z starts 3 cm above the slab so a target cannot be inside the table.
WORKSPACE = {
    "x": (0.20, 0.75),
    "y": (-0.45, 0.45),
    "z": (TABLE_TOP_Z + 0.03, 0.60),
    "yaw": (-math.pi / 2, math.pi / 2),
}


# ----------------------------------------------------------- scene names --
#
# Kept here (no Isaac import) so the task math can name things without
# booting Kit.

ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_JOINT_NAME = "finger_joint"
#: All six revolute gripper joints; only `finger_joint` is commanded.
GRIPPER_JOINT_NAMES = (
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
)
FLANGE_BODY_NAME = "panda_link8"
#: The two bodies whose pads touch the object, (left, right).
FINGER_BODY_NAMES = ("left_inner_finger", "right_inner_finger")
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = math.pi / 4

#: Scene-entity names of the bodies the collision cost is charged against, in
#: the order the task sums them; each carries a `contact_<name>` sensor.
OBSTACLE_NAMES = ("table", "cab_red", "cab_yellow", "cab_blue", "block", "banana")
#: Free objects other than the bowl, whose displacement is a second cost.
DISTRACTOR_NAMES = ("block", "banana")
