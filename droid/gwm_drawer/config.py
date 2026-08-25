"""gwm_drawer: shared constants for the drawer-opening selection experiment.

Standalone experiment folder: import-only
against the repo; nothing outside droid/gwm_drawer/ is modified except the
scene8_0.usd symlink in droid-sim-evals/assets/ that the scene loader's
filename convention requires.

Scene 8 = scene1 (bowl kept and re-placed, rubiks cube dropped) plus a
plain block, a YCB banana and three single-drawer cabinets of different
colour and size:

  - red     large, muted red, at +y
  - yellow  medium, yellow, in the middle, yawed -25 deg about z
  - blue    small, blue, at -y

Each drawer is a dynamic rigid body held by a prismatic joint to the world
(slide axis = the cabinet's local -x, toward the robot); the carcasses are
static colliders. Six candidate trajectories: three drawer pulls (reach the
knob, close, pull the drawer open) and three top-down object grasps (reach,
close, lift) that act as distractors. Six tasks: each drawer under two
referring expressions (colour, and size or position). World frame: robot base
at origin, table top z below.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # gwm-wiser
ASSETS_DIR = REPO / "droid" / "droid-sim-evals" / "assets"
CAPTURE_DIR = HERE / "captures" / "scene8_0"
RESULTS = HERE / "results"

TABLE_TOP_Z = 0.045141201291582375

SERVER_URL = "http://localhost:8901"
CAMS = ("external_cam", "external_cam_2")  # score both, fuse by mean
URDF = REPO / "droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf"

# ------------------------------------------------------------- cameras

# The two external cameras are re-posed at capture/execution time (the stock
# rig frames the whole room; this one frames the workspace): pulled in, aimed
# at the cabinet block, longer focal length, eye level with the drawers.
CAM_FOCAL = 2.6          # mm (stock 2.1); apertures stay 5.376 x 3.024
CAM_POSE = {
    "external_cam": {"pos": (0.08, 0.56, 0.44), "lookat": (0.55, -0.04, 0.32)},
    "external_cam_2": {"pos": (0.08, -0.56, 0.44), "lookat": (0.55, 0.04, 0.32)},
}


def cam_offset_quat(pos, lookat):
    """(w, x, y, z) OpenGL-convention camera rotation: -z toward lookat, world
    +z as up."""
    import numpy as np

    fwd = np.asarray(lookat, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    R = np.column_stack([right, true_up, -fwd])
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return (float(w), float(x), float(y), float(z))


def apply_camera_rig(scene_cfg):
    """Re-pose both external cameras on a parsed DROID env config."""
    for cam, spec in CAM_POSE.items():
        c = getattr(scene_cfg, cam)
        c.spawn.focal_length = CAM_FOCAL
        c.offset.pos = tuple(spec["pos"])
        c.offset.rot = cam_offset_quat(spec["pos"], spec["lookat"])
        c.offset.convention = "opengl"


# ---------------------------------------------------------------- cabinets

KNOB = {"depth": 0.030, "face": 0.026}  # block knob: x-protrusion, y/z edge


class Cabinet:
    """Box-built cabinet: static carcass + one inset drawer front.

    front_x, yc  world x of the closed drawer-front plane and world y of the
                 cabinet centre BEFORE the yaw; the cabinet is then yawed
                 about the vertical axis through its footprint centre
    width/height/depth  outer dimensions (y/z/x)
    t, plinth   panel thickness and solid base height (the drawer bay sits
                on top of the base)
    """

    def __init__(self, name, front_x, yc, width, height, depth,
                 t, plinth, tray_wall_h, pull, yaw=0.0):
        self.name = name
        self.front_x = front_x
        self.yc = yc
        self.width = width
        self.height = height
        self.depth = depth
        self.t = t
        self.plinth = plinth
        self.tray_wall_h = tray_wall_h
        self.pull = pull  # planned opening distance (m)
        self.yaw = yaw    # deg about z
        self.z0 = TABLE_TOP_Z
        self.bay_h = height - plinth - t

    # -- local frame: origin at the footprint centre (z = 0), x toward the back
    def frame(self):
        """(R, center): world = R @ local + center."""
        import numpy as np

        a = np.radians(self.yaw)
        R = np.array([[np.cos(a), -np.sin(a), 0.0],
                      [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        center = np.array([self.front_x + self.depth / 2, self.yc, 0.0])
        return R, center

    def to_world(self, p_local):
        import numpy as np

        R, c = self.frame()
        return (R @ np.asarray(p_local, float) + c).tolist()

    @property
    def front_x_local(self):
        return -self.depth / 2

    def bay_z(self):
        lo = self.z0 + self.plinth
        return lo, lo + self.bay_h

    def front_rect(self):
        """Inset drawer front, local: (z_lo, z_hi, y_lo, y_hi), 3 mm reveal."""
        z_lo, z_hi = self.bay_z()
        g = 0.003
        return (z_lo + g, z_hi - g,
                -self.width / 2 + self.t + g, self.width / 2 - self.t - g)

    def knob_local(self):
        z_lo, z_hi, y_lo, y_hi = self.front_rect()
        return (self.front_x_local - KNOB["depth"] / 2,
                (y_lo + y_hi) / 2, (z_lo + z_hi) / 2)

    def knob_center(self):
        """World centre of the block knob."""
        return self.to_world(self.knob_local())

    def slide_axis(self):
        """World unit vector the drawer moves along when opening."""
        R, _ = self.frame()
        return (R @ [-1.0, 0.0, 0.0]).tolist()


# Knob heights 0.40 / 0.31 / 0.24 m; sizes large / medium / small.
CAB_RED = Cabinet("cab_red", front_x=0.66, yc=0.33, width=0.30, height=0.45,
                  depth=0.18, t=0.016, plinth=0.275, tray_wall_h=0.08,
                  pull=0.14)
CAB_YELLOW = Cabinet("cab_yellow", front_x=0.70, yc=-0.02, width=0.24,
                     height=0.35, depth=0.14, t=0.014, plinth=0.190,
                     tray_wall_h=0.07, pull=0.10, yaw=-25.0)
CAB_BLUE = Cabinet("cab_blue", front_x=0.62, yc=-0.30, width=0.20, height=0.27,
                   depth=0.14, t=0.014, plinth=0.130, tray_wall_h=0.06,
                   pull=0.09)

# Big cabinet muted, small cabinet saturated.
COLORS = {
    "cab_red_body": (0.50, 0.17, 0.14), "cab_red_front": (0.62, 0.24, 0.20),
    "cab_yellow_body": (0.78, 0.58, 0.08), "cab_yellow_front": (0.92, 0.74, 0.16),
    "cab_blue_body": (0.07, 0.19, 0.62), "cab_blue_front": (0.12, 0.29, 0.80),
    "knob": (0.10, 0.10, 0.11),
    "tray": (0.55, 0.44, 0.30),
}

# The three drawer candidates, keyed by colour.
DRAWERS = {"red": CAB_RED, "yellow": CAB_YELLOW, "blue": CAB_BLUE}

# ---------------------------------------------------------------- objects

# Three stock objects on the table in front of the cabinets; each gets a
# top-down grasp-and-lift candidate that serves as a distractor trajectory.
# prim: the /World prim name in scene8_0.usd; xy: spawn position; straddle:
# world direction (deg from +x) along which the finger pads open; z_above:
# pads-mid height above the table top at the grasp; radial: grasp point
# offset from the settled object origin toward -x (bowl rim).
OBJECTS = {
    "block": {"prim": "basic_block", "xy": (0.369, 0.190), "straddle": 90.0,
              "z_above": 0.030, "radial": 0.0},
    "bowl": {"prim": "_24_bowl", "xy": (0.47, 0.05), "straddle": 0.0,
             "z_above": 0.045, "radial": 0.075},
    "banana": {"prim": "_11_banana", "xy": (0.33, -0.09), "straddle": 10.0,
               "z_above": 0.028, "radial": 0.0},
}
BLOCK_COLOR = (0.72, 0.72, 0.70)
GRASP_LIFT = 0.12  # m the object is lifted after the close

# ------------------------------------------------------------ scoring

# RAT window = WISER schedule x 3.0 from the trajectory start (8.85 s, frames
# at 0 / 1.65 / 3.45 / 5.25 / 7.05 / 8.85 s); timeline constants live in traj.py.
RAT_SCALE = 3.0
TASK_IMAGE = "current"

# ------------------------------------------------------------ tasks

# Six tasks: each drawer under two referring expressions — its colour, and a
# size (largest / smallest) or position (middle) attribute. Five phrasings per
# task are averaged (prompt ensemble) before any readout.
TASKS = {
    "red_color": {"target": "red", "phrases": [
        "open the drawer of the red cabinet",
        "open the red cabinet's drawer",
        "pull open the drawer of the red cabinet",
        "open the drawer in the red dresser",
        "slide out the red cabinet's drawer",
    ]},
    "red_size": {"target": "red", "phrases": [
        "open the drawer of the largest cabinet",
        "open the biggest cabinet's drawer",
        "pull open the drawer of the largest cabinet",
        "open the drawer in the biggest dresser",
        "slide out the largest cabinet's drawer",
    ]},
    "yellow_color": {"target": "yellow", "phrases": [
        "open the drawer of the yellow cabinet",
        "open the yellow cabinet's drawer",
        "pull open the drawer of the yellow cabinet",
        "open the drawer in the yellow dresser",
        "slide out the yellow cabinet's drawer",
    ]},
    "yellow_position": {"target": "yellow", "phrases": [
        "open the drawer of the middle cabinet",
        "open the middle cabinet's drawer",
        "pull open the drawer of the cabinet in the middle",
        "open the drawer in the center dresser",
        "slide out the middle cabinet's drawer",
    ]},
    "blue_color": {"target": "blue", "phrases": [
        "open the drawer of the blue cabinet",
        "open the blue cabinet's drawer",
        "pull open the drawer of the blue cabinet",
        "open the drawer in the blue dresser",
        "slide out the blue cabinet's drawer",
    ]},
    "blue_size": {"target": "blue", "phrases": [
        "open the drawer of the smallest cabinet",
        "open the smallest cabinet's drawer",
        "pull open the drawer of the smallest cabinet",
        "open the drawer in the smallest dresser",
        "slide out the smallest cabinet's drawer",
    ]},
}
