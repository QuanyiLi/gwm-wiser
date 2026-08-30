"""gwm_push_cem: shared constants for the three-cube directional push study.

Standalone experiment folder. Import-only against the repo: nothing outside
droid/gwm_push_cem/ is modified, except the scene9_0.usd symlink in
droid-sim-evals/assets/ that the scene loader's filename convention requires.

World frame: robot base at origin, +x away from the base, +y to the robot's
left. Table top spans x in [0.199, 0.899], y in [-0.478, 0.522] at
z = 0.045141.

Scene: a clean table, the closed gripper parked at HOME_XY a centimetre and a
half above the table top, and three 47 mm cubes -- red, green and blue -- at
CUBE_D metres in front of / to the left of / to the right of the gripper.

Candidate: the fingertip slides in a straight line at constant height from
HOME_XY to a search-region endpoint, gripper closed throughout. CEM searches
the endpoint xy only; height and tool orientation are frozen at their home
values.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # gwm-wiser
ASSETS_DIR = REPO / "droid" / "droid-sim-evals" / "assets"

SCENE_ID = 9
SCENE_VARIANT = 0
RESULTS = HERE / "results"
CAPTURE_DIR = HERE / "captures" / f"scene{SCENE_ID}_{SCENE_VARIANT}"

TABLE_TOP_Z = 0.045141201291582375

# Fingertip reference point. The IK target is the midpoint of the two inner
# finger pads; with the gripper closed the lowest point of the hand sits
# PAD_DROP below that midpoint, so TIP_CLEAR is the true table clearance.
TIP_CLEAR = 0.015
PAD_DROP = 0.0121
Z_PUSH = TABLE_TOP_Z + TIP_CLEAR + PAD_DROP

HOME_XY = (0.46, 0.00)

CUBE_SIZE = 0.047
CUBE_D = 0.20                 # cube centre distance from the gripper home
CUBE_SPAWN_Z = TABLE_TOP_Z + CUBE_SIZE / 2 + 0.002
CUBE_MASS = 0.025
CUBE_FRICTION = 0.45

# tag -> (x, y) world centre of the cube it names.
CUBES = {
    "front": (HOME_XY[0] + CUBE_D, HOME_XY[1]),
    "left":  (HOME_XY[0], HOME_XY[1] + CUBE_D),
    "right": (HOME_XY[0], HOME_XY[1] - CUBE_D),
}
# Unit push direction each prompt is meant to produce.
DIRECTIONS = {"front": (1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
CUBE_PRIMS = {tag: f"cube_{tag}" for tag in CUBES}

# One colour per position; the prompt names the colour.
CUBE_RGB = {"front": (0.78, 0.10, 0.08), "left": (0.10, 0.55, 0.16),
            "right": (0.08, 0.24, 0.75)}
CUBE_COLOR_NAME = {"front": "red", "left": "green", "right": "blue"}
PROMPTS = {
    "front": "push the red cube",
    "left":  "push the green cube",
    "right": "push the blue cube",
}

# Endpoint search region (x0, x1, y0, y1): a square of half-width REGION_HALF
# centred on HOME_XY.
REGION_HALF = 0.24
REGION = (HOME_XY[0] - REGION_HALF, HOME_XY[0] + REGION_HALF,
          HOME_XY[1] - REGION_HALF, HOME_XY[1] + REGION_HALF)
LATTICE_STEP = 0.02           # endpoint resolution; the score map uses the same
GRID_STEP = LATTICE_STEP

# Candidate trajectory: straight line in the table plane, uniform timeline.
TRAJ_DURATION = 4.0                  # s; identical for every candidate
TRAJ_STEPS = 31
GRIPPER = 1.0                        # closed at capture and throughout

# Execution
EXEC_HZ = 15.0
HOLD_S = 1.5                         # settle time after the last waypoint

SERVER_URL = "http://localhost:8902"
CAMS = ("external_cam", "external_cam_2")   # score both, fuse by mean

URDF = REPO / "droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf"

# Plot colours, one per prompt.
COLORS = {"front": "#d62728", "left": "#2ca02c", "right": "#1f77b4"}
