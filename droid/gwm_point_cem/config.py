"""gwm_point_cem: shared constants for the 4-image 2x2 pointing + CEM experiment.

Standalone experiment folder. Import-only against the repo: nothing outside
droid/gwm_point_cem/ is modified, except the scene7_0.usd symlink in
droid-sim-evals/assets/ that the scene loader's filename convention requires
(same pattern as scene6).

World frame: robot base at origin. Table top z = 0.045141 (bin_geom).
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # gwm-wiser
ASSETS_DIR = REPO / "droid" / "droid-sim-evals" / "assets"
CAPTURE_DIR = HERE / "captures" / "scene7_0"
RESULTS = HERE / "results"

TABLE_TOP_Z = 0.045141201291582375
IMG_LIFT = 0.0015          # image quad sits 1.5 mm above the table top
IMG_SIZE = 0.15            # 15 x 15 cm printed photo

# cell name -> (x, y) world centre; 0.20 m pitch.
CELLS = {
    "dog":        (0.37, 0.00),
    "panda":      (0.57, 0.00),
    "banana":     (0.37, -0.20),
    "strawberry": (0.57, -0.20),
}
IMG_FILES = {
    "dog":        HERE / "assets/img/train_dog.png",
    "panda":      HERE / "assets/img/train_panda.png",
    "banana":     HERE / "assets/img/train_banana.png",
    "strawberry": HERE / "assets/img/test_strawberry.png",
}

# Hover search region around the 2x2 (x0, x1, y0, y1) and lattice steps.
REGION = (0.29, 0.66, -0.29, 0.09)
GRID_STEP = 0.02                     # heatmap sweep
LATTICE_STEP = 0.01                  # CEM samples snap to this lattice

# Candidate trajectory: home -> hover, linear in joint space, uniform timeline.
HOVER_H = 0.05
Z_HOVER = TABLE_TOP_Z + HOVER_H      # fingertip height target
TRAJ_DURATION = 6.0                  # s; identical for every candidate
TRAJ_STEPS = 31
GRIPPER = 1.0                        # closed at capture and throughout every candidate

SERVER_URL = "http://localhost:8901"
CAMS = ("external_cam", "external_cam_2")   # score both, fuse by mean

URDF = REPO / "droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf"
BAR_URDF = HERE / "assets/panda_bar.urdf"
