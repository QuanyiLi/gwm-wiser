"""Overlay-validate the shared FrankaRobotRenderer against droid-sim's external_cam.

Reads the H5 written by droid-sim-evals/capture_external_cam.py, renders the robot
at the captured joint state with the captured K / camera pose, and writes
side-by-side + alpha-blend comparison images. Run inside the gwm-wiser venv:

    cd /root/code/gwm/gwm-wiser && .venv/bin/python \
        /root/code/gwm/gwm-wiser/droid/gwm_tiptop/validate_renderer_overlay.py \
        --h5 /root/code/gwm/gwm-wiser/droid/droid-sim-evals/tiptop_assets/external_scene1_0.h5 \
        --out-dir /root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/overlays
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

GWM_WISER_ROOT = Path("/root/code/gwm/gwm-wiser")
sys.path.insert(0, str(GWM_WISER_ROOT))

from real_data_train.renderer.franka_renderer import FrankaRobotRenderer  # noqa: E402

URDF = {
    # droid-sim variant: Robotiq standoff of 18.2 mm, matching the sim USD
    # (gwm-wiser's default URDF welds it at 4 mm).
    "panda": Path(__file__).parent / "assets/panda_robotiq_droidsim.urdf",
    "fr3": GWM_WISER_ROOT / "real_data_train/data/assets/fr3_robotiq.urdf",
}
def cam2world_cv_from_h5(g) -> np.ndarray:
    """IsaacLab pos_w + quat_w_ros (wxyz, ROS optical = OpenCV axes) -> cam2world_cv.

    NOTE: FrankaRobotRenderer's `cam2world_gl` parameter, despite the name,
    consumes CV-axis cam2world matrices (x right, y down, z forward):
    gl_to_sapien_pose maps forward = column z — the same mapping
    cv_pose_to_sapien_pose reuses verbatim. Pass the CV matrix directly; do NOT
    apply a CV->GL axis flip.
    """
    pos = np.asarray(g["pos_w"])
    w, x, y, z = np.asarray(g["quat_w_ros"])
    cam2world_cv = np.eye(4)
    cam2world_cv[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    cam2world_cv[:3, 3] = pos
    return cam2world_cv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cam", default="external_cam", choices=["external_cam", "external_cam_2"])
    ap.add_argument("--arms", nargs="+", default=["panda"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.h5).stem

    with h5py.File(args.h5) as f:
        sim_rgb = np.asarray(f[f"{args.cam}/rgb"])[..., :3]
        K = np.asarray(f[f"{args.cam}/intrinsic_matrix"])
        c2w_gl = cam2world_cv_from_h5(f[args.cam])
        arm_qpos = np.asarray(f["arm_joint_pos"])[None]  # (1, 7)
        joint_names = [n.decode() for n in f["joint_names"][:]]
        joint_pos = np.asarray(f["joint_pos"])
        root_pos = np.asarray(f["robot_root_pos_w"])
        root_quat = np.asarray(f["robot_root_quat_w"])  # wxyz

    ji = {n: i for i, n in enumerate(joint_names)}
    drivers = np.array([[joint_pos[ji["finger_joint"]],
                         joint_pos[ji["right_outer_knuckle_joint"]]]])
    base_pose = np.concatenate([root_pos, root_quat])[None]  # (1, 7) xyz + wxyz
    h, w = sim_rgb.shape[:2]

    for arm in args.arms:
        renderer = FrankaRobotRenderer(URDF[arm], arm=arm)
        robot_rgb = renderer.render(
            arm_qpos, drivers, K, c2w_gl, width=w, height=h, base_pose=base_pose
        )[0]

        mask = robot_rgb.max(axis=-1) > 8
        blend = sim_rgb.copy()
        blend[mask] = (0.45 * sim_rgb[mask] + 0.55 * robot_rgb[mask]).astype(np.uint8)
        side = np.concatenate([sim_rgb, robot_rgb, blend], axis=1)

        Image.fromarray(blend).save(out_dir / f"{tag}_{args.cam}_{arm}_blend.png")
        Image.fromarray(side).save(out_dir / f"{tag}_{args.cam}_{arm}_side.png")
        cov = float(mask.mean())
        print(f"[{arm}] robot pixel coverage {cov:.3%} -> {out_dir}/{tag}_{args.cam}_{arm}_blend.png")


if __name__ == "__main__":
    main()
