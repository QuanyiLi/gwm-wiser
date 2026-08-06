"""FrankaRobotRenderer: state -> robot-only RGB (ADR-0017).

One SAPIEN scene holding one welded arm+gripper articulation renders the
robot alone against black. Camera intrinsics/extrinsics/resolution are
injected per call; the URDF (Panda vs FR3 rig) is chosen per source. The
same class serves offline training-data generation and inference-time
candidate scoring — train/inference render homology is a hard requirement.

Kinematics only: qpos is set directly (no physics stepping), and the Robotiq
2F-85 linkage is reconstructed from its driver values through the mimic map
(the ManiSkill URDF ships six independent revolute joints).
"""

import numpy as np

# Robotiq 2F-85 mimic map: joint name -> (side_driver, sign). Drivers are the
# two outer knuckle joints; DROID's continuous gripper position maps onto
# them via DRIVER_RANGE_RAD.
MIMIC_MAP = {
    "left_outer_knuckle_joint": ("left", 1.0),
    "left_inner_knuckle_joint": ("left", 1.0),
    "left_inner_finger_joint": ("left", -1.0),
    "right_outer_knuckle_joint": ("right", 1.0),
    "right_inner_knuckle_joint": ("right", 1.0),
    "right_inner_finger_joint": ("right", -1.0),
}
DRIVER_RANGE_RAD = 0.8   # closed driver angle for gripper_pos = 1.0

ARM_JOINTS = {
    "panda": [f"panda_joint{i}" for i in range(1, 8)],
    "fr3": [f"fr3_joint{i}" for i in range(1, 8)],
}


def gl_to_sapien_pose(cam2world_gl: np.ndarray):
    """OpenGL camera-to-world (x right, y up, -z forward) -> SAPIEN pose.

    Mapping verified empirically (2026-08-06) against MolmoBot's paired
    extrinsic_cv/cam2world_gl: after set_entity_pose with this rotation,
    camera.get_extrinsic_matrix() reproduces extrinsic_cv to 0 and a sphere
    placed 1 m along the CV forward axis renders at the principal point.
    """
    import sapien

    m = np.asarray(cam2world_gl, dtype=np.float64)
    r = np.empty((3, 3))
    r[:, 0] = m[:3, 2]
    r[:, 1] = -m[:3, 0]
    r[:, 2] = -m[:3, 1]
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = m[:3, 3]
    return sapien.Pose(t)


def cv_pose_to_sapien_pose(xyzrpy: np.ndarray):
    """Camera pose (x,y,z,roll,pitch,yaw) in base frame, OpenCV camera axes
    (x right, y down, z forward) -> SAPIEN pose."""
    import sapien
    from scipy.spatial.transform import Rotation  # noqa: F401

    raise NotImplementedError(
        "DROID extrinsics are zero-filled in the release; this entry point "
        "lands with the DROID camera-recovery gate."
    )


def fit_arm_mount(renderer, arm_qpos, gripper, tcp_local,
                  max_residual_m: float = 0.002):
    """Recover the arm-mount translation from recorded TCP poses.

    MolmoBot's robot_base_pose is the platform frame, not the arm root; the
    arm root sits at a fixed translation above it. Solving
    ``tcp[t] = m + FK_flange(qpos[t]) + R_flange(qpos[t]) @ [0, 0, dz]``
    for (m, dz) by least squares recovers the mount and doubles as a
    kinematics gate: a residual above max_residual_m means our welded URDF
    does not reproduce the source kinematics for this episode.

    Returns (mount_translation (3,), tcp_dz, rms_residual_m); raises
    RuntimeError past the gate.
    """
    import sapien

    arm_qpos = np.asarray(arm_qpos)
    tcp_local = np.asarray(tcp_local)[:, :3]
    n = arm_qpos.shape[0]
    gripper = np.asarray(gripper)
    if gripper.ndim == 1:
        gripper = gripper[:, None]

    renderer.robot.set_root_pose(sapien.Pose())
    flange = {l.name: l for l in renderer.robot.get_links()}[
        FLANGE_LINK_NAME[renderer.arm]]
    p8 = np.empty((n, 3))
    r8 = np.empty((n, 3, 3))
    for i in range(n):
        renderer.robot.set_qpos(renderer.full_qpos(arm_qpos[i], gripper[i]))
        pose = flange.entity_pose
        p8[i] = pose.p
        r8[i] = pose.to_transformation_matrix()[:3, :3]

    a = np.zeros((3 * n, 4))
    b = (tcp_local - p8).reshape(-1)
    for ax in range(3):
        a[ax::3, ax] = 1
        a[ax::3, 3] = r8[:, ax, 2]
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    rms = float(np.sqrt(((b - a @ x) ** 2).mean()))
    if rms > max_residual_m:
        raise RuntimeError(
            f"arm-mount fit residual {rms * 1000:.2f} mm exceeds "
            f"{max_residual_m * 1000:.1f} mm: welded URDF does not "
            "reproduce the source kinematics for this episode"
        )
    return x[:3], float(x[3]), rms


FLANGE_LINK_NAME = {"panda": "panda_link8", "fr3": "fr3_link8"}


class FrankaRobotRenderer:
    def __init__(self, urdf_path, arm: str, shader: str = "default"):
        import sapien

        sapien.render.set_log_level("err")
        self.arm = arm
        self.scene = sapien.Scene()
        # Robot-only scene on black: no ground, no environment map. Neutral
        # lighting so the linkage silhouette and joints read clearly.
        self.scene.set_ambient_light([0.4, 0.4, 0.4])
        self.scene.add_directional_light([0.3, 0.2, -1.0], [1.5, 1.5, 1.5])
        self.scene.add_directional_light([-0.5, -0.6, -0.4], [0.6, 0.6, 0.6])

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(str(urdf_path))
        names = [j.name for j in self.robot.get_active_joints()]
        self._qpos_index = {n: i for i, n in enumerate(names)}
        for jn in ARM_JOINTS[arm]:
            assert jn in self._qpos_index, f"missing joint {jn} in {names}"
        self._camera = None
        self._camera_wh = None

    # -------------------------------------------------------------- qpos

    def full_qpos(self, arm_qpos, gripper) -> np.ndarray:
        """(7,) arm + gripper driver(s) -> full articulation qpos.

        gripper: scalar in [0, 1] (DROID continuous position) or a pair of
        driver angles in radians (MolmoBot qpos['gripper']).
        """
        q = np.zeros(len(self._qpos_index))
        for jn, v in zip(ARM_JOINTS[self.arm], np.asarray(arm_qpos).ravel()):
            q[self._qpos_index[jn]] = v
        g = np.atleast_1d(np.asarray(gripper, dtype=np.float64))
        if g.size == 1:
            drivers = {"left": g[0] * DRIVER_RANGE_RAD,
                       "right": g[0] * DRIVER_RANGE_RAD}
        elif g.size == 2:
            drivers = {"left": g[0], "right": g[1]}
        else:
            raise ValueError(f"gripper state has {g.size} values")
        for jn, (side, sign) in MIMIC_MAP.items():
            if jn in self._qpos_index:
                q[self._qpos_index[jn]] = sign * drivers[side]
        return q

    # ------------------------------------------------------------ camera

    def _get_camera(self, width, height):
        if self._camera is None or self._camera_wh != (width, height):
            if self._camera is not None:
                self.scene.remove_camera(self._camera)
            self._camera = self.scene.add_camera(
                "render_cam", width, height, 1.0, 0.01, 100.0
            )
            self._camera_wh = (width, height)
        return self._camera

    # ------------------------------------------------------------ render

    def render(
        self,
        arm_qpos: np.ndarray,       # (T, 7)
        gripper: np.ndarray,        # (T,) or (T, 2)
        intrinsics: np.ndarray,     # (3, 3) or (T, 3, 3), pixel units
        cam2world_gl: np.ndarray,   # (4, 4) or (T, 4, 4)
        width: int,
        height: int,
        base_pose: np.ndarray = None,   # (T, 7) xyz + wxyz quat, world frame
    ) -> np.ndarray:
        """Render robot-only RGB for T frames -> uint8 (T, H, W, 3)."""
        import sapien

        arm_qpos = np.asarray(arm_qpos)
        n = arm_qpos.shape[0]
        intr = np.broadcast_to(np.asarray(intrinsics), (n, 3, 3))
        c2w = np.broadcast_to(np.asarray(cam2world_gl), (n, 4, 4))
        gripper = np.asarray(gripper)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        if base_pose is not None:
            base_pose = np.broadcast_to(np.asarray(base_pose), (n, 7))

        cam = self._get_camera(width, height)
        out = np.empty((n, height, width, 3), dtype=np.uint8)
        for t in range(n):
            if base_pose is not None:
                self.robot.set_root_pose(
                    sapien.Pose(base_pose[t, :3], base_pose[t, 3:])
                )
            self.robot.set_qpos(self.full_qpos(arm_qpos[t], gripper[t]))
            k = intr[t]
            cam.set_perspective_parameters(
                0.01, 100.0, k[0, 0], k[1, 1], k[0, 2], k[1, 2], 0.0
            )
            cam.set_entity_pose(gl_to_sapien_pose(c2w[t]))
            self.scene.update_render()
            cam.take_picture()
            rgba = cam.get_picture("Color")  # float32 (H, W, 4)
            # Alpha-composite over black: the clear color is not black, the
            # background is only identifiable by alpha == 0.
            rgb = rgba[..., :3] * rgba[..., 3:4]
            out[t] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return out
