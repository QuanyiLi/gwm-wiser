"""FrankaRobotRenderer: state -> robot-only RGB.

One SAPIEN scene holding one welded arm+gripper articulation renders the
robot alone against black. Camera intrinsics/extrinsics/resolution are
injected per call; the URDF (Panda vs FR3 rig) is chosen per source. The
same class serves offline training-data generation and inference-time
candidate scoring — train/inference render homology is a hard requirement.

Kinematics only: qpos is set directly (no physics stepping), and the Robotiq
2F-85 linkage is reconstructed from its driver values through the mimic map
(the ManiSkill URDF ships six independent revolute joints).

A URDF that DOES declare `<mimic>` tags is honoured instead: the chain is read
off the file, so a different gripper needs no constants here. That is how the
`zhiwei` hardware rig's Robotiq 2F-140 renders
(`gwm_hardware/assets/panda_robotiq_2f_140.urdf`, one `finger_joint` driver and
five mimics). The 2F-85 path is untouched by this and stays byte-identical --
the ManiSkill URDF declares no mimics, so it falls through to MIMIC_MAP exactly
as before.
"""

import xml.etree.ElementTree as ET

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


def parse_mimic_chain(urdf_path):
    """Read a URDF's single-driver mimic chain.

    Returns ``(driver_joint, closed_angle_rad, [(joint, multiplier), ...])`` or
    ``None`` when the file declares no mimics (the 2F-85 case). `closed_angle`
    is the driver's own upper limit, i.e. the angle a fully closed gripper
    sits at, which is what the [0, 1] gripper command scales onto.
    """
    root = ET.parse(str(urdf_path)).getroot()
    chain, drivers = [], set()
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        drivers.add(mimic.get("joint"))
        chain.append((joint.get("name"), float(mimic.get("multiplier", 1.0))))
    if not chain:
        return None
    if len(drivers) != 1:
        raise ValueError(f"{urdf_path}: expected one mimic driver, found {sorted(drivers)}")
    driver = drivers.pop()
    upper = None
    for joint in root.findall("joint"):
        if joint.get("name") == driver:
            limit = joint.find("limit")
            if limit is not None:
                upper = float(limit.get("upper"))
    if not upper:
        raise ValueError(f"{urdf_path}: mimic driver {driver!r} has no usable upper limit")
    return driver, upper, chain

ARM_JOINTS = {
    "panda": [f"panda_joint{i}" for i in range(1, 8)],
    "fr3": [f"fr3_joint{i}" for i in range(1, 8)],
}


def gl_to_sapien_pose(cam2world_gl: np.ndarray):
    """OpenGL camera-to-world (x right, y up, -z forward) -> SAPIEN pose.

    Mapping verified empirically against MolmoBot's paired
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


def cv_pose_to_matrix(xyzrpy: np.ndarray) -> np.ndarray:
    """KarlP 6D extrinsic [x,y,z,rx,ry,rz] -> 4x4 cam2base (OpenCV camera
    axes: x right, y down, z forward). Euler convention is scipy "xyz",
    matching the release README verbatim."""
    from scipy.spatial.transform import Rotation

    v = np.asarray(xyzrpy, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", v[3:6]).as_matrix()
    m[:3, 3] = v[:3]
    return m


def cv_pose_to_sapien_pose(xyzrpy: np.ndarray):
    """Camera pose 6D [x,y,z,rx,ry,rz] in base frame, OpenCV camera axes
    (x right, y down, z forward) -> SAPIEN pose (x forward, y left, z up).

    The axis mapping (forward = column z, left = -column x, up = -column y)
    is the same one gl_to_sapien_pose applies, which was closed-loop-verified:
    camera.get_extrinsic_matrix() reproduces the inverse cam2world exactly.
    """
    return gl_to_sapien_pose(cv_pose_to_matrix(xyzrpy))


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
        self._mimic = parse_mimic_chain(urdf_path)
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
        if self._mimic is not None:
            # URDF-declared chain (2F-140): one driver, everything else follows
            # it. A two-value gripper state has no meaning on a single-driver
            # linkage, so it is averaged rather than silently taking one side.
            driver, closed, chain = self._mimic
            angle = float(g.mean()) * closed
            if driver in self._qpos_index:
                q[self._qpos_index[driver]] = angle
            for jn, mult in chain:
                if jn in self._qpos_index:
                    q[self._qpos_index[jn]] = mult * angle
            return q
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
        return_alpha: bool = False,
    ) -> np.ndarray:
        """Render robot-only RGB for T frames -> uint8 (T, H, W, 3).

        `return_alpha` additionally returns the (T, H, W) float32 coverage mask,
        for compositing the robot onto a real photograph. The scorer never asks
        for it -- the model is fed the over-black composite below and nothing
        else -- but a human checking WHERE the arm goes needs the arm drawn in
        the scene, and thresholding the composite would silently drop every
        dark part of the gripper. Off by default, so the scoring path is
        byte-identical.
        """
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
        alpha = np.empty((n, height, width), dtype=np.float32) if return_alpha else None
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
            if alpha is not None:
                alpha[t] = rgba[..., 3].astype(np.float32)
        return (out, alpha) if return_alpha else out
