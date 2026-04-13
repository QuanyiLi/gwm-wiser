"""
Lightweight robot-only renderer.

Renders a robot arm against a black background given ``pd_joint_pos``
actions – the same format accepted by ``WISEREnv.step()``.

Supports:
- **Panda**  (``my_panda_wristcam``): 8-dim action (7 arm + 1 normalised gripper)
- **XArm6** (``xarm6_robotiq_wristcam``): 7-dim action (6 arm + 1 raw gripper)

The output matches ``DomainRandEnv.get_obs()[image_1_robot_state]`` but does
**not** require physics simulation, segmentation, or scene objects.  Only the
robot articulation, a single camera and minimal lighting are created.
"""

from __future__ import annotations


import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import torch

import mani_skill.render.utils as render_utils
from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils.system.backend import parse_sim_and_render_backend
from mani_skill.render import set_shader_pack
from mani_skill.sensors.camera import Camera, CameraConfig
from mani_skill.utils import sapien_utils

from gwm_wiser.env.config import CAMERA_RESOLUTION, scene_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Robot-specific configuration
# ─────────────────────────────────────────────────────────────────────────────

# Panda gripper: normalised [-1, 1] → [-0.01, 0.04]
_PANDA_GRIPPER_LOWER = -0.01
_PANDA_GRIPPER_UPPER = 0.04


def _panda_gripper_unnormalise(action: torch.Tensor) -> torch.Tensor:
    """Map normalised gripper action [-1, 1] → raw finger qpos (Panda)."""
    low = _PANDA_GRIPPER_LOWER
    high = _PANDA_GRIPPER_UPPER
    action = torch.clamp(action, -1.0, 1.0)
    return 0.5 * (high + low) + 0.5 * (high - low) * action


# ── Per-robot configs ────────────────────────────────────────────────────────

_ROBOT_CONFIGS = {
    "panda": dict(
        urdf_path=f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf",
        base_pos=[-0.615, 0, 0],
        arm_dof=7,
        action_dim=8,  # 7 arm + 1 gripper (normalised)
        qpos_dim=9,  # 7 arm + 2 finger
        gripper_normalised=True,
        urdf_config=dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        ),
        loader_name="my_panda_wristcam",
    ),
    "xarm6": dict(
        urdf_path=f"{ASSET_DIR}/robots/xarm6/xarm6_robotiq.urdf",
        base_pos=[-0.565, 0, 0],
        arm_dof=6,
        action_dim=7,  # 6 arm + 1 gripper (raw)
        qpos_dim=12,  # 6 arm + 6 gripper (2 active + 4 passive)
        gripper_normalised=False,
        urdf_config=dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                left_inner_finger_pad=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                right_inner_finger_pad=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        ),
        loader_name="xarm6_robotiq_wristcam",
    ),
}


class RobotRenderer:
    """Parallel, lightweight robot‑only renderer.

    Creates a minimal SAPIEN scene containing **only** the robot and a
    fixed external camera identical to ``base_camera_0`` of the
    ``WISEREnv``.  No physics engine is stepped – articulation
    poses are set directly via ``set_qpos``.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments to render.
    device : str, optional
        Torch device, by default ``"cuda"``.
    robot : str, optional
        Robot type: ``"panda"`` or ``"xarm6"``.  Default ``"panda"``.
    match_env_lighting : bool, optional
        If ``True`` (default), use the same ambient + directional lighting as
        ``DomainRandEnv`` and apply a segmentation mask to zero out the
        background.  This produces images **visually identical** to
        ``image_1_robot_state``.
        If ``False``, ambient light is set to zero so empty space is naturally
        black (no masking needed).  The robot will appear darker because it
        receives no ambient fill.
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cuda",
        robot: str = "panda",
        match_env_lighting: bool = True,
    ):
        if robot not in _ROBOT_CONFIGS:
            raise ValueError(
                f"Unsupported robot '{robot}'. Choose from {list(_ROBOT_CONFIGS)}"
            )

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.match_env_lighting = match_env_lighting
        self.robot_type = robot
        self.rcfg = _ROBOT_CONFIGS[robot]

        # ── Choose backends ──────────────────────────────────────────────
        sim_backend = "physx_cuda" if num_envs > 1 else "physx_cpu"
        render_backend = "gpu"
        self.backend = parse_sim_and_render_backend(sim_backend, render_backend)

        if self.backend.sim_device.is_cuda() and not physx.is_gpu_enabled():
            physx.enable_gpu()

        # ── Build the scene(s) ───────────────────────────────────────────
        self._build_scene()

        # ── Load only the robot ──────────────────────────────────────────
        self._load_robot()

        # ── Initialise GPU sim buffers ───────────────────────────────────
        self.scene._setup(enable_gpu=self.scene.gpu_sim_enabled)

        # ── Lighting ─────────────────────────────────────────────────────
        self._load_lighting()

        # ── Camera ───────────────────────────────────────────────────────
        self._setup_camera()

    # ─────────────────────────────────────────────────────────────────────
    # Internal setup helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_scene(self):
        """Create a ManiSkillScene with minimal physics config."""
        physx.set_default_material(
            static_friction=0.3, dynamic_friction=0.3, restitution=0.0
        )

        if self.backend.sim_device.is_cuda():
            physx_system = physx.PhysxGpuSystem(device=self.backend.sim_device)
            sub_scenes = []
            grid = int(np.ceil(np.sqrt(self.num_envs)))
            for idx in range(self.num_envs):
                sx, sy = idx % grid - grid // 2, idx // grid - grid // 2
                systems = [physx_system]
                if render_utils.can_render(self.backend.render_device):
                    systems.append(
                        sapien.render.RenderSystem(self.backend.render_device)
                    )
                sc = sapien.Scene(systems=systems)
                physx_system.set_scene_offset(sc, [sx * 10.0, sy * 10.0, 0.0])
                sub_scenes.append(sc)
        else:
            physx_system = physx.PhysxCpuSystem()
            systems = [physx_system]
            if render_utils.can_render(self.backend.render_device):
                systems.append(sapien.render.RenderSystem(self.backend.render_device))
            sub_scenes = [sapien.Scene(systems)]

        from mani_skill.utils.structs.types import SimConfig

        self.scene = ManiSkillScene(
            sub_scenes,
            sim_config=SimConfig(),
            device=self.backend.device,
            parallel_in_single_scene=False,
            backend=self.backend,
        )

    def _load_robot(self):
        """Load the robot URDF into every sub-scene."""
        cfg = self.rcfg
        urdf_path = cfg["urdf_path"]
        base_pos = cfg["base_pos"]

        loader = self.scene.create_urdf_loader()
        loader.name = cfg["loader_name"]
        loader.fix_root_link = True
        loader.disable_self_collisions = False
        parsed_urdf_config = sapien_utils.parse_urdf_config(cfg["urdf_config"])
        sapien_utils.check_urdf_config(parsed_urdf_config)
        sapien_utils.apply_urdf_config(loader, parsed_urdf_config)

        builder = loader.parse(urdf_path)["articulation_builders"][0]
        builder.initial_pose = sapien.Pose(p=base_pos)
        self.robot = builder.build()

        # Disable gravity for all links (matches the env behaviour)
        if not self.scene.gpu_sim_enabled:
            for link in self.robot.links:
                link.disable_gravity = True

    def _load_lighting(self):
        """Set up scene lighting."""
        if self.match_env_lighting:
            self.scene.set_ambient_light([0.3, 0.3, 0.3])
        else:
            self.scene.set_ambient_light([0.0, 0.0, 0.0])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_camera(self):
        """Create and register a camera identical to ``base_camera_0``."""
        cam_cfg = scene_cfg
        eye = cam_cfg["sensor_cam_eye_pos"][0]
        target = cam_cfg["sensor_cam_target_pos"][0]
        fov_deg = cam_cfg.get("sensor_cam_fov", 38)
        fov_rad = np.pi * (fov_deg / 180.0)

        cam_pose = sapien_utils.look_at(eye=eye, target=target)
        shader = "default" if self.match_env_lighting else "minimal"
        config = CameraConfig(
            uid="base_camera_0",
            pose=cam_pose,
            width=CAMERA_RESOLUTION[0],
            height=CAMERA_RESOLUTION[1],
            fov=fov_rad,
            near=0.01,
            far=100,
            shader_pack=shader,
        )

        set_shader_pack(config.shader_config)
        self.camera = Camera(config, self.scene)
        self.scene.sensors = {"base_camera_0": self.camera}
        self.scene.human_render_cameras = {}

    # ─────────────────────────────────────────────────────────────────────
    # Action → qpos conversion
    # ─────────────────────────────────────────────────────────────────────

    def action_to_qpos(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert a ``pd_joint_pos`` action to full-DOF qpos.

        Parameters
        ----------
        actions : Tensor
            Shape ``(N, action_dim)`` where ``action_dim`` is 8 (Panda) or
            7 (XArm6).

        Returns
        -------
        qpos : Tensor
            Shape ``(N, qpos_dim)`` where ``qpos_dim`` is 9 (Panda) or
            12 (XArm6).
        """
        if self.robot_type == "panda":
            return self._panda_action_to_qpos(actions)
        else:
            return self._xarm6_action_to_qpos(actions)

    @staticmethod
    def _panda_action_to_qpos(actions: torch.Tensor) -> torch.Tensor:
        """Panda: 8-dim action → 9-dim qpos."""
        arm = actions[:, :7]  # raw joint pos
        gripper_normed = actions[:, 7:8]  # [-1, 1]
        finger = _panda_gripper_unnormalise(gripper_normed)
        return torch.cat([arm, finger, finger], dim=-1)  # mimic → both fingers

    @staticmethod
    def _xarm6_action_to_qpos(actions: torch.Tensor) -> torch.Tensor:
        """XArm6: 7-dim action → 12-dim qpos.

        Joint order in URDF:
            [0:6]  arm joints
            [6]    left_outer_knuckle_joint  (mimic of right)
            [7]    left_inner_knuckle_joint  (coupled)
            [8]    right_outer_knuckle_joint (control)
            [9]    right_inner_knuckle_joint (coupled)
            [10]   left_inner_finger_joint   (coupled)
            [11]   right_inner_finger_joint  (coupled)

        Action format:
            [0:6]  arm joint positions (raw, normalize_action=False)
            [6]    right_outer_knuckle_joint position (raw)

        The robotiq 2F gripper has a parallel linkage where the inner
        knuckle and inner finger joints are mechanically coupled to the
        outer knuckle joints.  Without physics stepping we manually
        compute these using the kinematic ratio from the joint limits:
            outer_knuckle range: [0, 0.81]
            inner_knuckle range: [0, 0.8757]   →  ratio = 0.8757 / 0.81
            inner_finger  range: [-0.8757, 0]  →  ratio = -0.8757 / 0.81
        """
        arm = actions[:, :6]  # (N, 6)
        gripper_raw = actions[:, 6:7]  # (N, 1) – raw pos for right_outer_knuckle

        # Mimic joint: left_outer = right_outer (multiplier=1, offset=0)
        mimic = gripper_raw

        # Coupled joints via parallel linkage
        ratio = 0.8757 / 0.81
        inner_knuckle = gripper_raw * ratio  # same direction
        inner_finger = gripper_raw * (-ratio)  # opposite direction

        # Assemble: [arm(6), left_outer(mimic), left_inner_knuckle,
        #            right_outer(control), right_inner_knuckle,
        #            left_finger, right_finger]
        qpos = torch.cat(
            [
                arm,  # [0:6]
                mimic,  # [6]   left_outer_knuckle (mimic)
                inner_knuckle,  # [7]   left_inner_knuckle (coupled)
                gripper_raw,  # [8]   right_outer_knuckle (control)
                inner_knuckle,  # [9]   right_inner_knuckle (coupled)
                inner_finger,  # [10]  left_inner_finger (coupled)
                inner_finger,  # [11]  right_inner_finger (coupled)
            ],
            dim=-1,
        )

        return qpos

    # ─────────────────────────────────────────────────────────────────────
    # Core render API
    # ─────────────────────────────────────────────────────────────────────

    def render(self, actions: torch.Tensor) -> torch.Tensor:
        """Render the robot for a batch of actions.

        Parameters
        ----------
        actions : Tensor of shape ``(N, action_dim)``
            Same action format as ``DomainRandEnv.step()``.

        Returns
        -------
        images : Tensor of shape ``(N, H, W, 3)``, dtype ``torch.uint8``
            RGB images of the robot on a black background.
        """
        actions = actions.to(device=self.device, dtype=torch.float32)
        qpos = self.action_to_qpos(actions)
        return self.render_qpos(qpos)

    def render_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        """Render the robot for explicit qpos values.

        Parameters
        ----------
        qpos : Tensor of shape ``(N, qpos_dim)``

        Returns
        -------
        images : Tensor of shape ``(N, H, W, 3)``, dtype ``torch.uint8``
            RGB images of the robot on a black background.
        """
        qpos = qpos.to(device=self.device, dtype=torch.float32)

        # ── Set qpos on the articulation ─────────────────────────────────
        self.robot.set_qpos(qpos)

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

        # ── Render ───────────────────────────────────────────────────────
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.camera.capture()

        need_seg = self.match_env_lighting
        obs = self.camera.get_obs(
            rgb=True,
            depth=False,
            position=False,
            segmentation=need_seg,
            apply_texture_transforms=True,
        )

        rgb = obs["rgb"]

        # ── Apply segmentation mask (if ambient light is used) ───────────
        if self.match_env_lighting:
            seg = obs["segmentation"]  # (N, H, W, 1)
            mask = seg[..., 0] > 0  # (N, H, W)
            rgb = rgb.clone()
            rgb[~mask] = 0

        return rgb

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    def close(self):
        """Release all SAPIEN resources."""
        self.scene = None
        self.robot = None
        self.camera = None
        import gc

        gc.collect()

    def __del__(self):
        self.close()
