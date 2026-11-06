"""
Shared helpers for planner evaluations that build GWM candidates from joint
trajectories instead of the skill library's pre-rendered videos.

- ``KeyframeRenderer``: renders robot-only images (the GWM candidate input) for
  arbitrary joint states with ``RobotRenderer`` and exposes TCP forward kinematics.
- ``smooth_joint_noise``: temporally smooth Gaussian noise for the arm joints.
- ``task_mode``: (cube slot, placard slot) of a WISER env index.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

ARM_DOF = 7
TCP_LINK = "panda_hand_tcp"


def keyframe_indices(num_future_frames: int = 60, num_keyframes: int = 6):
    """Same frame indices as ``PaddedLeRobotDataset`` (np.linspace, int)."""
    return np.linspace(0, num_future_frames - 1, num_keyframes, dtype=int).tolist()


def task_mode(env_idx: int):
    """WISER env ``e`` picks the cube in slot ``e // 3`` and places on placard slot ``e % 3``.

    Follows ``WorkSpaceBuilder._get_scenario_configs``: cube positions are rotated per
    group by ``repeat_to_length`` and interleaved 3x, destinations are tiled 4x.
    """
    return env_idx // 3, env_idx % 3


def mode_id(cube_slot: int, place_slot: int) -> int:
    return cube_slot * 3 + place_slot


def smooth_joint_noise(
    rng: np.random.Generator,
    length: int,
    sigma: float,
    smooth_steps: float = 8.0,
    ramp_steps: int = 10,
    dof: int = ARM_DOF,
) -> np.ndarray:
    """Gaussian noise smoothed along time, unit std before scaling, ramped in from zero.

    The ramp keeps every perturbed trajectory starting at the current joint state.
    Returns an array of shape (length, dof) in radians.
    """
    if sigma <= 0:
        return np.zeros((length, dof), dtype=np.float32)
    white = rng.standard_normal((length + 6 * int(smooth_steps), dof))
    smooth = gaussian_filter1d(white, smooth_steps, axis=0, mode="nearest")
    pad = 3 * int(smooth_steps)
    smooth = smooth[pad : pad + length]
    smooth = smooth / (smooth.std(axis=0, keepdims=True) + 1e-8)
    ramp = np.clip(np.arange(length) / max(ramp_steps, 1), 0.0, 1.0)[:, None]
    return (sigma * ramp * smooth).astype(np.float32)


class KeyframeRenderer:
    """Robot-only renders of joint states, matching the env's ``image_1_robot_state``.

    Uses a single-env (CPU physics, GPU render) ``RobotRenderer`` so it can live in
    the same process as the GPU-physics evaluation env.
    """

    def __init__(self, device: str = "cuda"):
        from gwm_wiser.utils.robot_renderer import RobotRenderer

        self.renderer = RobotRenderer(num_envs=1, device=device, robot="panda")
        self.tcp_link = self.renderer.robot.links_map[TCP_LINK]

    @staticmethod
    def actions_to_qpos(actions: torch.Tensor) -> torch.Tensor:
        from gwm_wiser.utils.robot_renderer import RobotRenderer

        return RobotRenderer._panda_action_to_qpos(actions.float())

    def render_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        """qpos (T, 9) -> images (T, 3, H, W) float in [0, 1] (dataset video format)."""
        frames = []
        for q in qpos:
            img = self.renderer.render_qpos(q[None].float())[0]  # (H, W, 3) uint8
            frames.append(img.permute(2, 0, 1).float().cpu() / 255.0)
        return torch.stack(frames, dim=0)

    def tcp_positions(self, qpos: torch.Tensor) -> np.ndarray:
        """qpos (T, 9) -> TCP positions (T, 3) via the articulation's kinematics."""
        robot = self.renderer.robot
        out = []
        for q in qpos:
            robot.set_qpos(q[None].float().to(self.renderer.device))
            out.append(self.tcp_link.pose.p[0].cpu().numpy().copy())
        return np.stack(out, axis=0)

    def close(self):
        self.renderer.close()
