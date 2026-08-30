"""Franka Panda kinematics in plain torch — the one model both the executors
and the GWM scoring client use.

Forward kinematics to the flange (``panda_link8``) and on to the tool centre
point of the DROID Robotiq 2F-85, a geometric Jacobian, a damped-least-squares
IK step, and the straight-line end-effector interpolation a macro action is
executed with. Everything is batched over a leading ``N`` and runs on whatever
device the inputs live on, so 2048 envs cost the same call as one.

The tool orientation the policy commands is always "top-down + yaw":
``R(yaw) = Rz(yaw) @ R_DOWN``. What the fingers' opening axis is in the flange
frame, and the flange-to-TCP offset, are measured against Isaac once
(`smoke.py --calibrate`) and pinned below.
"""

from __future__ import annotations

import math

import torch

# Modified DH (Craig) parameters of the Panda: rows (a_{i-1}, d_i, alpha_{i-1}).
_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -math.pi / 2),
    (0.0, 0.316, math.pi / 2),
    (0.0825, 0.0, math.pi / 2),
    (-0.0825, 0.384, -math.pi / 2),
    (0.0, 0.0, math.pi / 2),
    (0.088, 0.0, math.pi / 2),
)
#: panda_link7 -> panda_link8 (the flange), along z7.
FLANGE_D = 0.107

#: Flange -> tool centre point (midpoint between the open pads), along the
#: flange +z. 0.150 is `gwm_drawer/traj.py`'s D_TIP on the scoring URDF;
#: measured on the Isaac asset by `smoke.py`: 0.1493, pads 10.25 cm apart.
TCP_OFFSET = 0.150
#: Which flange axis the pads open along (0 = x, 1 = y); measured: y.
OPENING_AXIS = 1

JOINT_LOWER = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
#: Kept away from the hard stops so the PD never drives into them.
JOINT_MARGIN = 0.02

#: Flange orientation with the tool pointing straight down (a half turn about
#: x): flange +z -> world -z, flange +y -> world -y.
R_DOWN = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

#: DROID home pose (`nvidia_droid.py`): joints 2/4/6 at -pi/5, -4pi/5, 3pi/5.
HOME_Q = torch.tensor([0.0, -math.pi / 5, 0.0, -4 * math.pi / 5, 0.0, 3 * math.pi / 5, 0.0])


# ------------------------------------------------------------------ FK --


def _dh_matrix(theta: torch.Tensor, a: float, d: float, alpha: float) -> torch.Tensor:
    """RotX(alpha) TransX(a) RotZ(theta) TransZ(d), batched over theta."""
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    rows = [
        torch.stack([ct, -st, zero, a * one], dim=-1),
        torch.stack([st * ca, ct * ca, -sa * one, -sa * d * one], dim=-1),
        torch.stack([st * sa, ct * sa, ca * one, ca * d * one], dim=-1),
        torch.stack([zero, zero, zero, one], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def fk_frames(q: torch.Tensor) -> list[torch.Tensor]:
    """Joint frames 1..7 and the flange, each ``(N, 4, 4)`` in the base frame."""
    T = torch.eye(4, device=q.device, dtype=q.dtype).expand(q.shape[0], 4, 4)
    frames = []
    for i, (a, d, alpha) in enumerate(_DH):
        T = T @ _dh_matrix(q[:, i], a, d, alpha)
        frames.append(T)
    flange = T.clone()
    flange[:, :3, 3] += FLANGE_D * T[:, :3, 2]
    frames.append(flange)
    return frames


def fk_tcp(q: torch.Tensor, tcp_offset: float = TCP_OFFSET) -> tuple[torch.Tensor, torch.Tensor]:
    """TCP position ``(N, 3)`` and rotation ``(N, 3, 3)`` (= the flange's)."""
    flange = fk_frames(q)[-1]
    R = flange[:, :3, :3]
    p = flange[:, :3, 3] + tcp_offset * R[:, :, 2]
    return p, R


def jacobian(q: torch.Tensor, tcp_offset: float = TCP_OFFSET) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Geometric Jacobian of the TCP, ``(N, 6, 7)``, plus the TCP pose.

    Column ``i`` is ``[z_i x (p - o_i); z_i]`` for the revolute axis of joint
    ``i`` — ``z_i``/``o_i`` read off the frame after the full DH transform,
    which lies on the same axis line as the joint's own frame.
    """
    frames = fk_frames(q)
    flange = frames[-1]
    R = flange[:, :3, :3]
    p = flange[:, :3, 3] + tcp_offset * R[:, :, 2]
    cols = []
    for T in frames[:7]:
        z = T[:, :3, 2]
        o = T[:, :3, 3]
        cols.append(torch.cat([torch.linalg.cross(z, p - o, dim=-1), z], dim=-1))
    return torch.stack(cols, dim=-1), p, R


# ---------------------------------------------------------------- rotations --


def rotvec_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector of ``R``, ``(..., 3)``; exact away from a half turn."""
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    w = 0.5 * torch.stack(
        [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1
    )  # sin(theta) * axis
    sin = torch.sqrt((1.0 - cos * cos).clamp(min=0.0))
    scale = torch.where(sin > 1e-6, theta / sin.clamp(min=1e-6), torch.ones_like(sin))
    return w * scale.unsqueeze(-1)


def rot_z(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack(
        [torch.stack([c, -s, zero], -1), torch.stack([s, c, zero], -1), torch.stack([zero, zero, one], -1)], dim=-2
    )


def target_rotation(yaw: torch.Tensor) -> torch.Tensor:
    """Tool pointing down, rotated ``yaw`` about the world vertical: ``(N, 3, 3)``."""
    return rot_z(yaw) @ R_DOWN.to(yaw.device, yaw.dtype)


def yaw_of(R: torch.Tensor) -> torch.Tensor:
    """The yaw that best explains ``R`` as ``Rz(yaw) @ R_DOWN`` (tilt ignored)."""
    M = R @ R_DOWN.to(R.device, R.dtype).transpose(-1, -2)
    return torch.atan2(M[..., 1, 0], M[..., 0, 0])


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) unit quaternions -> rotation matrices, batched."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


# ------------------------------------------------------------------- IK --


def clamp_joints(q: torch.Tensor) -> torch.Tensor:
    lo = JOINT_LOWER.to(q.device, q.dtype) + JOINT_MARGIN
    hi = JOINT_UPPER.to(q.device, q.dtype) - JOINT_MARGIN
    return torch.minimum(torch.maximum(q, lo), hi)


def ik_step(
    q: torch.Tensor,
    p_target: torch.Tensor,
    R_target: torch.Tensor,
    *,
    iters: int = 3,
    damping: float = 0.05,
    max_dq: float = 0.12,
    rot_weight: float = 1.0,
    tcp_offset: float = TCP_OFFSET,
) -> torch.Tensor:
    """Damped-least-squares pull of the TCP toward a pose, bounded per call.

    ``iters`` inner Gauss-Newton iterations, then the total joint motion is
    clipped to ``max_dq`` per joint — 0.12 rad per 15 Hz tick is 1.8 rad/s,
    inside the Panda's limits. Joint limits are enforced with a margin.
    """
    q0 = q
    eye = torch.eye(6, device=q.device, dtype=q.dtype)
    for _ in range(iters):
        J, p, R = jacobian(q, tcp_offset)
        e_pos = p_target - p
        e_rot = rotvec_from_matrix(R_target @ R.transpose(-1, -2)) * rot_weight
        e = torch.cat([e_pos, e_rot], dim=-1).unsqueeze(-1)
        JJt = J @ J.transpose(-1, -2) + (damping**2) * eye
        dq = (J.transpose(-1, -2) @ torch.linalg.solve(JJt, e)).squeeze(-1)
        q = clamp_joints(q + dq)
    step = (q - q0).clamp(-max_dq, max_dq)
    return clamp_joints(q0 + step)


# ------------------------------------------------------------ interpolation --


def wrap_half_turn(d: torch.Tensor) -> torch.Tensor:
    """Yaw difference reduced modulo pi into (-pi/2, pi/2]: a parallel jaw
    rotated a half turn is the same grasp, so the tool always takes the
    shorter of the two equivalent rotations."""
    return torch.remainder(d + math.pi / 2, math.pi) - math.pi / 2


def interpolate_pose(
    p0: torch.Tensor, yaw0: torch.Tensor, p1: torch.Tensor, yaw1: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight line in position and yaw (modulo pi), ``n`` samples excluding
    the start: ``(N, n, 3)`` and ``(N, n)``, the last sample being the target."""
    s = torch.arange(1, n + 1, device=p0.device, dtype=p0.dtype) / n
    p = p0.unsqueeze(1) + (p1 - p0).unsqueeze(1) * s.view(1, n, 1)
    yaw = yaw0.unsqueeze(1) + wrap_half_turn(yaw1 - yaw0).unsqueeze(1) * s.view(1, n)
    return p, yaw


def plan_macro(
    q0: torch.Tensor,
    p_target: torch.Tensor,
    yaw_target: torch.Tensor,
    *,
    n_move: int,
    n_hold: int,
    tcp_offset: float = TCP_OFFSET,
    **ik_kwargs,
) -> torch.Tensor:
    """Open-loop joint trajectory of a macro action: ``(N, n_move + n_hold, 7)``.

    The executor does exactly this, one tick at a time, from the *measured*
    joints; run from the same start pose the two agree wherever the arm is
    free to follow, which is what makes a scored plan the executed one.
    """
    p0, R0 = fk_tcp(q0, tcp_offset)
    p_path, yaw_path = interpolate_pose(p0, yaw_of(R0), p_target, yaw_target, n_move)
    q = q0
    out = []
    for k in range(n_move):
        q = ik_step(q, p_path[:, k], target_rotation(yaw_path[:, k]), tcp_offset=tcp_offset, **ik_kwargs)
        out.append(q)
    for _ in range(n_hold):
        out.append(q)
    return torch.stack(out, dim=1)
