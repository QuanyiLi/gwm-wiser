"""Pick-up-the-bowl: parameters, predicates, the staged reward, the collision
cost, the observation fields and the per-env episode state.

torch only — Isaac arrives through the ``env`` argument at call time, so the
reward and predicates can be exercised without booting Kit. The shape follows
`isaaclab_M3/tasks/single_object/task_core.py` (staged reward, dwell success,
metrics drained per window); what is specific to this task:

- **The reach target is the bowl's rim, not its centre.** The 2F-85 opens
  8.5 cm and the bowl is 16 cm across, so the only graspable feature is the
  rim wall. The target is the point of the rim circle nearest the TCP, carried
  by the bowl's live pose, 1.2 cm *below* the rim plane so the pads straddle
  the wall rather than pinch its edge; the alignment term wants the tool
  pointing into the bowl with the pads opening radially — across the wall —
  and the grasp bonus is scaled by the depth actually reached.
- **Success is holding the bowl up**: lifted at least ``lift_height`` above its
  settled height with both pads in contact, for ``dwell_target`` consecutive
  ticks. No orientation requirement.
- **The collision cost is measured, never paid** (``collision_penalty`` is 0):
  contact force between the robot and every non-bowl body, integrated per
  episode, plus how far the distractors were moved.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

import torch

from gwm_rl import geometry as G
from gwm_rl.franka_kin import OPENING_AXIS, TCP_OFFSET, quat_to_matrix

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

TICK_DT = 8.0 / 120.0  # 15 Hz control over 120 Hz PhysX


@dataclasses.dataclass(frozen=True)
class TaskParams:
    """Thresholds and reward shaping; overridable per run via ``--env-set task_params.<k>=<v>``."""

    #: Bowl centre above its settled height that counts as lifted (one-sided).
    lift_height: float = 0.10
    #: Consecutive ticks lifted-and-grasped that count as success (8 = 0.53 s).
    dwell_target: int = 8
    #: Per-pad force against the bowl for `is_grasped`, and for the observation bits.
    grasp_force_threshold: float = 0.5
    engaged_force_threshold: float = 0.01
    #: Robot-vs-obstacle force above which a tick counts as "in contact".
    obstacle_force_threshold: float = 1.0
    #: How far below the rim plane the pads' midpoint should sit for a grasp
    #: that can lift: the reach target is this deep, and the grasp bonus is
    #: scaled by the depth actually reached (`grasp_depth_ok`: zero
    #: ``grasp_depth_slack`` above the plane, full from
    #: ``grasp_depth - 2 * grasp_depth_slack`` below it). A pinch on the rim's
    #: top edge satisfies `is_grasped` at 0.5 N yet cannot lift.
    grasp_depth: float = 0.012
    grasp_depth_slack: float = 0.005
    #: Weight of the lift term (linear in bowl height up to `lift_height`).
    lift_reward_weight: float = 1.0

    # -- staged reward (M3 numbers; the lift term is linear in height)
    reach_reward_scale: float = 5.0
    reach_far_weight: float = 0.5
    reach_far_scale: float = 2.0
    align_reward_weight: float = 0.6
    grasp_quality_weight: float = 0.6
    held_bonus: float = 1.0
    dwell_reward_weight: float = 1.5
    success_bonus: float = 10.0
    success_margin_weight: float = 3.0
    success_margin_scale: float = 12.0
    #: Reward per newton-second of obstacle impulse, subtracted; 0 = cost is only logged.
    collision_penalty: float = 0.0

    #: Control ticks per episode (220 = 5 macro-steps of 44).
    episode_ticks: int = 220
    tcp_offset: float = TCP_OFFSET


def resolve_params(overrides: dict | None) -> TaskParams:
    return dataclasses.replace(TaskParams(), **(overrides or {}))


# ------------------------------------------------------------------ math --


def rim_geometry(tcp_pos: torch.Tensor, bowl_pos: torch.Tensor, bowl_quat: torch.Tensor):
    """Nearest rim point to the TCP, the outward radial there, and the bowl's up axis."""
    R = quat_to_matrix(bowl_quat)
    axis = R @ tcp_pos.new_tensor(G.BOWL_AXIS_BODY)  # (N, 3), centre -> rim plane
    rim_center = bowl_pos + G.BOWL_HALF_HEIGHT * axis
    v = tcp_pos - rim_center
    v_perp = v - (v * axis).sum(-1, keepdim=True) * axis
    norm = torch.linalg.vector_norm(v_perp, dim=-1, keepdim=True)
    # Directly above the centre the radial is undefined; fall back to the
    # world -x side, where the drawer experiment's rim grasp was.
    fallback = tcp_pos.new_tensor([-1.0, 0.0, 0.0]).expand_as(v_perp)
    fallback = fallback - (fallback * axis).sum(-1, keepdim=True) * axis
    radial = torch.where(norm > 1e-4, v_perp / norm.clamp(min=1e-4), fallback)
    radial = radial / torch.linalg.vector_norm(radial, dim=-1, keepdim=True).clamp(min=1e-6)
    rim_point = rim_center + G.BOWL_RIM_RADIUS * radial
    return rim_point, radial, axis


def grasp_depth(tcp_pos: torch.Tensor, rim_point: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """How far the TCP sits *below* the rim plane, along the bowl's up axis (m)."""
    return ((rim_point - tcp_pos) * axis).sum(-1)


def grasp_alignment(R_tool: torch.Tensor, radial: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """In [0, 1]: tool z pointing into the bowl (along -axis) and the pad-opening
    axis parallel to the radial, either way round (a parallel jaw is symmetric)."""
    approach = (R_tool[:, :, 2] * (-axis)).sum(-1).clamp(0.0, 1.0)
    opening = (R_tool[:, :, OPENING_AXIS] * radial).sum(-1).abs().clamp(max=1.0)
    return 0.5 * (approach + opening)


def pick_reward(
    *,
    grasp_dist: torch.Tensor,
    grasp_align: torch.Tensor,
    grasp_depth_ok: torch.Tensor,
    is_grasped: torch.Tensor,
    height_reached: torch.Tensor,
    bowl_height: torch.Tensor,
    height_margin: torch.Tensor,
    dwell_fraction: torch.Tensor,
    success: torch.Tensor,
    obstacle_force: torch.Tensor,
    params: TaskParams,
) -> torch.Tensor:
    """Staged: reach+grasp tops out at 2.5, lifting adds up to
    ``lift_reward_weight`` (linear in height), holding up adds that plus
    1 + dwell, success replaces the lot with 10 + margin. Divided by 10.
    """
    p = params
    align = (1.0 - p.align_reward_weight) + p.align_reward_weight * grasp_align
    near = 1.0 - torch.tanh(p.reach_reward_scale * grasp_dist)
    far = 1.0 - torch.tanh(p.reach_far_scale * grasp_dist)
    quality = (1.0 - p.grasp_quality_weight) + p.grasp_quality_weight * grasp_align * grasp_depth_ok
    reward = near * align + p.reach_far_weight * far + is_grasped.float() * quality

    lift_mask = is_grasped & ~height_reached
    lift_r = p.lift_reward_weight * (bowl_height / p.lift_height).clamp(0.0, 1.0)
    reward = torch.where(lift_mask, reward + lift_r, reward)

    held_mask = is_grasped & height_reached
    reward = torch.where(
        held_mask, reward + p.lift_reward_weight + p.held_bonus + p.dwell_reward_weight * dwell_fraction, reward
    )

    margin = torch.tanh(p.success_margin_scale * height_margin)
    reward = torch.where(success, p.success_bonus + p.success_margin_weight * margin, reward)
    if p.collision_penalty > 0.0:
        reward = reward - p.collision_penalty * obstacle_force * TICK_DT
    return reward / 10.0


def rot6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


# --------------------------------------------------------------- metrics --


@dataclasses.dataclass(frozen=True)
class MetricSpec:
    """(reported name, predicate key): ``once`` ORs over the episode, ``last``
    and ``value`` take the final tick's value (bool / float)."""

    once: tuple[tuple[str, str], ...] = ()
    last: tuple[tuple[str, str], ...] = ()
    value: tuple[tuple[str, str], ...] = ()


class EpisodeMetrics:
    """Running per-env flags plus a window accumulator, drained destructively."""

    def __init__(self, spec: MetricSpec, num_envs: int, device):
        self.spec = spec
        self.once = {n: torch.zeros(num_envs, dtype=torch.bool, device=device) for n, _ in spec.once}
        self.last = {n: torch.zeros(num_envs, dtype=torch.bool, device=device) for n, _ in spec.last}
        self.value = {n: torch.zeros(num_envs, dtype=torch.float32, device=device) for n, _ in spec.value}
        self._zero_window()

    def _zero_window(self) -> None:
        self._finished = 0
        self._sums = {n: 0.0 for n in [*self.once, *self.last, *self.value]}

    def update(self, pred: dict) -> None:
        for n, k in self.spec.once:
            self.once[n] |= pred[k]
        for n, k in self.spec.last:
            self.last[n] = pred[k]
        for n, k in self.spec.value:
            self.value[n] = pred[k].float()

    def harvest(self, env, env_ids: torch.Tensor) -> None:
        finished = env_ids[env.episode_length_buf[env_ids] > 0]
        if finished.numel() == 0:
            return
        self._finished += int(finished.numel())
        for store in (self.once, self.last, self.value):
            for n, flags in store.items():
                self._sums[n] += float(flags[finished].sum().item())

    def clear(self, env_ids: torch.Tensor) -> None:
        for store in (self.once, self.last):
            for flags in store.values():
                flags[env_ids] = False
        for flags in self.value.values():
            flags[env_ids] = 0.0

    def drain(self) -> dict[str, float]:
        count = self._finished
        out = {"episodes": float(count)}
        for n, total in self._sums.items():
            out[n] = total / count if count else float("nan")
        self._zero_window()
        return out


METRIC_SPEC = MetricSpec(
    once=(
        ("success_once", "success"),
        ("grasped_once", "is_grasped"),
        ("lifted_once", "height_reached"),
        ("contact_once", "contact"),
    ),
    last=(
        ("success_at_end", "success"),
        ("is_grasped", "is_grasped"),
        ("height_reached", "height_reached"),
        ("lost_grip", "lost_grip"),
    ),
    value=(
        ("grasp_dist", "grasp_dist"),
        ("grasp_align", "grasp_align"),
        ("grasp_depth", "grasp_depth"),
        ("dwell_fraction", "dwell_fraction"),
        ("bowl_height", "bowl_height"),
        ("bowl_dist", "bowl_dist"),
        # collision cost, per episode
        ("obstacle_impulse", "impulse"),
        ("contact_ticks", "contact_ticks"),
        ("max_obstacle_force", "max_force"),
        ("block_disp", "block_disp"),
        ("banana_disp", "banana_disp"),
    ),
)


# ------------------------------------------------------------ task state --


class TaskState:
    """Per-env episode state: handles, dwell, cost accumulators, metrics."""

    def __init__(self, env: ManagerBasedRLEnv, params: TaskParams):
        self.params = params
        self.num_envs = env.num_envs
        self.device = env.device
        robot = env.scene["robot"]
        self.arm_ids, _ = robot.find_joints(list(G.ARM_JOINT_NAMES), preserve_order=True)
        self.finger_joint_ids, _ = robot.find_joints([G.FINGER_JOINT_NAME])
        self.gripper_joint_ids, _ = robot.find_joints(list(G.GRIPPER_JOINT_NAMES), preserve_order=True)
        self.joint_order = list(self.arm_ids) + list(self.finger_joint_ids)
        self.flange_id = robot.find_bodies(G.FLANGE_BODY_NAME)[0][0]
        self.finger_body_ids, _ = robot.find_bodies(list(G.FINGER_BODY_NAMES), preserve_order=True)
        for name in ("bowl", *G.DISTRACTOR_NAMES):
            assert name in env.scene.rigid_objects, f"scene is missing {name!r}"
        self.obstacle_sensors = [f"contact_{n}" for n in G.OBSTACLE_NAMES]
        for name in ("contact_left", "contact_right", *self.obstacle_sensors):
            assert name in env.scene.sensors, f"scene is missing sensor {name!r}"

        dev = self.device
        self.bowl_rest_z = float(G.BOWL_POS[2])
        self.distractor_rest = {
            "block": torch.tensor(G.BLOCK_POS, device=dev),
            "banana": torch.tensor(G.BANANA_POS, device=dev),
        }
        self.metrics = EpisodeMetrics(METRIC_SPEC, self.num_envs, dev)
        self.dwell = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.ever_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self.impulse = torch.zeros(self.num_envs, device=dev)
        self.contact_ticks = torch.zeros(self.num_envs, device=dev)
        self.max_force = torch.zeros(self.num_envs, device=dev)
        self._cache: dict | None = None

    # -- raw readings ------------------------------------------------------

    @staticmethod
    def _sensor_force(env, name: str) -> torch.Tensor:
        """Filtered contact force magnitude summed over the sensor's bodies and
        filter shapes: ``(N,)`` newtons."""
        m = env.scene.sensors[name].data.force_matrix_w  # (N, B, M, 3)
        return torch.linalg.vector_norm(m, dim=-1).sum(dim=(1, 2))

    def joint_state(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        robot = env.scene["robot"]
        return robot.data.joint_pos[:, self.joint_order], robot.data.joint_vel[:, self.joint_order]

    def arm_q(self, env) -> torch.Tensor:
        return env.scene["robot"].data.joint_pos[:, self.arm_ids]

    # -- predicates --------------------------------------------------------

    def predicates(self, env: ManagerBasedRLEnv, advance: bool = False) -> dict[str, torch.Tensor]:
        p = self.params
        origins = env.scene.env_origins
        robot = env.scene["robot"]

        flange_pos = robot.data.body_link_pos_w[:, self.flange_id] - origins
        R_tool = quat_to_matrix(robot.data.body_link_quat_w[:, self.flange_id])
        tcp_pos = flange_pos + p.tcp_offset * R_tool[:, :, 2]
        finger_pos = robot.data.body_link_pos_w[:, self.finger_body_ids] - origins.unsqueeze(1)

        bowl = env.scene["bowl"].data
        # Clamped to the scene's scale: a bowl knocked off the table must not
        # put metre-scale numbers into the observation.
        bowl_pos = (bowl.root_pos_w - origins).clamp(-2.0, 2.0)
        bowl_quat = bowl.root_quat_w
        bowl_lin_vel = bowl.root_lin_vel_w.clamp(-5.0, 5.0)
        bowl_ang_vel = bowl.root_ang_vel_w.clamp(-30.0, 30.0)
        rim_point, radial, axis = rim_geometry(tcp_pos, bowl_pos, bowl_quat)
        depth = grasp_depth(tcp_pos, rim_point, axis)
        # 0 at `grasp_depth_slack` above the rim plane, 1 from `grasp_depth - 2 * grasp_depth_slack` below it
        depth_ok = ((depth + p.grasp_depth_slack) / (p.grasp_depth - p.grasp_depth_slack)).clamp(0.0, 1.0)
        target = rim_point - p.grasp_depth * axis  # pads-mid below the rim plane: a grasp that lifts
        grasp_dist = torch.linalg.vector_norm(tcp_pos - target, dim=-1)
        grasp_align = grasp_alignment(R_tool, radial, axis)

        # Contact data is what PhysX reported on its last step; at the reset
        # observation no step has run since the teleport, so an env with
        # episode_length_buf == 0 would still show the previous episode's
        # final contacts. A just-reset env touches nothing by construction.
        fresh = (env.episode_length_buf == 0).float()
        left = self._sensor_force(env, "contact_left") * (1.0 - fresh)
        right = self._sensor_force(env, "contact_right") * (1.0 - fresh)
        is_grasped = (left > p.grasp_force_threshold) & (right > p.grasp_force_threshold)

        bowl_height = bowl_pos[:, 2] - self.bowl_rest_z
        height_reached = bowl_height >= p.lift_height
        dist_to_lift = (p.lift_height - bowl_height).clamp(min=0.0)
        height_margin = (bowl_height - p.lift_height).clamp(min=0.0)

        obstacle_force = torch.stack([self._sensor_force(env, s) for s in self.obstacle_sensors], dim=0).sum(0) * (1.0 - fresh)
        contact = obstacle_force > p.obstacle_force_threshold
        disp = {
            n: torch.linalg.vector_norm(env.scene[n].data.root_pos_w - origins - self.distractor_rest[n], dim=-1).clamp(max=5.0)
            for n in G.DISTRACTOR_NAMES
        }

        if advance:
            self.ever_grasped |= is_grasped
            self.dwell += 1
            self.dwell[~(is_grasped & height_reached)] = 0
            self.impulse += obstacle_force * TICK_DT
            self.contact_ticks += contact.float()
            self.max_force = torch.maximum(self.max_force, obstacle_force)

        return {
            "tcp_pos": tcp_pos,
            "R_tool": R_tool,
            "finger_pos": finger_pos,
            "bowl_pos": bowl_pos,
            "bowl_quat": bowl_quat,
            "bowl_lin_vel": bowl_lin_vel,
            "bowl_ang_vel": bowl_ang_vel,
            "bowl_dist": torch.linalg.vector_norm(bowl_pos, dim=-1),
            "rim_point": target,
            "radial": radial,
            "grasp_dist": grasp_dist,
            "grasp_align": grasp_align,
            "grasp_depth": depth,
            "grasp_depth_ok": depth_ok,
            "finger_forces": (left, right),
            "is_grasped": is_grasped,
            "bowl_height": bowl_height,
            "height_reached": height_reached,
            "dist_to_lift": dist_to_lift,
            "height_margin": height_margin,
            "obstacle_force": obstacle_force,
            "contact": contact,
            "impulse": self.impulse,
            "contact_ticks": self.contact_ticks,
            "max_force": self.max_force,
            "block_disp": disp["block"],
            "banana_disp": disp["banana"],
            "lost_grip": self.ever_grasped & ~is_grasped,
            "dwell_fraction": (self.dwell.float() / p.dwell_target).clamp(max=1.0),
            "success": self.dwell >= p.dwell_target,
        }

    def refresh(self, env, advance: bool = False) -> dict:
        self._cache = self.predicates(env, advance=advance)
        return self._cache

    def cached(self, env) -> dict:
        if self._cache is None:
            self._cache = self.predicates(env)
        return self._cache

    def invalidate(self) -> None:
        self._cache = None

    # -- episode bookkeeping ----------------------------------------------

    def harvest(self, env, env_ids: torch.Tensor) -> None:
        self.metrics.harvest(env, env_ids)

    def clear(self, env_ids: torch.Tensor) -> None:
        self.dwell[env_ids] = 0
        self.ever_grasped[env_ids] = False
        self.impulse[env_ids] = 0.0
        self.contact_ticks[env_ids] = 0.0
        self.max_force[env_ids] = 0.0
        self.metrics.clear(env_ids)

    def drain(self) -> dict[str, float]:
        return self.metrics.drain()


# ---------------------------------------------------------- observations --


def _arm_joint(env, state: TaskState) -> torch.Tensor:
    """qpos (8), qvel (8), pad opening (1) = 17."""
    qpos, qvel = state.joint_state(env)
    pads = state.cached(env)["finger_pos"]
    opening = torch.linalg.vector_norm(pads[:, 0] - pads[:, 1], dim=-1, keepdim=True)
    return torch.cat([qpos, qvel, opening], dim=-1)


def _tcp(env, state) -> torch.Tensor:
    pred = state.cached(env)
    return torch.cat([pred["tcp_pos"], rot6d(pred["R_tool"])], dim=-1)


def _fingertips(env, state) -> torch.Tensor:
    return state.cached(env)["finger_pos"].flatten(1)


def _bowl_state(env, state) -> torch.Tensor:
    pred = state.cached(env)
    return torch.cat(
        [pred["bowl_pos"], rot6d(quat_to_matrix(pred["bowl_quat"])), pred["bowl_lin_vel"], pred["bowl_ang_vel"]],
        dim=-1,
    )


def _rim(env, state) -> torch.Tensor:
    """Grasp target minus TCP (3) and the outward radial there (3)."""
    pred = state.cached(env)
    return torch.cat([pred["rim_point"] - pred["tcp_pos"], pred["radial"]], dim=-1)


def _phase(env, state) -> torch.Tensor:
    pred = state.cached(env)
    return torch.stack(
        [pred["is_grasped"].float(), pred["height_reached"].float(), pred["dwell_fraction"], pred["bowl_height"],
         pred["grasp_depth_ok"]],
        dim=-1,
    )


def _engaged(env, state) -> torch.Tensor:
    left, right = state.cached(env)["finger_forces"]
    t = state.params.engaged_force_threshold
    return torch.stack([(left > t).float(), (right > t).float()], dim=-1)


FIELDS: dict[str, Callable] = {
    "arm_joint": _arm_joint,
    "tcp": _tcp,
    "fingertips": _fingertips,
    "bowl_state": _bowl_state,
    "rim": _rim,
    "phase": _phase,
    "engaged": _engaged,
}
#: 17 + 9 + 6 + 15 + 6 + 5 + 2 = 60 features. The order is the checkpoint's contract.
POLICY_FIELDS: tuple[str, ...] = ("arm_joint", "tcp", "fingertips", "bowl_state", "rim", "phase", "engaged")
