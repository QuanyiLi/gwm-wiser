"""Manager terms: observation, reward, reset events. Thin adapters over `task.py`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from gwm_rl import geometry as G
from gwm_rl.task import FIELDS, TaskState, pick_reward, resolve_params

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

_STATE_ATTR = "_gwm_rl_task_state"


def task_state(env) -> TaskState:
    """The env's task state, built at first use (the ObservationManager's width
    probe during ``gym.make``, in practice) once the scene exists."""
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        state = TaskState(env, resolve_params(getattr(env.cfg, "task_params", None)))
        setattr(env, _STATE_ATTR, state)
    return state


def observation_fields(env: ManagerBasedRLEnv, fields: Sequence[str]) -> torch.Tensor:
    state = task_state(env)
    parts = [FIELDS[name](env, state) for name in fields]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def pick_reward_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The reward, and the once-per-tick side effects only it may have: the
    dwell counter, the cost accumulators and the metric flags advance here,
    because Isaac Lab evaluates rewards exactly once per step, before resets."""
    state = task_state(env)
    pred = state.refresh(env, advance=True)
    state.metrics.update(pred)
    return pick_reward(
        grasp_dist=pred["grasp_dist"],
        grasp_align=pred["grasp_align"],
        grasp_depth_ok=pred["grasp_depth_ok"],
        is_grasped=pred["is_grasped"],
        height_reached=pred["height_reached"],
        bowl_height=pred["bowl_height"],
        height_margin=pred["height_margin"],
        dwell_fraction=pred["dwell_fraction"],
        success=pred["success"],
        obstacle_force=pred["obstacle_force"],
        params=state.params,
    )


def reset_arm_gaussian(env: ManagerBasedEnv, env_ids: torch.Tensor, std: float) -> None:
    """Arm joints at the home pose plus N(0, std); every gripper joint exactly open."""
    robot = env.scene["robot"]
    state = task_state(env)
    qpos = robot.data.default_joint_pos[env_ids].clone()
    qpos[:, state.arm_ids] += torch.randn_like(qpos[:, state.arm_ids]) * std
    qpos[:, state.gripper_joint_ids] = G.GRIPPER_OPEN
    limits = robot.data.soft_joint_pos_limits[env_ids]
    qpos = torch.clamp(qpos, limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(qpos, torch.zeros_like(qpos), env_ids=env_ids)
    # The PD targets would otherwise still point at the previous episode's last command.
    robot.set_joint_position_target(qpos, env_ids=env_ids)


def reset_episode(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Bank the finished episodes' metrics and clear their running state."""
    state = task_state(env)
    state.harvest(env, env_ids)
    state.clear(env_ids)
    state.invalidate()
