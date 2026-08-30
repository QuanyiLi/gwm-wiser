"""The end-effector action space over the joint-target env.

The env (`env_cfg.py`) takes ``[7 absolute joint targets, gripper]`` per 15 Hz
tick. The policy acts in end-effector space with the tool held top-down:
:class:`MacroExecutor` turns one decision per ``n_move + n_hold`` ticks — an
**absolute** target ``(x, y, z, yaw)`` in the workspace box plus the gripper
state to switch to once the arm has arrived — into joint targets with
`franka_kin`, the same functions the GWM scoring client uses to turn a target
pose into the joint trajectory it renders, so what is scored is what is
executed. The arm moves on a straight line for ``n_move`` ticks, then holds
while the gripper goes to ``g``. The reward is the mean of the per-tick
rewards over the hold phase (the state the target reached); time-outs are kept
on the macro-step grid so an episode never ends inside a step.

The executor exposes the surface `FlashSacWrapper` needs: ``num_envs``,
``num_actions``, ``max_episode_length`` (in macro-steps), ``reset`` / ``step``
/ ``stagger``.
"""

from __future__ import annotations

from typing import Any

import torch

from gwm_rl import franka_kin as K
from gwm_rl import geometry as G
from gwm_rl.mdp import task_state


def _box_map(a: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """[-1, 1] -> [lo, hi]; inputs outside are clipped to the box."""
    return lo + (a.clamp(-1.0, 1.0) + 1.0) * 0.5 * (hi - lo)


class _Executor:
    def __init__(self, env: Any):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.num_envs = int(self.unwrapped.num_envs)
        self.state = task_state(self.unwrapped)
        self.robot = self.unwrapped.scene["robot"]
        self.ticks_per_step = 1
        self._closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # -- helpers
    def _arm_q(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.state.arm_ids]

    def _inner_action(self, q_cmd: torch.Tensor, closed: torch.Tensor) -> torch.Tensor:
        g = torch.where(closed, -1.0, 1.0).unsqueeze(-1).to(q_cmd.dtype)
        return torch.cat([q_cmd, g], dim=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._closed.zero_()
        return obs, info

    def close(self) -> None:
        self.env.close()

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    def stagger(self) -> None:
        buf = self.episode_length_buf
        buf.copy_(torch.randint_like(buf, high=self.max_episode_length) * self.ticks_per_step)


class MacroExecutor(_Executor):
    """Absolute target pose per macro-step: action ``(N, 5)`` in [-1, 1]."""

    num_actions = 5

    def __init__(self, env: Any, *, n_move: int = 30, n_hold: int = 14, reward_phase: str = "hold", ik_kwargs=None):
        super().__init__(env)
        self.n_move, self.n_hold = int(n_move), int(n_hold)
        if reward_phase not in ("hold", "all"):
            raise ValueError(f"reward_phase must be 'hold' or 'all', got {reward_phase!r}")
        self.reward_phase = reward_phase
        self.ticks_per_step = self.n_move + self.n_hold
        self.ik_kwargs = dict(ik_kwargs or {})
        inner = int(self.unwrapped.max_episode_length)
        if inner % self.ticks_per_step:
            raise ValueError(
                f"episode of {inner} ticks is not a whole number of {self.ticks_per_step}-tick macro-steps"
            )
        self.max_episode_length = inner // self.ticks_per_step
        self._box_lo = torch.tensor([G.WORKSPACE[k][0] for k in "xyz"], device=self.device)
        self._box_hi = torch.tensor([G.WORKSPACE[k][1] for k in "xyz"], device=self.device)

    def decode(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(target position (N, 3), target yaw (N,), close (N,)) from a normalized action."""
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        p_t = self._box_lo + (action[:, :3].clamp(-1, 1) + 1.0) * 0.5 * (self._box_hi - self._box_lo)
        yaw_t = _box_map(action[:, 3], *G.WORKSPACE["yaw"])
        return p_t, yaw_t, action[:, 4] < 0.0

    def step(self, action: torch.Tensor):
        p_t, yaw_t, close_after = self.decode(action)
        tcp = self.state.params.tcp_offset
        q0 = self._arm_q()
        p0, R0 = K.fk_tcp(q0, tcp)
        path_p, path_yaw = K.interpolate_pose(p0, K.yaw_of(R0), p_t, yaw_t, self.n_move)
        R_t = K.target_rotation(path_yaw[:, -1])  # yaw_t modulo pi, on the current side
        reward_sum = torch.zeros(self.num_envs, device=self.device)
        terminated_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated_any = torch.zeros_like(terminated_any)
        for k in range(self.ticks_per_step):
            q = self._arm_q()
            if k < self.n_move:
                q_cmd = K.ik_step(q, path_p[:, k], K.target_rotation(path_yaw[:, k]), tcp_offset=tcp, **self.ik_kwargs)
                closed = self._closed
            else:
                q_cmd = K.ik_step(q, p_t, R_t, tcp_offset=tcp, **self.ik_kwargs)
                closed = close_after
            obs, reward, terminated, truncated, info = self.env.step(self._inner_action(q_cmd, closed))
            if self.reward_phase == "all" or k >= self.n_move:
                reward_sum += reward
            terminated_any |= terminated
            truncated_any |= truncated
        done = terminated_any | truncated_any
        # After a reset the jaw is open; otherwise the commanded state carries over.
        self._closed = close_after & ~done
        info = dict(info or {})
        info["ticks"] = self.ticks_per_step
        n = self.ticks_per_step if self.reward_phase == "all" else self.n_hold
        return obs, reward_sum / n, terminated_any, truncated_any, info


def make_executor(env: Any, executor_cfg: dict | None = None) -> MacroExecutor:
    return MacroExecutor(env, **dict(executor_cfg or {}))
