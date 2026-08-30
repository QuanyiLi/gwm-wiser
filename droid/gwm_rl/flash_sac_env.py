"""Adapter between an executor (`executors.py`) and FlashSAC's flat-vector
contract; the observation, staggering and truncation-bootstrap rules of
`isaaclab_M3/utils/flash_sac_env.py`.

- Observations: the named groups are concatenated into one float32 tensor,
  ``[actor groups | critic-only tail]`` (FlashSAC's asymmetric mode slices the
  actor's input off the front).
- Staggered time limits: every env would otherwise reset together and the
  replay buffer would hold one phase of the task at a time.
- Truncation bootstrap: Isaac Lab resets a timed-out env inside ``step`` and
  returns the *next* episode's first observation, so on done rows
  ``info["bootstrap_next_obs"]`` carries ``o_t`` instead — the critic
  bootstraps ``gamma * V(s_t)``. Episodes here never terminate early, so the
  held state's value barely moves in one step and the approximation is close.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


class FlashSacWrapper:
    def __init__(self, executor: Any, *, actor_groups: Sequence[str], critic_groups: Sequence[str],
                 stagger_time_limits: bool = True, cfg_to_save: dict | None = None):
        import gymnasium as gym

        self.executor = executor
        self.unwrapped = executor.unwrapped
        self.device = executor.device
        self.num_envs = int(executor.num_envs)
        self.num_actions = int(executor.num_actions)
        self.max_episode_length = int(executor.max_episode_length)
        self.ticks_per_step = int(getattr(executor, "ticks_per_step", 1))
        self.cfg_to_save = cfg_to_save or {}
        self.actor_groups = tuple(actor_groups)
        self.critic_tail = tuple(g for g in critic_groups if g not in self.actor_groups)
        self.asymmetric_obs = bool(self.critic_tail)
        self._stagger = stagger_time_limits

        obs, _ = self.executor.reset()
        self.num_actor_obs = sum(int(obs[g].shape[-1]) for g in self.actor_groups)
        self.num_obs = self.num_actor_obs + sum(int(obs[g].shape[-1]) for g in self.critic_tail)
        self._last_obs = self._flatten(obs)

        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (self.num_obs,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.num_envs, self.num_obs), np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (self.num_actions,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.num_envs, self.num_actions), np.float32)

    def _flatten(self, obs: dict) -> torch.Tensor:
        parts = [obs[g] for g in self.actor_groups] + [obs[g] for g in self.critic_tail]
        flat = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return flat.to(dtype=torch.float32)

    @property
    def env_info(self) -> dict[str, Any]:
        return {"actor_observation_size": (self.num_actor_obs,), "asymmetric_obs": self.asymmetric_obs}

    def reset(self, *args, stagger: bool | None = None, **kwargs):
        obs, _ = self.executor.reset(*args, **kwargs)
        self._last_obs = self._flatten(obs)
        if self._stagger if stagger is None else stagger:
            self.stagger_time_limits()
        return self._last_obs, self.env_info

    def stagger_time_limits(self) -> None:
        self.executor.stagger()

    def random_actions(self) -> torch.Tensor:
        return torch.rand(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32) * 2.0 - 1.0

    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        obs, reward, terminated, truncated, _ = self.executor.step(actions)
        next_obs = self._flatten(obs)
        next_obs, reward = self._guard(next_obs, reward)
        done = terminated | truncated
        bootstrap = next_obs
        if bool(done.any()):
            bootstrap = torch.where(done.unsqueeze(-1), self._last_obs, next_obs)
        self._last_obs = next_obs
        return next_obs, reward, terminated, truncated, {"bootstrap_next_obs": bootstrap}

    def _guard(self, obs: torch.Tensor, reward: torch.Tensor):
        """A non-finite observation or reward would poison the replay buffer and
        crash the categorical critic (NaN -> bin index). Report the first few
        occurrences with the env and the columns involved, then sanitize."""
        bad_obs = ~torch.isfinite(obs)
        bad_rew = ~torch.isfinite(reward)
        if bool(bad_obs.any()) or bool(bad_rew.any()):
            self.num_nonfinite = getattr(self, "num_nonfinite", 0) + 1
            if self.num_nonfinite <= 5:
                envs = torch.nonzero(bad_obs.any(dim=-1) | bad_rew).flatten()[:8].tolist()
                cols = torch.nonzero(bad_obs.any(dim=0)).flatten()[:20].tolist()
                print(f"[guard] non-finite values: envs {envs} obs columns {cols} "
                      f"rewards {int(bad_rew.sum())}; sanitizing", flush=True)
                try:
                    from gwm_rl.mdp import task_state
                    st = task_state(self.unwrapped)
                    pred = st.cached(self.unwrapped)
                    e = envs[0]
                    print(f"[guard] env {e}: bowl {pred['bowl_pos'][e].tolist()} quat {pred['bowl_quat'][e].tolist()} "
                          f"v {pred['bowl_lin_vel'][e].tolist()} w {pred['bowl_ang_vel'][e].tolist()} "
                          f"tcp {pred['tcp_pos'][e].tolist()} q {self.unwrapped.scene['robot'].data.joint_pos[e].tolist()}",
                          flush=True)
                except Exception as exc:  # diagnostics must never take the run down
                    print(f"[guard] detail unavailable: {exc}", flush=True)
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, reward

    def close(self) -> None:
        self.executor.close()
