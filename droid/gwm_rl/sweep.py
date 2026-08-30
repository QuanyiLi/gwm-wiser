"""The deterministic sweep: one full-horizon episode in every env, fresh
unstaggered reset, deterministic policy; the metrics drained after it cover
exactly those episodes. Shared by the trainer and the checkpoint evaluator."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def policy_step(wrapper: Any, agent: Any, iteration: int = 0) -> Callable[[Any], Any]:
    def step(obs):
        actions = agent.sample_actions(iteration, {"next_observation": obs}, training=False)
        return wrapper.step(actions)[0]

    return step


def deterministic_episode(wrapper: Any, state: Any, step: Callable[[Any], Any], *, seed: int | None = None) -> dict:
    obs, _ = wrapper.reset(stagger=False) if seed is None else wrapper.reset(seed=seed, stagger=False)
    state.drain()  # the partial episodes the reset interrupted are not episodes
    for _ in range(wrapper.max_episode_length):
        obs = step(obs)
    return state.drain()


def aggregate(summaries: Sequence[dict]) -> dict[str, dict[str, float]]:
    if not summaries:
        return {}
    out: dict[str, dict[str, float]] = {}
    for key in sorted(summaries[0]):
        values = [float(s[key]) for s in summaries if key in s]
        out[key] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    return out
