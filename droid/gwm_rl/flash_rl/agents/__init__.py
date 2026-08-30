from typing import Any

import gymnasium as gym

from gwm_rl.flash_rl.agents.base_agent import BaseAgent
from gwm_rl.flash_rl.types import NDArray


def create_agent(
    observation_space: gym.spaces.Space[NDArray],
    action_space: gym.spaces.Space[NDArray],
    env_info: dict[str, Any],
    cfg: dict[str, Any],
) -> BaseAgent[Any]:
    """Build the FlashSAC agent from a plain config dict (no hydra)."""
    from gwm_rl.flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig

    cfg_dict = {str(k): v for k, v in dict(cfg).items()}
    cfg_dict.pop("agent_type", None)
    return FlashSACAgent(observation_space, action_space, env_info, FlashSACConfig(**cfg_dict))
