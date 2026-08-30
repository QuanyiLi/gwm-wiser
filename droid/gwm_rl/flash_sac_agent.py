"""Build the vendored FlashSAC agent from a recipe dict — no hydra.

The recipe (`configs/*.yaml`) lists the deviations from FlashSAC's published
defaults; everything else is filled in here with the values of
`FlashSAC/configs/agent/flashSAC.yaml`, so training and evaluation build
identical agents and a checkpoint loads back into what produced it.
"""

from __future__ import annotations

from typing import Any

_NUMERIC = {
    "num_envs": int, "wall_budget_s": float, "lr_horizon_env_steps": float, "eval_interval_s": float,
    "stop_at_success": float, "stop_patience": int, "min_explore_success": float, "seed": int,
}
_NUMERIC_AGENT = {
    "replay_ratio": float, "sample_batch_size": int, "buffer_capacity": float, "warmup_transitions": float,
    "n_step": int, "gamma": float, "temp_target_sigma": float, "temp_initial_value": float,
    "learning_rate": float, "learning_rate_end": float, "critic_hidden_dim": int, "actor_hidden_dim": int,
    "num_blocks": int, "actor_update_period": int, "critic_target_update_tau": float, "normalized_G_max": float,
    "critic_num_bins": int, "actor_noise_zeta_mu": float, "actor_noise_zeta_max": int, "actor_bc_alpha": float,
}


def coerce_numbers(cfg: dict) -> dict:
    """YAML 1.1 reads ``8e6`` as a string; force the numeric leaves."""
    for key, cast in _NUMERIC.items():
        if key in cfg:
            cfg[key] = cast(cfg[key])
    for key, cast in _NUMERIC_AGENT.items():
        if key in cfg.get("agent", {}):
            cfg["agent"][key] = cast(cfg["agent"][key])
    return cfg


def resolve_obs_groups(cfg: dict) -> tuple[list[str], list[str]]:
    groups = cfg["obs_groups"]
    return list(groups["policy"]), list(groups["critic"])


def updates_per_step(cfg: dict, num_envs: int) -> int:
    """Gradient updates per interaction step: replay ratio counts gradient
    samples per collected transition, so this scales with the env count."""
    return max(1, round(cfg["agent"]["replay_ratio"] * num_envs / cfg["agent"]["sample_batch_size"]))


def flash_sac_config(cfg: dict, wrapper: Any, updates: int) -> dict[str, Any]:
    """The full `FlashSACConfig` field dict."""
    a = cfg["agent"]
    # LR schedule horizon in *update* steps: interaction steps (transitions per
    # env the wall budget is expected to reach) times updates per step.
    horizon_transitions = float(cfg["lr_horizon_env_steps"]) / max(1, wrapper.ticks_per_step)
    interaction_steps = max(1, int(horizon_transitions / wrapper.num_envs))
    lr_steps = interaction_steps * updates
    g_max = float(a["normalized_G_max"])
    return dict(
        seed=int(cfg["seed"]),
        normalize_reward=True,
        normalized_G_max=g_max,
        asymmetric_observation=bool(wrapper.asymmetric_obs),
        device_type="cuda",
        buffer_max_length=int(a["buffer_capacity"]),
        buffer_min_length=int(a["warmup_transitions"]),
        buffer_device_type="cuda",
        sample_batch_size=int(a["sample_batch_size"]),
        learning_rate_init=float(a["learning_rate"]),
        learning_rate_peak=float(a["learning_rate"]),
        learning_rate_end=float(a["learning_rate_end"]),
        learning_rate_warmup_rate=1e-6,
        learning_rate_warmup_step=int(1e-6 * lr_steps),
        learning_rate_decay_rate=1.0,
        learning_rate_decay_step=int(lr_steps),
        actor_num_blocks=int(a["num_blocks"]),
        actor_hidden_dim=int(a["actor_hidden_dim"]),
        actor_bc_alpha=float(a.get("actor_bc_alpha", 0.0)),
        actor_noise_zeta_mu=float(a.get("actor_noise_zeta_mu", 2.0)),
        actor_noise_zeta_max=int(a.get("actor_noise_zeta_max", 16)),
        actor_update_period=int(a["actor_update_period"]),
        critic_num_blocks=int(a["num_blocks"]),
        critic_hidden_dim=int(a["critic_hidden_dim"]),
        critic_num_bins=int(a["critic_num_bins"]),
        critic_min_v=-g_max,
        critic_max_v=g_max,
        critic_target_update_tau=float(a["critic_target_update_tau"]),
        temp_initial_value=float(a.get("temp_initial_value", 0.01)),
        temp_target_sigma=float(a["temp_target_sigma"]),
        temp_target_entropy=None,  # derived from the sigma inside the agent
        gamma=float(a["gamma"]),
        n_step=int(a["n_step"]),
        use_compile=bool(a["use_compile"]),
        compile_mode="auto",
        use_amp=bool(a["use_amp"]),
        load_optimizer=True,
        load_reward_normalizer=True,
    )


def build_agent(wrapper: Any, flash_cfg: dict[str, Any]):
    from gwm_rl.flash_rl.agents import create_agent

    return create_agent(
        observation_space=wrapper.observation_space,
        action_space=wrapper.action_space,
        env_info=wrapper.env_info,
        cfg=flash_cfg,
    )
