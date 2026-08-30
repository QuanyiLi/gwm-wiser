"""Evaluate a checkpoint: deterministic sweeps, fresh episodes, mean +/- sd.

    OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 \\
    ../droid-sim-evals/.venv/bin/python eval.py experiments/pickbowl-s1 --headless --episodes 4

The run directory's ``config.yaml`` says which recipe and env overrides to
rebuild the agent with; ``models/best`` is loaded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", help="run directory (contains config.yaml and models/best)")
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--num-envs", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402

import gwm_rl.env_cfg as E  # noqa: E402
from gwm_rl.executors import make_executor  # noqa: E402
from gwm_rl.flash_sac_agent import build_agent, resolve_obs_groups  # noqa: E402
from gwm_rl.flash_sac_env import FlashSacWrapper  # noqa: E402
from gwm_rl.mdp import task_state  # noqa: E402
from gwm_rl.run_config import set_seed  # noqa: E402
from gwm_rl.sweep import aggregate, deterministic_episode, policy_step  # noqa: E402


def main() -> None:
    run = Path(args.run)
    saved = yaml.safe_load((run / "config.yaml").read_text())
    recipe, flash_cfg, env_saved = saved["recipe"], saved["flash_sac"], saved["env"]
    set_seed(int(recipe["seed"]) + 1000)
    num_envs = args.num_envs or int(env_saved["num_envs"])
    env_cfg = E.make_env_cfg(num_envs=num_envs, overrides=env_saved.get("env_set", []), seed=int(recipe["seed"]) + 1000)
    env = gym.make(E.TASK_ID, cfg=env_cfg)
    executor = make_executor(env, recipe.get("executor"))
    actor_groups, critic_groups = resolve_obs_groups(recipe)
    wrapper = FlashSacWrapper(executor, actor_groups=actor_groups, critic_groups=critic_groups)
    flash_cfg = dict(flash_cfg, load_optimizer=False)
    agent = build_agent(wrapper, flash_cfg)
    agent.load(str(run / "models" / "best"))
    state = task_state(wrapper.unwrapped)
    step = policy_step(wrapper, agent)
    episodes = [deterministic_episode(wrapper, state, step) for _ in range(args.episodes)]
    final = aggregate(episodes)
    print(f"\n=== {run.name}: best checkpoint on {args.episodes} x {num_envs} fresh episodes ===")
    for k in ("success_at_end", "success_once", "grasped_once", "lifted_once", "lost_grip",
              "obstacle_impulse", "contact_ticks", "block_disp", "banana_disp"):
        if k in final:
            print(f"  {k:18s} {final[k]['mean']:.4f} +/- {final[k]['sd']:.4f}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # No simulation_app.close(): Kit's shutdown hangs after this scene; the
    # process exit frees the GPU.
    os._exit(0)
