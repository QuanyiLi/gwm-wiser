"""Train pick-up-the-bowl with FlashSAC.

    OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 \\
    ../droid-sim-evals/.venv/bin/python train.py --headless --seed 1 --final-eval-episodes 4

The loop is `isaaclab_M3/scripts/train_flash_sac.py`'s: exploring rollouts
with the replay ratio's gradient updates drained after every step, a periodic
deterministic sweep (every env reset unstaggered, one full episode with the
noise-free policy) whose transitions are kept but whose updates are deferred
so the number belongs to one policy, the best sweep's checkpoint saved before
those updates run, and a wall-clock budget as the only stop.

What is recorded per evaluation, in ``evals.json``: the sweep's success and
stage metrics, the exploring rollout's, the collision cost per episode of
both, the running totals of that cost since the start (the "cost of
learning"), and the first crossing of each success level in env steps (ticks
summed over envs — the sample-efficiency axis).

Recipe leaves override from the command line:

    ... train.py --headless --set agent.gamma=0.97 num_envs=512
    ... train.py --headless --env-set task_params.lift_height=0.12
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None, help="recipe yaml (default: configs/macro.yaml)")
parser.add_argument("--exp-name", default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--wall-budget-s", type=float, default=None)
parser.add_argument("--eval-interval-s", type=float, default=None)
parser.add_argument("--stop-at-success", type=float, default=None)
parser.add_argument("--no-time-suffix", action="store_true")
parser.add_argument("--final-eval-episodes", type=int, default=0,
                    help="after the budget, reload the best checkpoint and roll this many deterministic episodes per env")
parser.add_argument("--profile", action="store_true", help="synchronise the GPU around each phase and report the split")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides into the recipe")
parser.add_argument("--env-set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="dotted overrides into the env config, python literals")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_t0 = time.perf_counter()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
_boot_s = time.perf_counter() - _t0

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import gwm_rl.env_cfg as E  # noqa: E402
from gwm_rl.executors import make_executor  # noqa: E402
from gwm_rl.flash_sac_agent import build_agent, coerce_numbers, flash_sac_config, resolve_obs_groups, updates_per_step  # noqa: E402
from gwm_rl.flash_sac_env import FlashSacWrapper  # noqa: E402
from gwm_rl.mdp import task_state  # noqa: E402
from gwm_rl.run_config import CONFIG_DIR, load_recipe, parse_overrides, set_seed  # noqa: E402
from gwm_rl.run_loop import RunLedger, RunSchedule  # noqa: E402
from gwm_rl.sweep import aggregate, deterministic_episode  # noqa: E402

COST_KEYS = ("obstacle_impulse", "contact_ticks", "block_disp", "banana_disp")
PRINT_KEYS = ("success_at_end", "success_once", "grasped_once", "lifted_once")


class Budget(Exception):
    """Raised once the run is done (threshold cleared or clock spent)."""


class CostTotals:
    """Running totals of the per-episode cost, episodes weighted, since the start."""

    def __init__(self):
        self.episodes = 0.0
        self.sums = {k: 0.0 for k in COST_KEYS}
        self.contact_episodes = 0.0

    def add(self, window: dict) -> None:
        n = float(window.get("episodes", 0.0))
        if n <= 0:
            return
        self.episodes += n
        for k in COST_KEYS:
            v = window.get(k, float("nan"))
            if v == v:
                self.sums[k] += n * v
        c = window.get("contact_once", float("nan"))
        if c == c:
            self.contact_episodes += n * c

    def snapshot(self) -> dict:
        return {"episodes": self.episodes, "contact_episodes": self.contact_episodes, **self.sums}


def main() -> None:
    overrides = parse_overrides(args.set)
    for flag, key in (
        (args.exp_name, "experiment_name"), (args.seed, "seed"), (args.num_envs, "num_envs"),
        (args.wall_budget_s, "wall_budget_s"), (args.eval_interval_s, "eval_interval_s"),
        (args.stop_at_success, "stop_at_success"),
    ):
        if flag is not None:
            overrides[key] = flag
    recipe_path = Path(args.config) if args.config else CONFIG_DIR / "macro.yaml"
    cfg = coerce_numbers(load_recipe(recipe_path, exp_name_add_time=not args.no_time_suffix, overrides=overrides))
    seed = set_seed(int(cfg["seed"]))

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    num_envs = int(cfg["num_envs"])
    actor_groups, critic_groups = resolve_obs_groups(cfg)
    env_cfg = E.make_env_cfg(num_envs=num_envs, overrides=args.env_set, seed=seed)
    _t_build = time.perf_counter()
    env = gym.make(E.TASK_ID, cfg=env_cfg)
    build_s = time.perf_counter() - _t_build

    executor = make_executor(env, cfg.get("executor"))
    wrapper = FlashSacWrapper(
        executor, actor_groups=actor_groups, critic_groups=critic_groups,
        cfg_to_save=dict(num_envs=num_envs, env_set=list(args.env_set), recipe=copy.deepcopy(cfg)),
    )
    state = task_state(wrapper.unwrapped)
    updates = updates_per_step(cfg, num_envs)
    flash_cfg = flash_sac_config(cfg, wrapper, updates)
    agent = build_agent(wrapper, flash_cfg)

    exp_dir = Path(cfg["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(
            {"launcher": vars(args), "recipe": cfg, "flash_sac": flash_cfg,
             "task_params": dict(vars(state.params)), "env": wrapper.cfg_to_save},
            handle, default_flow_style=False, sort_keys=False,
        )

    n_params = sum(p.numel() for net in (agent._critic, agent._actor) for p in net.network.parameters())
    ticks = wrapper.ticks_per_step
    print(
        f"[train] {num_envs} envs x {wrapper.max_episode_length} macro-steps "
        f"({wrapper.max_episode_length * ticks} ticks) | obs {wrapper.num_obs} | act {wrapper.num_actions}\n"
        f"[train] batch {flash_cfg['sample_batch_size']}, {updates} updates/iter "
        f"(replay ratio {cfg['agent']['replay_ratio']:g}), n_step {flash_cfg['n_step']}, gamma {flash_cfg['gamma']}, "
        f"sigma {flash_cfg['temp_target_sigma']}, warmup {flash_cfg['buffer_min_length']} transitions, "
        f"{n_params / 1e6:.2f}M params\n"
        f"[train] wall budget {cfg['wall_budget_s']:.0f}s, eval every {cfg['eval_interval_s']:.0f}s, "
        f"stop at {cfg['stop_metric']} >= {cfg['stop_at_success']}\n"
        f"[train] logging to {exp_dir} (app boot {_boot_s:.1f}s, scene build {build_s:.1f}s)",
        flush=True,
    )

    schedule = RunSchedule(
        eval_interval_s=cfg["eval_interval_s"], wall_budget_s=cfg["wall_budget_s"], stop_at=cfg["stop_at_success"],
        patience=cfg["stop_patience"], min_explore_success=cfg["min_explore_success"],
    )
    ledger = RunLedger(
        exp_dir / "evals.json", metric=cfg["stop_metric"],
        static={
            "args": vars(args), "config": cfg,
            "setup": {"app_boot_s": round(_boot_s, 1), "scene_build_s": round(build_s, 1),
                      "updates_per_iteration": updates, "obs_dim": wrapper.num_obs, "act_dim": wrapper.num_actions,
                      "ticks_per_step": ticks, "params_M": round(n_params / 1e6, 3)},
        },
    )
    timing = {"act": 0.0, "step": 0.0, "update": 0.0}
    counters = {"env_steps": 0, "transitions": 0, "iterations": 0, "updates": 0}
    cost_explore, cost_sweep = CostTotals(), CostTotals()
    update_counter = 0.0
    last_mark = (0.0, 0)

    obs, _ = wrapper.reset()
    state.drain()
    started = time.time()

    def _tick() -> float:
        if args.profile:
            torch.cuda.synchronize()
        return time.perf_counter()

    def drain_updates() -> None:
        nonlocal update_counter
        if not agent.can_start_training():
            update_counter = 0.0
            return
        t = _tick()
        while update_counter >= 1:
            agent.update()
            update_counter -= 1
            counters["updates"] += 1
        timing["update"] += _tick() - t

    def interact(it: int, *, exploring: bool, collect: bool, train: bool = True) -> None:
        nonlocal obs, update_counter
        t = _tick()
        if exploring and not agent.can_start_training():
            actions = wrapper.random_actions()
        else:
            actions = torch.as_tensor(
                agent.sample_actions(it, {"next_observation": obs}, training=exploring), device=wrapper.device
            )
        t_act = _tick()
        next_obs, reward, terminated, truncated, info = wrapper.step(actions)
        t_step = _tick()
        if collect:
            agent.process_transition({
                "observation": obs, "action": actions, "reward": reward, "terminated": terminated,
                "truncated": truncated, "next_observation": info["bootstrap_next_obs"],
            })
            counters["transitions"] += wrapper.num_envs
            counters["env_steps"] += wrapper.num_envs * ticks
            if agent.can_start_training():
                update_counter += updates
        obs = next_obs
        if train:
            drain_updates()
        counters["iterations"] += 1
        timing["act"] += t_act - t
        timing["step"] += t_step - t_act

    def deterministic_sweep(it: int, collect: bool | None = None) -> dict:
        nonlocal obs
        if collect is None:
            collect = bool(cfg["collect_during_eval"])

        def step(current):
            nonlocal obs
            obs = current
            interact(it, exploring=False, collect=collect, train=False)
            return obs

        summary = deterministic_episode(wrapper, state, step)
        if collect:
            cost_sweep.add(summary)
        wrapper.stagger_time_limits()
        return summary

    def write_evals(final: dict | None = None) -> None:
        elapsed = time.time() - started
        share = {k: round(v / max(elapsed, 1e-9), 4) for k, v in timing.items()}
        ledger.write(
            elapsed=elapsed, counters=counters,
            extra={"wall_share": share, "cost_explore": cost_explore.snapshot(), "cost_sweep": cost_sweep.snapshot(),
                   **({"final_eval": final} if final else {})},
        )

    def run_eval(it: int, prefix: str = "eval", force_sweep: bool = False) -> dict | None:
        nonlocal last_mark
        rollout = state.drain()
        cost_explore.add(rollout)
        explore = rollout.get(cfg.get("sweep_gate_metric", "grasped_once"), float("nan"))
        summary = deterministic_sweep(it) if (force_sweep or schedule.sweep_worthwhile(explore)) else None
        elapsed = time.time() - started
        rate = (counters["env_steps"] - last_mark[1]) / max(elapsed - last_mark[0], 1e-9)
        last_mark = (elapsed, counters["env_steps"])
        entry = {
            "iteration": it, "env_steps": counters["env_steps"], "transitions": counters["transitions"],
            "updates": counters["updates"], "wall_time_s": round(elapsed, 1), "env_steps_per_s": round(rate),
            "rollout": rollout, "cumulative": {"explore": cost_explore.snapshot(), "sweep": cost_sweep.snapshot()},
        }
        if summary is not None:
            entry[prefix] = summary
        ledger.record(entry)
        cost = f"impulse/ep {rollout.get('obstacle_impulse', float('nan')):.2f} N.s " \
               f"contact/ep {rollout.get('contact_ticks', float('nan')):.1f} ticks " \
               f"cum {cost_explore.sums['obstacle_impulse']:.0f} N.s"
        line = (" ".join(f"{k}={summary[k]:.4f}" for k in PRINT_KEYS if k in summary and summary[k] == summary[k])
                if summary is not None else "(no sweep yet)")
        print(f"  [{elapsed:6.1f}s {counters['env_steps'] / 1e6:6.2f}M {rate / 1e3:4.1f}k/s] {line}  "
              f"explore[grasp {rollout.get('grasped_once', float('nan')):.2f} lift {rollout.get('lifted_once', float('nan')):.2f} "
              f"succ {rollout.get('success_once', float('nan')):.3f} h {rollout.get('bowl_height', float('nan')):+.3f}]  {cost}",
              flush=True)
        if summary is None:
            write_evals()
            return None
        value = summary.get(cfg["stop_metric"], float("nan"))
        info = dict(wall_time_s=round(elapsed, 1), env_steps=counters["env_steps"], iteration=it)
        ledger.note_milestones(value, **info)
        if ledger.note_best(summary, **info):
            agent.save(str(exp_dir / "models" / "best"))
        drain_updates()  # the sweep's deferred updates, only after the checkpoint is on disk
        write_evals()
        if schedule.clears(value):
            ledger.note_reached(value=value, **info)
            raise Budget
        return summary

    it = 0
    try:
        while True:
            it += 1
            interact(it, exploring=True, collect=True)
            elapsed = time.time() - started
            if schedule.eval_due(elapsed):
                run_eval(it)
                schedule.note_eval(time.time() - started)
            if schedule.budget_spent(time.time() - started):
                raise Budget
    except Budget:
        pass

    total = time.time() - started
    if ledger.reached is None:
        try:
            run_eval(it, prefix="eval_final", force_sweep=True)
        except Budget:
            pass

    final = None
    if args.final_eval_episodes > 0 and ledger.best["wall_time_s"] is not None:
        agent.load(str(exp_dir / "models" / "best"))
        episodes = [deterministic_sweep(it, collect=False) for _ in range(args.final_eval_episodes)]
        final = aggregate(episodes)
        print(f"\n=== best checkpoint, held out on {args.final_eval_episodes} x {num_envs} episodes ===")
        for k in (*PRINT_KEYS, "lost_grip", *COST_KEYS):
            if k in final:
                print(f"  {k:18s} {final[k]['mean']:.4f} +/- {final[k]['sd']:.4f}")
    write_evals(final)

    best = ledger.best
    print(f"\n=== {cfg['experiment_name']} ===")
    for level, m in ledger.milestones.items():
        print(f"  {cfg['stop_metric']} >= {level:>4s} first at {m['wall_time_s']:.1f}s / {m['env_steps'] / 1e6:.2f}M ticks")
    print(f"  best {cfg['stop_metric']} {best['value']:.4f} at {best['wall_time_s']}s / {(best['env_steps'] or 0) / 1e6:.2f}M ticks")
    print(f"  {counters['env_steps'] / 1e6:.2f}M ticks, {counters['transitions'] / 1e6:.2f}M transitions, "
          f"{counters['updates']} updates, {counters['env_steps'] / max(total, 1e-9):.0f} ticks/s")
    ce = cost_explore.snapshot()
    print(f"  exploration cost: {ce['episodes']:.0f} episodes, {ce['obstacle_impulse']:.0f} N.s obstacle impulse, "
          f"{ce['contact_ticks']:.0f} contact ticks, {100 * ce['contact_episodes'] / max(ce['episodes'], 1):.1f}% episodes with contact")
    if args.profile:
        for name, seconds in timing.items():
            print(f"  {name:8s} {seconds:7.1f}s  {seconds / max(total, 1e-9) * 100:5.1f}%")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # No simulation_app.close(): Kit's shutdown hangs after this scene; the
    # process exit frees the GPU.
    os._exit(0)
