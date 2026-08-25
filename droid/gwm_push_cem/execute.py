"""execute: run every CEM rollout in Isaac and log where the cubes end up.

One boot, one episode per distinct trajectory: reset, settle while holding the
home pose with the gripper closed, track the timeline at 15 Hz, then hold still
while the cubes come to rest. Every cube's spawn and final position is written
out, so the physical outcome is read from sim state rather than from video.

Distinct trajectory, not distinct rollout: independent CEM runs often converge
on the same endpoint, and `reset_scene_to_default` puts every body back
bit-exactly (measured spawn spread across episodes: 0.0 mm), so a repeated
endpoint would reproduce its own episode exactly. Each one is therefore
simulated once and its outcome copied to every rollout that asked for it;
`--verify N` re-runs N of them to show the outcome really is reproducible.
Several plan files can be handed over at once so they share both the boot and
that pool of episodes.

Camera observations and the camera sensors are both switched off unless
episodes are selected with `--record`: the observation manager would otherwise
render two 1280x720 views on every step, which dominates the run time and
changes nothing physical.

    cd /root/code/gwm/gwm-wiser/droid/gwm_push_cem && \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
    ../droid-sim-evals/.venv/bin/python -u execute.py \
        --plans plans_winner.json,plans_sample.json \
        --out   exec_winner.json,exec_sample.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "droid-sim-evals"))
sys.path.insert(0, str(HERE))

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
CAM_TERMS = ("external_cam", "external_cam_2", "wrist_cam", "wrist_depth",
             "wrist_intrinsics", "wrist_cam_pos_w", "wrist_cam_quat_w")
CAM_SENSORS = ("external_cam", "external_cam_2", "wrist_cam")


def key_of(endpoint):
    return f"{endpoint[0]:.4f},{endpoint[1]:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="plans_winner.json",
                    help="comma-separated plan files, executed together")
    ap.add_argument("--out", default=None,
                    help="comma-separated output files, one per plan file")
    ap.add_argument("--only", default=None, help="comma-separated prompt tags")
    ap.add_argument("--limit", type=int, default=None, help="first N rollouts per prompt")
    ap.add_argument("--record", default="",
                    help='"all", or comma-separated endpoint keys ("0.6800,0.0000") to film')
    ap.add_argument("--verify", type=int, default=0,
                    help="re-run this many episodes to check reproducibility")
    ap.add_argument("--settle-steps", type=int, default=15)
    args, _ = ap.parse_known_args()

    from config import (CUBE_PRIMS, EXEC_HZ, HOLD_S, RESULTS, SCENE_ID,
                        SCENE_VARIANT)
    from sim_common import arm_home_qpos

    plan_files = [s.strip() for s in args.plans.split(",") if s.strip()]
    out_files = ([s.strip() for s in args.out.split(",")] if args.out else
                 [f.replace("plans", "exec") for f in plan_files])
    assert len(out_files) == len(plan_files), "--out must match --plans"

    plans, wanted = {}, {}
    for f in plan_files:
        d = json.loads((RESULTS / f).read_text())
        roll = d["rollouts"]
        if args.only:
            keep = {t.strip() for t in args.only.split(",")}
            roll = {k: v for k, v in roll.items() if k in keep}
        if args.limit:
            roll = {k: v[:args.limit] for k, v in roll.items()}
        d["rollouts"] = roll
        plans[f] = d
        for tag, runs in roll.items():
            for r in runs:
                wanted.setdefault(key_of(r["endpoint"]), d["trajectories"][key_of(r["endpoint"])])

    record = args.record.strip()
    n_roll = sum(len(v) for d in plans.values() for v in d["rollouts"].values())
    print(f"{len(plan_files)} plan file(s), {n_roll} rollouts, "
          f"{len(wanted)} distinct trajectories to simulate", flush=True)
    q_home = arm_home_qpos()

    from isaaclab.app import AppLauncher

    kit_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(kit_parser)
    args_cli, _ = kit_parser.parse_known_args()
    args_cli.enable_cameras = bool(record)
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import numpy as np
    import torch

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(SCENE_ID), SCENE_VARIANT)
    env_cfg.scene.robot.init_state.joint_pos.update(
        {name: float(v) for name, v in zip(ARM_JOINTS, q_home)})
    if not record:
        # A spawned camera makes Isaac demand --enable_cameras even when nothing
        # reads it, so the sensors go as well as the observation terms. The
        # scene builder skips None entries.
        for term in CAM_TERMS:
            setattr(env_cfg.observations.policy, term, None)
        for sensor in CAM_SENSORS:
            setattr(env_cfg.scene, sensor, None)
        env_cfg.rerender_on_reset = False
    env = gym.make("DROID", cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()

    device = env.unwrapped.device
    scene_h = env.unwrapped.scene
    hold_q = torch.tensor(q_home, dtype=torch.float32, device=device).unsqueeze(0)
    hold_action = torch.cat([hold_q, torch.ones((1, 1), device=device)], dim=-1)

    def np0(t):
        return t[0].cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t[0])

    def cube_xyz():
        return {tag: np0(scene_h.rigid_objects[prim].data.root_pos_w)[:3].copy()
                for tag, prim in CUBE_PRIMS.items()}

    imageio = None
    if record:
        import imageio.v2 as imageio  # noqa: F811

    def episode(cand, video_dir=None):
        P = np.asarray(cand["positions"], dtype=np.float64)
        T = np.asarray(cand["t"], dtype=np.float64)
        env.reset()
        for _ in range(args.settle_steps):
            env.step(hold_action)
        spawn = cube_xyz()
        writers = None
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
            writers = {c: imageio.get_writer(video_dir / f"{c}.mp4", fps=int(EXEC_HZ),
                                             codec="libx264", quality=8)
                       for c in ("external_cam", "external_cam_2")}
        n_steps = int((T[-1] + HOLD_S) * EXEC_HZ) + 1
        for k in range(n_steps):
            tk = min(k / EXEC_HZ, T[-1])
            q = np.array([np.interp(tk, T, P[:, j]) for j in range(7)])
            action = torch.tensor(np.concatenate([q, [1.0]]), dtype=torch.float32,
                                  device=device).unsqueeze(0)
            env.step(action)
            if writers is not None:
                for c, w in writers.items():
                    w.append_data(np0(scene_h.sensors[c].data.output["rgb"])[..., :3]
                                  .astype(np.uint8))
        if writers is not None:
            for w in writers.values():
                w.close()
        final = cube_xyz()
        return {
            "spawn": {t: [round(float(v), 5) for v in p] for t, p in spawn.items()},
            "final": {t: [round(float(v), 5) for v in p] for t, p in final.items()},
            "disp": {t: [round(float(final[t][i] - spawn[t][i]), 5) for i in range(3)]
                     for t in spawn},
        }

    def video_dir_for(key):
        if not record:
            return None
        if record != "all" and key not in {k.strip() for k in record.split(";")}:
            return None
        return RESULTS / "exec_video" / key.replace(",", "_")

    outcome, t_start = {}, time.time()
    keys = list(wanted)
    for n, key in enumerate(keys, 1):
        outcome[key] = episode(wanted[key], video_dir_for(key))
        if n % 20 == 0 or n == len(keys):
            print(f"[{n}/{len(keys)}] {key} "
                  f"[{time.time() - t_start:.0f}s, "
                  f"{(time.time() - t_start) / n:.1f} s/episode]", flush=True)

    repeat = {}
    if args.verify:
        step = max(1, len(keys) // args.verify)
        for key in keys[::step][:args.verify]:
            again = episode(wanted[key])
            repeat[key] = max(
                abs(again["final"][t][i] - outcome[key]["final"][t][i])
                for t in again["final"] for i in range(3))
        print(f"reproducibility over {len(repeat)} repeated episodes: "
              f"max |delta final| = {max(repeat.values()):.6f} m", flush=True)

    for f, o in zip(plan_files, out_files):
        d = plans[f]
        res = {}
        for tag, runs in d["rollouts"].items():
            res[tag] = [{"i": r["i"], "seed": r["seed"], "endpoint": r["endpoint"],
                         "obj": r.get("obj"), **outcome[key_of(r["endpoint"])]}
                        for r in runs]
        (RESULTS / o).write_text(json.dumps(res))
        print(f"wrote {RESULTS / o}")
    if repeat:
        (RESULTS / "exec_repeatability.json").write_text(json.dumps(
            {"max_abs_delta_m": repeat}, indent=2))
    print(f"{len(keys)} episodes for {n_roll} rollouts in {time.time() - t_start:.0f}s")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
