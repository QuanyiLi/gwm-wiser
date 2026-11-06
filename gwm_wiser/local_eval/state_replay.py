"""
Recover full low-dimensional state for the WISER training demos by replaying their
recorded ``pd_joint_pos`` actions in the env (``obs_mode="state"``).

The LeRobot demos only store the 9-D joint state. This script rebuilds, for every
step, the state vector used by the state-only proposal policy:

    qpos (9) | cube_slot_0..3 pose (4 x 7) | placard_slot_0..2 pose (3 x 7)

Slots are ordered by their fixed y coordinate (left to right in the robot frame), so
the vector never marks which cube is the instructed one. No velocities, goal, grasp
flag or static/dynamic flag are included.

Each episode is replayed in the env index whose instruction matches the episode's
task string (12 envs per config, 6 batches per config for the 1 clean + 5 noisy
demos), starting from the episode's recorded initial joint state.

    python -m gwm_wiser.local_eval.state_replay --dataset wiser_dataset/merged_train \
        --out /path/state_demos --start_config 0 --end_config 24
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from gwm_wiser.env.config import get_env_cfg, MAX_EPISODE_STEP_WORKSPACE
from gwm_wiser.env.wiser_env import WISEREnv

STATE_KEYS = (
    ["qpos"]
    + [f"cube_slot_{s}_pose" for s in range(4)]
    + [f"placard_slot_{s}_pose" for s in range(3)]
)
STATE_DIM = 9 + 4 * 7 + 3 * 7


def build_env(cfg_name, mode, obs_mode="state"):
    env_cfg = get_env_cfg(
        num_env=12,
        max_steps=MAX_EPISODE_STEP_WORKSPACE,
        obs_mode=obs_mode,
        scene_cfg_to_overwrite=dict(
            robot_init_qpos_noise=0.0, cube_size_noise=0.0, cfg_name=cfg_name, mode=mode
        ),
        robot_uids="my_panda_wristcam",
    )
    return WISEREnv(**env_cfg)


def placard_poses(env):
    """(3, 7) world poses of the placards, slot order (ascending y).

    The placard actors sit at the origin and carry the image as a visual offset
    (``WorkSpaceBuilder._build_image``), so the pose is the fixed visual pose.
    """
    from transforms3d.euler import euler2quat

    x = env.table_scene.cfg["cube_spawn_center"][0] + 0.25
    q = euler2quat(0, -np.pi / 2, -np.pi / 2)
    return torch.tensor([[x, y, 0.005, *q] for y in (-0.2, 0.0, 0.2)],
                        dtype=torch.float32, device=env.device)


def slot_orders(env):
    """Per env: cube group index for each cube slot (ascending y), and placard poses."""
    ts = env.table_scene
    init_y = torch.stack([c.pose.p[:, 1] for c in ts.all_cubes], dim=1)  # (E, 4)
    cube_order = torch.argsort(init_y, dim=1)  # slot -> group
    return cube_order, placard_poses(env)


def full_state(env, cube_order, placards):
    """(E, STATE_DIM) state vector in fixed slot order."""
    ts = env.table_scene
    qpos = env.agent.robot.get_qpos()
    cube_poses = torch.stack([c.pose.raw_pose for c in ts.all_cubes], dim=1)  # (E, 4, 7)
    cube_poses = torch.gather(cube_poses, 1, cube_order[..., None].expand(-1, -1, 7))
    pl = placards.flatten()[None].expand(qpos.shape[0], -1)
    return torch.cat([qpos, cube_poses.flatten(1), pl], dim=-1)


def load_episodes(dataset_root):
    files = sorted(glob.glob(os.path.join(dataset_root, "data", "*", "*.parquet")))
    df = pd.concat(
        [pd.read_parquet(f, columns=["observation.state", "action", "episode_index",
                                     "frame_index", "task_index"]) for f in files]
    )
    tasks = pd.read_parquet(os.path.join(dataset_root, "meta", "tasks.parquet"))
    task_str = {int(r.task_index): t for t, r in tasks.iterrows()}
    eps = {}
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        eps[int(ep)] = dict(
            task=task_str[int(g.task_index.iloc[0])],
            state=np.stack(g["observation.state"].to_numpy()).astype(np.float32),
            action=np.stack(g["action"].to_numpy()).astype(np.float32),
        )
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="train")
    ap.add_argument("--start_config", type=int, default=0)
    ap.add_argument("--end_config", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eps = load_episodes(args.dataset)
    print(f"loaded {len(eps)} episodes; state keys {STATE_KEYS} dim {STATE_DIM}", flush=True)

    for c in range(args.start_config, args.end_config):
        out_path = os.path.join(args.out, f"config_{c}_{args.mode}.npz")
        if os.path.exists(out_path):
            print(f"skip {out_path}")
            continue
        env = build_env(f"config_{c}", args.mode)
        env.reset(seed=0)
        instr = list(env.table_scene.task_instruction)
        by_env = {e: sorted(k for k, v in eps.items() if v["task"] == instr[e]) for e in range(12)}
        counts = {len(v) for v in by_env.values()}
        assert len(counts) == 1 and 0 not in counts, f"config_{c}: episode counts {by_env}"
        n_batches = counts.pop()

        all_states, all_actions, all_rec_qpos, ep_ids, env_ids, successes, max_err = [], [], [], [], [], [], []
        for b in range(n_batches):
            batch = [eps[by_env[e][b]] for e in range(12)]
            env.reset(seed=0)
            init_qpos = torch.tensor(np.stack([x["state"][0] for x in batch]), device=env.device)
            env.agent.reset(init_qpos)
            if env.gpu_sim_enabled:
                env.scene._gpu_apply_all()
                env.scene.px.gpu_update_articulation_kinematics()
                env.scene._gpu_fetch_all()
            cube_order, placards = slot_orders(env)
            T = batch[0]["action"].shape[0]
            states = [full_state(env, cube_order, placards)]
            success_once = torch.zeros(12, dtype=torch.bool, device=env.device)
            for t in range(T):
                a = torch.tensor(np.stack([x["action"][t] for x in batch]), device=env.device)
                _, _, _, _, info = env.step(a)
                success_once |= info["success"].bool()
                states.append(full_state(env, cube_order, placards))
            states = torch.stack(states, dim=1).cpu().numpy()  # (12, T+1, D)
            rec = np.stack([x["state"] for x in batch])  # (12, T, 9)
            err = np.abs(states[:, :T, :9] - rec).max(axis=(1, 2))
            all_states.append(states)
            all_actions.append(np.stack([x["action"] for x in batch]))
            all_rec_qpos.append(rec)
            ep_ids += [by_env[e][b] for e in range(12)]
            env_ids += list(range(12))
            successes.append(success_once.cpu().numpy())
            max_err.append(err)
            print(f"config_{c} batch {b}: replay success {success_once.float().mean().item():.3f} "
                  f"max |qpos replay - recorded| {err.max():.4f}", flush=True)

        np.savez_compressed(
            out_path,
            states=np.concatenate(all_states).astype(np.float32),
            actions=np.concatenate(all_actions).astype(np.float32),
            recorded_qpos=np.concatenate(all_rec_qpos).astype(np.float32),
            episode_index=np.array(ep_ids),
            env_index=np.array(env_ids),
            replay_success=np.concatenate(successes),
            replay_qpos_max_err=np.concatenate(max_err),
            state_keys=np.array(STATE_KEYS),
        )
        with open(out_path.replace(".npz", ".json"), "w") as f:
            json.dump(dict(instructions=instr, n_batches=n_batches,
                           replay_success=float(np.concatenate(successes).mean()),
                           qpos_max_err=float(np.concatenate(max_err).max())), f, indent=2)
        env.close()


if __name__ == "__main__":
    main()
