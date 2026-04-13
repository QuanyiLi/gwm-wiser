import glob
import json
import os
import time
from collections import defaultdict

import datasets
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from gwm_wiser.env.config import (
    CAMERA_RESOLUTION,
    HAND_CAMERA_RESOLUTION,
    MAX_EPISODE_STEP_WORKSPACE,
    get_env_cfg,
)
from gwm_wiser.planner.mplib import BatchPickCubePlanner
from gwm_wiser.utils.env import build_endless_env
from gwm_wiser.utils.helpers import batch_tensor_to_string
from gwm_wiser.utils.lerobot import obs_state_key, reward_key, action_key
from gwm_wiser.utils.lerobot import (
    task_key,
    image_1_key,
    image_2_key,
    wrist_image_key,
    image_1_robot_state,
    image_1_segmentation_mask_key,
)

# Runtime setup
datasets.disable_progress_bar()
os.environ["SVT_LOG"] = "0"

# the resolution in maniskill is width height, but her we use height width
base_cam_resolution = (CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3)
base_cam_seg_resolution = (CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3)
base_cam_mask_resolution = (CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3)
wrist_cam_resolution = (*HAND_CAMERA_RESOLUTION, 3)


def rollout(
    envs,
    policy,
    round_to_collect=1,
    demo_saving_dir=None,
    indices_to_save=[i for i in range(12)],
    only_save_robot_state_image=False,
):
    """
    Rollout expert policy and save demo data in lerobot format. Useful for BC and Dagger
    :param envs: The environments wrapped by RSLRLWrapper.
    :param demo_saving_dir: If not None, rollouts will be saved to LeRobot format. Note that inside this directory, a sub-directory "lerobot_data" will be created to save the data.
    :param policy: A policy class for rollout. Can be a mixed policy or any policy that takes in observations and outputs actions.
    :param round_to_collect: How many rounds to collect. Total number of episodes collected = num_env * round_to_collect
    :param indices_to_save: List of trajectory indices to save. Default saves all 12 envs.
    :param only_save_robot_state_image: Don't save any RGB data, set to zero for all RGB
    :return: rollout results in a dict.
    """
    num_env = envs.unwrapped.num_envs
    max_num_step = envs.unwrapped.max_episode_steps

    # Check if second camera exists by doing a test reset
    test_obs, _ = envs.reset()
    has_second_camera = image_2_key in test_obs

    # metrics
    eval_metrics = defaultdict(list)
    policy_infos = []
    episode_records = []  # Store per-episode metrics with task prompts

    # lerobot dataset features
    features = {
        image_1_key: {
            "dtype": "video",
            "shape": base_cam_resolution,
            "names": ["height", "width", "channel"],
        },
        image_1_robot_state: {
            "dtype": "video",
            "shape": base_cam_seg_resolution,
            "names": ["height", "width", "channel"],
        },
        image_1_segmentation_mask_key: {
            "dtype": "video",
            "shape": base_cam_mask_resolution,
            "names": ["height", "width", "channel"],
        },
        wrist_image_key: {
            "dtype": "video",
            "shape": (128, 128, 3)
            if "xarm" in envs.unwrapped.robot_uids
            else wrist_cam_resolution,
            "names": ["height", "width", "channel"],
        },
        obs_state_key: {
            "dtype": "float32",
            "shape": (12 if "xarm" in envs.unwrapped.robot_uids else 9,),
            "names": [
                "q_pos",
            ],
        },
        action_key: {
            "dtype": "float32",
            "shape": (7 if "xarm" in envs.unwrapped.robot_uids else 8,),
            "names": ["actions"],
        },
        reward_key: {"dtype": "float32", "shape": (1,), "names": ["reward"]},
    }

    # Only add second camera feature if it exists
    if has_second_camera:
        features[image_2_key] = {
            "dtype": "video",
            "shape": base_cam_resolution,
            "names": ["height", "width", "channel"],
        }

    # lerobot - create shared dataset
    if demo_saving_dir is not None:
        lerobot_dataset = LeRobotDataset.create(
            repo_id="demo_dataset",
            root=os.path.join(demo_saving_dir, "lerobot_data"),
            robot_type=envs.unwrapped.robot_uids,
            fps=envs.unwrapped.control_freq,
            # video_backend="pyav", # the default torchcodec maybe buggy
            use_videos=True,
            features=features,
        )
    elapsed_io = 0.0
    elapsed_rollout = 0.0

    # run
    pbar = tqdm(
        total=round_to_collect * num_env,
        desc=f"Episode Collected ({max_num_step} steps per episode)",
    )
    for round_index in range(round_to_collect):
        obs, _ = envs.reset()
        assert envs.unwrapped.elapsed_steps.sum() == 0, (
            "envs need to be reset before new round"
        )

        # stats
        episode_actions = []
        cam_1_images = []
        cam_1_robot_state_images = []
        cam_1_mask_images = []
        cam_2_images = []
        wrist_images = []
        state = []
        reward = []

        # rollout
        rollout_start_time = time.perf_counter()
        for i in range(max_num_step):
            with torch.no_grad():
                action_to_take, expert_action, policy_info = policy(obs)
                obs, rew, done, eval_infos = envs.step(action_to_take)
                policy_infos.append(policy_info)

            # collect data for lerobot
            # obs = obs.clone()  # clone to avoid reference bug
            episode_actions.append(expert_action.cpu())
            reward.append(rew.clone().cpu())
            cam_1_images.append(obs[image_1_key].cpu())
            cam_1_robot_state_images.append(obs[image_1_robot_state].cpu())
            cam_1_mask_images.append(obs[image_1_segmentation_mask_key].cpu())
            if has_second_camera:
                cam_2_images.append(obs[image_2_key].cpu())
            wrist_images.append(obs[wrist_image_key].cpu())
            state.append(obs[obs_state_key].cpu())
        elapsed_rollout += time.perf_counter() - rollout_start_time

        # collect metrics
        pbar.update(num_env)
        for k, v in eval_infos["episode"].items():
            eval_metrics[k].append(v)
        eval_metrics["near_goal"].append(eval_infos["near_goal"])
        eval_metrics["is_grasped"].append(eval_infos["is_grasped"])
        eval_metrics["tcp_near_goal"].append(eval_infos["tcp_near_goal"])

        # stack
        io_start_time = time.perf_counter()
        rew = torch.stack(reward, dim=1).numpy()[..., None]  # (num_env, T, 1)
        cam_1_images = torch.stack(cam_1_images, dim=1).numpy()
        cam_1_robot_state_images = torch.stack(cam_1_robot_state_images, dim=1).numpy()
        cam_1_mask_images = torch.stack(cam_1_mask_images, dim=1).numpy()
        cam_1_mask_images = cam_1_mask_images.repeat(
            3, axis=-1
        )  # repeat mask along the last dim

        if has_second_camera:
            cam_2_images = torch.stack(cam_2_images, dim=1).numpy()
        wrist_images = torch.stack(wrist_images, dim=1).numpy()
        state = torch.stack(state, dim=1).numpy()
        episode_actions = torch.stack(episode_actions, dim=1).numpy()
        task_name = batch_tensor_to_string(obs[task_key])

        if only_save_robot_state_image:
            cam_1_images = np.zeros_like(cam_1_images)
            if has_second_camera:
                cam_2_images = np.zeros_like(cam_2_images)
            wrist_images = np.zeros_like(wrist_images)
            cam_1_mask_images = np.zeros_like(cam_1_mask_images)

        # Collect per-episode metrics with task prompts
        for env_idx in range(num_env):
            episode_record = {
                "round": round_index,
                "env_idx": env_idx,
                "global_episode_idx": round_index * num_env + env_idx,
                "task_prompt": task_name[env_idx],
                "success_once": bool(
                    eval_infos["episode"]["success_once"][env_idx].item()
                ),
                "success_at_end": bool(
                    eval_infos["episode"]["success_at_end"][env_idx].item()
                ),
                "return": float(eval_infos["episode"]["return"][env_idx].item()),
                "near_goal": bool(eval_infos["near_goal"][env_idx].item()),
                "is_grasped": bool(eval_infos["is_grasped"][env_idx].item()),
                "tcp_near_goal": bool(eval_infos["tcp_near_goal"][env_idx].item()),
            }
            episode_records.append(episode_record)

        if demo_saving_dir is not None:  # do saving or not
            for env_idx in range(num_env):
                # Skip trajectories not in the specified indices list
                if env_idx not in indices_to_save:
                    continue

                for t in range(max_num_step):
                    frame_data = {
                        image_1_key: cam_1_images[env_idx][t],
                        image_1_robot_state: cam_1_robot_state_images[env_idx][t],
                        image_1_segmentation_mask_key: cam_1_mask_images[env_idx][t],
                        wrist_image_key: wrist_images[env_idx][t],
                        obs_state_key: state[env_idx][t],
                        action_key: episode_actions[env_idx][t],
                        reward_key: rew[env_idx][t],
                        task_key: task_name[env_idx],
                    }
                    if has_second_camera:
                        frame_data[image_2_key] = cam_2_images[env_idx][t]

                    lerobot_dataset.add_frame(frame_data)

                lerobot_dataset.save_episode()
        elapsed_io += time.perf_counter() - io_start_time

    # close
    pbar.close()
    if demo_saving_dir is not None:
        lerobot_dataset.finalize()

    # record metrics
    results = {}
    for k in [
        "near_goal",
        "success_at_end",
        "success_once",
        "is_grasped",
        "return",
        "tcp_near_goal",
    ]:
        v = eval_metrics[k]
        v = torch.cat(v)
        mean = v.to(torch.float32).mean()
        results[f"{k}_mean"] = mean.cpu().item()

    results["rollout_fps"] = round(
        num_env * round_to_collect * max_num_step / elapsed_rollout, 2
    )
    results["rollout_time"] = round(elapsed_rollout, 2)

    if demo_saving_dir is not None and elapsed_io > 1:
        results["io_time"] = round(elapsed_io, 2)
        results["io_fps"] = round(
            num_env * round_to_collect * max_num_step / elapsed_io, 2
        )

    # update policy execution info
    results.update(
        {k: np.mean([i[k] for i in policy_infos]) for k, v in policy_infos[0].items()}
    )

    # Save episode metrics to JSON
    if demo_saving_dir is not None:
        metrics_path = os.path.join(demo_saving_dir, "episode_metrics.json")
        metrics_data = {
            "summary": results,
            "episodes": episode_records,
            "config": {
                "num_envs": num_env,
                "rounds": round_to_collect,
                "max_steps_per_episode": max_num_step,
                "total_episodes": len(episode_records),
            },
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Episode metrics saved to {metrics_path}")

    return results


def rollout_with_mplib_expert(
    demo_saving_dir,
    cfg_name,
    mode,
    max_num_step=MAX_EPISODE_STEP_WORKSPACE,
    round_to_collect=1,
    robot_init_qpos=0.0,
    only_save_robot_state_image=False,
    robot_uids="my_panda_wristcam",
):
    # cfg and build endless envs
    scene_cfg = dict(
        robot_init_qpos_noise=robot_init_qpos,  # collect with noise for training lerobot VLA
        mode=mode,
        cfg_name=cfg_name,
        cube_size_noise=0.00,
    )
    env_cfg = get_env_cfg(
        num_env=12,
        max_steps=max_num_step,
        obs_mode="rgb+segmentation",
        scene_cfg_to_overwrite=scene_cfg,
    )
    env_cfg["robot_uids"] = robot_uids
    envs = build_endless_env(env_cfg, record_video=False, data_record_dir="test")

    # wrap policy to use privileged_obs
    policy = BatchPickCubePlanner.build_policy(envs)

    def wrapped_policy(*args, **kwargs):
        action_to_take = expert_action = policy()
        return action_to_take, expert_action, dict(use_expert_action=1)

    # rollout
    performance = rollout(
        envs,
        wrapped_policy,
        round_to_collect=round_to_collect,
        demo_saving_dir=demo_saving_dir,
        only_save_robot_state_image=only_save_robot_state_image,
    )
    for k, v in performance.items():
        print(f"{k}: {v}")
    envs.unwrapped.close()


def calculate_averages(folder_pattern):
    # Find all directories matching the pattern
    directories = glob.glob(folder_pattern)
    directories.sort()

    if not directories:
        print(f"No directories found for pattern: {folder_pattern}")
        return {}

    print(f"Found {len(directories)} directories for pattern '{folder_pattern}'")

    # Store all values for each metric
    metrics_sums = {}
    metrics_counts = {}
    metrics_mins = {}
    metrics_maxs = {}

    file_count = 0

    for directory in directories:
        metrics_file = os.path.join(directory, "episode_metrics.json")
        if not os.path.exists(metrics_file):
            print(f"Warning: {metrics_file} not found. Skipping.")
            continue

        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
                summary = data.get("summary", {})

                for key, value in summary.items():
                    if isinstance(value, (int, float)):
                        metrics_sums[key] = metrics_sums.get(key, 0) + value
                        metrics_counts[key] = metrics_counts.get(key, 0) + 1

                        if key not in metrics_mins or value < metrics_mins[key]:
                            metrics_mins[key] = value
                        if key not in metrics_maxs or value > metrics_maxs[key]:
                            metrics_maxs[key] = value

                file_count += 1
        except json.JSONDecodeError:
            print(f"Error decoding JSON in {metrics_file}. Skipping.")

    print(f"Processed {file_count} files.")
    print("-" * 30)
    print(f"Averages for {folder_pattern}:")
    final_results = {}
    for key in sorted(metrics_sums.keys()):
        if metrics_counts[key] > 0:
            avg = metrics_sums[key] / metrics_counts[key]
            min_val = metrics_mins[key]
            max_val = metrics_maxs[key]
            print(f"{key}: {avg:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")
            final_results[key] = {"avg": avg, "min": min_val, "max": max_val}
        else:
            print(f"{key}: No data")
    print("-" * 30)
    print("\n")
    return final_results
