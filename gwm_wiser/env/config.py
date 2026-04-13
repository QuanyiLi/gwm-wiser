MAX_EPISODE_STEP_WORKSPACE = 120
MAX_EPISODE_STEP_WORKSPACE_EVAL = MAX_EPISODE_STEP_WORKSPACE

CAMERA_RESOLUTION = (448, 224)
HAND_CAMERA_RESOLUTION = (224, 224)


def get_env_cfg(
    *,
    max_steps,
    num_env,
    obs_mode="state",
    table_builder=None,
    scene_cfg_to_overwrite=None,
    robot_uids="my_panda_wristcam",
):
    """
    Get env cfgs.
    :param max_steps: Episode will be ended after max_steps only.
    :param num_env: No larger than 64 to avoid OOM when saving videos.
    :param obs_mode: "state" or "rgb" or others, like "rgb+segmentation"
    :param table_builder: Optional, choosing from ComplexTableSceneBuilder, WorkSpaceBuilder
    :param scene_cfg_to_overwrite: Some options to overwrite the default scene config.
    :return: env_cfg
    """
    import os

    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda_available = False

    default_sim_backend = "physx_cuda" if cuda_available else "physx_cpu"
    default_render_backend = "gpu" if cuda_available else "cpu"

    sim_backend = os.environ.get("GWM_SIM_BACKEND", default_sim_backend)
    render_backend = os.environ.get("GWM_RENDER_BACKEND", default_render_backend)

    if sim_backend in {"cpu", "physx_cpu"} and num_env > 1:
        # ManiSkill disallows in-process vectorization for CPU PhysX; fall back to a single env.
        num_env = 1

    env_cfg = dict(
        max_episode_steps=max_steps,  # same to ppo batch steps
        max_step_option="terminate",  # "truncate" makes the robot take shortcut
        scene_cfg_to_overwrite=scene_cfg_to_overwrite or {},
        # simulation core config
        obs_mode=obs_mode,
        control_mode="pd_joint_pos",  # pd_joint_delta_pos, pd_ee_delta_pose
        render_mode="rgb_array",  # when evaluation, record video
        sim_backend=sim_backend,
        num_envs=num_env,
        render_backend=render_backend,
        enable_shadow=True,
        robot_uids=robot_uids,
        parallel_in_single_scene=False,
    )
    if table_builder is not None:
        env_cfg["table_builder"] = table_builder

    return env_cfg


scene_cfg = {
    "hide_goal_site": True,  # show the yellow ball as the place to go
    "cube_size_noise": 0.0,  # add noise to cube size
    # working space
    "cube_spawn_half_size": (0.25, 0.3),  # make it hard, x, y
    "cube_spawn_center": (-0.165, 0),  # max reach distance ~= 0.7
    # goal
    "goal_thresh": 0.025,
    "max_goal_height": 0.01,
    # "sensor_cam_eye_pos": [[1.0522, 0.7018, 0.5615]],
    # "sensor_cam_target_pos": [[-0.7401, -0.2693, 0.1630]],
    "sensor_cam_eye_pos": [[0.4700, 0.0000, 0.2000]],
    "sensor_cam_target_pos": [[0.0200, 0.0000, 0.0800]],
    "sensor_cam_fov": 38,
    "human_cam_eye_pos": [0.05, 0.35, 0.3],
    "human_cam_target_pos": [-0.1, 0.0, 0.0],
    # Expert Training and only valid when using complex table builder
    "robot_init_qpos_noise": 0.0,
    "robot_retract_prob": 0.5,
    "qpos_shuffle_across_envs": True,  # shuffle the initial qpos across envs by using rotate
    "scene_reset_freq": 1,  # 1 reset every episode or None for turning off
    "colors": ["red", "green"],
    # "cube_len": [0.02, 0.04],  # small, large
    "one_color_per_size": False,  # if True, each color/size group has one cube to pick
    "make_others_big": False,  # if True, all cubes are big except the target one, used to train better expert policy
    # mode and config
    "mode": "train",  # options: train, test
    "cfg_name": "config_0",  # default config name
    # lighting
    "randomize_light": False,  # if True, randomize ambient light color based on config index
}
