from typing import Any, Dict, Optional
from typing import Union
import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs.pose import Pose

from gwm_wiser.env.config import scene_cfg, CAMERA_RESOLUTION
from gwm_wiser.env.work_space_builder import WorkSpaceBuilder
from gwm_wiser.utils.lerobot import (
    obs_state_key,
    image_1_key,
    image_2_key,
    wrist_image_key,
    task_key,
    image_1_robot_state,
    image_1_segmentation_mask_key,
)


class WISEREnv(BaseEnv):
    agent: Union[Panda]

    def __init__(
        self,
        *,
        scene_cfg_to_overwrite: Dict = None,
        max_episode_steps=None,
        max_step_option="terminate",  # or terminate
        success_terminate=False,  # important, or the algorithm will take shortcut and never stay at the goal
        robot_uids="my_panda_wristcam",
        table_builder=WorkSpaceBuilder,
        **kwargs,
    ):

        # fix control mode
        default_control_mode = "pd_joint_pos"
        if "control_mode" in kwargs:
            assert kwargs["control_mode"] == default_control_mode, (
                f"Please only use {default_control_mode} for domain randomization pick env"
            )
        kwargs["control_mode"] = default_control_mode

        # fix control/sim freq
        default_control_freq = 20  # Hz
        default_sim_freq = (
            100  # Hz this may not be a good dt for sim2real and affect the integration
        )
        if "sim_config" in kwargs and "control_freq" in kwargs["sim_config"]:
            assert kwargs["sim_config"]["control_freq"] == default_control_freq, (
                f"Control freq is fixed to {default_control_freq}Hz for domain randomization pick env"
            )
        if "sim_config" in kwargs and "sim_freq" in kwargs["sim_config"]:
            assert kwargs["sim_config"]["sim_freq"] == default_sim_freq, (
                f"Sim freq is fixed to {default_sim_freq}Hz for domain randomization pick env"
            )
        kwargs["sim_config"] = {
            "control_freq": default_control_freq,
            "sim_freq": default_sim_freq,
        }

        minimum_env = 12
        if table_builder == WorkSpaceBuilder:
            assert (
                kwargs["num_envs"] >= minimum_env
                and kwargs["num_envs"] % minimum_env == 0
            ), f"At least {minimum_env} envs are required for domain randomization"

        # cfg and builders
        scene_cfg_to_overwrite = scene_cfg_to_overwrite or {}
        self.cfg = cfg = scene_cfg
        self.cfg.update(scene_cfg_to_overwrite)
        self.table_builder_class = table_builder
        assert not success_terminate, (
            "don't end when success or the algorithm will never stay at the goal"
        )

        # goal
        self.goal_thresh = cfg["goal_thresh"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.sensor_cam_fov = cfg.get("sensor_cam_fov", 58)  # default 58 degrees
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        self.success_terminate = success_terminate

        # First frame, many prompts referring objects/destinations with the first frame setting
        self._first_frame_image_1 = None

        # env init
        super().__init__(robot_uids=robot_uids, **kwargs)

        # termination/truncation setting
        self.max_episode_steps = max_episode_steps
        assert max_step_option in ["truncate", "terminate"]
        self.max_step_option = max_step_option
        if max_episode_steps is None:
            logger.warning(
                "No max episode steps provided. Episode will last until terminate!"
            )
        self.truncated = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )  # truncation buffer

    @property
    def _default_sensor_configs(self):
        cam_config = []
        fov_rad = np.pi * (self.sensor_cam_fov / 180)
        for i in range(len(self.sensor_cam_eye_pos)):
            pose = sapien_utils.look_at(
                eye=self.sensor_cam_eye_pos[i], target=self.sensor_cam_target_pos[i]
            )
            cam_config.append(
                CameraConfig(
                    f"base_camera_{i}",
                    pose,
                    *CAMERA_RESOLUTION,
                    fov_rad,
                    0.01,
                    100,
                    shader_pack="default",
                )
            )
        return cam_config

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig(
            "render_camera", pose, *CAMERA_RESOLUTION, np.pi * (100 / 180), 0.01, 100
        )

    def _load_lighting(self, options: dict):
        """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""
        if self.cfg["randomize_light"]:
            raise NotImplementedError
        else:
            super()._load_lighting(options)

    def _load_agent(self, options: dict):
        base_pos = [-0.615, 0, 0]
        if "xarm" in self.robot_uids:
            base_pos = [-0.565, 0, 0]
        super()._load_agent(options, sapien.Pose(p=base_pos))

    def _load_scene(self, options: dict):
        self.table_scene = self.table_builder_class(self, cfg=self.cfg)
        self.table_scene.build()

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[233 / 255, 240 / 255, 26 / 255, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        if self.cfg["hide_goal_site"]:
            self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self.goal_site.set_pose(Pose.create_from_pq(self.table_scene.goal_xyz))
        self._first_frame_image_1 = None

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.goal_site.pose.p - self.table_scene.cube.pose.p, dim=1
            )
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.table_scene.cube)
        is_robot_static = self.agent.is_static(0.2)

        dist_2d_to_goal = torch.linalg.norm(
            self.goal_site.pose.p[:, :2] - self.table_scene.cube.pose.p[:, :2], dim=1
        )
        near_goal_tolerance = 0.05
        near_goal = dist_2d_to_goal < near_goal_tolerance
        tcp_to_goal = torch.linalg.norm(
            self.goal_site.pose.p - self.agent.tcp_pose.p, dim=1
        )
        tcp_near_goal = tcp_to_goal < near_goal_tolerance

        target_height = 0.15  # TODO: consider the table height?
        height_tolerance = 0.02
        dist_to_target_height = torch.abs(
            target_height - self.table_scene.cube.pose.p[..., 2]
        )
        height_reached = dist_to_target_height < height_tolerance  # tolerance 2 cm
        return {
            "dist_2d_to_goal": dist_2d_to_goal,
            "dist_to_target_height": dist_to_target_height,
            "height_reached": height_reached,
            "target_height": target_height,
            "near_goal": near_goal,
            "tcp_near_goal": tcp_near_goal,
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # constants
        dist_2d_max = 0.8

        # status
        is_grasped = info["is_grasped"]
        near_goal = info["near_goal"]
        dist_to_target_height = info["dist_to_target_height"]
        target_height = info["target_height"]
        dist_2d_to_goal = info["dist_2d_to_goal"]
        height_reached = info["height_reached"]

        # Base reward for reaching and grasping
        tcp_to_obj_dist = torch.linalg.norm(
            self.table_scene.cube.pose.p - self.agent.tcp_pose.p, dim=1
        )
        reaching_reward = 1 - torch.tanh(tcp_to_obj_dist / dist_2d_max)
        grasp_reward = is_grasped.float()  # Use .float() for clarity
        reward = reaching_reward + grasp_reward

        # Phase 2: Move the object (active ONLY when grasped and not near_goal)
        lift_mask = is_grasped & ~near_goal & ~height_reached
        lift_reward = 1 - torch.tanh(dist_to_target_height / target_height)
        reward[lift_mask] += lift_reward[lift_mask]

        move_mask = is_grasped & ~near_goal & height_reached
        move_reward = 1 - torch.tanh(dist_2d_to_goal / dist_2d_max)
        reward[move_mask] += move_reward[move_mask] + 1

        # Phase 3: Place the object (active ONLY when near_goal)
        place_mask = is_grasped & near_goal
        dist_3d_to_goal = torch.linalg.norm(
            self.table_scene.cube.pose.p - self.goal_site.pose.p, dim=1
        )
        place_reward = 1 - torch.tanh(dist_3d_to_goal / 0.2)
        reward[place_mask] += (
            4 * place_reward[place_mask] + 2
        )  # +2 to bypass the effect of phase 2!!

        qvel = self.agent.robot.get_qvel()[..., :-2]
        static_reward = 1 - torch.tanh(torch.linalg.norm(qvel, dim=1) / dist_2d_max)
        reward += static_reward * info["is_obj_placed"]
        reward[info["success"]] = 10

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # 6 reward items in total
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        o, r, termination, _, info = super().step(action)
        info["env_idx"] = torch.arange(0, self.num_envs)
        if not self.success_terminate:
            termination = torch.zeros_like(termination, dtype=torch.bool)

        if self.max_episode_steps is not None:
            if self.max_step_option == "terminate":
                termination |= self.elapsed_steps >= self.max_episode_steps
            else:  # truncate
                self.truncated = self.elapsed_steps >= self.max_episode_steps
        return o, r, termination, self.truncated, info

    def get_obs(self, info: Optional[Dict] = None, unflattened: bool = False):
        obs = super().get_obs(info, True)
        obs["tcp_pose"] = self.agent.tcp_pose.raw_pose

        privileged_obs = {k: v.clone() for k, v in obs["agent"].items()}
        privileged_obs.update(
            is_grasped=info["is_grasped"],
            near_goal=info["near_goal"],
            height_reached=info["height_reached"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            half_size=self.table_scene.cube_half_sizes,
            obj_pose=self.table_scene.cube.pose.raw_pose,
            tcp_to_obj_pos=self.table_scene.cube.pose.p - self.agent.tcp_pose.p,
            obj_to_goal_pos=self.goal_site.pose.p - self.table_scene.cube.pose.p,
        )
        privileged_obs = common.flatten_state_dict(
            privileged_obs, use_torch=True, device=self.device
        )
        obs["privileged_obs"] = privileged_obs

        # Used to test distillation algorithm
        reduced_obs = {k: v.clone() for k, v in obs["agent"].items()}
        reduced_obs.update(
            is_grasped=info["is_grasped"],
            goal_pos=self.goal_site.pose.p,
            half_size=self.table_scene.cube_half_sizes,
            obj_pose=self.table_scene.cube.pose.raw_pose,
        )
        reduced_obs = common.flatten_state_dict(
            reduced_obs, use_torch=True, device=self.device
        )
        obs["reduced_obs"] = reduced_obs

        # obs used for lerobot policies
        if "sensor_data" in obs:
            obs[task_key] = self.table_scene.task_instruction_utf8_tensor.to(
                self.device
            )
            obs[image_1_key] = obs["sensor_data"]["base_camera_0"]["rgb"]
            if "segmentation" in obs["sensor_data"]["base_camera_0"]:
                seg = obs["sensor_data"]["base_camera_0"]["segmentation"]
                assert seg.size(-1) == 1
                s = seg[..., 0]
                mask = torch.isin(
                    s,
                    torch.arange(
                        1, len(self.agent.robot.links) + 1, device=self.device
                    ),
                )
                masked_rgb = obs[image_1_key].clone()
                masked_rgb[~mask] = 0
                obs[image_1_robot_state] = masked_rgb
                # Panda: 15 robot links (IDs 1-15), cubes start at 16
                # XArm6: 21 robot links (IDs 1-21), cubes start at 22

                # Add raw segmentation mask
                assert s.max() <= 255 and s.min() >= 0
                obs[image_1_segmentation_mask_key] = s[..., None].to(
                    torch.uint8
                )  # (H, W, 1)
            if "base_camera_1" in obs["sensor_data"]:
                obs[image_2_key] = obs["sensor_data"]["base_camera_1"]["rgb"]
            obs[wrist_image_key] = obs["sensor_data"]["hand_camera"]["rgb"]
            obs.pop("sensor_data")

            # Capture/update first frame when episode starts (elapsed_steps == 0)
            if self._first_frame_image_1 is None:
                self._first_frame_image_1 = obs[image_1_key].clone()

            # Add first frame to observations
            obs[image_1_key + "_first_frame"] = self._first_frame_image_1.clone()

            # Add elapsed steps to observations
            obs["elapsed_steps"] = self.elapsed_steps.clone()

            # use qpos/qvel and tcp_pose as state input
            obs[obs_state_key] = obs["agent"]["qpos"]
            # obs[obs_state_key] = common.flatten_state_dict(obs["agent"], use_torch=True, device=self.device)
            # obs[obs_state_key] = torch.concatenate([q_info, obs["tcp_pose"]], dim=-1)

        return obs

    # def _load_lighting(self, options: dict):
    #     for scene in self.scene.sub_scenes:
    #         scene.ambient_light = [0.6, 0.6, 0.6]
    #         scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _flatten_raw_obs(self, obs: Any):
        return obs  # don't flatten, let the wrapper do it

    @staticmethod
    def draw_rectangle(viewer, corner_1, corner_2, corner_3, corner_4):
        line_1 = np.stack([corner_1, corner_2])
        line_2 = np.stack([corner_2, corner_3])
        line_3 = np.stack([corner_3, corner_4])
        line_4 = np.stack([corner_4, corner_1])
        color = np.array([[0.8, 0.8, 0.8, 1], [0.8, 0.8, 0.8, 1]])

        lineset_1 = viewer.renderer_context.create_line_set(line_1, color)
        viewer.render_scene.add_line_set(lineset_1)
        lineset_2 = viewer.renderer_context.create_line_set(line_2, color)
        viewer.render_scene.add_line_set(lineset_2)
        lineset_3 = viewer.renderer_context.create_line_set(line_3, color)
        viewer.render_scene.add_line_set(lineset_3)
        lineset_4 = viewer.renderer_context.create_line_set(line_4, color)
        viewer.render_scene.add_line_set(lineset_4)

        x_vertices = np.array([[0, 0, 0], [1, 0, 0]])
        y_vertices = np.array([[0, 0, 0], [0, 1, 0]])
        z_vertices = np.array([[0, 0, 0], [0, 0, 1]])

        x_colors = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])
        y_colors = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
        z_colors = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])

        lineset_x = viewer.renderer_context.create_line_set(x_vertices, x_colors)
        viewer.render_scene.add_line_set(lineset_x)
        lineset_y = viewer.renderer_context.create_line_set(y_vertices, y_colors)
        viewer.render_scene.add_line_set(lineset_y)
        lineset_z = viewer.renderer_context.create_line_set(z_vertices, z_colors)
        viewer.render_scene.add_line_set(lineset_z)


if __name__ == "__main__":
    # mp.set_start_method("fork", force=True)  # must be in the main guard
    reconfigure_interval = None
    scene_cfg_to_overwrite = dict(
        mode="test",
        cfg_name="animal",
    )
    env = WISEREnv(
        num_envs=12,
        # obs_mode="state",
        obs_mode="rgb+segmentation",
        # robot_uids="panda",
        scene_cfg_to_overwrite=scene_cfg_to_overwrite,
        render_mode="human",
        # control_mode="pd_ee_delta_pose",  # pd_joint_delta_pos, pd_ee_delta_pose
        # parallel_in_single_scene=True,
        # table_builder=ComplexTableSceneBuilder
    )
    o = env.reset()
    been_drawn = False
    viewer = None
    t = 0
    while True:
        t += 1
        action = np.zeros_like(env.action_space.sample())
        action[:, :] = np.array(
            [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04]
        )
        o, r, done, _, info = env.step(action)

        if env.render_mode == "human":
            viewer = env.render()
        #
        # if not been_drawn and viewer is not None:
        #     x_half, y_half = scene_cfg["cube_spawn_half_size"]
        #     corner_1 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-x_half, -y_half, 0])
        #     corner_2 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([x_half, -y_half, 0])
        #     corner_3 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([x_half, y_half, 0])
        #     corner_4 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-x_half, y_half, 0])
        #     env.draw_rectangle(viewer, corner_1, corner_2, corner_3, corner_4)
        #
        #     corner_1 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-0.1 + 0.15, -0.1, 0])
        #     corner_2 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([0.1 + 0.15, -0.1, 0])
        #     corner_3 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([0.1 + 0.15, 0.1, 0])
        #     corner_4 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-0.1 + 0.15, 0.1, 0])
        #     env.draw_rectangle(viewer, corner_1, corner_2, corner_3, corner_4)
        #
        #     corner_1 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-0.1 - 0.15, -0.1, 0])
        #     corner_2 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([0.1 - 0.15, -0.1, 0])
        #     corner_3 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([0.1 - 0.15, 0.1, 0])
        #     corner_4 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-0.1 - 0.15, 0.1, 0])
        #     env.draw_rectangle(viewer, corner_1, corner_2, corner_3, corner_4)
        #
        #     corner_1 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-0.1 - 0.15, -0.1 + 0.2, 0])
        #     corner_2 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([0.1 - 0.15, -0.1 + 0.2, 0])
        #     corner_3 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([0.1 - 0.15, 0.1 + 0.2, 0])
        #     corner_4 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-0.1 - 0.15, 0.1 + 0.2, 0])
        #     env.draw_rectangle(viewer, corner_1, corner_2, corner_3, corner_4)
        #
        #     corner_1 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-0.1 - 0.15, -0.1 - 0.2, 0])
        #     corner_2 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([0.1 - 0.15, -0.1 - 0.2, 0])
        #     corner_3 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([0.1 - 0.15, 0.1 - 0.2, 0])
        #     corner_4 = np.array([*scene_cfg["cube_spawn_center"], 0]) + np.array([-0.1 - 0.15, 0.1 - 0.2, 0])
        #     env.draw_rectangle(viewer, corner_1, corner_2, corner_3, corner_4)
        #
        #     been_drawn = True
        #
        # if reconfigure_interval is not None and t % reconfigure_interval == 0:
        #     env.reset(options=dict(reconfigure=True))
        #     been_drawn = False
