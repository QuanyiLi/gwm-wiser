import os
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import mani_skill
import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.utils import common
from mani_skill.utils.scene_builder.scene_builder import SceneBuilder
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from gwm_wiser import ASSET_ROOT
from gwm_wiser.env.config_registry import ConfigRegistry
from gwm_wiser.utils.helpers import batch_string_to_tensor
from gwm_wiser.utils.helpers import repeat_to_length, generate_color, repeat_at_level

HEIGHT_OFFSET = 1e-4
img_p = os.path.join(ASSET_ROOT, "images")


@dataclass
class TableImage:
    path: str


class WorkSpaceBuilder(SceneBuilder):
    """A simple scene builder that adds a table to the scene such that the height of the table is at 0, and
    gives reasonable initial poses for robots."""

    def __init__(self, env, cfg):
        super().__init__(env, cfg["robot_init_qpos_noise"])
        self.cfg = cfg
        self.table = None
        self.all_cubes = None
        self.all_cube_half_sizes = None
        self.workspace = None

        # build cube params
        self.cube_to_pick_idx = 0  # grasp group 1
        self.num_cubes = 4
        self.all_scene_configurations = self._get_scenario_configs(cfg)

        # goal_xyz
        self.goal_xyz = None

        # language
        self.task_instruction = None
        self.task_instruction_utf8_tensor = None  # auto build at each episode reset

    def _get_scenario_configs(self, cfg):
        # Load config from YAML file
        assert "cfg_name" in cfg, (
            "Please provide the scene configuration name via 'cfg_name'."
        )
        yaml_cfg = ConfigRegistry.get_config(cfg.get("cfg_name", "config_0"))

        # Select section based on mode
        mode = cfg["mode"]
        if mode == "train":
            section = yaml_cfg["train"]
        elif mode == "test":
            section = yaml_cfg["test"]
        else:
            raise ValueError(f"Only train/test modes are supported, got: {mode}")

        # Extract config values
        colors = section["cube_colors"]
        assert len(colors) == 4, "the following code are hard coded for 4 colors."

        images = [
            TableImage(os.path.join(img_p, img["path"]))
            for img in section["table_images"]
        ]
        assert len(images) == 3, "the following code are hard coded for 3 table images."

        task_referring_expressions = section["task_referring_expressions"]
        assert len(task_referring_expressions) == 12, (
            "the following code are hard coded for 12 task expressions."
        )

        # Compute positions (same as before)
        spawn_center = cfg["cube_spawn_center"]
        position = np.array(
            [
                [spawn_center[0] + 0.05, -0.225, 0],
                [spawn_center[0] + 0.05, -0.075, 0],
                [spawn_center[0] + 0.05, 0.075, 0],
                [spawn_center[0] + 0.05, 0.225, 0],
            ]
        )
        assert len(position) == 4, "the following code are hard coded for 4 positions."

        destination = [
            [self.cfg["cube_spawn_center"][0] + 0.25, -0.2, 0],
            [self.cfg["cube_spawn_center"][0] + 0.25, 0, 0],
            [self.cfg["cube_spawn_center"][0] + 0.25, 0.2, 0],
        ]
        assert len(destination) == 3, (
            "the following code are hard coded for 3 destinations."
        )

        # Expand colors and positions for all envs (4 cubes × 3 destinations = 12 envs)
        colors = repeat_to_length(colors, len(colors))
        positions = repeat_to_length(position.tolist(), len(position))
        colors = repeat_at_level(colors, 3, -1)
        positions = repeat_at_level(positions, 3, -2)
        destination = destination * 4

        # Output
        scenario_configs = dict(
            cube_colors=colors,
            positions=positions,
            destination=destination,
            table_images=images,
            task_referring_expressions=task_referring_expressions,
        )
        return scenario_configs

    def build(self):
        self._build_workspace(need_physics=False, additional_height=0.001)

        # build cubes
        self.all_cubes = []
        self.all_cube_half_sizes = []

        # cube_to_pick_idx == 0 suggests the object to pick
        for i in range(self.num_cubes):
            physx_type = (
                "dynamic" if i == self.cube_to_pick_idx else "static"
            )  # don't move other cubes : )
            cube, half_x, half_y, half_z = self._build_cube(
                group_idx=i, physx_type=physx_type
            )
            self.all_cubes.append(cube)
            half_sizes = common.to_tensor(
                np.stack([half_x, half_y, half_z]), self.env.device
            ).T
            self.all_cube_half_sizes.append(half_sizes)
        self.all_cube_half_sizes = torch.stack(self.all_cube_half_sizes, dim=0)

        # destination
        dest_figures = self._build_images()

        # add to scene_objects
        self.scene_objects: List[sapien.Entity] = (
            [self.table, self.workspace] + self.all_cubes + dest_figures
        )

    def initialize(self, env_idx: torch.Tensor):
        # randomize the robot initial pose per episode
        if self.env.robot_uids in ["panda", "my_panda_wristcam"]:
            # Note: we enhance the determinism of the environment when setting the robot initial by default
            retract_qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            noise = self.env._episode_rng.normal(
                0,
                self.robot_init_qpos_noise,
                (self.env.num_envs, retract_qpos.shape[-1]),
            )
            retract_qpos = retract_qpos + noise
            retract_qpos[..., -2:] = 0.04  # keep gripper open

            # set to retract environment
            self.env.agent.reset(torch.from_numpy(retract_qpos).to(self.env.device))
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "xarm6_robotiq_wristcam":
            retract_qpos = np.array([0, 0.22, -1.23, 0, 1.01, 0, 0, 0, 0, 0, 0, 0])
            noise = self.env._episode_rng.normal(
                0,
                self.robot_init_qpos_noise,
                (self.env.num_envs, retract_qpos.shape[-1]),
            )
            retract_qpos = retract_qpos + noise
            retract_qpos[..., -6:] = 0  # keep gripper open

            self.env.agent.reset(torch.from_numpy(retract_qpos).to(self.env.device))
            self.env.agent.robot.set_pose(sapien.Pose([-0.565, 0, 0]))
        else:
            raise ValueError(
                f"Unsupported robot {self.env.robot_uids} for TableSceneBuilder."
            )

        # reset cube poses
        p = Pose.create_from_pq(
            p=self.all_scene_configurations["positions"][self.cube_to_pick_idx]
        )
        p.p[..., -1] += self.cube_half_sizes[..., -1] + HEIGHT_OFFSET
        self.cube.set_pose(p)

        # reset goal, consider the cube height
        self.goal_xyz = torch.tensor(
            self.all_scene_configurations["destination"], device=self.env.device
        )
        self.goal_xyz[..., -1] += self.cube_half_sizes[..., -1] + HEIGHT_OFFSET

        # build instruction
        self._build_language_instruction()

    def _build_language_instruction(self):
        # Use task_referring_expressions tuples directly from config
        task_exprs = self.all_scene_configurations["task_referring_expressions"]
        num_envs = self.env.num_envs

        assert num_envs == 12, (
            f"num_envs must be 12 to match task_referring_expressions, got {num_envs}"
        )
        assert len(task_exprs) == 12, (
            f"task_referring_expressions must have exactly 12 entries, got {len(task_exprs)}"
        )

        instructions = []
        for obj, dest in task_exprs:
            obj = obj.strip()
            if not obj.startswith("the"):
                obj = "the " + obj
            dest = dest.strip()
            if not dest.startswith("the"):
                dest = "the " + dest
            instructions.append(f"Pick up {obj} and place it onto {dest}.")

        self.task_instruction = instructions
        self.task_instruction_utf8_tensor = batch_string_to_tensor(
            self.task_instruction
        ).to(self.env.device)

    def _build_cube(self, group_idx, physx_type="dynamic", element_type="cube"):
        noise = self.env._batched_episode_rng.uniform(
            -self.cfg["cube_size_noise"], self.cfg["cube_size_noise"]
        )

        cubes = []
        half_size = []
        for i in range(self.env.num_envs):
            builder = self.scene.create_actor_builder()
            cube_len = 0.04  # TODO: fixed size for now

            half_length = (cube_len + noise[i]) / 2
            half_size.append(half_length)

            # physics
            builder.add_box_collision(half_size=[half_length] * 3)

            # visual
            mat = sapien.render.RenderMaterial(
                base_color=generate_color(
                    self.all_scene_configurations["cube_colors"][group_idx][i]
                ),
                roughness=0.5,
                specular=0.5,
            )
            if element_type == "cube":
                builder.add_box_visual(
                    sapien.Pose(), half_size=[half_length] * 3, material=mat
                )
            else:
                builder.add_cylinder_visual(
                    pose=sapien.Pose(q=euler2quat(0, np.pi / 2, 0)),
                    radius=half_length,
                    half_length=half_length,
                    material=mat,
                )

            # set position
            position = self.all_scene_configurations["positions"][group_idx][i].copy()
            position[2] += half_length + HEIGHT_OFFSET
            builder.initial_pose = sapien.Pose(p=position)

            # build
            builder.set_scene_idxs([i])
            if physx_type == "dynamic":
                cube = builder.build_dynamic(f"cube_{group_idx}_{i}")
            elif physx_type == "kinematic":
                cube = builder.build_kinematic(f"cube_{group_idx}_{i}")
            elif physx_type == "static":
                cube = builder.build_static(f"cube_{group_idx}_{i}")
            else:
                raise ValueError(f"Unsupported physx type {physx_type} for box.")
            self.env.remove_from_state_dict_registry(cube)
            cubes.append(cube)

        # merge
        cube = Actor.merge(cubes, f"cube_{group_idx}")
        self.env.add_to_state_dict_registry(cube)
        return cube, half_size, half_size, half_size

    @property
    def cube_half_sizes(self):
        return self.all_cube_half_sizes[self.cube_to_pick_idx]

    @property
    def cube(self):
        return self.all_cubes[self.cube_to_pick_idx]

    @property
    def cube_id(self):
        target_cube_id = self.cube.per_scene_id
        assert torch.diff(target_cube_id).sum() == 0, (
            "All envs should have the same target cube id."
        )
        return target_cube_id

    def _build_image(self, x, y, img_path, actor_name, scale=(0.0, 0.09, 0.09)):
        builder = self.scene.create_actor_builder()
        pose = sapien.Pose(p=[x, y, 0.005], q=euler2quat(0, -np.pi / 2, -np.pi / 2))
        # Read image into numpy array and create texture from it
        img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_bgr.shape[2] == 3:
            # Convert BGR to RGBA
            img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
        else:
            # Convert BGRA to RGBA
            img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA)
        img_rgba = cv2.flip(img_rgba, 1)
        texture = sapien.render.RenderTexture2D(img_rgba, "R8G8B8A8Unorm", srgb=True)
        mat = sapien.render.RenderMaterial(roughness=10, specular=0.5)
        mat.set_base_color_texture(texture)
        builder.add_plane_visual(material=mat, scale=scale, pose=pose)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[0, 0, 0, 0])
        return builder.build_static(name=actor_name)

    def _build_workspace(self, need_physics, additional_height):
        builder = self.scene.create_actor_builder()
        model_dir = (
            Path(osp.dirname(mani_skill.__file__)) / "utils/scene_builder/table/assets"
        )
        table_model_file = str(model_dir / "table.glb")
        scale = 1.75

        # table
        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )
        table = builder.build_static(name="table")
        self.table = table

    def _build_images(self):
        table_conf = self.all_scene_configurations["table_images"]
        x = self.cfg["cube_spawn_center"][0] + 0.25
        right_img = self._build_image(x, -0.2, table_conf[0].path, "right-figure")
        middle_img = self._build_image(x, 0, table_conf[1].path, "middle-figure")
        left_img = self._build_image(x, 0.2, table_conf[2].path, "left-figure")
        return [middle_img, left_img, right_img]
