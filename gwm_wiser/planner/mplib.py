import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.xarm6.motionplanner import (
    XArm6RobotiqMotionPlanningSolver,
)
from mani_skill.utils.structs.pose import to_sapien_pose

from gwm_wiser.env.wiser_env import WISEREnv


class BatchPickCubePlanner:
    """
    It runs in sequential to simulate the parallel planning for multiple envs.
    """

    def __init__(self, env: WISEREnv):
        import packaging.version

        # Check numpy version for mplib compatibility
        if packaging.version.parse(np.__version__) != packaging.version.parse("1.26.4"):
            raise RuntimeError(
                f"numpy version {np.__version__} is NOT 1.26.4, which is required by mplib. "
                "Please install it with `pip install numpy==1.26.4`"
            )

        FINGER_LENGTH = 0.025
        self.env = env = env.unwrapped
        if "panda" in self.env.robot_uids:
            planner_cls = PandaArmMotionPlanningSolver
        elif "xarm6_robotiq_wristcam" in self.env.robot_uids:
            planner_cls = XArm6RobotiqMotionPlanningSolver
        else:
            raise ValueError(f"Planner unsupported for robot: {self.env.robot_uids}")

        self.planner = planner_cls(
            self.env,
            debug=False,
            vis=False,
            base_pose=env.unwrapped.agent.robot.pose.raw_pose[0],
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )

        env_cubes = [
            env.table_scene.cube.scene.actors[f"cube_0_{env_index}"]
            for env_index in range(env.num_envs)
        ]
        obbs = [get_actor_obb(cube) for cube in env_cubes]

        approaching = np.array([0, 0, -1])
        target_closings = (
            env.agent.tcp.pose.to_transformation_matrix()[:, :3, 1].cpu().numpy()
        )

        self.reach_pose = []
        self.grasp_poses = []
        self.lift_poses = []
        self.above_goal_poses = []
        self.goal_poses = []
        for env_index in range(env.num_envs):
            grasp_info = compute_grasp_info_by_obb(
                obbs[env_index],
                approaching=approaching,
                target_closing=target_closings[env_index],
                depth=FINGER_LENGTH,
            )
            closing, center = grasp_info["closing"], grasp_info["center"]
            grasp_pose = env.agent.build_grasp_pose(
                approaching, closing, env_cubes[env_index].pose.sp.p
            )
            grasp_pose = grasp_pose * sapien.Pose(
                [0, 0, 0.0]
            )  # offset to avoid collision with the table
            reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])  # approach from above
            lift_pose = grasp_pose * sapien.Pose(
                [0, 0, -0.08]
            )  # lift up after grasping
            self.grasp_poses.append(grasp_pose)
            self.reach_pose.append(reach_pose)
            self.lift_poses.append(lift_pose)

            # goal
            if "xarm6_robotiq_wristcam" in self.env.robot_uids:
                goal_pose = sapien.Pose(
                    self.env.goal_site.pose.p[env_index].cpu(), grasp_pose.q
                )
            else:
                goal_pose = env.agent.build_grasp_pose(
                    approaching, closing, self.env.goal_site.pose.p[env_index].cpu()
                )
            above_goal_pose = goal_pose * sapien.Pose([0, 0, -0.12])
            self.above_goal_poses.append(above_goal_pose)
            self.goal_poses.append(goal_pose)

    def batch_plan(self, batch_pose, batch_initial_qpos=None):
        results = []
        if batch_initial_qpos is None:
            batch_initial_qpos = self.planner.robot.get_qpos().cpu().numpy()
        batch_plan_results = self.batch_move_to_pose_with_screw(
            batch_pose, batch_initial_qpos=batch_initial_qpos
        )
        for env_index in range(self.env.num_envs):
            qpos = batch_plan_results[env_index]["position"]
            action = np.hstack(
                [qpos, np.full((qpos.shape[0], 1), self.planner.gripper_state)]
            )
            results.append(torch.from_numpy(action).to(self.env.device))
        last_qpos = [action[-1, :-1].cpu().numpy() for action in results]
        return results, last_qpos

    def batch_close_gripper(self, t=6, batch_initial_qpos=None):
        self.planner.gripper_state = self.planner.__class__.CLOSED
        if batch_initial_qpos is None:
            if "panda" in self.env.robot_uids:
                batch_initial_qpos = self.planner.robot.get_qpos()[:, :-2].cpu().numpy()
            elif "xarm6_robotiq_wristcam" in self.env.robot_uids:
                batch_initial_qpos = self.planner.robot.get_qpos()[:, :-6].cpu().numpy()
        batch_initial_qpos = np.array(batch_initial_qpos)
        action = np.hstack(
            [
                batch_initial_qpos,
                np.full((batch_initial_qpos.shape[0], 1), self.planner.gripper_state),
            ]
        )[:, None]
        action = torch.from_numpy(action).to(self.env.device)
        results = action.repeat(1, t, 1)
        return results

    def batch_move_to_pose_with_screw(self, poses: sapien.Pose, batch_initial_qpos):
        results = []
        for env_index in range(self.env.num_envs):
            pose = poses[env_index]
            pose = to_sapien_pose(pose)
            if "xarm6_robotiq_wristcam" in self.env.robot_uids:
                result = self.planner.planner.plan_screw(
                    np.concatenate([pose.p, pose.q]),
                    batch_initial_qpos[env_index],
                    time_step=self.env.control_timestep,
                    use_point_cloud=False,
                )
            elif "panda" in self.env.robot_uids:
                full_qpos = np.concatenate(
                    [batch_initial_qpos[env_index], np.array([0.4, 0.4])], axis=-1
                )
                result = self.planner.planner.plan_qpos_to_pose(
                    np.concatenate([pose.p, pose.q]),
                    full_qpos,
                    time_step=self.env.control_timestep,
                    use_point_cloud=False,
                    wrt_world=True,
                )
            else:
                raise ValueError(
                    f"Planner unsupported for robot: {self.env.robot_uids}"
                )
            reason = result["status"]
            assert result["status"] == "Success", f"Planning failed: {reason}"
            results.append(result)
        return results

    def batch_generate_actions(self):
        """
        Generate the pick and place trajectory for all envs in batch.
        :return: actions that can be consumed by env.step(), shape (T, N, action_dim), valid mask (N, T)
        """

        self.planner.gripper_state = self.planner.__class__.OPEN
        if "panda" in self.env.robot_uids:
            initial_qpos = self.env.agent.robot.get_qpos()[:, :-2].cpu().numpy()
        elif "xarm6_robotiq_wristcam" in self.env.robot_uids:
            initial_qpos = self.env.agent.robot.get_qpos()[:, :-6].cpu().numpy()
        planning_results = []

        # reach
        plan_results, initial_qpos = self.batch_plan(self.reach_pose, initial_qpos)
        planning_results.append(plan_results)

        # grasp
        plan_results, initial_qpos = self.batch_plan(self.grasp_poses, initial_qpos)
        planning_results.append(plan_results)

        # close
        plan_results = self.batch_close_gripper(batch_initial_qpos=initial_qpos)
        planning_results.append(plan_results)

        # lift
        plan_results, initial_qpos = self.batch_plan(self.lift_poses, initial_qpos)
        planning_results.append(plan_results)

        # place
        # plan_results, initial_qpos = self.batch_plan(self.above_goal_poses, initial_qpos)
        # planning_results.append(plan_results)

        plan_results, _ = self.batch_plan(self.goal_poses, initial_qpos)
        planning_results.append(plan_results)

        # merge all plans
        merged_planning_results = []
        for env_index in range(self.env.num_envs):
            env_actions = []
            for stage in planning_results:
                env_actions.append(stage[env_index])
            merged_planning_results.append(torch.concatenate(env_actions, dim=0))
        results = merged_planning_results

        # record valid steps
        num_valid_step = [len(r) for r in results]
        max_num_valid_step = max(num_valid_step)
        valid = torch.zeros(
            (self.env.num_envs, max_num_valid_step),
            dtype=torch.bool,
            device=self.env.device,
        )

        # padding to the same length
        for env_index in range(self.env.num_envs):
            pad_len = max_num_valid_step - len(results[env_index])
            valid[env_index, : num_valid_step[env_index]] = True
            if pad_len > 0:
                pad_action = results[env_index][-1].unsqueeze(0).repeat(pad_len, 1)
                results[env_index] = torch.cat([results[env_index], pad_action], dim=0)
            else:
                assert valid[env_index].all()
        results = torch.stack(results, dim=1)
        return results, valid

    @classmethod
    def build_policy(cls, env: BaseEnv):
        """
        Align with the policy interface in mani_skill.
        """
        planner = cls(env)
        planning_results = None
        step = 0

        def policy(*args, **kwargs):
            nonlocal env
            nonlocal planning_results
            nonlocal step
            nonlocal planner
            if env.unwrapped.elapsed_steps.sum() == 0:  # reset policy
                planning_results, _ = planner.batch_generate_actions()
                step = 0

            action = planning_results[step if step < planning_results.size(0) else -1]
            step += 1
            return action.to(torch.float32)

        return policy
