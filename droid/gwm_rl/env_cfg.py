"""The env config and its gym registration.

One environment serves both baselines: its action is ``[7 absolute joint
targets, 1 binary gripper]`` at 15 Hz (the DROID convention), and the
end-effector action spaces are wrappers over it (`executors.py`) that turn a
pose into joint targets with `franka_kin`. Rewards, observations and the
collision cost are computed here, per tick, whichever wrapper drives it.
"""

from __future__ import annotations

import gymnasium as gym
import isaaclab.envs.mdp as il_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from gwm_rl import geometry as G
from gwm_rl import mdp
from gwm_rl.scene import PickBowlSceneCfg
from gwm_rl.task import POLICY_FIELDS, TICK_DT, resolve_params

TASK_ID = "GwmRl-PickBowl-v0"
_SIM_DT = 1.0 / 120.0
_DECIMATION = 8
assert abs(_SIM_DT * _DECIMATION - TICK_DT) < 1e-12


@configclass
class ActionsCfg:
    arm = il_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(G.ARM_JOINT_NAMES),
        preserve_order=True,
        use_default_offset=False,
    )
    # Stock convention: action < 0 closes.
    gripper = il_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[G.FINGER_JOINT_NAME],
        open_command_expr={G.FINGER_JOINT_NAME: G.GRIPPER_OPEN},
        close_command_expr={G.FINGER_JOINT_NAME: G.GRIPPER_CLOSED},
    )


@configclass
class PolicyObsCfg(ObsGroup):
    state = ObsTerm(func=mdp.observation_fields, params={"fields": POLICY_FIELDS})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: PolicyObsCfg = PolicyObsCfg()


@configclass
class EventsCfg:
    """Reset-mode terms, in order: scene defaults, arm draw, episode bookkeeping."""

    reset_all = EventTerm(func=il_mdp.reset_scene_to_default, mode="reset")
    reset_robot = EventTerm(func=mdp.reset_arm_gaussian, mode="reset", params={"std": 0.05})
    reset_episode = EventTerm(func=mdp.reset_episode, mode="reset")


@configclass
class RewardsCfg:
    # 1/dt cancels the RewardManager's weight * dt: the per-tick value is the raw one.
    pick = RewTerm(func=mdp.pick_reward_term, weight=1.0 / TICK_DT)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=il_mdp.time_out, time_out=True)


def _camera(name: str, env_index: int) -> CameraCfg:
    spec = G.CAM_POSE[name]
    return CameraCfg(
        prim_path=f"/World/envs/env_{env_index}/{name}",
        height=G.CAM_RES[1],
        width=G.CAM_RES[0],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=G.CAM_FOCAL, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024
        ),
        offset=CameraCfg.OffsetCfg(
            pos=spec["pos"], rot=G.cam_offset_quat(spec["pos"], spec["lookat"]), convention="opengl"
        ),
    )


@configclass
class PickBowlEnvCfg(ManagerBasedRLEnvCfg):
    #: Overrides onto `TaskParams` (`--env-set task_params.lift_height=0.12`).
    task_params: dict = {}
    #: Mount the drawer experiment's two external cameras in envs 0..K-1 —
    #: the GWM capture hook. 0 in training configs: a mounted camera costs
    #: throughput even when nobody reads it.
    capture_envs: int = 0

    scene: PickBowlSceneCfg = PickBowlSceneCfg(num_envs=2048, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(1.8, 1.6, 1.3), lookat=(0.45, 0.0, 0.2))

    def __post_init__(self):
        task = resolve_params(self.task_params)
        self.decimation = _DECIMATION
        self.sim.dt = _SIM_DT
        self.sim.render_interval = _DECIMATION
        # ceil(episode_length_s / step_dt) must land exactly on episode_ticks.
        self.episode_length_s = (task.episode_ticks - 0.5) * TICK_DT
        # PhysX GPU buffers as droid-sim-evals sets them.
        self.sim.physx.gpu_temp_buffer_capacity = 2**26
        self.sim.physx.gpu_heap_capacity = 2**26
        self.sim.physx.gpu_collision_stack_size = 2**26
        for i in range(self.capture_envs):
            for name in G.CAM_POSE:
                setattr(self.scene, f"{name}_{i}", _camera(name, i))


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": PickBowlEnvCfg},
    disable_env_checker=True,
)


def make_env_cfg(*, num_envs: int, overrides=(), capture_envs: int = 0, seed: int | None = None) -> PickBowlEnvCfg:
    from gwm_rl.overrides import apply_overrides

    cfg = PickBowlEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.capture_envs = capture_envs
    if seed is not None:
        cfg.seed = seed
    apply_overrides(cfg, list(overrides))
    cfg.__post_init__()
    return cfg
