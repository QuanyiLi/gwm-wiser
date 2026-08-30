"""The DROID robot as `droid-sim-evals` configures it: Franka Panda + Robotiq
2F-85 (flattened USD, 13 DoF, 8 actuated), gravity off, stiff PD.

Copied from `droid-sim-evals/src/sim_evals/environments/nvidia_droid.py` so
this folder does not import that package; the numbers are theirs. The
gripper's five follower joints have no actuator entry and run on the USD's
own drives, exactly as in the drawer experiment's executions.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from gwm_rl.geometry import ROBOT_USD

#: Every rigid body of the robot except the fixed base, as explicit prim
#: paths: PhysX filtered contact reporting wants one pattern per filter body.
_GRIPPER = "{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/"
ROBOT_BODY_PATHS = [f"{{ENV_REGEX_NS}}/robot/panda_link{i}" for i in range(1, 9)] + [
    _GRIPPER + name
    for name in (
        "base_link",
        "left_outer_knuckle", "right_outer_knuckle",
        "left_outer_finger", "right_outer_finger",
        "left_inner_finger", "right_inner_finger",
        "left_inner_knuckle", "right_inner_knuckle",
    )
]

DROID_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * math.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * math.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * math.pi,
            "panda_joint7": 0.0,
            "finger_joint": 0.0,
            "right_outer.*": 0.0,
            "left_inner.*": 0.0,
            "right_inner.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=1000.0,
            damping=None,
            velocity_limit=1.0,
        ),
    },
)
