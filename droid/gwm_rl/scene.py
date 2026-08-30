"""The scene: robot, table, three static cabinets, bowl + two distractors,
scene8's lights, and the contact sensors the task and the cost read.

Obstacles (table, cabinets) are *kinematic rigid bodies* rather than static
colliders: PhysX contact reporting is body-to-body, so a sensor mounted on an
obstacle and filtered against the robot yields exactly "how hard is the arm
pushing on this thing" — the collision cost — with no guessing about which
robot link did it. Filtered reporting is one-to-many per sensor with one
pattern per filter body, hence one sensor per obstacle and one per finger.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from gwm_rl import geometry as G
from gwm_rl.geometry import FINGER_BODY_NAMES
from gwm_rl.robot import DROID_ROBOT_CFG, ROBOT_BODY_PATHS

_ASSETS = G.ASSETS

# A free body wedged between the arm (a stiff, gravity-free pusher) and a
# kinematic obstacle gets launched by the solver; these caps keep any such
# event inside the scene's scale (the arm itself moves at < 1 m/s).
_FREE_BODY_PROPS = sim_utils.RigidBodyPropertiesCfg(
    max_depenetration_velocity=1.0, max_linear_velocity=3.0, max_angular_velocity=20.0
)

def _kinematic(usd: str, pos, rot=(1.0, 0.0, 0.0, 0.0), prim: str = "") -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSETS / usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos), rot=tuple(rot)),
    )


def _obstacle_sensor(name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        filter_prim_paths_expr=list(ROBOT_BODY_PATHS),
    )


@configclass
class PickBowlSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = DROID_ROBOT_CFG

    # -- obstacles (kinematic compound bodies)
    table = _kinematic("table.usda", G.TABLE_POS, prim="table")
    cab_red = _kinematic("cab_red.usda", G.CAB_RED.world_pos, G.CAB_RED.world_quat, prim="cab_red")
    cab_yellow = _kinematic("cab_yellow.usda", G.CAB_YELLOW.world_pos, G.CAB_YELLOW.world_quat, prim="cab_yellow")
    cab_blue = _kinematic("cab_blue.usda", G.CAB_BLUE.world_pos, G.CAB_BLUE.world_quat, prim="cab_blue")

    # -- free objects, reset onto their settled poses
    bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSETS / "bowl.usda"), activate_contact_sensors=True, rigid_props=_FREE_BODY_PROPS
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=G.BOWL_POS, rot=G.BOWL_QUAT),
    )
    banana = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/banana",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSETS / "banana.usda"), activate_contact_sensors=True, rigid_props=_FREE_BODY_PROPS
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=G.BANANA_POS, rot=G.BANANA_QUAT),
    )
    block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/block",
        spawn=sim_utils.CuboidCfg(
            size=(G.BLOCK_SIZE, G.BLOCK_SIZE, G.BLOCK_SIZE),
            rigid_props=_FREE_BODY_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=G.BLOCK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=G.BLOCK_COLOR, roughness=0.6),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=G.BLOCK_POS),
    )

    # -- the floor, where the table's legs end: an object knocked off the table
    # -- lands 63 cm down instead of falling for the rest of the episode
    # -- (an invisible collider, so the photos keep the HDRI room's floor)
    floor = AssetBaseCfg(
        prim_path="/World/Floor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, G.FLOOR_Z - 0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(400.0, 400.0, 0.02), collision_props=sim_utils.CollisionPropertiesCfg(), visible=False
        ),
    )

    # -- lights, as scene8 + the DROID env's sphere light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, texture_file=str(G.HDRI), texture_format="latlong"),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.38268343236508984, 0.0, 0.0, 0.9238795325112867)),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.6532814824381883, 0.2705980500730985, 0.27059805007309845, 0.6532814824381882)
        ),
    )
    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(intensity=5000.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    # -- contact sensors: the fingers against the bowl (grasp detection) ...
    contact_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/" + FINGER_BODY_NAMES[0],
        filter_prim_paths_expr=["{ENV_REGEX_NS}/bowl"],
    )
    contact_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/" + FINGER_BODY_NAMES[1],
        filter_prim_paths_expr=["{ENV_REGEX_NS}/bowl"],
    )
    # ... and every obstacle against the whole robot (collision cost)
    contact_table = _obstacle_sensor("table")
    contact_cab_red = _obstacle_sensor("cab_red")
    contact_cab_yellow = _obstacle_sensor("cab_yellow")
    contact_cab_blue = _obstacle_sensor("cab_blue")
    contact_block = _obstacle_sensor("block")
    contact_banana = _obstacle_sensor("banana")
