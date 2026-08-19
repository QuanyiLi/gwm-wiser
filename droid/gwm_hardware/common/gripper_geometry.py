"""How far the CLOSED gripper reaches past the frame the planner plans with.

`grasp_frame` is cuRobo's `ee_link`, and `build_2f140.py` calibrates it to the
fingertip plane **with the gripper open** -- deliberately, because that is the
convention the 2F-85 the whole stack was tuned on uses. The cuRobo config then
locks `finger_joint: 0.0`, so the planner's collision model is the OPEN
gripper too. Consistent, and correct for a pick, which approaches open.

It is wrong for anything CARRIED. The 2F-140 is a four-bar linkage: closing
swings each finger inward along an arc that also drops it, so the real
fingertips sit BELOW `grasp_frame` by an amount the planner cannot see.

Measured on this rig's generated model:

    driver 0.00 (open)    fingertip 212.0 mm == grasp_frame
    driver 0.35 (half)    fingertip 229.5 mm    +17.5 mm
    driver 0.70 (closed)  fingertip 235.7 mm    +23.7 mm

That 23.7 mm is a systematic error in the dangerous direction, and it is what
drove the first hardware place into the table: the planner believed the
fingertips cleared the container floor by 28.4 mm while they actually cleared
it by 4.7 mm -- less than this rig's own hand-eye residual. The trajectory
faulted mid-descent.

This is the same fact `inspect_plan` learned the hard way from the other side
(docs/tiptop-modifications.md): judging the gripper open called a perfectly
good cup grasp impossible, because open is the one configuration where the
pads sit highest.

Measured from the model rather than hardcoded, so it follows the URDF: change
the gripper, regenerate, and the number changes with it.
"""

import logging

_log = logging.getLogger(__name__)

_CACHE: dict = {}

# Driver angles, from the URDF's own limit (0 = fully open, 0.7 = fully closed).
_OPEN, _CLOSED = 0.0, 0.7
_FINGER_LINKS = [f"{s}_{p}" for s in ("left", "right")
                 for p in ("outer_finger", "inner_finger", "inner_finger_pad")]


def _fingertip_z(robot, driver: float) -> tuple[float, float]:
    """(lowest fingertip, grasp_frame) along the gripper axis, in the base link."""
    import numpy as np
    import trimesh

    robot.update_cfg({"finger_joint": driver})
    tip = -np.inf
    for name in _FINGER_LINKS:
        T = robot.get_transform(name, "robotiq_140_base_link")
        for col in robot.link_map[name].collisions:
            geom = col.geometry
            if geom.mesh is not None:
                mesh = trimesh.load(robot._filename_handler(geom.mesh.filename), force="mesh")
                verts = mesh.vertices
            elif geom.box is not None:
                verts = trimesh.creation.box(extents=geom.box.size).vertices
            else:
                continue
            origin = col.origin if col.origin is not None else np.eye(4)
            world = (T @ origin @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
            tip = max(tip, float(world[:, 2].max()))
    return tip, float(robot.get_transform("grasp_frame", "robotiq_140_base_link")[2, 3])


def closed_tip_overhang(urdf_path=None) -> float:
    """Metres the closed fingertips reach beyond `grasp_frame`. >= 0.

    Add this to any clearance measured from `grasp_frame` while the gripper is
    holding something, or the clearance is fictitious by exactly this much.
    """
    from yourdfpy import URDF

    from gwm_hardware.common.paths import ASSETS

    path = str(urdf_path or (ASSETS / "panda_robotiq_2f_140.urdf"))
    if path in _CACHE:
        return _CACHE[path]

    robot = URDF.load(path, load_meshes=True, build_scene_graph=True)
    tip_open, gf = _fingertip_z(robot, _OPEN)
    tip_closed, _ = _fingertip_z(robot, _CLOSED)
    overhang = max(0.0, tip_closed - gf)
    _log.info(
        f"closed-gripper overhang {overhang * 1000:.1f} mm "
        f"(open tip {tip_open * 1000:.1f}, closed tip {tip_closed * 1000:.1f}, "
        f"grasp_frame {gf * 1000:.1f}) -- the planner locks the gripper OPEN, so this "
        "much of any carried-clearance is invisible to it"
    )
    _CACHE[path] = overhang
    return overhang


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"{closed_tip_overhang() * 1000:.1f} mm")


if __name__ == "__main__":
    main()
