"""Static workspace obstacles for the `zhiwei` rig.

`tiptop/workspace.py` dispatches `panda_robotiq` to `fr3_workspace()`, which is
MIT LIS's bench: a Vention table, a wall, an iPad, a camera pillar. On this rig
that geometry is wrong in both directions at once -- it invents obstacles where
there is free space, and leaves the real table edges unmodelled. Installed over
that dispatch by `install_rig_workspace.py`; the geometry itself lives here so
the pristine tiptop worktree only carries a three-line patch.

Frame: `panda_link0`. +x forward (the direction the arm reaches at q_home),
+y to the robot's LEFT, +z up. The robot is bolted to the table, so the table
top is z = 0.

Tape measurements:

    table top, floor to surface   0.61 m
    ROBOT BASE, floor to mount    0.555 m     <- the robot is NOT on this table
    => table top in base frame    +0.055 m
    table edge to the LEFT        0.50 m      (1 m wide, centred on the base)
    table edge to the RIGHT       0.50 m
    table extent BEHIND the base  0.50 m
    forward                       beyond the arm's reach
    overhead                      nothing
    external camera               outside the workspace, beyond the side
                                  keep-out

Over-approximate rather than under-: TiPToP's own guidance, and the asymmetry
of the cost -- a phantom obstacle loses a plan, a missing one loses hardware.
"""

from curobo.geom.types import Cuboid
from cutamp.envs.utils import unit_quat

# --- measured -------------------------------------------------------------
# The robot base mounts 55.5 cm above the floor and the table top is at 61 cm,
# so the working surface sits 55 mm ABOVE the base plane. Getting this wrong is
# the dangerous direction: with the table modelled at z = 0 the planner believes
# it has 55 mm more clearance than it has, and drives the fingers into it.
#
# Corroborated by the wrist camera: fitting the table plane to its depth at
# q_home puts the camera 359.2 mm from the surface, which places it 136 mm
# above the TCP. That matches where the camera physically is -- on the coupling,
# behind the 212 mm-long 2F-140's fingertips. Under the z = 0 assumption the
# same fit gives 81 mm, which would put the camera halfway down the fingers.
TABLE_TOP_Z = 0.055
TABLE_HEIGHT = 0.61
EDGE_LEFT_Y = 0.50         # +y  -- 1 m table, centred on the base
EDGE_RIGHT_Y = -0.50       # -y
TABLE_BACK_X = -0.50       # table extends this far behind the base

# ASSUMED. The table does not pass under the robot -- checked against the pose
# the arm is physically parked in, where two base spheres sit below the
# corrected table top yet the hardware is plainly not in contact. So the slab
# has to start clear of the base. This is the near edge, measured forward from
# the base axis; replace it with a tape measurement.
TABLE_NEAR_X = 0.15
KEEPOUT_HEIGHT = 1.20      # taller than anything the arm can reach

# How far below the real surface to sink the collision slab.
#
# tiptop inserts its OWN table into the collision world, from the RANSAC fit of
# the live point cloud, and deliberately sinks it 20 mm:
#
#     segmentation.py:237
#     height_offset = surface_z - table_center[2] - extents[2] / 2 - 0.02
#
# That 20 mm is the clearance a grasp needs -- fingers have to close around an
# object that is *resting on* the surface, so a collision table flush with the
# surface makes every top-down grasp a collision.
#
# Our slab is a second table on top of that one. Flush with TABLE_TOP_Z it
# would sit 20 mm HIGHER than the one tiptop has just carved clearance into,
# re-blocking exactly that gap and failing every pick with
# MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION. Matching tiptop's own
# 20 mm hands the tabletop back to the detected table, which is the one that
# tracks the real surface.
TABLE_COLLISION_SINK = 0.020

# --- ASSUMED ---------------------------------------------------------------
TABLE_FRONT_X = 1.00       # past the arm's reach, so the exact value is moot
CEILING_Z = 1.20           # nothing is physically overhead; this only caps
                           # cuRobo from planning absurd excursions
SIDE_WALL_THICK = 0.10


def _slab(name, x0, x1, y0, y1, z0, z1, color):
    return Cuboid(
        name,
        dims=[x1 - x0, y1 - y0, z1 - z0],
        pose=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, *unit_quat],
        color=color,
    )


def zhiwei_workspace() -> tuple[Cuboid, ...]:
    # Top sunk by TABLE_COLLISION_SINK so tiptop's detected table governs the
    # surface. This slab's job is the volume BELOW the table, and the region
    # outside whatever the camera happened to see.
    table = _slab("table_body",
                  TABLE_NEAR_X, TABLE_FRONT_X, EDGE_RIGHT_Y, EDGE_LEFT_Y,
                  TABLE_TOP_Z - TABLE_HEIGHT, TABLE_TOP_Z - TABLE_COLLISION_SINK,
                  color=[222, 184, 135])

    # Both sides are open air. Nothing to collide with, but nothing to catch
    # a dropped object either, and past the edge is a 0.61 m fall -- so they stay
    # modelled as keep-outs rather than being deleted.
    left_keepout = _slab("left_keepout",
                         TABLE_BACK_X, TABLE_FRONT_X,
                         EDGE_LEFT_Y, EDGE_LEFT_Y + SIDE_WALL_THICK,
                         TABLE_TOP_Z, TABLE_TOP_Z + KEEPOUT_HEIGHT,
                         color=[255, 0, 255])

    right_keepout = _slab("right_keepout",
                          TABLE_BACK_X, TABLE_FRONT_X,
                          EDGE_RIGHT_Y - SIDE_WALL_THICK, EDGE_RIGHT_Y,
                          TABLE_TOP_Z, TABLE_TOP_Z + KEEPOUT_HEIGHT,
                          color=[255, 0, 255])

    # Behind the base is where the operator stands and the cables run.
    back_keepout = _slab("back_keepout",
                         TABLE_BACK_X - SIDE_WALL_THICK, TABLE_BACK_X,
                         EDGE_RIGHT_Y - SIDE_WALL_THICK, EDGE_LEFT_Y + SIDE_WALL_THICK,
                         TABLE_TOP_Z, TABLE_TOP_Z + KEEPOUT_HEIGHT,
                         color=[255, 0, 255])

    ceiling = _slab("ceiling",
                    TABLE_BACK_X, TABLE_FRONT_X,
                    EDGE_RIGHT_Y, EDGE_LEFT_Y,
                    CEILING_Z, CEILING_Z + 0.02,
                    color=[225, 225, 225])

    return (table, left_keepout, right_keepout, back_keepout, ceiling)
