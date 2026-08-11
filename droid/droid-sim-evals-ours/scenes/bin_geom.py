"""bin_geom: shared geometry for the recolored KLT bins used as placement targets.

The stock `small_KLT_visual_collision` footprint is a 0.198 x 0.297 rectangle.
Uniformly scaling it keeps that 3:2 aspect, which forces the long edge to point
at something -- at scene6's occupancy the only orientation that fits puts it
within 33 mm of the banana. Scaling NON-uniformly to a square footprint gives a
smaller, orientation-free bin.

WHICH FRAME THE SCALE ACTS IN DEPENDS ON xformOpOrder, and the two container
assets in this repo disagree -- the LAST op in the list is applied to the
geometry FIRST (innermost):

    _24_bowl  ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
              -> scale innermost, acts in the asset's LOCAL frame. Confirmed on
                 scene5's non-uniformly scaled bowl (0.018717684, 0.012486185,
                 0.018717684): local-frame scaling predicts a 0.302 x 0.302 x
                 0.069 world bbox vs 0.302 x 0.302 x 0.070 measured, whereas
                 parent-frame scaling would predict 0.302 x 0.202.
    KLT bin   ["xformOp:translate", "xformOp:scale", "xformOp:orient"]
              -> orient innermost, so the scale acts in the PARENT frame, AFTER
                 the yaw. Read straight off the composed local matrix: with
                 scale (0.38769, 0.58128, ...) the linear part is
                 [[0, 0.38769, 0], [-0.58128, 0, 0], ...], i.e. component 0
                 scales world X and component 1 scales world Y.

So for the bins the scale vector is simply per-world-axis, and since yaw +90
puts the asset's long axis (KLT_Y) on world X, a square footprint of edge `size`
needs
    scale = (size / KLT_Y, size / KLT_X, height / KLT_Z)
"""

TABLE_TOP_Z = 0.045141201291582375
# small_KLT_visual_collision outer bbox in asset-local metres; origin at the bbox centre
KLT_X, KLT_Y, KLT_Z = 0.19784, 0.29663, 0.14636
# inner clear opening and the wall thickness that produces it
KLT_INNER_X, KLT_INNER_Y = 0.180, 0.262
KLT_WALL_X, KLT_WALL_Y = 0.0012, 0.0011
KLT_INNER_DEPTH = 0.1442
BIN_DROP = 0.005  # spawn clearance above the table (stock KLT uses 0.0071)

DEFAULT_SIZE = 0.115
DEFAULT_HEIGHT = 0.068

# Settled world AABBs of scene6's three stock objects, ((x0,x1),(y0,y1),(z0,z1)):
# captures/scene6_0/objects.json centres + the assets' USD bboxes.
STOCK = {
    "rubiks_cube": ((0.3328, 0.4049), (0.1542, 0.2263), (0.0451, 0.1172)),
    "_24_bowl": ((0.4217, 0.5828), (0.0337, 0.1952), (0.0463, 0.1013)),
    "_11_banana": ((0.5120, 0.5820), (-0.3400, -0.1460), (0.0260, 0.0999)),
}


def bin_scale(size: float = DEFAULT_SIZE, height: float = DEFAULT_HEIGHT):
    """xformOp:scale for a square `size` x `size` world footprint.

    Parent-frame, per world axis (see module docstring): world X gets the
    asset's long axis KLT_Y after the yaw, world Y gets KLT_X.
    """
    return size / KLT_Y, size / KLT_X, height / KLT_Z


def bin_spawn_z(height: float = DEFAULT_HEIGHT) -> float:
    return TABLE_TOP_Z + height / 2 + BIN_DROP


def bin_aabb(x: float, y: float, size: float = DEFAULT_SIZE, height: float = DEFAULT_HEIGHT):
    """World AABB ((x0,x1),(y0,y1),(z0,z1)). Square footprint, so yaw is irrelevant."""
    h = size / 2
    z = bin_spawn_z(height)
    return (x - h, x + h), (y - h, y + h), (z - height / 2, z + height / 2)


def bin_report(size: float = DEFAULT_SIZE, height: float = DEFAULT_HEIGHT) -> str:
    """sx/sy are per world axis, so world-X quantities come from the asset's Y extent."""
    sx, sy, sz = bin_scale(size, height)
    return (
        f"footprint {size:.3f}x{size:.3f} m, height {height:.3f} m, scale=({sx:.4f},{sy:.4f},{sz:.4f})\n"
        f"    inner opening {KLT_INNER_Y * sx:.3f} (X) x {KLT_INNER_X * sy:.3f} (Y) m, "
        f"depth {KLT_INNER_DEPTH * sz:.3f} m\n"
        f"    walls {KLT_WALL_Y * sx * 1000:.2f} mm (X) / {KLT_WALL_X * sy * 1000:.2f} mm (Y) "
        f"[stock 1.10/1.20 mm; 0.72 mm settled with zero drift in the uniform-0.6 rev]"
    )
