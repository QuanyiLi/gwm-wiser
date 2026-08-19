"""Mask-free scene decomposition: table plane + DBSCAN clusters (plan G-2/G-3).

Replaces the Gemini/SAM2 path: no labels, no masks. The table is found
geometrically (horizontal RANSAC plane near the robot base — the DROID rig
bolts the robot to the table, so the plane lives in a narrow height band
around z=0), and objects are anonymous point-cloud clusters above it.
"""

import logging

import numpy as np
import open3d as o3d
import trimesh

from tiptop.perception.segmentation import aabb_to_cuboid, augment_with_base_projections

_log = logging.getLogger(__name__)


def find_table_plane(
    xyz_world: np.ndarray,
    rgb: np.ndarray,
    max_planes: int = 5,
    normal_z_min: float = 0.90,
    z_band: tuple[float, float] = (-0.25, 0.15),
    workspace_radius: float = 1.4,
    voxel_size: float = 0.005,
) -> tuple[trimesh.primitives.Box, float]:
    """Mask-free variant of segment_table_with_ransac.

    Iterative RANSAC over the workspace-cropped cloud; candidate planes must be
    near-horizontal (|n_z| >= normal_z_min) with height inside z_band (the
    robot-base-relative band where the DROID table lives); the winner is the
    one with the most inliers. The trailing box construction mirrors the
    original (outlier removal, largest DBSCAN cluster, percentile AABB).
    """
    if xyz_world.ndim != 3:
        raise ValueError(f"Expected structured (H, W, 3) cloud, got {xyz_world.shape}")

    valid = ~np.isnan(xyz_world).any(axis=2)
    xyz = xyz_world[valid]
    col = rgb[valid]
    near = np.linalg.norm(xyz[:, :2], axis=1) < workspace_radius
    xyz, col = xyz[near], col[near]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(col)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    remaining = pcd
    best_pcd, best_inliers, best_plane = None, -1, None
    for i in range(max_planes):
        if len(remaining.points) < 50:
            break
        (a, b, c, d), idxs = remaining.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        n = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        inlier_pcd = remaining.select_by_index(idxs)
        plane_z = np.asarray(inlier_pcd.points)[:, 2].mean()
        horizontal = abs(n[2]) >= normal_z_min
        in_band = z_band[0] <= plane_z <= z_band[1]
        _log.debug(
            f"Plane {i}: n_z={n[2]:+.2f} z={plane_z:+.3f} inliers={len(idxs)} "
            f"horizontal={horizontal} in_band={in_band}"
        )
        if horizontal and in_band and len(idxs) > best_inliers:
            best_inliers = len(idxs)
            best_pcd = inlier_pcd
            # Keep the fitted plane, oriented +z up and normalised, so callers
            # can measure height ABOVE THE TABLE rather than above world z.
            # On a perfectly level table the two are the same and nothing
            # changes; on the zhiwei rig the perceived plane is tilted 2.88 deg
            # (the hand-eye rotational residual), which is 48 mm of world-z
            # spread across an 0.85 m footprint -- three times the clearance
            # that separates an object from the table. See cluster_objects.
            s = np.sign(c) or 1.0
            best_plane = np.array([a, b, c, d], dtype=np.float64) * s / np.linalg.norm([a, b, c])
        remaining = remaining.select_by_index(idxs, invert=True)

    if best_pcd is None:
        raise RuntimeError(
            f"No horizontal plane found in z band {z_band} within {workspace_radius} m "
            f"of the robot base (tried {max_planes} planes)."
        )
    _log.info(f"Table plane selected with {best_inliers} inliers")

    table_pcd, _ = best_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    labels = np.array(table_pcd.cluster_dbscan(eps=3 * voxel_size, min_points=10))
    if len(labels) == 0 or labels.max() < 0:
        raise RuntimeError("DBSCAN found no clusters in table plane inliers.")
    largest = int(np.bincount(labels[labels >= 0]).argmax())
    table_pcd = table_pcd.select_by_index(np.where(labels == largest)[0])

    pts = np.asarray(table_pcd.points)
    xy_min = np.percentile(pts[:, :2], 2, axis=0)
    xy_max = np.percentile(pts[:, :2], 98, axis=0)
    aabb = np.stack([np.append(xy_min, pts[:, 2].min()), np.append(xy_max, pts[:, 2].max())])
    surface_z = pts[:, 2].mean()

    table_box = aabb_to_cuboid(aabb, "table")
    extents = table_box.extents
    center = table_box.center_mass
    table_box.apply_translation([0, 0, surface_z - center[2] - extents[2] / 2 - 0.02])
    table_box.visual.face_colors = np.append(
        (np.asarray(table_pcd.colors).mean(0) * 255).astype(np.uint8), 255
    )
    tilt = np.degrees(np.arccos(np.clip(best_plane[2], -1.0, 1.0)))
    _log.info(f"Table surface at z = {surface_z:.3f}, dims = {table_box.extents}, "
              f"plane tilt {tilt:.2f} deg")
    # The fitted plane rides along on the box's metadata rather than in the
    # return tuple, so every existing two-value caller is untouched.
    table_box.metadata = {**(table_box.metadata or {}),
                          "plane": best_plane.tolist(), "surface_z": float(surface_z),
                          "tilt_deg": float(tilt)}
    # Return the true surface height alongside the box: the box top is
    # deliberately sunk 2 cm below the surface (tiptop collision convention),
    # so it must NOT be used as a segmentation boundary.
    return table_box, surface_z


def _merge_xy_overlapping_clusters(points: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Union clusters whose XY convex hulls contain each other's points.

    A partially occluded object (e.g. a bowl seen as two rim arcs) splits into
    DBSCAN clusters that are far apart in 3D but whose XY hulls span the same
    footprint — the occluded body connects them. Distinct adjacent objects have
    disjoint XY hulls, so a small eps stays safe.
    """
    from scipy.spatial import Delaunay, QhullError

    hulls = {}
    for cl in range(n_clusters):
        xy = points[labels == cl][:, :2]
        if len(xy) >= 4:
            try:
                hulls[cl] = Delaunay(xy)
            except QhullError:
                pass

    parent = list(range(n_clusters))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    from scipy.spatial import KDTree

    xy_trees = {
        cl: KDTree(points[labels == cl][:, :2]) for cl in range(n_clusters) if (labels == cl).sum() > 0
    }
    for a in range(n_clusters):
        for b in range(a + 1, n_clusters):
            if a not in xy_trees or b not in xy_trees:
                continue
            xy_a = points[labels == a][:, :2]
            xy_b = points[labels == b][:, :2]
            # Rule 1: occluded-body containment (opposite arcs of a hollow object)
            frac = 0.0
            if a in hulls:
                frac = max(frac, (hulls[a].find_simplex(xy_b) >= 0).mean())
            if b in hulls:
                frac = max(frac, (hulls[b].find_simplex(xy_a) >= 0).mean())
            # Rule 2: rim slivers — 3D depth discontinuity but XY-adjacent
            min_xy_dist = xy_trees[b].query(xy_a, k=1)[0].min()
            if frac > 0.15 or min_xy_dist < 0.008:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
                    _log.info(
                        f"Merging clusters {a} and {b} (XY overlap {frac:.0%}, min XY gap {min_xy_dist * 1000:.1f} mm)"
                    )

    merged = labels.copy()
    for cl in range(n_clusters):
        merged[labels == cl] = find(cl)
    return merged


def cluster_objects(
    xyz_world: np.ndarray,
    rgb: np.ndarray,
    table_box: trimesh.primitives.Box,
    table_top_z: float,
    eps: float = 0.015,
    min_points: int = 40,
    min_cluster_points: int = 120,
    max_objects: int = 8,
    xy_margin: float = 0.02,
    voxel_size: float = 0.004,
    use_plane_normal: bool = False,
    robot_spheres: np.ndarray | None = None,
    robot_margin: float = 0.02,
) -> tuple[dict[str, trimesh.Trimesh], dict[str, o3d.geometry.PointCloud]]:
    """DBSCAN the above-table points into anonymous objects `object_0..N`.

    Mirrors segment_pointcloud_by_masks' output contract (hull trimeshes +
    per-object pcds) so everything downstream — grasp association, cuTAMP env
    construction — is unchanged. Clusters are ordered by size; tiny clusters
    and clusters outside the table's XY footprint are dropped.

    `robot_spheres` ((N,4) xyz+radius, the robot's own collision spheres at the
    capture configuration) replaces the height heuristic for identifying the
    ARM. The heuristic deletes any cluster whose lowest point sits more than
    `resting_tolerance` above the table, which assumes every object rests on
    the table AND is seen down to within 4 cm of it. A tall object viewed from
    a top-down wrist camera fails the second half: the camera sees its top face
    and almost none of its vertical sides, so its lowest VISIBLE point is high.
    Measured on the zhiwei rig, an upended 93 mm box was deleted as "robot arm"
    at a lowest visible point of 59 mm, and the instruction that referred to it
    then had no referent to find. Where the robot actually is, is not something
    to infer from height -- FK knows it. Off by default so sim reproduces.

    `use_plane_normal` measures "above the table" perpendicular to the plane
    `find_table_plane` fitted, instead of along world z. Off by default so sim
    results reproduce exactly (droid-sim's table is level, where the two agree
    to floating point). Hardware needs it on: the zhiwei rig's perceived table
    is tilted 2.88 deg, which spreads its own surface over 48 mm of world z
    across the capture footprint — more than three times the 15 mm clearance
    that is supposed to separate an object from the table. With a horizontal
    cut the high end of the table survives as a phantom "object" the size of
    the tabletop, and merges into whatever real object sits near it.
    """
    valid = ~np.isnan(xyz_world).any(axis=2)
    xyz = xyz_world[valid]
    col = rgb[valid]

    plane = None
    if use_plane_normal:
        meta = table_box.metadata or {}
        if "plane" not in meta:
            raise ValueError("use_plane_normal needs a table_box from find_table_plane")
        plane = np.asarray(meta["plane"], dtype=np.float64)
        clearance = table_top_z - float(meta["surface_z"])

    def _height(pts: np.ndarray) -> np.ndarray:
        """Height above the table: perpendicular to the fitted plane, or world z."""
        if plane is None:
            return pts[:, 2] - table_top_z
        return pts @ plane[:3] + plane[3] - clearance

    (x0, y0), (x1, y1) = table_box.bounds[0, :2], table_box.bounds[1, :2]
    keep = (
        (_height(xyz) > 0)
        & (xyz[:, 0] > x0 + xy_margin) & (xyz[:, 0] < x1 - xy_margin)
        & (xyz[:, 1] > y0 + xy_margin) & (xyz[:, 1] < y1 - xy_margin)
    )
    if keep.sum() < min_cluster_points:
        raise RuntimeError(f"Only {int(keep.sum())} points above the table — nothing to cluster.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[keep])
    pcd.colors = o3d.utility.Vector3dVector(col[keep])
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    n_clusters = int(labels.max()) + 1 if len(labels) and labels.max() >= 0 else 0
    if n_clusters == 0:
        raise RuntimeError("DBSCAN found no object clusters above the table.")

    # Drop floating (robot-arm) clusters BEFORE merging: the arm hangs directly
    # above objects at the capture pose, so the XY-containment merge rule would
    # otherwise absorb arm fragments into the object below them.
    pts_all = np.asarray(pcd.points)
    resting_tolerance = 0.04
    arm_tree = None
    if robot_spheres is not None:
        from scipy.spatial import KDTree

        sph = np.asarray(robot_spheres, dtype=np.float64)
        sph = sph[sph[:, 3] > 0.0]      # cuRobo pads the buffer with negative radii
        arm_tree = (KDTree(sph[:, :3]), sph[:, 3])

    for cl in range(n_clusters):
        sel = labels == cl
        if not sel.any():
            continue
        pts = pts_all[sel]
        if arm_tree is not None:
            tree, radii = arm_tree
            dist, idx = tree.query(pts)
            on_arm = float((dist <= radii[idx] + robot_margin).mean())
            if on_arm > 0.5:
                _log.info(
                    f"Skipping cluster {cl} ({int(sel.sum())} pts): {on_arm:.0%} of it "
                    "lies inside the robot's own collision spheres"
                )
                labels[sel] = -1
            continue
        # Written as two branches rather than one `_height(...) > tol` so the
        # default path is the ORIGINAL comparison, floating-point included.
        lowest = pts[:, 2].min() if plane is None else _height(pts).min()
        floating = (lowest > table_top_z + resting_tolerance) if plane is None \
            else (lowest > resting_tolerance)
        if floating:
            _log.info(
                f"Skipping floating cluster {cl} (lowest point {lowest:.3f}, "
                f"{int(sel.sum())} pts) — likely robot arm"
            )
            labels[sel] = -1

    labels = _merge_xy_overlapping_clusters(pts_all, labels, n_clusters)

    sizes = np.bincount(labels[labels >= 0])
    order = np.argsort(sizes)[::-1]
    meshes: dict[str, trimesh.Trimesh] = {}
    pcds: dict[str, o3d.geometry.PointCloud] = {}
    obj_idx = 0
    for cl in order:
        if sizes[cl] < min_cluster_points or obj_idx >= max_objects:
            continue
        opcd = pcd.select_by_index(np.where(labels == cl)[0])
        opcd, _ = opcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        if len(opcd.points) < min_cluster_points:
            continue
        pts, cols = augment_with_base_projections(
            np.asarray(opcd.points), np.asarray(opcd.colors)
        )
        hull_pcd = o3d.geometry.PointCloud()
        hull_pcd.points = o3d.utility.Vector3dVector(pts)
        hull_pcd.colors = o3d.utility.Vector3dVector(cols)
        try:
            hull, _ = hull_pcd.compute_convex_hull()
        except Exception as e:  # degenerate clusters (planar slivers)
            _log.warning(f"Skipping cluster {cl}: convex hull failed ({e})")
            continue
        name = f"object_{obj_idx}"
        mesh = trimesh.Trimesh(
            vertices=np.asarray(hull.vertices), faces=np.asarray(hull.triangles), process=True
        )
        mesh.metadata = {"name": name, "centroid": np.asarray(hull.vertices).mean(0).tolist()}
        mesh.visual.face_colors = np.append(
            (np.asarray(opcd.colors).mean(0) * 255).astype(np.uint8), 255
        )
        meshes[name] = mesh
        pcds[name] = hull_pcd
        _log.info(f"{name}: {len(opcd.points)} pts, centroid {np.round(mesh.metadata['centroid'], 3)}")
        obj_idx += 1

    if not meshes:
        raise RuntimeError("All clusters were rejected (too small or degenerate).")
    return meshes, pcds
