"""Place-only proposals from a single wrist RGB-D: held object -> every cluster.

PERCEPTION-ONLY CONTRACT (audit 2026-08-11): ground truth about the scene --
how many objects exist, what they are, where they sit -- is judge-side
knowledge and must never reach this proposer. Its only inputs are the wrist
observation (rgb/depth/K/extrinsics/q_init) and the robot's own model
(kinematics + collision spheres). No objects.json, no hardcoded destination
list, no heights lifted from asset dimensions.

The held object is WELDED to the gripper (weld_held_block.py) and never
released, so no grasps are needed -- no M2T2, no cuTAMP. Each candidate is a
deterministic two-segment cuRobo plan:

    [gripper close, MoveHolding(home -> pre-place above dest), Place(descend)]

- In-hand geometry is measured, not assumed: cloud points near the FK end
  effector that are (a) floating above the table and (b) outside the robot's
  own collision spheres are the held object. Its xy centroid and bottom
  height give the EE->object offset that turns landing targets into EE goals.
  The weld makes that offset constant across the plan.
- Destinations are ALL perceived clusters, mirroring the pick proposer's
  every-object candidate set: the proposer does not know which one the
  instruction means -- selection is the scorer's job. Per cluster the landing
  surface is read off the points: if the central region dips >= 3 cm below
  the rim the cluster is an open container (land on its inner floor), else
  it is solid (land on its top face). Landing xy is the centroid of that
  surface's points, so an off-centre visible floor patch aims where the
  camera actually saw floor.
- The close comes FIRST because the gripper transports the object with
  fingers shut (an open 2F-85 is wider than a small container mouth); the
  executor holds pose on a leading gripper step (tiptop_websocket._step_plan).
- The descent reuses cuTAMP's constrained-plan trick verbatim: PoseCostMetric
  holding x/y/rotation with z free is a straight vertical drop. Only the
  destination cluster is disabled during its own descent; everything else
  stays an obstacle. Approach planning keeps the full perceived world.
- No GoToInitial and no release: the episode ends with the object held inside
  the destination, which is what the judge scores (place_eval.py) and what
  the GWM RAT window needs -- the last sampled frame is the arm over the
  chosen destination, the most discriminative pose.

Candidate targets add a deterministic xy-offset pattern (no RNG, re-runs are
byte-stable). The --k-total budget is scene-independent (16, the same
whole-scene number as the pick proposer); it is split floor+remainder over
however many clusters perception found.

Run inside the droid/tiptop pixi shell (no servers needed):

    python -m gwm_tiptop.place_propose \
        --h5-path .../captures/scene6_1/wrist_obs.h5 \
        --output-dir .../gwm_integrate_doc/proposals/scene6_place_v2
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from scipy.spatial import ConvexHull, Delaunay, KDTree, QhullError

from tiptop.config import tiptop_cfg
from tiptop.motion_planning import build_curobo_solvers
from tiptop.perception.utils import (
    convert_trimesh_box_to_curobo_cuboid,
    convert_trimesh_to_curobo_mesh,
    depth_to_xyz,
)
from tiptop.planning import save_tiptop_plan

from gwm_tiptop.perception_geometric import cluster_objects, find_table_plane
from gwm_tiptop.propose_from_h5 import load_h5_observation, save_cluster_viz

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_tiptop.place_propose")

# Deterministic per-destination xy offsets (m), applied around the perceived
# landing centroid. Small enough to stay inside any mouth the gripper itself
# fits through.
XY_OFFSETS = [
    (0.0, 0.0), (0.018, 0.0), (-0.018, 0.0), (0.0, 0.018),
    (0.0, -0.018), (0.013, 0.013), (-0.013, 0.013), (0.013, -0.013), (-0.013, -0.013),
]
# Robot-side policy margins (scene-agnostic): how far the held object's bottom
# clears a destination's rim on approach, and hovers above the landing surface
# at the end of the descent.
APPROACH_CLEARANCE = 0.055
LANDING_CLEARANCE = 0.010
# A cluster whose central region dips at least this far below its rim is an
# open container (land inside); shallower ones are solid (land on top).
HOLLOW_MIN_DEPTH = 0.030
# In-hand detection: crop radius around the FK end effector, minimum height
# above the table (shared with perception_geometric's floating filter), and
# padding added to the robot's own collision spheres before rejecting a point
# as "self".
HAND_CROP_RADIUS = 0.20
FLOAT_TOLERANCE = 0.04
SELF_SPHERE_PAD = 0.010


def estimate_held_object(xyz: np.ndarray, ee_pos: np.ndarray, self_spheres: np.ndarray,
                         table_z: float) -> dict:
    """Measure the in-hand object from the raw world cloud.

    Points near the end effector, floating above the table, and outside the
    robot's padded collision spheres are clustered; the largest cluster is the
    held object. Returns its xy centroid, a robust bottom height, and the
    point count. Raises if nothing qualifies -- a place proposer without an
    object in hand has nothing to plan.
    """
    d_ee = np.linalg.norm(xyz - ee_pos, axis=1)
    keep = (d_ee < HAND_CROP_RADIUS) & (xyz[:, 2] > table_z + FLOAT_TOLERANCE) \
        & (xyz[:, 2] < ee_pos[2] + 0.12)
    pts = xyz[keep]
    if len(pts):
        d_sph = np.linalg.norm(pts[:, None, :] - self_spheres[None, :, :3], axis=2)
        pts = pts[(d_sph > self_spheres[None, :, 3] + SELF_SPHERE_PAD).all(axis=1)]
    if len(pts) < 40:
        raise RuntimeError(
            f"No in-hand object: {len(pts)} non-self floating points near the EE "
            f"(is anything actually held in this capture?)"
        )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    labels = np.array(pcd.cluster_dbscan(eps=0.008, min_points=25))
    if labels.max() < 0:
        raise RuntimeError("In-hand points did not form a cluster")
    sel = labels == np.bincount(labels[labels >= 0]).argmax()
    obj = pts[sel]
    held = {
        "xy": obj[:, :2].mean(axis=0),
        "bottom_z": float(np.percentile(obj[:, 2], 2)),
        "top_z": float(np.percentile(obj[:, 2], 98)),
        "npts": int(sel.sum()),
    }
    _log.info(
        f"held object: {held['npts']} pts, xy {np.round(held['xy'], 4).tolist()}, "
        f"z [{held['bottom_z']:.4f}, {held['top_z']:.4f}] "
        f"(EE at {np.round(ee_pos, 4).tolist()})"
    )
    return held


def landing_surface(label: str, hull_xy: np.ndarray, xyz: np.ndarray, table_z: float) -> dict:
    """Read a destination's landing surface off the raw cloud.

    Raw points inside the cluster's xy footprint (between table and rim) are
    the true observed surfaces -- the cluster pcds are hull-augmented and
    would fake a floor under solid objects. Hollow vs solid is decided by the
    INTERIOR points, eroded away from the footprint boundary: a container's
    floor is interior, while a solid object's only low points are its side
    faces at the boundary. The landing xy is the centroid of the points at
    landing height, i.e. where floor/top was actually seen -- an off-centre
    visible floor patch aims where the camera actually saw floor.

    The caller must pass a cloud with the robot hand region already removed:
    the held object hovers at the capture pose and its footprint can overlap
    a destination's, which would otherwise fake a rim at gripper height.
    """
    try:
        hull = Delaunay(hull_xy)
    except QhullError:
        return None
    inside = hull.find_simplex(xyz[:, :2]) >= 0
    # No above-table cut on the landing points: a thin-shelled container's
    # floor sits AT the detected plane height (measured: KLT floor within 1 mm
    # of the table RANSAC plane), so any z threshold either eats the floor or
    # admits table noise. The enclosure check below is what separates a floor
    # from open table inside a crescent footprint.
    pts = xyz[inside & (xyz[:, 2] > table_z - 0.010)]
    if len(pts) < 60:
        return None
    above = pts[pts[:, 2] > table_z + 0.015]
    if len(above) < 40:
        return None  # nothing meaningfully above the table in this footprint
    rim_z = float(np.percentile(above[:, 2], 98))
    boundary = ConvexHull(hull_xy)
    poly = hull_xy[boundary.vertices]
    segs = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(2, int(np.linalg.norm(b - a) / 0.004))
        segs.append(a + (b - a) * np.linspace(0.0, 1.0, n, endpoint=False)[:, None])
    d_boundary = KDTree(np.vstack(segs)).query(pts[:, :2])[0]
    interior = pts[d_boundary > 0.015]
    low = interior[interior[:, 2] <= rim_z - HOLLOW_MIN_DEPTH] if len(interior) else interior
    hollow, land_z, surf = False, rim_z, pts
    coverage = 0.0
    if len(low) >= 30:
        # Enclosure check: a real container's rim surrounds its interior 360
        # degrees; a crescent (banana, a split bowl arc) leaves the gap open,
        # so table points inside its convex footprint must not count as floor.
        c = low[:, :2].mean(axis=0)
        rim_band = pts[pts[:, 2] > rim_z - 0.015]
        if len(rim_band) >= 30:
            ang = np.arctan2(rim_band[:, 1] - c[1], rim_band[:, 0] - c[0])
            coverage = len(np.unique((ang // (2 * np.pi / 24)).astype(int))) / 24.0
        if coverage >= 0.8:
            hollow, land_z, surf = True, float(np.percentile(low[:, 2], 10)), low
    band = surf[np.abs(surf[:, 2] - land_z) < 0.012]
    target_xy = band[:, :2].mean(axis=0) if len(band) >= 20 else pts[:, :2].mean(axis=0)
    out = {"target_xy": target_xy, "land_z": land_z, "rim_z": rim_z, "hollow": bool(hollow),
           "rim_coverage": round(coverage, 2)}
    _log.info(
        f"{label}: {'hollow' if hollow else 'solid'}, rim_z {rim_z:.3f}, land_z {land_z:.3f}, "
        f"rim_coverage {coverage:.2f}, target_xy {np.round(target_xy, 3).tolist()}"
    )
    return out


def emit_plan(q_init: np.ndarray, dest: str, results: list) -> dict:
    steps = [{"type": "gripper", "label": f"Close(held@{dest})", "action": "close"}]
    for label, res in results:
        plan = res.get_interpolated_plan()
        steps.append({
            "type": "trajectory",
            "label": label,
            "positions": plan.position.cpu().numpy(),
            "velocities": plan.velocity.cpu().numpy(),
            "dt": res.interpolation_dt,
        })
    return {"version": "1.0.0", "q_init": q_init, "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-path", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k-total", type=int, default=16)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = tiptop_cfg()
    tensor_args = TensorDeviceType()
    obs = load_h5_observation(args.h5_path)

    # Perceived world: table plane + clusters from the home wrist RGB-D.
    depth = obs["depth"].copy()
    depth[~np.isfinite(depth)] = np.nan
    depth[(depth <= 0.05) | (depth > 4.0)] = np.nan
    xyz_map = depth_to_xyz(depth, obs["K"])
    xyz_map = xyz_map @ obs["world_from_cam"][:3, :3].T + obs["world_from_cam"][:3, 3]
    rgb_map = obs["rgb"].astype(np.float32) / 255.0
    xyz_flat = xyz_map[np.isfinite(xyz_map).all(axis=2)]

    table_trimesh, surface_z = find_table_plane(xyz_map, rgb_map)
    table_cuboid = convert_trimesh_box_to_curobo_cuboid(table_trimesh, name="table")
    object_trimeshes, object_pcds = cluster_objects(xyz_map, rgb_map, table_trimesh, surface_z + 0.015)
    save_cluster_viz(obs, object_pcds, args.output_dir / "clusters.png")

    obstacles = {l: convert_trimesh_to_curobo_mesh(m, l) for l, m in object_trimeshes.items()}

    from curobo.geom.types import WorldConfig

    _, motion_gen, _ = build_curobo_solvers(num_particles=32, num_spheres=64, include_workspace=False)
    motion_gen.update_world(WorldConfig(cuboid=[table_cuboid], mesh=list(obstacles.values())))

    # Robot-side quantities: FK end effector and the robot's own collision
    # spheres at the capture configuration (proprioception, not scene GT).
    q_init = tensor_args.to_device(obs["q_init"]).float()
    state = motion_gen.kinematics.get_state(q_init[None])
    ee0_pos = state.ee_pose.position[0].cpu().numpy()
    ee0_quat = state.ee_pose.quaternion  # (1, 4) on-device; targets reuse home orientation
    self_spheres = state.get_link_spheres()[0].cpu().numpy()  # (n, 4) xyzr

    held = estimate_held_object(xyz_flat, ee0_pos, self_spheres, surface_z)
    d_xy = ee0_pos[:2] - held["xy"]        # EE offset from held-object centre
    d_bottom = ee0_pos[2] - held["bottom_z"]  # EE height above held-object bottom

    # Landing surfaces for every cluster; budget split floor+remainder in
    # cluster order (largest first, deterministic). The hand region (gripper +
    # held object) hovers over the table at the capture pose and may overlap a
    # destination's footprint in xy, so cut it out of the cloud first.
    xyz_scene = xyz_flat[np.linalg.norm(xyz_flat - ee0_pos, axis=1) >= HAND_CROP_RADIUS]
    landings = {}
    for label, mesh in object_trimeshes.items():
        surf = landing_surface(label, np.asarray(mesh.vertices)[:, :2], xyz_scene, surface_z)
        if surf is not None:
            landings[label] = surf
    if not landings:
        raise SystemExit("no destination clusters with a readable landing surface")
    n_dest = len(landings)
    quotas = [args.k_total // n_dest + (1 if i < args.k_total % n_dest else 0) for i in range(n_dest)]
    if max(quotas) > len(XY_OFFSETS):
        raise SystemExit(f"quota {max(quotas)} exceeds the {len(XY_OFFSETS)} deterministic offsets")
    _log.info(f"{n_dest} destinations, quotas {quotas} (budget {args.k_total})")

    plan_cfg = MotionGenPlanConfig(
        timeout=2.0, enable_finetune_trajopt=False, time_dilation_factor=cfg.robot.time_dilation_factor
    )
    descend_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=tensor_args.to_device([0.1, 0.1, 0.1, 0.1, 0.1, 0.0]),
        project_to_goal_frame=True,
    )
    descend_cfg = plan_cfg.clone()
    descend_cfg.pose_cost_metric = descend_metric

    def ee_pose_for_held(bottom_xyz: np.ndarray) -> Pose:
        """EE goal that puts the held object's BOTTOM at bottom_xyz."""
        target = np.array([*(bottom_xyz[:2] + d_xy), bottom_xyz[2] + d_bottom])
        return Pose(position=tensor_args.to_device(target).float()[None], quaternion=ee0_quat)

    js_init = JointState.from_position(q_init[None])
    index, n_fail = [], 0
    for (dest, surf), quota in zip(landings.items(), quotas):
        approach_z = surf["rim_z"] + APPROACH_CLEARANCE
        place_z = surf["land_z"] + LANDING_CLEARANCE
        for k, (ox, oy) in enumerate(XY_OFFSETS[:quota]):
            target_xy = surf["target_xy"] + np.array([ox, oy])
            approach = motion_gen.plan_single(
                js_init, ee_pose_for_held(np.array([*target_xy, approach_z])), plan_cfg.clone()
            )
            if not approach.success.item():
                _log.warning(f"{dest}[{k}]: approach failed ({approach.status}); skipping")
                n_fail += 1
                continue
            js_pre = JointState.from_position(approach.get_interpolated_plan().position[-1:])
            motion_gen.world_coll_checker.enable_obstacle(enable=False, name=dest)
            try:
                descend = motion_gen.plan_single(
                    js_pre, ee_pose_for_held(np.array([*target_xy, place_z])), descend_cfg.clone()
                )
            finally:
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=dest)
            if not descend.success.item():
                _log.warning(f"{dest}[{k}]: descend failed ({descend.status}); skipping")
                n_fail += 1
                continue

            i = len(index)
            plan = emit_plan(obs["q_init"], dest, [
                (f"MoveHolding(held, {dest})", approach),
                (f"Place(held, {dest})", descend),
            ])
            dur = sum(len(st["positions"]) * st["dt"] for st in plan["steps"] if st["type"] == "trajectory")
            name = f"plan_{i:02d}_{dest}.json"
            save_tiptop_plan(plan, args.output_dir / name)
            index.append({"file": name, "target": dest, "grasp_confidence": 1.0,
                          "offset": [ox, oy], "traj_s": round(dur, 2),
                          "landing": {"target_xy": np.round(surf["target_xy"], 4).tolist(),
                                      "land_z": round(surf["land_z"], 4),
                                      "rim_z": round(surf["rim_z"], 4),
                                      "hollow": surf["hollow"]}})
            _log.info(f"{name}: {dur:.1f}s of trajectory (+1.33s close)")

    with open(args.output_dir / "proposals_index.json", "w") as f:
        json.dump({
            "h5": str(args.h5_path), "num_proposals": len(index),
            "held": {"d_xy": np.round(d_xy, 4).tolist(), "d_bottom": round(float(d_bottom), 4),
                     "npts": held["npts"]},
            "proposals": index,
        }, f, indent=2)
    _log.info(f"Wrote {len(index)} proposals ({n_fail} failures) to {args.output_dir}")
    if not index:
        raise SystemExit("no proposals produced")


if __name__ == "__main__":
    main()
