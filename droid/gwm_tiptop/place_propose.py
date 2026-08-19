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
from scipy.spatial.transform import Rotation
import open3d as o3d
import torch
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from scipy.spatial import ConvexHull, Delaunay, KDTree, QhullError

from tiptop.config import tiptop_cfg
from gwm_tiptop.robot_fk import default_planning_solvers
from tiptop.perception.utils import (
    convert_trimesh_box_to_curobo_cuboid,
    convert_trimesh_to_curobo_mesh,
    depth_to_xyz,
)
from tiptop.planning import save_tiptop_plan

from gwm_tiptop.scene_cache import scene
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
# How far a candidate's first waypoint may sit from the pose the arm is
# actually in. Anything beyond this is a commanded jump, not a motion.
START_TOL_RAD = 0.02
# Rise across the held object's own footprint radius, above which a SOLID
# destination is not a surface anything would stay on. Dimensionless so it does
# not move with the payload; see the measurement table in `landing_surface`.
DEFAULT_MAX_SUPPORT_SLOPE = 0.18
# How near a cluster's hull has to sit to the measured in-hand points before
# that cluster IS the held object. Generous, because the two point sets come
# from the same depth frame and differ only by hull augmentation.
HELD_MATCH_RADIUS = 0.01


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
        "points": obj,
        "xy": obj[:, :2].mean(axis=0),
        "r90": float(np.percentile(np.linalg.norm(obj[:, :2] - obj[:, :2].mean(axis=0), axis=1), 90)),
        "bottom_z": float(np.percentile(obj[:, 2], 2)),
        "top_z": float(np.percentile(obj[:, 2], 98)),
        "npts": int(sel.sum()),
    }
    _log.info(
        f"held object: {held['npts']} pts, r90 {held['r90']*1000:.0f} mm, "
        f"xy {np.round(held['xy'], 4).tolist()}, "
        f"z [{held['bottom_z']:.4f}, {held['top_z']:.4f}] "
        f"(EE at {np.round(ee_pos, 4).tolist()})"
    )
    return held


def landing_surface(label: str, hull_xy: np.ndarray, xyz: np.ndarray, table_z: float,
                    r_need: float | None = None) -> dict:
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
    # Is the patch the held object would rest on actually FLAT? Every cluster
    # becomes a destination, so a 60 mm ball's apex was a placement target as
    # readily as a tray: it passes on area (its cap is 24 cm2, the held tomato
    # needs 14) and on plane tilt (a cap is symmetric, so the fit is level).
    # What it fails is how much the patch RISES across the footprint the object
    # has to rest on -- expressed as a SLOPE, peak-to-valley over that radius.
    #
    # Slope rather than millimetres, because the radius is the held object's own
    # and it changes between turns. Measured over four captures, held r90
    # ranging 18 - 36 mm:
    #
    #                             p2v            slope
    #     real container floors   1.5 - 3.6 mm   0.076 - 0.120   (7 samples)
    #     ball apex               5.7 - 7.3 mm   0.294 - 0.347
    #     table-edge clutter       19 -  29 mm   0.756 - 1.364
    #
    # In millimetres those bands very nearly touch, and on 2026-08-19 they did:
    # a smaller held object (r90 18 mm, down from 21) shrank the disc the ball's
    # cap was measured over, its p2v fell under a 5 mm limit, and the ball became
    # a destination again and took 5 of the 16 candidates. As a slope the same
    # ball reads 0.294 there and 0.347 elsewhere -- it does not move with the
    # payload.
    p2v = None
    if r_need is not None:
        disc = pts[(np.linalg.norm(pts[:, :2] - target_xy, axis=1) <= r_need)
                   & (np.abs(pts[:, 2] - land_z) < 0.030)]
        if len(disc) >= 20:
            p2v = float(np.percentile(disc[:, 2], 98) - np.percentile(disc[:, 2], 2))

    out = {"target_xy": target_xy, "land_z": land_z, "rim_z": rim_z, "hollow": bool(hollow),
           "rim_coverage": round(coverage, 2), "p2v": p2v,
           "slope": None if (p2v is None or not r_need) else p2v / float(r_need)}
    _log.info(
        f"{label}: {'hollow' if hollow else 'solid'}, rim_z {rim_z:.3f}, land_z {land_z:.3f}, "
        f"rim_coverage {coverage:.2f}, target_xy {np.round(target_xy, 3).tolist()}"
        + ("" if p2v is None else
           f", landing rise {p2v*1000:.1f} mm over {r_need*1000:.0f} mm (slope {p2v/r_need:.3f})")
    )
    return out


def snapshot(res) -> dict:
    """Copy a planner result's interpolated trajectory off the GPU, NOW.

    cuRobo's result objects reference planner-owned buffers, and this loop
    plans repeatedly with two different configs on one `MotionGen`. Read late,
    a result no longer describes the motion that was planned: on 2026-08-19
    every place candidate after the first came back with a 0.06 rad
    "approach" that began at the PREVIOUS candidate's end instead of at the
    capture pose -- up to 0.885 rad (51 deg) from where the arm actually was,
    which the controller refused to execute. Snapshotting between the plan and
    the next `plan_single` is what makes each candidate independent.
    """
    ip = res.get_interpolated_plan()
    return {"positions": ip.position.cpu().numpy().copy(),
            "velocities": ip.velocity.cpu().numpy().copy(),
            "dt": res.interpolation_dt}


def emit_plan(q_init: np.ndarray, dest: str, results: list, skip_close: bool = False) -> dict:
    # droid-sim starts with the gripper OPEN around a welded block, so the plan
    # has to close it before transporting. On hardware the gripper is ALREADY
    # holding the object -- the controller says so (is_grasped) -- and the
    # leading close costs twice over: it re-squeezes a real object for no
    # reason, and it prepends 1.33 s of stationary timeline that lands 2 of the
    # 6 RAT frames on a pose identical across every candidate, so a third of
    # the scoring evidence carries no signal at all. Place margins on this rig
    # sat at +0.0001 to +0.0021.
    steps = ([] if skip_close else
             [{"type": "gripper", "label": f"Close(held@{dest})", "action": "close"}])
    for label, snap in results:
        steps.append({"type": "trajectory", "label": label, **snap})
    return {"version": "1.0.0", "q_init": q_init, "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-path", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k-total", type=int, default=16)
    ap.add_argument("--use-plane-normal", action="store_true",
                    help="measure height above the FITTED table plane rather than world z. "
                         "Required on a rig whose perceived table is tilted; a no-op on a "
                         "level one. MUST match what the pick side used")
    ap.add_argument("--use-robot-arm-filter", action="store_true",
                    help="identify the arm by the robot's own collision spheres rather "
                         "than by cluster height")
    ap.add_argument("--max-support-slope", type=float, default=0.0,
                    help="rise over the held object's footprint radius, above which a SOLID "
                         "destination stops counting as a placement surface. Dimensionless, so "
                         "it does not drift when the payload changes size (0 = off, droid-sim "
                         "default; 0.18 on this rig separates container floors at 0.076-0.120 "
                         "from ball tops at 0.294-0.347)")
    ap.add_argument("--skip-leading-close", action="store_true",
                    help="omit the plan's opening gripper-close. Correct wherever the "
                         "gripper is ALREADY holding the object, which is every hardware "
                         "place; droid-sim needs it because its block is welded into an "
                         "open gripper")
    ap.add_argument("--release-above-rim", type=float, default=0.0,
                    help="metres above a CONTAINER's rim to stop at, so the object is "
                         "dropped in rather than carried to the floor. 0 reproduces "
                         "droid-sim (descend to the inner floor); a container is the only "
                         "case it applies to, solid tops are unaffected")
    ap.add_argument("--closed-tip-overhang", type=float, default=0.0,
                    help="metres the CLOSED fingertips reach beyond grasp_frame. The "
                         "planner locks the gripper open, so this much of any carried "
                         "clearance is invisible to it. 0 reproduces droid-sim; measure "
                         "yours with gwm_hardware.common.gripper_geometry")
    ap.add_argument("--anchor-descent", action="store_true",
                    help="build the constrained descent's goal from the pose the approach "
                         "actually reached, not the one it was asked for. Required wherever "
                         "the approach lands off its request by more than cuRobo's "
                         "hold_partial_pose tolerance, which is every descent into a "
                         "container on the zhiwei rig")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = tiptop_cfg()
    tensor_args = TensorDeviceType()
    obs = load_h5_observation(args.h5_path)

    # Perceived world: table plane + clusters from the home wrist RGB-D.
    #
    # The DESTINATIONS are these clusters, so a mis-segmented scene does not
    # degrade the placement -- it aims at the wrong thing entirely. On the
    # zhiwei rig with droid-sim's defaults it did exactly that: the perceived
    # table is tilted 2.88 deg, the world-z cut then loses real containers and
    # invents slivers, and a "place into the yellow box" landed on bare table
    # beside it because the box was never a cluster (2026-08-19). These are the
    # same two options the pick proposer and the grasp gate already take, and
    # for the same reason; both default OFF so droid-sim is unchanged.
    robot_spheres = None
    if args.use_robot_arm_filter:
        from gwm_tiptop.robot_fk import fk_model

        robot_spheres = (fk_model(tensor_args).get_state(tensor_args.to_device(obs["q_init"]))
                         .get_link_spheres()[0].cpu().numpy().astype(np.float64))

    sc = scene(args.h5_path, use_plane_normal=args.use_plane_normal,
               robot_spheres=robot_spheres)
    xyz_map, rgb_map = sc["xyz_map"], sc["rgb_map"]
    xyz_flat = xyz_map[np.isfinite(xyz_map).all(axis=2)]
    table_trimesh, surface_z = sc["table_box"], sc["surface_z"]
    # Copies: `scene()` caches these dicts and the grasp gate and debug viewer
    # read the same entry. Dropping the carried object below must not reach
    # them -- a proposer that quietly edits a shared decomposition is exactly
    # the class of bug the cache was keyed carefully to avoid.
    object_trimeshes, object_pcds = dict(sc["meshes"]), dict(sc["pcds"])
    table_cuboid = convert_trimesh_box_to_curobo_cuboid(table_trimesh, name="table")
    save_cluster_viz(obs, object_pcds, args.output_dir / "clusters.png")

    from curobo.geom.types import WorldConfig

    # Cached, not built fresh. `build_curobo_solvers` constructs an IK solver
    # AND a MotionGen and warms both; called directly it allocates ~1 GB of GPU
    # state that nothing ever frees. As a one-shot script that is invisible --
    # the process exits. Run IN-PROCESS by the hardware session, which is how
    # every stage runs since the pipeline was made to build once, it means a
    # fresh solver stack stranded on every PLACE turn, and the session grew
    # until FoundationStereo could not get its 1.72 GB and the turn died with
    # a CUDA OOM (2026-08-19). The pick proposer has used the cache since it
    # was written; this was the one caller left out.
    #
    # And it asks for the SAME key as everything else, not its own (32, 64,
    # no-workspace). A distinct key is a distinct solver stack, so caching two
    # of them saves nothing -- the session held both, 3.9 GB, and still OOMed.
    # Sharing is free here: the next line replaces the world outright, so
    # `include_workspace` cannot reach the plans; `num_spheres` sizes the
    # ATTACHED-object budget, which a place never uses; and only the IK solver
    # depends on `num_particles`, which this function never touches.
    _, motion_gen, _ = default_planning_solvers()

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

    # The object in the gripper is a cluster like any other, and until this it
    # was treated like any other: added to cuRobo's world as a STATIC mesh at
    # the capture pose, and offered as a destination.
    #
    # Both are wrong, and the first is the one that bites. The carried object
    # travels WITH the arm, so a fixed mesh where it currently sits is a phantom
    # the gripper starts inside and then has to plan around for the rest of the
    # motion -- an obstacle that exists only because the robot is holding it.
    # The second is worse in intent than in effect: it invites the proposer to
    # place the object on top of itself.
    #
    # `cluster_objects` already drops the ARM by collision-sphere membership,
    # but it cannot drop the held object that way: being OUTSIDE the robot's own
    # spheres is precisely how `estimate_held_object` recognises it.
    #
    # An image-space gripper mask is not the tool here either -- the place
    # branch captures with `--no-gripper-mask` on purpose, because those are the
    # pixels `estimate_held_object` measures `d_bottom` and `r90` from. Masking
    # them would delete the measurement this whole step depends on.
    #
    # Identified by OVERLAP with the points `estimate_held_object` already
    # isolated, not by distance to the end effector. Distance looked clean here
    # -- carried cluster 0.029 m from the EE against 0.457 m for the nearest
    # table cluster -- but it is a property of THIS capture pose, which sits
    # 0.43 m above the table. droid-sim's capture pose is at z 0.273, and its
    # nearest bin is 0.246 m from the end effector: a 0.20 m radius would have
    # cleared it by 46 mm, and a slightly different layout not at all. Overlap
    # has no such dependence.
    held_tree = KDTree(held["points"])
    carried = {}
    for label, mesh in object_trimeshes.items():
        v = np.asarray(mesh.vertices)
        frac = float((held_tree.query(v)[0] <= HELD_MATCH_RADIUS).mean())
        if frac >= 0.5:
            carried[label] = frac
    for label, frac in carried.items():
        _log.info(f"{label}: this is the CARRIED object ({frac*100:.0f}% of its hull within "
                  f"{HELD_MATCH_RADIUS*1000:.0f} mm of the measured in-hand points) -- "
                  f"not an obstacle, not a destination")
        object_trimeshes.pop(label)
        object_pcds.pop(label, None)

    obstacles = {l: convert_trimesh_to_curobo_mesh(m, l) for l, m in object_trimeshes.items()}
    motion_gen.update_world(WorldConfig(cuboid=[table_cuboid], mesh=list(obstacles.values())))

    # Landing surfaces for every cluster; budget split floor+remainder in
    # cluster order (largest first, deterministic). The hand region (gripper +
    # held object) hovers over the table at the capture pose and may overlap a
    # destination's footprint in xy, so cut it out of the cloud first.
    xyz_scene = xyz_flat[np.linalg.norm(xyz_flat - ee0_pos, axis=1) >= HAND_CROP_RADIUS]
    landings, rejected = {}, []
    for label, mesh in object_trimeshes.items():
        surf = landing_surface(label, np.asarray(mesh.vertices)[:, :2], xyz_scene, surface_z,
                               r_need=held["r90"] if args.max_support_slope > 0 else None)
        if surf is None:
            continue
        # A hollow destination's floor was already validated by the 360-degree
        # enclosure check, and its interior is the support by construction.
        # A SOLID one has to earn it: landing "on top of" something is only a
        # placement if the top can hold the object.
        if (not surf["hollow"] and args.max_support_slope > 0 and surf["slope"] is not None
                and surf["slope"] > args.max_support_slope):
            rejected.append((label, surf["p2v"], surf["slope"]))
            continue
        landings[label] = surf
    for label, p2v, slope in rejected:
        _log.info(f"{label}: NOT a placement destination -- its top rises {p2v*1000:.1f} mm "
                  f"across the {held['r90']*1000:.0f} mm the held object rests on, a slope of "
                  f"{slope:.3f} (limit {args.max_support_slope:.2f}); nothing would stay there")
    if not landings and rejected:
        # Refusing every destination is worse than planning a doubtful one: the
        # caller gets no candidates at all and no way to see why. Keep them and
        # say so, rather than turning a perception limit into a dead end.
        _log.warning("every destination failed the flatness check -- keeping them all. "
                     "Either nothing in this scene can hold the object, or a real "
                     "container was read as solid (its interior unseen from this view).")
        for label, mesh in object_trimeshes.items():
            surf = landing_surface(label, np.asarray(mesh.vertices)[:, :2], xyz_scene, surface_z)
            if surf is not None:
                landings[label] = surf
    if not landings:
        raise SystemExit("no destination clusters with a readable landing surface")
    n_dest = len(landings)
    quotas = [args.k_total // n_dest + (1 if i < args.k_total % n_dest else 0) for i in range(n_dest)]
    # The budget splits over destinations, so FEWER destinations means a bigger
    # share each -- and each share is drawn from a fixed table of deterministic
    # xy offsets. Once the scene narrows to one or two real containers (which
    # is what the carried-object and support-slope filters are for) the share
    # can exceed the table and there is simply nothing more to offer.
    #
    # That used to abort the turn. It should not: a scene with one valid
    # destination is a perfectly good scene to place in -- the choice is just
    # trivial. Cap and say what was lost, rather than refusing to act.
    if max(quotas) > len(XY_OFFSETS):
        capped = [min(q, len(XY_OFFSETS)) for q in quotas]
        _log.warning(
            f"budget {args.k_total} over {n_dest} destination(s) wants up to {max(quotas)} "
            f"candidates each, but only {len(XY_OFFSETS)} deterministic offsets exist -- "
            f"emitting {sum(capped)} instead of {args.k_total}"
            + ("; with one destination the selection is trivial anyway" if n_dest == 1 else ""))
        quotas = capped
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

    def start_state() -> JointState:
        """A FRESH JointState at the capture pose, for every single plan call.

        `JointState.from_position` wraps the tensor it is given rather than
        copying it, and cuRobo writes back into a start state it has been
        handed. Hoisting one `js_init` out of the loop -- which is what this
        did, and what droid-sim gets away with because its scenes never
        triggered the write -- means candidate k+1 is planned from wherever
        candidate k left the buffer.
        
        Measured on this rig, 2026-08-19: at k=0 the buffer held the capture
        pose; from k=1 on it held a pose up to 0.885 rad (51 deg) away, so 15
        of 16 place candidates were trajectories the arm could not begin. The
        controller refused the winner outright ("Trajectory execution failed",
        which names nothing) and the arm never moved.
        """
        return JointState.from_position(
            tensor_args.to_device(np.asarray(obs["q_init"], dtype=np.float64)).float()[None])

    index, n_fail = [], 0
    for (dest, surf), quota in zip(landings.items(), quotas):
        approach_z = surf["rim_z"] + APPROACH_CLEARANCE
        # Clearance is to the LOWEST REAL GEOMETRY, which is not always the
        # object. `place_z` positions the held object's BOTTOM, but the closed
        # fingertips reach `--closed-tip-overhang` past `grasp_frame` while the
        # planner has the gripper locked open and cannot see it. Whichever of
        # the two hangs lower is what has to clear the landing surface.
        #
        # Measured on the first hardware place: overhang 23.7 mm against a
        # held-object drop of 18.4 mm, so the fingers hung 5.3 mm below the
        # tomato. The planner believed it had 28.4 mm of clearance and actually
        # had 4.7 mm, and the descent faulted into the table.
        finger_drop = float(max(0.0, args.closed_tip_overhang - float(d_bottom)))
        if surf["hollow"] and args.release_above_rim > 0.0:
            # Hover over the mouth and let it fall, instead of carrying it down
            # to the floor. For a CONTAINER this is both safer and no less
            # accurate:
            #
            #  * the gripper never enters, so neither the closed fingers' 23.7 mm
            #    of invisible reach nor the mouth's width can bite -- the first
            #    hardware place faulted into the table on exactly that;
            #  * the descent shortens from ~121 mm to a few tens, and the
            #    tilt-induced lateral projection shrinks with it;
            #  * droid-sim already measured the precision. The baseline TiPToP
            #    arm releases ~65 mm above the mouth and its blocks land 5-12 mm
            #    from the bin centre -- BETTER than the GWM arm's carried poses
            #    at 14-35 mm (G-32). Free fall from the rim is not the sloppy
            #    option.
            #
            # Still measured to the lowest real geometry, so the fingers clear
            # the rim rather than the object merely clearing it.
            place_z = surf["rim_z"] + args.release_above_rim + finger_drop
            drop_mode = "over-rim"
        else:
            place_z = surf["land_z"] + LANDING_CLEARANCE + finger_drop
            drop_mode = "to-surface"
        for k, (ox, oy) in enumerate(XY_OFFSETS[:quota]):
            target_xy = surf["target_xy"] + np.array([ox, oy])
            approach = motion_gen.plan_single(
                start_state(), ee_pose_for_held(np.array([*target_xy, approach_z])),
                plan_cfg.clone()
            )
            if not approach.success.item():
                _log.warning(f"{dest}[{k}]: approach failed ({approach.status}); skipping")
                n_fail += 1
                continue
            approach_snap = snapshot(approach)
            js_pre = JointState.from_position(
                tensor_args.to_device(approach_snap["positions"][-1:]).float())

            # Anchor the descent goal to where the approach ACTUALLY ended, not
            # to where it was asked to end. `hold_partial_pose` requires the
            # HELD dimensions -- x, y and orientation -- to be equal between
            # start and goal, and the approach lands within a planner tolerance
            # of its request, not on it. Building the goal from the request
            # therefore asks cuRobo to hold a dimension that already differs,
            # and it refuses: every descent into the one correctly-detected
            # container failed with INVALID_PARTIAL_POSE_COST_METRIC
            # (2026-08-19), preceded by its own "Partial position between start
            # and goal is not equal" warning.
            #
            # Anchoring is also the honest statement of the intent: a
            # constrained descent means "straight down from HERE". The landing
            # xy then moves by the approach tolerance, which is sub-millimetre
            # against the +/-18 mm offset pattern the candidates already span.
            if args.anchor_descent:
                _ach = motion_gen.kinematics.get_state(js_pre.position).ee_pose
                _q = _ach.quaternion[0].detach().cpu().numpy()
                _axis = Rotation.from_quat([_q[1], _q[2], _q[3], _q[0]]).as_matrix()[:, 2]
                _p0 = _ach.position[0].detach().cpu().numpy()
                # Travel far enough ALONG THE GRIPPER'S OWN AXIS to reach the
                # landing height. Not along world z: cuRobo evaluates the held
                # dimensions in the GOAL frame, so a world-vertical descent by a
                # gripper tilted t degrees registers as lateral motion of
                # depth*sin(t) and is rejected past 5 mm. Measured here: 2.7 deg
                # of tilt, harmless over the 45 mm drop onto a solid top
                # (2.1 mm) and fatal over the 121 mm drop into a container
                # (5.7 mm) -- which is why only the container ever failed.
                #
                # Descending along the approach axis is also the better
                # motion: it is the direction the fingers point and the
                # direction the object was carried in, so it cannot scrape a
                # container wall the way a world-vertical drop from a tilted
                # gripper can. It costs depth*sin(t) of lateral drift, ~6 mm
                # here, against the +/-18 mm offsets the candidates span.
                _target_z = float(place_z + d_bottom)
                if abs(_axis[2]) < 1e-6:
                    _log.warning(f"{dest}[{k}]: gripper axis is horizontal; skipping")
                    n_fail += 1
                    continue
                _pos = _p0 + _axis * ((_target_z - _p0[2]) / _axis[2])
                descend_goal = Pose(position=tensor_args.to_device(_pos).float()[None],
                                    quaternion=_ach.quaternion.clone())
            else:
                descend_goal = ee_pose_for_held(np.array([*target_xy, place_z]))

            motion_gen.world_coll_checker.enable_obstacle(enable=False, name=dest)
            try:
                descend = motion_gen.plan_single(js_pre, descend_goal, descend_cfg.clone())
            finally:
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=dest)
            if not descend.success.item():
                # Turn cuRobo's opaque status into the numbers it actually
                # tested. INVALID_PARTIAL_POSE_COST_METRIC means the start,
                # projected into the GOAL frame, violated a held dimension:
                # angular distance > 0.05 rad, or |x| / |y| > 5 mm
                # (motion_gen.update_pose_cost_metric). Knowing WHICH, and by
                # how much, is the difference between a fix and a guess.
                detail = ""
                try:
                    _sp = motion_gen.compute_kinematics(js_pre).ee_pose.clone()
                    _pp = descend_goal.compute_local_pose(_sp)
                    _ang = float(_pp.angular_distance(
                        Pose.from_list([0, 0, 0, 1, 0, 0, 0],
                                       tensor_args=tensor_args)).max())
                    _lin = _pp.position[0].detach().cpu().numpy()
                    detail = (f"  [start in goal frame: angular {_ang:.4f} rad (limit 0.05), "
                              f"dx {_lin[0]*1000:+.1f} dy {_lin[1]*1000:+.1f} dz {_lin[2]*1000:+.1f} mm "
                              f"(x,y limit 5 mm)]")
                except Exception as _e:      # noqa: BLE001 - diagnostics only
                    detail = f"  [could not evaluate the constraint: {_e}]"
                _log.warning(f"{dest}[{k}]: descend failed ({descend.status}); skipping{detail}")
                n_fail += 1
                continue

            descend_snap = snapshot(descend)

            # The arm has to be able to START this plan. A candidate whose
            # first waypoint is not the capture pose is not a plan for this
            # robot at this moment, whatever else is right about it.
            gap = float(np.abs(approach_snap["positions"][0] - obs["q_init"]).max())
            if gap > START_TOL_RAD:
                _log.warning(f"{dest}[{k}]: discarded -- its first waypoint is {gap:.4f} rad "
                             f"({np.degrees(gap):.1f} deg) from the capture pose, so it "
                             f"commands a jump rather than a motion")
                n_fail += 1
                continue
            gap = float(np.abs(descend_snap["positions"][0] - approach_snap["positions"][-1]).max())
            if gap > START_TOL_RAD:
                _log.warning(f"{dest}[{k}]: discarded -- the descent starts {gap:.4f} rad "
                             f"from where the approach ends")
                n_fail += 1
                continue

            i = len(index)
            plan = emit_plan(obs["q_init"], dest, [
                (f"MoveHolding(held, {dest})", approach_snap),
                (f"Place(held, {dest})", descend_snap),
            ], skip_close=args.skip_leading_close)
            dur = sum(len(st["positions"]) * st["dt"] for st in plan["steps"] if st["type"] == "trajectory")
            name = f"plan_{i:02d}_{dest}.json"
            save_tiptop_plan(plan, args.output_dir / name)
            index.append({"file": name, "target": dest, "grasp_confidence": 1.0,
                          "offset": [ox, oy], "traj_s": round(dur, 2),
                          "landing": {"anchored": bool(args.anchor_descent),
                                      "target_xy": np.round(surf["target_xy"], 4).tolist(),
                                      "land_z": round(surf["land_z"], 4),
                                      "finger_drop": round(finger_drop, 4),
                                      "mode": drop_mode,
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
