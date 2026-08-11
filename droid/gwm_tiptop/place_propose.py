"""Place-only proposals for scene 6 variant 1: held block -> one of two bins.

The block is WELDED to the gripper (weld_held_block.py) and never released, so
this proposer needs no grasps at all -- no M2T2, no cuTAMP, no particle
machinery. Each candidate is a deterministic two-segment cuRobo plan:

    [gripper close, MoveHolding(home -> pre-place above bin), Place(descend)]

- The close comes FIRST because the gripper transports the block with fingers
  shut: the OPEN gripper is 0.144 m wide at the knuckles and cannot enter the
  0.105 m bin mouth; closed on the 30 mm block it is ~0.089 m. The executor
  handles a leading gripper step by holding the observed joint pose
  (tiptop_websocket._step_plan), and closing on the welded block just adds a
  symmetric clamp that agrees with the weld.
- The descent reuses cuTAMP's constrained-plan trick verbatim: a PoseCostMetric
  holding x/y/rotation in the goal frame with z free is a straight vertical
  drop. Both bin obstacles are disabled during the descent (the gripper must
  enter the mouth; clearances were sized offline: 7.9 mm a side), then
  re-enabled. Approach planning keeps the full perceived world.
- No GoToInitial and no release: the episode ends with the block held inside
  the bin, which is exactly what the judge scores (place_eval.py) and what the
  GWM RAT window needs -- the last sampled frame is the arm over the chosen
  bin, the most discriminative pose, instead of a shared home pose. Plans land
  ~7.5 s, under SCHEDULE[-1]*3.0 = 8.85 s, so sample_rat_times' shrink-to-fit
  branch fires and the 6 frames tile the whole plan.

Obstacles come from the same geometric perception as the pick proposer
(find_table_plane + cluster_objects on the home wrist RGB-D); clusters are
labeled by matching centroids against the capture's objects.json so the two
bin clusters can be identified and toggled. The held block never appears as a
cluster -- perception_geometric's floating-cluster filter discards anything
hovering above the table, which is where the welded block lives.

Candidate targets are a deterministic xy-offset pattern per bin (no RNG, so
re-runs are byte-stable). Diversity across candidates is placement pose, the
only thing the GWM scorer can see for a place task anyway.

Run inside the droid/tiptop pixi shell (no servers needed):

    python -m gwm_tiptop.place_propose \
        --h5-path .../captures/scene6_1/wrist_obs.h5 \
        --objects-json .../captures/scene6_1/objects.json \
        --output-dir .../gwm_integrate_doc/proposals/scene6_place
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

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

BINS = ("red_bin", "green_bin")
# Deterministic per-bin xy offsets (m). Judge tol is 0.05, bin half-width
# 0.0575, block half-edge 0.015: |offset| <= 0.018 keeps every candidate both
# physically inside the bin and inside the judge band with margin.
XY_OFFSETS = [
    (0.0, 0.0), (0.018, 0.0), (-0.018, 0.0), (0.0, 0.018),
    (0.0, -0.018), (0.013, 0.013), (-0.013, 0.013), (0.013, -0.013), (-0.013, -0.013),
]
Z_PREPLACE = 0.190  # block mesh centre; bottom clears the 0.118 rim by 57 mm
Z_PLACE = 0.075     # block mesh centre in-bin; z_rel to bin centre ~ -0.005, floor clearance ~14 mm
BLOCK_MESH_OFFSET_Z = -0.039214913  # basic_block prim origin -> mesh centre, at stock scale


def match_clusters(object_trimeshes: dict, objects_json: dict) -> dict:
    """cluster label -> unique obstacle name, by nearest settled xy centroid (<= 8 cm).

    Ambiguity is fatal only for the BINS (they are toggled by name during the
    descent). Anything else is obstacle-only, so a scene object that perception
    splits into several clusters (the bowl rim splits into two arcs in the
    wrist view) just gets numbered suffixes.
    """
    known = {n: np.asarray(d["pos_w"][:2]) for n, d in objects_json.items() if n != "held_block"}
    out, seen = {}, {}
    for label, mesh in object_trimeshes.items():
        c = mesh.bounding_box.centroid[:2]
        name, dist = min(((n, np.linalg.norm(c - p)) for n, p in known.items()), key=lambda t: t[1])
        if dist > 0.08:
            name = label
        n = seen[name] = seen.get(name, 0) + 1
        if n > 1:
            if name in BINS:
                raise RuntimeError(f"bin {name} matched {n} clusters -- cannot toggle it by name")
            name = f"{name}_{n}"
        out[label] = name
        _log.info(f"cluster {label} -> {name} (d={dist * 100:.1f} cm)")
    return out


def emit_plan(q_init: np.ndarray, bin_name: str, results: list) -> dict:
    steps = [{"type": "gripper", "label": "Close(held_block)", "action": "close"}]
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
    ap.add_argument("--objects-json", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k-per-bin", type=int, default=8)
    ap.add_argument("--block-edge", type=float, default=0.030)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.k_per_bin > len(XY_OFFSETS):
        raise SystemExit(f"--k-per-bin max {len(XY_OFFSETS)}")

    cfg = tiptop_cfg()
    tensor_args = TensorDeviceType()
    obs = load_h5_observation(args.h5_path)
    objects = json.loads(args.objects_json.read_text())

    # Perceived world: table plane + clusters from the home wrist RGB-D.
    depth = obs["depth"].copy()
    depth[~np.isfinite(depth)] = np.nan
    depth[(depth <= 0.05) | (depth > 4.0)] = np.nan
    xyz_map = depth_to_xyz(depth, obs["K"])
    xyz_map = xyz_map @ obs["world_from_cam"][:3, :3].T + obs["world_from_cam"][:3, 3]
    rgb_map = obs["rgb"].astype(np.float32) / 255.0

    table_trimesh, surface_z = find_table_plane(xyz_map, rgb_map)
    table_cuboid = convert_trimesh_box_to_curobo_cuboid(table_trimesh, name="table")
    object_trimeshes, object_pcds = cluster_objects(xyz_map, rgb_map, table_trimesh, surface_z + 0.015)
    save_cluster_viz(obs, object_pcds, args.output_dir / "clusters.png")

    label_to_name = match_clusters(object_trimeshes, objects)
    missing = [b for b in BINS if b not in label_to_name.values()]
    if missing:
        raise RuntimeError(f"bins not found among clusters: {missing}")
    obstacles = {label_to_name[l]: convert_trimesh_to_curobo_mesh(m, label_to_name[l])
                 for l, m in object_trimeshes.items()}

    from curobo.geom.types import WorldConfig

    _, motion_gen, _ = build_curobo_solvers(num_particles=32, num_spheres=64, include_workspace=False)
    motion_gen.update_world(WorldConfig(cuboid=[table_cuboid], mesh=list(obstacles.values())))

    # Grasp geometry: block mesh centre relative to the EE at the capture pose.
    q_init = tensor_args.to_device(obs["q_init"]).float()
    ee0 = motion_gen.kinematics.get_state(q_init[None]).ee_pose
    ee0_pos = ee0.position[0].cpu().numpy()
    ee0_quat = ee0.quaternion  # (1, 4) kept on-device; all targets reuse the home orientation
    blk = objects["held_block"]
    s = args.block_edge / 0.047
    # Settled block attitude is identity to within 0.8 deg (PD sag), so the
    # prim-origin -> mesh-centre offset is applied along world z unrotated
    # (the lateral leakage at that tilt is ~0.3 mm).
    block0 = np.asarray(blk["pos_w"]) + np.array([0.0, 0.0, BLOCK_MESH_OFFSET_Z * s])
    d = ee0_pos - block0  # EE offset from block mesh centre, constant under the weld
    _log.info(f"EE at capture: {np.round(ee0_pos, 4).tolist()}, block mesh {np.round(block0, 4).tolist()}, d={np.round(d, 4).tolist()}")

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

    def ee_pose_for_block(target_xyz: np.ndarray) -> Pose:
        return Pose(position=tensor_args.to_device(target_xyz + d).float()[None], quaternion=ee0_quat)

    js_init = JointState.from_position(q_init[None])
    index, n_fail = [], 0
    for bin_name in BINS:
        bxy = np.asarray(objects[bin_name]["pos_w"][:2])
        for k, (ox, oy) in enumerate(XY_OFFSETS[: args.k_per_bin]):
            target_xy = bxy + np.array([ox, oy])
            approach = motion_gen.plan_single(
                js_init, ee_pose_for_block(np.array([*target_xy, Z_PREPLACE])), plan_cfg.clone()
            )
            if not approach.success.item():
                _log.warning(f"{bin_name}[{k}]: approach failed ({approach.status}); skipping")
                n_fail += 1
                continue
            js_pre = JointState.from_position(approach.get_interpolated_plan().position[-1:])
            for b in BINS:
                motion_gen.world_coll_checker.enable_obstacle(enable=False, name=b)
            try:
                descend = motion_gen.plan_single(
                    js_pre, ee_pose_for_block(np.array([*target_xy, Z_PLACE])), descend_cfg.clone()
                )
            finally:
                for b in BINS:
                    motion_gen.world_coll_checker.enable_obstacle(enable=True, name=b)
            if not descend.success.item():
                _log.warning(f"{bin_name}[{k}]: descend failed ({descend.status}); skipping")
                n_fail += 1
                continue

            i = len(index)
            plan = emit_plan(obs["q_init"], bin_name, [
                (f"MoveHolding(held_block, {bin_name})", approach),
                (f"Place(held_block, {bin_name})", descend),
            ])
            dur = sum(len(st["positions"]) * st["dt"] for st in plan["steps"] if st["type"] == "trajectory")
            name = f"plan_{i:02d}_{bin_name}.json"
            save_tiptop_plan(plan, args.output_dir / name)
            index.append({"file": name, "target": bin_name, "grasp_confidence": 1.0,
                          "offset": [ox, oy], "traj_s": round(dur, 2)})
            _log.info(f"{name}: {dur:.1f}s of trajectory (+1.33s close)")

    with open(args.output_dir / "proposals_index.json", "w") as f:
        json.dump({"h5": str(args.h5_path), "num_proposals": len(index), "proposals": index}, f, indent=2)
    _log.info(f"Wrote {len(index)} proposals ({n_fail} failures) to {args.output_dir}")
    if not index:
        raise SystemExit("no proposals produced")


if __name__ == "__main__":
    main()
