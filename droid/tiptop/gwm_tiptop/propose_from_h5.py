"""GI-3 driver: H5 observation -> semantic-free pick proposals.

Mirrors the tiptop-h5 offline path but with the Gemini/SAM2 stage replaced by
geometric scene decomposition (perception_geometric) and single-plan cuTAMP
replaced by run_proposals. Requires the M2T2 server. Run inside `pixi shell`:

    python -m gwm_tiptop.propose_from_h5 \
        --h5-path /root/code/gwm/gwm-wiser/droid/droid-sim-evals/tiptop_assets/smoke_test.h5 \
        --output-dir ../gwm_integrate_doc/proposals/scene1
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from curobo.types.base import TensorDeviceType
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from tiptop.config import tiptop_cfg
from tiptop.motion_planning import build_curobo_solvers
from tiptop.perception.m2t2 import generate_grasps, m2t2_to_tiptop_transform
from tiptop.perception.utils import (
    convert_trimesh_box_to_curobo_cuboid,
    convert_trimesh_to_curobo_mesh,
    depth_to_xyz,
    get_o3d_pcd,
)
from tiptop.planning import build_tamp_config, save_tiptop_plan, serialize_plan

from gwm_tiptop.perception_geometric import cluster_objects, find_table_plane
from gwm_tiptop.proposals import run_proposals

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_tiptop.propose")


def load_h5_observation(h5_path: Path) -> dict:
    with h5py.File(h5_path) as f:
        rgb = np.asarray(f["rgb"])
        depth = np.asarray(f["depth"])
        if depth.ndim == 3:
            depth = depth[..., 0]
        K = np.asarray(f["intrinsic_matrix"])
        pos = np.asarray(f["pos_w"])
        w, x, y, z = np.asarray(f["quat_w_ros"])
        world_from_cam = np.eye(4)
        world_from_cam[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        world_from_cam[:3, 3] = pos
        # Match the websocket client's grasp-frame correction
        world_from_cam[:3, 3] -= np.array([0.0, 0.0, 0.015])
        q_init = np.asarray(f["q_init"])
    return {"rgb": rgb, "depth": depth, "K": K, "world_from_cam": world_from_cam, "q_init": q_init}


def associate_grasps(grasps: dict, object_pcds: dict, object_meshes_curobo: dict, contact_threshold_m: float) -> dict:
    """Assign whole-scene M2T2 grasps to clusters by contact-point proximity (lean copy of process_scene_geometry)."""
    tensor_args = TensorDeviceType()
    labels = list(object_pcds.keys())
    all_points, point_label = [], []
    for label, pcd in object_pcds.items():
        pts = np.asarray(pcd.points)
        all_points.append(pts)
        point_label.extend([label] * len(pts))
    kdtree = KDTree(np.vstack(all_points))
    point_label = np.array(point_label)

    poses_l, confs_l, contacts_l, labels_l = [], [], [], []
    for _, gd in grasps.items():
        poses, confs, contacts = gd["poses"], gd["confidences"], gd["contacts"]
        if len(contacts) == 0:
            continue
        dists, idxs = kdtree.query(contacts)
        near = dists < contact_threshold_m
        poses_l.append(poses[near])
        confs_l.append(confs[near])
        contacts_l.append(contacts[near])
        labels_l.append(point_label[idxs[near]])

    filtered: dict = {}
    if poses_l:
        poses = np.concatenate(poses_l)
        confs = np.concatenate(confs_l)
        contacts = np.concatenate(contacts_l)
        glabels = np.concatenate(labels_l)
    else:
        poses = np.zeros((0, 4, 4))
        confs = np.zeros(0)
        contacts = np.zeros((0, 3))
        glabels = np.zeros(0, dtype=str)

    for label in labels:
        m = glabels == label
        entry = {"poses": poses[m], "confidences": confs[m], "contacts": contacts[m]}
        curobo_pose = np.array(object_meshes_curobo[label].pose)
        assert np.allclose(curobo_pose[3:], np.array([1.0, 0.0, 0.0, 0.0]))
        world_from_obj = np.eye(4)
        world_from_obj[:3, 3] = curobo_pose[:3]
        world_from_grasp = entry["poses"] @ m2t2_to_tiptop_transform()
        obj_from_grasp = np.linalg.inv(world_from_obj) @ world_from_grasp
        entry["grasps_obj"] = tensor_args.to_device(obj_from_grasp)
        entry["confidences_pt"] = tensor_args.to_device(entry["confidences"])
        filtered[label] = entry
        _log.info(f"{label}: {int(m.sum())} grasps associated")
    return filtered


def save_cluster_viz(obs: dict, object_pcds: dict, out_path: Path) -> None:
    """Project cluster points into the observation image for a quick visual audit."""
    import cv2

    K, w2c = obs["K"], np.linalg.inv(obs["world_from_cam"])
    img = obs["rgb"].copy()
    palette = [(255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 255, 0),
               (255, 0, 255), (0, 255, 255), (255, 160, 0), (160, 0, 255)]
    for i, (label, pcd) in enumerate(object_pcds.items()):
        pts = np.asarray(pcd.points)
        pc = (w2c[:3, :3] @ pts.T + w2c[:3, 3:4])
        front = pc[2] > 0.05
        uv = K @ pc[:, front]
        uv = (uv[:2] / uv[2]).astype(int)
        ok = (uv[0] >= 0) & (uv[0] < img.shape[1]) & (uv[1] >= 0) & (uv[1] < img.shape[0])
        img[uv[1][ok], uv[0][ok]] = palette[i % len(palette)]
        if ok.any():
            cx, cy = int(uv[0][ok].mean()), int(uv[1][ok].mean())
            cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    _log.info(f"Cluster viz saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-path", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k-total", type=int, default=16)
    ap.add_argument("--num-particles", type=int, default=256)
    ap.add_argument("--max-planning-time", type=float, default=60.0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = tiptop_cfg()
    obs = load_h5_observation(args.h5_path)

    # Depth -> world cloud (GT depth in sim; no FoundationStereo). Sim depth
    # carries inf where rays miss geometry — mark those invalid up front.
    depth = obs["depth"].copy()
    depth[~np.isfinite(depth)] = np.nan
    depth[(depth <= 0.05) | (depth > 4.0)] = np.nan
    xyz_map = depth_to_xyz(depth, obs["K"])
    xyz_map = xyz_map @ obs["world_from_cam"][:3, :3].T + obs["world_from_cam"][:3, 3]
    rgb_map = obs["rgb"].astype(np.float32) / 255.0

    # M2T2 whole-scene grasps (mask-free by nature)
    finite = np.isfinite(xyz_map).all(axis=2)
    pcd_ds = get_o3d_pcd(xyz_map[finite], rgb_map[finite], cfg.perception.voxel_downsample_size)
    grasps = generate_grasps(
        cfg.perception.m2t2.url,
        scene_xyz=np.asarray(pcd_ds.points),
        scene_rgb=np.asarray(pcd_ds.colors),
        apply_bounds=cfg.perception.m2t2.apply_bounds,
    )

    # Geometric scene decomposition (no Gemini, no SAM2)
    table_trimesh, surface_z = find_table_plane(xyz_map, rgb_map)
    table_cuboid = convert_trimesh_box_to_curobo_cuboid(table_trimesh, name="table")
    config = build_tamp_config(
        num_particles=args.num_particles,
        max_planning_time=args.max_planning_time,
        opt_steps=500,
        robot_type=cfg.robot.type,
        time_dilation_factor=cfg.robot.time_dilation_factor,
        near_placement=False,
    )
    cluster_z = surface_z + 0.015
    object_trimeshes, object_pcds = cluster_objects(xyz_map, rgb_map, table_trimesh, cluster_z)
    save_cluster_viz(obs, object_pcds, args.output_dir / "clusters.png")

    object_meshes = {l: convert_trimesh_to_curobo_mesh(m, l) for l, m in object_trimeshes.items()}
    filtered_grasps = associate_grasps(grasps, object_pcds, object_meshes, cfg.perception.contact_threshold_m)
    # Objects with no reachable grasps cannot form Pick goals
    movables = [m for l, m in object_meshes.items() if len(filtered_grasps[l]["poses"]) > 0]
    dropped = sorted(set(object_meshes) - {m.name for m in movables})
    if dropped:
        _log.warning(f"Dropping graspless clusters: {dropped}")

    env = TAMPEnvironment(
        name="gwm_tiptop_proposals",
        movables=movables,
        statics=[table_cuboid],
        type_to_objects={"Movable": movables, "Surface": [table_cuboid]},
        goal_state=frozenset({HandEmpty.ground()}),
    )

    ik_solver, motion_gen, _ = build_curobo_solvers(
        config.num_particles, config.coll_n_spheres, include_workspace=False
    )
    proposals = run_proposals(
        env, config, obs["q_init"], ik_solver, filtered_grasps, motion_gen,
        all_surfaces=[table_cuboid], k_total=args.k_total,
    )

    index = []
    for i, prop in enumerate(proposals):
        serialized = serialize_plan(prop["steps"], obs["q_init"])
        name = f"plan_{i:02d}_{prop['target']}.json"
        save_tiptop_plan(serialized, args.output_dir / name)
        index.append({"file": name, "target": prop["target"], "grasp_confidence": prop["grasp_confidence"]})
    with open(args.output_dir / "proposals_index.json", "w") as f:
        json.dump({"h5": str(args.h5_path), "num_proposals": len(index), "proposals": index}, f, indent=2)
    _log.info(f"Wrote {len(index)} proposals to {args.output_dir}")


if __name__ == "__main__":
    main()
