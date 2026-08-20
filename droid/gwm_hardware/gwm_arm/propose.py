"""Hardware pick proposer: wrist h5 -> 12-16 executable candidates.

The method is `gwm_tiptop`'s, unchanged and imported: geometric table plane,
DBSCAN clusters, whole-scene M2T2 grasps associated by contact, cuTAMP
particle optimisation, cuRobo refinement, all successes collected. What this
driver adds is the two things the real rig needs and droid-sim does not.

**1. The above-table cut follows the table, not world z.** `find_table_plane`
fits a plane; on this rig that plane comes out tilted 2.88 deg, matching the
hand-eye rotational residual documented in `docs/tiptop-modifications.md`. It
is a genuine plane -- 1.87 mm rms perpendicular residual -- but 2.88 deg across
an 0.85 m capture footprint is 48 mm of world-z spread, and the clearance that
is supposed to separate an object from the table is 15 mm. Cut horizontally and
the high end of the tabletop stands 24 mm proud of the cut and becomes a
phantom cluster the size of the table. Measured on the 2026-08-18 blue-cup
capture, that is exactly what happened. `use_plane_normal=True` measures height
perpendicular to the fitted plane instead, which is the same thing on a level
table and the right thing on a tilted one.

**2. The workspace obstacles are in the collision world.** droid-sim plans with
`include_workspace=False` -- there is nothing to hit outside the scene. Here
there is: table edges, keep-outs, the camera mount (`common/rig_workspace.py`,
installed over tiptop's MIT-LIS default). The baseline arm plans with them, so
the A/B arm must too, and a proposal that clips the bench is not a proposal.

    python -m gwm_hardware.gwm_arm.propose \
        --h5-path runs/gwm/scene01/wrist_obs.h5 \
        --output-dir runs/gwm/scene01/proposals --k-total 16
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty

from tiptop.config import tiptop_cfg
from tiptop.motion_planning import build_curobo_solvers
from tiptop.perception.m2t2 import generate_grasps
from tiptop.perception.utils import (
    convert_trimesh_box_to_curobo_cuboid,
    convert_trimesh_to_curobo_mesh,
    depth_to_xyz,
    get_o3d_pcd,
)
from tiptop.planning import build_tamp_config, save_tiptop_plan, serialize_plan

from gwm_tiptop.propose_from_h5 import associate_grasps, load_h5_observation, save_cluster_viz
from gwm_tiptop.proposals import run_proposals
from gwm_tiptop.robot_fk import planning_solvers
from gwm_tiptop.scene_cache import scene

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_arm.propose")


def check_surface_z(surface_z: float, tol: float) -> None:
    """Refuse to plan on a table the fit does not put where the table is.

    The table is bolted down; its height in the base frame is a measured rig
    constant (`rig_workspace.TABLE_TOP_Z`, tape 2026-08-18). So a fitted
    surface far from it means the DEPTH is wrong, and every downstream number
    -- clusters, grasps, collision meshes -- is then wrong in a way nothing
    else notices.

    This is not hypothetical. On 2026-08-19 one capture came back with a table
    fitted 50 mm high (0.1044 vs 0.055) at an unchanged camera pose; the
    cluster cut then sat above two of the three objects and they vanished, and
    the instruction that referred to one of them silently had no referent. The
    two captures either side of it were normal, so it was a transient depth
    fault -- exactly the kind of thing that should stop the run rather than
    quietly change the answer.
    """
    from gwm_hardware.common.rig_workspace import TABLE_TOP_Z

    drift = surface_z - TABLE_TOP_Z
    if abs(drift) <= tol:
        _log.info(f"table fit {surface_z:.4f} m, {drift * 1000:+.1f} mm from the "
                  f"measured {TABLE_TOP_Z:.3f} m -- OK")
        return
    raise SystemExit(
        f"REFUSING TO PLAN: the fitted table surface is {surface_z:.4f} m, "
        f"{drift * 1000:+.1f} mm from this rig's measured table top "
        f"({TABLE_TOP_Z:.3f} m), tolerance {tol * 1000:.0f} mm.\n"
        "The table has not moved, so the DEPTH is wrong, and everything built on "
        "it (clusters, grasps, collision meshes) would be wrong with it.\n"
        "Re-capture. If it repeats: check `rs_preflight` (IR saturation guts the "
        "stereo pattern on a white table), and that FoundationStereo is warm.\n"
        "Override with --max-surface-drift if you genuinely moved the table."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-path", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--k-total", type=int, default=16)
    ap.add_argument("--num-particles", type=int, default=256)
    ap.add_argument("--max-planning-time", type=float, default=60.0)
    ap.add_argument("--horizontal-cut", dest="use_plane_normal", action="store_false",
                    help="cut above the table along world z, the droid-sim behaviour "
                         "(see the tilt note in this module's docstring before using it)")
    ap.add_argument("--height-arm-filter", dest="robot_arm_filter", action="store_false",
                    help="identify the arm by cluster height (droid-sim's rule) instead "
                         "of by the robot's own collision spheres")
    ap.add_argument("--max-surface-drift", type=float, default=0.02,
                    help="metres the fitted table may sit from rig_workspace.TABLE_TOP_Z "
                         "before the run is refused as a depth fault")
    ap.add_argument("--no-workspace", dest="include_workspace", action="store_false",
                    help="plan without the rig's keep-out volumes -- offline analysis only")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = tiptop_cfg()
    _t = {}
    _mark = time.perf_counter

    def _lap(name, t0):
        _t[name] = time.perf_counter() - t0
        return time.perf_counter()

    _t0 = _mark()
    obs = load_h5_observation(args.h5_path)
    if obs["extrinsics_z_correction"] != 0.0:
        _log.warning(
            f"this h5 carries a {obs['extrinsics_z_correction']:+.3f} m extrinsics "
            "correction. That belongs to droid-sim's websocket client; a hardware "
            "capture should read 0.0 (gwm_hardware.gwm_arm.capture writes it)."
        )

    # The arm's real pose, so the clusterer does not have to guess it from
    # height (see cluster_objects' robot_spheres note: the height rule silently
    # deleted a 93 mm upended box on this rig).
    robot_spheres = None
    if args.robot_arm_filter:
        from curobo.types.base import TensorDeviceType

        from gwm_tiptop.robot_fk import fk_model

        _ta = TensorDeviceType()
        robot_spheres = (fk_model(_ta).get_state(_ta.to_device(obs["q_init"]))
                         .get_link_spheres()[0].cpu().numpy().astype(np.float64))

    sc = scene(args.h5_path, use_plane_normal=args.use_plane_normal,
               robot_spheres=robot_spheres)
    xyz_map, rgb_map = sc["xyz_map"], sc["rgb_map"]
    object_trimeshes, object_pcds = sc["meshes"], sc["pcds"]
    _log.info(f"depth: {sc['valid_frac']:.1%} valid")
    _t0 = _lap("load+cloud+perception", _t0)

    finite = np.isfinite(xyz_map).all(axis=2)
    pcd_ds = get_o3d_pcd(xyz_map[finite], rgb_map[finite], cfg.perception.voxel_downsample_size)
    # grasp_threshold / num_runs come from tiptop.yml when present; the .get
    # fallbacks are the client's own defaults, so a config without the keys
    # behaves exactly as before. Measured on the 2026-08-20 peg capture
    # (9 repeats each): 0.035/5 left the yellow peg graspless 3/9 and the
    # green peg 7/9; 0.02/10 cut both to 2/9.
    _m2t2_kwargs = dict(
        grasp_threshold=float(cfg.perception.m2t2.get("grasp_threshold", 0.035)),
        num_runs=int(cfg.perception.m2t2.get("num_runs", 5)),
        apply_bounds=cfg.perception.m2t2.apply_bounds,
    )
    grasps = generate_grasps(
        cfg.perception.m2t2.url,
        scene_xyz=np.asarray(pcd_ds.points),
        scene_rgb=np.asarray(pcd_ds.colors),
        **_m2t2_kwargs,
    )
    _t0 = _lap("m2t2", _t0)

    table_trimesh, surface_z = sc["table_box"], sc["surface_z"]
    check_surface_z(surface_z, args.max_surface_drift)
    table_cuboid = convert_trimesh_box_to_curobo_cuboid(table_trimesh, name="table")
    config = build_tamp_config(
        num_particles=args.num_particles,
        max_planning_time=args.max_planning_time,
        opt_steps=500,
        robot_type=cfg.robot.type,
        time_dilation_factor=cfg.robot.time_dilation_factor,
        near_placement=False,
    )
    save_cluster_viz(obs, object_pcds, args.output_dir / "clusters.png")

    object_meshes = {l: convert_trimesh_to_curobo_mesh(m, l) for l, m in object_trimeshes.items()}
    filtered_grasps = associate_grasps(grasps, object_pcds, object_meshes,
                                       cfg.perception.contact_threshold_m)

    # M2T2's scene sampling is stochastic, and small flat objects sit right on
    # its edge: the same cloud gives a 69x39 mm peg 0..112 contact-associated
    # grasps across repeated calls. A cluster with zero grasps this draw is
    # therefore not proven ungraspable -- re-sample and MERGE before giving up
    # on it. Each retry costs one M2T2 round-trip (~2.6 s at num_runs 10) and
    # runs only while something is still graspless.
    for _retry in range(int(cfg.perception.m2t2.get("graspless_retries", 0))):
        graspless = sorted(l for l in object_pcds if len(filtered_grasps[l]["poses"]) == 0)
        if not graspless:
            break
        _log.warning(f"{graspless} got no grasps this M2T2 draw -- re-sampling "
                     f"(retry {_retry + 1})")
        more = generate_grasps(
            cfg.perception.m2t2.url,
            scene_xyz=np.asarray(pcd_ds.points),
            scene_rgb=np.asarray(pcd_ds.colors),
            **_m2t2_kwargs,
        )
        grasps = {**grasps, **{f"retry{_retry}_{k}": v for k, v in more.items()}}
        filtered_grasps = associate_grasps(grasps, object_pcds, object_meshes,
                                           cfg.perception.contact_threshold_m)

    movables = [m for l, m in object_meshes.items() if len(filtered_grasps[l]["poses"]) > 0]
    dropped = sorted(set(object_meshes) - {m.name for m in movables})
    if dropped:
        _log.warning(f"Dropping graspless clusters: {dropped}")

    env = TAMPEnvironment(
        name="gwm_arm_proposals",
        movables=movables,
        statics=[table_cuboid],
        type_to_objects={"Movable": movables, "Surface": [table_cuboid]},
        goal_state=frozenset({HandEmpty.ground()}),
    )
    _t0 = _lap("associate", _t0)
    ik_solver, motion_gen, _ = planning_solvers(
        config.num_particles, config.coll_n_spheres,
        include_workspace=args.include_workspace,
    )
    _t0 = _lap("build_solvers", _t0)
    proposals = run_proposals(
        env, config, obs["q_init"], ik_solver, filtered_grasps, motion_gen,
        all_surfaces=[table_cuboid], k_total=args.k_total,
    )

    _t0 = _lap("run_proposals", _t0)
    index = []
    for i, prop in enumerate(proposals):
        serialized = serialize_plan(prop["steps"], obs["q_init"])
        name = f"plan_{i:02d}_{prop['target']}.json"
        save_tiptop_plan(serialized, args.output_dir / name)
        index.append({"file": name, "target": prop["target"],
                      "grasp_confidence": prop["grasp_confidence"]})

    meta = table_trimesh.metadata or {}
    with open(args.output_dir / "proposals_index.json", "w") as f:
        json.dump({
            "h5": str(args.h5_path),
            "num_proposals": len(index),
            "proposals": index,
            "perception": {
                "clusters": sorted(object_meshes),
                "graspless_clusters": dropped,
                "surface_z": float(surface_z),
                "table_plane": meta.get("plane"),
                "table_tilt_deg": meta.get("tilt_deg"),
                "cut": "plane-normal" if args.use_plane_normal else "world-z",
            },
            "planning": {"include_workspace": bool(args.include_workspace),
                         "k_total": args.k_total,
                         "num_particles": args.num_particles},
        }, f, indent=2)
    per_object: dict[str, int] = {}
    for e in index:
        per_object[e["target"]] = per_object.get(e["target"], 0) + 1
    _log.info(f"Wrote {len(index)} proposals to {args.output_dir}  {per_object}")
    _log.info("timing: " + "  ".join(f"{k} {v:.1f}s" for k, v in _t.items())
              + f"  TOTAL {sum(_t.values()):.1f}s")
    if len(index) < args.k_total:
        _log.warning(
            f"{len(index)} candidates, asked for {args.k_total} -- some clusters "
            "produced no refinable grasp. Check clusters.png against the scene "
            "before reading anything into the selection."
        )


if __name__ == "__main__":
    main()
