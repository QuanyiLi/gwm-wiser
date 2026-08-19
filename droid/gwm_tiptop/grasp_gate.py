"""Closing-line grasp gate: veto physically fragile grasp candidates.

Motivation (G-24 plan_14 1/10, G-26 nearbowl plan_12 0/5): GWM retrieval picks
the right OBJECT but cannot see grasp robustness -- the RAT frames are
robot-only, so an edge/corner grasp that shoves the target scores as well as a
centred one. This gate reads the failure mode directly off geometry the system
already has: at the closing configuration, does the target's perceived point
mass sit squarely between the finger pads?

PERCEPTION-ONLY CONTRACT: inputs are the wrist h5 (same capture the proposer
used), the saved plans, and the robot's own model. No GT object info.

Per pick plan:
  q_close = last waypoint of the trajectory step preceding the gripper close.
  FK gives the EE pose; the finger-pad geometry (closing axis, pad window,
  open half-gap) is SELF-CALIBRATED once from the robot's collision spheres
  in the EE frame -- no hardcoded gripper numbers.
  The target cluster's raw points are expressed in the EE frame and cut to
  the capture box (the volume the pads sweep while closing). Metrics:
    n_slab      -- points inside the capture box (is there object to grab?)
    thickness   -- p95-p5 extent along the closing axis (how much material
                   the pads actually squeeze; a corner clip is thin)
    center_off  -- |mean| along the closing axis (asymmetry: one pad hits
                   first and shoves the object)
    ortho_off   -- |mean| across the pad width (edge clip sideways)
Pass requires all four within thresholds. --apply TAG re-picks the winner as
the most confident (M2T2) PASSING plan of the object score_client selected,
and rewrites winner_TAG.json (original kept implicitly in the ranking; falls
back with a warning if nothing passes). Since score_client already ranks
within an object by M2T2 confidence (G-28), the gate now only changes the
winner when the most confident candidate is also a fragile one.

Run inside the droid/tiptop pixi env from the gwm-wiser repo root:

    python -m gwm_tiptop.grasp_gate \
        --proposals-dir droid/gwm_integrate_doc/proposals/scene6_rev2 \
        --h5-path droid/droid-sim-evals-ours/scenes/captures/scene6_0/wrist_obs.h5 \
        --apply refer6_nearbowl
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.transform import Rotation

from curobo.types.base import TensorDeviceType

from tiptop.perception.utils import depth_to_xyz

from gwm_tiptop.perception_geometric import cluster_objects, find_table_plane
from gwm_tiptop.robot_fk import fk_model
from gwm_tiptop.propose_from_h5 import load_h5_observation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_tiptop.grasp_gate")

# Robot-side policy thresholds (scene-agnostic). Calibrated on the scene6_rev2
# pool where the fragile nearbowl winner (plan_12, 0/5) must fail and its
# robust siblings must pass; see G-26 follow-up in the ledger.
MIN_SLAB_PTS = 150      # raw cloud points inside the capture box
MIN_THICKNESS = 0.015   # m of material along the closing axis
MAX_CENTER_OFF = 0.015  # m slab-centroid asymmetry between the pads
MAX_ORTHO_OFF = 0.012   # m slab-centroid offset across the pad width
# Capture-box slop added to the self-calibrated pad window.
BOX_PAD = 0.005
# Hand-region crop radius around the CAPTURE-pose EE (the gripper hovers in
# frame at capture; its points must not masquerade as scene geometry).
HAND_CROP_RADIUS = 0.20


def ee_frame(state):
    t = state.ee_pose.position[0].cpu().numpy().astype(np.float64)
    w, x, y, z = state.ee_pose.quaternion[0].cpu().numpy().astype(np.float64)
    R = Rotation.from_quat([x, y, z, w]).as_matrix()  # EE -> world
    return R, t


def calibrate_pads(kin, q, tensor_args) -> dict:
    """Finger-pad geometry in the EE frame from the robot's own spheres."""
    state = kin.get_state(tensor_args.to_device(q).float()[None])
    R, t = ee_frame(state)
    sph = state.get_link_spheres()[0].cpu().numpy().astype(np.float64)
    valid = sph[:, 3] > 0.0  # cuRobo pads the buffer with negative-radius spheres
    c_ee = (sph[valid, :3] - t) @ R
    r = sph[valid, 3]
    # Fingertip pads: small spheres near the TCP plane. In this robot model the
    # EE z axis points from the arm PAST the fingertips (arm spheres at
    # z about -0.6), and z=0 IS the grasp point at the fingertip plane, so the
    # pads live in a thin band just below it.
    tips = (c_ee[:, 2] > -0.06) & (r < 0.03)
    if tips.sum() < 4:
        raise RuntimeError(f"Only {int(tips.sum())} fingertip spheres found -- calibration failed")
    xy = c_ee[tips, :2]
    xy_c = xy - xy.mean(axis=0)
    # Closing axis = dominant horizontal spread direction of the tip spheres.
    _, _, vt = np.linalg.svd(xy_c, full_matrices=False)
    v = np.array([vt[0, 0], vt[0, 1], 0.0])
    v /= np.linalg.norm(v)
    u = np.array([-v[1], v[0], 0.0])
    proj = xy_c @ v[:2]
    left, right = proj[proj < 0], proj[proj >= 0]
    if not len(left) or not len(right):
        raise RuntimeError("Fingertip spheres did not split into two pads")
    open_half = (np.abs(left.mean()) + np.abs(right.mean())) / 2
    if open_half < 0.02:
        raise RuntimeError(f"Pad separation {open_half:.3f} m too small -- gripper not open?")
    pad_z = float(c_ee[tips, 2].mean())
    pad_len = float(np.percentile(c_ee[tips, 2], 95) - np.percentile(c_ee[tips, 2], 5)) + 2 * float(r[tips].mean())
    pad_halfw = float(np.abs(xy_c @ u[:2]).max() + r[tips].mean())
    cal = {"v": v, "u": u, "open_half": float(open_half), "pad_z": pad_z,
           "pad_len": max(pad_len, 0.02), "pad_halfw": pad_halfw}
    _log.info(
        f"pad calibration: closing axis {np.round(v, 3).tolist()} (EE frame), open half-gap "
        f"{open_half:.4f}, pad_z {pad_z:.4f}, pad_len {cal['pad_len']:.4f}, pad_halfw {pad_halfw:.4f}"
    )
    return cal


def q_at_close(plan: dict):
    prev_traj = None
    for step in plan["steps"]:
        if step["type"] == "trajectory":
            prev_traj = step
        elif step["type"] == "gripper" and step.get("action") == "close":
            if prev_traj is None:
                return None  # place-style plan: close before any motion
            return np.asarray(prev_traj["positions"])[-1]
    return None


def slab_metrics(pts_ee: np.ndarray, cal: dict) -> dict:
    c = pts_ee @ cal["v"]
    o = pts_ee @ cal["u"]
    z = pts_ee[:, 2]
    m = (np.abs(c) <= cal["open_half"] + BOX_PAD) \
        & (np.abs(o) <= cal["pad_halfw"] + BOX_PAD) \
        & (np.abs(z - cal["pad_z"]) <= cal["pad_len"] / 2 + BOX_PAD)
    n = int(m.sum())
    if n == 0:
        return {"n_slab": 0, "thickness": 0.0, "center_off": 1.0, "ortho_off": 1.0}
    return {
        "n_slab": n,
        "thickness": float(np.percentile(c[m], 95) - np.percentile(c[m], 5)),
        "center_off": float(abs(c[m].mean())),
        "ortho_off": float(abs(o[m].mean())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals-dir", required=True, type=Path)
    ap.add_argument("--h5-path", required=True, type=Path)
    ap.add_argument("--apply", default=None, metavar="TAG",
                    help="rewrite winner_TAG.json to the best-scoring PASSING plan")
    # The four thresholds are class-D magic numbers (magic_numbers.md #8):
    # MIN_SLAB_PTS is an absolute point count, so it tracks the rig's cloud
    # density, and MIN_THICKNESS encodes a solid-body bias. Exposed so a new
    # rig can be re-calibrated without editing a module droid-sim shares; the
    # defaults are the scene6_rev2 values every sim result was produced under.
    ap.add_argument("--min-slab-pts", type=int, default=MIN_SLAB_PTS)
    ap.add_argument("--min-thickness", type=float, default=MIN_THICKNESS)
    ap.add_argument("--max-center-off", type=float, default=MAX_CENTER_OFF)
    ap.add_argument("--max-ortho-off", type=float, default=MAX_ORTHO_OFF)
    ap.add_argument("--use-robot-arm-filter", action="store_true",
                    help="identify the arm by the robot's own collision spheres rather "
                         "than by cluster height. MUST match what the proposer used, or "
                         "the gate judges plans against a different scene")
    ap.add_argument("--use-plane-normal", action="store_true",
                    help="measure height above the FITTED table plane rather than world z "
                         "when re-clustering. Required on a rig whose perceived table is "
                         "tilted (zhiwei: 2.88 deg); a no-op on a level one")
    args = ap.parse_args()

    obs = load_h5_observation(args.h5_path)
    depth = obs["depth"].copy()
    depth[~np.isfinite(depth)] = np.nan
    depth[(depth <= 0.05) | (depth > 4.0)] = np.nan
    xyz_map = depth_to_xyz(depth, obs["K"])
    xyz_map = xyz_map @ obs["world_from_cam"][:3, :3].T + obs["world_from_cam"][:3, 3]
    rgb_map = obs["rgb"].astype(np.float32) / 255.0
    xyz_flat = xyz_map[np.isfinite(xyz_map).all(axis=2)]

    # FK only -- this gate never plans. See gwm_tiptop.robot_fk.
    tensor_args = TensorDeviceType()
    kin = fk_model(tensor_args)

    # The gate re-derives the clusters itself, so it MUST decompose the scene the
    # same way the proposer did -- otherwise it judges plans against a scene
    # that does not contain their target. It happened: with the proposer using
    # the robot-sphere arm filter and the gate still using the height rule, an
    # upended box survived proposal and vanished at gate time, and all five of
    # its candidates came back "no raw points", so nothing passed and the gate
    # silently fell through to the ungated winner.
    table_trimesh, surface_z = find_table_plane(xyz_map, rgb_map)
    robot_spheres = None
    if args.use_robot_arm_filter:
        robot_spheres = (kin.get_state(tensor_args.to_device(obs["q_init"]))
                         .get_link_spheres()[0].cpu().numpy().astype(np.float64))
    object_trimeshes, _ = cluster_objects(xyz_map, rgb_map, table_trimesh, surface_z + 0.015,
                                          use_plane_normal=args.use_plane_normal,
                                          robot_spheres=robot_spheres)

    # Hand-region crop (capture-pose gripper hovers in frame).
    ee0 = ee_frame(kin.get_state(tensor_args.to_device(obs["q_init"]).float()[None]))[1]
    xyz_scene = xyz_flat[np.linalg.norm(xyz_flat - ee0, axis=1) >= HAND_CROP_RADIUS]

    # Raw in-footprint points per cluster (the hulls are base-augmented, so the
    # true observed surfaces come from re-cutting the raw cloud).
    cluster_pts = {}
    for label, mesh in object_trimeshes.items():
        try:
            hull = Delaunay(np.asarray(mesh.vertices)[:, :2])
        except QhullError:
            continue
        inside = hull.find_simplex(xyz_scene[:, :2]) >= 0
        pts = xyz_scene[inside & (xyz_scene[:, 2] > surface_z + 0.008)]
        if len(pts) >= 60:
            cluster_pts[label] = pts

    index = json.loads((args.proposals_dir / "proposals_index.json").read_text())
    cal = None
    results = {}
    for entry in index["proposals"]:
        plan = json.loads((args.proposals_dir / entry["file"]).read_text())
        q = q_at_close(plan)
        if q is None:
            results[entry["file"]] = {"skip": "no pre-close trajectory (place-style plan)"}
            continue
        if cal is None:
            cal = calibrate_pads(kin, q, tensor_args)
        if entry["target"] not in cluster_pts:
            results[entry["file"]] = {"skip": f"no raw points for {entry['target']}"}
            continue
        state = kin.get_state(tensor_args.to_device(q).float()[None])
        R, t = ee_frame(state)
        pts_ee = (cluster_pts[entry["target"]] - t) @ R
        m = slab_metrics(pts_ee, cal)
        m["pass"] = bool(
            m["n_slab"] >= args.min_slab_pts
            and m["thickness"] >= args.min_thickness
            and m["center_off"] <= args.max_center_off
            and m["ortho_off"] <= args.max_ortho_off
        )
        results[entry["file"]] = m
        _log.info(
            f"{entry['file']} [{entry['target']}]: n={m['n_slab']:5d} thick={m['thickness']*1000:5.1f}mm "
            f"center={m['center_off']*1000:5.1f}mm ortho={m['ortho_off']*1000:5.1f}mm -> "
            f"{'PASS' if m['pass'] else 'FAIL'}"
        )

    (args.proposals_dir / "gate.json").write_text(json.dumps({
        "h5": str(args.h5_path),
        "thresholds": {"min_slab": args.min_slab_pts, "min_thickness": args.min_thickness,
                       "max_center_off": args.max_center_off, "max_ortho_off": args.max_ortho_off},
        "use_plane_normal": bool(args.use_plane_normal),
        "results": results,
    }, indent=2))
    _log.info(f"gate.json written ({sum(1 for r in results.values() if r.get('pass'))} pass / "
              f"{sum(1 for r in results.values() if r.get('pass') is False)} fail)")

    if args.apply:
        # Two-stage selection: GWM decides the OBJECT (its semantic margin is
        # between objects); the gate only re-picks WITHIN that object, so a
        # gate-hostile target (e.g. a thin-walled bowl whose rim pinches all
        # fail) can never flip the selection to a distractor. The object comes
        # from score_client's `selected_target` (object-level aggregate);
        # pre-2026-08-11 score files only have the per-candidate ranking.
        scores = json.loads((args.proposals_dir / f"scores_{args.apply}.json").read_text())
        old = scores.get("winner_file", scores["argmax_file"])
        target = scores.get("selected_target", scores["ranking"][0]["target"])
        # Rank the survivors the way score_client ranks within an object: M2T2
        # confidence first, GWM score only as tie-break (`ranking` is GWM-
        # descending and max keeps the first maximal element).
        conf = {e["file"]: e.get("grasp_confidence") for e in index["proposals"]}
        passing = [r for r in scores["ranking"]
                   if r["target"] == target and results.get(r["file"], {}).get("pass")]
        winner = max(passing, key=lambda r: (conf.get(r["file"]) if conf.get(r["file"]) is not None
                                             else float("-inf"))) if passing else None
        if winner is None:
            _log.warning(
                f"--apply {args.apply}: no {target} plan passes the gate; keeping {old} (ungated)"
            )
            return
        shutil.copy(args.proposals_dir / winner["file"], args.proposals_dir / f"winner_{args.apply}.json")
        _log.info(
            f"--apply {args.apply}: winner {old} (ungated) -> {winner['file']} "
            f"(score {winner['score']:+.4f}, target {target}) written to winner_{args.apply}.json"
        )


if __name__ == "__main__":
    main()
