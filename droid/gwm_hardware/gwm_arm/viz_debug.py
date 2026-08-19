"""See what the GWM arm decided, and why.

Baseline TiPToP opens a Rerun window and shows its intermediate artefacts. This
arm keeps that window and adds the things that are specific to it -- the parts
of the decision that are otherwise a number in a JSON file:

  * the anonymous clusters the geometric perception found, one colour each,
    against the cloud they came from;
  * EVERY candidate trajectory still in the scene, drawn as the path its TCP
    takes, **coloured by the score GWM gave it against the instruction**;
  * the two-stage selection laid out: the per-object aggregate that chose the
    object, then the M2T2 confidence that chose the grasp within it;
  * the closing-line gate's verdict per candidate, and where the finger pads
    end up at the closing pose.

Two outputs, because they answer different questions:

  Rerun 3D  -- geometry. Is that cluster the object I think it is, does that
               path go where I expect, do the pads straddle the target.
  score_overlay.png -- the picture to look at first, and the one to keep: the
               candidate paths drawn over the camera image, coloured by score.

**Colour is RELATIVE, on purpose.** GWM's cosine scores across a candidate set
span about 0.01 (G-28), so an absolute colour map would render every candidate
the same shade and hide exactly the structure worth seeing. The ramp is
stretched over the observed min..max of THIS set and the range is printed on
the legend, so a wide spread and a hair-thin one look different on the page
rather than looking identical. Read the numbers, not the hue, for magnitude.

Runs in the droid/tiptop pixi env (it needs cuRobo FK), from the repo root:

    python -m gwm_hardware.gwm_arm.viz_debug \
        --proposals-dir runs/gwm/scene01/proposals \
        --h5-path runs/gwm/scene01/wrist_obs.h5 \
        --external-h5 runs/gwm/scene01/external_obs.h5 \
        --tag pick_cup --rerun
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_arm.viz_debug")

# Blue (worst) -> grey -> red (best). Chosen over a rainbow because the only
# question the colour answers is "better or worse than the others", which is
# one axis, and because it survives being printed in greyscale as lightness.
RAMP = np.array([[60, 90, 200], [140, 150, 170], [220, 60, 50]], dtype=np.float64)


def ramp_colour(x: float) -> np.ndarray:
    """x in [0, 1] -> RGB uint8 along RAMP."""
    x = float(np.clip(x, 0.0, 1.0)) * (len(RAMP) - 1)
    i = min(int(x), len(RAMP) - 2)
    return (RAMP[i] + (RAMP[i + 1] - RAMP[i]) * (x - i)).astype(np.uint8)


def plan_waypoints(plan: dict) -> tuple[np.ndarray, int | None]:
    """All trajectory waypoints in order -> ((T,7), index of the gripper close)."""
    q, close_at = [], None
    for step in plan["steps"]:
        if step["type"] == "trajectory":
            q.extend(step["positions"])
        elif step["type"] == "gripper" and step.get("action") == "close" and close_at is None:
            close_at = max(len(q) - 1, 0)
    return np.asarray(q, dtype=np.float64), close_at


def tcp_path(kin, tensor_args, q: np.ndarray) -> np.ndarray:
    """(T,7) joint waypoints -> (T,3) TCP positions, one batched FK call."""
    state = kin.get_state(tensor_args.to_device(q).float())
    return state.ee_pose.position.cpu().numpy().astype(np.float64)


def project(points: np.ndarray, K: np.ndarray, world_from_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points -> (uv (N,2) int, in_front_and_in_frame mask)."""
    w2c = np.linalg.inv(world_from_cam)
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    ok = pc[:, 2] > 0.05
    uv = np.zeros((len(points), 2))
    p = (K @ pc[ok].T)
    uv[ok] = (p[:2] / p[2]).T
    return uv, ok


def load_context(proposals_dir: Path, tag: str | None) -> dict:
    """Everything the viewer knows: proposals, scores, gate -- any of them missing."""
    index = json.loads((proposals_dir / "proposals_index.json").read_text())
    ctx = {"index": index, "scores": None, "gate": None}
    if tag:
        sp = proposals_dir / f"scores_{tag}.json"
        if sp.exists():
            ctx["scores"] = json.loads(sp.read_text())
        else:
            _log.warning(f"no {sp.name}: colouring by M2T2 grasp confidence instead of GWM score")
    gp = proposals_dir / "gate.json"
    if gp.exists():
        ctx["gate"] = json.loads(gp.read_text())
    return ctx


def candidate_table(ctx: dict) -> list[dict]:
    """One row per candidate, in the order they will be drawn (worst first)."""
    index, scores, gate = ctx["index"], ctx["scores"], ctx["gate"]
    by_file = {e["file"]: e for e in index["proposals"]}
    rows = []
    if scores:
        ranking = {r["file"]: r for r in scores["ranking"]}
        for f, e in by_file.items():
            r = ranking.get(f, {})
            rows.append({"file": f, "target": e["target"],
                         "conf": e.get("grasp_confidence"),
                         "score": r.get("score"), "softmax": r.get("softmax")})
    else:
        for f, e in by_file.items():
            rows.append({"file": f, "target": e["target"],
                         "conf": e.get("grasp_confidence"), "score": None, "softmax": None})
    if gate:
        for row in rows:
            g = gate["results"].get(row["file"], {})
            row["gate"] = g.get("pass")
            row["gate_metrics"] = {k: g[k] for k in
                                   ("n_slab", "thickness", "center_off", "ortho_off") if k in g}
    key = (lambda r: (r["score"] if r["score"] is not None else -np.inf))
    if not scores:
        key = (lambda r: (r["conf"] if r["conf"] is not None else -np.inf))
    return sorted(rows, key=key)


def colour_values(rows: list[dict]) -> tuple[list[float], str, tuple[float, float]]:
    """Normalised colour position per row, the field used, and its raw range."""
    field = "score" if any(r["score"] is not None for r in rows) else "conf"
    vals = [r[field] for r in rows]
    if any(v is None for v in vals):
        vals = [v if v is not None else min(x for x in vals if x is not None) for v in vals]
    lo, hi = float(min(vals)), float(max(vals))
    span = hi - lo
    norm = [0.5 if span <= 0 else (v - lo) / span for v in vals]
    return norm, field, (lo, hi)


# ------------------------------------------------------------------ 2D overlay


def draw_overlay(out: Path, rgb: np.ndarray, K: np.ndarray, world_from_cam: np.ndarray,
                 rows: list[dict], paths: dict[str, np.ndarray], ctx: dict,
                 instruction: str | None, tail_frac: float = 0.3) -> None:
    import cv2

    img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    norm, field, (lo, hi) = colour_values(rows)
    selected = (ctx["scores"] or {}).get("selected_target")
    winner = (ctx["scores"] or {}).get("winner_file")

    # Worst first, so the best candidate is drawn last and sits on top.
    for row, x in zip(rows, norm):
        entry = paths.get(row["file"])
        if entry is None:
            continue
        pts, close_at = entry
        # Everything after the gripper closes is the retract, which is the same
        # motion for every candidate and says nothing about the choice.
        pts = pts[:(close_at + 1) if close_at is not None else len(pts)]
        if len(pts) < 2:
            continue
        uv, ok = project(pts, K, world_from_cam)
        col = tuple(int(c) for c in ramp_colour(x)[::-1])  # BGR
        is_winner = row["file"] == winner
        # Every candidate starts from the same q_init, so the first two thirds
        # of all 16 paths are near-identical transit and just ink over the
        # scene. The APPROACH is what distinguishes them, so the transit is
        # drawn hairline and the tail solid -- nothing is hidden, but the part
        # that carries the decision is the part that reads.
        cut = int(len(pts) * (1.0 - tail_frac))
        for a in range(len(pts) - 1):
            b = a + 1
            if not (ok[a] and ok[b]):
                continue
            tail = a >= cut
            cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), col,
                     (4 if is_winner else 2) if tail else 1, cv2.LINE_AA)
        if ok[-1]:
            end = tuple(uv[-1].astype(int))
            cv2.drawMarker(img, end, col, cv2.MARKER_CROSS,
                           20 if is_winner else 12, 2, cv2.LINE_AA)
            cv2.circle(img, end, 10 if is_winner else 5, col, 2, cv2.LINE_AA)
            if is_winner:
                cv2.circle(img, end, 16, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend. Sorted best first here, which is the order a reader wants, even
    # though the drawing order above is the opposite for occlusion reasons.
    pad, lh = 10, 20
    rows_desc = rows[::-1]
    box_h = lh * (len(rows_desc) + 5) + 2 * pad
    box_w = 470
    panel = img[pad:pad + box_h, pad:pad + box_w].copy()
    cv2.rectangle(img, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    img[pad:pad + box_h, pad:pad + box_w] = cv2.addWeighted(
        img[pad:pad + box_h, pad:pad + box_w], 0.45, panel, 0.55, 0)

    def line(i, text, colour=(255, 255, 255), scale=0.45):
        cv2.putText(img, text, (pad + 8, pad + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, 1, cv2.LINE_AA)

    i = 0
    line(i, (instruction or "(no instruction)")[:60], (120, 255, 255), 0.5); i += 1
    line(i, f"colour = {'GWM score' if field == 'score' else 'M2T2 confidence'} "
            f"stretched over {lo:+.4f}..{hi:+.4f}  (span {hi - lo:.4f})", (200, 200, 200)); i += 1
    line(i, f"selected object: {selected or '-'}    winner: {winner or '-'}",
         (120, 255, 120)); i += 1
    line(i, f"path drawn to the gripper CLOSE (retract omitted); solid = last "
            f"{tail_frac:.0%}; cross = grasp", (200, 200, 200)); i += 1
    line(i, f"{'candidate':<26}{'target':<11}{'score':>9} {'conf':>6} gate", (180, 180, 180)); i += 1
    for row, x in zip(rows_desc, norm[::-1]):
        col = tuple(int(c) for c in ramp_colour(x)[::-1])
        gate = "" if row.get("gate") is None else ("PASS" if row["gate"] else "FAIL")
        mark = "*" if row["file"] == winner else " "
        score_s = "   -   " if row["score"] is None else f"{row['score']:+.4f}"
        conf_s = "  -  " if row["conf"] is None else f"{row['conf']:.3f}"
        line(i, f"{mark}{row['file'][:25]:<26}{row['target'][:10]:<11}"
                f"{score_s:>9} {conf_s:>6} {gate}", col)
        i += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    _log.info(f"overlay -> {out}")


# -------------------------------------------------------------------- 3D view


def log_rerun(ctx: dict, rows: list[dict], paths: dict[str, np.ndarray],
              obs: dict, xyz_map: np.ndarray, rgb_map: np.ndarray,
              object_pcds: dict, table_box, instruction: str | None,
              external: tuple | None) -> None:
    import rerun as rr

    rr.init("gwm_arm_debug", spawn=True)
    rr.log("wrist/rgb", rr.Image(obs["rgb"]), static=True)
    if external is not None:
        rr.log("external/rgb", rr.Image(external[0]), static=True)

    finite = np.isfinite(xyz_map).all(axis=2)
    rr.log("world/cloud", rr.Points3D(positions=xyz_map[finite].reshape(-1, 3),
                                      colors=rgb_map[finite].reshape(-1, 3)), static=True)
    palette = [(255, 80, 80), (80, 220, 80), (90, 150, 255), (240, 230, 60),
               (230, 90, 230), (70, 230, 230), (255, 165, 40), (170, 90, 255)]
    for i, (label, pcd) in enumerate(object_pcds.items()):
        rr.log(f"world/clusters/{label}",
               rr.Points3D(positions=np.asarray(pcd.points),
                           colors=np.tile(palette[i % len(palette)], (len(pcd.points), 1)),
                           radii=0.0025), static=True)

    norm, field, (lo, hi) = colour_values(rows)
    winner = (ctx["scores"] or {}).get("winner_file")
    for row, x in zip(rows, norm):
        entry = paths.get(row["file"])
        if entry is None:
            continue
        pts, close_at = entry
        pts = pts[:(close_at + 1) if close_at is not None else len(pts)]
        if len(pts) < 2:
            continue
        col = ramp_colour(x)
        parts = [row["file"], row["target"]]
        if row["score"] is not None:
            parts.append(f"score {row['score']:+.4f}")
        if row["conf"] is not None:
            parts.append(f"conf {row['conf']:.3f}")
        label = "  ".join(parts)
        rr.log(f"world/candidates/{Path(row['file']).stem}",
               rr.LineStrips3D([pts], colors=[col],
                               radii=0.004 if row["file"] == winner else 0.0015,
                               labels=[label]), static=True)
        rr.log(f"world/candidates/{Path(row['file']).stem}/grasp",
               rr.Points3D(positions=pts[-1:], colors=[col],
                           radii=0.012 if row["file"] == winner else 0.006), static=True)

    lines = [f"instruction: {instruction or '-'}",
             f"colour: {field} over {lo:+.4f}..{hi:+.4f}"]
    if ctx["scores"]:
        lines.append(f"selected object: {ctx['scores'].get('selected_target')}")
        lines.append(f"winner: {ctx['scores'].get('winner_file')}")
        for d in ctx["scores"].get("object_ranking", []):
            lines.append(f"  {d['score']:+.4f}  {d['target']}  (n={d['n']}, best {d['best']:+.4f})")
    for row in rows[::-1]:
        g = "" if row.get("gate") is None else ("  gate PASS" if row["gate"] else "  gate FAIL")
        sc = float("nan") if row["score"] is None else row["score"]
        cf = float("nan") if row["conf"] is None else row["conf"]
        lines.append(f"{row['file']:<28}{row['target']:<11}{sc:+.4f}  conf {cf:.3f}{g}")
    rr.log("selection", rr.TextDocument("\n".join(lines)), static=True)
    _log.info("Rerun viewer up: world/cloud, world/clusters, world/candidates, selection")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proposals-dir", required=True, type=Path)
    ap.add_argument("--h5-path", required=True, type=Path, help="the wrist capture")
    ap.add_argument("--external-h5", type=Path,
                    help="draw the overlay on the scoring view instead of the wrist view")
    ap.add_argument("--cam", default=None, help="which group of the external h5")
    ap.add_argument("--tag", default=None, help="scores_TAG.json to colour by")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--out", type=Path, default=None)
    # On by default, because `tiptop_run` spawns its Rerun viewer unconditionally
    # and this arm is supposed to keep that: the interactive window is how the
    # intermediate artefacts get looked at, and a debug view you have to
    # remember to ask for is a debug view nobody sees.
    ap.add_argument("--no-rerun", dest="rerun", action="store_false",
                    help="skip the 3D viewer (e.g. over a headless ssh session)")
    ap.add_argument("--tail-frac", type=float, default=0.3,
                    help="fraction of each path drawn solid, measured from the grasp back")
    ap.add_argument("--horizontal-cut", dest="use_plane_normal", action="store_false")
    args = ap.parse_args()

    import h5py
    from curobo.types.base import TensorDeviceType
    from scipy.spatial.transform import Rotation

    from tiptop.motion_planning import build_curobo_solvers
    from tiptop.perception.utils import depth_to_xyz

    from gwm_tiptop.perception_geometric import cluster_objects, find_table_plane
    from gwm_tiptop.propose_from_h5 import load_h5_observation
    from gwm_hardware.gwm_arm.capture import EXTERNAL_CAM

    ctx = load_context(args.proposals_dir, args.tag)
    instruction = args.instruction or (ctx["scores"] or {}).get("instruction")
    rows = candidate_table(ctx)

    obs = load_h5_observation(args.h5_path)
    depth = obs["depth"].copy()
    depth[~np.isfinite(depth)] = np.nan
    depth[(depth <= 0.05) | (depth > 4.0)] = np.nan
    xyz_map = depth_to_xyz(depth, obs["K"])
    xyz_map = xyz_map @ obs["world_from_cam"][:3, :3].T + obs["world_from_cam"][:3, 3]
    rgb_map = obs["rgb"].astype(np.float32) / 255.0

    tensor_args = TensorDeviceType()
    _, motion_gen, _ = build_curobo_solvers(num_particles=32, num_spheres=64,
                                            include_workspace=False)
    # (tcp path, index of the gripper close). The close index matters: a
    # tiptop pick plan is MoveFree -> Pick, and Pick RETRACTS afterwards, so
    # the last waypoint of every candidate is the same retract pose. Marking
    # the end of the path as "the grasp" would put all 16 crosses on one pixel
    # off the bottom of the frame -- which is exactly what the first version
    # of this viewer did.
    paths = {}
    for row in rows:
        plan = json.loads((args.proposals_dir / row["file"]).read_text())
        q, close_at = plan_waypoints(plan)
        if len(q):
            paths[row["file"]] = (tcp_path(motion_gen.kinematics, tensor_args, q), close_at)

    # Which image to draw on. The scoring view is the honest one -- it is what
    # GWM actually saw -- but it needs the external extrinsics, so the wrist
    # view is the fallback that always works.
    if args.external_h5:
        cam = args.cam or EXTERNAL_CAM
        with h5py.File(args.external_h5) as f:
            rgb = np.asarray(f[f"{cam}/rgb"])[..., :3]
            K = np.asarray(f[f"{cam}/intrinsic_matrix"])
            w, x, y, z = np.asarray(f[f"{cam}/quat_w_ros"])
            c2w = np.eye(4)
            c2w[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            c2w[:3, 3] = np.asarray(f[f"{cam}/pos_w"])
        view = (rgb, K, c2w, f"external:{cam}")
    else:
        view = (obs["rgb"], obs["K"], obs["world_from_cam"], "wrist")
        _log.info("drawing on the WRIST view (no --external-h5); this is not the "
                  "viewpoint GWM scored from")

    out = args.out or args.proposals_dir / f"score_overlay{'_' + args.tag if args.tag else ''}.png"
    draw_overlay(out, view[0], view[1], view[2], rows, paths, ctx, instruction,
                 tail_frac=args.tail_frac)

    if args.rerun:
        table_box, surface_z = find_table_plane(xyz_map, rgb_map)
        _, object_pcds = cluster_objects(xyz_map, rgb_map, table_box, surface_z + 0.015,
                                         use_plane_normal=args.use_plane_normal)
        log_rerun(ctx, rows, paths, obs, xyz_map, rgb_map, object_pcds, table_box,
                  instruction, None if not args.external_h5 else view)


if __name__ == "__main__":
    main()
