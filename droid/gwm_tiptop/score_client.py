"""Thin HTTP client for gwm-server + GI-4 driver: score saved proposals, pick the winner.

    python -m gwm_tiptop.score_client \
        --proposals-dir /root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene1 \
        --external-h5 /root/code/gwm/gwm-wiser/droid/droid-sim-evals/tiptop_assets/external_scene1_0.h5 \
        --instruction "pick up the cube" --tag cube_s1.0_start

Candidates are sent as execution timelines (positions + per-waypoint time and
gripper state, including the ~1.33 s gripper-action pauses the websocket
client inserts). RAT sampling is the single hyperparameter --rat-scale
(default 3.0 = WISER schedule x3 from the trajectory start, G-20; the literal
"none" = uniform 6 frames over the full trajectory, whatever its length),
forwarded per request so configs compare without restarting the server.

Selection is TWO-STAGE, and the two stages use DIFFERENT signals (G-28):

- WHICH OBJECT — GWM, reduced per object by --object-score (default `mean`).
  Semantic grounding is what GWM is for. The old behaviour (global argmax over
  individual candidates) is `--object-score max`.
- WHICH GRASP of that object — **M2T2 grasp confidence**, not GWM (GWM score
  only breaks ties). GWM's RAT frames are robot-only, so grasp robustness is
  information-theoretically invisible to it: on scene6_rev2's cube its
  within-object argmax is `plan_12` (M2T2 conf 0.341, a corner clip that
  shoves the cube — 0/5 in G-26) while M2T2's own ranking puts `plan_10`
  first (conf 0.778 — 5/5 in G-27). Within-object M2T2 confidence agrees with
  the closing-line gate on every object of that pool. Since
  `se3_fps_indices` seeds its farthest-point sampling at the highest-
  confidence pose, this is usually each object's first emitted candidate.

Why: `proposals.se3_fps_indices` samples each object's candidates by
DIVERSITY (confidence-weighted SE(3) farthest-point sampling), so with a small
per-object quota the candidates sit at the extremes of the object's grasp
family, not at its mode. A per-candidate argmax then compares order statistics
of extremes across objects, and at the ~0.01 score spread GWM produces within
a family that comparison is noise-limited. Measured on scene6_rev2 (16
candidates over 5 clusters, 10 referring expressions), re-reducing the SAME
saved scores: max 6/10 correct objects, mean/median/logsumexp/top2-mean all
9/10 — the four banana tasks lost to a red-bin candidate by +0.003..+0.008
under max. `mean` is the default because it is size-normalised (quotas are
floor+remainder, so families differ by one) and had the best worst-case margin
(+0.0052 vs median's +0.0015). The 4 place tasks are 4/4 under every rule.

HARDWARE CAVEAT (2026-08-19, user-reported): `mean` assumes an object's
candidates are comparable samples of how good that object is. They are not
when the object is hard to grasp. On *"pick up the object between the two
containers"* GWM ranked the tomato **first and second** overall (+0.7645,
+0.7541) and the blue cup third through seventh -- the grounding was right --
but the tomato's five candidates spread 0.0396 against the cup's 0.0118,
because M2T2's confidence in them was 0.16-0.37 against the cup's 0.46-0.78.
A small round object has a poor, diverse grasp family, `se3_fps_indices`
samples it for diversity, and the tail drags the mean down: cup +0.7445,
tomato +0.7436, and the robot went for the cup by 0.0008. `max` (+0.0138) and
`top2` (+0.0115) both get it right; `median` does not. Across the 25 scored
hardware runs to date the four rules disagree on 11, so one case does not
settle the general question -- but `top2` matched `mean`'s 9/10 in sim while
`max` managed 6/10, and it wins here, so of the two it is the better-supported
choice. **The HARDWARE session defaults to `top2` from 2026-08-19** (user
decision). This module's own default stays `mean` so droid-sim scripts that
call it directly reproduce unchanged.

VIEWPOINT (--cam, default `external_cam_2` since 2026-08-11, G-29): the DROID
rig carries two third-person cameras and the capture h5 stores both, so the
scene image the scorer sees is a free choice — and it is a FIRST-ORDER one.
Scored from `external_cam` the banana sits small, distant and inside the
gripper's shadow while the two bins dominate the frame, and the four
banana instructions win by only +0.014…+0.019 (`yellow` loses outright);
from `external_cam_2` the same instructions win by +0.091…+0.105 and object
accuracy goes 9/10 -> 10/10, with the cube/bowl tasks unchanged. The GI-2
renderer overlay was re-validated on `external_cam_2` before the switch
(overlays/scene6_external_cam_2/, 13.34 % robot coverage, no ghosting).
Caveat for any claim built on this: cam_2 was chosen AFTER looking at both
frames, so the 10/10 is not an unbiased viewpoint estimate. The unbiased
version — pass BOTH cameras (`--cam external_cam,external_cam_2`) and let the
fusion below average them — needs no such choice, is also 10/10 here, and is
markedly more robust (see the fusion comment in main()).

Writes winner_{tag}.json (a serialize_plan file, servable via
gwm_tiptop/policy_server.py --select fixed) and scores_{tag}.json next to the
proposals. scores_{tag}.json carries both the per-candidate `ranking` and the
`object_ranking` the choice was actually made on; `selected_target` is the
winning object (grasp_gate reads it to stay inside the same object).
"""

import argparse
import base64
import io
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation

GRIPPER_PAUSE_S = 20 / 15.0  # websocket client: 20 action steps at 15 Hz
GRIPPER_PAUSE_SUBSTEPS = 7


def score_candidates(
    server_url: str,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    world_from_cam: np.ndarray,
    instruction: str,
    candidates: list[dict],
    sampling: dict | None = None,
    timeout_s: float = 1800.0,
) -> dict:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    payload = {
        "rgb_png_b64": base64.b64encode(buf.getvalue()).decode(),
        "intrinsics": np.asarray(intrinsics).tolist(),
        "world_from_cam": np.asarray(world_from_cam).tolist(),
        "instruction": instruction,
        "candidates": candidates,
        **(sampling or {}),
    }
    resp = requests.post(f"{server_url.rstrip('/')}/score", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def plan_to_candidate(plan: dict, drop_static_prefix: bool = False,
                      append_release: bool = False) -> dict:
    """serialize_plan dict -> execution timeline for scoring.

    Trajectory steps advance time by their own dt per waypoint; gripper steps
    hold the last qpos for GRIPPER_PAUSE_S while the gripper value ramps to
    its target, mirroring how the websocket client executes the plan.

    Two hardware-only corrections, both default OFF so droid-sim is untouched:

    `drop_static_prefix` starts the timeline at the moment the arm first MOVES.
    A leading gripper step holds q_init -- the capture pose -- for 1.33 s, and
    the RAT window is laid over the whole timeline, so on a 2.8 s place plan
    two of the six frames landed on a pose that is identical across every
    candidate. A third of the evidence, carrying zero information about which
    candidate it was.

    `append_release` puts the RELEASE into the scored timeline. A hardware
    place plan ends the moment the gripper arrives above the target: the open
    is issued afterwards by the session, so it was never scored. Since the
    renderer draws the robot only and never the carried object, that left a
    place looking, pixel for pixel, exactly like a grasp approach -- empty
    gripper descending onto an object and stopping, still closed. The open is
    the one frame in which the two differ, and it was outside the window.
    """
    import numpy as np
    positions, times, gripper = [], [], []
    t, g, close_t = 0.0, 0.0, None
    for step in plan["steps"]:
        if step["type"] == "trajectory":
            dt = step["dt"]
            for p in step["positions"]:
                positions.append(p)
                times.append(t)
                gripper.append(g)
                t += dt
        elif step["type"] == "gripper":
            target = 1.0 if step["action"] == "close" else 0.0
            if step["action"] == "close" and close_t is None:
                close_t = t
            # A gripper step holds the arm where it already is. When the step
            # is FIRST -- which every place plan's leading close is -- there is
            # no previous waypoint, and falling back to zeros renders the arm
            # bolt upright in a configuration it never occupies. Those frames
            # then land inside the RAT window: measured on a hardware place,
            # 2 of the 6 sampled frames were the all-zero pose, identical
            # across every candidate, i.e. a third of the evidence was shared
            # noise. Place margins sat at ~0.0004 while picks, which have no
            # leading gripper step, scored ~0.03.
            last = positions[-1] if positions else list(plan["q_init"])
            for k in range(GRIPPER_PAUSE_SUBSTEPS):
                positions.append(last)
                times.append(t)
                gripper.append(g + (target - g) * (k + 1) / GRIPPER_PAUSE_SUBSTEPS)
                t += GRIPPER_PAUSE_S / GRIPPER_PAUSE_SUBSTEPS
            g = target

    if append_release and g != 0.0:
        last = positions[-1] if positions else list(plan["q_init"])
        for k in range(GRIPPER_PAUSE_SUBSTEPS):
            positions.append(last)
            times.append(t)
            gripper.append(g * (1.0 - (k + 1) / GRIPPER_PAUSE_SUBSTEPS))
            t += GRIPPER_PAUSE_S / GRIPPER_PAUSE_SUBSTEPS

    if drop_static_prefix and positions:
        P = np.asarray(positions, dtype=np.float64)
        moved = np.abs(P - P[0]).max(axis=1) > 1e-6
        # Keep the last stationary waypoint, so the window opens on the instant
        # of departure rather than one step after it.
        i = max(0, int(np.argmax(moved)) - 1) if moved.any() else 0
        positions, gripper = positions[i:], gripper[i:]
        times = [x - times[i] for x in times[i:]]

    return {
        "positions": [list(map(float, p)) for p in positions],
        "t": [round(float(x), 4) for x in times],
        "gripper": [round(float(x), 4) for x in gripper],
        "grasp_close_t": None if close_t is None else round(float(close_t), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals-dir", required=True, type=Path)
    ap.add_argument("--external-h5", required=True, type=Path)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--server-url", default="http://localhost:8901")
    ap.add_argument("--cam", default="external_cam_2",
                    help="which external camera(s) of the capture h5 the scene is scored from; "
                         "comma-separate for multi-view fusion (mean of per-candidate scores), "
                         "e.g. external_cam,external_cam_2. See the viewpoint note in the docstring")
    ap.add_argument("--tag", default="gwm")
    ap.add_argument("--rat-scale", default="3.0",
                    help="WISER schedule scale from trajectory start (G-20 default 3.0); "
                         "'none' = uniform 6 frames over the full trajectory")
    ap.add_argument("--task-image", default="current", choices=["current", "none"])
    ap.add_argument("--object-score", default="mean",
                    choices=["mean", "max", "median", "top2"],
                    help="how each object's candidates reduce to one object score; "
                         "max = pre-2026-08-11 global per-candidate argmax. `top2` is the "
                         "mean of an object's two best candidates -- it scored the same "
                         "9/10 as `mean` in sim (G-28) and, unlike `mean`, is not thrown "
                         "off by an object whose grasp family is wide because its grasps "
                         "are poor (see the hardware note below)")
    ap.add_argument("--dump-dir", type=Path)
    ap.add_argument("--drop-static-prefix", action="store_true",
                    help="open the RAT window where the arm first MOVES, not where the "
                         "timeline starts (hardware; see plan_to_candidate)")
    ap.add_argument("--append-release", action="store_true",
                    help="score the gripper OPENING at the target too -- the frame that "
                         "tells a place apart from a grasp (hardware; see plan_to_candidate)")
    args = ap.parse_args()
    rat_scale = None if args.rat_scale.strip().lower() == "none" else float(args.rat_scale)

    index = json.loads((args.proposals_dir / "proposals_index.json").read_text())
    plans = []
    for entry in index["proposals"]:
        plan = json.loads((args.proposals_dir / entry["file"]).read_text())
        plans.append((entry, plan))

    cams = [c.strip() for c in args.cam.split(",") if c.strip()]
    views = []
    with h5py.File(args.external_h5) as f:
        for cam in cams:
            pos = np.asarray(f[f"{cam}/pos_w"])
            w, x, y, z = np.asarray(f[f"{cam}/quat_w_ros"])
            c2w = np.eye(4)
            c2w[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            c2w[:3, 3] = pos
            views.append((cam, np.asarray(f[f"{cam}/rgb"])[..., :3],
                          np.asarray(f[f"{cam}/intrinsic_matrix"]), c2w))

    sampling = {"rat_scale": rat_scale, "task_image": args.task_image}
    timeline = {"drop_static_prefix": args.drop_static_prefix,
                "append_release": args.append_release}
    if not plans:
        raise SystemExit(
            f"{args.proposals_dir} has no proposals to score. The proposer found nothing "
            "it could plan for -- read its output, not this one. (Scoring an empty "
            "candidate list reaches gwm-server as a malformed request and comes back "
            "as an opaque 500.)")
    candidates = [plan_to_candidate(plan, **timeline) for _, plan in plans]
    per_view = []
    for cam, rgb, K, c2w in views:
        s = dict(sampling)
        if args.dump_dir:
            s["dump_dir"] = str(args.dump_dir if len(views) == 1 else args.dump_dir / cam)
        per_view.append((cam, score_candidates(args.server_url, rgb, K, c2w,
                                               args.instruction, candidates, s)))
    if args.dump_dir:
        sampling["dump_dir"] = str(args.dump_dir)
    sampling.update(timeline)   # so a viewer can rebuild the exact scored timeline

    # Multi-view fusion (G-30): plain arithmetic mean of each candidate's score
    # across views, then the unchanged two-stage selection. The views score the
    # SAME candidate set, so averaging per candidate and averaging each object's
    # per-view aggregate are the same number; per-candidate is the natural place.
    # Why the mean and not min/max/softmax-product/Borda: measured on this pool,
    # the mean had the lowest wrong-object rate under score noise at every level
    # (0.02 % vs cam-2-alone's 2.28 % at sigma 0.01) and adds no hyperparameter.
    # It is worth doing because the two views DISAGREE where it matters: pooled
    # correlation of their per-object deviations is only r=+0.58 (r~0.35 on the
    # four banana tasks, where one view is gripper-shadowed; r>0.85 where the
    # target is plainly visible in both), and the between-view spread (0.032) is
    # as large as the semantic signal itself (0.031).
    result = per_view[0][1]
    if len(per_view) > 1:
        scores = np.mean([r["scores"] for _, r in per_view], axis=0)
        sm = np.mean([r["softmax"] for _, r in per_view], axis=0)
        result = {**result,
                  "scores": [float(v) for v in scores],
                  "softmax": [float(v) for v in sm / sm.sum()],
                  "argmax": int(np.argmax(scores)),
                  "stats": {**result["stats"], "fused_views": cams,
                            "per_view_scores": {c: r["scores"] for c, r in per_view}},
                  "elapsed_s": sum(r["elapsed_s"] for _, r in per_view)}

    ranked = sorted(
        zip(result["scores"], result["softmax"], (e for e, _ in plans)),
        key=lambda t: -t[0],
    )
    print(f"\ninstruction: {args.instruction!r}  backend={result['stats']['backend']} "
          f"cams={cams} rat_scale={rat_scale} task_image={args.task_image}")
    for score, sm, entry in ranked:
        print(f"  {score:+.4f} (p={sm:.3f})  {entry['file']}  target={entry['target']}")

    reduce = {"mean": np.mean, "max": np.max, "median": np.median,
              "top2": lambda v: np.mean(sorted(v, reverse=True)[:2])}[args.object_score]
    per_object: dict[str, list[float]] = {}
    for score, _, entry in ranked:
        per_object.setdefault(entry["target"], []).append(score)
    object_ranking = sorted(
        ({"target": t, "score": float(reduce(v)), "n": len(v),
          "best": float(max(v)), "worst": float(min(v))} for t, v in per_object.items()),
        key=lambda d: -d["score"],
    )
    selected = object_ranking[0]["target"]
    # Winner = the selected object's most confident M2T2 grasp. `ranked` is
    # GWM-score-descending and max() keeps the first maximal element, so the
    # GWM score is the tie-break (and the sole criterion if the proposer
    # recorded no confidences).
    conf = {e["file"]: e.get("grasp_confidence") for e in index["proposals"]}
    win_entry = max((e for _, _, e in ranked if e["target"] == selected),
                    key=lambda e: (conf.get(e["file"]) if conf.get(e["file"]) is not None else float("-inf")))

    print(f"\nobject ranking ({args.object_score} over each object's candidates):")
    for d in object_ranking:
        print(f"  {d['score']:+.4f}  {d['target']}  (n={d['n']}, best {d['best']:+.4f})")

    shutil.copy(args.proposals_dir / win_entry["file"], args.proposals_dir / f"winner_{args.tag}.json")
    (args.proposals_dir / f"scores_{args.tag}.json").write_text(json.dumps({
        "instruction": args.instruction,
        "sampling": sampling,
        "server": result["stats"],
        "object_score": args.object_score,
        "cameras": cams,
        "selected_target": selected,
        "winner_file": win_entry["file"],
        "argmax_file": plans[result["argmax"]][0]["file"],  # per-candidate argmax, provenance only
        "elapsed_s": result["elapsed_s"],
        "object_ranking": object_ranking,
        "ranking": [{"file": e["file"], "target": e["target"], "score": s, "softmax": p}
                    for s, p, e in ranked],
    }, indent=2))
    print(f"\nwinner: {win_entry['file']} (target {selected}) -> "
          f"{args.proposals_dir / f'winner_{args.tag}.json'}")


if __name__ == "__main__":
    main()
