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

Writes winner_{tag}.json (a serialize_plan file, servable via
gwm_tiptop/policy_server.py --select fixed) and scores_{tag}.json next to the
proposals.
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


def plan_to_candidate(plan: dict) -> dict:
    """serialize_plan dict -> execution timeline for scoring.

    Trajectory steps advance time by their own dt per waypoint; gripper steps
    hold the last qpos for GRIPPER_PAUSE_S while the gripper value ramps to
    its target, mirroring how the websocket client executes the plan.
    """
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
            last = positions[-1] if positions else [0.0] * 7
            for k in range(GRIPPER_PAUSE_SUBSTEPS):
                positions.append(last)
                times.append(t)
                gripper.append(g + (target - g) * (k + 1) / GRIPPER_PAUSE_SUBSTEPS)
                t += GRIPPER_PAUSE_S / GRIPPER_PAUSE_SUBSTEPS
            g = target
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
    ap.add_argument("--cam", default="external_cam")
    ap.add_argument("--tag", default="gwm")
    ap.add_argument("--rat-scale", default="3.0",
                    help="WISER schedule scale from trajectory start (G-20 default 3.0); "
                         "'none' = uniform 6 frames over the full trajectory")
    ap.add_argument("--task-image", default="current", choices=["current", "none"])
    ap.add_argument("--dump-dir", type=Path)
    args = ap.parse_args()
    rat_scale = None if args.rat_scale.strip().lower() == "none" else float(args.rat_scale)

    index = json.loads((args.proposals_dir / "proposals_index.json").read_text())
    plans = []
    for entry in index["proposals"]:
        plan = json.loads((args.proposals_dir / entry["file"]).read_text())
        plans.append((entry, plan))

    with h5py.File(args.external_h5) as f:
        rgb = np.asarray(f[f"{args.cam}/rgb"])[..., :3]
        K = np.asarray(f[f"{args.cam}/intrinsic_matrix"])
        pos = np.asarray(f[f"{args.cam}/pos_w"])
        w, x, y, z = np.asarray(f[f"{args.cam}/quat_w_ros"])
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        c2w[:3, 3] = pos

    sampling = {"rat_scale": rat_scale, "task_image": args.task_image}
    if args.dump_dir:
        sampling["dump_dir"] = str(args.dump_dir)
    candidates = [plan_to_candidate(plan) for _, plan in plans]
    result = score_candidates(args.server_url, rgb, K, c2w, args.instruction, candidates, sampling)

    ranked = sorted(
        zip(result["scores"], result["softmax"], (e for e, _ in plans)),
        key=lambda t: -t[0],
    )
    print(f"\ninstruction: {args.instruction!r}  backend={result['stats']['backend']} "
          f"rat_scale={rat_scale} task_image={args.task_image}")
    for score, sm, entry in ranked:
        print(f"  {score:+.4f} (p={sm:.3f})  {entry['file']}  target={entry['target']}")

    win_entry, _ = plans[result["argmax"]]
    shutil.copy(args.proposals_dir / win_entry["file"], args.proposals_dir / f"winner_{args.tag}.json")
    (args.proposals_dir / f"scores_{args.tag}.json").write_text(json.dumps({
        "instruction": args.instruction,
        "sampling": sampling,
        "server": result["stats"],
        "argmax_file": win_entry["file"],
        "elapsed_s": result["elapsed_s"],
        "ranking": [{"file": e["file"], "target": e["target"], "score": s, "softmax": p}
                    for s, p, e in ranked],
    }, indent=2))
    print(f"\nwinner: {win_entry['file']} -> {args.proposals_dir / f'winner_{args.tag}.json'}")


if __name__ == "__main__":
    main()
