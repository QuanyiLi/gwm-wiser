"""Thin HTTP client for gwm-server + GI-4 driver: score saved proposals, pick the winner.

    python -m gwm_tiptop.score_client \
        --proposals-dir /root/code/gwm/gwm-wiser/droid/gwm_integrate_doc/proposals/scene1 \
        --external-h5 /root/code/gwm/gwm-wiser/droid/droid-sim-evals/tiptop_assets/external_scene1_0.h5 \
        --instruction "pick up the Rubik's cube"

Writes winner.json (a serialize_plan file, replayable via droid-sim-evals
replay_json_traj.py) and scores.json next to the proposals.
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


def score_candidates(
    server_url: str,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    world_from_cam: np.ndarray,
    instruction: str,
    candidates: list[dict],
    timeout_s: float = 600.0,
) -> dict:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    payload = {
        "rgb_png_b64": base64.b64encode(buf.getvalue()).decode(),
        "intrinsics": np.asarray(intrinsics).tolist(),
        "world_from_cam": np.asarray(world_from_cam).tolist(),
        "instruction": instruction,
        "candidates": candidates,
    }
    resp = requests.post(f"{server_url.rstrip('/')}/score", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def plan_to_candidate(plan: dict) -> dict:
    """serialize_plan dict -> flat joint trajectory for scoring (trajectory steps concatenated)."""
    positions = [np.asarray(s["positions"]) for s in plan["steps"] if s["type"] == "trajectory"]
    dt = next(s["dt"] for s in plan["steps"] if s["type"] == "trajectory")
    return {"positions": np.concatenate(positions).tolist(), "dt": dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals-dir", required=True, type=Path)
    ap.add_argument("--external-h5", required=True, type=Path)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--server-url", default="http://localhost:8901")
    ap.add_argument("--cam", default="external_cam")
    args = ap.parse_args()

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

    candidates = [plan_to_candidate(plan) for _, plan in plans]
    result = score_candidates(args.server_url, rgb, K, c2w, args.instruction, candidates)

    ranked = sorted(
        zip(result["scores"], result["softmax"], (e for e, _ in plans)),
        key=lambda t: -t[0],
    )
    print(f"\ninstruction: {args.instruction!r}  ({result['stats']['backend']} backend)")
    for score, sm, entry in ranked:
        print(f"  {score:.4f} (p={sm:.3f})  {entry['file']}  target={entry['target']}")

    win_entry, _ = plans[result["argmax"]]
    shutil.copy(args.proposals_dir / win_entry["file"], args.proposals_dir / "winner.json")
    (args.proposals_dir / "scores.json").write_text(json.dumps({
        "instruction": args.instruction,
        "server": result["stats"],
        "argmax_file": win_entry["file"],
        "elapsed_s": result["elapsed_s"],
        "ranking": [{"file": e["file"], "target": e["target"], "score": s, "softmax": p}
                     for s, p, e in ranked],
    }, indent=2))
    print(f"\nwinner: {win_entry['file']} -> {args.proposals_dir / 'winner.json'}")


if __name__ == "__main__":
    main()
