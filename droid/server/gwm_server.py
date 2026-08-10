"""gwm-server: trajectory scoring microservice (integration plan D9 / gwm_integrate_doc GI-4).

POST /score with the current external-cam frame, camera params, the task
instruction, and serialize_plan-style candidates; returns per-candidate scores
and the argmax. Backends:

- ``dummy``  — renders each candidate's robot-only RAT frames with the shared
  FrankaRobotRenderer (exercising the real render seam and its latency), then
  scores with a deterministic hash of the trajectory. Selection is meaningless
  by design; the mechanics are real. Use until the run-1 checkpoint lands.
- ``gwm``    — full Qwen3-VL + GWM scoring (to be wired when the retrained
  checkpoint is available; G-14).

Run inside the gwm venv:

    cd /root/code/gwm/gwm-wiser && .venv/bin/python -m droid.server.gwm_server \
        --backend dummy --urdf /root/code/gwm/gwm-wiser/droid/tiptop/gwm_tiptop/assets/panda_robotiq_droidsim.urdf
"""

import argparse
import base64
import hashlib
import io
import logging
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from real_data_train.renderer.franka_renderer import FrankaRobotRenderer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_server")

NUM_RAT_FRAMES = 6  # [current full RGB, 5 robot-only renders]


class Candidate(BaseModel):
    positions: list  # (T, 7) joint positions
    gripper_open_mask: list | None = None  # optional (T,) 1=open, 0=closed
    dt: float = 0.02


class ScoreRequest(BaseModel):
    rgb_png_b64: str  # current external-cam frame, PNG base64
    intrinsics: list  # (3, 3)
    world_from_cam: list  # (4, 4) CV-axis cam2world
    instruction: str
    candidates: list[Candidate]


class DummyBackend:
    """Renders real RAT frames, scores with a deterministic trajectory hash."""

    def __init__(self, urdf_path: str, arm: str = "panda"):
        self.renderer = FrankaRobotRenderer(urdf_path, arm=arm)

    def score(self, req: ScoreRequest, rgb: np.ndarray) -> tuple[list[float], dict]:
        h, w = rgb.shape[:2]
        K = np.asarray(req.intrinsics, dtype=np.float64)
        c2w = np.asarray(req.world_from_cam, dtype=np.float64)
        scores, render_ms = [], []
        for cand in req.candidates:
            qpos = np.asarray(cand.positions, dtype=np.float64)
            # Uniform 6 frames over the full trajectory (G-10); frame 0 is the
            # current full frame, so 5 robot-only renders from frames 1..5.
            idxs = np.linspace(0, len(qpos) - 1, NUM_RAT_FRAMES).round().astype(int)[1:]
            grip = np.zeros(len(idxs))  # pick approach renders with open gripper
            t0 = time.perf_counter()
            frames = self.renderer.render(qpos[idxs], grip, K, c2w, width=w, height=h)
            render_ms.append((time.perf_counter() - t0) * 1000)
            assert frames.shape == (len(idxs), h, w, 3)
            digest = hashlib.sha256(qpos.tobytes() + req.instruction.encode()).digest()
            scores.append(int.from_bytes(digest[:8], "little") / 2**64)
        stats = {"render_ms_per_candidate": render_ms, "backend": "dummy"}
        return scores, stats


def build_app(backend) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "healthy", "backend": type(backend).__name__}

    @app.post("/score")
    def score(req: ScoreRequest):
        from PIL import Image

        t0 = time.perf_counter()
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(req.rgb_png_b64))))[..., :3]
        raw_scores, stats = backend.score(req, rgb)
        raw = np.asarray(raw_scores)
        z = np.exp(raw - raw.max())
        softmax = (z / z.sum()).tolist()
        elapsed = time.perf_counter() - t0
        _log.info(
            f"Scored {len(req.candidates)} candidates for '{req.instruction}' in {elapsed:.2f}s "
            f"(argmax {int(np.argmax(raw))})"
        )
        return {
            "scores": raw_scores,
            "softmax": softmax,
            "argmax": int(np.argmax(raw)),
            "elapsed_s": elapsed,
            "stats": stats,
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["dummy", "gwm"], default="dummy")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--arm", default="panda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8901)
    args = ap.parse_args()

    if args.backend == "dummy":
        backend = DummyBackend(args.urdf, args.arm)
    else:
        raise NotImplementedError("gwm backend lands with the run-1 checkpoint (G-14)")
    uvicorn.run(build_app(backend), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
