"""gwm-server: trajectory scoring microservice (integration plan D9 / gwm_integrate_doc GI-4).

POST /score with the current external-cam frame, camera params, the task
instruction, and execution-timeline candidates; returns per-candidate scores
and the argmax. Backends:

- ``dummy``  — renders each candidate's robot-only RAT frames with the shared
  FrankaRobotRenderer (exercising the real render seam and its latency), then
  scores with a deterministic hash of the trajectory. Selection is meaningless
  by design; the mechanics are real.
- ``gwm``    — Qwen3-VL-Embedding + trained GWM. RAT condition =
  [current external RGB, 5 robot-only renders] sampled on the training-time
  WISER schedule (real_data_train.windows.SCHEDULE x scale), anchor-resized to
  624x352 and preprocessed with the training pixel budget so the video lands
  on the canonical (3,18,30) = 1620-token grid. Task embedding = the verbatim
  instruction (+ optionally the current frame) under the retrieval planner's
  text instruction; score = cosine(task, predicted video embedding).

RAT sampling is controlled by the single per-request hyperparameter
``rat_scale`` (G-20): default 3.0 = WISER schedule x3 from the trajectory
start (8.85 s window, the sim-source training ceiling); None = uniform 6
frames over the full trajectory, whatever its length. Windows longer than the
plan shrink to fit, mirroring training's fit-fallback (D-33).

Run inside the gwm venv:

    cd /root/code/gwm/gwm-wiser && .venv/bin/python -m droid.server.gwm_server \
        --backend gwm --urdf droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf \
        --ckpt /root/exp_ret/0810_gwm/checkpoint.pt
"""

import argparse
import base64
import hashlib
import io
import logging
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from real_data_train.renderer.franka_renderer import FrankaRobotRenderer
from real_data_train.windows import SCHEDULE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("gwm_server")

NUM_RAT_FRAMES = 6  # [current full RGB, 5 robot-only renders]

# Task embeddings are keyed by (instruction, mode, frame). A turn uses two --
# the instruction and the empty-instruction prior -- so this holds a handful of
# recent scenes, which is all the reuse there is to have.
TASK_CACHE_MAX = 16

# Retrieval planner's embedding-side instruction (gwm_wiser/planner/retrieval.py)
TEXT_INSTRUCTION = (
    "Retrieve the video which can best finish the manipulation task specified by the user, "
    "given the layout of the workspace and the current frame observation."
)


class Candidate(BaseModel):
    positions: list  # (T, 7) joint positions
    t: list | None = None  # (T,) seconds on the execution timeline (incl. gripper pauses)
    gripper: list | None = None  # (T,) 0=open .. 1=closed
    grasp_close_t: float | None = None  # first gripper-close time on the timeline
    gripper_open_mask: list | None = None  # legacy
    dt: float = 0.02  # legacy fallback when t is missing


class ScoreRequest(BaseModel):
    rgb_png_b64: str  # current external-cam frame, PNG base64
    intrinsics: list  # (3, 3)
    world_from_cam: list  # (4, 4) CV-axis cam2world
    instruction: str
    candidates: list[Candidate]
    rat_scale: float | None = 3.0  # WISER schedule x scale from the start; None = uniform over the full trajectory (G-20)
    task_image: str = "current"  # current | none
    dump_dir: str | None = None  # save the exact RAT strips fed to the model
    # Appended (with a space) to TEXT_INSTRUCTION for BOTH the task embedding
    # and the empty-instruction prior, so debias subtracts it back out. The
    # hardware session uses it to pin down "left/right" for its camera; absent
    # (the default, and droid-sim never sends it) the text path is unchanged.
    text_instruction_extra: str | None = None


def candidate_timeline(cand: Candidate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(qpos (T,7), t (T,), gripper (T,)) with legacy-field fallbacks."""
    qpos = np.asarray(cand.positions, dtype=np.float64)
    n = len(qpos)
    t = np.asarray(cand.t, dtype=np.float64) if cand.t is not None else np.arange(n) * cand.dt
    if cand.gripper is not None:
        grip = np.asarray(cand.gripper, dtype=np.float64)
    elif cand.gripper_open_mask is not None:
        grip = 1.0 - np.asarray(cand.gripper_open_mask, dtype=np.float64)
    else:
        grip = np.zeros(n)
    return qpos, t, grip


def sample_rat_times(rat_scale: float | None, t: np.ndarray) -> np.ndarray:
    """Six timeline indices for the RAT window (G-20 hyperparameter).

    rat_scale None discretizes the whole trajectory uniformly; a number places
    the WISER schedule x scale at the trajectory start, shrinking the scale
    when the window would overrun the plan (training's fit-fallback, D-33).
    """
    if rat_scale is None:
        return np.linspace(0, len(t) - 1, NUM_RAT_FRAMES).round().astype(int)
    scale = rat_scale
    if t[-1] > 0 and SCHEDULE[-1] * scale > t[-1]:
        scale = t[-1] / SCHEDULE[-1]
    times = np.asarray(SCHEDULE) * scale
    return np.array([int(np.abs(t - x).argmin()) for x in times])


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
            qpos, t, grip = candidate_timeline(cand)
            idxs = sample_rat_times(req.rat_scale, t)[1:]
            t0 = time.perf_counter()
            frames = self.renderer.render(qpos[idxs], grip[idxs], K, c2w, width=w, height=h)
            render_ms.append((time.perf_counter() - t0) * 1000)
            assert frames.shape == (len(idxs), h, w, 3)
            digest = hashlib.sha256(qpos.tobytes() + req.instruction.encode()).digest()
            scores.append(int.from_bytes(digest[:8], "little") / 2**64)
        stats = {"render_ms_per_candidate": render_ms, "backend": "dummy"}
        return scores, stats


class GwmBackend:
    """Qwen3-VL-Embedding + trained GWM scoring over rendered RAT candidates."""

    def __init__(self, urdf_path: str, arm: str, ckpt_path: str, embedder_path: str,
                 head_dtype: str = "fp32"):
        import torch
        from PIL import Image  # noqa: F401  (fail early if missing)

        from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLEmbedder
        from real_data_train.gwm_model import load_canonical_like_planner
        from real_data_train.qwen_rat import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, _apply_pixel_budget
        from real_data_train.rendered import anchor_resize

        self.torch = torch
        self._apply_pixel_budget = _apply_pixel_budget
        self._anchor_resize = anchor_resize
        self._pixel_budget = (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)

        self.renderer = FrankaRobotRenderer(urdf_path, arm=arm)
        _log.info(f"Loading Qwen3-VL embedder from {embedder_path} ...")
        self.embedder = Qwen3VLEmbedder(embedder_path, torch_dtype=torch.bfloat16)
        _log.info(f"Loading GWM checkpoint {ckpt_path} ...")
        model, ckpt = load_canonical_like_planner(ckpt_path)
        # The GWM head is the only weight on this server that is NOT already
        # bf16 -- the Qwen embedder is loaded bf16 above and is ~16 GB of the
        # ~20 GB total. fp32 is the numeric path every droid-sim result was
        # produced on, so it stays the default; bf16 halves the head's ~1.4 GB.
        self.head_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[head_dtype]
        self.gwm = model.to(self.head_dtype).eval().to(self.embedder.model.device)
        _log.info(f"GWM head in {head_dtype}")
        _log.info(f"GWM ready: step={ckpt.get('step')} params on {self.embedder.model.device}")
        self._task_cache: dict = {}

    # ---------------------------------------------------------------- frames

    def _condition_inputs(self, frames_u8: np.ndarray) -> dict:
        """(6, H, W, 3) uint8 -> budgeted Qwen video inputs on the model device.

        Training parity: anchor-resize to 624x352 (D-29), then the qwen_rat
        pixel budget pins the (3,18,30) grid (ADR-0019).
        """
        from PIL import Image

        t = self.torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 255.0
        t = self._anchor_resize(t)
        pil = [Image.fromarray(
            (f.permute(1, 2, 0).clamp(0, 1) * 255).round().to(self.torch.uint8).numpy()
        ) for f in t]
        conversation = self.embedder.format_model_input(video=pil)
        self._apply_pixel_budget(conversation, *self._pixel_budget)
        processed = self.embedder._preprocess_inputs([conversation])
        return {k: v.to(self.embedder.model.device) for k, v in processed.items()
                if isinstance(v, self.torch.Tensor)}

    def _task_embedding(self, instruction: str, rgb: np.ndarray, mode: str,
                        sys_extra: str | None = None):
        from PIL import Image

        sys_instruction = TEXT_INSTRUCTION if not sys_extra \
            else f"{TEXT_INSTRUCTION} {sys_extra}"

        # The FRAME is part of the key, not just the text. With
        # task_image="current" this embedding is computed from the instruction
        # AND the scene photo, so keying on (instruction, mode) alone returns a
        # embedding built from a DIFFERENT scene the second time an instruction
        # is repeated. This server outlives many scenes -- it is started once
        # and answers every turn of a session -- and instructions repeat
        # constantly ("pick up the tomato" ran six times on 2026-08-19), so
        # every repeat after the first was scored against the first scene's
        # photo. Silent, and it only ever makes a repeat look more like its
        # predecessor.
        #
        # Hashing 2.7 MB costs ~2 ms against a ~150 ms embed, and the cache
        # still does its job: within one scene, all 16 candidates and the
        # empty-instruction prior share one embedding.
        key = (instruction, sys_instruction, mode,
               hashlib.sha1(np.ascontiguousarray(rgb)).hexdigest()
               if mode == "current" else "")
        if key in self._task_cache:
            return self._task_cache[key]
        images = [Image.fromarray(rgb)] if mode == "current" else None
        emb = self.embedder.process([
            {"text": instruction, "image": images, "instruction": sys_instruction}
        ])[0]
        if len(self._task_cache) >= TASK_CACHE_MAX:
            self._task_cache.pop(next(iter(self._task_cache)))
        self._task_cache[key] = emb
        return emb

    # ----------------------------------------------------------------- score

    def score(self, req: ScoreRequest, rgb: np.ndarray) -> tuple[list[float], dict]:
        torch = self.torch
        h, w = rgb.shape[:2]
        K = np.asarray(req.intrinsics, dtype=np.float64)
        c2w = np.asarray(req.world_from_cam, dtype=np.float64)
        task_emb = self._task_embedding(req.instruction, rgb, req.task_image,
                                        req.text_instruction_extra)

        # The instruction-independent part of every candidate's score, measured
        # against the SAME video embedding. Free: one extra task embedding per
        # request (`_task_cache` keys on the frame, so it is shared by every
        # candidate of this scene), then one cosine each. Returned always,
        # used only if the
        # caller asks -- it is the number that separates "the model grounded
        # this" from "this candidate scores high whatever you ask for".
        prior_emb = self._task_embedding("", rgb, req.task_image,
                                         req.text_instruction_extra)

        scores, priors, per_cand = [], [], []
        dump_dir = Path(req.dump_dir) if req.dump_dir else None
        if dump_dir:
            dump_dir.mkdir(parents=True, exist_ok=True)

        for ci, cand in enumerate(req.candidates):
            t0 = time.perf_counter()
            qpos, t, grip = candidate_timeline(cand)
            idxs = sample_rat_times(req.rat_scale, t)
            renders = self.renderer.render(qpos[idxs[1:]], grip[idxs[1:]], K, c2w, width=w, height=h)
            frames = np.concatenate([rgb[None], renders], axis=0)
            render_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            with torch.no_grad():
                proc = self._condition_inputs(frames)
                video_embeds, deepstack = self.embedder.encode_video_to_latent(**proc)
                encoded = torch.cat([*deepstack, video_embeds], dim=0)
                if encoded.shape[0] != self.gwm.positions.shape[0]:
                    grid = proc.get("video_grid_thw")
                    raise RuntimeError(
                        f"visual token count {tuple(encoded.shape)} != canonical "
                        f"{tuple(self.gwm.positions.shape)} (grid {grid}) — pixel budget broke"
                    )
                pred = self.gwm(encoded[None].to(self.head_dtype))[0].to(torch.bfloat16)
                f1, f2, f3, latent = pred.chunk(4, dim=0)
                outputs = self.embedder.embed_video_latent(latent, [f1, f2, f3], **proc)
                vemb = self.embedder.pooling_video_latent(outputs)[0]
                _v = vemb.float().unsqueeze(0)
                score = torch.nn.functional.cosine_similarity(
                    task_emb.float().unsqueeze(0), _v).item()
                prior = torch.nn.functional.cosine_similarity(
                    prior_emb.float().unsqueeze(0), _v).item()
            embed_ms = (time.perf_counter() - t1) * 1000

            scores.append(score)
            priors.append(prior)
            per_cand.append({
                "times": [round(float(t[i]), 2) for i in idxs],
                "close_t": cand.grasp_close_t,
                "render_ms": round(render_ms, 1),
                "embed_ms": round(embed_ms, 1),
            })
            if dump_dir:
                from PIL import Image

                small = self.torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
                small = self._anchor_resize(small)
                arr = (small.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(self.torch.uint8).numpy()
                n, sh, sw, _ = arr.shape
                strip = arr.transpose(1, 0, 2, 3).reshape(sh, n * sw, 3)
                Image.fromarray(strip).save(dump_dir / f"cand{ci:02d}_s{score:+.4f}.png")

        stats = {
            "backend": "gwm",
            "priors": priors,
            "rat_scale": req.rat_scale,
            "task_image": req.task_image,
            "per_candidate": per_cand,
        }
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
    ap.add_argument("--ckpt", help="GWM canonical checkpoint (gwm backend)")
    ap.add_argument("--embedder", default="Qwen/Qwen3-VL-Embedding-8B")
    ap.add_argument("--head-dtype", default="fp32", choices=["fp32", "bf16"],
                    help="dtype for the GWM head. fp32 (default) is the numeric path "
                         "every droid-sim result was produced on; bf16 saves ~0.7 GB of "
                         "the server's ~20 GB, almost all of which is the bf16 Qwen "
                         "embedder and not reachable this way")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8901)
    args = ap.parse_args()

    if args.backend == "dummy":
        backend = DummyBackend(args.urdf, args.arm)
    else:
        if not args.ckpt:
            ap.error("--backend gwm requires --ckpt")
        backend = GwmBackend(args.urdf, args.arm, args.ckpt, args.embedder,
                             head_dtype=args.head_dtype)
    uvicorn.run(build_app(backend), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
