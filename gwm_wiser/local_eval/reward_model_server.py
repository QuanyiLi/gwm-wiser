"""
HTTP scoring server for a pretrained Robometer reward model (LeRobot integration).

Runs in its own environment (LeRobot main with the ``robometer`` extra); it imports
nothing from this repo. One request scores a batch of candidate frame stacks against
their task strings with the model's documented recipe (LeRobot's
``RobometerEncoderProcessorStep``: progress prompt, multi-image input with a progress
token after each frame) and returns, per candidate:

    progress  last-frame progress (expectation over the discrete progress bins, in [0, 1]),
              the model's documented trajectory-level reward (``reward_output="progress"``)
    success   last-frame success probability (sigmoid of the success logit), logged only

    python reward_model_server.py --pretrained lerobot/Robometer-4B --port 8950
    POST /score  body = np.savez(frames=uint8 (N, T, H, W, 3), tasks=str (N,)); JSON response
"""

import argparse
import io

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request

from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.robometer.modeling_robometer import (
    ROBOMETER_FEATURE_PREFIX,
    ROBOMETER_INPUT_KEYS,
    decode_progress_outputs,
)
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep


def build(pretrained: str, device: str):
    cfg = RobometerConfig(pretrained_path=pretrained, device=device, reward_output="progress")
    model = RobometerRewardModel.from_pretrained(cfg.pretrained_path, config=cfg).to(device).eval()
    encoder = RobometerEncoderProcessorStep(
        base_model_id=cfg.base_model_id,
        use_multi_image=cfg.use_multi_image,
        use_per_frame_progress_token=cfg.use_per_frame_progress_token,
        max_frames=cfg.max_frames,
    )
    return cfg, model, encoder


@torch.no_grad()
def score(cfg, model, encoder, frames: np.ndarray, tasks: list[str]) -> dict:
    """Same steps as ``RobometerRewardModel.compute_reward``, returning both heads."""
    encoded = encoder.encode_samples([(f, t) for f, t in zip(frames, tasks)])
    batch = {f"{ROBOMETER_FEATURE_PREFIX}{k}": v for k, v in encoded.items()}
    inputs = {k: batch[f"{ROBOMETER_FEATURE_PREFIX}{k}"] for k in ROBOMETER_INPUT_KEYS
              if f"{ROBOMETER_FEATURE_PREFIX}{k}" in batch}
    device = next(model.model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    model.eval()
    progress_logits, success_logits = model._compute_rbm_logits(inputs)
    decoded = decode_progress_outputs(progress_logits, success_logits,
                                      is_discrete_mode=cfg.use_discrete_progress)
    progress = [float(torch.as_tensor(s, dtype=torch.float32)[-1].clamp(0.0, 1.0)) for s in decoded["progress_pred"]]
    success = [float(torch.as_tensor(s, dtype=torch.float32)[-1]) for s in decoded["success_probs"]]
    return {"progress": progress, "success": success,
            "progress_per_frame": [np.asarray(s, dtype=np.float32).tolist() for s in decoded["progress_pred"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="lerobot/Robometer-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=8950)
    ap.add_argument("--max_batch", type=int, default=12)
    args = ap.parse_args()
    cfg, model, encoder = build(args.pretrained, args.device)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "pretrained": args.pretrained, "max_frames": cfg.max_frames,
                "reward_output": cfg.reward_output}

    @app.post("/score")
    async def score_route(request: Request):
        req = np.load(io.BytesIO(await request.body()), allow_pickle=False)
        frames, tasks = np.asarray(req["frames"], dtype=np.uint8), [str(t) for t in req["tasks"]]
        out = {"progress": [], "success": [], "progress_per_frame": []}
        for i in range(0, len(tasks), args.max_batch):
            r = score(cfg, model, encoder, frames[i:i + args.max_batch], tasks[i:i + args.max_batch])
            for k in out:
                out[k] += r[k]
        return out

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
