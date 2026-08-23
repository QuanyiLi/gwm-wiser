"""V-JEPA 2-AC (Assran et al. 2025) loaded from a local checkpoint.

Mirrors `src/hub/backbones._make_vjepa2_ac_model` in facebookresearch/vjepa2
(the hub loader at the pinned commit points at a localhost test URL, so the
checkpoint is fetched by hand to checkpoints/vjepa2-ac-vitg.pt).

Conventions (from app/vjepa_droid/droid.py, notebooks/utils/mpc_utils.py):
  state  s_t = [x, y, z, roll, pitch, yaw, gripper]   absolute EEF pose in the
               robot base frame (DROID `cartesian_position`, scipy extrinsic
               "xyz" Euler) + gripper closedness in [0, 1]
  action a_t = [dx, dy, dz, d_euler, d_gripper]       s_{t+1} - s_t, where the
               rotation delta is euler_xyz(R_{t+1} R_t^T), i.e. applied in the
               base frame; one action per token-frame, 4/15 s apart in training
  frame  one RGB image -> one tubelet (the frame is duplicated along time) ->
               16x16 = 256 patch tokens of dim 1408, layer-normed per token
  energy mean |z_pred - z_goal| over tokens and channels (the repo's `l1`)

Predictor call: predictor(z[B, T*256, D], actions[B, T, 7], states[B, T, 7])
returns [B, T*256, D]; with the frame-causal mask the last 256 tokens are the
prediction for frame T given frames 0..T-1 and the action taken at frame T-1.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
VJEPA2_ROOT = os.path.join(os.path.dirname(_HERE), "vjepa2")
DEFAULT_CKPT = os.path.join(os.path.dirname(_HERE), "checkpoints", "vjepa2-ac-vitg.pt")

if VJEPA2_ROOT not in sys.path:
    sys.path.insert(0, VJEPA2_ROOT)

from src.models import ac_predictor as vit_ac_predictor  # noqa: E402
from src.models import vision_transformer as vit_encoder  # noqa: E402

from .preprocess import to_model_input  # noqa: E402

PATCH = 16
IMG = 256
TOKENS_PER_FRAME = (IMG // PATCH) ** 2  # 256
MAX_FRAMES = 64 // 2  # predictor attention mask is built for num_frames/tubelet = 32 frames
STATE_DIM = 7


def _clean_keys(state_dict):
    out = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        out[k] = v
    return out


def l1_energy(z, z_goal):
    """mean |z - z_goal| over the last two dims; z [..., N, D], z_goal broadcastable."""
    return torch.mean(torch.abs(z.float() - z_goal.float()), dim=(-2, -1))


class VJEPA2AC:
    def __init__(self, ckpt=DEFAULT_CKPT, device="cuda", dtype=torch.bfloat16, encoder_key="encoder"):
        self.device = torch.device(device)
        self.dtype = dtype
        enc_kwargs = dict(
            patch_size=PATCH,
            img_size=(IMG, IMG),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )
        self.encoder = vit_encoder.vit_giant_xformers(**enc_kwargs)
        self.predictor = vit_ac_predictor.vit_ac_predictor(
            img_size=(IMG, IMG),
            patch_size=PATCH,
            num_frames=64,
            tubelet_size=2,
            embed_dim=self.encoder.embed_dim,
        )
        sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
        enc_missing = self.encoder.load_state_dict(_clean_keys(sd[encoder_key]), strict=False)
        # the checkpoint carries a sincos pos_embed the RoPE encoder does not use
        unexpected = [k for k in enc_missing.unexpected_keys if k != "pos_embed"]
        assert not enc_missing.missing_keys and not unexpected, (enc_missing.missing_keys, unexpected)
        self.predictor.load_state_dict(_clean_keys(sd["predictor"]), strict=True)
        self.ckpt_meta = {k: sd[k] for k in ("epoch", "loss", "batch_size", "world_size", "lr") if k in sd}
        del sd
        self.encoder.to(self.device).eval()
        self.predictor.to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.encoder.embed_dim

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, frames, crop_mode="full_aa", batch_size=8):
        """uint8 frames [T, H, W, 3] (or one [H, W, 3]) -> layer-normed tokens [T, 256, D] (fp32)."""
        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[None]
        x = to_model_input(frames, mode=crop_mode)  # [T, 3, 256, 256]
        outs = []
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size].to(self.device, non_blocking=True)
            # one image -> one 2-frame tubelet: [B, C, 2, H, W]
            clip = xb.unsqueeze(2).repeat(1, 1, 2, 1, 1)
            with torch.autocast("cuda", dtype=self.dtype):
                h = self.encoder(clip)  # [B, 256, D]
            h = F.layer_norm(h.float(), (h.size(-1),))
            outs.append(h)
        return torch.cat(outs, dim=0)

    # ----------------------------------------------------------------- predict
    @torch.no_grad()
    def predict_next(self, z_ctx, actions, states):
        """z_ctx [B, T, 256, D], actions [B, T, 7], states [B, T, 7] -> z_{T} [B, 256, D].

        Tokens for frames 0..T-1 plus the action taken at each frame; the
        returned tokens are the prediction for frame T.
        """
        B, T, N, D = z_ctx.shape
        assert T <= MAX_FRAMES, f"context of {T} frames exceeds the predictor's {MAX_FRAMES}-frame mask"
        z = z_ctx.reshape(B, T * N, D)
        with torch.autocast("cuda", dtype=self.dtype):
            out = self.predictor(
                z.to(self.device),
                actions.to(self.device, torch.float32),
                states.to(self.device, torch.float32),
            )
        out = out[:, -N:].float()
        return F.layer_norm(out, (out.size(-1),))

    @torch.no_grad()
    def rollout(self, z0, states, actions, context_window=MAX_FRAMES, teacher_z=None):
        """Autoregressive open-loop rollout of one action sequence.

        z0       [256, D]     encoded current frame (frame 0)
        states   [T+1, 7]     absolute EEF states s_0..s_T (s_T is unused by the
                              predictor but kept for a uniform interface)
        actions  [T, 7]       a_t = s_{t+1} - s_t
        context_window        max frames kept as context (<= 32); the oldest
                              frames are dropped once the window is full
        teacher_z [T+1,256,D] optional encoded *observed* frames; when given,
                              the context uses them instead of the model's own
                              predictions (teacher forcing, one-step errors)
        returns  z_pred [T, 256, D]  predicted frames 1..T
        """
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        T = actions.shape[0]
        assert states.shape[0] >= T, (states.shape, actions.shape)
        K = int(min(context_window, MAX_FRAMES))
        frames = [z0.to(self.device)]  # context frames (observed or predicted)
        preds = []
        for t in range(T):
            lo = max(0, t + 1 - K)
            z_ctx = torch.stack(frames[lo : t + 1], dim=0)[None]  # [1, k, 256, D]
            a_ctx = actions[lo : t + 1][None]
            s_ctx = states[lo : t + 1][None]
            z_next = self.predict_next(z_ctx, a_ctx, s_ctx)[0]
            preds.append(z_next)
            frames.append(teacher_z[t + 1].to(self.device) if teacher_z is not None else z_next)
        return torch.stack(preds, dim=0)

    @torch.no_grad()
    def rollout_batch(self, z0, states, actions, context_window=MAX_FRAMES):
        """Same as `rollout` for B candidates of equal length: z0 [B,256,D], states [B,T+1,7], actions [B,T,7]."""
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        B, T = actions.shape[:2]
        K = int(min(context_window, MAX_FRAMES))
        frames = [z0.to(self.device)]
        preds = []
        for t in range(T):
            lo = max(0, t + 1 - K)
            z_ctx = torch.stack(frames[lo : t + 1], dim=1)  # [B, k, 256, D]
            z_next = self.predict_next(z_ctx, actions[:, lo : t + 1], states[:, lo : t + 1])
            preds.append(z_next)
            frames.append(z_next)
        return torch.stack(preds, dim=1)  # [B, T, 256, D]
