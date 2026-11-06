"""
GWM candidate score with gradients from the 6 input frames to the cosine score.

Mirrors ``GWMBasedPlanner.get_video_embedding`` + the planner's cosine, with three
replacements that keep the graph intact:

- ``torch_video_preprocess``: the frames' preprocessing in float torch ops instead of
  tensor -> PIL -> ``process_vision_info`` -> processor. The path it reproduces:
  PIL bicubic (a = -0.5) resize 224x448 -> 256x448 (``fetch_image`` with factor 64),
  torchvision bicubic antialias resize -> 288x480 (``fetch_video`` min-token budget),
  rescale 1/255, normalize mean = std = 0.5, temporal (2) and spatial (16, merge 2)
  patchify exactly as ``Qwen3VLVideoProcessor._preprocess``. No uint8 rounding.
- the model calls without ``@torch.no_grad`` (vision tower, GWM, LLM readout with
  deepstack, out-of-place ``masked_scatter``), weights frozen.
- a batch of candidates per call (identical token layout for every candidate).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from gwm_wiser.planner.retrieval import RetrievalBasedPlanner
from gwm_wiser.utils.gwm_data import tensor_images_to_pil

FRAMES, H_IN, W_IN = 6, 224, 448
H_PIL, H_VID, W_VID = 256, 288, 480


def _pil_bicubic(x):
    a, x = -0.5, abs(x)
    if x < 1:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


def pil_resample_matrix(n_in, n_out):
    """(n_out, n_in) weights of PIL's ``precompute_coeffs`` for BICUBIC (float, unrounded)."""
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = 2.0 * fs
    M = np.zeros((n_out, n_in))
    for xx in range(n_out):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), n_in) - xmin
        k = np.array([_pil_bicubic((x + xmin - center + 0.5) / fs) for x in range(xmax)])
        if k.sum() != 0:
            k = k / k.sum()
        M[xx, xmin:xmin + xmax] = k
    return M


class TorchVideoPreprocess(torch.nn.Module):
    def __init__(self, patch=16, merge=2, temporal=2, device="cuda"):
        super().__init__()
        self.register_buffer("Wv", torch.tensor(pil_resample_matrix(H_IN, H_PIL), dtype=torch.float32, device=device))
        self.patch, self.merge, self.temporal = patch, merge, temporal

    def forward(self, frames):
        """frames (B, T=6, 3, 224, 448) in [0, 1] -> pixel_values_videos (B * 405 * 4, 1536)."""
        B, T, C, H, W = frames.shape
        x = frames.float() * 255.0
        x = torch.einsum("oh,btchw->btcow", self.Wv, x).clamp(0, 255)  # PIL pass (height only)
        x = F.interpolate(x.reshape(B * T, C, H_PIL, W), size=(H_VID, W_VID), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 255)
        x = (x / 255.0 - 0.5) / 0.5
        x = x.view(B, T, C, H_VID, W_VID)
        p, m, tp = self.patch, self.merge, self.temporal
        gt, gh, gw = T // tp, H_VID // p, W_VID // p
        x = x.view(B, gt, tp, C, gh // m, m, p, gw // m, m, p).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return x.reshape(B * gt * gh * gw, C * tp * p * p)


class DiffGWMScorer:
    """Grad-enabled GWM scoring on top of a loaded ``GWMBasedPlanner`` (its embedder + GWM)."""

    def __init__(self, planner):
        self.planner = planner
        self.embedder = planner.embedder
        self.model = planner.embedder.model  # Qwen3VLForEmbedding
        self.gwm = planner.gwm_model
        self.dtype = planner.dtype
        for mod in (self.model, self.gwm):
            for prm in mod.parameters():
                prm.requires_grad_(False)
        self.device = self.model.device
        self.pre = TorchVideoPreprocess(device=self.device)
        blank = [np.zeros((H_IN, W_IN, 3), np.uint8)] * FRAMES
        from PIL import Image

        tmpl = self.processor_inputs([[Image.fromarray(b) for b in blank]])
        self.input_ids = tmpl["input_ids"]
        self.attention_mask = tmpl["attention_mask"]
        self.grid = tmpl["video_grid_thw"]
        self.tokens_per_video = int(self.grid.prod(-1).item()) // 4

    # ── reference (processor) path ───────────────────────────────────────
    def processor_inputs(self, videos_pil):
        convs = [self.embedder.format_model_input(video=v) for v in videos_pil]
        out = self.embedder._preprocess_inputs(convs)
        return {k: v.to(self.device) for k, v in out.items()}

    def processor_pixels(self, frames):
        """frames (B, 6, 3, H, W) in [0, 1] -> the processor's pixel_values_videos (same as today)."""
        videos = [tensor_images_to_pil(f.detach().cpu().float()) for f in frames]
        return self.processor_inputs(videos)["pixel_values_videos"]

    # ── task embeddings (no gradient needed) ─────────────────────────────
    @torch.no_grad()
    def task_embeddings(self, task_prompt, first_img, cur_img):
        p = self.planner
        g = p._get_task_embedding(RetrievalBasedPlanner._parse_grasp_prompt(task_prompt), first_img, cur_img)
        pl = p._get_task_embedding(RetrievalBasedPlanner._parse_place_prompt(task_prompt), first_img, cur_img)
        return g, pl

    # ── grad-enabled embedding ───────────────────────────────────────────
    def video_embedding(self, pixel_values_videos, B):
        """pixel_values_videos for B candidates -> pooled, L2-normalised embeddings (B, 4096)."""
        m = self.model.model
        grid = self.grid.repeat(B, 1)
        pv = pixel_values_videos.to(m.visual.dtype)
        video_embeds, deepstack = m.get_video_features(pv, grid)
        n = self.tokens_per_video
        ve = torch.cat(video_embeds, 0).view(B, n, -1)
        ds = [d.view(B, n, -1) for d in deepstack]
        encoded = torch.cat(ds + [ve], dim=1)  # (B, 4 * n, 4096), order as encode_trajectory
        pred = self.gwm(encoded.to(self.dtype))
        f1, f2, f3, latent = pred.to(self.model.dtype).chunk(4, dim=1)
        input_ids = self.input_ids.repeat(B, 1)
        attn = self.attention_mask.repeat(B, 1)
        inputs_embeds = m.get_input_embeddings()(input_ids)
        lat = latent.reshape(B * n, -1).to(inputs_embeds.dtype)
        _, video_mask = m.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=lat)
        enc = dict(input_ids=input_ids, attention_mask=attn, inputs_embeds=inputs_embeds,
                   video_grid_thw=grid, visual_pos_masks=video_mask[..., 0], video_mask=video_mask)
        out = self.model.embed_video_latent(
            lat, [f.reshape(B * n, -1).to(inputs_embeds.dtype) for f in (f1, f2, f3)], enc)
        emb = self.embedder._pooling_last(out.last_hidden_state, attn)
        return F.normalize(emb.float(), dim=-1)

    def score(self, frames, task_emb, torch_preprocess=True):
        """frames (B, 6, 3, H, W) in [0, 1] -> cosine with task_emb (B,), differentiable in frames."""
        B = frames.shape[0]
        pv = self.pre(frames) if torch_preprocess else self.processor_pixels(frames)
        v = self.video_embedding(pv, B)
        t = F.normalize(task_emb.float(), dim=-1)
        return v @ t
