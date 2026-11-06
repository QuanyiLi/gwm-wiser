"""
Small state-only diffusion policy (Diffusion-Policy-style conditional 1-D temporal
U-Net) used as a language-free proposal generator for GWM.

Input: the last ``obs_horizon`` full low-dimensional states (see ``state_replay``).
Output: an action chunk of ``pred_horizon`` steps in the demos' ``pd_joint_pos``
action space (7 arm joints + normalised gripper).

    python -m gwm_wiser.local_eval.state_dp train --data /path/state_demos --out /path/run
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler


# ── network ──────────────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, cin, cout, k, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(n_groups, cout), nn.Mish()
        )

    def forward(self, x):
        return self.block(x)


class CondResBlock1D(nn.Module):
    """Residual block with FiLM conditioning (scale and bias)."""

    def __init__(self, cin, cout, cond_dim, k=5, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Conv1dBlock(cin, cout, k, n_groups), Conv1dBlock(cout, cout, k, n_groups)]
        )
        self.cout = cout
        self.cond = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        out = self.blocks[0](x)
        scale, bias = self.cond(c).unsqueeze(-1).chunk(2, dim=1)
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.res(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim, global_cond_dim, step_emb_dim=128,
                 down_dims=(64, 128, 256), kernel_size=5, n_groups=8):
        super().__init__()
        self.step_emb = nn.Sequential(
            SinusoidalPosEmb(step_emb_dim), nn.Linear(step_emb_dim, step_emb_dim * 4),
            nn.Mish(), nn.Linear(step_emb_dim * 4, step_emb_dim),
        )
        cond_dim = step_emb_dim + global_cond_dim
        dims = [action_dim] + list(down_dims)
        pairs = list(zip(dims[:-1], dims[1:]))
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(pairs):
            last = i == len(pairs) - 1
            self.down.append(nn.ModuleList([
                CondResBlock1D(a, b, cond_dim, kernel_size, n_groups),
                CondResBlock1D(b, b, cond_dim, kernel_size, n_groups),
                nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity(),
            ]))
        mid = dims[-1]
        self.mid = nn.ModuleList([
            CondResBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
            CondResBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
        ])
        self.up = nn.ModuleList()
        for a, b in reversed(pairs[1:]):  # one upsample per downsample
            self.up.append(nn.ModuleList([
                CondResBlock1D(b * 2, a, cond_dim, kernel_size, n_groups),
                CondResBlock1D(a, a, cond_dim, kernel_size, n_groups),
                nn.ConvTranspose1d(a, a, 4, 2, 1),
            ]))
        self.final = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, x, t, global_cond):
        # x: (B, T, A) -> (B, A, T)
        x = x.transpose(1, 2)
        c = torch.cat([self.step_emb(t), global_cond], dim=-1)
        skips = []
        for r1, r2, down in self.down:
            x = r2(r1(x, c), c)
            skips.append(x)
            x = down(x)
        for m in self.mid:
            x = m(x, c)
        for r1, r2, up in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = r2(r1(x, c), c)
            x = up(x)
        return self.final(x).transpose(1, 2)


# ── normalisation ────────────────────────────────────────────────────────────
class MinMax:
    """Per-dimension affine map to [-1, 1]; constant dimensions map to 0."""

    def __init__(self, lo, hi):
        self.lo = torch.as_tensor(lo, dtype=torch.float32)
        self.hi = torch.as_tensor(hi, dtype=torch.float32)
        rng = self.hi - self.lo
        self.const = rng < 1e-6
        self.scale = torch.where(self.const, torch.ones_like(rng), rng)

    def to(self, device):
        for k in ("lo", "hi", "scale", "const"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def norm(self, x):
        y = 2 * (x - self.lo) / self.scale - 1
        return torch.where(self.const, torch.zeros_like(y), y)

    def unnorm(self, y):
        x = (y + 1) / 2 * self.scale + self.lo
        return torch.where(self.const, self.lo.expand_as(x), x)

    def state_dict(self):
        return dict(lo=self.lo.cpu(), hi=self.hi.cpu())


# ── policy wrapper ───────────────────────────────────────────────────────────
class StateDiffusionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, obs_horizon=2, pred_horizon=64,
                 down_dims=(64, 128, 256), train_steps=100, infer_steps=16):
        super().__init__()
        self.cfg = dict(state_dim=state_dim, action_dim=action_dim, obs_horizon=obs_horizon,
                        pred_horizon=pred_horizon, down_dims=list(down_dims),
                        train_steps=train_steps, infer_steps=infer_steps)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.net = ConditionalUnet1D(action_dim, state_dim * obs_horizon, down_dims=down_dims)
        sched = dict(num_train_timesteps=train_steps, beta_schedule="squaredcos_cap_v2",
                     clip_sample=True, prediction_type="epsilon")
        self.train_sched = DDPMScheduler(**sched)
        self.infer_sched = DDIMScheduler(**sched)
        self.infer_steps = infer_steps
        self.state_norm = None
        self.action_norm = None

    def set_normalizers(self, state_norm, action_norm):
        self.state_norm, self.action_norm = state_norm, action_norm

    def loss(self, obs, actions):
        """obs (B, To, S) and actions (B, Tp, A), both normalised."""
        noise = torch.randn_like(actions)
        t = torch.randint(0, self.train_sched.config.num_train_timesteps,
                          (actions.shape[0],), device=actions.device)
        noisy = self.train_sched.add_noise(actions, noise, t)
        pred = self.net(noisy, t, obs.flatten(1))
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, obs_raw, num_samples=1, generator=None, infer_steps=None):
        """obs_raw (B, To, S) unnormalised -> actions (B, num_samples, Tp, A) unnormalised."""
        B = obs_raw.shape[0]
        obs = self.state_norm.norm(obs_raw).flatten(1)
        obs = obs.repeat_interleave(num_samples, dim=0)
        x = torch.randn((B * num_samples, self.pred_horizon, self.action_dim),
                        device=obs.device, generator=generator)
        self.infer_sched.set_timesteps(infer_steps or self.infer_steps)
        for t in self.infer_sched.timesteps:
            eps = self.net(x, t.expand(x.shape[0]).to(obs.device), obs)
            x = self.infer_sched.step(eps, t, x).prev_sample
        act = self.action_norm.unnorm(x.clamp(-1, 1))
        return act.view(B, num_samples, self.pred_horizon, self.action_dim)

    def save(self, path, extra=None):
        torch.save(dict(cfg=self.cfg, model=self.state_dict(),
                        state_norm=self.state_norm.state_dict(),
                        action_norm=self.action_norm.state_dict(), extra=extra or {}), path)

    @classmethod
    def load(cls, path, device="cuda"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        pol = cls(**ck["cfg"])
        pol.load_state_dict(ck["model"])
        pol.set_normalizers(MinMax(**ck["state_norm"]).to(device), MinMax(**ck["action_norm"]).to(device))
        return pol.to(device).eval()


# ── data ─────────────────────────────────────────────────────────────────────
def load_state_demos(data_dir, mode="train"):
    files = sorted(glob.glob(os.path.join(data_dir, f"config_*_{mode}.npz")))
    states, actions = [], []
    for f in files:
        d = np.load(f)
        states.append(d["states"][:, :-1])  # (E, T, S): state before action t
        actions.append(d["actions"])
    return np.concatenate(states), np.concatenate(actions), len(files)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, obs_horizon, pred_horizon):
        self.s, self.a = torch.from_numpy(states), torch.from_numpy(actions)
        self.To, self.Tp = obs_horizon, pred_horizon
        E, T = self.a.shape[:2]
        self.T = T
        self.index = [(e, t) for e in range(E) for t in range(T)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        oi = torch.clamp(torch.arange(t - self.To + 1, t + 1), 0, self.T - 1)
        ai = torch.clamp(torch.arange(t, t + self.Tp), 0, self.T - 1)
        return self.s[e, oi], self.a[e, ai]


def train(args):
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    states, actions, n_cfg = load_state_demos(args.data)
    print(f"demos: {states.shape[0]} episodes from {n_cfg} configs, state {states.shape}, action {actions.shape}")
    device = "cuda"
    pol = StateDiffusionPolicy(states.shape[-1], actions.shape[-1], args.obs_horizon, args.pred_horizon,
                               tuple(args.down_dims), args.train_steps, args.infer_steps)
    s_flat, a_flat = states.reshape(-1, states.shape[-1]), actions.reshape(-1, actions.shape[-1])
    pol.set_normalizers(MinMax(s_flat.min(0), s_flat.max(0)).to(device),
                        MinMax(a_flat.min(0), a_flat.max(0)).to(device))
    pol = pol.to(device)
    n_params = sum(p.numel() for p in pol.net.parameters())
    print(f"parameters: {n_params / 1e6:.2f} M")
    ema = torch.optim.swa_utils.AveragedModel(
        pol, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema))

    ds = WindowDataset(states, actions, args.obs_horizon, args.pred_horizon)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(pol.net.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    step, t0, log = 0, time.time(), []
    while step < args.steps:
        for obs, act in dl:
            obs, act = obs.to(device), act.to(device)
            loss = pol.loss(pol.state_norm.norm(obs), pol.action_norm.norm(act))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.net.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update_parameters(pol)
            step += 1
            if step % 500 == 0:
                rec = dict(step=step, loss=loss.item(), lr=sched.get_last_lr()[0], elapsed=time.time() - t0)
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                ema_pol = ema.module
                ema_pol.set_normalizers(pol.state_norm, pol.action_norm)
                ema_pol.save(os.path.join(args.out, f"ema_{step:06d}.pt"),
                             extra=dict(step=step, params=n_params, train_seconds=time.time() - t0))
            if step >= args.steps:
                break
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(dict(args=vars(args), params=n_params, train_seconds=time.time() - t0, log=log), f, indent=2)
    print(f"done: {step} steps in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--obs_horizon", type=int, default=2)
    tr.add_argument("--pred_horizon", type=int, default=64)
    tr.add_argument("--down_dims", type=int, nargs="+", default=[64, 128, 256])
    tr.add_argument("--train_steps", type=int, default=100)
    tr.add_argument("--infer_steps", type=int, default=16)
    tr.add_argument("--batch_size", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--ema", type=float, default=0.999)
    tr.add_argument("--steps", type=int, default=60000)
    tr.add_argument("--save_every", type=int, default=10000)
    tr.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
