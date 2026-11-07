"""
Feasibility probe: can the GWM score of a WISER candidate be differentiated w.r.t. its
actions? Offline checks on replans dumped by ``state_diffusion_proposals --dump_replans``.

Candidate frames are the planner's: frame 0 = current RGB, frames 1-5 = robot-only renders of the
action targets chunk[10, 22, 34, 46, 58] (keyframes linspace(0, 59, 6), the action that
reaches keyframe k is chunk[k - 1]), with ``RobotRenderer.action_to_qpos`` semantics.

Variants of the frame source:
  sapien  - RobotRenderer (the original pipeline; no gradient)
  soft    - SoftRobotRenderer (pure-PyTorch soft rasterizer; differentiable)
  st      - straight-through: forward = sapien, gradient = soft

    python -m gwm_wiser.local_eval.differentiable_render_probe c1 --replans DIR --out DIR
    (stages: c1 c2 c3 c4 c5 cost)
"""

import argparse
import glob
import json
import math
import os
import time
import zlib

import numpy as np
import torch

from gwm_wiser import PROJECT_ROOT

ACT_IDX = [10, 22, 34, 46, 58]  # action targets that reach keyframes 11, 23, 35, 47, 59
EMBEDDER = os.environ.get("GWM_EMBEDDER", "Qwen/Qwen3-VL-Embedding-8B")
GWM_CKPT = os.environ.get("GWM_CKPT", "checkpoints/gwm/checkpoint.pt")


# ── loading ───────────────────────────────────────────────────────────────────
def load_planner(dtype):
    import gwm_wiser.local_eval  # noqa: F401  (checkpoint config alias)
    from gwm_wiser.models.qwen3_vl_embedding import Qwen3VLEmbedder
    from gwm_wiser.planner.gwm import GWMBasedPlanner

    if dtype == torch.float32:  # cuDNN TF32 (default on) makes the Conv3d patch embedding non-smooth
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    emb = Qwen3VLEmbedder(EMBEDDER, torch_dtype=dtype)
    return GWMBasedPlanner(
        dataset_root=os.path.join(PROJECT_ROOT, "gwm_skills/config_0_train/lerobot_data"),
        embedder=emb, gwm_checkpoint_path=GWM_CKPT, k=12, replan_horizon=20,
        video_frame_subsample=6, num_future_frames=60, device="cuda", dtype=dtype)


def load_replans(d, limit=None):
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    out = []
    for f in files[:limit]:
        z = np.load(f, allow_pickle=True)
        r = {k: z[k] for k in z.files}
        r["file"] = os.path.basename(f)
        r["task"] = str(r["task"])
        r["is_grasped"] = bool(r["is_grasped"])
        out.append(r)
    return out


class Renderers:
    """RobotRenderer (+ mask) and the soft renderer with the same SAPIEN camera."""

    def __init__(self, voxel=0.0015, sigma=0.3, cull=False, srgb=True, pixel_offset=0.0, gamma=1e-2):
        from gwm_wiser.local_eval.soft_robot_render import SoftRobotRenderer
        from gwm_wiser.local_eval.traj_tools import KeyframeRenderer

        self.kr = KeyframeRenderer()
        prm = self.kr.renderer.camera.get_params()
        E = prm["extrinsic_cv"][0].cpu().numpy()
        K = prm["intrinsic_cv"][0].cpu().numpy().copy()
        K[0, 2] += pixel_offset
        K[1, 2] += pixel_offset
        self.E, self.K = E, K
        self.soft = SoftRobotRenderer(E, K, voxel=voxel, sigma=sigma, gamma=gamma, cull_backfaces=cull, srgb=srgb)

    @torch.no_grad()
    def sapien(self, qpos):
        """qpos (N, 9) -> (images (N, 3, H, W) in [0, 1], masks (N, H, W) bool)."""
        r = self.kr.renderer
        imgs, masks = [], []
        for q in qpos:
            r.robot.set_qpos(q[None].float().to(r.device))
            r.scene.update_render(update_sensors=True, update_human_render_cameras=False)
            r.camera.capture()
            o = r.camera.get_obs(rgb=True, depth=False, position=False, segmentation=True,
                                 apply_texture_transforms=True)
            m = o["segmentation"][0, ..., 0] > 0
            rgb = o["rgb"][0].clone()
            rgb[~m] = 0
            imgs.append(rgb.permute(2, 0, 1).float().cpu() / 255.0)
            masks.append(m.cpu())
        return torch.stack(imgs), torch.stack(masks)


def actions_to_qpos(a):
    from gwm_wiser.utils.robot_renderer import RobotRenderer

    return RobotRenderer._panda_action_to_qpos(a)


def keyframe_actions(chunks):
    return chunks[:, ACT_IDX]  # (K, 5, 8)


def frames_from(chunks, cur_rgb, R, mode, kf_actions=None):
    """chunks (K, 60, 8) -> frames (K, 6, 3, 224, 448) in [0, 1].

    ``kf_actions`` (K, 5, 8) may carry gradient (soft / st); otherwise taken from chunks.
    """
    K = chunks.shape[0]
    a = keyframe_actions(chunks) if kf_actions is None else kf_actions
    q = actions_to_qpos(a.reshape(K * 5, 8).float())
    cur = torch.from_numpy(cur_rgb).permute(2, 0, 1).float().cuda()[None, None] / 255.0
    if mode == "sapien":
        img = R.sapien(q.detach())[0].cuda()
    elif mode == "soft":
        img = R.soft(q.cuda())
    else:  # straight-through
        soft = R.soft(q.cuda())
        img = R.sapien(q.detach())[0].cuda() + (soft - soft.detach())
    img = img.view(K, 5, 3, img.shape[-2], img.shape[-1])
    return torch.cat([cur.expand(K, 1, -1, -1, -1), img], dim=1)


def original_scores(planner, chunks, cur_rgb, first_rgb, task, R, frames=None):
    """The planner's exact scoring path (RobotRenderer frames, PIL + processor, bf16 planner)."""
    from types import SimpleNamespace

    from PIL import Image

    from gwm_wiser.planner.retrieval import RetrievalBasedPlanner as RP

    cur, first = Image.fromarray(cur_rgb), Image.fromarray(first_rgb)
    g_emb = planner._get_task_embedding(RP._parse_grasp_prompt(task), first, cur)
    p_emb = planner._get_task_embedding(RP._parse_place_prompt(task), first, cur)
    if frames is None:
        frames = frames_from(chunks, cur_rgb, R, "sapien")
    g, p = [], []
    with torch.no_grad():
        for k in range(frames.shape[0]):
            cand = SimpleNamespace(images_main=None, images_wrist=None, images_robot_state=frames[k, :].cpu())
            v = planner.get_video_embedding(cand, cur)
            g.append(torch.nn.functional.cosine_similarity(g_emb[None].float(), v[None].float()).item())
            p.append(torch.nn.functional.cosine_similarity(p_emb[None].float(), v[None].float()).item())
    return np.array(g), np.array(p)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ── C1: preprocessing parity ──────────────────────────────────────────────────
def stage_c1(args):
    from gwm_wiser.local_eval.diff_gwm_score import DiffGWMScorer

    planner = load_planner(torch.bfloat16)
    sc = DiffGWMScorer(planner)
    R = Renderers()
    reps = load_replans(args.replans)[: args.n]
    max_diff, mean_diff, dscore, dself, top1 = [], [], [], [], []
    for r in reps:
        chunks = torch.from_numpy(r["chunks"]).float()
        fr = frames_from(chunks, r["cur_rgb"], R, "sapien")
        with torch.no_grad():
            pv_proc = sc.processor_pixels(fr)
            pv_torch = sc.pre(fr)
            d = (pv_proc.float() - pv_torch.float()).abs()
            max_diff.append(float(d.max()))
            mean_diff.append(float(d.mean()))
            te = stage_score_fn(sc, r)
            s_proc = (sc.video_embedding(pv_proc, fr.shape[0]) @ torch.nn.functional.normalize(te.float(), dim=-1)).cpu().numpy()
            s_torch = (sc.video_embedding(pv_torch, fr.shape[0]) @ torch.nn.functional.normalize(te.float(), dim=-1)).cpu().numpy()
        g, p = original_scores(planner, chunks, r["cur_rgb"], r["first_rgb"], r["task"], R, frames=fr)
        s_orig = p if r["is_grasped"] else g
        dscore.append(np.abs(s_proc - s_torch))
        top1.append(int(np.argmax(s_proc) == np.argmax(s_torch)))
        dself.append(np.abs(s_proc - s_orig))
        print(f"{r['file']}: pv max {max_diff[-1]:.4f} score |d| {dscore[-1].mean():.5f} "
              f"(reimpl vs planner {dself[-1].mean():.6f})", flush=True)
    ds, dsf = np.concatenate(dscore), np.concatenate(dself)
    res = dict(n_sets=len(reps), n_candidates=int(ds.size),
               pixel_values_max_abs_diff=float(np.max(max_diff)),
               pixel_values_max_abs_diff_in_uint8=float(np.max(max_diff)) * 127.5,
               pixel_values_mean_abs_diff=float(np.mean(mean_diff)),
               score_abs_diff_mean=float(ds.mean()), score_abs_diff_max=float(ds.max()),
               top1_agreement=float(np.mean(top1)),
               grad_enabled_reimpl_vs_planner_score_abs_diff_mean=float(dsf.mean()),
               grad_enabled_reimpl_vs_planner_score_abs_diff_max=float(dsf.max()))
    json.dump(res, open(os.path.join(args.out, "c1.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


# ── C2: render parity + score parity ──────────────────────────────────────────
def stage_c2(args):
    reps = load_replans(args.replans)
    rng = np.random.default_rng(0)
    # >= 200 robot states: keyframe targets of random samples
    states = []
    for r in reps:
        k = rng.integers(0, r["chunks"].shape[0])
        states.append(actions_to_qpos(torch.from_numpy(r["chunks"][k, ACT_IDX]).float()))
    states = torch.cat(states)[: max(args.n_states, 200)]
    states = torch.cat([states, states])[: max(args.n_states, 200)] if len(states) < 200 else states
    res = {"n_states": int(states.shape[0]), "settings": {}}
    for voxel, sigma in [(0.0015, 0.3), (0.0015, 0.15), (0.00075, 0.15)]:
        R = Renderers(voxel=voxel, sigma=sigma)
        hard_iou, soft_iou, mad, t_soft = [], [], [], []
        for i in range(0, len(states), 5):
            q = states[i: i + 5]
            img_s, m_s = R.sapien(q)
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                img_d, a_d = R.soft(q.cuda(), return_alpha=True)
            torch.cuda.synchronize(); t_soft.append((time.time() - t0) / len(q))
            for m_d, acc in ((R.soft.last_hard.cpu(), hard_iou), ((a_d > 0.5).cpu(), soft_iou)):
                acc += ((m_s & m_d).sum((1, 2)).float() / (m_s | m_d).sum((1, 2)).float().clamp(min=1)).tolist()
            m_d = (a_d > 0.5).cpu()
            region = (m_s | m_d)[:, None].expand(-1, 3, -1, -1)
            mad += [float((img_s[j] - img_d[j].cpu()).abs()[region[j]].mean()) * 255 for j in range(len(q))]
        key = f"voxel{voxel * 1000:.2f}mm_sigma{sigma}"
        res["settings"][key] = dict(voxel=voxel, sigma=sigma, faces=int(R.soft.mesh.faces.shape[0]),
                                    faces_full=int(R.soft.mesh.n_faces_full),
                                    hard_mask_iou_mean=float(np.mean(hard_iou)), hard_mask_iou_min=float(np.min(hard_iou)),
                                    soft_alpha_gt_0_5_iou_mean=float(np.mean(soft_iou)),
                                    robot_region_mean_abs_pixel_diff_0_255=float(np.mean(mad)),
                                    soft_render_ms_per_frame=1000 * float(np.mean(t_soft)))
        print(key, res["settings"][key], flush=True)
        del R
    res["chosen"] = dict(voxel=args.voxel, sigma=args.sigma, gamma=args.gamma, pixel_offset=0.0, cull=False)
    # FK check vs SAPIEN
    R = Renderers(**res["chosen"])
    tcp_s = np.stack([R.kr.tcp_positions(q[None])[0] for q in states[:50]])
    tcp_d = R.soft.mesh.tcp(states[:50].cuda()).detach().cpu().numpy()
    res["fk_tcp_max_abs_diff_m"] = float(np.abs(tcp_s - tcp_d).max())
    # score parity on >= 50 replans x 16 samples
    planner = load_planner(torch.bfloat16)
    sp, top1, dabs = [], [], []
    for r in reps[: args.n]:
        chunks = torch.from_numpy(r["chunks"]).float()
        fs = frames_from(chunks, r["cur_rgb"], R, "sapien")
        with torch.no_grad():
            fd = frames_from(chunks, r["cur_rgb"], R, "soft").cpu()
        gs, ps = original_scores(planner, chunks, r["cur_rgb"], r["first_rgb"], r["task"], R, frames=fs)
        gd, pd = original_scores(planner, chunks, r["cur_rgb"], r["first_rgb"], r["task"], R, frames=fd)
        a, b = (ps, pd) if r["is_grasped"] else (gs, gd)
        sp.append(spearman(a, b)); top1.append(int(np.argmax(a) == np.argmax(b))); dabs.append(np.abs(a - b).mean())
        print(f"{r['file']}: spearman {sp[-1]:.3f} top1 {top1[-1]} |d| {dabs[-1]:.4f}", flush=True)
    res["score_parity"] = dict(n_sets=len(sp), spearman_mean=float(np.mean(sp)), spearman_median=float(np.median(sp)),
                               top1_agreement=float(np.mean(top1)), mean_abs_score_diff=float(np.mean(dabs)))
    json.dump(res, open(os.path.join(args.out, "c2.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


def chosen_renderer(args):
    c2 = json.load(open(os.path.join(args.out, "c2.json")))
    ch = dict(c2["chosen"])
    if args.sigma_override:
        ch["sigma"] = args.sigma_override
    return Renderers(**ch)


def stage_score_fn(sc, r):
    from PIL import Image

    ge, pe = sc.task_embeddings(r["task"], Image.fromarray(r["first_rgb"]), Image.fromarray(r["cur_rgb"]))
    return pe if r["is_grasped"] else ge


# ── C3: gradient correctness ──────────────────────────────────────────────────
def stage_c3(args):
    from gwm_wiser.local_eval.diff_gwm_score import DiffGWMScorer

    planner = load_planner(torch.float32)
    sc = DiffGWMScorer(planner)
    R = chosen_renderer(args)
    reps = load_replans(args.replans)
    rng = np.random.default_rng(1)
    picks = [reps[i] for i in rng.choice(len(reps), size=args.n, replace=False)]
    res = {"fd_pure_soft": [], "grad_norms": [], "st_vs_original_fd": []}

    def score_of(kf, chunks, r, te, mode):
        fr = frames_from(chunks, r["cur_rgb"], R, mode, kf_actions=kf)
        return sc.score(fr, te)

    for ci, r in enumerate(picks):
        chunks = torch.from_numpy(r["chunks"][:1]).float()  # one candidate per replan
        te = stage_score_fn(sc, r)
        # autograd, pure soft pipeline
        kf = keyframe_actions(chunks).clone().cuda().requires_grad_(True)
        s = score_of(kf, chunks, r, te, "soft")[0]
        (g,) = torch.autograd.grad(s, kf)
        g = g[0].cpu().numpy()
        res["grad_norms"].append(dict(per_keyframe=np.linalg.norm(g, axis=1).tolist(),
                                      arm=float(np.linalg.norm(g[:, :7])), gripper=float(np.linalg.norm(g[:, 7]))))
        # central differences, eps = 1e-3 rad, several coordinates
        coords = [(k, j) for k in range(5) for j in range(8)]
        for (k, j) in [coords[i] for i in rng.choice(len(coords), size=args.coords, replace=False)]:
            ad = float(g[k, j])
            row = dict(set=ci, keyframe=k, joint=j, autograd=ad)
            for eps, tag in ((args.eps, ""), (args.eps / 10, "_small")):
                vals = []
                for sgn in (1, -1):
                    kp = keyframe_actions(chunks).clone().cuda()
                    kp[0, k, j] += sgn * eps
                    with torch.no_grad():
                        vals.append(float(score_of(kp, chunks, r, te, "soft")[0]))
                fd = (vals[0] - vals[1]) / (2 * eps)
                row["fd" + tag] = fd
                row["rel_err" + tag] = abs(ad - fd) / max(abs(ad), abs(fd), 1e-12)
            res["fd_pure_soft"].append(row)
            print(f"c3 set {ci} kf {k} j {j}: ad {ad:+.4e} fd {row['fd']:+.4e} fd_small {row['fd_small']:+.4e}", flush=True)
        # straight-through gradient vs central differences of the ORIGINAL pipeline (fp32, eps 1e-2)
        if ci < args.n_st:
            kf = keyframe_actions(chunks).clone().cuda().requires_grad_(True)
            s = score_of(kf, chunks, r, te, "st")[0]
            (gst,) = torch.autograd.grad(s, kf)
            gst = gst[0, :, :7].cpu().numpy().reshape(-1)
            fd_vec = []
            for k in range(5):
                for j in range(7):
                    vals = []
                    for sgn in (1, -1):
                        kp = keyframe_actions(chunks).clone()
                        kp[0, k, j] += sgn * args.eps_orig
                        with torch.no_grad():
                            fr = frames_from(chunks, r["cur_rgb"], R, "sapien", kf_actions=kp)
                            vals.append(float(sc.score(fr, te, torch_preprocess=False)[0]))
                    fd_vec.append((vals[0] - vals[1]) / (2 * args.eps_orig))
            fd_vec = np.array(fd_vec)
            cos = float(gst @ fd_vec / (np.linalg.norm(gst) * np.linalg.norm(fd_vec) + 1e-12))
            res["st_vs_original_fd"].append(dict(set=ci, cosine=cos, st_norm=float(np.linalg.norm(gst)),
                                                 fd_norm=float(np.linalg.norm(fd_vec))))
            print(f"c3 set {ci}: cos(st grad, original FD) = {cos:.3f}", flush=True)
    rel = np.array([d["rel_err"] for d in res["fd_pure_soft"]])
    rel_s = np.array([d["rel_err_small"] for d in res["fd_pure_soft"]])
    gn = res["grad_norms"]
    res["summary"] = dict(
        n_candidates=len(picks), n_fd_coords=int(rel.size), rel_err_median=float(np.median(rel)),
        rel_err_mean=float(rel.mean()), frac_rel_err_below_0_1=float((rel < 0.1).mean()),
        eps=args.eps, eps_small=args.eps / 10, rel_err_small_median=float(np.median(rel_s)),
        frac_rel_err_small_below_0_1=float((rel_s < 0.1).mean()),
        grad_norm_per_keyframe_mean=np.mean([g["per_keyframe"] for g in gn], 0).tolist(),
        grad_norm_arm_mean=float(np.mean([g["arm"] for g in gn])),
        grad_norm_gripper_mean=float(np.mean([g["gripper"] for g in gn])),
        st_vs_original_fd_cosine_mean=float(np.mean([d["cosine"] for d in res["st_vs_original_fd"]])) if res["st_vs_original_fd"] else None,
        st_vs_original_fd_cosine_all=[round(d["cosine"], 3) for d in res["st_vs_original_fd"]])
    json.dump(res, open(os.path.join(args.out, f"c3{args.tag}.json"), "w"), indent=2)
    print(json.dumps(res["summary"], indent=2))


# ── helpers for C4 / C5 ───────────────────────────────────────────────────────
def smooth_offset(kf_offset, T=60):
    """(K, 5, 7) keyframe arm offsets -> (K, T, 7): 0 at the current state (index -1),
    linear through the 5 action indices, held after the last."""
    xs = np.array([-1] + ACT_IDX, dtype=float)
    K = kf_offset.shape[0]
    pts = torch.cat([torch.zeros(K, 1, 7, device=kf_offset.device, dtype=kf_offset.dtype), kf_offset], 1)
    t = torch.arange(T, dtype=kf_offset.dtype, device=kf_offset.device)
    out = torch.empty(K, T, 7, device=kf_offset.device, dtype=kf_offset.dtype)
    for i in range(T):
        seg = min(np.searchsorted(xs, i, side="right") - 1, len(xs) - 2)
        x0, x1 = xs[seg], xs[seg + 1]
        w = float(np.clip((i - x0) / (x1 - x0), 0, 1))
        out[:, i] = (1 - w) * pts[:, seg] + w * pts[:, seg + 1]
    return out


def event_xy_dist(chunk, mesh, r, slots_cube, slots_placard):
    """Distance (m) from the TCP xy at the stage's event to the instructed target, or None."""
    q = actions_to_qpos(chunk.float()).cuda()
    with torch.no_grad():
        tcp = mesh.tcp(q)[:, :2].cpu().numpy()
    closed = chunk[:, 7].numpy() < 0
    if not r["is_grasped"]:
        idx = np.nonzero(closed)[0]
        t = idx[0] if len(idx) else len(closed) - 1
        return float(np.linalg.norm(tcp[t] - slots_cube[int(r["target_cube_slot"])]))
    idx = np.nonzero(~closed)[0]
    if not len(idx):
        return None
    return float(np.linalg.norm(tcp[idx[0]] - slots_placard[int(r["target_place_slot"])]))


def sample_modes(chunks, mesh, is_grasped):
    from gwm_wiser.local_eval.state_diffusion_proposals import chunk_mode

    modes = []
    for c in chunks:
        q = actions_to_qpos(c[::2].float()).cuda()
        with torch.no_grad():
            tcp = mesh.tcp(q).cpu().numpy()
        modes.append(chunk_mode(tcp[:, :2], c[::2, 7].numpy(), is_grasped))
    return modes


def st_gradient(sc, R, chunks, r, te, batch):
    """d score / d keyframe actions (K, 5, 8), straight-through frames, in batches."""
    grads, scores = [], []
    for i in range(0, chunks.shape[0], batch):
        c = chunks[i: i + batch]
        kf = keyframe_actions(c).clone().cuda().requires_grad_(True)
        s = sc.score(frames_from(c, r["cur_rgb"], R, "st", kf_actions=kf), te)
        (g,) = torch.autograd.grad(s.sum(), kf)
        grads.append(g.detach().cpu()); scores.append(s.detach().cpu())
    g = torch.cat(grads)
    st_gradient.nonfinite = int((~torch.isfinite(g)).flatten(1).any(1).sum())
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), torch.cat(scores)


# ── C4: local direction ───────────────────────────────────────────────────────
def stage_c4(args):
    from gwm_wiser.local_eval.diff_gwm_score import DiffGWMScorer
    from gwm_wiser.local_eval.state_diffusion_proposals import CUBE_SLOTS_XY, PLACARD_SLOTS_XY

    planner = load_planner(torch.bfloat16)
    sc = DiffGWMScorer(planner)
    R = chosen_renderer(args)
    reps = load_replans(args.replans)
    rng = np.random.default_rng(2)
    picks = [reps[i] for i in sorted(rng.choice(len(reps), size=min(args.n, len(reps)), replace=False))]
    steps = [0.01, 0.03, 0.1]
    rows = {s: dict(rise=[], dscore=[], wrong_ddist=[]) for s in steps}
    nonfinite = 0
    for r in picks:
        chunks = torch.from_numpy(r["chunks"]).float()
        te = stage_score_fn(sc, r)
        g, _ = st_gradient(sc, R, chunks, r, te, args.batch)
        nonfinite += st_gradient.nonfinite
        ga = g[:, :, :7]
        dirn = ga / ga.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)  # unit L-inf per sample
        g0, p0 = original_scores(planner, chunks, r["cur_rgb"], r["first_rgb"], r["task"], R)
        s0 = p0 if r["is_grasped"] else g0
        modes = sample_modes(chunks, R.soft.mesh, r["is_grasped"])
        tgt = int(r["target_place_slot"] if r["is_grasped"] else r["target_cube_slot"])
        wrong = [((m[1] if r["is_grasped"] else m[0]) not in (None, tgt)) for m in modes]
        d0 = [event_xy_dist(c, R.soft.mesh, r, CUBE_SLOTS_XY, PLACARD_SLOTS_XY) for c in chunks]
        for s in steps:
            new = chunks.clone()
            new[:, :, :7] += smooth_offset(s * dirn, T=chunks.shape[1])
            g1, p1 = original_scores(planner, new, r["cur_rgb"], r["first_rgb"], r["task"], R)
            s1 = p1 if r["is_grasped"] else g1
            rows[s]["rise"] += list(s1 > s0)
            rows[s]["dscore"] += list(s1 - s0)
            for k in range(len(chunks)):
                if wrong[k] and d0[k] is not None:
                    d1 = event_xy_dist(new[k], R.soft.mesh, r, CUBE_SLOTS_XY, PLACARD_SLOTS_XY)
                    if d1 is not None:
                        rows[s]["wrong_ddist"].append(d1 - d0[k])
        print(f"c4 {r['file']}: " + " ".join(f"{s}: rise {np.mean(rows[s]['rise']):.3f}" for s in steps), flush=True)
    res = {"n_replans": len(picks), "n_samples": len(rows[steps[0]]["rise"]), "nonfinite_gradients": nonfinite,
           "step_normalisation": "unit L-inf over the 5x7 keyframe arm gradient (max joint offset = step)"}
    for s in steps:
        w = np.array(rows[s]["wrong_ddist"])
        res[f"step_{s}"] = dict(frac_score_rises=float(np.mean(rows[s]["rise"])),
                                mean_dscore=float(np.mean(rows[s]["dscore"])),
                                n_wrong_target=int(w.size),
                                wrong_target_mean_change_in_dist_m=float(w.mean()) if w.size else None,
                                wrong_target_frac_closer=float((w < 0).mean()) if w.size else None)
    json.dump(res, open(os.path.join(args.out, "c4.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


# ── C5: guided sampling smoke test ────────────────────────────────────────────
def guided_sample(policy, sc, R, r, te, seed, beta, guided_steps, batch, infer_steps=None, K=16):
    """DDIM (eta 0) with guidance on x0_hat; same noise for every setting of a replan.

    ``guided_steps``: iterable of DDIM step indices (one gradient step each) or a dict
    {step index: n_g gradient steps}. Each gradient step: straight-through d score / d x0_hat
    on the 5 keyframe arm targets, unit L-inf, applied as the C4 smooth offset of ``beta`` rad;
    DDIM then continues from the moved x0_hat with the predicted eps.
    """
    sched = policy.infer_sched
    sched.set_timesteps(infer_steps or policy.infer_steps)
    gsteps = guided_steps if isinstance(guided_steps, dict) else {i: 1 for i in guided_steps}
    obs = torch.from_numpy(r["history"]).float().cuda()[None]
    cond = policy.state_norm.norm(obs).flatten(1).repeat(K, 1)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((K, policy.pred_horizon, policy.action_dim), device="cuda", generator=gen)
    an = policy.action_norm
    acp = sched.alphas_cumprod.cuda()
    ts = sched.timesteps
    for i, t in enumerate(ts):
        with torch.no_grad():
            eps = policy.net(x, t.expand(K).cuda(), cond)
        a_t = acp[t]
        a_prev = acp[ts[i + 1]] if i + 1 < len(ts) else sched.final_alpha_cumprod.cuda()
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        for _ in range(gsteps.get(i, 0) if beta > 0 else 0):
            acts = an.unnorm(x0)[:, :60].cpu()
            g, _ = st_gradient(sc, R, acts, r, te, batch)
            ga = g[:, :, :7]
            dirn = ga / ga.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
            off = smooth_offset(beta * dirn.cuda(), T=60)  # radians
            scale = (an.scale[:7] / 2).cuda()  # d(normalised)/d(radians) = 2 / range
            x0 = x0.clone()
            x0[:, :60, :7] = (x0[:, :60, :7] + off / scale).clamp(-1, 1)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return an.unnorm(x.clamp(-1, 1))[:, :60].detach().cpu()


def panda_limits():
    import xml.etree.ElementTree as ET

    from mani_skill import PACKAGE_ASSET_DIR

    u = ET.parse(f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf").getroot()
    lim = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
           for j in u.findall("joint") if j.find("limit") is not None}
    lo = np.array([lim[f"panda_joint{i}"][0] for i in range(1, 8)])
    hi = np.array([lim[f"panda_joint{i}"][1] for i in range(1, 8)])
    return lo, hi


SCHEDULES = {  # name: (DDIM steps, {guided step index: n_g})
    "unguided16": (16, {}),
    "unguided32": (32, {}),
    "g0x1": (16, {0: 1}),
    "g0x2": (16, {0: 2}),
    "g3x1": (16, {3: 1}),
    "g0g16x1_d32": (32, {0: 1, 16: 1}),
}


def stage_c5s(args):
    """Guided-point schedule family on matched replans/seeds (extends c5; also the offline screen)."""
    from gwm_wiser.local_eval.diff_gwm_score import DiffGWMScorer
    from gwm_wiser.local_eval.state_dp import StateDiffusionPolicy

    planner = load_planner(torch.bfloat16)
    sc = DiffGWMScorer(planner)
    R = chosen_renderer(args)
    policy = StateDiffusionPolicy.load(args.policy)
    reps = load_replans(args.replans)
    if args.subset == "c5":  # the same place / grasp replans as stage_c5
        place = [r for r in reps if r["is_grasped"]
                 and sum(int(m[1]) == int(r["target_place_slot"]) for m in r["sample_modes"]) <= 4]
        grasp = [r for r in reps if not r["is_grasped"]]
        rng = np.random.default_rng(3)
        place = [place[i] for i in sorted(rng.choice(len(place), size=min(args.n_place, len(place)), replace=False))]
        grasp = [grasp[i] for i in sorted(rng.choice(len(grasp), size=min(args.n_grasp, len(grasp)), replace=False))]
        reps = place + grasp
    lo, hi = panda_limits()
    settings = []
    for name in args.schedules:
        steps, gp = SCHEDULES[name]
        if not gp:
            settings.append((name, steps, gp, 0.0))
        else:
            settings += [(f"{name}_b{b}", steps, gp, b) for b in args.betas]
    rows = []
    out_path = os.path.join(args.out, f"c5s{args.tag}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(d["file"], d["setting"]) for d in map(json.loads, open(out_path))}
    fh = open(out_path, "a")
    for r in reps:
        te = stage_score_fn(sc, r)
        seed = zlib.crc32(r["file"].encode()) % (2 ** 31)
        tgt = int(r["target_place_slot"] if r["is_grasped"] else r["target_cube_slot"])
        for name, steps, gp, beta in settings:
            if (r["file"], name) in done:
                continue
            t0 = time.time()
            chunks = guided_sample(policy, sc, R, r, te, seed, beta, gp, args.batch, infer_steps=steps)
            dt = time.time() - t0
            modes = sample_modes(chunks, R.soft.mesh, r["is_grasped"])
            on = [(m[1] if r["is_grasped"] else m[0]) == tgt for m in modes]
            g, p = original_scores(planner, chunks, r["cur_rgb"], r["first_rgb"], r["task"], R)
            sc_ = p if r["is_grasped"] else g
            sel = int(np.argmax(sc_))
            arm = chunks[:, :, :7].numpy()
            rec = dict(file=r["file"], setting=name, stage="place" if r["is_grasped"] else "grasp",
                       target_frac=float(np.mean(on)), selected_on_target=bool(on[sel]), best_score=float(sc_[sel]),
                       seconds=dt, joint_limit_violation_frac=float(((arm < lo) | (arm > hi)).any(-1).mean()),
                       max_step_joint_change=float(np.abs(np.diff(arm, axis=1)).max()),
                       mean_max_step_joint_change=float(np.abs(np.diff(arm, axis=1)).max(-1).max(-1).mean()))
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"c5s {rec['stage']} {r['file']} {name}: on-target {rec['target_frac']:.3f} sel {int(rec['selected_on_target'])} "
                  f"best {rec['best_score']:.4f} jump {rec['mean_max_step_joint_change']:.4f} ({dt:.0f}s)", flush=True)
    fh.close()
    recs = [json.loads(l) for l in open(out_path)]
    summ = {}
    for name in sorted({d["setting"] for d in recs}):
        for stage in ("place", "grasp", "all"):
            rs = [d for d in recs if d["setting"] == name and (stage == "all" or d["stage"] == stage)]
            if not rs:
                continue
            summ[f"{name}|{stage}"] = {k: float(np.mean([d[k] for d in rs])) for k in
                                       ("target_frac", "selected_on_target", "best_score", "seconds",
                                        "joint_limit_violation_frac", "mean_max_step_joint_change")}
            summ[f"{name}|{stage}"]["n"] = len(rs)
    json.dump(summ, open(os.path.join(args.out, f"c5s{args.tag}_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))


def stage_c5(args):
    from gwm_wiser.local_eval.diff_gwm_score import DiffGWMScorer
    from gwm_wiser.local_eval.state_dp import StateDiffusionPolicy

    planner = load_planner(torch.bfloat16)
    sc = DiffGWMScorer(planner)
    R = chosen_renderer(args)
    policy = StateDiffusionPolicy.load(args.policy)
    reps = load_replans(args.replans)
    place = [r for r in reps if r["is_grasped"]
             and sum(int(m[1]) == int(r["target_place_slot"]) for m in r["sample_modes"]) <= 4]
    grasp = [r for r in reps if not r["is_grasped"]]
    rng = np.random.default_rng(3)
    place = [place[i] for i in sorted(rng.choice(len(place), size=min(args.n_place, len(place)), replace=False))]
    grasp = [grasp[i] for i in sorted(rng.choice(len(grasp), size=min(args.n_grasp, len(grasp)), replace=False))]
    n_steps = policy.infer_steps
    settings = [("unguided", 0.0, ())]
    for b in args.betas:
        settings.append((f"all_b{b}", b, tuple(range(n_steps))))
    settings.append((f"early_b{args.betas[-1]}", args.betas[-1], tuple(range(n_steps // 2))))
    settings.append((f"late_b{args.betas[-1]}", args.betas[-1], tuple(range(n_steps // 2, n_steps))))
    out = {name: dict(target_frac=[], best_score=[], seconds=[]) for name, _, _ in settings}
    for group, rs in (("place", place), ("grasp", grasp)):
        for r in rs:
            te = stage_score_fn(sc, r)
            seed = zlib.crc32(r["file"].encode()) % (2 ** 31)
            tgt = int(r["target_place_slot"] if r["is_grasped"] else r["target_cube_slot"])
            for name, beta, rng_steps in settings:
                t0 = time.time()
                chunks = guided_sample(policy, sc, R, r, te, seed, beta, rng_steps, args.batch)
                dt = time.time() - t0
                modes = sample_modes(chunks, R.soft.mesh, r["is_grasped"])
                frac = float(np.mean([(m[1] if r["is_grasped"] else m[0]) == tgt for m in modes]))
                g, p = original_scores(planner, chunks, r["cur_rgb"], r["first_rgb"], r["task"], R)
                best = float((p if r["is_grasped"] else g).max())
                for k, v in (("target_frac", frac), ("best_score", best), ("seconds", dt)):
                    out[name][k].append((group, v))
                print(f"c5 {group} {r['file']} {name}: target frac {frac:.3f} best {best:.4f} ({dt:.0f}s)", flush=True)
    res = {"n_place": len(place), "n_grasp": len(grasp), "guidance": "reconstruction guidance on x0_hat; "
           "straight-through score gradient w.r.t. the 5 keyframe arm targets, unit L-inf, applied as the C4 "
           "smooth offset of beta rad, DDIM eta 0 continued from the moved x0_hat; same noise per replan"}
    for name in out:
        for group in ("place", "grasp"):
            tf = [v for g, v in out[name]["target_frac"] if g == group]
            bs = [v for g, v in out[name]["best_score"] if g == group]
            sec = [v for g, v in out[name]["seconds"] if g == group]
            res[f"{name}|{group}"] = dict(target_frac_mean=float(np.mean(tf)), best_of_16_score_mean=float(np.mean(bs)),
                                         seconds_per_replan=float(np.mean(sec)))
    json.dump(res, open(os.path.join(args.out, "c5.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


# ── cost ──────────────────────────────────────────────────────────────────────
def stage_cost(args):
    from gwm_wiser.local_eval.diff_gwm_score import DiffGWMScorer

    res = {}
    for dtype in (torch.bfloat16, torch.float32):
        planner = load_planner(dtype)
        sc = DiffGWMScorer(planner)
        R = chosen_renderer(args)
        r = load_replans(args.replans, limit=1)[0]
        te = stage_score_fn(sc, r)
        chunks = torch.from_numpy(r["chunks"]).float()
        name = str(dtype).split(".")[-1]
        res[name] = {}
        for B in (1, 2, 4, 8, 16, 32):
            c = chunks[:B] if B <= 16 else chunks.repeat(2, 1, 1)[:B]
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                st_gradient(sc, R, c, r, te, B)  # warm-up
                torch.cuda.synchronize(); t0 = time.time()
                st_gradient(sc, R, c, r, te, B)
                torch.cuda.synchronize()
                res[name][B] = dict(seconds_per_candidate=(time.time() - t0) / B,
                                    peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
                print(name, B, res[name][B], flush=True)
            except torch.cuda.OutOfMemoryError:
                res[name][B] = "OOM"
                print(name, B, "OOM", flush=True)
                break
        # breakdown at B = 1: soft render forward+backward alone
        q = actions_to_qpos(keyframe_actions(chunks[:1]).reshape(5, 8)).cuda().requires_grad_(True)
        torch.cuda.synchronize(); t0 = time.time()
        R.soft(q).sum().backward()
        torch.cuda.synchronize()
        res[name]["soft_render_5_frames_fwd_bwd_s"] = time.time() - t0
        del planner, sc
        torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(args.out, "cost.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["c1", "c2", "c3", "c4", "c5", "c5s", "cost"])
    ap.add_argument("--replans", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n_states", type=int, default=200)
    ap.add_argument("--coords", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--eps_orig", type=float, default=1e-2)
    ap.add_argument("--n_st", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--policy", default=os.environ.get("DP_POLICY"), help="state-DP checkpoint (or set DP_POLICY)")
    ap.add_argument("--n_place", type=int, default=20)
    ap.add_argument("--n_grasp", type=int, default=20)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.01, 0.03])
    ap.add_argument("--voxel", type=float, default=0.0015)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--sigma_override", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--schedules", nargs="+", default=list(SCHEDULES))
    ap.add_argument("--subset", default="c5", choices=["c5", "all"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"c1": stage_c1, "c2": stage_c2, "c3": stage_c3, "c4": stage_c4, "c5": stage_c5, "c5s": stage_c5s, "cost": stage_cost}[args.stage](args)


if __name__ == "__main__":
    main()
