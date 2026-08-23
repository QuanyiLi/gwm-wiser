"""score_vjepa: V-JEPA 2-AC energies of every candidate plan against every
candidate's executed end frame (the goal bank), for one scene family.

Inputs
  --replay-dir   output of sim/replay_candidates.py (frames, final frames, judge)
  --plans-dir    the candidate pool (plan_*.json + proposals_index.json)
Outputs (in --out-dir)
  energies.npz   E_pred [C, G, T]  energy of candidate c's PREDICTED frame t vs goal g
                 E_obs  [C, G, T]  energy of candidate c's OBSERVED frame t vs goal g (oracle)
                 E_cur  [G]        energy of the current frame vs goal g (no-motion baseline)
                 E_pred_lift / E_obs_lift / E_cur_lift   the same against the LIFT goal bank
                 E_pred_h{1.5,3,6} / E_obs_h{...}        the same against the HORIZON goal banks
                                                         (goal = goal candidate's frame at H s; t <= H only)
                 E_track [C, T]    |z_pred_c[t] - z_obs_c[t]|   predictor vs reality on its own trajectory
                 E_still [C, T]    |z_obs_c[0]  - z_obs_c[t]|   "nothing moves" baseline for E_track
                 n_frames [C], close_frame [C], lift_frame [C], t [C, T]; candidate / goal names
  z_final.npz    fp16 final predicted / observed / current embeddings (for later analyses)
  goal images, model-view PNGs, config.json

Goal banks. `final`: goal g = the final external_cam_2 frame of candidate g
after the harness's post-plan hold (what the judge scored). `lift` (pick
family only): the frame LIFT_DELAY_S after candidate g's gripper-close
command, i.e. the object just lifted with the arm still above its place -- a
shorter-horizon, candidate-specific goal. `h<H>`: the frame H seconds into
the goal candidate's execution (arm part-way to / at the object), scored
against the rollout's first H seconds only -- the short-horizon "reach"
setting closest to how the paper uses the model. Frame t of a candidate is the frame
at control step t*stride (stride 4 -> 3.75 fps, DROID's training cadence);
the last frame is the final one.

    .venv/bin/python score_vjepa.py --family pick --replay-dir runs/replay_pick \
        --plans-dir ../gwm_integrate_doc/proposals/scene6_rev2 --out-dir runs/vjepa_pick/w32_s4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vjepa_sel.model import MAX_FRAMES, VJEPA2AC, l1_energy
from vjepa_sel.plan_stepper import load_plan
from vjepa_sel.preprocess import model_view_uint8
from vjepa_sel.traj import action_stats, plan_to_sequence


LIFT_DELAY_S = 3.0  # 1.33 s gripper action + ~1.7 s of the lift-out trajectory
HORIZONS_S = (1.5, 3.0, 6.0)


def load_candidates(plans_dir):
    index = json.loads((plans_dir / "proposals_index.json").read_text())
    return [(p["file"], p["target"], p.get("grasp_confidence")) for p in index["proposals"]]


def observed_frames(replay_dir, stem, cam, stride):
    d = replay_dir / stem
    fr = np.load(d / f"frames_{cam}.npz")["frames"]
    idx = np.load(d / "frame_index.npy")
    final = np.asarray(Image.open(d / f"final_{cam}.png"))[..., :3]
    judge = json.loads((d / "judge.json").read_text())
    assert judge["frame_stride"] == 4, judge["frame_stride"]
    if stride % 4 != 0:
        raise ValueError("stride must be a multiple of the recorded frame stride (4)")
    sub = stride // 4
    fr, idx = fr[::sub], idx[::sub]
    n_total = judge["n_steps_total"]
    frames = np.concatenate([fr, final[None]], axis=0)
    steps = np.append(idx, n_total)
    return frames, steps, judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["pick", "place"], required=True)
    ap.add_argument("--replay-dir", type=Path, required=True)
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cam", default="external_cam_2")
    ap.add_argument("--crop-mode", default="full_aa", choices=["full_aa", "train", "full", "square"])
    ap.add_argument("--tcp-offset", type=float, default=0.0,
                    help="state point along the tool axis from the flange (m); 0 = panda_link8")
    ap.add_argument("--window", type=int, default=MAX_FRAMES, help="context window (frames) for the AR rollout")
    ap.add_argument("--stride", type=int, default=4, help="control steps per model step (4 = 3.75 fps)")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    initial_gripper = 1.0 if args.family == "place" else 0.0

    cands = load_candidates(args.plans_dir)
    names = [c[0] for c in cands]
    C = len(cands)
    t0 = time.time()
    model = VJEPA2AC(**({"ckpt": args.ckpt} if args.ckpt else {}))
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    # ---- observed frames and their embeddings -------------------------------
    z_obs, steps_obs, judges = [], [], {}
    first_frames = []
    for stem, _, _ in [(Path(n).stem, None, None) for n in names]:
        frames, steps, judge = observed_frames(args.replay_dir, stem, args.cam, args.stride)
        first_frames.append(frames[0])
        judges[stem] = judge
        z = model.encode(frames, crop_mode=args.crop_mode)  # [T+1, 256, D]
        z_obs.append(z)
        steps_obs.append(steps)
        print(f"encoded {stem}: {len(frames)} frames", flush=True)
    # the current frame must be the same for every candidate (same reset + settle)
    diffs = [float(np.abs(first_frames[0].astype(np.int16) - f.astype(np.int16)).mean()) for f in first_frames]
    print(f"first-frame mean |diff| across candidates: max {max(diffs):.3f} (0 = identical)", flush=True)
    z_cur = z_obs[0][0]
    z_goal = torch.stack([z[-1] for z in z_obs])  # [G, 256, D]
    E_cur = l1_energy(z_cur[None], z_goal).cpu().numpy()  # [G]
    Image.fromarray(first_frames[0]).save(args.out_dir / "current_frame.png")
    Image.fromarray(model_view_uint8(first_frames[0][None], args.crop_mode)[0]).save(args.out_dir / "current_frame_model_view.png")
    goals_dir = args.out_dir / "goals"
    goals_dir.mkdir(exist_ok=True)
    for stem, z in zip([Path(n).stem for n in names], z_obs):
        final = np.asarray(Image.open(args.replay_dir / stem / f"final_{args.cam}.png"))[..., :3]
        Image.fromarray(model_view_uint8(final[None], args.crop_mode)[0]).save(goals_dir / f"{stem}_model_view.png")

    # ---- planned sequences ----------------------------------------------------
    T_max = max(len(z) for z in z_obs)
    n_frames = np.zeros(C, np.int64)
    close_frame = np.full(C, -1, np.int64)
    lift_frame = np.full(C, -1, np.int64)
    t_arr = np.full((C, T_max), np.nan, np.float32)
    seqs, seq_stats = [], {}
    for ci, (fname, target, conf) in enumerate(cands):
        stem = Path(fname).stem
        plan = load_plan(args.plans_dir / fname)
        seq = plan_to_sequence(plan, stride=args.stride, initial_gripper=initial_gripper, tcp_offset=args.tcp_offset)
        states, actions = seq["states"], seq["actions"]
        T1 = len(states)
        if T1 != len(z_obs[ci]):
            # the offline timeline and the replay can differ by a step at gripper holds;
            # align on the shorter one and report it
            print(f"WARNING {stem}: planned {T1} samples vs {len(z_obs[ci])} observed; truncating", flush=True)
            T1 = min(T1, len(z_obs[ci]))
            states, actions = states[:T1], actions[: T1 - 1]
        n_frames[ci] = T1
        t_arr[ci, :T1] = seq["t"][:T1]
        if seq["close_t"] is not None:
            close_frame[ci] = int(np.searchsorted(seq["t"][:T1], seq["close_t"]))
            if seq["close_t"] > 0:  # pick family: the object is lifted LIFT_DELAY_S after the close command
                lift_frame[ci] = min(T1 - 1, int(np.searchsorted(seq["t"][:T1], seq["close_t"] + LIFT_DELAY_S)))
        seqs.append((states, actions))
        seq_stats[stem] = {
            "target": target, "grasp_confidence": conf, "n_samples": int(T1),
            "close_t": seq["close_t"], "lift_frame": int(lift_frame[ci]), "n_loop_steps": int(seq["n_loop"]),
            "actions": action_stats(actions),
            "state0": states[0].tolist(), "stateT": states[-1].tolist(),
        }
    has_lift = bool((lift_frame >= 0).all())
    z_goal_lift = torch.stack([z_obs[g][int(lift_frame[g])] for g in range(C)]) if has_lift else None
    E_cur_lift = l1_energy(z_cur[None], z_goal_lift).cpu().numpy() if has_lift else np.full(C, np.nan, np.float32)
    if has_lift:
        for stem, z, lf in zip([Path(n).stem for n in names], z_obs, lift_frame):
            fr, _, _ = observed_frames(args.replay_dir, stem, args.cam, args.stride)
            Image.fromarray(model_view_uint8(fr[int(lf)][None], args.crop_mode)[0]).save(goals_dir / f"{stem}_lift_model_view.png")

    # horizon goal banks: frame index of H seconds (same cadence for every candidate)
    h_idx = {H: int(np.searchsorted(t_arr[0, : int(n_frames[0])], H - 1e-6)) for H in HORIZONS_S}
    h_idx = {H: i for H, i in h_idx.items() if i < int(n_frames.min())}
    z_goal_h = {H: torch.stack([z_obs[g][i] for g in range(C)]) for H, i in h_idx.items()}
    E_cur_h = {H: l1_energy(z_cur[None], zg).cpu().numpy() for H, zg in z_goal_h.items()}
    E_pred_h = {H: np.full((C, C, T_max), np.nan, np.float32) for H in h_idx}
    E_obs_h = {H: np.full((C, C, T_max), np.nan, np.float32) for H in h_idx}

    # ---- rollouts ---------------------------------------------------------------
    E_pred = np.full((C, C, T_max), np.nan, np.float32)
    E_obs = np.full((C, C, T_max), np.nan, np.float32)
    E_pred_lift = np.full((C, C, T_max), np.nan, np.float32)
    E_obs_lift = np.full((C, C, T_max), np.nan, np.float32)
    E_track = np.full((C, T_max), np.nan, np.float32)
    E_still = np.full((C, T_max), np.nan, np.float32)
    z_final_pred = np.zeros((C, z_cur.shape[0], z_cur.shape[1]), np.float16)
    z_final_obs = np.zeros_like(z_final_pred)
    z_lift_pred = np.zeros_like(z_final_pred)
    for ci, (fname, target, conf) in enumerate(cands):
        stem = Path(fname).stem
        states, actions = seqs[ci]
        T1 = int(n_frames[ci])
        tr = time.time()
        z_pred = model.rollout(z_cur, states, actions, context_window=args.window)  # [T, 256, D]
        z_seq = torch.cat([z_cur[None], z_pred], dim=0)  # index t = frame t (0 = current)
        # energies vs every goal
        for t in range(T1):
            E_pred[ci, :, t] = l1_energy(z_seq[t][None], z_goal).cpu().numpy()
            E_obs[ci, :, t] = l1_energy(z_obs[ci][t][None], z_goal).cpu().numpy()
            if has_lift:
                E_pred_lift[ci, :, t] = l1_energy(z_seq[t][None], z_goal_lift).cpu().numpy()
                E_obs_lift[ci, :, t] = l1_energy(z_obs[ci][t][None], z_goal_lift).cpu().numpy()
        for H, i in h_idx.items():
            for t in range(i + 1):
                E_pred_h[H][ci, :, t] = l1_energy(z_seq[t][None], z_goal_h[H]).cpu().numpy()
                E_obs_h[H][ci, :, t] = l1_energy(z_obs[ci][t][None], z_goal_h[H]).cpu().numpy()
        E_track[ci, :T1] = l1_energy(z_seq[:T1], z_obs[ci][:T1]).cpu().numpy()
        E_still[ci, :T1] = l1_energy(z_obs[ci][0][None].expand(T1, -1, -1), z_obs[ci][:T1]).cpu().numpy()
        z_final_pred[ci] = z_seq[T1 - 1].cpu().numpy().astype(np.float16)
        z_final_obs[ci] = z_obs[ci][T1 - 1].cpu().numpy().astype(np.float16)
        if has_lift:
            z_lift_pred[ci] = z_seq[int(lift_frame[ci])].cpu().numpy().astype(np.float16)
        print(f"rollout {stem} ({target}): {T1-1} steps in {time.time()-tr:.1f}s; "
              f"E_pred(final) vs own goal {E_pred[ci, ci, T1-1]:.4f}, track(final) {E_track[ci, T1-1]:.4f}, "
              f"still(final) {E_still[ci, T1-1]:.4f}"
              + (f"; lift: E_pred {E_pred_lift[ci, ci, int(lift_frame[ci])]:.4f} vs E_cur {E_cur_lift[ci]:.4f}" if has_lift else ""),
              flush=True)

    h_arrays = {}
    for H, i in h_idx.items():
        tag = f"h{H:g}"
        h_arrays[f"E_pred_{tag}"] = E_pred_h[H]
        h_arrays[f"E_obs_{tag}"] = E_obs_h[H]
        h_arrays[f"E_cur_{tag}"] = E_cur_h[H]
        h_arrays[f"idx_{tag}"] = np.int64(i)
    np.savez(
        args.out_dir / "energies.npz",
        E_pred=E_pred, E_obs=E_obs, E_cur=E_cur, E_track=E_track, E_still=E_still,
        E_pred_lift=E_pred_lift, E_obs_lift=E_obs_lift, E_cur_lift=E_cur_lift,
        n_frames=n_frames, close_frame=close_frame, lift_frame=lift_frame, t=t_arr,
        horizons=np.asarray(sorted(h_idx)), **h_arrays,
        names=np.asarray(names), targets=np.asarray([c[1] for c in cands]),
        confidences=np.asarray([c[2] if c[2] is not None else np.nan for c in cands], np.float32),
    )
    np.savez_compressed(args.out_dir / "z_final.npz", z_cur=z_cur.cpu().numpy().astype(np.float16),
                        z_final_pred=z_final_pred, z_final_obs=z_final_obs, z_lift_pred=z_lift_pred,
                        z_goal_lift=(z_goal_lift.cpu().numpy().astype(np.float16) if has_lift else np.zeros(0, np.float16)))
    (args.out_dir / "config.json").write_text(json.dumps({
        "family": args.family, "cam": args.cam, "crop_mode": args.crop_mode, "window": args.window,
        "stride": args.stride, "tcp_offset": args.tcp_offset, "horizon_frame_index": {f"{H:g}": i for H, i in h_idx.items()},
        "replay_dir": str(args.replay_dir), "plans_dir": str(args.plans_dir),
        "ckpt_meta": {k: (float(v) if not isinstance(v, int) else v) for k, v in model.ckpt_meta.items()},
        "initial_gripper": initial_gripper, "sequences": seq_stats,
        "judges": judges, "first_frame_max_mean_absdiff": max(diffs),
    }, indent=1))
    print(f"done in {time.time()-t0:.0f}s -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
