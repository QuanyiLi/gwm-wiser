"""Draw the arm GWM actually saw, back onto the photograph it actually saw.

The scorer's evidence for one candidate is six frames: the external RGB, then
five robot-only renders sampled on the WISER schedule. `gwm_server --dump-dir`
already saves them, but as the model gets them -- the robot alpha-composited
over BLACK. That is faithful and nearly unreadable: you can see the arm's
silhouette, and nothing about where it is relative to the container it is
supposed to be reaching into.

This composites the same five renders onto the same photograph, so each
horizon can be judged in the scene. Nothing here changes what was scored: the
timeline comes from `plan_to_candidate`, the frame indices from
`sample_rat_times`, the poses from the same render URDF and the same camera
pose, so panel k is pixel-for-pixel the arm in dump frame k -- only the
background differs.

Read it for the two things the black strips cannot answer:

  * does the horizon COVER the placement? A window that ends while the arm is
    still travelling shows the model a trajectory whose destination is not in
    any frame, and every candidate then looks alike.
  * is the target visible at all, or does the arm cross in front of it?
    Occlusion by the robot's own body is how the sim's `yellow` failure worked
    (G-29), and it is invisible in a robot-on-black render by construction.

    cd /home/quanyi/gwm-wiser
    PYTHONPATH=.:droid ./.venv/bin/python -m gwm_hardware.gwm_arm.viz_rat \
        --run-dir droid/gwm_hardware/runs/session/20260819_154219_01
    PYTHONPATH=.:droid ./.venv/bin/python -m gwm_hardware.gwm_arm.viz_rat --all-places

Writes `rat_overlay/cand<i>_<target>_s<score>.png` next to the run's proposals.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

# Blue -> red across the five horizons, so the trail panel reads as time.
RAMP = [(80, 150, 255), (110, 220, 220), (150, 230, 120), (255, 200, 70), (255, 90, 80)]


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def composite(rgb: np.ndarray, render: np.ndarray, alpha: np.ndarray,
              tint=None, strength: float = 1.0) -> np.ndarray:
    """Alpha-composite one robot render over the scene photo.

    `tint` recolours the arm (trail panel); without it the arm keeps the shaded
    colour the model is shown, which is what makes a panel comparable to its
    dump-strip twin.
    """
    a = np.clip(alpha, 0.0, 1.0)[..., None] * strength
    fg = render.astype(np.float32)
    if tint is not None:
        lum = fg.mean(axis=2, keepdims=True) / 255.0
        fg = np.asarray(tint, dtype=np.float32) * (0.35 + 0.65 * lum)
    return np.clip(rgb.astype(np.float32) * (1 - a) + fg * a, 0, 255).astype(np.uint8)


def sheet(panels, labels, header, cell_w=560, cols=4, pad=8) -> Image.Image:
    """Contact sheet: one cell per panel, header line on top."""
    h, w = panels[0].shape[:2]
    cell_h = int(round(cell_w * h / w))
    rows = (len(panels) + cols - 1) // cols
    top = 34
    out = Image.new("RGB", (cols * cell_w + (cols + 1) * pad,
                            top + rows * cell_h + (rows + 1) * pad), (18, 18, 20))
    d = ImageDraw.Draw(out)
    d.text((pad, 8), header, fill=(235, 235, 235), font=_font(20))
    for i, (p, lab) in enumerate(zip(panels, labels)):
        r, c = divmod(i, cols)
        x, y = pad + c * (cell_w + pad), top + pad + r * (cell_h + pad)
        out.paste(Image.fromarray(p).resize((cell_w, cell_h), Image.LANCZOS), (x, y))
        d.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(70, 70, 78))
        d.rectangle([x, y, x + cell_w - 1, y + 24], fill=(0, 0, 0))
        d.text((x + 6, y + 4), lab, fill=(240, 240, 240), font=_font(17))
    return out


def view_from_h5(path: Path, cam: str):
    with h5py.File(path) as f:
        if cam not in f:
            raise SystemExit(f"{path} has no camera {cam!r} (has {list(f)})")
        pos = np.asarray(f[f"{cam}/pos_w"])
        qw, qx, qy, qz = np.asarray(f[f"{cam}/quat_w_ros"])
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w[:3, 3] = pos
        return (np.asarray(f[f"{cam}/rgb"])[..., :3],
                np.asarray(f[f"{cam}/intrinsic_matrix"]), c2w)


def scores_for(proposals_dir: Path) -> tuple[dict, dict]:
    """(file -> score, the scores json) for the run's single scoring pass."""
    files = sorted(proposals_dir.glob("scores_*.json"))
    if not files:
        return {}, {}
    s = json.loads(files[-1].read_text())
    return {r["file"]: r["score"] for r in s["ranking"]}, s


def run_dir_is_place(run_dir: Path) -> bool:
    _, s = scores_for(run_dir / "proposals")
    return bool(s) and s["instruction"].strip().lower().startswith("place")


def overlay_run(run_dir: Path, renderer, rat_scale, cam_override, out_name: str) -> Path:
    from gwm_tiptop.score_client import plan_to_candidate
    from droid.server.gwm_server import Candidate, candidate_timeline, sample_rat_times

    proposals = run_dir / "proposals"
    index = json.loads((proposals / "proposals_index.json").read_text())
    score_of, sjson = scores_for(proposals)
    cam = cam_override or (sjson.get("cameras") or ["external_cam"])[0]
    scale = rat_scale if rat_scale is not None else (sjson.get("sampling") or {}).get("rat_scale", 3.0)

    rgb, K, c2w = view_from_h5(run_dir / "external_obs.h5", cam)
    h, w = rgb.shape[:2]
    out_dir = run_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    instruction = sjson.get("instruction", "(unscored)")
    finals = []
    print(f"\n{run_dir.name}  {instruction!r}  cam={cam} rat_scale={scale}")

    for ci, entry in enumerate(index["proposals"]):
        plan = json.loads((proposals / entry["file"]).read_text())
        cand = Candidate(**plan_to_candidate(plan))
        qpos, t, grip = candidate_timeline(cand)
        idxs = sample_rat_times(scale, t)
        renders, alphas = renderer.render(qpos[idxs[1:]], grip[idxs[1:]], K, c2w,
                                          width=w, height=h, return_alpha=True)

        panels = [rgb]
        labels = [f"0  scene photo (t=0.00)"]
        for k in range(5):
            panels.append(composite(rgb, renders[k], alphas[k]))
            labels.append(f"{k+1}  t={t[idxs[k+1]]:.2f}s  grip={grip[idxs[k+1]]:.2f}"
                          f"  cover={alphas[k].mean()*100:.1f}%")

        trail = rgb.copy()
        for k in range(5):
            trail = composite(trail, renders[k], alphas[k], tint=RAMP[k], strength=0.85)
        panels.append(trail)
        labels.append("trail  blue=first .. red=last")

        sc = score_of.get(entry["file"])
        head = (f"{entry['file']}   target={entry['target']}   "
                f"score={'n/a' if sc is None else f'{sc:+.4f}'}   "
                f"traj={entry['traj_s']}s  window={t[idxs[-1]]:.2f}s of {t[-1]:.2f}s   "
                f"{instruction!r}")
        name = f"cand{ci:02d}_{entry['target']}_s{'na' if sc is None else f'{sc:+.4f}'}.png"
        sheet(panels, labels, head).save(out_dir / name)
        finals.append((sc, entry, panels[-2]))
        print(f"  {name}  window {t[idxs[-1]]:.2f}/{t[-1]:.2f}s  "
              f"coverage {np.mean([a.mean() for a in alphas])*100:.1f}%")

    # Every candidate's LAST horizon frame, score-descending. The per-candidate
    # sheets say what one plan does; this says how much the sixteen differ,
    # which is the question a 0.010 score spread actually turns on.
    finals.sort(key=lambda f: -(f[0] if f[0] is not None else 0.0))
    sheet([p for _, _, p in finals],
          [f"{'na' if s0 is None else f'{s0:+.4f}'}  {e['target']}  {e['file'][:11]}"
           for s0, e, _ in finals],
          f"last RAT frame of every candidate, best first   {instruction!r}   cam={cam}",
          cell_w=460).save(out_dir / "compare_final.png")
    return out_dir


def main() -> None:
    import sys
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="a session run dir; default = the most recent place run")
    ap.add_argument("--all-places", action="store_true",
                    help="every session run whose instruction starts with 'place'")
    ap.add_argument("--sessions", type=Path,
                    default=Path("droid/gwm_hardware/runs/session"))
    ap.add_argument("--cam", default=None, help="default: the camera the run was scored from")
    ap.add_argument("--rat-scale", type=float, default=None,
                    help="default: the scale the run was scored with")
    ap.add_argument("--urdf", default=None, help="default: the render URDF the server uses")
    ap.add_argument("--out-name", default="rat_overlay")
    args = ap.parse_args()

    if args.run_dir:
        runs = [args.run_dir]
    else:
        cand = sorted((d for d in args.sessions.iterdir()
                       if (d / "proposals" / "proposals_index.json").exists()), reverse=True)
        runs = [d for d in cand if run_dir_is_place(d)]
        if not args.all_places:
            runs = runs[:1]
    if not runs:
        raise SystemExit("no place run found; pass --run-dir")

    from gwm_hardware.gwm_arm.render_model import ensure_render_urdf
    from real_data_train.renderer.franka_renderer import FrankaRobotRenderer

    renderer = FrankaRobotRenderer(str(args.urdf or ensure_render_urdf()), arm="panda")
    for run in runs:
        print(overlay_run(run, renderer, args.rat_scale, args.cam, args.out_name))


if __name__ == "__main__":
    main()
