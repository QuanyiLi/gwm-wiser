"""Draw the arm GWM actually saw, back onto the photograph it actually saw.

The scorer's evidence for one candidate is six frames: the external RGB, then
five robot-only renders sampled on the WISER schedule. `gwm_server --dump-dir`
already saves them, but as the model gets them -- the robot alpha-composited
over BLACK. That is faithful and nearly unreadable: you can see the arm's
silhouette, and nothing about where it is relative to the container it is
supposed to be reaching into.

This writes ONE figure per trajectory: the six sampled poses, each composited
onto the scene photograph, with the grasp frame ringed so the gripper is
findable at a glance. Nothing here changes what was scored -- the timeline
comes from `plan_to_candidate`, the frame indices from `sample_rat_times`, the
poses from the same render URDF and the same camera pose, so panel k is
pixel-for-pixel the arm in dump frame k. Only the background differs, and
panel 0, where the model is handed the photo itself rather than a render.

Read it for the two things the black strips cannot answer:

  * does the horizon COVER the placement, or does it burn frames on a
    stationary arm and stop before the gripper opens?
  * is the target visible at all, or does the arm cross in front of it?
    Occlusion by the robot's own body is a real failure mode, and it is
    invisible in a robot-on-black render by construction.

    cd /home/quanyi/gwm-wiser
    PYTHONPATH=.:droid ./.venv/bin/python -m gwm_hardware.gwm_arm.viz_rat \
        --run-dir droid/gwm_hardware/runs/session/<run>
    PYTHONPATH=.:droid ./.venv/bin/python -m gwm_hardware.gwm_arm.viz_rat --all-places

Writes `rat_overlay/cand<i>_<target>_s<score>.png` per candidate, plus
`compare_final.png` -- every candidate's last frame, best first, which is what
a 0.010 score spread actually turns on.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

TCP_LINK = "grasp_frame"


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def composite(rgb: np.ndarray, render: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    return np.clip(rgb.astype(np.float32) * (1 - a)
                   + render.astype(np.float32) * a, 0, 255).astype(np.uint8)


def tcp_pixels(renderer, qpos, grip, K, c2w) -> list:
    """Where the grasp frame lands in the image, per pose. None if behind."""
    import sapien

    links = {l.name: l for l in renderer.robot.get_links()}
    if TCP_LINK not in links:
        return [None] * len(qpos)
    link = links[TCP_LINK]
    renderer.robot.set_root_pose(sapien.Pose())
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    K = np.asarray(K, dtype=np.float64)
    out = []
    for q, g in zip(np.asarray(qpos), np.atleast_1d(np.asarray(grip))):
        renderer.robot.set_qpos(renderer.full_qpos(q, np.atleast_1d(g)))
        p = np.asarray(link.entity_pose.p, dtype=np.float64)
        pc = w2c[:3, :3] @ p + w2c[:3, 3]
        uv = K @ pc
        out.append((float(uv[0] / uv[2]), float(uv[1] / uv[2])) if uv[2] > 0.05 else None)
    return out


def sheet(panels, labels, header, marks=None, cell_w=860, cols=3, pad=10) -> Image.Image:
    """One figure: a cell per panel, with the grasp frame ringed."""
    h, w = panels[0].shape[:2]
    cell_h = int(round(cell_w * h / w))
    rows = (len(panels) + cols - 1) // cols
    top = 40
    out = Image.new("RGB", (cols * cell_w + (cols + 1) * pad,
                            top + rows * cell_h + (rows + 1) * pad), (16, 16, 18))
    d = ImageDraw.Draw(out)
    d.text((pad, 10), header, fill=(240, 240, 240), font=_font(24))
    s = cell_w / w
    for i, (p, lab) in enumerate(zip(panels, labels)):
        r, c = divmod(i, cols)
        x, y = pad + c * (cell_w + pad), top + pad + r * (cell_h + pad)
        out.paste(Image.fromarray(p).resize((cell_w, cell_h), Image.LANCZOS), (x, y))
        m = (marks or [None] * len(panels))[i]
        if m is not None:
            u, v = x + m[0] * s, y + m[1] * s
            for rr, col, wd in ((34, (0, 0, 0), 7), (34, (255, 60, 200), 3)):
                d.ellipse([u - rr, v - rr, u + rr, v + rr], outline=col, width=wd)
            d.line([u - 52, v, u - 40, v], fill=(255, 60, 200), width=3)
            d.line([u + 40, v, u + 52, v], fill=(255, 60, 200), width=3)
        d.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(70, 70, 78))
        d.rectangle([x, y, x + cell_w - 1, y + 30], fill=(0, 0, 0))
        d.text((x + 8, y + 5), lab, fill=(245, 245, 245), font=_font(21))
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
    files = sorted(proposals_dir.glob("scores_*.json"))
    if not files:
        return {}, {}
    s = json.loads(files[-1].read_text())
    return {r["file"]: r["score"] for r in s["ranking"]}, s


def run_dir_is_place(run_dir: Path) -> bool:
    _, s = scores_for(run_dir / "proposals")
    return bool(s) and s["instruction"].strip().lower().startswith("place")


def overlay_run(run_dir: Path, renderer, args) -> Path:
    from droid.server.gwm_server import Candidate, candidate_timeline, sample_rat_times
    from gwm_tiptop.score_client import plan_to_candidate

    proposals = run_dir / "proposals"
    index = json.loads((proposals / "proposals_index.json").read_text())
    score_of, sjson = scores_for(proposals)
    sampling = sjson.get("sampling") or {}
    cam = args.cam or (sjson.get("cameras") or ["external_cam"])[0]
    scale = args.rat_scale if args.rat_scale is not None else sampling.get("rat_scale", 3.0)
    # Mirror the timeline the run was scored with unless told otherwise, so a
    # picture never shows frames the model was not given.
    timeline = {
        "drop_static_prefix": (args.drop_static_prefix
                              if args.drop_static_prefix is not None
                              else bool(sampling.get("drop_static_prefix", False))),
        "append_release": (args.append_release
                           if args.append_release is not None
                           else bool(sampling.get("append_release", False))),
    }

    rgb, K, c2w = view_from_h5(run_dir / "external_obs.h5", cam)
    h, w = rgb.shape[:2]
    out_dir = run_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    instruction = sjson.get("instruction", "(unscored)")
    # A figure drawn on a timeline the run was NOT scored with still carries the
    # saved score in its header, and that pairing is a lie unless it is labelled.
    stale = any(bool(sampling.get(k, False)) != v for k, v in timeline.items())
    warn = "   [SCORE IS FROM THE RECORDED TIMELINE, NOT THIS ONE]" if stale else ""
    print(f"\n{run_dir.name}  {instruction!r}  cam={cam} rat_scale={scale} "
          f"drop_static_prefix={timeline['drop_static_prefix']} "
          f"append_release={timeline['append_release']}")
    finals = []

    for ci, entry in enumerate(index["proposals"]):
        plan = json.loads((proposals / entry["file"]).read_text())
        cand = Candidate(**plan_to_candidate(plan, **timeline))
        qpos, t, grip = candidate_timeline(cand)
        idxs = sample_rat_times(scale, t)
        renders, alphas = renderer.render(qpos[idxs], grip[idxs], K, c2w,
                                          width=w, height=h, return_alpha=True)
        marks = tcp_pixels(renderer, qpos[idxs], grip[idxs], K, c2w)

        panels, labels = [], []
        for k in range(len(idxs)):
            panels.append(composite(rgb, renders[k], alphas[k]))
            note = "  <- model gets the PHOTO here" if k == 0 else ""
            moved = "" if k == 0 else (
                "  STILL" if np.allclose(qpos[idxs[k]], qpos[idxs[k - 1]], atol=1e-6) else "")
            labels.append(f"{k}  t={t[idxs[k]]:.2f}s  grip={grip[idxs[k]]:.2f}{moved}{note}")

        sc = score_of.get(entry["file"])
        stxt = "na" if sc is None else f"{sc:+.4f}"
        head = (f"{entry['file']}   target={entry['target']}   score={stxt}   "
                f"window {t[idxs[-1]]:.2f}s of {t[-1]:.2f}s   {instruction!r}{warn}")
        sheet(panels, labels, head, marks=marks).save(
            out_dir / f"cand{ci:02d}_{entry['target']}_s{stxt}.png")
        finals.append((sc, entry, panels[-1], marks[-1]))

        still = sum(1 for k in range(1, len(idxs))
                    if np.allclose(qpos[idxs[k]], qpos[idxs[k - 1]], atol=1e-6))
        print(f"  cand{ci:02d} {entry['target']:9s} {stxt}  window {t[idxs[-1]]:.2f}/{t[-1]:.2f}s"
              f"  grip {grip[idxs[0]]:.2f}->{grip[idxs[-1]]:.2f}"
              f"  {still} repeated frame(s)")

    # Every candidate's LAST frame, score-descending. The per-candidate figures
    # say what one plan does; this says how much the sixteen differ.
    finals.sort(key=lambda f: -(f[0] if f[0] is not None else 0.0))
    sheet([p for _, _, p, _ in finals],
          [f"{'na' if s0 is None else f'{s0:+.4f}'}  {e['target']}  {e['file'][:11]}"
           for s0, e, _, _ in finals],
          f"last RAT frame of every candidate, best first   {instruction!r}   cam={cam}",
          marks=[m for _, _, _, m in finals], cell_w=520, cols=4).save(out_dir / "compare_final.png")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="a session run dir; default = the most recent place run")
    ap.add_argument("--all-places", action="store_true")
    ap.add_argument("--sessions", type=Path, default=Path("droid/gwm_hardware/runs/session"))
    ap.add_argument("--cam", default=None, help="default: the camera the run was scored from")
    ap.add_argument("--rat-scale", type=float, default=None,
                    help="default: the scale the run was scored with")
    ap.add_argument("--drop-static-prefix", action="store_true", default=None,
                    help="default: whatever the run was scored with")
    ap.add_argument("--append-release", action="store_true", default=None,
                    help="default: whatever the run was scored with")
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
        print(overlay_run(run, renderer, args))


if __name__ == "__main__":
    main()
