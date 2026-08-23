"""make_bundle: assemble vjepa_ret/ (the tracked results bundle) from runs/.

Copies, per family and config: summary.md, selection.json, config.json,
energies.npz, csv/, goals/ (model-view goal images) and the current frame;
per replayed candidate: judge.json, traj.npz and the first / final
external_cam_2 frames downscaled to 640x360 JPEG. The raw 720p frame arrays
stay in runs/ (gitignored, ~4 GB).

    .venv/bin/python make_bundle.py
"""

import json
import shutil
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
RET = HERE / "vjepa_ret"


def copy_replay(family, replay_dir):
    out = RET / "replay" / family
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for cand in sorted(p for p in replay_dir.iterdir() if p.is_dir() and (p / "judge.json").exists()):
        dst = out / cand.name
        dst.mkdir(exist_ok=True)
        shutil.copy(cand / "judge.json", dst / "judge.json")
        shutil.copy(cand / "traj.npz", dst / "traj.npz")
        shutil.copy(cand / "cameras.json", dst / "cameras.json")
        for tag in ("first_external_cam_2", "final_external_cam_2", "final_external_cam", "final_wrist_cam"):
            im = Image.open(cand / f"{tag}.png").convert("RGB").resize((640, 360), Image.BILINEAR)
            im.save(dst / f"{tag}.jpg", quality=90)
        index[cand.name] = json.loads((cand / "judge.json").read_text())
    (out / "replay_index.json").write_text(json.dumps(index, indent=1))
    return index


def copy_scores(family, score_root):
    out = RET / family
    out.mkdir(parents=True, exist_ok=True)
    for cfg in sorted(p for p in score_root.iterdir() if p.is_dir() and (p / "selection.json").exists()):
        dst = out / cfg.name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir()
        for f in ("summary.md", "selection.json", "config.json", "energies.npz") + (
                ("current_frame_model_view.png",) if cfg.name in ("w32_s4", "w32_s4_cam1", "w32_s4_crop135") else ()):
            if (cfg / f).exists():
                shutil.copy(cfg / f, dst / f)
        if (cfg / "csv").exists():
            shutil.copytree(cfg / "csv", dst / "csv")
        # goal / current images only for the configs whose view or crop differs
        # (the others see identical frames)
        if cfg.name in ("w32_s4", "w32_s4_cam1", "w32_s4_crop135"):
            if (cfg / "goals").exists():
                shutil.copytree(cfg / "goals", dst / "goals")
            im = Image.open(cfg / "current_frame.png").convert("RGB").resize((640, 360), Image.BILINEAR)
            im.save(dst / "current_frame.jpg", quality=90)


def main():
    RET.mkdir(exist_ok=True)
    for f in ("diag_onestep.json", "diag_onestep_full_aa.json", "action_stats.json", "tables.md"):
        if (HERE / "runs" / f).exists():
            shutil.copy(HERE / "runs" / f, RET / f)
    for fam in ("pick", "place"):
        rd = HERE / "runs" / f"replay_{fam}"
        if rd.exists():
            copy_replay(fam, rd)
        sr = HERE / "runs" / f"vjepa_{fam}"
        if sr.exists():
            copy_scores(fam, sr)
    print("bundle ->", RET)


if __name__ == "__main__":
    main()
