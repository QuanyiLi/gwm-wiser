"""plot_figs: scene, score-matrix and execution figures for gwm_drawer.

Reads captures/scene8_0, results/selection.json and results/exec/, writes
fig_scene.png, fig_matrix.png and fig_exec.png into results/.

Run with the droid-sim-evals venv (matplotlib + imageio):

    ../droid-sim-evals/.venv/bin/python plot_figs.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import CAPTURE_DIR, RESULTS, TASKS

CAND_LABELS = {"red": "red\ndrawer", "yellow": "yellow\ndrawer", "blue": "blue\ndrawer",
               "block": "grasp\nblock", "bowl": "grasp\nbowl", "banana": "grasp\nbanana"}
TASK_LABELS = {t: f'"{TASKS[t]["phrases"][0]}"' for t in TASKS}
CLEAN_CAM = {"red": "ext", "yellow": "ext", "blue": "ext2",
             "block": "ext2", "bowl": "ext", "banana": "ext"}


def fig_scene():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, png, title in zip(axes, ("ext.png", "ext2.png"),
                              ("external_cam", "external_cam_2")):
        ax.imshow(Image.open(CAPTURE_DIR / png))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("scene 8: three single-drawer cabinets + three table objects",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_scene.png", dpi=160)
    plt.close(fig)


def fig_matrix():
    sel = json.loads((RESULTS / "selection.json").read_text())
    names = sel["candidates"]
    tasks = list(sel["tasks"])
    S = np.array([[sel["matrix"][t][n] for n in names] for t in tasks])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    im = ax.imshow(S, cmap="viridis", aspect="auto")
    for i, t in enumerate(tasks):
        for j in range(len(names)):
            ax.text(j, i, "%.3f" % S[i, j], ha="center", va="center", fontsize=8.5,
                    color="w" if S[i, j] < S.max() - 0.3 * np.ptp(S) else "k")
        j_r = names.index(sel["argmax"][t])
        ax.plot(j_r + 0.4, i - 0.36, marker="*", color="w", ms=9, mec="k")
        j_t = names.index(sel["tasks"][t])
        ax.add_patch(plt.Rectangle((j_t - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=2.0))
    ax.set_xticks(range(len(names)), [CAND_LABELS[n] for n in names], fontsize=8.5)
    ax.set_yticks(range(len(tasks)), [TASK_LABELS[t] for t in tasks], fontsize=8)
    ax.set_xlabel("candidate trajectory", fontsize=9)
    ax.set_title("GWM scores, 5-phrase ensemble, both-camera fusion; "
                 "star = argmax, red = target", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_matrix.png", dpi=160)
    plt.close(fig)


def fig_exec():
    import imageio.v2 as imageio

    sel = json.loads((RESULTS / "selection.json").read_text())
    rows = []
    for task in sel["candidates"]:
        cam = CLEAN_CAM[task]
        rd = imageio.get_reader(RESULTS / "exec" / task / f"{cam}.mp4")
        n = rd.count_frames()
        frames = [rd.get_data(i) for i in (0, int(n * 0.52), n - 2)]
        rd.close()
        rows.append((task, frames))
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 2.3 * len(rows)))
    for r, (task, frames) in enumerate(rows):
        for c, fr in enumerate(frames):
            ax = axes[r][c]
            ax.imshow(fr)
            ax.axis("off")
            if c == 0:
                ax.set_title(CAND_LABELS[task].replace("\n", " ") + " candidate",
                             loc="left", fontsize=9)
    fig.suptitle("execution of the six candidates (start / grasp / end)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_exec.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_scene()
    fig_matrix()
    try:
        fig_exec()
    except Exception as e:
        print(f"fig_exec skipped: {e}")
    print(f"wrote figures to {RESULTS}")
