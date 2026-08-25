"""plot_figs: the figures for the directional push study.

All plan views are drawn top-down as seen from behind the robot: world +x
(away from the base, "front") is up, world +y (the robot's left) is left.

    ../droid-sim-evals/.venv/bin/python plot_figs.py \
        --exec exec_winner.json --suffix _winner   # matplotlib lives in this venv
"""

import argparse
import json
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from config import (COLORS, CUBE_COLOR_NAME, CUBE_D, CUBE_RGB, CUBE_SIZE, CUBES,
                    HOME_XY, PROMPTS, REGION, RESULTS)

TAGS = ("front", "left", "right")
MOVED_M = 0.03   # analyze.py's genuine-push threshold: blade-carried, not grazed
CUBE_LABEL = {t: f"{CUBE_COLOR_NAME[t]} cube" for t in TAGS}
SCENE_NOTE = f"three colour-named cubes {CUBE_D * 100:.0f} cm from the gripper"


def plan_axes(ax, pad=0.03, extra=None):
    """Top-down frame: horizontal = world y (left positive, drawn leftwards).

    `extra` is an (n, 2) array of world (x, y) that the frame must also cover.
    """
    x0, x1, y0, y1 = REGION
    if extra is not None and len(extra):
        e = np.asarray(extra, dtype=float)
        x0, x1 = min(x0, e[:, 0].min()), max(x1, e[:, 0].max())
        y0, y1 = min(y0, e[:, 1].min()), max(y1, e[:, 1].max())
    ax.set_xlim(y1 + pad, y0 - pad)                    # +y on the left
    ax.set_ylim(x0 - pad, x1 + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("world y  (m)   <- robot's left        robot's right ->")
    ax.set_ylabel("world x  (m)   (away from the base ->)")
    ax.grid(alpha=0.15, lw=0.5)


def draw_scene(ax, cubes=True, region=True):
    if region:
        ax.add_patch(Rectangle((REGION[2], REGION[0]),
                               REGION[3] - REGION[2], REGION[1] - REGION[0],
                               fill=False, ec="0.75", ls="--", lw=1.0,
                               label="_endpoint search region"))
    if cubes:
        h = CUBE_SIZE / 2
        for tag, (cx, cy) in CUBES.items():
            rgb = CUBE_RGB[tag]
            face = tuple(1 - 0.45 * (1 - c) for c in rgb)   # the cube's own paint, washed out
            ax.add_patch(Rectangle((cy - h, cx - h), CUBE_SIZE, CUBE_SIZE,
                                   fc=face, ec="0.35", lw=1.0, zorder=1))
            ax.text(cy, cx - h - 0.012, CUBE_LABEL[tag], ha="center", va="top",
                    fontsize=7.5, color="0.35")
    ax.plot([HOME_XY[1]], [HOME_XY[0]], marker="P", ms=11, color="k", zorder=6)
    ax.text(HOME_XY[1], HOME_XY[0] + 0.015, "gripper home", ha="center",
            va="bottom", fontsize=8)


def counted(points, nd=3):
    """Collapse coincident points so repeats show as heavier markers."""
    c = Counter(tuple(np.round(p, nd)) for p in points)
    pts = np.array(list(c.keys()))
    return pts, np.array(list(c.values()))


def fig_cube_final(ex, path):
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    every = [r["final"][t][:2] for t in TAGS for r in ex.get(t, [])]
    plan_axes(ax, extra=every)
    draw_scene(ax)
    for tag in TAGS:
        recs = ex.get(tag)
        if not recs:
            continue
        col = COLORS[tag]
        spawn = np.array(recs[0]["spawn"][tag][:2])
        fin = np.array([r["final"][tag][:2] for r in recs])
        mag = np.linalg.norm(fin - spawn, axis=1)
        moved, still = fin[mag >= MOVED_M], fin[mag < MOVED_M]

        # Rollouts that never touched the cube all sit on its spawn; one marker
        # with the count, so they cannot swamp the ones that moved it.
        if len(still):
            ax.scatter([spawn[1]], [spawn[0]], s=90, facecolors="none",
                       edgecolors=col, lw=1.6, alpha=0.9, zorder=3)
            ax.text(spawn[1], spawn[0], f"{len(still)}", color=col, fontsize=7,
                    ha="center", va="center", zorder=4)
        for f in moved:
            ax.plot([spawn[1], f[1]], [spawn[0], f[0]], color=col, alpha=0.30,
                    lw=0.9, zorder=2)
        if len(moved):
            pts, n = counted(moved)
            ax.scatter(pts[:, 1], pts[:, 0], s=18 + 10 * np.sqrt(n), c=col,
                       alpha=0.8, edgecolors="white", lw=0.4, zorder=5)
        m = fin.mean(axis=0)
        ax.annotate("", xy=(m[1], m[0]), xytext=(spawn[1], spawn[0]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.6, color=col,
                                    shrinkA=0, shrinkB=0), zorder=7)
        d = m - spawn
        ax.scatter([], [], s=60, c=col, alpha=0.8, edgecolors="white",
                   label=f'"{PROMPTS[tag]}"\n    {len(moved)}/{len(recs)} moved it '
                         f'>= {MOVED_M * 100:.0f} cm; mean '
                         f'({d[0] * 100:+.1f}, {d[1] * 100:+.1f}) cm')
    ax.set_title("Where the named cube ends up\n"
                 "filled = rollouts that moved it, ring = count that did not,\n"
                 "arrow = mean displacement over all 100\n"
                 f"{SCENE_NOTE}", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"wrote {path}")


def fig_cube_final_zoom(ex, path, pad=0.055):
    """One panel per prompt, zoomed on that cube, so small pushes stay legible."""
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.4))
    # One window size for all three panels so the pushes compare directly.
    half = pad
    for tag in TAGS:
        for r in ex.get(tag, []):
            f = np.asarray(r["final"][tag][:2])
            half = max(half, float(np.abs(f - np.asarray(CUBES[tag])).max()) + 0.02)
    for ax, tag in zip(axes, TAGS):
        recs = ex.get(tag)
        col = COLORS[tag]
        h = CUBE_SIZE / 2
        cx, cy = CUBES[tag]
        ax.add_patch(Rectangle((cy - h, cx - h), CUBE_SIZE, CUBE_SIZE,
                               fc=tuple(1 - 0.45 * (1 - c) for c in CUBE_RGB[tag]),
                               ec="0.35", lw=1.0, zorder=1))
        ax.set_aspect("equal")
        ax.grid(alpha=0.15, lw=0.5)
        ax.set_xlim(cy + half, cy - half)
        ax.set_ylim(cx - half, cx + half)
        ax.set_xlabel("world y (m)   <- left    right ->")
        if not recs:
            continue
        spawn = np.array(recs[0]["spawn"][tag][:2])
        fin = np.array([r["final"][tag][:2] for r in recs])
        mag = np.linalg.norm(fin - spawn, axis=1)
        moved = fin[mag >= MOVED_M]
        for f in moved:
            ax.plot([spawn[1], f[1]], [spawn[0], f[0]], color=col, alpha=0.28,
                    lw=1.0, zorder=2)
        ax.scatter([spawn[1]], [spawn[0]], s=140, facecolors="none",
                   edgecolors=col, lw=1.8, zorder=3)
        ax.text(spawn[1], spawn[0], f"{len(fin) - len(moved)}", color=col,
                fontsize=8, ha="center", va="center", zorder=4)
        if len(moved):
            pts, n = counted(moved, nd=4)
            ax.scatter(pts[:, 1], pts[:, 0], s=26 + 12 * np.sqrt(n), c=col,
                       alpha=0.82, edgecolors="white", lw=0.5, zorder=5)
        m = fin.mean(axis=0)
        ax.annotate("", xy=(m[1], m[0]), xytext=(spawn[1], spawn[0]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.6, color=col,
                                    shrinkA=0, shrinkB=0), zorder=7)
        d = (m - spawn) * 100
        ax.set_title(f'"{PROMPTS[tag]}"\n{len(moved)}/{len(fin)} pushed >= 3 cm; '
                     f"mean ({d[0]:+.1f}, {d[1]:+.1f}) cm", fontsize=9.5)
    axes[0].set_ylabel("world x (m)   (away from the base ->)")
    fig.suptitle(f"Per-prompt close-up  |  {SCENE_NOTE}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=185)
    plt.close(fig)
    print(f"wrote {path}")


def fig_endpoints(ex, path):
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    plan_axes(ax)
    draw_scene(ax)
    for tag in TAGS:
        recs = ex.get(tag)
        if not recs:
            continue
        col = COLORS[tag]
        ends = np.array([r["endpoint"] for r in recs])
        pts, n = counted(ends)
        ax.scatter(pts[:, 1], pts[:, 0], s=16 + 11 * n, c=col, alpha=0.6,
                   edgecolors="none", zorder=4, label=f"{tag}  (n={len(recs)})")
        m = ends.mean(axis=0)
        ax.annotate("", xy=(m[1], m[0]), xytext=(HOME_XY[1], HOME_XY[0]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col,
                                    shrinkA=0, shrinkB=0), zorder=7)
    ax.set_title("What CEM searched for\n"
                 "slide endpoint chosen by each run; arrow = mean endpoint\n"
                 f"{SCENE_NOTE}", fontsize=10.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    print(f"wrote {path}")


def fig_scoremaps(grid, path, objective="lang"):
    xs, ys = np.array(grid["xs"]), np.array(grid["ys"])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6))
    for ax, tag in zip(axes, TAGS):
        d = grid["prompts"].get(tag)
        if d is None:
            plan_axes(ax)
            continue
        # M is indexed [x, y] so that imshow's rows run along world x (vertical)
        # and its columns along world y (horizontal), matching the plan view.
        M = np.full((len(xs), len(ys)), np.nan)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                e = d["points"].get(f"{round(float(x), 4)},{round(float(y), 4)}")
                if e:
                    M[i, j] = e["score"] if objective == "raw" else e["score"] - e["prior"]
        dx = (xs[1] - xs[0]) / 2 if len(xs) > 1 else 0.005
        dy = (ys[1] - ys[0]) / 2 if len(ys) > 1 else 0.005
        im = ax.imshow(M, origin="lower", cmap="viridis", aspect="equal",
                       extent=[ys[0] - dy, ys[-1] + dy, xs[0] - dx, xs[-1] + dx])
        plan_axes(ax)          # after imshow: it would otherwise reset the limits
        draw_scene(ax, region=False)
        im_i, jm = np.unravel_index(np.nan_to_num(M, nan=-1e9).argmax(), M.shape)
        ax.plot([ys[jm]], [xs[im_i]], marker="*", ms=17, mfc="white",
                mec="k", mew=1.1, zorder=9)
        ax.set_title(f'"{PROMPTS[tag]}"', fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"GWM score of the slide endpoint ({objective} objective); "
                 f"star = argmax  |  {SCENE_NOTE}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", dest="exec_file", default="exec_winner.json")
    ap.add_argument("--grid", default="grid.json")
    ap.add_argument("--objective", default="lang", choices=["lang", "raw"])
    ap.add_argument("--suffix", default="_winner")
    args = ap.parse_args()

    s = args.suffix
    grid_path = RESULTS / args.grid
    if grid_path.exists():
        fig_scoremaps(json.loads(grid_path.read_text()),
                      RESULTS / "fig_scoremap.png", args.objective)
    exec_path = RESULTS / args.exec_file
    if exec_path.exists():
        ex = json.loads(exec_path.read_text())
        fig_cube_final(ex, RESULTS / f"fig_cube_final{s}.png")
        fig_cube_final_zoom(ex, RESULTS / f"fig_cube_zoom{s}.png")
        fig_endpoints(ex, RESULTS / f"fig_endpoints{s}.png")


if __name__ == "__main__":
    main()
