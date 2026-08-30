"""plot_figs: render the gwm_point_cem figures from results/*.json.

Run with the droid-sim venv (has matplotlib):
    ../droid-sim-evals/.venv/bin/python plot_figs.py [--bar-grid grid_bar.json]

Rendering conventions:
  - score maps are an alpha-graded single-hue overlay (transparent where cold,
    saturated blue where hot) on the top-down table; per-panel relative scale
    (2/98 percentiles);
  - the reported signal is the fused score ("raw"); score-minus-prior
    variants are emitted alongside as *_debiased.png, and the prior map is
    its own control panel;
  - the cross-prompt partition z-scores each map before taking the argmax;
  - prompts use a fixed categorical palette + distinct marker shapes; the CEM
    path inside a single-prompt panel is drawn in a contrasting accent.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

from config import CELLS, IMG_FILES, IMG_SIZE, REGION, RESULTS  # noqa: E402

HEAT_CMAP = LinearSegmentedColormap.from_list("heatblue", [
    (0.00, to_rgba("#cde2fb", 0.00)),
    (0.35, to_rgba("#5598e7", 0.30)),
    (0.65, to_rgba("#2a78d6", 0.62)),
    (1.00, to_rgba("#0d366b", 0.90)),
])
TAGS = list(CELLS)  # prompt tags in cell order
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
_MARKERS = ["o", "s", "^", "D"]
PROMPT_COLOR = dict(zip(TAGS, _PALETTE))
PROMPT_MARKER = dict(zip(TAGS, _MARKERS))
INK, INK2 = "#0b0b0b", "#52514e"
TABLE_FILL = "#f1ede4"
HALF = IMG_SIZE / 2

plt.rcParams.update({
    "font.size": 10, "text.color": INK, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def load_grid(path):
    g = json.loads((RESULTS / path).read_text())
    xs, ys = np.asarray(g["xs"]), np.asarray(g["ys"])
    maps = {}
    for tag, row in g["prompts"].items():
        raw = np.full((len(xs), len(ys)), np.nan)
        prior = np.full((len(xs), len(ys)), np.nan)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                e = row["points"].get(f"{round(float(x), 4)},{round(float(y), 4)}")
                if e:
                    raw[i, j] = e["score"]
                    prior[i, j] = e["prior"]
        maps[tag] = {"raw": raw, "prior": prior, "lang": raw - prior,
                     "instruction": row["instruction"]}
    return xs, ys, maps


def draw_table(ax, photo_half=0.6 * HALF, title_pad=None):
    """Top-down underlay: table fill, photo thumbnails, true cell borders.

    Axes: horizontal = world y (m, +y on the LEFT — the top-down view with
    world x pointing up), vertical = world x (up = away from the robot base).
    Thumbnails render smaller than the physical 15 cm cell; the outline marks
    the true cell.
    """
    x0, x1, y0, y1 = REGION
    pad = 0.035
    ax.add_patch(patches.Rectangle((y0 - pad, x0 - pad), (y1 - y0) + 2 * pad,
                                   (x1 - x0) + 2 * pad, facecolor=TABLE_FILL,
                                   edgecolor="none", zorder=0))
    for name, (cx, cy) in CELLS.items():
        img = np.asarray(Image.open(IMG_FILES[name]).convert("RGB"))
        h = photo_half
        ax.imshow(img, extent=(cy + h, cy - h, cx - h, cx + h),
                  origin="upper", zorder=4, interpolation="bilinear")
        ax.add_patch(patches.Rectangle((cy - HALF, cx - HALF), IMG_SIZE, IMG_SIZE,
                                       fill=False, edgecolor="white", lw=1.6, zorder=3))
        ax.add_patch(patches.Rectangle((cy - HALF, cx - HALF), IMG_SIZE, IMG_SIZE,
                                       fill=False, edgecolor=INK2, lw=0.5, zorder=3))
    ax.set_xlim(y1 + pad, y0 - pad)
    ax.set_ylim(x0 - pad, x1 + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("world y (m)")
    ax.set_ylabel("world x (m)")


def heat(ax, xs, ys, m):
    lo, hi = np.nanpercentile(m, 2), np.nanpercentile(m, 98)
    step = xs[1] - xs[0]
    im = ax.imshow(m, cmap=HEAT_CMAP, vmin=lo, vmax=hi, origin="lower",
                   extent=(ys[0] - step / 2, ys[-1] + step / 2,
                           xs[0] - step / 2, xs[-1] + step / 2),
                   interpolation="bicubic", zorder=2)
    return im


def cem_overlay(ax, run, color, marker, label=None, samples=True, lw=1.6):
    hist = run["history"]
    if samples:
        for it, h in enumerate(hist):
            f = it / max(len(hist) - 1, 1)
            pts = np.asarray(h["samples"])
            ax.scatter(pts[:, 1], pts[:, 0], s=11, marker=marker, color=color,
                       alpha=0.2 + 0.6 * f, linewidths=0, zorder=5)
    # Path = the per-iteration means; the landing point = the best-scoring
    # sample of the whole run.
    path = [h["mean"] for h in hist] + [run["final_mean"]]
    if run.get("winner") is not None:
        path = path + [run["winner"]]
    path = np.asarray(path)
    ax.plot(path[:, 1], path[:, 0], color=color, lw=lw, alpha=0.9, zorder=7)
    for k in range(len(path) - 1):
        ax.annotate("", xy=(path[k + 1, 1], path[k + 1, 0]),
                    xytext=(path[k, 1], path[k, 0]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    shrinkA=0, shrinkB=0), zorder=7)
    land = path[-1]
    ax.scatter([land[1]], [land[0]], s=200, marker="*", color=color,
               edgecolor="white", linewidths=1.2, zorder=8)
    if label:
        ax.annotate(label, (means[0, 1], means[0, 0]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=color, zorder=9)


def fig_combined(cem, out, subtitle):
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    draw_table(ax, photo_half=0.72 * HALF)
    for tag in TAGS:
        if tag in cem:
            cem_overlay(ax, cem[tag], PROMPT_COLOR[tag], PROMPT_MARKER[tag])
    handles = [Line2D([], [], color=PROMPT_COLOR[t], marker=PROMPT_MARKER[t],
                      ls="-", ms=6, label=f'"…{t}"')
               for t in TAGS if t in cem]
    handles.append(Line2D([], [], color=INK2, marker="*", ls="none", ms=11,
                          markerfacecolor="white",
                          label="selected point (best sample)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    ax.set_title('CEM over EEF hover (x, y) — "point at the image of the …"\n'
                 + subtitle, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(RESULTS / out, dpi=220)
    plt.close(fig)
    print("wrote", RESULTS / out)


def fig_perprompt(xs, ys, maps, cem, out, signal="raw",
                  tags=None,
                  cem_color="#eb6834"):
    tags = tags or TAGS
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 10.6))
    im = None
    for ax, tag in zip(axes.ravel(), tags):
        draw_table(ax)
        im = heat(ax, xs, ys, maps[tag][signal])
        if cem and tag in cem:
            cem_overlay(ax, cem[tag], cem_color, PROMPT_MARKER.get(tag, "o"),
                        samples=False, lw=1.4)
        m = maps[tag][signal]
        i, j = np.unravel_index(np.nanargmax(m), m.shape)
        ax.scatter([ys[j]], [xs[i]], s=80, marker="o", facecolor="none",
                   edgecolor=INK, linewidths=1.6, zorder=9)
        ax.set_title(f'"{maps[tag]["instruction"]}"', fontsize=10)
    label = {"raw": "GWM score (fused)", "lang": "score − prior"}[signal]
    cb = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cb.set_label(f"{label} — per-panel relative scale", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("Score map over hover (x, y) — ○ grid argmax, orange = CEM mean path",
                 fontsize=11, y=0.995)
    fig.savefig(RESULTS / out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", RESULTS / out)


def fig_controls(xs, ys, maps, out, signal="raw"):
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4))
    panels = [("prior", maps["dog"]["prior"], "no prompt (prior: empty instruction)"),
              ("animal", maps["animal"][signal], f'"{maps["animal"]["instruction"]}"'),
              ("fruit", maps["fruit"][signal], f'"{maps["fruit"]["instruction"]}"')]
    for ax, (tag, m, title) in zip(axes, panels):
        draw_table(ax)
        im = heat(ax, xs, ys, m)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Controls: instruction-independent prior; deliberately 2-way "
                 "ambiguous prompts (relative scale per panel)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / out, dpi=200)
    plt.close(fig)
    print("wrote", RESULTS / out)


def fig_argmax(xs, ys, maps, out, signal="raw"):
    tags = TAGS
    stack = np.stack([(maps[t][signal] - np.nanmean(maps[t][signal]))
                      / np.nanstd(maps[t][signal]) for t in tags])
    win = np.argmax(stack, axis=0)
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    draw_table(ax)
    step = xs[1] - xs[0]
    rgba = np.zeros(win.shape + (4,))
    for k, t in enumerate(tags):
        rgba[win == k] = to_rgba(PROMPT_COLOR[t], alpha=0.5)
    ax.imshow(rgba, origin="lower",
              extent=(ys[0] - step / 2, ys[-1] + step / 2,
                      xs[0] - step / 2, xs[-1] + step / 2),
              interpolation="nearest", zorder=2)
    handles = [patches.Patch(facecolor=PROMPT_COLOR[t], alpha=0.5,
                             label=f'"…{t}" wins') for t in tags]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    ax.set_title("Which prompt prefers each hover point\n"
                 "(argmax over the four maps, z-scored per prompt)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(RESULTS / out, dpi=220)
    plt.close(fig)
    print("wrote", RESULTS / out)


def fig_bar_compare(xs, ys, maps_g, maps_b, out, signal="raw"):
    tags = TAGS
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 9.6))
    for col, tag in enumerate(tags):
        for row, (maps, label) in enumerate([(maps_g, "closed gripper"),
                                             (maps_b, "bar EEF")]):
            ax = axes[row, col]
            draw_table(ax)
            im = heat(ax, xs, ys, maps[tag][signal])
            m = maps[tag][signal]
            i, j = np.unravel_index(np.nanargmax(m), m.shape)
            ax.scatter([ys[j]], [xs[i]], s=60, marker="o", facecolor="none",
                       edgecolor=INK, linewidths=1.4, zorder=9)
            if row == 0:
                ax.set_title(f'"…{tag}"', fontsize=10)
            if col == 0:
                ax.text(-0.26, 0.5, label, transform=ax.transAxes, rotation=90,
                        va="center", fontsize=11, color=INK)
            cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cb.ax.tick_params(labelsize=6)
    fig.suptitle("Renderer EEF ablation: closed 2F-85 (top) vs rigid bar (bottom) on "
                 "IDENTICAL trajectories — fused score, relative scale, ○ = argmax",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(RESULTS / out, dpi=180)
    plt.close(fig)
    print("wrote", RESULTS / out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="grid.json")
    ap.add_argument("--cem", default="cem.json")
    ap.add_argument("--bar-grid", default=None)
    args = ap.parse_args()

    xs, ys, maps = load_grid(args.grid)
    cem = json.loads((RESULTS / args.cem).read_text()) \
        if (RESULTS / args.cem).exists() else {}

    if cem:
        fig_combined(cem, "fig_cem_combined.png",
                     "dots: samples (faint → solid) · path: iteration means "
                     "→ selected point")
    fig_perprompt(xs, ys, maps, cem, "fig_maps_perprompt.png", signal="raw")
    fig_perprompt(xs, ys, maps, {}, "fig_maps_perprompt_debiased.png",
                  signal="lang")
    if "animal" in maps and "fruit" in maps:
        fig_controls(xs, ys, maps, "fig_controls.png", signal="raw")
    fig_argmax(xs, ys, maps, "fig_argmax_partition.png", signal="raw")
    fig_argmax(xs, ys, maps, "fig_argmax_partition_debiased.png", signal="lang")
    if args.bar_grid and (RESULTS / args.bar_grid).exists():
        _, _, maps_b = load_grid(args.bar_grid)
        fig_bar_compare(xs, ys, maps, maps_b, "fig_bar_compare.png", signal="raw")


if __name__ == "__main__":
    main()
