"""analyze: quantitative summary of the grid + CEM results -> results/stats.json.

Per prompt and per signal (raw fused score; debiased = score - prior):
  - grid_argmax + whether it falls inside the prompt's 15 cm cell;
  - top10_in_cell: fraction of the 10 best grid points inside the cell;
  - cell_margin: mean over the cell minus the best OTHER cell's mean;
  - CEM: hit by final mean / by best-ever sample.
The same summary is also computed for each camera alone.
"""

import json

import numpy as np

from config import CAMS, CELLS, IMG_SIZE, RESULTS
from plot_figs import load_grid

HALF = IMG_SIZE / 2


def cell_mask(xs, ys, cx, cy):
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return (np.abs(gx - cx) <= HALF) & (np.abs(gy - cy) <= HALF)


def load_grid_cam(path, cam):
    g = json.loads((RESULTS / path).read_text())
    xs, ys = np.asarray(g["xs"]), np.asarray(g["ys"])
    maps = {}
    for tag, row in g["prompts"].items():
        raw = np.full((len(xs), len(ys)), np.nan)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                e = row["points"].get(f"{round(float(x), 4)},{round(float(y), 4)}")
                if e:
                    raw[i, j] = e["per_cam"][cam]["score"]
        maps[tag] = {"raw": raw}
    return xs, ys, maps


def summarize(xs, ys, maps, cem, signal="raw"):
    out = {}
    for tag, (cx, cy) in CELLS.items():
        if tag not in maps:
            continue
        m = maps[tag][signal]
        i, j = np.unravel_index(np.nanargmax(m), m.shape)
        in_cell = bool(abs(xs[i] - cx) <= HALF and abs(ys[j] - cy) <= HALF)
        order = np.argsort(-m, axis=None)[:10]
        oi, oj = np.unravel_index(order, m.shape)
        top10 = float(np.mean([(abs(xs[a] - cx) <= HALF and abs(ys[b] - cy) <= HALF)
                               for a, b in zip(oi, oj)]))
        means = {t: float(np.nanmean(m[cell_mask(xs, ys, *CELLS[t])])) for t in CELLS}
        margin = means[tag] - max(v for t, v in means.items() if t != tag)
        row = {"argmax": [float(xs[i]), float(ys[j])], "argmax_in_cell": in_cell,
               "top10_in_cell": top10, "cell_means": means,
               "cell_margin": float(margin)}
        if cem and tag in cem:
            n_evals = sum(len(h["samples"]) for h in cem[tag]["history"])
            row["cem"] = {"hit_final_mean": cem[tag]["hit_final_mean"],
                          "hit_winner": cem[tag]["hit_winner"],
                          "final_mean": cem[tag]["final_mean"],
                          "samples_drawn": n_evals}
        out[tag] = row
    return out


def print_table(title, rows):
    print(f"\n[{title}]")
    print(f"{'prompt':<12} {'argmax in cell':<15} {'top10 in cell':<14} "
          f"{'cell margin':<12} {'CEM hit(mean/winner)'}")
    for tag, r in rows.items():
        c = r.get("cem", {})
        print(f"{tag:<12} {str(r['argmax_in_cell']):<15} "
              f"{r['top10_in_cell']:<14.2f} {r['cell_margin']:<+12.4f} "
              f"{c.get('hit_final_mean')}/{c.get('hit_winner')}")


def main() -> None:
    xs, ys, maps = load_grid("grid.json")
    cem = json.loads((RESULTS / "cem.json").read_text()) \
        if (RESULTS / "cem.json").exists() else {}
    stats = {"raw": summarize(xs, ys, maps, cem, "raw"),
             "debiased": summarize(xs, ys, maps, cem, "lang"),
             "per_cam": {}}
    for cam in CAMS:
        _, _, mc = load_grid_cam("grid.json", cam)
        stats["per_cam"][cam] = summarize(xs, ys, mc, {}, "raw")
    (RESULTS / "stats.json").write_text(json.dumps(stats, indent=2))

    print_table("raw (fused)", stats["raw"])
    print_table("debiased (fused)", stats["debiased"])
    for cam in CAMS:
        print_table(f"raw ({cam})", stats["per_cam"][cam])


if __name__ == "__main__":
    main()
