"""plot_figs: figures for one scored configuration.

  fig_curves_<fam>_<tag>.png   energy of every candidate's rollout (top: predicted,
                               bottom: observed frames) against the designated goal
                               of each target object, over time
  fig_matrix_<fam>_<tag>.png   final-frame energy matrices (predicted / observed)
  fig_track_<fam>_<tag>.png    predictor-vs-reality along each trajectory vs the
                               'nothing moved' baseline

    .venv/bin/python plot_figs.py --family pick --energy-dir runs/vjepa_pick/w32_s4 --out-dir vjepa_ret/figs
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vjepa_sel.tasks import clusters, tasks  # noqa: E402

# fixed categorical order (reference palette slots 1-6), one per cluster id
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TEXT = "#0b0b0b"
MUTED = "#52514e"


def short(name):
    return Path(name).stem.replace("plan_", "")


def plot_matrix(fam, tag, names, targets, color, order, n_frames, Ep, Eo, t_end, label, bank, out_dir):
    C = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (arm, E) in zip(axes, (("predicted rollout", Ep), ("observed frames (oracle)", Eo))):
        M = np.array([[E[c, g, (int(n_frames[c]) - 1) if t_end is None else t_end] for g in order] for c in order])
        Mplot = M.copy()
        np.fill_diagonal(Mplot, np.nan)  # self-goal is trivially 0 for the oracle
        im = ax.imshow(Mplot, cmap="Blues_r", aspect="equal")
        ax.set_xticks(range(C))
        ax.set_xticklabels([short(names[c]) for c in order], rotation=90, fontsize=7)
        ax.set_yticks(range(C))
        ax.set_yticklabels([short(names[c]) for c in order], fontsize=7)
        for lab in ax.get_xticklabels():
            lab.set_color(color[targets[order[int(lab.get_position()[0])]]])
        for lab in ax.get_yticklabels():
            lab.set_color(color[targets[order[int(lab.get_position()[1])]]])
        ax.set_xlabel(f"goal = {label} frame of candidate")
        ax.set_ylabel("candidate (rolled out / executed)")
        ax.set_title(f"{label} energy — {arm}", fontsize=9)
        for j in range(C):
            i = int(np.nanargmin(Mplot[:, j]))
            ax.plot(j, i, marker="s", ms=9, mfc="none", mec="#e34948", mew=1.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="L1 energy (lower = closer)")
    fig.suptitle(f"{fam} pool, config {tag}: red square = argmin candidate for each goal (diagonal excluded)", fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_matrix_{fam}_{tag}{'' if bank == 'final' else '_' + bank}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--energy-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.energy_dir.name
    fam = args.family
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ez = np.load(args.energy_dir / "energies.npz")
    sel = json.loads((args.energy_dir / "selection.json").read_text())
    names = [str(n) for n in ez["names"]]
    targets = [str(t) for t in ez["targets"]]
    n_frames = ez["n_frames"]
    t_arr = ez["t"]
    C = len(names)
    cl = clusters(fam)
    cluster_ids = sorted(set(targets))
    color = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(cluster_ids)}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})

    # ---- energy curves per target object ------------------------------------
    target_clusters = []
    for tg, _, k, _ in tasks(fam):
        if k not in target_clusters:
            target_clusters.append(k)
    headline = sel["results"]["pred/final:final/two_stage"]["excl"]
    goal_of = {}
    for tg, d in headline.items():
        if d["single"] is not None:
            goal_of[d["target"]] = d["single"]["goal"]
    fig, axes = plt.subplots(2, len(target_clusters), figsize=(4.2 * len(target_clusters), 6.2), sharey="row", squeeze=False)
    for j, k in enumerate(target_clusters):
        if k not in goal_of:
            for r in range(2):
                axes[r, j].set_title(f"{cl.get(k)} — no successful goal candidate")
            continue
        g = names.index(goal_of[k])
        for r, (arm, E) in enumerate((("predicted rollout", ez["E_pred"]), ("observed frames (oracle)", ez["E_obs"]))):
            ax = axes[r, j]
            for c in range(C):
                if c == g:
                    continue
                n = int(n_frames[c])
                correct = targets[c] == k
                ax.plot(t_arr[c, :n], E[c, g, :n], color=color[targets[c]], lw=2.0 if correct else 1.2,
                        alpha=1.0 if correct else 0.75, zorder=3 if correct else 2)
            ax.axhline(float(ez["E_cur"][g]), color=MUTED, lw=1, ls=":", zorder=1)
            ax.text(0.02, float(ez["E_cur"][g]), "current frame vs goal", color=MUTED, fontsize=7, va="bottom")
            ax.set_title(f"goal: {cl.get(k)} ({short(names[g])}) — {arm}", fontsize=9, color=TEXT)
            ax.set_xlabel("time from plan start (s)")
            if j == 0:
                ax.set_ylabel("L1 energy to goal embedding")
            ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    handles = [plt.Line2D([], [], color=color[k], lw=2, label=f"{k} = {cl.get(k)}") for k in cluster_ids]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(cluster_ids), 3), frameon=False, fontsize=8)
    fig.suptitle(f"V-JEPA 2-AC energies, {fam} pool, config {tag} (bold = candidates of the goal's object)", fontsize=10, color=TEXT)
    fig.tight_layout(rect=(0, 0.10 if len(cluster_ids) > 3 else 0.06, 1, 0.96))
    fig.savefig(args.out_dir / f"fig_curves_{fam}_{tag}.png", dpi=150)
    plt.close(fig)

    # ---- energy matrices: final-frame bank and the short-horizon banks -------
    order = sorted(range(C), key=lambda c: (targets[c], c))
    matrix_banks = [("final", "final-frame", ez["E_pred"], ez["E_obs"], None)]
    for H in (ez["horizons"].tolist() if "horizons" in ez else []):
        hk = f"h{H:g}"
        matrix_banks.append((hk, f"{H:g} s horizon", ez[f"E_pred_{hk}"], ez[f"E_obs_{hk}"], int(ez[f"idx_{hk}"])))
    for bank, label, Ep, Eo, t_end in matrix_banks:
        plot_matrix(fam, tag, names, targets, color, order, n_frames, Ep, Eo, t_end, label, bank, args.out_dir)

    # ---- predictor vs reality -----------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    Et, Es = ez["E_track"], ez["E_still"]
    for c in range(C):
        n = int(n_frames[c])
        ax.plot(t_arr[c, :n], Et[c, :n], color=PALETTE[0], lw=0.8, alpha=0.5)
        ax.plot(t_arr[c, :n], Es[c, :n], color=PALETTE[1], lw=0.8, alpha=0.5)
    ax.plot([], [], color=PALETTE[0], lw=2, label="|z_pred(t) − z_obs(t)|  (rollout vs what the sim showed)")
    ax.plot([], [], color=PALETTE[1], lw=2, label="|z_obs(0) − z_obs(t)|  (‘nothing moved’ baseline)")
    ax.set_xlabel("time from plan start (s)")
    ax.set_ylabel("L1 energy")
    ax.set_title(f"open-loop predictor vs observed frames, all {C} candidates ({fam}, {tag})", fontsize=9, color=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"fig_track_{fam}_{tag}.png", dpi=150)
    plt.close(fig)
    print("figures ->", args.out_dir)


if __name__ == "__main__":
    main()
