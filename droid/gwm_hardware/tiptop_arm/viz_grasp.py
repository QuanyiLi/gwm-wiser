"""Draw a saved plan's grasp against the scene, so it can be judged by eye.

`inspect_plan` reduces a grasp to numbers; this renders the same thing. Views
are axis-aligned and to scale, because the question is almost always metric --
does that pad actually get below the rim -- and a perspective view hides
exactly that.

Shows the object's own points, the M2T2 candidate grasp origins the optimiser
started from, and the finger pads at the configuration that will be executed.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.tiptop_arm.viz_grasp \
        droid/tiptop/tiptop_outputs/eval/<timestamp> --object blue_cup
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--object", required=True)
    ap.add_argument("--out", type=Path, default=Path.home() / "Desktop/grasp_check.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import yaml
    from curobo.types.base import TensorDeviceType
    from cutamp.robots import load_panda_robotiq_container

    from tiptop.perception.m2t2 import m2t2_to_tiptop_transform

    env = pickle.loads((a.run_dir / "perception/cutamp_env.pkl").read_bytes())
    target = next(m for m in env.movables if m.name == a.object)
    V = np.asarray(target.get_mesh().vertices) + np.asarray(target.pose)[:3]
    others = [(m.name, np.asarray(m.get_mesh().vertices) + np.asarray(m.pose)[:3])
              for m in env.type_to_objects["Movable"] if m.name != a.object]

    plan = json.loads((a.run_dir / "tiptop_plan.json").read_text())
    k = next(i for i, s in enumerate(plan["steps"]) if s.get("type") == "gripper")
    q = np.asarray(plan["steps"][k - 1]["positions"][-1])

    ta = TensorDeviceType()
    kin = load_panda_robotiq_container(ta).kin_model
    st = kin.get_state(ta.to_device(q[None]))
    tcp = st.ee_pose.get_numpy_matrix()[0][:3, 3]
    S = st.link_spheres_tensor[0].cpu().numpy()

    from gwm_hardware.common.paths import ASSETS

    cfgk = yaml.safe_load((ASSETS / "panda_robotiq_2f_140.yml")
                          .read_text())["robot_cfg"]["kinematics"]
    pads, i = {}, 0
    for n in cfgk["collision_link_names"]:
        m = len(cfgk["collision_spheres"].get(n, []))
        if m and "pad" in n:
            pads[n] = S[i:i + m]
        i += m

    g = torch.load(a.run_dir / "perception/grasps.pt", weights_only=False)[a.object]
    cand = (g["poses"] @ m2t2_to_tiptop_transform()[None])[:, :3, 3]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (h, v, hl, vl) in zip(axes, [(0, 2, "x (m)", "z (m)  -- height"),
                                         (0, 1, "x (m)", "y (m)  -- top view")]):
        ax.scatter(V[:, h], V[:, v], s=3, c="tab:blue", label=f"{a.object} (target)")
        for nm, W in others:
            ax.scatter(W[:, h], W[:, v], s=2, c="0.75", label=nm)
        ax.scatter(cand[:, h], cand[:, v], s=4, c="tab:green", alpha=.35,
                   label=f"M2T2 candidates ({len(cand)})")
        for nm, sp in pads.items():
            side = "left" if "left" in nm else "right"
            col = "tab:red" if side == "left" else "tab:orange"
            for s in sp:
                ax.add_patch(plt.Circle((s[h], s[v]), s[3], color=col, alpha=.25))
            ax.scatter([], [], c=col, s=40, label=f"{side} finger pad")
        ax.scatter([tcp[h]], [tcp[v]], marker="x", s=160, c="k", lw=2.5, label="grasp TCP")
        if v == 2:
            ax.axhline(V[:, 2].max(), ls="--", lw=1, c="tab:blue")
            ax.annotate(f"object top  z={V[:, 2].max():.4f}",
                        (ax.get_xlim()[0], V[:, 2].max()), fontsize=9, va="bottom", color="tab:blue")
        # Frame on the target and the gripper. The workspace keep-outs are metres
        # across and would otherwise shrink the thing being judged to a smudge.
        allpts = np.vstack([V[:, [h, v]]] + [np.stack([sp[:, h], sp[:, v]], 1) for sp in pads.values()])
        c = 0.5 * (allpts.min(0) + allpts.max(0))
        r = 0.55 * max((allpts.max(0) - allpts.min(0)).max(), 0.25)
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_xlabel(hl); ax.set_ylabel(vl); ax.set_aspect("equal"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{a.run_dir.name}   {a.object}   TCP z={tcp[2]:.4f}, object top={V[:, 2].max():.4f} "
                 f"({(tcp[2] - V[:, 2].max()) * 1000:+.1f} mm)")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
