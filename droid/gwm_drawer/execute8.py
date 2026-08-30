"""execute8: run the six candidates in Isaac, record videos.

One boot, one episode per candidate: reset, settle, then track the candidate
timeline at 15 Hz (absolute joint targets; gripper binary from the timeline's
close), plus a 2 s hold at the end. Both external cameras are recorded to
mp4. Every drawer's displacement along its slide axis and every object's
rise are logged, so the physical outcome (which drawer opened / which object
was lifted, and that nothing else moved) is judged from sim state.

    cd /root/code/gwm/gwm-wiser/droid/gwm_drawer && \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
    ../droid-sim-evals/.venv/bin/python -u execute8.py
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "droid-sim-evals"))
sys.path.insert(0, str(HERE))

HOLD_S = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=8)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--tasks", default=None, help="comma-separated candidate names")
    args, _ = ap.parse_known_args()

    from config import DRAWERS, GRASP_LIFT, OBJECTS, RESULTS, apply_camera_rig

    cands = json.loads((RESULTS / "candidates.json").read_text())
    tasks = args.tasks.split(",") if args.tasks else list(cands)

    from isaaclab.app import AppLauncher

    kit_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(kit_parser)
    args_cli, _ = kit_parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.sim_utils import settle_sim

    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(args.scene), args.variant)
    apply_camera_rig(env_cfg.scene)
    env = gym.make("DROID", cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()

    def np0(t):
        return t[0].cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t[0])

    scene_h = env.unwrapped.scene
    axes = {k: np.asarray(cab.slide_axis()) for k, cab in DRAWERS.items()}
    prims = {**{k: f"{k}_drawer" for k in DRAWERS},
             **{k: spec["prim"] for k, spec in OBJECTS.items()}}

    def positions():
        return {k: np0(scene_h.rigid_objects[p].data.root_pos_w).copy()
                for k, p in prims.items()}

    outcomes = {}
    for task in tasks:
        cand = cands[task]["candidate"]
        P = np.asarray(cand["positions"], dtype=np.float64)
        T = np.asarray(cand["t"], dtype=np.float64)
        G = np.asarray(cand["gripper"], dtype=np.float64)

        obs, _ = env.reset()
        obs = settle_sim(env, obs, steps=60)
        spawn = positions()

        out_dir = RESULTS / "exec" / task
        out_dir.mkdir(parents=True, exist_ok=True)
        writers = {
            "external_cam": imageio.get_writer(out_dir / "ext.mp4", fps=15,
                                               codec="libx264", quality=8),
            "external_cam_2": imageio.get_writer(out_dir / "ext2.mp4", fps=15,
                                                 codec="libx264", quality=8),
        }
        n_steps = int((T[-1] + HOLD_S) * 15) + 1
        for k in range(n_steps):
            tk = min(k / 15.0, T[-1])
            q = np.array([np.interp(tk, T, P[:, j]) for j in range(7)])
            g = 1.0 if np.interp(tk, T, G) > 0.5 else 0.0
            action = torch.tensor(np.concatenate([q, [g]]), dtype=torch.float32,
                                  device=env.unwrapped.device).unsqueeze(0)
            obs, _, _, _, _ = env.step(action)
            for cam_name, w in writers.items():
                rgb = np0(scene_h.sensors[cam_name].data.output["rgb"])[..., :3].astype(np.uint8)
                w.append_data(rgb)
                if k in (0, n_steps - 1):
                    tag = "first" if k == 0 else "last"
                    Image.fromarray(rgb).save(out_dir / f"{cam_name}_{tag}.png")
        for w in writers.values():
            w.close()

        final = positions()
        moved = {}
        for k in DRAWERS:
            moved[k] = round(float(np.dot(final[k] - spawn[k], axes[k])), 4)
        for k in OBJECTS:
            moved[k] = round(float(final[k][2] - spawn[k][2]), 4)
        full = DRAWERS[task].pull if task in DRAWERS else GRASP_LIFT
        others = max(abs(moved[k]) for k in moved if k != task)
        outcomes[task] = {
            "kind": "drawer" if task in DRAWERS else "grasp",
            "moved": moved, "target_moved_m": moved[task],
            "target_frac": round(moved[task] / full, 3),
            "max_other_m": round(others, 4),
        }
        (out_dir / "displacements.json").write_text(json.dumps(outcomes[task], indent=2))
        what = "opened" if task in DRAWERS else "lifted"
        print(f"{task}: {what} {moved[task] * 100:.1f} cm "
              f"({moved[task] / full * 100:.0f}% of full), "
              f"max other body {others * 100:.1f} cm", flush=True)

    (RESULTS / "exec" / "outcomes.json").write_text(json.dumps(outcomes, indent=2))
    print(f"wrote {RESULTS / 'exec' / 'outcomes.json'}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
