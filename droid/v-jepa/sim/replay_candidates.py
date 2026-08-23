"""replay_candidates: execute candidate plans in the DROID Isaac sim and record what
the world-model evaluation needs -- the current frame, the frames along the
executed trajectory at the model's 4 fps cadence, the end frame (goal image /
oracle target), the executed robot states, and the judge verdict.

One Isaac boot per invocation; every candidate is a fresh reset + settle,
then a fixed-plan replay with the same stepping as the eval harness
(`TiptopWebsocketClient._step_plan`: 15 Hz control, waypoint stride 3,
20-step gripper actions, 30-step hold after the plan), so the recorded end
state is the one the scene-6 judges scored for the GWM arms.

Cameras: `external_cam_2` (the DROID-style view, the one GWM cam2 scored)
and `external_cam` are fetched every `--frame-stride` control steps (4 ->
3.75 fps, DROID's "4 fps"); all three cameras are saved at t=0 and after the
final hold. The default eval tier deletes external_cam_2; here all cameras
stay at full rate.

    cd /root/code/gwm/gwm-wiser/droid/v-jepa && \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
    ../droid-sim-evals/.venv/bin/python -u sim/replay_candidates.py \
        --variant 0 --plans-dir ../gwm_integrate_doc/proposals/scene6_rev2 \
        --out-dir runs/replay_pick

Output per candidate `<out>/<plan_stem>/`:
  frames_external_cam_2.npz   uint8 [F, 720, 1280, 3] frames at the stride cadence
  frames_external_cam.npz     same for external_cam
  frame_index.npy             control-step index of each frame (time = idx / 15 s)
  first_*.png / final_*.png   all three cameras at t=0 and after the hold
  traj.npz                    per control step: t, q_cmd, grip_cmd, q_meas,
                              finger_joint, link8 pos/quat (wxyz), object centres
  cameras.json                K, pos_w, quat_w_ros per camera
  judge.json                  lift / place verdicts for every object, phases, timing
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VJEPA_ROOT = HERE.parent
DROID = VJEPA_ROOT.parent
sys.path.insert(0, str(DROID / "droid-sim-evals"))
sys.path.insert(0, str(VJEPA_ROOT))

import numpy as np  # noqa: E402

from vjepa_sel.plan_stepper import (  # noqa: E402
    HOLD_STEPS_AFTER_PLAN,
    SIM_CONTROL_HZ,
    PlanStepper,
    load_plan,
)

LIFT_M = 0.15  # scene-6 pick judge: target mesh centre >= 0.15 m above its settled height
PLACE_XY_TOL = 0.05
PLACE_Z_BAND = (-0.03, 0.03)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=6)
    ap.add_argument("--variant", type=int, required=True, help="0 = pick scene, 1 = place scene (held block welded)")
    ap.add_argument("--plans-dir", required=True, type=Path)
    ap.add_argument("--plans", nargs="*", default=None, help="plan file names to run (default: all plan_*.json)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--frame-stride", type=int, default=4, help="control steps between recorded frames (4 -> 3.75 fps)")
    ap.add_argument("--episode-cap-s", type=float, default=90.0)
    ap.add_argument("--overwrite", action="store_true")
    args, _ = ap.parse_known_args()

    plan_files = sorted(args.plans_dir.glob("plan_*.json")) if not args.plans else [args.plans_dir / p for p in args.plans]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in plan_files if args.overwrite or not (args.out_dir / p.stem / "judge.json").exists()]
    print(f"[replay] {len(plan_files)} plans, {len(todo)} to run", flush=True)
    if not todo:
        return

    if args.variant == 1:
        # weld the held block at settle time, exactly as place_eval.py does
        sys.path.insert(0, str(DROID / "droid-sim-evals-ours"))
        import weld_held_block  # noqa: F401
    else:
        weld_held_block = None

    from isaaclab.app import AppLauncher

    kit_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(kit_parser)
    args_cli, _ = kit_parser.parse_known_args([])
    args_cli.enable_cameras = True
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import torch
    from PIL import Image

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from src.sim_evals.sim_utils import settle_sim

    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(args.scene), args.variant)
    env_cfg.episode_length_s = args.episode_cap_s
    env = gym.make("DROID", cfg=env_cfg)
    env.reset()
    env.reset()  # second render cycle for correct materials
    scene = env.unwrapped.scene
    robot = scene["robot"]
    max_steps = env.unwrapped.max_episode_length
    body_names = list(robot.body_names)
    link8 = body_names.index("panda_link8")
    finger_idx = [i for i, n in enumerate(robot.data.joint_names) if n == "finger_joint"][0]
    obj_names = list(scene.rigid_objects.keys())
    print(f"[replay] rigid objects: {obj_names}; bodies: {body_names}", flush=True)

    def np0(t):
        return t[0].detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t[0])

    def cam_rgb(name):
        return np0(scene.sensors[name].data.output["rgb"])[..., :3].astype(np.uint8).copy()

    def cam_meta(name):
        cam = scene.sensors[name]
        return {
            "K": np0(cam.data.intrinsic_matrices).tolist(),
            "pos_w": np0(cam.data.pos_w).tolist(),
            "quat_w_ros": np0(cam.data.quat_w_ros).tolist(),
            "quat_w_world": np0(cam.data.quat_w_world).tolist() if hasattr(cam.data, "quat_w_world") else None,
            "height": int(cam.cfg.height),
            "width": int(cam.cfg.width),
        }

    def link_pose(i):
        d = robot.data
        if hasattr(d, "body_link_pose_w"):
            p = d.body_link_pose_w[0, i].detach().cpu().numpy()
            return p[:3].copy(), p[3:7].copy()
        return d.body_pos_w[0, i].detach().cpu().numpy().copy(), d.body_quat_w[0, i].detach().cpu().numpy().copy()

    # mesh-centre offsets (USD bbox) per rigid object, as the harness judge does
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    offset_body = {}
    for n, obj in scene.rigid_objects.items():
        prim = stage.GetPrimAtPath(obj.cfg.prim_path.replace("env_.*", "env_0"))
        rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        center = np.array([(mn[i] + mx[i]) / 2 for i in range(3)])
        p = obj.data.root_pos_w[0].cpu().numpy()
        R = quat_to_mat(obj.data.root_quat_w[0].cpu().numpy())
        offset_body[n] = R.T @ (center - p)

    def object_centres():
        out = {}
        for n, obj in scene.rigid_objects.items():
            p = obj.data.root_pos_w[0].cpu().numpy()
            q = obj.data.root_quat_w[0].cpu().numpy()
            out[n] = p + quat_to_mat(q) @ offset_body[n]
        return out

    for plan_path in todo:
        t_wall = time.perf_counter()
        out = args.out_dir / plan_path.stem
        out.mkdir(parents=True, exist_ok=True)
        plan = load_plan(plan_path)
        print(f"[replay] === {plan_path.name} ===", flush=True)

        obs, _ = env.reset()
        obs = settle_sim(env, obs, reset_episode_buf=True)  # 100 steps; welds the block on variant 1
        q_init = np0(obs["policy"]["arm_joint_pos"]).astype(np.float32)
        q_drift = float(np.abs(q_init - plan["q_init"]).max())
        print(f"[replay] q_init max|drift| vs plan = {q_drift:.2e} rad", flush=True)

        cams = {n: cam_meta(n) for n in ("external_cam", "external_cam_2", "wrist_cam")}
        (out / "cameras.json").write_text(json.dumps(cams, indent=1))
        for n, tag in (("external_cam", "first_external_cam"), ("external_cam_2", "first_external_cam_2"), ("wrist_cam", "first_wrist_cam")):
            Image.fromarray(cam_rgb(n)).save(out / f"{tag}.png")

        centres0 = object_centres()
        stepper = PlanStepper(plan)
        rec = {k: [] for k in ("t", "q_cmd", "grip_cmd", "q_meas", "finger_joint", "link8_pos", "link8_quat", "phase", "label")}
        obj_rec = {n: [] for n in obj_names}
        frames = {"external_cam_2": [], "external_cam": []}
        frame_index = []
        capped = False
        ret = None
        with torch.no_grad():
            for k in range(max_steps):
                jp = np0(obs["policy"]["arm_joint_pos"])
                gp = np0(obs["policy"]["gripper_pos"])
                ret = stepper.step(jp, gp)
                if stepper.plan_done:
                    break
                if k % args.frame_stride == 0:
                    for n in frames:
                        frames[n].append(cam_rgb(n))
                    frame_index.append(k)
                p8, q8 = link_pose(link8)
                rec["t"].append(k / SIM_CONTROL_HZ)
                rec["q_cmd"].append(ret[:7].copy())
                rec["grip_cmd"].append(float(ret[7]))
                rec["q_meas"].append(jp.astype(np.float32))
                rec["finger_joint"].append(float(robot.data.joint_pos[0, finger_idx]))
                rec["link8_pos"].append(p8)
                rec["link8_quat"].append(q8)
                rec["phase"].append(stepper.phase)
                rec["label"].append(str(stepper.label))
                for n, c in object_centres().items():
                    obj_rec[n].append(c)
                obs, _, term, trunc, _ = env.step(torch.tensor(ret)[None])
                if term or trunc:
                    capped = True
                    print("[replay] WARNING: episode cap hit before plan end", flush=True)
                    break
                if weld_held_block is not None:
                    weld_held_block.maybe_release(scene)
            n_loop = len(rec["t"])
            if not capped:
                hold = torch.tensor(ret)[None]
                for h in range(HOLD_STEPS_AFTER_PLAN):
                    k = n_loop + h
                    if k % args.frame_stride == 0:
                        for n in frames:
                            frames[n].append(cam_rgb(n))
                        frame_index.append(k)
                    p8, q8 = link_pose(link8)
                    rec["t"].append(k / SIM_CONTROL_HZ)
                    rec["q_cmd"].append(ret[:7].copy())
                    rec["grip_cmd"].append(float(ret[7]))
                    rec["q_meas"].append(np0(obs["policy"]["arm_joint_pos"]).astype(np.float32))
                    rec["finger_joint"].append(float(robot.data.joint_pos[0, finger_idx]))
                    rec["link8_pos"].append(p8)
                    rec["link8_quat"].append(q8)
                    rec["phase"].append("hold")
                    rec["label"].append("hold")
                    for n, c in object_centres().items():
                        obj_rec[n].append(c)
                    obs, _, term, trunc, _ = env.step(hold)
                    if term or trunc:
                        capped = True
                        break
                    if weld_held_block is not None:
                        weld_held_block.maybe_release(scene)

        # final state (after the hold), the frame the judge looks at
        final_centres = object_centres()
        p8, q8 = link_pose(link8)
        for n, tag in (("external_cam", "final_external_cam"), ("external_cam_2", "final_external_cam_2"), ("wrist_cam", "final_wrist_cam")):
            Image.fromarray(cam_rgb(n)).save(out / f"{tag}.png")

        for n in frames:
            np.savez_compressed(out / f"frames_{n}.npz", frames=np.stack(frames[n]) if frames[n] else np.zeros((0, 720, 1280, 3), np.uint8))
        np.save(out / "frame_index.npy", np.asarray(frame_index, dtype=np.int64))
        np.savez_compressed(
            out / "traj.npz",
            t=np.asarray(rec["t"], np.float32),
            q_cmd=np.stack(rec["q_cmd"]).astype(np.float32),
            grip_cmd=np.asarray(rec["grip_cmd"], np.float32),
            q_meas=np.stack(rec["q_meas"]).astype(np.float32),
            finger_joint=np.asarray(rec["finger_joint"], np.float32),
            link8_pos=np.stack(rec["link8_pos"]).astype(np.float32),
            link8_quat=np.stack(rec["link8_quat"]).astype(np.float32),
            phase=np.asarray(rec["phase"]),
            label=np.asarray(rec["label"]),
            q_init=q_init,
            link8_pos_final=p8.astype(np.float32),
            link8_quat_final=q8.astype(np.float32),
            object_names=np.asarray(obj_names),
            object_centres=np.stack([np.stack(obj_rec[n]) for n in obj_names], axis=1).astype(np.float32),  # [N, n_obj, 3]
            object_centres_0=np.stack([centres0[n] for n in obj_names]).astype(np.float32),
            object_centres_final=np.stack([final_centres[n] for n in obj_names]).astype(np.float32),
        )

        # judge: which objects were lifted (pick rule), where the block is (place rule)
        lifted = {}
        for n in obj_names:
            z_rel = float(final_centres[n][2] - centres0[n][2])
            xy_move = float(np.linalg.norm(final_centres[n][:2] - centres0[n][:2]))
            lifted[n] = {"z_rel": round(z_rel, 4), "xy_moved": round(xy_move, 4), "lifted": z_rel >= LIFT_M}
        place = None
        if "held_block" in final_centres:
            b = final_centres["held_block"]
            per = {}
            for bin_name in [n for n in obj_names if n.endswith("_bin")]:
                c = final_centres[bin_name]
                xy = float(np.linalg.norm(b[:2] - c[:2]))
                zr = float(b[2] - c[2])
                per[bin_name] = {"xy": round(xy, 4), "z_rel": round(zr, 4),
                                 "inside": xy <= PLACE_XY_TOL and PLACE_Z_BAND[0] <= zr <= PLACE_Z_BAND[1]}
            inside = [n for n, d in per.items() if d["inside"]]
            place = {"per_bin": per, "landed_in": inside[0] if len(inside) == 1 else (inside or None),
                     "released": bool(getattr(weld_held_block, "_released", False)) if weld_held_block else None}
        close_steps = [i for i, g in enumerate(rec["grip_cmd"]) if g > 0.5]
        judge = {
            "plan": plan_path.name,
            "variant": args.variant,
            "q_init_drift_rad": q_drift,
            "capped": capped,
            "n_loop_steps": n_loop,
            "n_steps_total": len(rec["t"]),
            "duration_s": len(rec["t"]) / SIM_CONTROL_HZ,
            "close_t": (close_steps[0] / SIM_CONTROL_HZ) if close_steps else None,
            "n_frames": len(frame_index),
            "frame_stride": args.frame_stride,
            "lifted": lifted,
            "place": place,
            "wall_s": round(time.perf_counter() - t_wall, 1),
        }
        (out / "judge.json").write_text(json.dumps(judge, indent=1))
        summary = {n: d["z_rel"] for n, d in lifted.items() if abs(d["z_rel"]) > 0.02}
        print(f"[replay] RESULT {plan_path.name}: steps={len(rec['t'])} capped={capped} "
              f"moved={summary} place={place and place.get('landed_in')} wall={judge['wall_s']}s", flush=True)

    env.close()
    simulation_app.close()
    print("[replay] DONE", flush=True)


if __name__ == "__main__":
    main()
