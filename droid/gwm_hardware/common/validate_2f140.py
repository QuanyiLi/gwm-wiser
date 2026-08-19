"""Validate the generated Panda + Robotiq 2F-140 model against cuRobo.

Checks, in order of how badly each would bite on the real robot:

1. the config loads and cuRobo builds kinematics from it;
2. FK puts the TCP where the 2F-85 convention says it should be, and the
   2F-140/2F-85 TCP gap matches the measured 62 mm;
3. the gripper opens and closes through the mimic chain (a static gripper is
   the classic sign of a broken mimic map);
4. IK round-trips a reachable pose;
5. cuRobo actually plans a collision-free motion over a table -- and how long
   that takes relative to the stock 2F-85, since this model carries far more
   collision spheres.

    cd /home/quanyi/gwm-wiser
    pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.common.validate_2f140
"""

import time
from pathlib import Path

import numpy as np
import torch

from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from gwm_hardware.common import robot_2f140

from gwm_hardware.common.paths import CUTAMP_ASSETS
TCP_2F85_M = 0.150          # grasp_frame offset in cuTAMP's 2F-85 model
EXPECTED_TCP_2F140_M = 0.212
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def table_world():
    return WorldConfig(cuboid=[
        Cuboid(name="table", pose=[0.5, 0.0, -0.05, 1, 0, 0, 0], dims=[1.2, 1.2, 0.1]),
    ])


def main() -> None:
    print("1. config + kinematics")
    cfg = robot_2f140.curobo_cfg()
    kin_cfg = cfg["robot_cfg"]["kinematics"]
    n_spheres = sum(len(v) for v in kin_cfg["collision_spheres"].values())
    check("config loads", True,
          f"{len(kin_cfg['collision_link_names'])} collision links, {n_spheres} spheres")
    model = robot_2f140.get_kinematics_model()
    check("cuRobo kinematics builds", model is not None,
          f"ee_link={kin_cfg['ee_link']}, dof={len(model.joint_names)}")
    print(f"       joints: {model.joint_names}")

    print("2. TCP convention")
    from yourdfpy import URDF
    urdf = URDF.load(str(robot_2f140.ASSETS_DIR / "panda_robotiq_2f_140.urdf"),
                     load_meshes=False, build_scene_graph=True)
    urdf.update_cfg({robot_2f140.DRIVER_JOINT: robot_2f140.DRIVER_OPEN})
    tcp_z = float(urdf.get_transform("grasp_frame", "robotiq_140_base_link")[2, 3])
    check("TCP at the 2F-85's convention (fingertip plane, open)",
          abs(tcp_z - EXPECTED_TCP_2F140_M) < 1e-3,
          f"{tcp_z * 1000:.1f} mm (expected {EXPECTED_TCP_2F140_M * 1000:.1f})")
    check("TCP sits further out than the 2F-85, by the measured gap",
          abs((tcp_z - TCP_2F85_M) - 0.062) < 2e-3,
          f"+{(tcp_z - TCP_2F85_M) * 1000:.1f} mm vs 2F-85")

    print("2b. grasp_frame ORIENTATION vs the 2F-85 the pipeline was tuned on")
    # This is the check that would have caught the 90 deg bug of 2026-08-18.
    # None of the others did: the closing axes coincide at equal joint angles,
    # so the two models look identical side by side. The error only appears
    # once the planner is asked to satisfy a grasp pose -- it rotates the wrist
    # 90 deg and closes across the wrong axis of the object. Since
    # `m2t2_to_tiptop_transform()` is gripper-agnostic, grasp_frame's
    # orientation convention is not ours to choose.
    from scipy.spatial.transform import Rotation as _R
    q_neutral = {f"panda_joint{i+1}": v
                 for i, v in enumerate(robot_2f140.NEUTRAL_JOINT_POSITIONS)}
    ref = URDF.load(str(CUTAMP_ASSETS / "panda_robotiq_2f_85.urdf"),
                    load_meshes=False, build_scene_graph=True)
    ref.update_cfg({**q_neutral, "robotiq_85_left_knuckle_joint": 0.0})
    urdf.update_cfg({**q_neutral, robot_2f140.DRIVER_JOINT: robot_2f140.DRIVER_OPEN})
    R85 = ref.get_transform("grasp_frame", "panda_link8")[:3, :3]
    R140 = urdf.get_transform("grasp_frame", "panda_link8")[:3, :3]
    rel = np.degrees(np.linalg.norm(_R.from_matrix(R85.T @ R140).as_rotvec()))
    check("grasp_frame orientation matches the 2F-85 convention", rel < 1.0,
          f"{rel:.2f} deg apart (a 90 deg error here rotates every grasp)")

    print("3. gripper actuation through the mimic chain")
    spans = {}
    for label, driver in (("open", robot_2f140.DRIVER_OPEN),
                          ("closed", robot_2f140.DRIVER_CLOSED)):
        urdf.update_cfg({robot_2f140.DRIVER_JOINT: driver})
        pads = [urdf.get_transform(f"{s}_inner_finger_pad", "robotiq_140_base_link")[:3, 3]
                for s in ("left", "right")]
        spans[label] = float(np.linalg.norm(pads[0] - pads[1]))
        print(f"       pad separation {label:6s} = {spans[label] * 1000:6.1f} mm")
    check("gripper closes when the driver moves", spans["closed"] < spans["open"] - 0.05,
          f"{spans['open'] * 1000:.0f} -> {spans['closed'] * 1000:.0f} mm")
    check("open span is consistent with a 140 mm stroke",
          0.12 <= spans["open"] <= 0.16, f"{spans['open'] * 1000:.0f} mm")

    print("4. IK round-trip")
    ik = robot_2f140.get_ik_solver(table_world())
    target = Pose(
        position=torch.tensor([[0.45, 0.0, 0.35]], device=TensorDeviceType().device),
        quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=TensorDeviceType().device),
    )
    t0 = time.perf_counter()
    sol = ik.solve_batch(target)
    dt_ik = time.perf_counter() - t0
    ok = bool(sol.success.any())
    err = float(sol.position_error[sol.success].min()) if ok else float("nan")
    check("IK finds a collision-free solution", ok and err < 5e-3,
          f"position error {err * 1000:.2f} mm, {dt_ik * 1000:.0f} ms")

    print("5. motion planning, and its cost relative to the stock 2F-85")
    timings = {}
    for label, robot_cfg in (
        ("2F-140 (ours)", RobotConfig.from_dict(robot_2f140.curobo_cfg()["robot_cfg"])),
        ("2F-85 (cuTAMP)", None),
    ):
        if robot_cfg is None:
            stock = load_yaml(str(CUTAMP_ASSETS / "panda_robotiq_2f_85.yml"))
            for key in ("external_asset_path", "external_robot_configs_path"):
                stock["robot_cfg"]["kinematics"].setdefault(key, str(CUTAMP_ASSETS))
            robot_cfg = RobotConfig.from_dict(stock["robot_cfg"])
        mg = MotionGen(MotionGenConfig.load_from_robot_config(
            robot_cfg, table_world(), interpolation_dt=0.02))
        mg.warmup(warmup_js_trajopt=False)
        # The gripper driver is locked in both configs, so cuRobo exposes the
        # 7 arm joints only -- build the start state from its own joint list.
        start = JointState.from_position(
            torch.tensor([list(robot_2f140.NEUTRAL_JOINT_POSITIONS)
                          [:len(mg.kinematics.joint_names)]],
                         device=TensorDeviceType().device, dtype=torch.float32),
            joint_names=mg.kinematics.joint_names)
        # Median of repeated plans: the first call after warmup can still pay
        # JIT and allocation costs, which would flatter whichever model runs
        # second.
        samples, ok, n_wp = [], False, 0
        for _ in range(5):
            t0 = time.perf_counter()
            res = mg.plan_single(start, target, MotionGenPlanConfig(max_attempts=3))
            samples.append(time.perf_counter() - t0)
            ok = bool(res.success.item())
            if ok:
                n_wp = int(res.optimized_plan.position.shape[0])
        timings[label] = (ok, float(np.median(samples)), n_wp)
        del mg
        torch.cuda.empty_cache()
    for label, (ok, dt, n) in timings.items():
        print(f"       {label:16s} success={ok}  {dt * 1000:7.0f} ms  {n} waypoints")
    check("2F-140 plans a collision-free trajectory", timings["2F-140 (ours)"][0])
    ratio = timings["2F-140 (ours)"][1] / max(timings["2F-85 (cuTAMP)"][1], 1e-6)
    check("planning cost stays within 4x of the 2F-85 model", ratio < 4.0,
          f"{ratio:.2f}x ({timings['2F-140 (ours)'][1] * 1000:.0f} ms vs "
          f"{timings['2F-85 (cuTAMP)'][1] * 1000:.0f} ms)")

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
