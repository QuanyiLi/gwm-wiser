# gwm_rl — pick up the bowl with FlashSAC, in the drawer scene

The scene8 drawer scene (three cabinets, a bowl, a block and a banana on the
DROID table) as an Isaac Lab environment, and FlashSAC trained on "pick up the
bowl" with a macro-action end-effector action space. This is the un-guided
baseline; the hooks for GWM-guided exploration are in `capture.py`.

## Layout

```
geometry.py      scene numbers (table, cabinets, objects, camera rig, workspace
                 box, robot/joint/body names) — pure python, no Isaac
franka_kin.py    Panda FK / Jacobian / DLS-IK / EE interpolation in torch
robot.py         DROID Franka + Robotiq 2F-85 articulation cfg
scene.py         InteractiveSceneCfg: robot, kinematic table + cabinets, bowl,
                 block, banana, floor, lights, 8 contact sensors
task.py          TaskParams, rim-grasp predicates, staged reward, collision
                 cost accounting, observation fields, per-env episode state
mdp.py           manager terms (observation / reward / reset events)
env_cfg.py       ManagerBasedRLEnvCfg + gym id `GwmRl-PickBowl-v0`; the raw
                 action is [7 absolute joint targets, gripper] at 15 Hz
executors.py     MacroExecutor: the target-pose action space over that env
flash_sac_*.py   FlashSAC adapter + config; flash_rl/ = vendored FlashSAC (MIT)
train.py         trainer; eval.py (checkpoint evaluation); report.py
                 (sim-free reader of evals.json); smoke.py (bring-up checks)
capture.py       GWM hook: photo + camera model + candidate format
configs/macro.yaml   the recipe
assets/          make_assets.py -> *.usda, fetch_assets.sh -> ycb/
slurm/train.sbatch   one run, with a startup watchdog
tests/test_kin.py    sim-free checks
```

A run lands in `experiments/<name>/` (`GWM_RL_EXPERIMENT_ROOT` overrides):
`config.yaml`, `evals.json` (history, milestones, cost totals), `models/best/`.

## Running

Requirements: Python 3.11, `isaaclab[all,isaacsim]==2.2.0` (pip,
`--extra-index-url https://pypi.nvidia.com`), torch 2.7 cu118, gymnasium 1.2,
numpy 1.26, `pyyaml`, `pillow`, `matplotlib`; a driver with CUDA 12 support.
Physics-only training never renders; cameras (`capture.py` / `capture_envs`)
need RTX rendering, which works headless on an H100 via a slower path.

```bash
bash assets/fetch_assets.sh && python assets/make_assets.py     # once: YCB meshes + textures, usda files
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export ISAAC_PYTHON=/path/to/isaac/python

$ISAAC_PYTHON smoke.py --headless --num_envs 4              # build, FK check, scripted grasp, sensors
$ISAAC_PYTHON train.py --headless --seed 1 --final-eval-episodes 4
sbatch slurm/train.sbatch 1                                   # the same with the startup watchdog
$ISAAC_PYTHON eval.py experiments/pickbowl-s1 --headless --episodes 4
$ISAAC_PYTHON report.py 'experiments/pickbowl-*' --plot experiments/curves.png
$ISAAC_PYTHON capture.py --headless --out captures/home     # photo + camera model for the GWM server
python tests/test_kin.py                                      # sim-free checks
```

Protocol: 5 seeds at 2048 envs, 700 s wall each; `report.py` prints mean ± sd
of `success_at_end` and of the ticks to each success level. One PhysX GPU
process per GPU. The scripts hard-exit (`os._exit`) because Kit's shutdown
hangs after this scene; run python with `-u` under redirection.

## Environment

- Robot: DROID Franka + Robotiq 2F-85, 15 Hz control (PhysX 120 Hz,
  decimation 8), gravity off, stiffness 400 / damping 80. Episode 220 ticks
  = 14.67 s = 5 macro-steps. Randomization: N(0, 0.05 rad) on the arm joints
  at reset; objects reset to their settled scene8 poses.
- Scene: table and cabinets are kinematic compound bodies (drawers closed and
  fixed); bowl (YCB 024, convex decomposition, 0.235 kg, friction 1.3 / 1.2),
  block (4.7 cm, 20 g) and banana are free bodies with velocity caps (3 m/s,
  20 rad/s, depenetration 1 m/s); an invisible floor collider 2 cm below the
  table's legs.
- Observation (60): joint pos/vel (8+8) + pad opening, TCP pose (3 + rot6d),
  the two pad positions, bowl pose + twist (clamped), grasp target − TCP and
  the rim radial, phase bits (grasped, lifted, dwell fraction, bowl height,
  grasp depth), pad contact bits.
- Success: bowl centre ≥ 0.10 m above its settled height with both pads in
  contact, 8 consecutive ticks. `success_at_end` (the metric) = true on the
  episode's last tick; `success_once` = ever true.
- Reward (`task.pick_reward`): staged reach (distance to the grasp target =
  nearest rim point 1.2 cm below the rim plane; tool pointing into the bowl,
  pads across the rim wall) → grasp (bonus scaled by depth reached) → lift
  (linear in height) → hold + dwell → success 10 + margin, all /10. No
  collision term.
- Collision cost, logged only: `contact_<obstacle>` sensors on the table, the
  three cabinets, block and banana, filtered against the 17 robot bodies; per
  episode the integrated force (`obstacle_impulse`, N·s), ticks above 1 N
  (`contact_ticks`), peak force, block/banana displacement. `evals.json`
  keeps per-window means and running totals (`cost_explore`, `cost_sweep`).
  `--env-set task_params.collision_penalty=<w>` makes it a reward penalty.
- Action (`executors.MacroExecutor`): `(x, y, z, yaw, g)` in [−1, 1]⁵ →
  an absolute target pose in `geometry.WORKSPACE` (x 0.20–0.75, y ±0.45,
  z table+0.03…0.60, yaw ±90°); 30 ticks of straight-line EE motion (one
  DLS-IK step per tick) + 14 ticks holding while the gripper switches to `g`
  (`g < 0` closes). Reward = mean over the 14 hold ticks. Yaw is modulo π.
  `MacroExecutor.decode(action)` → `(p_t, yaw_t, close)`.
- `task.TaskState.predicates` zeroes contact forces of envs with
  `episode_length_buf == 0` (the sensors hold the pre-reset step's data).
  `FlashSacWrapper._guard` sanitizes non-finite observations / rewards.

## GWM hook

`droid/server/gwm_server.py` scores candidates = joint trajectories
`(positions (T,7), t (T,), gripper (T,) in [0,1])` against one external
photo + intrinsics + `world_from_cam` (OpenCV axes, robot base at the origin)
and an instruction, rendering the robot with
`droid/gwm_tiptop/assets/panda_robotiq_droidsim.urdf` (the same 2F-85).

- `capture.camera_view(env, "external_cam", i)` returns the photo and both
  camera matrices for env `i` in the server's conventions. Cameras
  (`geometry.CAM_POSE`, 1280 × 720) exist only when the env cfg was built
  with `capture_envs=K` (envs 0..K−1; launch with `--enable_cameras`; keep
  K small).
- `capture.macro_candidate(q0, target_xyz, yaw, close)` turns one macro
  action into the server's candidate through `franka_kin.plan_macro` — the
  same interpolation + IK the executor runs. One sample per tick at 15 Hz;
  the gripper switches at tick 31.
- `capture.score_request(view, instruction, candidates, rat_scale=1.0)`
  builds the request body.

The seam for guided exploration is `train.py::interact`: for env `i < K`,
at a macro-step boundary, photograph, pick a target pose with the server's
scores, execute it through `MacroExecutor.step` and store the transition as
usual (FlashSAC is off-policy). `agent.actor_bc_alpha` (0 by default) pulls
the actor toward those transitions.
