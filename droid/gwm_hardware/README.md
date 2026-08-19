# `gwm_hardware/` — the real-robot rig

Everything specific to running on the hardware rig at `zhiwei`. Kept apart from
`gwm_tiptop/` (the droid-sim integration, whose results are recorded in
`gwm_integrate_doc/plan.md` G-19…G-32) so those sim results stay reproducible
from an unchanged tree.

## The rig

| | |
|---|---|
| Arm | Franka Emika **Panda**, robot system version 4.2.2 |
| Gripper | Robotiq **2F-140** (the stack elsewhere assumes a 2F-85) |
| Controller | Bamboo on a separate PREEMPT_RT machine, **libfranka 0.9.2** |
| Cameras | RealSense D435i `035422072950` (wrist) + D435 `348522073586` (external), spare D435i `134322070906` |
| GPU workstation | RTX 5090 32 GB, Ubuntu 22.04 |

## Two experiments, one rig

```
gwm_hardware/
  common/       the RIG: robot model, calibration, cameras, workspace, the
                five installers that deviate the pristine tiptop tree
  tiptop_arm/   baseline TiPToP (Gemini + SAM2 + cuTAMP) — the A/B control
  gwm_arm/      GWM x TiPToP — geometric perception + M2T2 + cuTAMP proposals
                scored by GWM. The method under test
  assets/       generated, gitignored — machine-local URDF / cuRobo yml / spheres
  config/       versioned — tiptop.yml (symlinked into the tiptop tree), extrinsics
  docs/
```

`gwm_arm` imports `common`; it never imports `tiptop_arm`, and neither arm
imports the other. The **method** lives in `droid/gwm_tiptop/` and is shared
with droid-sim unmodified — `gwm_arm` is only the rig plumbing the method needs
on real hardware and does not need in sim (live capture, the external camera's
extrinsics, the renderer overlay gate, the run driver, the debug viewer).

## Where to start

| Doc | For |
|---|---|
| [`docs/hardware-bringup.md`](docs/hardware-bringup.md) | bringing the rig up from nothing |
| [`docs/gwm-arm.md`](docs/gwm-arm.md) | running the GWM arm — status, commands, what is still blocked on the robot |
| [`docs/tiptop-modifications.md`](docs/tiptop-modifications.md) | **the reproduction record**: every deviation this rig makes to the pristine `droid/tiptop/` worktree, which installer makes it, and whether it affects droid-sim. Read it before running anything here or in droid-sim-evals |
| [`docs/tcp-convention.md`](docs/tcp-convention.md) | where `grasp_frame` goes, and why Franka Desk's TCP is a separate unused number — read before changing any end-effector offset |
| [`docs/bamboo-handover.md`](docs/bamboo-handover.md), [`docs/rc-handover-2.md`](docs/rc-handover-2.md), [`docs/rc-handover-4.md`](docs/rc-handover-4.md) | the controller machine |

## Contents

### `common/` — the rig

| File | What it is |
|---|---|
| `paths.py` | where `assets/` and `config/` live; import these rather than recomputing `__file__`-relative paths |
| `build_2f140.py` | generates the Panda + 2F-140 **URDF** — arm reused verbatim from cuTAMP's 2F-85 model, gripper expanded from cuTAMP's own `robotiq_2f_140.xacro` |
| `build_2f140_cfg.py` | generates the matching **cuRobo config**: collision spheres fitted to the 2F-140 meshes, self-collision table, mimic/locked joints, cspace |
| `robot_2f140.py` | loads that model with `cutamp.robots.franka_robotiq`'s API shape |
| `validate_2f140.py` | kinematics / TCP / gripper-actuation / IK / motion-planning checks |
| `rig_workspace.py` + `install_rig_workspace.py` | this bench's obstacles, over tiptop's MIT-LIS default |
| `install_rig_config.py` | symlinks `config/tiptop.yml` into the tiptop tree |
| `install_2f140_cutamp.py` | points cuTAMP's `panda_robotiq` at the 2F-140 |
| `install_charuco_params.py` | points `calibrate-wrist-cam` at this rig's 11x8 / 34.31 mm board |
| `install_client_lifetime_fix.py` | `go_to_q` closed the shared Bamboo client |
| `build_gripper_mask.py` | the wrist gripper mask, segmented by depth |
| `measure_charuco.py` | measures the board's checker size with the depth camera |
| `find_capture_pose.py` | plan/move/look loop that settled `q_capture` |
| `aim_camera.py` | live preview with framing guides, for aiming the external camera |
| `check_calibration.py` | hand-eye residual diagnostic |
| `rs_preflight.py` | puts the RealSense IR pair into a state FoundationStereo can use |
| `warm_servers.py` | absorbs the first-call PTX JIT cost on M2T2 / FoundationStereo |

### `tiptop_arm/` — the baseline

| File | What it is |
|---|---|
| `services.sh` | servers + preflight + warm-up, then `tiptop-run` |
| `inspect_plan.py` | where a saved plan actually puts the fingers |
| `viz_grasp.py` | the same question, drawn to scale |

### `gwm_arm/` — the method under test

| File | What it is |
|---|---|
| `capture.py` | rig observation → the wrist / external h5 files `gwm_tiptop` reads; also replays a saved baseline run, so the whole arm can be exercised with the robot off |
| `extcam_calib.py` | the external camera's extrinsics in the base frame, via the wrist camera and a Charuco board |
| `render_model.py` | the RENDER-only 2F-140 URDF (SAPIEN rejects cuTAMP's `panda_link3` inertia) |
| `propose.py` | the hardware pick proposer — `gwm_tiptop`'s method with the plane-normal table cut and the rig's keep-out volumes |
| `overlay_gate.py` | the hard gate: does the rendered robot land on the real one |
| `viz_debug.py` | every candidate drawn and coloured by its GWM score; the Rerun 3D view |
| `execute.py` | run a selected plan, with the checks an offline plan needs |
| `run_real.py` | capture → propose → score → gate → viz → execute |
| `services.sh` | the shared stack plus `gwm-server` |

cuTAMP is never forked and `tiptop/` is never patched by hand (G-4 / G-7 /
G-18); this package sits alongside them and is resolved through a
site-packages symlink (G-21 — never a `.pth`).

## Why a separate gripper model at all

cuTAMP ships only `panda_robotiq_2f_85.{urdf,yml}` and
`get_robotiq_2f_85_gripper_spheres`, and tiptop's `panda_robotiq` embodiment
resolves to those. Measured on the generated models:

```
2F-85   grasp_frame 150.0 mm from the gripper base  (= its fingertip plane, open: 149.3 mm)
2F-140  fingertip plane, open                        212.0 mm
```

So planning a 2F-140 with the 2F-85 model would drive the real fingers
**62 mm** past where the planner thinks they are, on every grasp, and check
collisions against a gripper 62 mm too short. That is the whole reason this
directory exists — and the same 62 mm is why the GWM arm renders the 2F-140
rather than the 2F-85 `real_data_train/renderer/assets.py` welds.

## Regenerate + verify

The generated model derives from the gitignored cuTAMP/cuRobo clones, so
re-run this after any `install-cutamp` / `install-curobo`:

```bash
cd /home/quanyi/gwm-wiser
export PATH="$HOME/.pixi/bin:$PATH"
P="pixi run --manifest-path droid/tiptop/pixi.toml python"
$P -m gwm_hardware.common.build_2f140
$P -m gwm_hardware.common.build_2f140_cfg
$P -m gwm_hardware.common.validate_2f140          # expect 10/10
```

Last run: **10/10 checks passed** — TCP 212.0 mm (+62.0 mm vs 2F-85), gripper
136 → 7 mm through the mimic chain, IK error 0.00 mm, `grasp_frame`
orientation matching the 2F-85 convention.

## Open items

- **The flange standoff is 0.** The gripper is bolted straight to the flange on
  this rig, so `--flange-offset` defaults to 0; recover the true value from the
  renderer overlay gate (`gwm_arm/overlay_gate.py`) if the rendered gripper
  sits proud of the real one.
- **The 2F-140 is out of distribution for GWM's scorer.** Its training corpus
  (MolmoAct2-DROID, MolmoBot) is all 2F-85, so the robot-only RAT frames will
  show a gripper the model has not seen. Accepted deliberately; watch the
  selection margins on the first real-robot runs.
- **~3 deg hand-eye rotational residual**, worth 10-20 mm at the edges of the
  capture footprint. Systematic, reproducible, and not fixed by recalibration;
  see the end of `docs/tiptop-modifications.md`. It is also what forced the
  GWM arm's plane-normal table cut (`gwm_arm/propose.py`).
- **One usable external camera.** The sim's best configuration was two-camera
  score fusion (G-30); the spare D435i is currently in a backlit window
  position and fails `rs_preflight` on IR saturation. Moving it would buy back
  that configuration — `score_client --cam` already takes a comma-separated
  list.
- **No droid-sim regression run.** droid-sim's robot is a 2F-85, and IsaacSim
  5.0 does not run on this GPU (its torch 2.7.0+cu126 has no `sm_120` cubin and
  no PTX fallback). Skipped by user decision; the shared-code changes the GWM
  arm needed were instead verified bit-identical on saved sim captures, see
  `docs/gwm-arm.md`.
