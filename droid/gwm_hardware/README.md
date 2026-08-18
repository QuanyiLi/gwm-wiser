# `gwm_hardware/` — the real-robot rig

Everything specific to running GWM × TiPToP on the hardware rig at `zhiwei`.
Kept apart from `gwm_tiptop/` (the droid-sim integration, whose results are
recorded in `gwm_integrate_doc/plan.md` G-19…G-32) so those sim results stay
reproducible from an unchanged tree.

## The rig

| | |
|---|---|
| Arm | Franka Emika **Panda**, robot system version 4.2.2 |
| Gripper | Robotiq **2F-140** (the stack elsewhere assumes a 2F-85) |
| Controller | Bamboo on a separate PREEMPT_RT machine, **libfranka 0.9.2** |
| Cameras | RealSense D435i `035422072950` (wrist) + D435 `348522073586` (external) |
| GPU workstation | RTX 5090 32 GB, Ubuntu 22.04 |

**[`docs/tiptop-modifications.md`](docs/tiptop-modifications.md) is the reproduction record**: every deviation this rig makes to the pristine `droid/tiptop/` worktree, which installer makes it, and whether it affects droid-sim. Read it before running anything here or in droid-sim-evals.

Start with [`docs/hardware-bringup.md`](docs/hardware-bringup.md); the
controller machine gets [`docs/bamboo-handover.md`](docs/bamboo-handover.md)
then [`docs/rc-handover-2.md`](docs/rc-handover-2.md).
[`docs/tcp-convention.md`](docs/tcp-convention.md) settles where `grasp_frame`
goes and why Franka Desk's TCP is a separate, unused number — read it before
changing any end-effector offset.

## Contents

| File | What it is |
|---|---|
| `build_2f140.py` | generates the Panda + 2F-140 **URDF** — arm reused verbatim from cuTAMP's 2F-85 model, gripper expanded from cuTAMP's own `robotiq_2f_140.xacro` |
| `build_2f140_cfg.py` | generates the matching **cuRobo config**: collision spheres fitted to the 2F-140 meshes, self-collision table, mimic/locked joints, cspace |
| `robot_2f140.py` | loads that model with `cutamp.robots.franka_robotiq`'s API shape, so anything taking a cuTAMP loader can take this |
| `validate_2f140.py` | kinematics / TCP / gripper-actuation / IK / motion-planning checks |
| `rig_workspace.py` + `install_rig_workspace.py` | this bench's obstacles, over tiptop's MIT-LIS default |
| `install_rig_config.py` | symlinks `config/tiptop.yml` into the tiptop tree |
| `install_charuco_params.py` | points `calibrate-wrist-cam` at this rig's 11x8 / 34.31 mm board |
| `measure_charuco.py` | measures the board's checker size with the depth camera |
| `find_capture_pose.py` | plan/move/look loop that settled `q_capture` |
| `aim_camera.py` | live preview with framing guides, for aiming the external camera |
| `rs_preflight.py` | puts the RealSense IR pair into a state FoundationStereo can use |
| `warm_servers.py` | absorbs the first-call PTX JIT cost on M2T2 / FoundationStereo |
| `assets/` | **generated, gitignored** — machine-local URDF + yml (absolute mesh paths) |

cuTAMP is never forked and `tiptop/` is never patched (G-4 / G-7 / G-18); this
package sits alongside them and is resolved through a site-packages symlink
(G-21 — never a `.pth`).

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
directory exists.

## Regenerate + verify

The generated model derives from the gitignored cuTAMP/cuRobo clones, so
re-run this after any `install-cutamp` / `install-curobo`:

```bash
cd /home/quanyi/gwm-wiser
export PATH="$HOME/.pixi/bin:$PATH"
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.build_2f140
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.build_2f140_cfg
pixi run --manifest-path droid/tiptop/pixi.toml python -m gwm_hardware.validate_2f140
```

Last run: **9/9 checks passed** — TCP 212.0 mm (+62.0 mm vs 2F-85), gripper
136 → 7 mm through the mimic chain, IK error 0.00 mm, motion planning
successful at 1.94x the stock 2F-85's cost (92 ms vs 48 ms, median of 5).

## Open items

- **The flange standoff is 0.** The gripper is bolted straight to the flange on
  this rig, so `--flange-offset` defaults to 0; recover the true value from the
  renderer overlay gate if the rendered gripper sits proud of the real one.
- **The 2F-140 is out of distribution for GWM's scorer.** Its training corpus
  (MolmoAct2-DROID, MolmoBot) is all 2F-85, so the robot-only RAT frames will
  show a gripper the model has not seen. Accepted deliberately; watch the
  selection margins on the first real-robot runs.
- **~3 deg hand-eye rotational residual**, worth 10-20 mm at the edges of the
  capture footprint. Systematic, reproducible, and not fixed by recalibration;
  see the end of `docs/tiptop-modifications.md`. Left open on purpose — running
  tiptop answers whether it matters faster than more metrology does.
- **No droid-sim regression run.** droid-sim's robot is a 2F-85
  (`franka_robotiq_2f_85_flattened.usd`), so it could only have validated the
  stack, not this gripper — and IsaacSim 5.0 does not run on this GPU anyway
  (its torch 2.7.0+cu126 has no `sm_120` cubin and no PTX fallback). Skipped by
  user decision; the stack gets exercised instead by the phase-D exit criterion
  (baseline `tiptop-run` succeeding three times in a row on the real robot).
