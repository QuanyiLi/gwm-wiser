# Every modification this rig makes to `droid/tiptop/`

`droid/tiptop/` is a pristine upstream worktree by policy (G-18/G-21), and the
droid-sim results in `gwm_integrate_doc/plan.md` are reproducible from it
unchanged. A hardware rig cannot leave it completely untouched — tiptop reads
its robot config, its calibration and its workspace from files inside the
package — so every deviation is made by a **versioned, idempotent installer**
in this directory, each with `--restore`.

This file is the complete list. Reading it plus running the installers should
reproduce the rig from a fresh clone.

## Rebuild from scratch, in order

```bash
cd /home/quanyi/gwm-wiser
export PATH="$HOME/.pixi/bin:$PATH"
P="pixi run --manifest-path droid/tiptop/pixi.toml"

# 0. environments (see droid/README.md "Rig 2 — zhiwei" for the three gotchas)
#    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TIPTOP, the pinned cuRobo, sm_120.

# 1. package resolution: site-packages symlinks, never a .pth (G-21)
SP=$(droid/tiptop/.pixi/envs/default/bin/python -c 'import site; print(site.getsitepackages()[0])')
ln -sfn $PWD/droid/gwm_tiptop  "$SP/gwm_tiptop"
ln -sfn $PWD/droid/gwm_hardware "$SP/gwm_hardware"

# 2. the Panda + Robotiq 2F-140 model (cuTAMP ships only a 2F-85)
$P python -m gwm_hardware.build_2f140
$P python -m gwm_hardware.build_2f140_cfg
$P python -m gwm_hardware.validate_2f140          # expect 10/10

# 3. the five tiptop deviations
$P python -m gwm_hardware.install_rig_config      # tiptop.yml  -> symlink
$P python -m gwm_hardware.install_rig_workspace   # workspace.py dispatch
$P python -m gwm_hardware.install_2f140_cutamp    # cuTAMP robot model
$P python -m gwm_hardware.install_charuco_params --checker-mm 34.31
$P python -m gwm_hardware.install_client_lifetime_fix   # go_to_q closes the shared client

# 4. rig calibration (needs the robot and a human)
#    pixi run calibrate-wrist-cam        -> config/assets/calibration_info.json
$P python -m gwm_hardware.build_gripper_mask --install
```

---

## 1. `tiptop/config/tiptop.yml` — replaced by a symlink

**Installer:** `install_rig_config.py`  ·  **Backup:** `config/tiptop.yml.upstream`

`tiptop.config` reads this path and offers no override flag or env var, and
`tiptop-config` writes straight back to it. Rather than let rig values live in
the pristine tree, the file is replaced by a symlink to
`gwm_hardware/config/tiptop.yml`, so the tiptop worktree carries a one-line
pointer and the content is versioned here. `tiptop-config` writes through the
symlink into our copy.

Values that differ from upstream: `robot.type: panda_robotiq`, `robot.host`
(the RT machine, `192.168.68.132`), the two RealSense serials,
`time_dilation_factor: 0.2`, and `q_capture`.

**droid-sim impact:** none for `host`/`cameras`/`q_capture` — the sim drives
tiptop over the websocket and supplies its own observation and `q_init`.
`robot.type` **is** read (`tiptop_websocket_server.py:81`); see §3.

## 2. `tiptop/workspace.py` — one dispatch line

**Installer:** `install_rig_workspace.py`  ·  **Backup:** `workspace.py.orig`

`workspace_cuboids()` sends `panda_robotiq` to `fr3_workspace()`, which is MIT
LIS's bench: a Vention table, a wall, an iPad, a camera pillar. On this rig
that geometry is wrong in both directions — it invents obstacles that are not
here and omits the table edges that are. The dispatch is redirected to
`gwm_hardware.rig_workspace.zhiwei_workspace`; the geometry itself lives here,
so the diff in the tiptop tree is three lines.

**droid-sim impact: none.** `tiptop_websocket_server` defaults to
`include_workspace=False`, so the sim path never calls `workspace_cuboids()`.

### The tabletop must be sunk 20 mm — `rig_workspace.TABLE_COLLISION_SINK`

Non-obvious and it silently kills every pick. tiptop puts its **own** table in
the collision world, from the RANSAC fit of the live cloud, and deliberately
sinks it 20 mm below the detected surface (`segmentation.py:237`,
`... - extents[2] / 2 - 0.02`). That 20 mm is the clearance a top-down grasp
needs: the fingers close around an object *resting on* the surface, so a
collision table flush with the surface makes every grasp a collision.

Our slab is a **second** table on top of that one. Built to `TABLE_TOP_Z` it
sat 20 mm higher than the table tiptop had just carved clearance into, and
re-blocked exactly the gap. First `tiptop-run` on this rig: perception fine,
224 particles satisfied the constraints, **zero** survived motion refinement —
every one `MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION` on the retract,
whose start state is the grasp configuration.

Fixed by sinking our slab by the same 20 mm, so the detected table governs the
surface and ours only owns the volume below it and the region outside whatever
the camera saw.

## 3. cuTAMP `robots/franka_robotiq.py` — the gripper model ⚠️

**Installer:** `install_2f140_cutamp.py`  ·  **Backup:** `franka_robotiq.py.orig`

The rig carries a Robotiq **2F-140**; cuTAMP ships only a 2F-85, and tiptop's
`panda_robotiq` embodiment resolves to it. Measured on the generated models,
the 2F-140's TCP sits **62 mm** further from the flange, so the stock model
would drive the real fingers 62 mm past where the planner believes they are on
every grasp. Two literals are redirected: the config path, and the gripper
spheres cuTAMP uses to reject grasps that collide with the target.

cuTAMP is not forked — G-4 holds, the algorithm is untouched, only which robot
it loads. The clone is gitignored and rebuilt by `install-cutamp.sh`, so this
is a replayable install step; **re-run it after any cuTAMP reinstall.**

`urdf_path` in the generated `panda_robotiq_2f_140.yml` is **absolute**, and
has to be: cuTAMP's `load_panda_robotiq_rerun()` resolves it against its own
`robots/assets/` dir, where our model does not live, so a bare filename makes
`tiptop-run` die in `get_robot_rerun()` right after the instruction prompt.
Both cuRobo's `join_path` and `pathlib` drop the prefix when the suffix is
absolute, so one absolute path satisfies every consumer.

**⚠️ droid-sim impact: this is the one that matters.** The redirect is global
to `panda_robotiq`, and the sim reads the same robot type while its simulated
robot is a 2F-85 (`franka_robotiq_2f_85_flattened.usd`). Escape hatch:

```bash
GWM_TIPTOP_GRIPPER=2f85 <your droid-sim command>
```

which restores upstream behaviour without un-patching (verified: default gives
`panda_robotiq_2f_140.urdf`, spheres reaching 214.5 mm; with the variable set,
`panda_robotiq_2f_85.urdf`, 130.0 mm). The 2F-140 is the default because the
failure directions are not symmetric — a sim run with the wrong gripper gives
bad numbers, a hardware run with the wrong gripper drives into the table.

## 4. `tiptop/scripts/calibrate_wrist_cam.py` — board params and sweep

**Installer:** `install_charuco_params.py`  ·  **Backup:** `calibrate_wrist_cam.py.orig`

Upstream hard-codes DROID's board (14x9, 20 mm checker, 15 mm marker). This rig
has **11x8, DICT_5X5_100**, identified from a photo with the aruco detector
(44 markers, all 70 interior corners recovered) rather than by counting.

Its checker size was measured **with the depth camera**, because the tape
readings disagreed by 0.8 % and that lands directly in the hand-eye solve:
unprojecting the detected corners with metric depth gives **34.31 mm**
(FoundationStereo 34.31 ± 0.04, RealSense ASIC 34.32 ± 0.05 over 5 shots).

Also raised `calibration_traj`'s `angle_scale` from 0.2 to 0.45 (wrist sweep
±8° → ±17°): hand-eye conditions its rotational part on the rotation *between*
poses, and ±8° conditions it poorly.

**droid-sim impact: none.** Sim never calibrates a physical camera.

## 5. `tiptop/config/assets/calibration_info.json` — one added entry

Written by `calibrate-wrist-cam`, keyed by camera serial. Our entry is
`035422072950`; the two pre-existing keys are upstream ZED serials and are
untouched. **droid-sim impact: none** — the sim supplies camera poses in its
observation.

## 6. `tiptop/config/assets/gripper_mask.png` — replaced

**Builder:** `build_gripper_mask.py --install`  ·  **Backup:** `config/gripper_mask.png.upstream`

tiptop zeroes the cloud wherever this is True. The shipped mask is DROID's
2F-85 + ZED silhouette covering **20.6 %** of the frame; on this rig that
region is clean tabletop, so it silently deleted a fifth of the scene.

`compute-gripper-mask` (Gemini + SAM) does not apply either, because the 2F-140
is barely in the wrist frame at all. The replacement is segmented **by depth**:
the table sits at 0.61 m and the gripper at 0.15-0.25 m, a 3x separation, so a
threshold cuts them apart observationally. Result: **0.74 %**, two blobs at
image x 303-397 and x 1240-1280, matching where the fingers actually graze the
frame edge.

**droid-sim impact: none.** `tiptop_websocket_server.py:250` passes
`gripper_mask=None`.

## 7. `tiptop/motion_planning.py` — `go_to_q` closed the shared client

**Installer:** `install_client_lifetime_fix.py`  ·  **Backup:** `motion_planning.py.orig`

`go_to_q` called `client.close()` right after executing its trajectory.
`tiptop.utils.get_robot_client` is `@cache`d, so that is the same
`BambooFrankaClient` object `tiptop_run` holds as `container.robot`, and
`close()` is not a soft close — it does `zmq_context.term()`, and the gripper
path has no recreate logic. So:

```
go_to_capture(...)              # closes the shared client
container.robot.open_gripper()  # zmq.error.ZMQError: Socket operation on non-socket
```

This blocks `tiptop-run` outright — it dies before the instruction prompt. The
traceback is invisible because `_sync_entrypoint`'s `finally` calls
`sys.exit(exit_code)` with `exit_code` still 1, which replaces the propagating
exception: the symptom is a clean-looking exit 1 straight after
"Executed trajectory on the robot".

Harmless for `scripts/go_to_conf.py`, a one-shot CLI that exits immediately
afterwards, which is very likely where it came from. `tiptop_run` closes the
client itself in its own `finally` (`tiptop_run.py:817`).

**droid-sim impact: none.** The sim does not drive a Bamboo client.

---

## Baseline status

First successful pick on hardware, 2026-08-18 20:24 (`tiptop_outputs/eval/2026-08-18_20-24-39`):
`grasping the tomato` -> `holding(tomato)`, perception 7.0 s, plan found, executed
in 17.7 s at `time_dilation_factor` 0.2. Gemini, SAM2, FoundationStereo, M2T2,
cuTAMP and Bamboo all in the loop.

Check a plan before letting it execute -- `tiptop-run` goes straight from a
returned plan to `execute_cutamp_plan` with no confirmation
(`tiptop_run.py:680`):

```bash
pixi run tiptop-run --no-execute-plan       # then
$P python -m gwm_hardware.inspect_plan droid/tiptop/tiptop_outputs/eval/<timestamp>
```

`inspect_plan` reports where the finger **pads** land, not just the TCP. On the
run before this one the TCP looked reasonable (49 mm from the centre of a 100 mm
bowl, i.e. on the rim) while the pad over the bowl's opening sat 9 mm above the
rim and could never enter it; the plan would have pushed the bowl. The tomato
run scored 4 mm off centre with +16 mm of pad overlap on both sides.

Run it **interactively** from a terminal, not with piped stdin: after a
`Holding` goal it asks `Open gripper? [y]`, and that prompt is the only thing
holding the object up.

---

## Known unresolved issue: ~3° hand-eye rotational residual

Reconstructing the tabletop through FK → `ee_from_cam` → depth gives a plane
that is flat to **0.7-0.8 mm rms** but sits **2.9-3.0° off** the base frame's
vertical, ~5 mm high. Measured at several arm poses, the tilt's *magnitude*
stays constant while its *direction rotates with the wrist* — the signature of
a rotational error in the extrinsic, not a tilted table. Two independent
calibration runs agree with each other to 0.36 °, so it is systematic, not
noise.

Consequence: roughly **10-20 mm of position error** at the edges of the capture
footprint, less near the centre.

What was tried and did not fix it: re-running the calibration with the board
propped at an angle and the sweep widened to ±17° (changed the result by only
0.36°); solving for a single fixed rotation correction (the EE-frame normals
agree to 1.6°, not the <1° a pure fixed rotation would give); a six-pose joint
solve for extrinsic error *and* physical table tilt (diverged to a nonsense
98°, because poses at yaw +135/+180 swing part of the frame off the table edge
and corrupt the plane fit — `check_calibration.py` now drops any pose whose
plane fit keeps under 95 % of points).

Left open deliberately. The question that decides whether it matters is whether
grasps actually fail, and in a pattern consistent with a systematic height
error — edge positions scuffing or missing while centre positions succeed.
That is cheaper to answer by running tiptop than by more metrology.

Diagnostic: `python -m gwm_hardware.check_calibration --execute`.
