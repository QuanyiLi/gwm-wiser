# droid-sim-evals-ours

Custom DROID-sim evaluation tasks for TiPToP, layered on [`../droid-sim-evals`](../droid-sim-evals)
(same IsaacLab `.venv`, same runner, same CSV schema — `grasp_eval.py` imports `batch_eval_v2`
and swaps in a lift-based success judge).

## Tasks (scene 1)

| task id | instruction | success rule |
|---|---|---|
| grasp_cube | pick up the cube | cube center ≥ 0.15 m above its settled height at trial end |
| grasp_bowl | pick up the bowl | bowl center ≥ 0.15 m above its settled height at trial end |

TiPToP handles these natively: Gemini grounds "pick up X" to a `holding(x)` atom and cuTAMP
plans against a `Holding` goal (no place), ending with retract + go-home while still gripping —
a successful grasp carries the object well above the lift threshold, a dropped one falls back
during the 30-step post-plan hold.

## Run

Servers must be up first (M2T2 on :8123, tiptop-server on :8765 — see [`../README.md`](../README.md)).

```bash
PHASE=smoke ./run_grasp_tasks.sh   # 1 trial per task, with video — eyeball first
PHASE=batch ./run_grasp_tasks.sh   # fill to $TRIALS (default 5) in --fast, resumes from CSV
./run_grasp_tasks.sh               # both phases back to back
```

Results land in `runs/grasp_v1/results_<task>.csv` (schema `task,trial,success,plan_failed,detail`),
smoke videos in `runs/grasp_v1/videos_<task>/`.

## Scene 6 — scene 1 + banana (referring-expression tasks)

`scenes/make_scene6.py` authors `scenes/scene6_0.usd` (= stock scene1 byte-identical
+ the YCB `_11_banana` from scene3, all asset paths absolutized) and symlinks it
into `../droid-sim-evals/assets/`, so every stock tool takes it via `--scene 6`.
Layout: banana on the opposite side of the bowl from the cube (settled: cube
(0.369,0.190), bowl (0.502,0.114), banana (0.547,-0.243) at long-axis 100°;
|banana-bowl| 0.360 ≥ |bowl-cube| 0.153). Placement rationale (home-wrist-FOV
visibility, SEGMENT-to-center spawn clearance) is in the script docstring; the
banana spawns at its measured resting attitude, so the scene is static from
step 0 (traced 100-step settle: zero drift; the stock bowl/cube keep their
stock ~4 mm / 2 cm vertical drops).

`scenes/capture_scene6.py` (one Isaac boot) saves to `scenes/captures/scene6_0/`:
layout PNGs (`ext/ext2/wrist.png`), `wrist_obs.h5` (smoke_test.h5-format input
for `gwm_tiptop.propose_from_h5`), `external_obs.h5` (save_h5_obs-format input
for `gwm_tiptop.score_client`), and `objects.json` (settled poses).

```bash
/root/code/gwm/gwm-wiser/.venv/bin/python scenes/make_scene6.py       # author USD (usd-core env)
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y OMNI_KIT_ALLOW_ROOT=1 \
  ../droid-sim-evals/.venv/bin/python -u scenes/capture_scene6.py --scene 6 --variant 0
```

Success rules extend unchanged: `{"objects":["banana"],"lift":0.15}` resolves
`_11_banana` by substring, like `cube`/`bowl` do.
