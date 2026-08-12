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

## Scene 6 rev2 — two colored bins + place eval (2026-08-11)

`make_scene6.py` now authors two variants:
- `scene6_0` (pick): stock rev1 objects + `red_bin` (0.395, -0.055) and
  `green_bin` (0.305, -0.250) — squared-off 0.115 m KLT copies recolored via
  UsdPreviewSurface (layout solved by `scenes/optimize_bin_layout.py` for
  external-camera pixel separation under wrist-visibility / gripper-shadow /
  clearance constraints). Two refer6 instructions reworded for the new
  containers (see `refer6_tasks.sh` header); pre-rev2 results not comparable.
- `scene6_1` (place): same + `held_block`, a 30 mm blue block spawned inside
  the open gripper. `weld_held_block.py` (imported by `place_eval.py` /
  `scenes/capture_place.py`) welds it to the Robotiq base_link with a
  runtime-authored FixedJoint at first settle. Since G-31 the weld is released
  the first time the gripper is COMMANDED open after having closed — judged on
  the commanded joint target, not the measured angle, because PhysX can pin
  the fingers shut while they squeeze a block rigidly welded to their own
  base (v1/v2 measured-angle hooks missed 3/40 releases that way). GWM place
  candidates never reopen, so they end holding the block inside a bin, while
  TiPToP's native plan releases and drops it in; the same judge band covers
  both end states unchanged. 30 mm (not stock 47) because the gripper closed
  on the block must pass the 0.105 m bin mouth: width is 0.0588 + edge.

Pick pipeline (GWM arm), scene6_0 → `proposals/scene6_rev2`:
1. `gwm_tiptop/propose_from_h5.py` — 16 whole-scene candidates, the budget split
   floor+remainder over the perceived clusters (5 here → quotas 4,3,3,3,3).
2. `run_refer6_score.sh` — per instruction, gwm-server scores every candidate;
   the OBJECT is chosen by the **mean** of its candidates' scores and the winner
   is that object's best candidate (`score_client --object-score`, G-28). The
   older global per-candidate argmax (`--object-score max`) scored 6/10 correct
   objects against mean's 9/10: `proposals.se3_fps_indices` samples each
   object's quota for SE(3) *diversity*, so a per-candidate max compares grasp
   families by their extremes at GWM's ~0.01 within-family spread. Same script
   then runs `grasp_gate --apply` (G-27), which re-picks within the chosen
   object only.
3. `run_refer6_gwm.sh` — fixed-plan replay via policy_server, judged by
   `grasp_eval.py`; `analyze_refer6.py` aggregates. `run_refer6_tiptop.sh` is
   the baseline arm (online tiptop planning per trial, no policy server).

Place pipeline (GWM arm, no M2T2 / no cuTAMP — the "grasp" is the weld):
1. `scenes/capture_place.py --scene 6 --variant 1` — welded captures.
2. `gwm_tiptop/place_propose.py` (tiptop pixi env) — PERCEPTION-ONLY since the
   2026-08-11 audit: inputs are the wrist h5 + the robot's own model, nothing
   else (no objects.json, no hardcoded bin list — GT object count/pose is
   judge-side only). The in-hand object is measured from the cloud (points
   near the FK EE, outside the robot's padded collision spheres); every
   perceived cluster becomes a destination (hollow → land on its inner floor,
   solid → on its top face; heights read off the points); the 16-candidate
   whole-scene budget is split floor+remainder over the clusters. Candidates:
   [gripper close, approach above dest, constrained straight descent]. No
   GoToInitial: plans are ~7.5 s < 8.85 s, so the GWM RAT window's
   shrink-to-fit branch finally fires and the last frame is the discriminative
   in-dest pose. v1 (GT-target proposer) is preserved at
   `proposals/scene6_place` for provenance; v2 lives at
   `proposals/scene6_place_v2` with results under `runs/place_v2/`.
3. `run_place_score.sh` — gwm-server selection per instruction (4 tasks,
   `place_tasks.sh`, destination referring expressions split 2-2 across bins).
   Same two-stage `score_client` as the pick arm, but stage 2 is a no-op here:
   place candidates carry a constant `grasp_confidence` of 1.0 (there is no
   M2T2 grasp — the "grasp" is the weld), so the winner falls back to the
   chosen destination's best-scoring candidate.
4. `run_place_gwm.sh` — fixed-plan replay via policy_server on scene6_1,
   judged by `place_eval.py` (stock placement SuccessTracker + per-candidate
   `_landed_in` bookkeeping); `analyze_place.py` aggregates. TRIALS defaults
   to 1 (deterministic replay); raise for formal runs.

Debug bring-up 2026-08-11: hand-picked winners (`plan_00_red_bin` /
`plan_08_green_bin`) replayed clean — both tasks success=True, block 5–6 mm
from bin centre, `_landed_in` correct, 60 s/trial.
