# GWM×TiPToP Semantic-Free Proposer — droid-sim Milestone (Plan of Record)

*Settled 2026-08-09 after a three-round design grill (all recommendations adopted). Companion to the system-level [TiPToP×GWM integration plan](/root/code/gwm/gwm-wiser/real_data_train/docs/tiptop-gwm-integration-plan.md) (M0–M4) and the [M2 retraining plan](/root/code/gwm/gwm-wiser/real_data_train/docs/plan.md). This document owns the lightened M0 + all of M1; it supersedes the integration plan's D3 and revises D4's mechanics (recorded there).*

## Objective

On droid-sim (IsaacLab), replace every semantic foundation model in TiPToP's loop with GWM:

> anonymous point-cloud clusters → 12–16 executable pick trajectories (one to a few per cluster) → GWM scores each against the verbatim task description → argmax executes, open-loop, one planning call per episode.

End-state inference components: **two neural networks** — M2T2 (class-agnostic whole-scene grasp generation) and the GWM stack (frozen Qwen3-VL-Embedding-8B + GWM transformer, the *only* semantic component). Everything else is classical: depth→point-cloud projection, mask-free RANSAC table plane, DBSCAN clustering, convex hulls, KDTree grasp association, SE(3) FPS, cuTAMP particle optimization, cuRobo motion refinement. Gemini (detection + task translation), SAM2 (segmentation), and FoundationStereo (sim uses GT depth) are all cut.

## Verified system facts (2026-08-09, code-level)

### tiptop (this repo)

- The semantic boundary is three plain data structures — `masks (N,1,H,W)`, `bboxes [{box_2d,label}]`, `grounded_atoms [{predicate,args}]` — produced solely by `detect_and_segment` (`tiptop/perception_wrapper.py:21`; Gemini + SAM2 are lazily imported there and nowhere else on the pipeline).
- Masks have exactly three *non-semantic* consumers, all replaceable by geometry:
  1. table RANSAC scores candidate planes by per-object contact points and **raises without masks** (`tiptop/perception/segmentation.py:115`);
  2. per-object convex-hull collision meshes for cuTAMP movables/attached-object checks (`segment_pointcloud_by_masks`, same file);
  3. grasp→object association via KDTree over per-object point clouds (`process_scene_geometry`, `tiptop/tiptop_run.py:311`).
- M2T2 is already whole-scene and mask-free (~200 grasps from the full downsampled cloud, `tiptop/perception_wrapper.py:104`).
- Plan schema: `serialize_plan` (`tiptop/planning.py:129`) → `{version, q_init, steps:[{type:"trajectory", positions (T,7), velocities, dt} | {type:"gripper", action}]}`.
- Gemini cost is a non-issue for the baseline arm: Robotics-ER 1.6 at $1/M input, $5/M output ≈ $0.005/episode.

### cuTAMP v0.0.6 (external dep, source walkthrough)

- **Zero language anywhere** — the task planner (`cutamp/task_planning/search.py`) is symbolic BFS from `initial_state` to `goal_state` over grounded operators. All semantics enter tiptop-side (Gemini's `grounded_atoms`), which we replace with enumeration.
- **No fork needed.** `run_cutamp` is only an orchestration function; every building block is public and importable: `setup_cutamp` (world build; reusable across goals — `task_plan_generator(initial_state, goal_state, operators)` takes the goal explicitly), `ParticleInitializer`, `ParticleOptimizer`, `get_ranked_satisfying_particles` (**all** satisfying particles, ranked by summed M2T2 confidence, `algorithm.py:151`), `solve_curobo` (`motion_solver.py:38`). The stock single-plan behavior is just the refinement loop's first-success `break` (`algorithm.py:639-656`); our own orchestrator collects successes instead.
- Version pin mechanism: `REQUIRED_CUTAMP_VERSION = "0.0.6"` in `tiptop/utils.py:26`, enforced at startup — satisfied unchanged since we don't modify cuTAMP.
- License: NVIDIA License (non-commercial), same family as M2T2/FoundationStereo checkpoints — fine for research; note for any publication artifacts.
- ⚠️ cuRobo is **not pinned** (`install/install-curobo.sh` tracks `williamshen-nz/curobo` main) — record the commit at install time and pin it ourselves (GI-0).

### droid-sim-evals (fork of arhanjain/sim-evals; investigated remotely 2026-08-09)

- Client sends **only the wrist camera** (1280×720, fx≈667, GT depth) to the tiptop server (`src/sim_evals/inference/tiptop_websocket.py`). A static **`external_cam` already exists and renders every step**: 1280×720, fx=fy=500.0 exactly, DROID-style left-shoulder view at pos (0.05, 0.57, 0.66), plus a mirrored `external_cam_2`; both currently viz-only, RGB-only. Adding external RGB/K/pose to the request is ~20 lines (copy the wrist ObsTerm pattern).
- **All five stock scenes are natively pick-and-place** ("Put the Rubik's cube in the bowl", …, "Put 3 blocks in the bowl"); instructions are free-form CLI strings, so a custom pick-only instruction set over the same scenes is trivial (~50–100 episode definitions across 5 scenes × 5–8 graspable objects).
- **No automatic success detection** (TerminationsCfg has only timeout; success = watch the MP4) and **no batch runner** (`tiptop_eval.py` = one episode per invocation, restarting IsaacLab each time, 1–5 min startup).
- One planning call per episode, open-loop execution of cached waypoints at 15 Hz (cuRobo ~50 Hz subsampled ×3); episode cap 90 s.
- Install: Python 3.11, `uv`, `isaaclab[all,isaacsim]==2.2.0`, cu118 torch, assets ~69 MB.

### GWM scorer seam (gwm-wiser)

- `score_trajectories` (`gwm_wiser/planner/retrieval.py:473`): splits the prompt into grasp/place prompts, hard 0/1 weighting by `is_grasped` — pick-only ⇒ `is_grasped=False` ⇒ grasp path only. Candidates are duck-typed (`traj.images_robot_state`).
- `_get_task_embedding` (retrieval.py:429): task text + fixed retrieval-style instruction + up to two frames (first, current). ⚠️ Known bug to fix during extraction: the embedding cache is keyed on prompt text only, ignoring the images — clears/keys must include the frame identity or be reset per episode.
- `GWMBasedPlanner.get_video_embedding` (gwm.py:121): RAT = `[current external-cam RGB, 5 robot-only renders]` → `encode_trajectory` → GWM forward → prediction chunked ×4 → `embed_video_latent` → `pooling_video_latent` → (4096,) cosine.
- Loader: `load_canonical_like_planner` (`real_data_train/gwm_model.py:84`), strict, requires `checkpoint["config"]`.
- Local weights status: Qwen3-VL-Embedding-8B cached; only reduced smoke checkpoints (512/1024/2) exist locally; the full-scale run-1 checkpoint is imminent from cluster training.

### GI-2 findings (2026-08-09, scene-1 overlay validation)

- **droid-sim's robot is a Panda, not an FR3** — live sim joint names are `panda_joint1..7` and the flattened USD is NVIDIA's Panda + Robotiq. The integration plan's per-source URDF table (Panda only for real DROID; FR3 for droid-sim/MolmoSpaces) is corrected for droid-sim; MolmoSpaces (real FR3 eval rig) is unaffected.
- **Camera + arm validated exactly**: renderer FK vs live sim `body_pos_w` agrees to **0.0 mm on all arm links 0–8**, and the branded upper-arm band template-matches the sim image at (0,0) px. IsaacLab's `quat_w_ros` is CV-axis cam2world; it feeds the renderer directly.
- **Renderer pose-convention gotcha**: `FrankaRobotRenderer.render(cam2world_gl=...)`, despite the parameter name, consumes **CV-axis** cam2world matrices (`gl_to_sapien_pose` maps forward = column z — the exact mapping `cv_pose_to_sapien_pose` reuses, closed-loop-verified against MolmoBot). Do not apply a CV→GL flip.
- **Robotiq mount standoff for droid-sim is 18.2 mm**, not the 4 mm smoke-stage default in gwm-wiser's welded URDF (two independent ground truths: live `body_pos_w` gives 18.1 mm; the sim USD's `base_link` rel `panda_link8` gives 18.2 mm). Fixed in `gwm_tiptop/assets/panda_robotiq_droidsim.urdf` — gwm-wiser's default is untouched (its 4 mm was validated against MolmoBot pixels; the standoff is per-rig). Residual visual difference at the wrist is the sim's DROID wrist-camera mount, absent from our URDF by design (robot-only renders need the arm+gripper silhouette, not rig accessories).

### Local machine

RTX 3090 24 GB, 208 GB free disk, 40 GB RAM, Ubuntu 26.04 (⚠️ IsaacLab officially supports 22.04/24.04), no pixi yet, Vulkan OK headless.

## Target pipeline

```
droid-sim episode (fork):
  obs = { wrist RGB+GT depth+K+pose,  external_cam RGB+K+pose,  task string, q_init }
        │
        ▼  gwm_tiptop server (pixi env, no Gemini / no SAM2)
  depth → world point cloud
  table plane: iterative RANSAC, horizontal-normal filter, height band, max inliers   (G-3)
  DBSCAN clusters above plane → anonymous object_0..N → convex hulls                  (G-2)
  M2T2 whole-scene grasps → KDTree contact association to clusters
        │
        ▼  run_proposals (cuTAMP as unmodified library)                               (G-4)
  one setup_cutamp world; for each cluster i:
      goal = Holding(object_i) → task_plan_generator → [MoveFree, Pick(object_i)]     (G-5)
      ParticleOptimizer → get_ranked_satisfying_particles
      confidence-weighted SE(3) FPS → ceil(16/N) diverse particles                    (G-6)
      solve_curobo each; collect ALL successes (no first-success break)
  → 12–16 executable candidates (serialize_plan format each)
        │
        ▼  gwm-server (gwm-wiser repo, own env, sequential phase on the same GPU)     (G-7, G-8)
  per candidate: uniform 6 qpos frames over the full trajectory                       (G-10)
      → FrankaRobotRenderer robot-only renders @ external_cam K/pose
      → RAT [external RGB, renders 1..5] → Qwen encode → GWM forward → (4096,)
  task embedding from verbatim task description (is_grasped=False)                    (G-11)
  cosine → softmax across candidates → argmax
        │
        ▼
  execute winning plan open-loop (single planning call per episode)                   (G-9)
  auto success: target object lifted ≥ 10 cm sustained ≥ 1 s (sim rigid-body state)   (G-13)
```

## Decision ledger

| # | Decision |
| --- | --- |
| G-1 | **Light reproduction first**: original TiPToP (with Gemini; user-provided `GOOGLE_API_KEY`) on 2–3 variants per scene, websocket mode — for pipeline bring-up and system understanding, not statistics. The statistical baseline is deferred to the 4-arm A/B (original TiPToP joins {GWM, confidence-only, random} at ≥100 episodes/arm), so nothing is run twice |
| G-2 | **Perception is pure geometry — no SAM2, no Gemini in the new system** (supersedes integration-plan D3's automask): table plane → DBSCAN clusters above it → anonymous `object_0..N`. Under-segmentation (touching objects merging) is tolerable for pick-only + GWM scoring. Recorded fallback: SAM2 automask, switchable in ~a day if clustering proves brittle in sim |
| G-3 | **Mask-free table plane rule** (reversible default): iterative RANSAC; keep planes with near-vertical normals within a workspace height band (default z ∈ [0.1, 1.2] m); choose max inliers |
| G-4 | **cuTAMP consumed as an unmodified library** (revises integration-plan D4's mechanics — no fork, no `break_on_satisfying` config change): own orchestrator `run_proposals` in `gwm_tiptop/` imports `setup_cutamp` / `task_plan_generator` / `ParticleInitializer` / `ParticleOptimizer` / `get_ranked_satisfying_particles` / `solve_curobo`; one shared world, per-cluster goals; refinement successes are collected, never broken on. Pin the cuRobo commit ourselves at install |
| G-5 | **Goal enumeration `Holding(object_i)` per cluster is scaffolding, not selection**: every cluster gets candidates, all candidates go to GWM, selection is 100 % GWM's. The symbolic BFS stays (it is fast, non-semantic, and free) |
| G-6 | **Candidate budget 12–16 total, split evenly across clusters** (`ceil(K/N)` each): the proposer must not pre-judge object importance; M2T2 confidence is used only *within* a cluster (ranking + confidence-weighted SE(3) FPS for diversity) |
| G-7 | **Code layout**: new package `gwm_tiptop/` in this repo on a dedicated branch; original `tiptop/` untouched (baseline arm stays runnable). `gwm-server` lives in the gwm-wiser repo (owns the pinned `transformers==4.57.6` env); tiptop side gets a thin HTTP client |
| G-8 | **Single-3090 orchestration is sequential two-phase** for this milestone: propose → persist candidates (offline-H5-style) → free planner → score → replay. Planner stack (~6–10 GB) and gwm-server (~18 GB) never co-resident; closed-loop latency is explicitly out of scope until the M2 A/B |
| G-9 | **Single open-loop planning per episode** (same as stock TiPToP). GWM conditions on the static `external_cam` (1280×720, fx=fy=500, DROID-style shoulder view) — never the wrist camera (GWM's training corpus excludes wrist cams; RAT assumes a fixed camera). Planning geometry still comes from the wrist camera's GT depth |
| G-10 | **Chunk convention default: uniform 6 frames over the full trajectory** (droid-sim picks land within the time-scale augmentation's 1.5–4.4 s coverage, plan.md D-30); fixed-3 s window kept as the A/B alternative; final choice stays deferred per plan.md |
| G-11 | **Prompt: `task_description` verbatim**, `is_grasped=False` (grasp path only); the grasp/place-split machinery is bypassed in the gwm-server extraction; fix the task-embedding cache-key bug (prompt-only key ignores images) while extracting |
| G-12 | **Episode protocol: custom pick-only instruction set** ("pick up the X") over the 5 stock scenes (~50–100 definitions); native put-X-in-Y instructions are used only during G-1 understanding runs. All four A/B arms share the same instruction set |
| G-13 | **Fork droid-sim-evals** with three additions: ① external_cam RGB/K/pose in the server request (~20 lines); ② automatic pick success = target object lifted ≥ 10 cm sustained ≥ 1 s, read from sim rigid-body state; ③ batch runner keeping IsaacLab resident across a scene×variant×instruction list, emitting a success-rate CSV. Estimated 1–2 days |
| G-14 | **Checkpoint strategy**: build and smoke the mechanics with the local reduced (512/1024/2) checkpoint now; swap in the incoming full-scale run-1 canonical checkpoint via `load_canonical_like_planner` (strict) when it lands. Scores are meaningless until then — GI-4 validates mechanics only |
| G-15 | **Hardware gate deferred, direction recorded**: debug the full system to ≈ original-TiPToP parity on droid-sim before the real robot; MolmoSpaces stays in the plan (only public reference point vs 46.1 %), its priority vs hardware to be ordered on A/B evidence. A/B protocol: 4 arms ≥100 episodes each ≈ 1–2 days of 3090 time with the batch runner |
| G-16 | **Monorepo layout (2026-08-10, supersedes "clone under /root/code/gwm/")**: tiptop, M2T2, droid-sim-evals absorbed into `gwm-wiser/droid/` (their `.git` dirs, incl. nested curobo/cutamp, backed up at `/root/code/gwm/upstream-git-backups/`; provenance table in `droid/README.md`). Old paths remain as symlinks so the 47 GB of pixi/uv environments keep resolving — no reinstall. `real_world_gwm` renamed `real_data_train`; gwm-server moved to `droid/server/` (imports `real_data_train.renderer`, serves `gwm_tiptop/score_client.py` over HTTP). Everything versioned and pushed on gwm-wiser branch `hardware` |
| G-17 | **Custom grasp-only eval tasks live in `droid/droid-sim-evals-ours/`** (2026-08-10; first slice of G-12/G-13's pick-only protocol, layered on droid-sim-evals instead of patching it): `grasp_eval.py` reuses `batch_eval_v2.main()` and swaps in a lift-based `GraspTracker` — success = target mesh center ≥ 0.15 m (user-set) above settled height at trial end. TiPToP needs no changes: Gemini grounds "pick up X" → `holding(x)` → cuTAMP `Holding` goal, and every plan ends retract+go-home still gripping. Scene-1 results (5 trials each, ~75 s/trial fast): grasp_cube "pick up the cube" 4/5, grasp_bowl "pick up the bowl" 5/5 |
| G-18 | **GWM additions extracted from the tiptop worktree (2026-08-10) — `tiptop/` is pristine upstream again**: `gwm_tiptop/` and `gwm_integrate_doc/` moved up to `droid/` level (git renames, all cross-references fixed); `our-droid-sim-evals` renamed `droid-sim-evals-ours`. tiptop worktree diffs empty against upstream `d8f5afd` (verified via backup git). `import gwm_tiptop` resolves in the tiptop pixi env through a one-line `gwm_tiptop.pth` (recreate recipe in `droid/README.md`); canonical cwd for its drivers stays `droid/tiptop` (the upstream `tiptop` package itself is cwd-resolved, unchanged). Prep for the gwm_tiptop test/debug phase |
| G-19 | **First selection A/B with the run-1 checkpoint (2026-08-10, scene-1 grasp tasks; all changes deliberately uncommitted)**. Mechanics: `gwm_tiptop/policy_server.py` speaks the tiptop websocket protocol but serves the 16 precomputed GI-3 candidates (batch_eval_v2/grasp_eval unchanged, `--ws-port`; per-request q_init vs plan q_init drift check read 0.0, so the deterministic reset matches the capture exactly). Random arm = uniform draw, GWM arm = argmax of `gwm-server --backend gwm` (ckpt `/root/exp_ret/0810_gwm/checkpoint.pt`, step 34000) over execution-timeline candidates from `score_client`. **RAT preprocessing is training-parity by construction**: frames anchor-resized to 624×352 (D-29), qwen_rat pixel budget pins the (3,18,30)=1620-token grid (runtime-asserted), WISER schedule available for sampling. **Sampling-interval sweep** (the knob the user flagged): schedule×{0.5,1.0} start, ×{1.0,1.5} pre-close, uniform-full × both instructions — argmax correct in 9/10 variants (miss: bowl@scale-0.5; 1.5 s window shows too little motion), argmax margins 0.004→0.030 (cube) and wrong→0.066 (bowl). **uniform-6 over the full trajectory was the only variant solid on both tasks and was executed** — effective time-scale ≈4.7. Interpretation (user-corrected 2026-08-10): D-33 scale ranges are per-source — droid 0.5–1.5 but **sim/molmobot 1.0–3.0** — and our smooth cuRobo trajectories resemble the sim source's dynamics, so uniform sits only ~1.6× beyond the nearest training ceiling, not 3×. G-10's A/B stays open; pre-close@1.0-1.5 is the strongest bowl scorer; schedule×3.0 start-anchored was tested next and adopted (G-20). **Results**: rand_cube 9/20 = 45 % (right object 9/12, wrong object 0/8; n=20 from an accidental-but-valid doubled batch), rand_bowl 5/10 = 50 % (5/5 vs 0/5) → GWM arm **10/10 cube (plan_10) and 10/10 bowl (plan_05)**. Both scored/served artifacts under `proposals/scene1/scores_*|winner_*`, run logs+CSVs under `droid-sim-evals-ours/runs/gwm_ab/` |
| G-20 | **RAT sampling standard = WISER schedule ×3.0, start-anchored (2026-08-10, user-directed follow-up to G-19)**: 6 frames at t = 0/1.65/3.45/5.25/7.05/8.85 s — the sim-source training ceiling (D-33 molmobot [1,3], spans 3.0–8.9 s), covering approach + gripper close + early lift and skipping the semantically empty go-home tail. Scoring: best cube object-separation of every variant tried (mean sep **+0.039** vs uniform's +0.013; top-7 all cube), bowl winner correct (sep +0.019). Executed: **cube 10/10 (plan_11, new plan) and bowl 10/10 (plan_05, same as uniform's pick)** — ties uniform on success rate on this candidate set, wins on margin robustness while staying inside training coverage → default for future scoring, exposed as the **single sampling hyperparameter `rat_scale`** (ScoreRequest field / `score_client --rat-scale`): default 3.0; `None` ("none" on the CLI) = uniform 6 over the full trajectory whatever its length (the G-10 alternative); a window longer than the plan shrinks to fit (mirrors D-33's fit-fallback). Refactor regression: default-flag scoring reproduces the s30 archive bit-exactly (identical ranking, 0.0 score diff). Determinism note: the bowl batch reproduced the uniform-arm batch's per-trial z_rel *exactly* (same plan + same trial index → same physics outcome) |

## Mini-milestones

| # | Content | Exit criteria |
| --- | --- | --- |
| GI-0 ✅ | Environment bring-up: pixi + tiptop + `setup-planners` + M2T2 (+weights) + droid-sim-evals fork + assets; record & pin cuRobo commit — **pinned 2026-08-09: `williamshen-nz/curobo @ b5fad1df2a3ac4d3e33e369918b7d62d0e59ebd1` (2026-03-21); cuTAMP tag v0.0.6** | **Done 2026-08-09.** cuTAMP pick_block GPU smoke passed; `tiptop-run -h` imports; M2T2 healthy; all 7 GI-1 episodes reached the server |
| GI-1 ✅* | Light repro of original TiPToP (G-1), native instructions, websocket mode | **Done 2026-08-09** ([gi1-repro-log.md](gi1-repro-log.md)): 5 scenes, 4/7 success, failures match TiPToP's published taxonomy; latency logged. *User pipeline walkthrough pending* |
| GI-2 ✅ | Renderer↔droid-sim overlay validation (hard gate, from system-plan M0.3): render `q_init` with `external_cam` K/pose over sim RGB | **Passed 2026-08-09.** All 5 scenes visually aligned ([overlays/](overlays/)); FK vs sim `body_pos_w` 0.0 mm on arm links 0–8; Robotiq standoff corrected to 18.2 mm (GI-2 findings above) |
| GI-3 ✅ | `gwm_tiptop/` semantic-free proposer: mask-free perception + `run_proposals` | **Passed 2026-08-09.** 16–20 executable candidates per scene on all 5 scenes (scene1/2/3: 2 objects ea.; scene4: 5; scene5: 3), zero Gemini/SAM2 imports, every cluster covered, cluster viz under [proposals/](proposals/). Perception lessons and cuTAMP landmines recorded below |
| GI-4 ✅ | gwm-server extraction + sequential scoring loop | **Dummy-backend loop closed 2026-08-09**; **`--backend gwm` landed 2026-08-10 with the run-1 checkpoint** (G-19): Qwen3-VL-Embedding-8B + canonical GWM on the 3090 (~19 GB), ~2.5 s/candidate end-to-end, RAT strips dumpable per request for visual audit. Selection A/B vs random executed same day (G-19: 45 %→100 % cube, 50 %→100 % bowl on the fixed scene-1 candidate set) |
| GI-5 | A/B readiness: fork's auto-success + batch runner + pick-only instruction set (G-12/G-13) | 10-episode dry run emits a correct success-rate CSV unattended |

Beyond GI-5 the work hands over to the system plan's M2.5 selection A/B (4 arms, ≥100 episodes/arm, ≥5 pp exit criterion) once the run-1 checkpoint is in.

## GI-3/GI-4 implementation lessons (2026-08-09)

- **cuTAMP v0.0.6 landmine — `get_world_cfg(env, include_movables=True)` mutates `env.movables` in place** (`obstacles = env.movables; obstacles += env.statics`). Statics leak into the movables list; the cost function then requests collision spheres for the table and KeyErrors. `run_cutamp` only survives because it calls this at the last moment before refinement. Our `run_proposals` passes a throwaway env snapshot. Also: `ParticleOptimizer` leaks the `optimization_step` timer on its early-satisfied exit (second call raises "Timer already started") — we stop it defensively after each call.
- **Never use the collision table box as a segmentation boundary**: tiptop's table cuboid top is deliberately sunk 2 cm below the detected surface. `find_table_plane` returns the true `surface_z` separately; clustering cuts at `surface_z + 1.5 cm`. Getting this wrong floods the clusters with tabletop points.
- **Cluster merge rules (occlusion vs adjacency)**: DBSCAN at eps 1.5 cm splits partially occluded objects (a bowl becomes two rim arcs, split by depth discontinuity, XY gap ≈ 4 mm). Merge when XY-hull containment > 15 % (opposite arcs of hollow objects) OR min XY gap < 8 mm (rim slivers). eps 3 cm is NOT a substitute — it fuses adjacent objects (cube 2 cm from bowl).
- **Filter robot-arm clusters BEFORE merging**: the arm hangs directly above objects at the capture pose; the XY-containment rule otherwise absorbs floating arm fragments into the object below (scene 4's sugar box grew to z=0.39 and put q_init in start-state collision, killing every refinement). Resting test: cluster min-z ≤ surface + 4 cm.
- **Single-GPU VRAM discipline**: with the GI-1 tiptop-server (3.4 GB) + M2T2 + gwm-server + an IsaacLab replay resident, `MotionGen` construction intermittently fails deep in cuRobo's STOMP covariance init (`torch.inverse` → "lu_solve pivots" — cusolver under memory pressure). Kill idle servers before proposer batches; the error is spurious, not scene-dependent.
- Observed proposer wall-clock: ~2 min/scene (perception + M2T2 + 2–5 × (optimize ≈ 3–7 s + K × refine)).

## Risks

| Risk | Mitigation |
| --- | --- |
| DBSCAN under-/over-segmentation (touching objects, flat objects near plane) | tolerable for pick-only scoring; tune eps/min-points on sim scenes in GI-3; recorded fallback = SAM2 automask (G-2), swappable ~1 day |
| Ubuntu 26.04 vs IsaacLab 2.2.0 (supports 22.04/24.04) | try native first; fallback = official Isaac Sim container / older-glibc conda env |
| VRAM squeeze in websocket mode (Isaac Sim + M2T2 + planner ≈ 17 GB on one 3090) | headless, trim camera count during repro; fallback = offline H5 mode (natural time-sharing) |
| cuRobo unpinned upstream | pin commit at GI-0; record in this doc |
| GWM scores meaningless until run-1 ckpt | GI-4 is mechanics-only by design (G-14); no selection-quality conclusions before the ckpt swap |
| Non-commercial licenses (cuTAMP NVIDIA License, M2T2/FoundationStereo checkpoints) | research use OK; flag before releasing artifacts |
| Isaac startup dominating batch wall-clock | resident batch runner (G-13 ③) |

## Deferred / iteration knobs

- Chunk convention final choice (uniform-6 vs fixed-3 s) — decided at the M2 selection A/B (G-10).
- Hardware entry ordering vs MolmoSpaces (G-15) — decided on A/B evidence.
- Pick-and-place scope extension (goal pairs `On(object_i, surface_j)`; candidate budget explodes) — after pick-only v1 lands.
- Automask fallback trigger (G-2) — only on clustering failure evidence.
- Co-resident single-GPU serving / latency work — M2 A/B phase or second GPU.
