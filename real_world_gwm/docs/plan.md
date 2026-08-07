# GWM Retraining Plan (Plan of Record)

*Supersedes the deleted phase-one and interim phase-two plans (both 2026-08-06; the ADR chain retains their decision history). Companion to the [TiPToP integration plan](tiptop-gwm-integration-plan.md), the system-level plan whose M2 milestone this document owns.*

**This is Stage 1 of the whole effort and starts first** (integration plan, revised execution order): renderer build → pre-flight gates → training-set construction → pre-launch gates → training. Stage 2 (droid-sim pipeline, M0+M1) runs in parallel and consumes the renderer built here; Stage 3 (MolmoSpaces + iteration back into this model) and Stage 4 (hardware) follow.

*Implementation status (2026-08-06): the Stage-1 stack is built and E2E-smoke-verified on the dev machine — renderer + MolmoBot pre-flight PASSED, rendered tree + audit + training + held-out eval all green on the test-split subset; kuma launchers ready. MolmoAct2-DROID is **postponed by decision** (D-26): run 1 trains MolmoBot-only; DROID camera recovery is a later work item.*

Everything under `real_world_gwm/` is organized around three pillars:

1. **Data** — MolmoAct2-DROID (real) + MolmoBot-Data Franka subset (sim). Nothing else is on the training path; VRS and WISER survive only in documentation.
2. **Sim benchmark** — droid-sim (IsaacLab, debugging and selection A/B) and the MolmoSpaces leaderboard (scoring target).
3. **Framework** — the TiPToP system, for both simulation and real hardware. GWM is its only semantic component; this repo supplies the trained scorer and stays a read-only dependency of tiptop.

WISER is fully retired (ADR-0018): no open-loop WISER metrics during training, no closed-loop WISER evaluation, and therefore no WISER-contamination boundary to police. The GWM/Qwen/RAT model interface and the training machinery are inherited unchanged; what changed is corpus, robot-appearance derivation, token-grid anchor, and evaluation.

## Objective

Train a GWM that scores externally proposed Franka pick trajectories in the DROID camera/embodiment setting well enough that GWM candidate selection beats confidence-only and random baselines on droid-sim, with the MolmoSpaces Pick subset (match-or-beat TiPToP's published 46.1% reference) as the downstream target.

## Corpus (ADR-0016)

| Source | Kind | Scale | Streams | Calibration | License |
| --- | --- | --- | --- | --- | --- |
| [MolmoAct2-DROID-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-DROID-Dataset) | Real (quality-filtered DROID) | 74,604 episodes / 17.8 M frames (mean ≈ 15.9 s), LeRobot v3.0 | `wrist_left`, `exterior_1_left`, `exterior_2_left` — **320×180 @ 15 fps, AV1 MP4** | **no usable calibration: the per-frame `camera_extrinsics.*` columns exist but are zero-filled across the entire release** (meta/stats.json min = max = mean = 0, verified 2026-08-06 on-disk), no intrinsics, and DROID episode IDs / camera serials were dropped in conversion. Continuous gripper position in `observation.state` [8]. **Streams stay un-admitted until the camera-recovery gate (below) passes.** | Apache-2.0 |
| [MolmoBot-Data](https://huggingface.co/datasets/allenai/MolmoBot-Data) Franka tabletop subset | Sim (**MuJoCo**, scripted motion-planned, ~15.15 Hz — `policy_dt_ms` 66) | ~1.55 M episodes / ~263 M frames | 5 cams @ 624×352 (table below) | per-frame `cam2world_gl` per camera; per-camera vertical FOV in `obs_scene.frozen_config` (**the h5 `intrinsic_cv` belongs to a different internal 480×480 render and must not be used** — verified by re-projection overlay); per-timestep qpos (arm 7 + **2 Robotiq driver joints**); base-local `tcp_pose` (mount recovery); absolute `joint_pos` actions | ODC-BY |

MolmoBot Franka camera rig (`FrankaOmniPurposeCameraSystem`, all 624×352):

| Stream | Mount / randomization | FOV (vertical, per-episode in `frozen_config`) |
| --- | --- | --- |
| `wrist_camera_zed_mini` | wrist (excluded, as all wrist cams) | ~55.5° |
| `droid_shoulder_light_randomization` | DROID-style shoulder mount, light pose noise (±5 cm, ±8°) | 71° |
| `randomized_zed2_analogue_1/2` | freely placed, full 360° azimuth | 64–72° |
| `randomized_gopro_analogue_1` | freely placed | 137–140° |

**No distortion is baked into MolmoBot-Data**: only pinhole `intrinsic_cv` is published, and the GoPro-analogue stream is rendered undistorted (`is_warped=False` — "baked in warping not yet implemented"); fisheye warping exists only as MolmoBot's own training-time augmentation. The earlier "OmniCam fisheye" risk therefore collapses to a per-camera pre-flight formality. Franka episode lengths at 15 Hz: Pick avg 4.8 s (~72 steps), PnP 17.1 s, PnP-NextTo 20.1 s, PnP-Color 17.4 s. Partial download is per task config (`FrankaPickOmniCamConfig/` etc., own parquet index + tar shards; `bulk_download.py --config ...`), and individual packages can be streamed via HTTP Range without bulk download.

Verified embodiment facts (2026-08-06, primary sources): MolmoBot's Franka is an **FR3 with a Robotiq 2F-85** in MuJoCo — identical to the MolmoSpaces evaluation rig ("specifically set up as a DROID system") and the TiPToP hardware rig. The real DROID platform is a **Panda** arm ("Franka Panda 7DoF robot arm", droid-dataset.github.io) with Robotiq 2F-85 and ZED cameras, at 15 Hz.

MolmoBot-Data is pure sim with zero DROID/OXE overlap; training on it puts our entry in the MolmoSpaces "trained on MolmoBot data" class — record this when comparing against TiPToP's not-MolmoBot-trained 46.1%. MolmoBot-Data totals 10.3 TB; the Franka subset is subsampled per run with the policy recorded in the audit manifest (HF parquet indices count packages of ~6–11 episodes, not episodes).

### MolmoAct2-DROID camera recovery (DEFERRED — DROID postponed by decision, 2026-08-06)

**Decision (D-26 settled): MolmoAct2-DROID is postponed.** Run 1 trains on MolmoBot-Data alone — the Franka subset is large enough on its own — and DROID admission is revisited later as its own work item. The recovery paths below are the recorded options for when it is picked up; the recommended one remains the overlay-verified KarlP join.

Implementation-time discovery (2026-08-06): the release publishes **neither usable extrinsics nor intrinsics** — the `camera_extrinsics.*` columns are zero-filled across the whole dataset (global stats min = max = 0), which supersedes the earlier "extrinsics only" understanding. The prior plan (nominal ZED K + refine against known extrinsics) is void: full camera recovery — 6-DoF pose *and* focal — is required per episode-stream before any DROID rendering.

Candidate paths, decision pending:

1. **Verified join against the raw-DROID calibration release** ([KarlP/droid](https://huggingface.co/KarlP/droid), keyed by episode ID + camera serial). The conversion dropped those keys, so the join must be fuzzy (`building`/`collector_id`/`date`/`length`/instruction) — previously rejected as unverifiable, but the per-stream **pre-flight overlay gate now verifies each joined calibration anyway** (URDF render vs observed robot), converting an unverifiable join into a per-episode self-verifying one. Wrong joins fail the overlay and are excluded, not silently admitted. *(Recommended.)*
2. **Full self-calibration**: optimize pose + focal per episode-stream by aligning URDF renders (known qpos) against the observed robot. Strictly harder than the overlay check itself; no external data needed.
3. **Fallback at scale**: raw DROID + KarlP calibration with exact keys (costs: loses the quality filter and LeRobot packaging, TB-scale storage, CC-BY).

Until one path passes the gate, MolmoAct2-DROID episode-streams are enumerated but carry `calibrated=False` and the render pipeline refuses them (`render_actions.py` reports the count). Training runs sim-only in the interim; the run-1 corpus-mix decision (D-2) re-applies once DROID unlocks.

## Robot appearance (ADR-0017)

Robot-only RGB is state-rendered by the shared `FrankaRobotRenderer` (SAPIEN): batched `qpos → robot-only RGB (N,H,W,3)` against black, camera intrinsics/extrinsics/resolution injected per call, per-source URDF. The same renderer serves training-data generation and inference scoring — render homology is a hard requirement. No mask, depth, or scene geometry clips the rendered robot; robot-only RGB is exempt from photometric augmentation; one appearance provenance (renderer version + URDF hash) is recorded per source per run.

Per-source URDF (verified 2026-08-06):

| Context | Arm | Gripper |
| --- | --- | --- |
| MolmoAct2-DROID (real DROID rigs) | Panda | Robotiq 2F-85 |
| MolmoBot-Data / droid-sim / MolmoSpaces / hardware | FR3 | Robotiq 2F-85 |

Gripper linkage reconstruction goes through the 2F-85 mimic-joint model in both cases: from MolmoBot's 2 driver-joint values, and from MolmoAct2-DROID's continuous `observation.state.gripper_position`.

**Render resolution (settled 2026-08-06):** robot-only frames are rendered at the source's native resolution (320×180 / 624×352) and then pass through the **identical** pixel-budget preprocessing as the paired full RGB — the two streams share one resize path, and overlay/audit comparisons stay pixel-corresponding. Rendering directly at post-budget resolution is rejected (divergent interpolation paths, broken overlay checks).

### Pre-flight verification (hard gate)

Per camera stream, on sample episodes, before any large-scale rendering: URDF re-projection must pixel-align with the released RGB. Status:

- **MolmoBot-Data: PASSED (2026-08-06)** on the smoke subset, with two corrections the gate itself surfaced: (a) the released mp4s do **not** follow the h5 `intrinsic_cv` — the working K comes from each camera's `frozen_config` vertical FOV (`f = H/2 / tan(fov/2)`, centered); (b) `robot_base_pose` is the platform frame, not the arm root — the arm mount is recovered per episode by least squares against `tcp_pose` (pure translation ≈ (0, 0, 0.581) m for the Pick config, fit RMS ≈ 0.09 mm, which simultaneously validates the welded FR3+2F-85 URDF kinematics to sub-mm). The mount fit runs inside `render_actions.py` on every episode with a ≤2 mm residual gate — a per-episode kinematics check at rendering time, forever.
- **MolmoAct2-DROID: BLOCKED** by the camera-recovery gate above.
- The SAPIEN camera-pose convention for `cam2world_gl` is locked by a closed-loop test (set pose → `get_extrinsic_matrix()` reproduces `extrinsic_cv` to 0; sphere at 1 m along the CV forward axis renders at the principal point).

## Token operating grid (ADR-0019)

One shared operating grid anchors all training sources and inference scoring, chosen from the tiptop inference cameras. Both are ~16:9 — droid-sim observes at **1280×720** (external cams fx=fy=500, cx=640, cy=360; verified in `droid_environment.py`) and MolmoSpaces evaluates at **624×352** (`FrankaEvalCameraSystem`, deterministic DROID-style shoulder `exo_camera_1`, not OmniCam-randomized) — so the grid is pinned at **`(3,18,30)` = 1,620 tokens**, coinciding with the existing fixed-length interface. The mechanism is the existing per-source pixel budget (ADR-0014 — unchanged); only the anchor moves from "WISER scale" to "inference scale":

- MolmoAct2-DROID 320×180 (16:9) is **upsampled** onto the grid — no information added, but token count and position distribution match inference.
- MolmoBot-Data 624×352 (≈16:9) lands on it natively.
- Inference RGB (1280×720 droid-sim; 624×352 MolmoSpaces) is budgeted **down/onto** the same grid by `gwm-server`.

Architecturally the model is variable-length (`VariableLenGWM`, ADR-0006) — 1,620 is not a hard requirement — but the learned flat-index position distribution makes train/inference grid mismatch a real risk, so one shared grid is policy. The 2,048-token fail-fast ceiling and audit token histogram are unchanged.

**Sequence-length / batching policy (settled 2026-08-06):** the canonical transformer interface is fixed at 1,620; rather than exercising variable-length training, pin **every** sample to exactly `(3,18,30)` by tuning each source's pixel budget until the audit's empirically measured grid (the production preprocessing path, including its per-frame factor-64 rounding) lands there. Off-grid samples are a fail-fast error (`audit.py` enforces it). Payoff: ordinary uniform batching returns (no batch-size-1 fallback), and training runs at the exact point where `VariableLenGWM` is bit-exact with the canonical model, making canonical export trivially safe. **Measured (2026-08-06): 624×352 through the default budget lands exactly on `[[3,18,30]]`** — no tweak or crop needed for MolmoBot; the DROID 320×180 measurement happens when that source unlocks (exact 16:9, expected to land).

**Token scale (settled default; designated iteration knob):** with WISER gone, 1,620 is no longer a compatibility constraint — the grid is ours to choose. Run 1 stays at 1,620 anyway: the real half of the corpus is 320×180 (no information above the current grid — extra tokens would upsample noise), the frozen-Qwen embedding floor and GWM attention cost grow super-linearly with tokens, and inference latency is K× per decision. Raising the grid (e.g., toward MolmoBot's native 624×352 detail) is a recorded Stage-3 iteration knob, triggered only if error analysis implicates small-object detail; it moves the anchor (ADR-0019), requires raising the 2,048 ceiling, retraining, and redeploying `gwm-server` on the same new grid — never a training-side-only change.

## Sources, the rendered tree, and temporal sampling

**Normalized rendered tree (settled at implementation, revises the runtime-adapter shape):** per-source logic lives entirely in the *preparation* scripts — `scripts/setup_data.py` (provisioning) and `scripts/render_actions.py` (offline rendering) via the thin readers `sources/molmoact2_droid.py` / `sources/molmobot.py`. Rendering writes one normalized on-disk contract, `data/rendered/<source>/<clip_id>/{robot_only.mkv, meta.json}` (per-clip FFV1 lossless video, D-27), where `meta.json` pairs the robot-only stream with its source RGB video (path + frame offset + timestamps) and records full render provenance. The training side has exactly **one** dataset (`rendered.py: RenderedWindowDataset`) that never touches source formats; full RGB stays in the source videos and is decoded on the fly (torchcodec). Rendering itself never subsamples time — every frame of every admitted stream is rendered; frame/stride selection happens at training.

Both sources are timestamped, so the ADR-0012 elapsed-time path applies — no ordinal fallback exists on the training path. Time-based sampling absorbs fps differences (DROID 15 Hz exact; MolmoBot 66 ms steps, whose worst 3-s-schedule offset lands 32 ms off — inside the ±33 ms tolerance, watched by the audit's `max_schedule_error_s`).

Held-out membership is a deterministic hash — `sha256(episode_uid) mod 1000 < 20` (2%) — computed on the camera-independent episode uid, so all streams of one episode land on the same side, on any machine, with no split files.

**Multi-camera contract (settled 2026-08-06, revises the phase-one one-main-camera rule):** each admitted exterior stream yields its own Training Clips — both MolmoAct2-DROID exteriors, and MolmoBot's shoulder + both ZED2 analogues (the 137–140° GoPro analogue is held out of run 1 as an optional diversity ablation). Every stream passes its own pre-flight gate; a clip never mixes streams; wrist cameras are always excluded.

Carried over from the retired phase-one contract (still binding):

- The training core never inspects source paths, actions, proprioception, calibration, or auxiliary streams — adapters normalize everything.
- No frame duplication or tail-repetition to complete a window; windows that miss the configured offsets/tolerance are rejected and reported by the audit.
- Six-frame RAT cardinality is fixed; changing it is a separate interface decision.
- Mandatory per-source machine-readable audit (`audit.py`) with manifest hash embedded in checkpoints; new fields: renderer version, URDF hash, camera-stream selection, intrinsics provenance (nominal / self-calibrated / per-frame), subsampling policy, pre-flight reference.
- Mandatory per-source human-inspection visualization from the exact training samples, now showing the rendered robot composited over the observation.
- No hidden experiment constants: offsets, tolerance, stride, pixel budget, augmentation probabilities, sampling controls all externally configurable and serialized into checkpoints.

Temporal configuration (settled 2026-08-06): both sources default to the legacy schedule `[0.00, 0.55, 1.15, 1.75, 2.35, 2.95]` s, reviewed against the audit's per-source motion statistics before launch. Window stride is per-source: 0.5–1.0 s sliding for MolmoAct2-DROID (real data is the scarce resource), ≈ non-overlapping (≥3 s) for MolmoBot. Timestamp matching tolerance is half the frame interval (±33 ms at 15 Hz); windows beyond it are rejected, never padded. The chunk convention (fixed 3 s vs uniform-6-over-trajectory) is deliberately deferred to the Stage-2/3 selection A/B with the fixed-window default — a late switch forces a window-resampling retrain (accepted, recorded risk).

Augmentation (settled revision): **horizontal flip is disabled** — a mirrored robot lies outside the shared renderer's manifold, violating ADR-0017 homology; viewpoint diversity comes from multi-stream admission instead. Photometric jitter stays p=0.5 on full RGB only; robot-only RGB is never altered.

## Stage-1 order of work and pre-launch gates

Order: **renderer → pre-flight → adapters + bulk rendering → audit + visualization → local gates → cluster training.** Nothing launches on the cluster until every gate below passes. This restores and extends the retired phase-one acceptance discipline; development stays two-staged (everything through small-data training validated locally on the RTX 3090; the cluster is for formal runs only).

Renderer gates (new — the renderer is now training-critical):

- Unit tests: batched qpos → image shapes; camera-parameter injection; **2F-85 mimic-joint linkage reconstruction** from MolmoBot's 2 driver joints and from DROID's continuous gripper position (known open/closed configurations render to expected finger geometry).
- Extrinsics-convention tests: MolmoBot publishes both `cam2world_gl` (OpenGL) and `extrinsic_cv` (CV) — the renderer must consume one convention explicitly and a unit test locks it; MolmoAct2-DROID's 6-D `camera_extrinsics.*` encoding (translation + rotation parameterization) is verified against re-projection in pre-flight, not assumed.
- Per-stream pre-flight re-projection sheets (the hard gate of ADR-0017), produced by the same code path as training-data rendering.

Data gates (carried over from phase one, adapted):

- Unit coverage: window indexing over timestamps, tolerance rejection (no tail-repetition), augmentation invariants — **full-RGB jitter leaves robot-only RGB byte-identical**, manifest generation, failure modes.
- Audit on real released data per stream: empirically measured Qwen grid (must equal `(3,18,30)` exactly — see the sequence-length policy), token histogram, window counts, motion statistics under the configured schedule, exclusions with reasons, manifest hash.
- Visualization contact sheets from the **exact training samples** (same adapter, same transforms): frame order + source IDs, requested offsets vs selected timestamps + matching errors, rendered robot composited over the observation, robot-only on black, RAT condition vs full-RGB target side by side, pre/post-augmentation pairs. Human inspection per admitted stream — a synthetic fixture never satisfies admission.
- Download-budget audit for MolmoBot: measured GB-per-usable-window on one shard per candidate config before any bulk pull.

Model/training gates (carried over unchanged):

- 1,620-token parity test: `VariableLenGWM` vs canonical model, numerically matching outputs for identical parameters and inputs.
- Canonical strict-load test through the planner-identical loader.
- Real-data integration smoke: forward, backward, save, resume, load; a tiny overfit run on real windows shows a clear loss decrease (reduced GWM config where 24 GB is insufficient; full size on cluster).
- Sustained throughput of online Qwen embedding + optimizer step measured on the target GPU class before committing cluster budget (the frozen vision tower is the training-loop floor).

## Training

`train.py` was rewritten in place (2026-08-06, decision D-16) rather than resurrecting `gwm_wiser/scripts/gwm_train.py`: frozen Qwen3-VL-Embedding-8B online embedding, token-level MSE with cosine logging, Muon+AuxAdam, bf16, DDP, step-granular checkpoint/resume, canonical export — now consuming the rendered tree through the single `RenderedWindowDataset`, with the audit auto-run at startup and held-out open-loop evaluation replacing every WISER hook. **All legacy VRS/WISER code is deleted** (D-3 executed: `adapters/`, VRS tests, both old slurm launchers, the `--wiser_dev_dataset_root`/`--dataset_adapter` flags, the VRS prototype review doc); `tests/evaluate_open_loop.py` now evaluates any canonical checkpoint against the hash held-out split. Run 1 (settled): **~1–2 M windows at 1:1 real:sim by window count**, scaled on signal — sim-only until DROID camera recovery unlocks. Checkpoint contract (settled): **canonical fixed-1,620 export is retained as the deployment format** — `gwm-server` reuses the planner-identical loader unchanged. Contract reminders: canonical export embeds `config`; the pickled `gwm_wiser.models.transformer.TransformerConfig` import path never moves; `transformers` pinned at `4.57.6`. Reference recipe: 3×4 H100, batch 32/GPU (uniform batching restored by the exact-grid policy), launched via `slurm/submit_gwm_molmo.run`; large-scale rendering via the `slurm/submit_render_actions.run` array (disjoint resumable shards).

**E2E smoke record (RTX 3090, 2026-08-06, test-split data):** audit `[[3,18,30]]` exact; overfit-one-batch 50 steps mse 0.200 → 0.082 monotonic; 200-step run (4+ epochs over 48 train windows, reduced 512/1024/2 model) train mse 0.20 → 0.06 with held-out cos 0.24 → 0.33 improving; canonical export + planner-identical strict load verified at every save; standalone `evaluate_open_loop` reproduces the training-time held-out numbers exactly.

## Evaluation and milestones (ADR-0018)

1. **Routine diagnostic**: held-out MSE/cosine on windows from both corpora, real and sim reported separately. Splits are **episode-level minimum, scene/lab-level for DROID where metadata allows; window-level splits are forbidden** (settled — adjacent windows leak).
2. **Development milestone**: the canonical checkpoint in `gwm-server`, A/B on droid-sim pick tasks against {M2T2-confidence-only, random candidate}. **Exit criterion (settled): ≥100 episodes per arm; GWM beats the best baseline by ≥5 pp with confidence intervals reported.** Selection-quality tuning (chunk convention, candidate count, FPS K') happens here.
3. **Downstream target** (Stage 3, integration plan M3): MolmoSpaces Pick subset, match-or-beat TiPToP (≤2 pp = match). Stage-2/3 performance feeds back into this model — corpus mix, chunk convention, stream admission, and candidate count are the designated iteration knobs before any architecture change is considered.
4. Hardware evaluation rides the TiPToP framework (Stage 4, integration plan M4); hardware-checkpoint selection is deferred.

If the mixed-corpus signal is weak: real-only vs sim-only vs mixed ablation. There is no VRS arm.

## Decision record (frontier closed 2026-08-06)

Every grill-tracked decision is settled; the standing recommendations were adopted in full. The operative sections above carry the details; this table is the ledger.

| # | Decision |
| --- | --- |
| D-1 | `FrankaRobotRenderer` lives under `real_world_gwm/`; tiptop imports read-only |
| D-2 | Run 1: ~1–2 M windows, 1:1 real:sim by window count, scale on signal |
| D-3 | Legacy code (VRS adapter/tests/slurm, WISER hooks, wiser-repro) frozen + de-gated now; deleted when the Molmo adapters land |
| D-4 | Temporal schedule: legacy 3 s table for both sources, audit-reviewed before launch |
| D-5 | Stride: 0.5–1.0 s sliding (real) / ≈ non-overlapping (MolmoBot) |
| D-6 | Timestamp tolerance: half frame interval (±33 ms @ 15 Hz), reject beyond |
| D-7 | MolmoBot streams: shoulder + both ZED2 analogues; GoPro (137–140°) held out of run 1 |
| D-8 | Canonical fixed-1,620 export retained as the deployment checkpoint format |
| D-9 | Held-out splits: episode-level minimum, scene/lab-level for DROID where possible; window-level forbidden |
| D-10 | droid-sim exit criterion: ≥100 episodes/arm, ≥5 pp over the best baseline, CI reported |
| D-11 | Each admitted exterior stream is its own Training Clip (Main Camera glossary term revised) |
| D-12 | Horizontal flip disabled (render-homology); jitter p=0.5 full-RGB only |
| D-13 | All four Franka tabletop task configs admitted; download order may favor byte-efficient configs |
| D-14 | Sequence length pinned at exactly `(3,18,30)` = 1,620; off-grid fail-fast; uniform batching |
| D-15 | Robot appearance: SAPIEN, offline render-once, lossless cache (container per download-budget audit); rendered at native resolution through the shared preprocessing path |

Stage-1 implementation round (settled 2026-08-06, second grill round — recommendations adopted in full):

| # | Decision |
| --- | --- |
| D-16 | Trainer: rewrite `real_world_gwm/train.py` in place; `gwm_wiser/scripts/gwm_train.py` untouched |
| D-17 | VRS purged radically: all code/tests/slurm/prototype-review deleted; ADR chain retained |
| D-18 | Normalized rendered tree is the single training-side data contract; per-source logic lives in setup/render scripts only |
| D-19 | Robot-only storage: PNG per frame at native resolution + `meta.json` with renderer provenance and RGB pairing |
| D-20 | Held-out: deterministic `sha256(episode_uid) mod 1000 < 20` split, camera-independent, no split files |
| D-21 | MolmoBot camera model: K from `frozen_config` per-camera vertical FOV (h5 `intrinsic_cv` unusable); arm mount recovered per episode from `tcp_pose` with a ≤2 mm kinematics gate |
| D-22 | URDF assets: welded in-repo from ManiSkill Panda + FR3Env FR3 + ManiSkill-Robotiq_2F (no public combined URDF exists); mimic linkage expanded in code from driver values |
| D-23 | Test-split fixtures: DROID episodes 0–148 both exteriors (~560 MB) + MolmoBot Pick val shard 0, 12 packages (~46 MB); same code path as full-scale provisioning |
| D-24 | Rendering never subsamples time; frame/stride selection is a training-side parameter |
| D-25 | Smoke recipe on the 3090: reduced 512/1024/2 model, batch 1, overfit gate then multi-epoch run with held-out eval |

| D-26 | **MolmoAct2-DROID postponed** (its calibration columns are zero-filled; recovery is real work): run 1 is MolmoBot-only, DROID admission is a later work item. Recorded recovery paths: overlay-verified KarlP join (recommended) / full self-calibration / raw-DROID fallback. Supersedes the 1:1 real:sim mixture of D-2 for run 1; the mixture question re-opens when DROID unlocks |
| D-27 | **Rendered-tree container: one FFV1 lossless MKV per clip** (schema v2), revising D-19's PNG-per-frame after cluster quota reality: run-1 renders drop from ~60 M files / ~1.5 TB to ~250 k files / ~0.8 TB. Bit-exactness through the torchcodec decode path is verified per clip at write time (a failed verification blocks the clip); YUV-subsampled "lossless" modes measurably break bit-exactness and are banned. v1 PNG trees remain readable |

**Deliberately deferred / iteration knobs** (revisited from Stage-2/3 evidence, never changed training-side-only):

- Chunk convention: fixed 3 s (default) vs uniform-6-over-trajectory — decided at the selection A/B; late switch forces window-resampling retrain.
- Token scale: raise the operating grid beyond 1,620 only if error analysis implicates small-object detail (re-anchor ADR-0019, raise ceiling, retrain, redeploy together).
- Lightweight online renderer (nvdiffrast/pyrender): only if disk or corpus scale demands it; swaps training AND `gwm-server` together.
- GoPro-analogue stream admission: optional diversity ablation.
- Corpus-mix ablation (real-only / sim-only / mixed): only if the mixed signal is weak.

All previously pending facts are resolved (2026-08-06 research reports; see references.md): eval resolutions pin the grid at 1,620; DROID arm is Panda; both corpora run at 15 Hz; MolmoBot camera roster, episode lengths, absolute joint-position actions, per-config partial download, and the absence of baked-in distortion are all verified. Remaining unconfirmed detail: MolmoAct2-DROID per-episode length distribution beyond mean/max (audit will produce it).

## Risks

| Risk | Mitigation |
| --- | --- |
| DROID camera recovery fails or joins too few episodes | overlay gate rejects wrong joins per stream; full self-calibration and raw-DROID + KarlP fallbacks documented; training proceeds sim-only meanwhile |
| 320×180 real data too coarse for semantic outcomes | held-out curves + droid-sim A/B give the evidence; fallback shares the raw-DROID path |
| Training-vs-eval camera FOV gap (fully randomized MolmoBot cams vs deterministic eval mounts) | FOV-bracketed stream admission (D-7); shoulder stream matches the eval mount family directly |
| Rendered robot-only synthetic even for real RGB | same renderer at train and inference (ADR-0017); visual audit |
| Wrong per-source URDF (Panda vs FR3, gripper linkage) | pre-flight re-projection is a hard gate; mimic-joint reconstruction verified visually |
| Chunk-convention late switch | deliberately deferred (decision record); decided before scaled training |
| Checkpoint `config` pickle path breakage | never move the module; single loader |
| Sim-dominant corpus trains a sim-biased model | 1:1 starting mixture (D-2); real/sim held-out reported separately; mix ablation |
