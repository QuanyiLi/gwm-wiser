# References

## System of record

- [TiPToP × GWM integration plan](tiptop-gwm-integration-plan.md) — the system-level plan (M0–M4) this retraining serves (this repo owns its M2).
- [Plan of record](plan.md) — corpus, renderer, operating grid, evaluation, open decisions.

## Selected corpus (ADR-0016)

- [MolmoAct2-DROID-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-DROID-Dataset) (paper <https://arxiv.org/abs/2605.02881>) — real training source: quality-filtered DROID, LeRobot, Apache-2.0.
- [MolmoBot-Data](https://huggingface.co/datasets/allenai/MolmoBot-Data) (paper <https://arxiv.org/abs/2603.16861>, code <https://github.com/allenai/MolmoBot>) — simulation training source: Franka tabletop subset.
- [KarlP/droid calibration release](https://huggingface.co/KarlP/droid) — the DROID authors' post-hoc calibration + annotation release, keyed by episode ID (`{recording_folderpath}--{file_path}`): `cam2base_extrinsics.json` (~36k, quality metrics per entry), `cam2base_extrinsic_superset.json` (~24k unique eps / ~48k poses), `cam2cam_extrinsics.json` (~90k), `intrinsics.json` (~72k, per-serial ZED from SVO), the triple language annotations (~75k eps), `keep_ranges_1_0_1.json` (idle filtering). Joins to MolmoAct2-DROID via the verbatim `language_instruction{,_2,_3}` triple (verified 99.6% unique over the 1,454 local test-split episodes; `building`/`collector_id`/`date` are empty release-wide). Extrinsics format: 6D `[xyz, euler-xyz]` cam→base. Known-imperfect for some episodes ([droid#39](https://github.com/droid-dataset/droid/issues/39)) — the per-stream overlay gate is the admission criterion.
- DROID camera hardware: exterior = ZED 2 @1280×720, factory fx ≈ 531.7 (± ~1% per unit; nominal vFOV 68.4°); **extrinsics vary per scene** (tripods repositioned across 1,417 scenes) — per-episode extrinsics are mandatory, nominal intrinsics are tolerable (~1 px @320×180).
- [DROID](https://droid-dataset.github.io/) — the real platform behind MolmoAct2-DROID.

## Evaluation targets (ADR-0018)

- [droid-sim-evals](https://github.com/tiptop-robot/droid-sim-evals) (upstream <https://github.com/arhanjain/sim-evals>) — IsaacLab Franka/DROID debugging environment for the selection A/B.
- [MolmoSpaces](https://github.com/allenai/molmospaces) (paper <https://arxiv.org/abs/2602.11337>, leaderboard <https://molmospaces.allen.ai/leaderboard>, eval guide <https://allenai.github.io/molmospaces/evaluation_guide/>, data format <https://allenai.github.io/molmospaces/data_format/>) — leaderboard target; TiPToP's published 46.1% is the reference baseline.

## Verified dataset and evaluation facts (2026-08-06, primary-source research reports)

MolmoAct2-DROID:

- 74,604 episodes / 17,758,044 frames (mean ≈ 15.9 s), 15 fps, AV1 MP4; streams `wrist_left`, `exterior_1_left`, `exterior_2_left`, all 320×180 (`meta/info.json`).
- **No usable camera calibration (verified on disk 2026-08-06)**: the per-frame `camera_extrinsics.{wrist_left,exterior_1_left,exterior_2_left}` columns exist but are zero-filled across the entire release (`meta/stats.json` global min = max = mean = 0); no intrinsics anywhere in the 43-feature schema; the episode-metadata parquet (311 columns) retains no DROID episode IDs or camera serials. Recovery paths and the open D-26 decision live in plan.md ("MolmoAct2-DROID camera recovery").
- LeRobot **v3.0** repo layout: multi-episode parquet/mp4 files (`data/chunk-000/file-XXX.parquet`, concatenated AV1 videos sliced by `from_timestamp`/`to_timestamp` in `meta/episodes/`); lerobot 0.4.3 loads it directly; a ~560 MB subset (episodes 0–148, both exteriors) is the smoke fixture.
- `observation.state` [8] = 7 joints + continuous gripper position (also split out as `observation.state.gripper_position`); actions in joint/cartesian position and velocity forms; language instructions live in `meta/tasks_annotated.parquet`.
- The real DROID rig: "Franka **Panda** 7DoF robot arm" + Robotiq 2F-85, two ZED 2 exterior + ZED Mini wrist cameras, 15 Hz control.

MolmoBot-Data (Franka tabletop):

- Generated in **MuJoCo** ("data generation and benchmarking are only supported for Mujoco"); the arm is an **FR3 with a Robotiq 2F-85** (`franka_droid` config, `RobotIQGripperGroup`, assets `franka_fr3` + `robotiq_2f85_v4`).
- Camera rig `FrankaOmniPurposeCameraSystem`, all 624×352: `wrist_camera_zed_mini` (52°), `droid_shoulder_light_randomization` (±5 cm/±8° pose noise), `randomized_zed2_analogue_1/2` (64–72°, full azimuth), `randomized_gopro_analogue_1` (137–140°).
- **Only pinhole calibration exists**: per-frame `extrinsic_cv (T,3,4)` / `cam2world_gl (T,4,4)`; the GoPro stream is rendered undistorted (`is_warped=False`) — fisheye is MolmoBot's training-time augmentation, not a data property. **Caution (verified by overlay 2026-08-06): the h5 `intrinsic_cv (T,3,3)` (fx≈369, c=(240,240), a 480×480 internal render) does NOT describe the released 624×352 mp4s** — working K comes from `obs_scene.frozen_config` per-camera vertical FOV: `f = 176/tan(fov/2)`, c=(312,176).
- `robot_base_pose (T,7)` is the platform frame, not the arm root: the FR3 arm mount is a pure translation above it (≈ (0, 0, 0.581) m for the Pick config), recovered per episode by least squares against base-local `tcp_pose` (fit RMS ≈ 0.09 mm — which also validates the welded FR3+2F-85 URDF kinematics; `renderer/franka_renderer.py: fit_arm_mount`).
- States are null-padded JSON per step inside uint8 arrays (`qpos = {"arm": [7], "base": [], "gripper": [2]}`); `policy_dt_ms` = 66 (≈15.15 fps, mp4 `r_frame_rate` 303/20); scene packages are `.tar.zst` members inside plain shard tars, extractable locally or via HTTP Range (parquet index: `shard_id`/`offset`/`size`).
- Per-timestep `obs/agent/qpos` = JSON dict {arm 7, gripper 2 Robotiq driver joints}; `actions/joint_pos` provides absolute joint-position commands (arm 7 + gripper 1).
- Episode lengths at 15 Hz: Pick 4.8 s avg (~72 steps), PnP 17.1 s, PnP-NextTo 20.1 s, PnP-Color 17.4 s.
- Storage: per-task-config directories (e.g. `FrankaPickOmniCamConfig/`) with parquet package indices + tar shards; `bulk_download.py --config` for subsets; HTTP-Range streaming of individual packages is supported. Parquet indices count packages (~6–11 episodes), not episodes.

Evaluation side:

- droid-sim observes at **1280×720**: `external_cam`/`external_cam_2` fx=fy=500.0, cx=640, cy=360; wrist fx≈666.67 (`droid_environment.py`).
- MolmoSpaces evaluates at **624×352** with `FrankaEvalCameraSystem`: deterministic DROID-style shoulder `exo_camera_1` (71°) + wrist; not OmniCam-randomized unless opted in.
- MolmoSpaces/MolmoBot rig is "specifically set up as a DROID system"; training on MolmoBot-Data places entries in the leaderboard's "trained on MolmoBot data" class.

## Model interface

- [Grounded World Model paper](https://arxiv.org/html/2604.11751) — GWM, RAT, and the original planning results.
- [Qwen3-VL-Embedding-8B model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) · [configuration](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/blob/main/config.json) — frozen embedder.

## Relevant local sources

- [Repository overview](../../README.md) — existing GWM workflows.
- [GWM model](../../gwm_wiser/models/gwm.py) — fixed 1,620-token wrapper and canonical learned modules.
- [Qwen embedding wrapper](../../gwm_wiser/models/qwen3_vl_embedding.py) · [video preprocessing](../../gwm_wiser/models/qwen_video_utils.py) — preprocessing, latent extraction, pooling.
- [GWM data pipeline](../../gwm_wiser/utils/gwm_data.py) — six-frame condition/target construction and latent concatenation.
- [GWM training](../../gwm_wiser/scripts/gwm_train.py) — the reference trainer `real_world_gwm/train.py` mirrors.
- [Retrieval planner](../../gwm_wiser/planner/retrieval.py) — the scoring seam `gwm-server` extracts.
- [Robot-only renderer](../../gwm_wiser/utils/robot_renderer.py) — the pattern the shared `FrankaRobotRenderer` follows.

## Established facts affecting the design

- The existing GWM trains on Qwen internal visual tokens, not the final pooled retrieval vector; the wrapper fixes 1,620 tokens while its learned layers are sequence-length agnostic.
- The existing preprocessing maps 16:9 sources under the default pixel budget exactly onto the `(3,18,30)` = 1,620-token grid (verified empirically; unbudgeted 16:9 sources reach 8,880). Per-frame sizing rounds to factor 64 (round-half-even) before the factor-32 video window — budgeted grids can differ from the uninjected path for the same input size.
- Checkpoints are loaded with `weights_only=False`; the pickled `gwm_wiser.models.transformer.TransformerConfig` import path is part of the compatibility contract. Preprocessing (and token counts) are `transformers`-version-dependent — hence the `4.57.6` pin.

## Historical (retired sources)

VRS/RobotSeg ([repo](https://github.com/showlab/RobotSeg), [paper §3](https://arxiv.org/html/2511.22950v2#S3)) was phase one's ready-now corpus candidate: 2,812 videos / 138,707 frames, ten upstream datasets, 80% Franka/DROID, whole-robot mask category `002`, DINOv3 pseudo train masks, no timestamps, OneDrive/Baidu only, research-only licensing (ADR-0009). Retired to documentation by ADR-0016; its grill-established practices (audit-informed step choice, window rejection over tail-padding, measured pixel-budget behavior) carry over as configuration reference for the Molmo adapters. The deprecated phase plans, the VRS adapter code, and the prototype review doc were all deleted in the 2026-08-06 radical purge (plan decision D-17); the ADR chain retains the decision history. Other phase-one survey leads (RoboMIND, BridgeData V2, LIBERO, RoboCasa, AgiBot World, …) were dropped without adapters when the survey was concluded by ADR-0016.
