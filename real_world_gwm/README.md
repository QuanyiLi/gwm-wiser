# Real-World GWM

This folder adapts the Grounded World Model (GWM) into the semantic scorer of the TiPToP planning system. Everything here is organized around three pillars:

1. **Data** — [MolmoAct2-DROID](https://huggingface.co/datasets/allenai/MolmoAct2-DROID-Dataset) (real) + the [MolmoBot-Data](https://huggingface.co/datasets/allenai/MolmoBot-Data) Franka subset (sim), with robot appearance state-rendered by the shared Franka renderer (ADR-0016/0017).
2. **Sim benchmark** — droid-sim (selection A/B) and the MolmoSpaces leaderboard (ADR-0018).
3. **Framework** — TiPToP, for both simulation and real hardware; this repo is a read-only dependency of it.

WISER and VRS are fully retired from the workflow and survive only in documentation; the plan of record is [docs/plan.md](docs/plan.md), and the system-level [TiPToP integration plan](docs/tiptop-gwm-integration-plan.md) (milestones M0–M4) lives alongside it. No file under `gwm_wiser/` is changed by this folder.

## Environment setup

Replicates the verified dev-machine environment (RTX 3090, CUDA driver ≥ cu126). All commands run from the repo root; the code is imported via `PYTHONPATH`, not installed.

```bash
conda create -n gwm python=3.11 -y
conda activate gwm

# Order matters: lerobot pins torch<2.8 and resolves torch 2.7.1+cu126.
pip install "lerobot==0.4.3" tensordict
pip install "transformers==4.57.6"   # pinned: preprocessing/token counts are version-dependent
pip install accelerate pillow "huggingface_hub[cli]" pytest
pip install zstandard h5py           # MolmoBot scene-package extraction / h5 states
pip install "sapien==3.0.3"          # shared Franka renderer (offscreen Vulkan)

# FFmpeg shared libraries for torchcodec (video decoding; needs an AV1 decoder,
# libdav1d — the conda-forge builds have it)
conda install -c conda-forge -y "ffmpeg>=6,<8"

export PYTHONPATH=/path/to/gwm-wiser:$PYTHONPATH
```

Resolved versions verified on this setup: `python 3.11.15`, `torch 2.7.1+cu126`, `torchvision 0.22.1`, `transformers 4.57.6`, `lerobot 0.4.3`, `sapien 3.0.3`, `zstandard 0.25.0`, `h5py 3.16.0`.

Data provisioning (everything lands under the gitignored `real_world_gwm/data/`):

```bash
# Frozen embedder (~16 GB, cached under ~/.cache/huggingface)
hf download Qwen/Qwen3-VL-Embedding-8B

# Smoke subset (~600 MB): DROID episodes 0-148 (both exteriors) +
# MolmoBot Pick val shard 0 (12 scene packages)
python -m real_world_gwm.scripts.setup_data --test-split

# Full-scale selectors (cluster): see the module docstring
python -m real_world_gwm.scripts.setup_data --source molmobot \
    --configs FrankaPickOmniCamConfig --split train --shards 0 10
```

Then render the robot-only streams and inspect the data supply:

```bash
# Offline URDF rendering -> data/rendered/ (every frame, no subsampling;
# URDF source repos are cloned into data/assets/ on first run)
python -m real_world_gwm.scripts.render_actions --source molmobot

# Contact sheets from EXACT training samples (RGB / robot-only / overlay)
python -m real_world_gwm.scripts.visualize_dataloader --num 12 --out viz/
```

Verify the setup with `pytest real_world_gwm/tests/` (no GPU needed).

## Implementation

Modules (all under `real_world_gwm/`, importable from the repo root). VRS/WISER code is fully deleted (plan decisions D-3/D-17).

Data pipeline (per-source logic lives here and only here — decision D-18):

- `scripts/setup_data.py` — provisions both corpora into `data/` (`--test-split` = fixed smoke subset; generic selectors for full scale; MolmoBot shards extracted through the HF cache in the authors' `bulk_download.py` layout).
- `sources/molmoact2_droid.py` — LeRobot v3.0 reader (episodes metadata ∩ files on disk, concatenated-video frame offsets, per-frame states; **streams stay `calibrated=False` until the DROID camera-recovery gate passes** — the release's extrinsics columns are zero-filled).
- `sources/molmobot.py` — scene-package reader (h5 JSON states, per-camera FOV from `frozen_config` → mp4 intrinsics, `cam2world_gl`, base pose, TCP pose).
- `renderer/assets.py` — clones the URDF source repos (ManiSkill Panda, FR3Env FR3, ManiSkill-Robotiq_2F) and welds per-rig arm+gripper URDFs (no public combined URDF exists).
- `renderer/franka_renderer.py` — the shared `FrankaRobotRenderer` (SAPIEN offscreen, per-call camera K/pose/resolution, 2F-85 mimic expansion from driver values; decision D-1 — consumed read-only by tiptop's `gwm-server`) plus `fit_arm_mount` (per-episode mount recovery ≤2 mm kinematics gate).
- `scripts/render_actions.py` — offline rendering of EVERY frame into the normalized rendered tree `data/rendered/<source>/<clip_id>/` as one FFV1 lossless video per clip, bit-exact-verified at write time (D-27; resumable; `--shard-index/--num-shards` for slurm arrays).
- `lossless_video.py` — the FFV1 write/read/verify helpers behind D-27.

Training core (source-agnostic — consumes only the rendered tree):

- `windows.py` — timestamped six-frame RAT windows (legacy 3 s schedule, ±33 ms reject-beyond tolerance) and the RAT condition/target pair.
- `rendered.py` — rendered-tree discovery, deterministic episode-level hash held-out split, and the single `RenderedWindowDataset` (source RGB decoded on the fly via torchcodec).
- `augment.py` — probability-gated color jitter on full RGB only; horizontal flip is deliberately gone (render homology, D-12).
- `qwen_rat.py` — RAT tensors → preprocessed Qwen inputs through the unchanged `gwm_wiser` preprocessing, pixel budget via the `min_pixels`/`max_pixels` hooks (ADR-0019).
- `gwm_model.py` — `VariableLenGWM` plus canonical checkpoint export and the planner-identical strict loader.
- `audit.py` — machine-readable audit over the rendered tree; enforces the exact-`(3,18,30)` operating grid and the token ceiling; auto-run by `train.py` when `--manifest` is absent, manifest hash embedded in every checkpoint.
- `train.py` — training entry (frozen Qwen online embedding, MSE + cosine logging, Muon+AuxAdam, bf16, DDP via torchrun), step-granular checkpoint/resume, canonical export, held-out open-loop evaluation.
- `scripts/visualize_dataloader.py` — contact sheets (RGB / robot-only / overlay) from exact training samples.
- `tests/evaluate_open_loop.py` — standalone open-loop MSE/cosine for a saved canonical checkpoint against the hash held-out split.
- `slurm/submit_setup_data.run`, `slurm/submit_render_actions.run`, `slurm/submit_gwm_molmo.run` — kuma launchers for the full pipeline (CPU-only download → rendering array → 3×4 H100 training); every step runs as a slurm job, nothing on the login node.

Tests: `pytest real_world_gwm/tests/` — unit tests need no GPU and no Qwen weights (the synthetic rendered-tree fixture builds a real mp4 via imageio).

## Documentation map

- [CONTEXT.md](CONTEXT.md) defines the canonical domain language.
- [Plan of record](docs/plan.md) — corpus, renderer, operating grid, evaluation, pre-launch gates, and the settled decision ledger (D-1…D-15).
- [ADR index](docs/adr/README.md) links the accepted architectural decisions and their rationale.
- [References](docs/references.md) lists the primary papers, repositories, verified dataset facts, and relevant local code.
- [TiPToP integration plan](docs/tiptop-gwm-integration-plan.md) — the system-level milestones M0–M4.
- Deprecated history: the phase-one/two plans and the VRS prototype (code and review doc) were deleted; decision history lives in the ADR chain.

## Documentation policy

ADRs record decisions that are costly to reverse, surprising without context, and based on a real trade-off. Reversible defaults and operational details belong in the plan of record. When a decision changes, add or supersede an ADR instead of silently rewriting its history.

Dataset downloads happen only through `scripts/setup_data.py` into the gitignored `data/` tree. No existing file under `gwm_wiser/` is changed by this folder.
