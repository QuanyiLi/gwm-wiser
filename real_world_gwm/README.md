# Real-World GWM

This folder is the documentation and implementation home for adapting the Grounded World Model (GWM) to broader robot-video corpora for eventual real-world use. The phase-one implementation (VRS adapter, audit, variable-length training, canonical export) lives here; no file under `gwm_wiser/` is changed by it.

## Environment setup

Replicates the verified dev-machine environment (RTX 3090, CUDA driver ≥ cu126). All commands run from the repo root; the code is imported via `PYTHONPATH`, not installed.

```bash
conda create -n gwm python=3.11 -y
conda activate gwm

# Order matters: lerobot pins torch<2.8 and resolves torch 2.7.1+cu126.
pip install "lerobot==0.4.3" tensordict
pip install "transformers==4.57.6"   # pinned: preprocessing/token counts are version-dependent
pip install accelerate pillow "huggingface_hub[cli]" pytest

# FFmpeg shared libraries for torchcodec (LeRobot video decoding)
conda install -c conda-forge -y "ffmpeg>=6,<8"

export PYTHONPATH=/path/to/gwm-wiser:$PYTHONPATH
```

Resolved versions verified on this setup: `python 3.11.15`, `torch 2.7.1+cu126`, `torchvision 0.22.1`, `transformers 4.57.6`, `lerobot 0.4.3`, `wandb 0.24.2` (pulled in by lerobot). `mani_skill` is not needed for anything under `real_world_gwm/` (only for closed-loop `gwm_eval.py`).

Assets and data:

```bash
# Frozen embedder (~16 GB, cached under ~/.cache/huggingface)
hf download Qwen/Qwen3-VL-Embedding-8B

# WISER dataset (open-loop dev evaluation + the wiser repro adapter)
hf download Shady0057/WISER --repo-type dataset \
    --include "merged_train/**" "merged_test/**" --local-dir wiser_dataset

# VRS (RobotSeg): OneDrive/Baidu release only — see the RobotSeg README
# (https://github.com/showlab/RobotSeg). Extract to e.g. /data/vrs so that
# /data/vrs/test/{image,mask_gt} and /data/vrs/train/{image,mask_gt_dinov3}
# exist. test.zip is 0.74 GB (105 videos), train.zip is 21.8 GB.
```

Verify the setup with `pytest real_world_gwm/tests/` (see Tests below).

## Phase-one implementation

Modules (all under `real_world_gwm/`, importable from the repo root):

- `adapters/vrs/dataset.py` — VRS (RobotSeg) source adapter: clip discovery over the released `image/` + `mask_gt`/`mask_gt_dinov3` trees (whole-robot category `002`, provenance recorded), ordinal six-frame windows (`frame_step`/`window_stride`, incomplete windows rejected), robot-only derivation, RAT condition/target assembly.
- `adapters/vrs/visualize.py` — human-inspection contact sheets rendered from the exact training samples.
- `augment.py` — flip applied to all streams; probability-gated color jitter on full RGB only (robot-only is never altered).
- `qwen_rat.py` — RAT tensors → preprocessed Qwen inputs through the unchanged `gwm_wiser` preprocessing, with the per-source pixel budget (ADR-0014) injected via the existing `min_pixels`/`max_pixels` hooks. Default budget lands sources near the WISER token scale (all released VRS grids stay ≤ 2,048 tokens).
- `gwm_model.py` — `VariableLenGWM` (dynamic `(feature_level, flattened_index)` positions for `4 × N` tokens; parameter names/shapes identical to the canonical model) plus canonical checkpoint export and the planner-identical strict loader.
- `audit.py` — machine-readable corpus audit: `python -m real_world_gwm.audit --roots <vrs_tree>... --out audit_manifest.json`. Standalone runs add per-clip motion statistics for the ordinal `frame_step` choice; `train.py` runs the same audit automatically at startup when `--manifest` is not given (single-command slurm launch), embeds the manifest hash in every checkpoint, and fail-fasts on token-ceiling violations.
- `train.py` — training entry mirroring `gwm_wiser/scripts/gwm_train.py` (frozen Qwen online embedding via the shared `gwm_data.compute_embeddings_sequentially`, MSE + cosine logging, Muon+AuxAdam, bf16, optional DDP via torchrun), step-granular checkpoint/resume, canonical export every `--save_every` steps, optional in-training WISER-dev open-loop metrics via `--wiser_dev_dataset_root`. See its docstring for a local smoke command.
- `tests/evaluate_open_loop.py` — standalone WISER-dev open-loop MSE/cosine for an already-saved canonical checkpoint (the same metrics run online during training via `train.py --wiser_dev_dataset_root`).

The implementation shares the WISER pipeline's helpers (`tensor_images_to_pil`, `compute_embeddings_sequentially`, `PaddedLeRobotDataset`, `MuonWithAuxAdam`, canonical `GroundedWorldModel`) instead of duplicating them, so it requires the `[gwm+wiser]` extras (lerobot 0.4.3) plus FFmpeg for LeRobot video decoding.

Cluster launch: `slurm/submit_gwm_vrs.run` mirrors `gwm_wiser/scripts/slurm/submit_gwm.run` (sbatch → srun torchrun, c10d rendezvous); paths are overridable via environment variables and the audit runs automatically inside the job. `slurm/submit_gwm_wiser_repro.run` is the pipeline-debug comparison: `train.py --dataset_adapter wiser` on WISER `merged_train` with parameters aligned to the reference `submit_gwm.run` recipe, so its curves can be compared against `gwm_train.py`'s (checkpoints from that path are WISER-contaminated and are never phase-one candidates).

Tests: `pytest real_world_gwm/tests/` (unit tests need no GPU and no Qwen weights; `tests/test_preflight_e2e.py` is the local acceptance gate and needs the GPU plus a real VRS tree, default `/root/data/vrs/test`, overridable via `VRS_TEST_ROOT`).

## Documentation map

- [CONTEXT.md](CONTEXT.md) defines the canonical domain language.
- [Phase-one plan](docs/phase-1-plan.md) records the current, adjustable experiment and implementation plan.
- [Retired VRS prototype review](docs/prior-dev-vrs-prototype.md) preserves the useful findings and rejected choices from the former `origin/dev` branch.
- [ADR index](docs/adr/README.md) links the accepted architectural decisions and their rationale.
- [References](docs/references.md) lists the primary papers, repositories, and relevant local code.

## Documentation policy

ADRs record decisions that are costly to reverse, surprising without context, and based on a real trade-off. Reversible defaults and operational details belong in the phase-one plan. When a decision changes, add or supersede an ADR instead of silently rewriting its history.

No dataset or model download is performed by this folder. No existing file under `gwm_wiser/` is changed by the planned adaptation.
