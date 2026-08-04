# Phase-One Plan: Corpus-selected GWM on WISER-dev

## Objective

Adapt the existing GWM training pipeline to large-scale out-of-domain robot video while preserving its RAT representation, frozen Qwen latent, MSE objective, checkpoint format, and WISER closed-loop planner. The primary experiment asks whether real, simulated, or mixed-domain pretraining can produce a transferable semantic forward model that scores logged Franka WISER proposals well enough to reach at least 70% end-of-episode success without any WISER training samples.

This is an engineering development milestone, not a held-out benchmark estimate or final real-world claim.

## Scope

Phase one includes:

- A dataset signal survey and explicit training-corpus selection.
- A source-neutral robot-video training contract and source adapters for the selected corpus.
- Candidate-source discovery, validation, audit, visualization, and six-frame window construction.
- Source-side robot-only RGB derivation by masking or calibrated state rendering.
- Online Qwen latent extraction.
- Variable-length GWM training with canonical checkpoint export.
- Optional WISER-dev open-loop MSE/cosine feedback.
- Compatibility with the unchanged closed-loop `gwm_eval.py`.

Phase one excludes:

- Automatic dataset or model downloading by the training entry point.
- Training or evaluating the established WISER-trained checkpoint.
- Trajectory proposal research.
- Physical feasibility, control, safety, or hardware execution.
- Selection of the eventual hardware checkpoint.

## Dataset signal survey

VRS is the ready-to-use candidate, not a locked dependency. Before implementing the final training adapter or launching full training, conduct a separate survey that may replace or supplement it with higher-quality supervision. The survey itself is deferred and is not performed in this documentation session.

Initial discovery leads include:

- Robot datasets indexed by `datasets.bot`.
- Larger or more complete Franka corpora with synchronized RGB and joint configurations.
- MolmoBot and other simulation corpora with recoverable robot geometry and camera state.
- The original upstream sources behind any catalog entry; catalog metadata alone is not sufficient evidence.

Real and simulated video are both eligible, separately or in a documented mixture, provided the pretraining corpus is outside the WISER data domain. Compare candidates using a written matrix containing at least:

- Usable consecutive clip and six-frame-window counts after validation.
- Main-camera RGB quality, resolution, viewpoint, scene, object, task, and motion diversity.
- Franka coverage and diversity; other embodiments are reported descriptively but are not a primary phase-one objective.
- Availability and synchronization of qpos or reconstructable absolute joint-position commands.
- Availability and accuracy of URDF or meshes, camera intrinsics/extrinsics, robot base pose, and timestamps.
- Availability and provenance of whole-robot masks when rendering is not possible.
- Visual alignment quality of derived robot-only RGB under human inspection.
- Robot-only derivation provenance, prioritizing state rendering whenever a valid renderable state bundle exists.
- License, research-use, redistribution, access, storage, and preprocessing constraints.
- Any overlap with WISER; all WISER samples and WISER-generated rollouts are ineligible regardless of split or score.

Record the survey later in `docs/dataset-survey.md`. Corpus identity is selected by usable RAT training signal rather than by dataset popularity or nominal frame count. If multiple sources are chosen, their mixture and sampling policy require a separate decision before training.

### WISER contamination boundary

The selected corpus and gradient path must exclude:

- WISER RGB, robot masks, robot-only RGB, Qwen latents, and logged trajectories.
- Rollouts generated from WISER environments, WISER-specific scenes or object assets, or WISER task configurations.
- Any derivative whose target visual outcomes originate from those samples or rollouts.

The implementation may reuse the existing GWM and Qwen code, RAT construction, generic Panda URDF or meshes, and robot-only renderer machinery. Those shared interface assets do not supply WISER outcome supervision. WISER-dev is permitted only through the development-evaluation path described below and never participates in gradient computation.

## Corpus contract

The training core receives ordered clips from exactly one selected main camera whose frames contain aligned full-scene RGB and robot-only RGB. It must not inspect VRS paths, numeric mask categories, original split names, actions, proprioception, camera calibration, robot state, or auxiliary camera streams. Each source adapter is responsible for producing this normalized representation.

Phase one does not support multi-view input, camera fusion, or treating each auxiliary camera as an additional sample. A source adapter selects one main-camera key; every other camera stream is ignored by this pipeline.

An adapter may use either of two derivation paths without changing the training core:

- Apply an observed or predicted whole-robot mask to the aligned full RGB.
- Render synchronized robot configurations using the correct URDF or meshes, camera intrinsics and extrinsics, robot base pose, and visual materials in a robot-only scene against black.

Generic numeric actions are not sufficient for the rendering path. Direct rendering requires per-frame joint configurations, or absolute joint-position commands whose semantics are known to reconstruct those configurations; delta-pose, velocity, or torque commands require additional state reconstruction outside GWM.

The derivation paths have a strict priority. If the source exposes a valid renderable state bundle, use state-rendered robot-only RGB even when masks are present. Do not intersect that render with an observed or predicted mask, scene depth, or scene geometry: future occlusion and contact are prediction targets, not RAT condition signals. Masks are the primary derivation only for sources without usable qpos/rendering metadata; when both exist, masks are retained for alignment QA rather than substituted into training. A run fixes and records one provenance per source instead of changing it between frames.

### Temporal sampling

Every source adapter declares whether it has a reliable monotonic clock:

- A timestamped source selects the nearest valid frames for six configurable elapsed-time offsets. The phase-one defaults reproduce the existing WISER schedule: `[0.00, 0.55, 1.15, 1.75, 2.35, 2.95]` seconds.
- A source without reliable timestamps uses configurable ordinal `frame_step` and `window_stride`. This fallback is never labeled with seconds.

Timestamp matching must not duplicate frames or repeat a tail to manufacture a complete window. A window that cannot satisfy the configured offsets and matching tolerance is invalid and is reported by the audit.

If VRS is selected or included, its adapter uses every clip from the official train and test trees and all available embodiments. For each clip:

1. Sort consecutive RGB frames by their released ordinal index.
2. Load the whole-robot mask (`002`) for every selected frame, accepting both manual and propagated masks.
3. Generate every complete window with indices:

   ```text
   [i + k * frame_step for k in range(6)]
   ```

4. Default `frame_step=1` and `window_stride=1`; expose both as configuration.
5. Reject incomplete windows instead of repeating the last frame.
6. Sample valid windows uniformly. Do not balance by embodiment, source, clip, or window count.

For RGB frames `rgb[0:6]` and masks `mask[0:6]`:

```text
robot_only[t] = rgb[t] where mask[t] is robot, black elsewhere
condition = [rgb[0], robot_only[1], robot_only[2], ..., robot_only[5]]
target    = [rgb[0], rgb[1], rgb[2], ..., rgb[5]]
```

The mask removes only the background; it must retain every visible robot part.

### Human inspection

Every supported source adapter must include its own visualization script. Before that source is admitted to training, a human must be able to inspect sampled clips showing at least:

- Frame order and source identifiers.
- The selected main-camera key and a statement that auxiliary views are excluded.
- Source timestamps when available, requested temporal offsets, selected frame indices, and matching errors.
- Full RGB with the whole-robot mask overlaid when the adapter uses masks, or with the rendered robot composited over the observation when it uses robot state.
- A rendered-versus-mask comparison when both are available, clearly marking the state-rendered image as the training input.
- Examples where scene objects occlude the observed robot, demonstrating that the geometry-only render remains unclipped.
- The resulting robot-only RGB on black.
- Full-RGB frames before and after photometric augmentation, alongside the unchanged robot-only RGB.
- The six selected frames and their ordinal indices.
- The final RAT condition and full-RGB target side by side.

The visualization must use the same adapter, window selection, masking, and configured transforms as training so it cannot silently validate a separate preprocessing path. Human inspection complements, rather than replaces, the machine-readable audit.

## Preprocessing audit

Audit is a mandatory gate before training. It must write a machine-readable manifest containing at least:

- Source name, source split, video ID, inferred embodiment, and frame count.
- Selected main-camera key, calibration provenance, and ignored auxiliary-camera keys.
- Original frame dimensions and detected dimension inconsistencies inside a clip.
- Timestamp/FPS provenance, monotonicity failures, configured temporal schedule, and per-frame matching error when available.
- Missing, unreadable, or shape-mismatched full-RGB/robot-only pairs and any source assets used to derive them.
- Robot-only provenance: observed mask, predicted mask, or state renderer, including source-specific calibration identifiers where applicable.
- Valid-window count under the configured frame step and window stride.
- Qwen video grid and four-level concatenated token count.
- Token-count histogram and the set of distinct batch shapes.
- Every exclusion and its reason.
- A stable manifest hash recorded by subsequent checkpoints.

The first token ceiling is 2,048 after concatenating the three DeepStack levels and final visual level. Exceeding it is a fail-fast error containing the video ID, input dimensions, Qwen grid, and resulting token count; the first implementation does not resize or skip the sample automatically.

## Configuration policy

No source adapter may hide experiment choices in code constants. At minimum, source roots and adapter choice, main-camera key, six temporal offsets, ordinal frame step, window-start stride, timestamp tolerance, token ceiling, augmentation probabilities, sampling controls, batch size, and training duration are externally configurable. Every run serializes its fully resolved configuration and the audit-manifest hash into its checkpoints and logs.

The six-frame cardinality remains fixed for the phase-one WISER-compatible model. Future real-robot experiments may tune the exposed temporal schedule and train a new checkpoint; changing the number of RAT frames requires a separate interface decision.

## Augmentation

Apply the same spatial transformation to full RGB, robot-only RGB, and masks when present. Full-RGB color jitter is enabled by default with probability `0.5`, using the existing ranges (`brightness=0.4`, `contrast=0.4`, `saturation=0.4`, `hue=0.1`) and one sampled transformation shared by all six frames. Its probability and ranges are configurable.

Photometric augmentation must never alter or recompute robot-only RGB. Robot-only color is an invariant of the RAT interface because future actions may be produced by a URDF renderer; there is no configuration that enables robot-only color jitter. Horizontal flip remains configurable with the existing default probability and applies consistently to full RGB, robot-only RGB, and masks when present.

## Model and training

- Reuse the frozen `Qwen/Qwen3-VL-Embedding-8B` preprocessing and embedding path.
- Concatenate its three DeepStack visual levels and final visual level exactly as the existing pipeline does.
- Preserve the existing GWM projections, transformer layers, dimensions, optimizer, scheduler, precision, and DDP behavior where possible.
- Generate the existing two-coordinate positions dynamically as `(feature_level, flattened_visual_index)` for sequence length `4 * visual_token_count`.
- Optimize only token-level MSE. Log token-level cosine similarity without adding it to the loss.
- Do not provide task text or captions to the GWM forward pass. The current full RGB remains part of RAT; downstream, the unchanged scorer separately builds a task-query embedding from language and visual context to rank predicted trajectory embeddings.
- Compute Qwen embeddings online; do not introduce an embedding cache.
- Save a canonical checkpoint every epoch and support resuming model, optimizer, scheduler, epoch, and random state.
- Begin with the existing three-epoch cluster recipe, while keeping epochs configurable for longer selected-corpus training.

### Batch-shape policy

The audit determines the first batching path:

- If all preprocessed samples have the same grid, reuse ordinary batching and the configured batch size.
- If grids differ, use batch size one. Setting a larger batch size must fail with a clear explanation instead of silently padding or resizing.

Padding and length bucketing are explicitly deferred until real audit results show that their complexity is necessary.

## Checkpoint compatibility

The training wrapper may accept variable sequence lengths, but learned module names and parameter shapes must remain identical to the original `GroundedWorldModel`. On save:

1. Instantiate the canonical fixed-1,620-token model with its original position buffer.
2. Copy the trained learned parameters into it.
3. Save the canonical state dictionary and existing `TransformerConfig` fields.
4. Verify strict loading with the same logic used by `GWMBasedPlanner`.

At exactly 1,620 tokens, the training wrapper and canonical model must produce numerically matching outputs for identical parameters and inputs.

## Development evaluation

Training data and optional development data use separate configuration:

```text
dataset_adapter:           selected_source
dataset_root:              /path/to/selected_corpus
wiser_dev_dataset_root:    /path/to/wiser_dataset  # optional
```

When configured, WISER-dev evaluation reads only `wiser_dev_dataset_root/merged_test`, computes the existing open-loop MSE and cosine metrics, and never backpropagates them. These signals may select a phase-one development checkpoint, so they must not be described as held-out evaluation.

Closed-loop evaluation is performed by the existing `gwm_wiser/scripts/gwm_eval.py` without modifying its planner, logged skill library, proposal horizon, replanning interval, or task environments. Report both WISER train and test task results; the primary phase-one threshold is:

```text
WISER-dev test success_at_end_mean >= 0.70
```

Report `success_once_mean`, grasp, reach, train/test gap, open-loop MSE, and cosine as secondary diagnostics. The established WISER-trained model is outside this implementation's execution scope.

This result measures cross-domain semantic forward-model transfer within the Franka/Panda embodiment. It is not evidence of cross-embodiment generalization or physical-dynamics transfer and need not become a paper claim. An optional UR5 simulation diagnostic is deferred because the unchanged evaluator currently supports only Panda and XArm6; adding UR5 would require a separately scoped evaluation integration and would not alter the primary milestone.

## Planned module interface

All new implementation will remain below `real_world_gwm/`. Its intended shared interface has three entry points:

- `audit.py` inspects the explicitly selected source adapter and emits the manifest.
- `train.py` trains, resumes, optionally evaluates WISER-dev, and saves canonical checkpoints.
- `evaluate_open_loop.py` evaluates a saved checkpoint against an explicitly supplied development dataset.

Each dataset integration lives behind a source adapter and includes a corresponding human-inspection visualization script, such as `adapters/<source>/visualize.py`. RobotSeg directory conventions, mask selection, augmentation, dynamic positions, and canonical checkpoint conversion remain internal implementation details.

## Debugging and verification

The planned implementation must support limited videos/windows, one-batch overfitting, and a one-step dry run. Dataset auditing and unit tests should run without loading the full Qwen model; full integration requires a GPU and locally available Qwen and selected-corpus assets.

Implementation is complete when:

- Window indexing, masking, augmentation, manifest generation, and failure modes have unit coverage, including a test that full-RGB jitter leaves robot-only RGB byte-identical.
- Every selected source's visualization script renders the exact normalized clips and RAT samples consumed by training.
- A small-model test covers dynamic sequence lengths and MSE.
- The 1,620-token wrapper parity test passes.
- Canonical checkpoint strict-loading passes.
- A real-data integration smoke test completes forward, backward, save, resume, and load.
- A tiny overfit run produces a clear loss decrease.

Full selected-corpus training and the 70% closed-loop milestone are subsequent experiment runs, not code-level completion criteria.

## Deferred phase-two decisions

Franka-class real-hardware work begins only after phase one. It must separately decide the trajectory proposal method, source of camera calibration and URDF state, rendering interface, real validation corpus, hardware-checkpoint selection, and operational safety procedures. None of those decisions may be inferred from WISER-dev performance.
