# Phase-One Plan: Corpus-selected GWM on WISER-dev

## Objective

Adapt the existing GWM training pipeline to broader out-of-domain robot video while preserving its RAT representation, frozen Qwen latent, MSE objective, checkpoint format, and WISER closed-loop planner. The primary experiment asks whether real, simulated, or mixed-domain pretraining can produce a transferable semantic forward model that scores logged Franka WISER proposals well enough to reach at least 70% end-of-episode success without any WISER training samples.

This is an engineering development milestone, not a held-out benchmark estimate or final real-world claim.

## Scope

Phase one includes:

- A dataset signal survey and explicit training-corpus selection.
- A source-neutral robot-video training contract and source adapters for the selected corpus.
- Candidate-source discovery, validation, audit, visualization, and six-frame window construction.
- Source-side robot-only RGB derivation from observed or predicted whole-robot masks; calibrated state rendering is deferred to phase two (ADR-0015).
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

The survey validates three signal hypotheses in parallel, one per first-tier lead:

- **VRS (real, ready now)**: the only zero-preparation real-video source and the pipeline's first adapter; known costs are DINOv3 pseudo training masks, roughly 118k step-1 windows, and 80% Franka/DROID weighting.
- **DROID with RobotSeg-predicted masks (real, scale)**: about 100x the VRS Franka scale under CC-BY 4.0; the RobotSeg model was trained on DROID-dominated VRS, so its predictions are near-domain, but predicted-mask quality must pass audit and human inspection before admission.
- **One ground-truth-segmentation simulation source (simulation, perfect masks)**: RoboCasa (different engine) or MolmoBot-Data (qpos-complete; engine proximity must be tagged under the same-engine rule); simulator segmentation yields exact occlusion-clipped whole-robot masks matching the WISER-dev condition.

Second-tier leads are surveyed on paper but receive no adapter until a first-tier line fails or a recorded decision supersedes this one: RoboMIND (107k real trajectories, 52.9k Franka, RGB plus joint states), BridgeData V2, NVIDIA PhysicalAI Robotics Manipulation Augmented, LIBERO, Exylos Bimanual Table Spill Cleanup, AgiBot World, and Dexora (low confidence). State-rich corpora previously dismissed solely for lacking released robot masks enter phase one through predicted whole-robot masks, while their qpos and render bundles are recorded as phase-two readiness. Robot datasets indexed by `datasets.bot` remain a catalog index; the original upstream sources behind any catalog entry must be inspected directly, because catalog metadata alone is not sufficient evidence.

Real and simulated video are both eligible, separately or in a documented mixture, provided the pretraining corpus is outside the WISER data domain. Simulation sources sharing WISER's engine or renderer (ManiSkill/SAPIEN) are eligible under the ADR-0011 boundary — their scenes, assets, and tasks must not be WISER's — but carry a recorded engine/renderer-proximity tag, and any mixture documentation states the same-engine share. Compare candidates using a written matrix containing at least:

- Usable consecutive clip and six-frame-window counts after validation.
- Main-camera RGB quality, resolution, viewpoint, scene, object, task, and motion diversity.
- Franka coverage and diversity; other embodiments are reported descriptively but are not a primary phase-one objective.
- Availability and provenance of whole-robot masks — released, propagated, or predictable by a segmentation model — as the primary phase-one derivation (ADR-0015).
- Visual alignment quality of derived robot-only RGB under human inspection.
- Availability and synchronization of qpos or reconstructable absolute joint-position commands, reported as phase-two readiness.
- Availability and accuracy of URDF or meshes, camera intrinsics/extrinsics, robot base pose, and timestamps, reported as phase-two readiness.
- Simulation-source engine/renderer proximity to WISER, tagged explicitly for same-engine sources.
- License, research-use, redistribution, access, storage, and preprocessing constraints.
- Any overlap with WISER; all WISER samples and WISER-generated rollouts are ineligible regardless of split or score.

Record the survey later in `docs/dataset-survey.md`. Corpus identity is selected by usable RAT training signal rather than by dataset popularity or nominal frame count. If multiple sources are chosen, their mixture and sampling policy require a separate decision before training.

### Evidence from the retired VRS prototype

The former `origin/dev` branch is reviewed in [Review of the Retired VRS Prototype](prior-dev-vrs-prototype.md). It demonstrated only a loader-shaped smoke fixture, not successful GWM training or WISER performance. Its reusable observations are already reflected here: VRS uses ordered frames and whole-robot mask `002`, lacks a reliable corpus-wide clock, has mask-provenance risk, and should remain only one candidate in the broader tiered signal survey.

Do not merge that implementation directly. Its fixed `224x224` resize, synthetic 3-second clip normalization, one-window-per-clip sampling, tail repetition, split handling, mask-category fallback, state/action placeholders, and edits to the existing trainer all conflict with this plan.

### WISER contamination boundary

The selected corpus and gradient path must exclude:

- WISER RGB, robot masks, robot-only RGB, Qwen latents, and logged trajectories.
- Rollouts generated from WISER environments, WISER-specific scenes or object assets, or WISER task configurations.
- Any derivative whose target visual outcomes originate from those samples or rollouts.

The implementation may reuse the existing GWM and Qwen code, RAT construction, generic Panda URDF or meshes, and robot-only renderer machinery. Those shared interface assets do not supply WISER outcome supervision. WISER-dev is permitted only through the development-evaluation path described below and never participates in gradient computation.

## Corpus contract

The training core receives ordered clips from exactly one selected main camera whose frames contain aligned full-scene RGB and robot-only RGB. It must not inspect VRS paths, numeric mask categories, original split names, actions, proprioception, camera calibration, robot state, auxiliary camera streams, or zero-valued compatibility placeholders for any of those fields. Each source adapter is responsible for producing this normalized representation.

Phase one does not support multi-view input, camera fusion, or treating each auxiliary camera as an additional sample. A source adapter selects one main-camera key; every other camera stream is ignored by this pipeline.

An adapter may implement either of two derivation paths without changing the training core:

- Apply an observed or predicted whole-robot mask to the aligned full RGB.
- Render synchronized robot configurations using the correct URDF or meshes, camera intrinsics and extrinsics, robot base pose, and visual materials in a robot-only scene against black.

Generic numeric actions are not sufficient for the rendering path. Direct rendering requires per-frame joint configurations, or absolute joint-position commands whose semantics are known to reconstruct those configurations; delta-pose, velocity, or torque commands require additional state reconstruction outside GWM.

Phase-one training supervision uses the mask path (ADR-0015). The WISER-dev evaluator conditions the GWM on robot appearances cut from observed segmentation — occlusion-clipped by scene objects — so mask-derived robot-only RGB keeps the training condition distribution consistent with the unchanged evaluator. A source without released whole-robot masks may be admitted through predicted masks (for example a RobotSeg-model labeling pass) with provenance recorded. Synchronized qpos, calibration, and render-bundle quality are still surveyed and audited because they determine phase-two readiness, where the state renderer becomes the training and proposal condition and is never intersected with masks, depth, or scene geometry. A run fixes and records one robot-only provenance per source instead of changing it between frames.

### Temporal sampling

Every source adapter declares whether it has a reliable monotonic clock:

- A timestamped source selects the nearest valid frames for six configurable elapsed-time offsets. The phase-one defaults reproduce the existing WISER schedule: `[0.00, 0.55, 1.15, 1.75, 2.35, 2.95]` seconds.
- A source without reliable timestamps uses configurable ordinal `frame_step` and `window_stride`. This fallback is never labeled with seconds.

Ordinal defaults are not chosen blindly. For each ordinal source, the audit reports per-window robot-motion statistics (for example whole-robot-mask displacement and frame-difference magnitude) for candidate `frame_step` values, and source admission records a step whose windows visually match the motion span of WISER-dev candidate skills under human inspection. The software default remains `frame_step=1`, but training on an ordinal source requires an explicitly recorded, audit-informed step choice; the starting hypothesis for VRS is a step of 2-3 pending audit. An explicitly configured multi-step mixture is permitted as a documented sampling policy — the RAT condition encodes each window's span through its future robot appearances, so mixed spans are not ambiguous to the model — and defaults to off.

Timestamp matching must not duplicate frames or repeat a tail to manufacture a complete window. A window that cannot satisfy the configured offsets and matching tolerance is invalid and is reported by the audit.

Never normalize a clockless clip by uniformly stretching its first and last frames over the WISER three-second horizon. An ordinal adapter must not synthesize source timestamps, `raw_fps`, or elapsed-time claims from clip length. The selected six frames continue through the existing list-frame Qwen preprocessing path unless a separately validated source with real timing metadata requires an explicit timing policy.

If VRS is selected or included, its adapter uses every clip from the official train and test trees and all available embodiments. For each clip:

1. Sort consecutive RGB frames by their released ordinal index.
2. Load the whole-robot mask (`002`) for every selected frame, accepting manual first-frame annotations and the released DINOv3 pseudo masks (`mask_gt_dinov3`); record which kind each frame used.
3. Generate every complete window with indices:

   ```text
   [i + k * frame_step for k in range(6)]
   ```

4. Expose `frame_step` and `window_stride` as configuration (software default one each); the trained step follows the audit-informed choice above.
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
- Full RGB with the whole-robot mask overlaid; for a phase-two state-rendered adapter, the rendered robot composited over the observation.
- Mask-provenance examples (manual, propagated, or model-predicted) and frames where scene objects occlude the robot, showing the occlusion-clipped mask behavior that matches the WISER-dev condition.
- A rendered-versus-mask comparison when a render bundle is also available, clearly marking which image is the training input.
- The resulting robot-only RGB on black.
- Full-RGB frames before and after photometric augmentation, alongside the unchanged robot-only RGB.
- The six selected frames and their ordinal indices.
- The final RAT condition and full-RGB target side by side.

The visualization must use the same adapter, window selection, masking, and configured transforms as training so it cannot silently validate a separate preprocessing path. Human inspection complements, rather than replaces, the machine-readable audit.

A synthetic image, overview GIF crop, or heuristically generated mask may test basic file and tensor plumbing, but it does not satisfy source admission. At least one inspection and audit path must exercise the released source records and the exact robot-only derivation used for training.

## Preprocessing audit

Audit is a mandatory gate before training. It must write a machine-readable manifest containing at least:

- Source name, source split, video ID, inferred embodiment, and frame count.
- Selected main-camera key, calibration provenance, and ignored auxiliary-camera keys.
- Original frame dimensions and detected dimension inconsistencies inside a clip.
- Timestamp/FPS provenance, monotonicity failures, configured temporal schedule, and per-frame matching error when available.
- Missing, unreadable, or shape-mismatched full-RGB/robot-only pairs and any source assets used to derive them.
- Robot-only provenance: observed mask, predicted mask, or state renderer, including source-specific calibration identifiers where applicable.
- Valid-window count under the configured frame step and window stride.
- Per-window robot-motion statistics under the configured and candidate frame steps for ordinal sources.
- Qwen video grid and four-level concatenated token count.
- Token-count histogram and the set of distinct batch shapes.
- Every exclusion and its reason.
- A stable manifest hash recorded by subsequent checkpoints.

Every source adapter declares an aspect-preserving pixel budget applied through the existing Qwen preprocessing hooks (per-video `min_pixels`/`max_pixels`), defaulting to the WISER-scale window that lands near 405 tokens per level; the budget is recorded in the manifest and checkpoints (ADR-0014). The first token ceiling is 2,048 after concatenating the three DeepStack levels and final visual level. Exceeding it is a fail-fast error containing the video ID, input dimensions, configured pixel budget, Qwen grid, and resulting token count; the implementation must not silently skip a sample or downscale it outside the declared budget. The ceiling is a reversible experiment control that later experiments may raise. Audit token counts are computed with the exact production preprocessing path, including its per-frame factor-64 rounding, rather than re-derived formulas.

## Configuration policy

No source adapter may hide experiment choices in code constants. At minimum, source roots and adapter choice, main-camera key, six temporal offsets, ordinal frame step, window-start stride, timestamp tolerance, per-source pixel budget, token ceiling, augmentation probabilities, sampling controls, batch size, and training duration are externally configurable. Every run serializes its fully resolved configuration and the audit-manifest hash into its checkpoints and logs.

The six-frame cardinality remains fixed for the phase-one WISER-compatible model. Future real-robot experiments may tune the exposed temporal schedule and train a new checkpoint; changing the number of RAT frames requires a separate interface decision.

## Augmentation

Apply the same spatial transformation to full RGB, robot-only RGB, and masks when present. Full-RGB color jitter is enabled by default with probability `0.5`, using the existing ranges (`brightness=0.4`, `contrast=0.4`, `saturation=0.4`, `hue=0.1`) and one sampled transformation shared by all six frames. Its probability and ranges are configurable.

Note: the existing WISER trainer applies color jitter unconditionally — its `jitter_prob` parameter is unused dead code — so the `0.5` gate is a deliberate behavior change in this implementation, not a reproduction of current behavior.

Photometric augmentation must never alter or recompute robot-only RGB. Robot-only color is an invariant of the RAT interface because future actions may be produced by a URDF renderer; there is no configuration that enables robot-only color jitter. Horizontal flip remains configurable with the existing default probability and applies consistently to full RGB, robot-only RGB, and masks when present.

## Training-run matrix

Phase one runs a staged matrix of at most four headline training runs:

1. VRS-only first. This run doubles as the pipeline's first end-to-end exercise — audit, training, canonical export, and WISER-dev open-loop plus closed-loop evaluation — and produces the first cross-domain transfer signal.
2. One single-source run for each remaining first-tier adapter that passes audit and human inspection.
3. At most one mixture run, decided only after the single-source results exist; the mixture and sampling policy remain a separate recorded decision.

Each run starts from the existing three-epoch-equivalent budget with the WISER-dev open-loop curve logged; only the best-performing line is extended. If every line lands under the 70% threshold, the response is analysis first — open-loop diagnostics and per-config breakdowns — not a silent threshold change.

## Model and training

- Reuse the frozen `Qwen/Qwen3-VL-Embedding-8B` preprocessing and embedding path.
- Concatenate its three DeepStack visual levels and final visual level exactly as the existing pipeline does.
- Preserve the existing GWM projections, transformer layers, dimensions, optimizer, scheduler, precision, and DDP behavior where possible.
- Generate the existing two-coordinate positions dynamically as `(feature_level, flattened_visual_index)` for sequence length `4 * visual_token_count`.
- Optimize only token-level MSE. Log token-level cosine similarity without adding it to the loss.
- Do not provide task text or captions to the GWM forward pass. The current full RGB remains part of RAT; downstream, the unchanged scorer separately builds a task-query embedding from language and visual context to rank predicted trajectory embeddings.
- Compute Qwen embeddings online; do not introduce an embedding cache.
- Schedule, checkpoint, evaluate, and resume at optimizer-step granularity: cosine LR annealing over the configured total steps, a canonical checkpoint plus optional WISER-dev open-loop evaluation every N steps (N configurable), and resume restoring model, optimizer, scheduler, step counter, sampler position, and random state.
- Express the initial budget as the step count equivalent to the existing three-epoch cluster recipe on the selected corpus, keeping total steps configurable for longer training.

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

Canonical export always embeds the `config` key: the evaluation loader silently falls back to default `TransformerConfig` values (`dim=256`) when it is missing and then fails strict loading far from the real cause. Because checkpoints are loaded with `weights_only=False`, the pickled import path `gwm_wiser.models.transformer.TransformerConfig` is part of the compatibility contract and must never move. Every run records the exact versions of `transformers` (pinned `4.57.6`), `torch`, and other preprocessing-relevant libraries in its metadata, because preprocessing behavior — and therefore token counts — is version-dependent.

## Development evaluation

Training data and optional development data use separate configuration:

```text
dataset_adapter:           selected_source
dataset_root:              /path/to/selected_corpus
wiser_dev_dataset_root:    /path/to/wiser_dataset  # optional
```

When configured, WISER-dev evaluation reads only `wiser_dev_dataset_root/merged_test`, computes the existing open-loop MSE and cosine metrics, and never backpropagates them. These signals may select a phase-one development checkpoint, so they must not be described as held-out evaluation.

Closed-loop evaluation is performed by the existing `gwm_wiser/scripts/gwm_eval.py` without modifying its planner, logged skill library, proposal horizon, replanning interval, or task environments. The evaluator conditions the GWM on candidate robot appearances cut from observed segmentation (`image_1_robot_state`), which are occlusion-clipped by scene objects; phase-one mask-based training supervision matches this condition provenance by design (ADR-0015).

Closed-loop readiness is an operational precondition, not part of the milestone: the skill-library videos under `gwm_skills/*/lerobot_data/videos/` are gitignored, so a fresh checkout must regenerate them with `save_skill.py` or copy them before evaluation, and a `--use_gt` oracle run (which loads no GWM) is the recommended toolchain check on a new machine. A known evaluator limitation is documented rather than fixed: `--eval_rounds > 1` raises because the planner is not reset between rounds, so multiple evaluation rounds run as separate invocations. Report both WISER train and test task results; the primary phase-one threshold is:

```text
WISER-dev test success_at_end_mean >= 0.70
```

Report `success_once_mean`, `is_grasped_mean`, `near_goal_mean`, `tcp_near_goal_mean`, the train/test gap, open-loop MSE, and cosine as secondary diagnostics; these are the metric keys the evaluator actually emits. Each split contains 24 configs x 12 task instances = 288 episodes under the default single-round protocol, so a 0.70 success rate carries a 95% confidence half-width of roughly ±0.05; the threshold is judged on the point estimate with the interval reported alongside. The established WISER-trained model is outside this implementation's execution scope.

This result measures cross-domain semantic forward-model transfer within the Franka/Panda embodiment. It is not evidence of cross-embodiment generalization or physical-dynamics transfer and need not become a paper claim. An optional UR5 simulation diagnostic is deferred because the unchanged evaluator currently supports only Panda and XArm6; adding UR5 would require a separately scoped evaluation integration and would not alter the primary milestone.

## Planned module interface

All new implementation will remain below `real_world_gwm/`. Its intended shared interface has three entry points:

- `audit.py` inspects the explicitly selected source adapter and emits the manifest.
- `train.py` trains, resumes, optionally evaluates WISER-dev, and saves canonical checkpoints.
- `evaluate_open_loop.py` evaluates a saved checkpoint against an explicitly supplied development dataset.

Each dataset integration lives behind a source adapter and includes a corresponding human-inspection visualization script, such as `adapters/<source>/visualize.py`. RobotSeg directory conventions, mask selection, augmentation, dynamic positions, and canonical checkpoint conversion remain internal implementation details.

## Debugging and verification

Development is deliberately two-staged and kept simple for prototyping. Everything through small-data training is validated locally on the development machine (single RTX 3090, 24 GB); the compute cluster is used only for formal training runs, checkpoint selection, and closed-loop evaluation. The implementation must support limited videos/windows, one-batch overfitting, and a one-step dry run; dataset auditing and unit tests run without loading the full Qwen model.

Local acceptance gate (all on the development machine):

- Window indexing, masking, augmentation, manifest generation, and failure modes have unit coverage, including a test that full-RGB jitter leaves robot-only RGB byte-identical.
- Every selected source's visualization script renders the exact normalized clips and RAT samples consumed by training, and audit plus human inspection pass on real released source data.
- A small-model test covers dynamic sequence lengths and MSE.
- The 1,620-token wrapper parity test passes.
- Canonical checkpoint strict-loading passes.
- A real-data integration smoke test completes forward, backward, save, resume, and load, and a tiny overfit run produces a clear loss decrease. Where 24 GB does not fit the frozen embedder plus the full 4096-dim GWM training state, the smoke test uses a reduced GWM configuration; the full-size configuration is exercised on the cluster.
- Sustained throughput of online embedding plus an optimizer step is measured on the target GPU class before committing cluster budget, since the frozen vision-tower embedding is the training-loop floor.

Cluster stage: the training-run matrix, step-based checkpointing, WISER-dev open-loop checkpoint selection, and closed-loop evaluation. Full selected-corpus training and the 70% closed-loop milestone are experiment outcomes, not code-level completion criteria.

## Deferred phase-two decisions

Franka-class real-hardware work begins only after phase one. It must separately decide the trajectory proposal method, source of camera calibration and URDF state, rendering interface, real validation corpus, hardware-checkpoint selection, and operational safety procedures. None of those decisions may be inferred from WISER-dev performance.
