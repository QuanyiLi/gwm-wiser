# Review of the Retired VRS Prototype

This review preserves the useful evidence from the former `origin/dev` branch before that branch is retired. The reviewed tip was `05c6d4f` with merge base `ead1d76` on `origin/main`; its three commits added a RobotSeg loader, a resumable OneDrive helper, and an attempted WISER timing alignment.

The branch is historical implementation evidence, not the implementation base for phase one. No commit from it should be merged wholesale into `hardware`.

## What the prototype implemented

- A `RobotSegGwmDataset` that discovered several guessed VRS-style directory layouts, naturally ordered image names, paired RGB with a mask, formed robot-only RGB, and returned WISER-shaped sample keys.
- A `dataset_format=robotseg` branch inside the existing `gwm_train.py`.
- Extra `sample_fps` and `raw_fps` plumbing in the shared Qwen preprocessor.
- A resumable downloader for temporary OneDrive train/test URLs.
- A preparation script that generated a small loader fixture from the official RobotSeg overview GIF and saved tensor/PNG previews.
- A mask-first survey of RobotSeg/VRS and several other robot datasets.

## What was and was not demonstrated

The preparation script exercised tensor assembly on a generated fixture. It cropped an xArm panel from the overview GIF and derived a foreground mask by thresholding its difference from white; it did not read a released VRS clip and its released masks. The branch README reported a batch shape of `[1, 6, 3, 224, 224]` for full RGB, mask-derived robot-only RGB, and repeated-channel masks.

No committed evidence demonstrates any of the following:

- Successful loading or auditing of the complete released VRS corpus.
- An exact-path visualization of samples consumed by training.
- A Qwen-to-GWM forward or backward pass on the new data.
- A falling training loss, saved compatible checkpoint, or resume path.
- WISER open-loop or closed-loop performance.

The fixed `224x224` fixture would produce a latent length different from the original fixed 1,620-token GWM interface, while the prototype did not add the dynamic-length wrapper or canonical checkpoint conversion. The branch therefore establishes a useful loader sketch, not a runnable or evaluated training pipeline.

## Conclusions retained in the current plan

- Observation-only GWM pretraining can be constructed from aligned consecutive full RGB and robot-only RGB without exposing raw actions to GWM. Phase one now derives robot-only RGB from whole-robot masks to match the WISER-dev evaluation condition; state rendering is the phase-two path (ADR-0015).
- VRS has ordered frame indices but no reliable corpus-wide FPS or timestamp contract. Its windows must remain ordinal; clip length must not be reinterpreted as elapsed seconds.
- The useful VRS convention is the whole-robot mask category `002`. Arm and gripper categories are not substitutes for a complete robot appearance.
- VRS train-time dense masks may include propagated or pseudo labels, so mask provenance and alignment require audit and human inspection.
- Dataset acquisition helpers may be resumable and source-specific, but downloading remains separate from the training entry point.
- The earlier survey named Exylos Bimanual Table Spill Cleanup and NVIDIA PhysicalAI Robotics Manipulation Augmented as state-rich leads. These are discovery leads only and must be revalidated against the current qpos-, calibration-, usable-window-, and license-first matrix.
- State-rich corpora that the earlier mask-first survey dismissed solely for lacking released robot masks must still be reconsidered: predicted whole-robot masks can admit them in phase one, and their renderable state bundles matter for phase two.

### Historical survey leads

The old survey did not download and inspect most candidates, so its ranking is not preserved. Its discovery list is retained with a disposition that matches the current experiment:

| Lead | Current disposition |
| --- | --- |
| RobotSeg/VRS | Ready mask-based fallback candidate; ordinal time, pseudo-mask provenance, and full-corpus usability still require audit |
| Exylos Bimanual Table Spill Cleanup | Revisit as a state- and mask-rich Franka lead; verify single-main-camera suitability, qpos, calibration, and license |
| NVIDIA PhysicalAI Robotics Manipulation Augmented | Revisit as a state-rich simulation lead; verify render metadata, usable semantic diversity, labels, and license |
| DROID, BridgeData, and Open-X-Embodiment sources | Reconsider sources that were excluded only for missing masks; phase one requires whole-robot masks (released or predicted), while renderable state and camera metadata count toward phase-two readiness |
| Dexora | Keep as a low-confidence state-rich lead; robot identity, main-camera signal, state semantics, and mask coverage were not inspected |
| RoboEngine/RoboSeg and RoVi-Aug | Useful segmentation or data-generation references, but the old survey did not establish a consecutive, ready-to-train corpus for this experiment |
| Roboflow Robot Arm Segmentation and synthetic DaVinci instruments | Do not prioritize for phase one: the former is small and non-sequential, while the latter is outside the Franka tabletop target |

## Prototype choices not carried forward

| Former `dev` behavior | Current requirement |
| --- | --- |
| Add a RobotSeg switch to the existing trainer | Keep all adaptation below `real_world_gwm/` behind a source-neutral adapter |
| Resize every frame to `224x224` | Apply an explicit aspect-preserving per-source pixel budget through the existing preprocessing and audit the resulting token count (ADR-0014); never force a fixed square shape |
| Uniformly map every complete clip onto 60 synthetic 20 Hz steps | Use real elapsed seconds only with a reliable clock; otherwise use explicitly ordinal windows |
| Produce one stretched window per clip | Enumerate every complete six-frame window under configurable `frame_step` and `window_stride` |
| Repeat the last frame for a short window | Reject incomplete windows and report them |
| Train on VRS `train` while treating VRS `test` as evaluation | If VRS is selected, absorb both released trees into the pretraining corpus; WISER-dev is the separate development evaluation |
| Search `robot`, `arm`, and `gripper` paths and accept the first match | Require deterministic whole-robot mask `002` for the VRS mask route |
| Return zero-valued state and action placeholders | Normalize adapters to aligned full RGB and robot-only RGB; do not leak source-specific compatibility fields into the training core |
| Derive `sample_fps`, `raw_fps`, and timestamps from a synthetic 3-second clip mapping | Do not fabricate a source clock; preserve the existing six-frame Qwen preprocessing unless real timing metadata justifies a source-specific policy |
| Validate with a GIF crop and generated foreground mask | Visualize and audit released data through the exact training adapter; synthetic fixtures test plumbing only |
| Optimize for mask availability and cross-embodiment pretraining | Select by usable RAT signal — mask-first in phase one (ADR-0015) — and evaluate cross-domain Franka transfer |

## Concepts worth reusing

Natural frame ordering, explicit source-frame identifiers, deterministic RGB/mask pairing, manifest generation, resumable acquisition, and side-by-side previews remain useful implementation ideas. They must be rebuilt inside the standalone adapter boundary with deterministic mask precedence, non-destructive paths, current temporal semantics, native-aspect latent handling, and automated coverage.

No new ADR is needed for this review. The relevant decisions are already recorded by [ADR-0005](adr/0005-use-ordinal-six-frame-windows.md), [ADR-0006](adr/0006-support-variable-length-native-aspect-latents.md), [ADR-0007](adr/0007-isolate-the-adaptation-and-export-canonical-checkpoints.md), [ADR-0010](adr/0010-keep-the-training-corpus-provisional-pending-signal-survey.md), [ADR-0012](adr/0012-sample-by-elapsed-time-when-the-source-has-a-clock.md), and [ADR-0013](adr/0013-prioritize-state-rendered-robot-appearance.md).
