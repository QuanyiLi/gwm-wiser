# Real-World Grounded World Model

This context describes a semantic outcome model pretrained from real and simulated robot video and used, inside the TiPToP planning system, to score externally proposed robot trajectories. Its language separates semantic prediction from trajectory generation, physics, control, and safety.

## Model and representations

**Grounded World Model (GWM)**:
A model that predicts a candidate trajectory's future visual outcome in a language-aligned latent space.
_Avoid_: Physics model, action generator, controller

**Semantic Outcome**:
The language-relevant visual consequence of a candidate robot trajectory, independent of whether that trajectory is physically feasible.
_Avoid_: Dynamics rollout, success guarantee

**Qwen Visual Latent**:
The token-level internal visual representation produced by the frozen Qwen3-VL embedding model.
_Avoid_: Pooled retrieval vector, pixel prediction

**Rendering-based Action Tokenization (RAT)**:
A candidate-trajectory representation consisting of the current full scene followed by future robot-only appearances.
_Avoid_: Raw action conditioning, action tokens

**RAT Window**:
An ordered six-frame observation window used to construct one GWM condition and target pair.
_Avoid_: Action chunk, timed rollout

**Full-RGB Trajectory**:
The ordered full-scene RGB frames that define the target semantic outcome.
_Avoid_: Robot rendering, segmentation video

**Robot-only RGB**:
An RGB image showing only the robot on black, rendered from robot state with no scene object, depth, or observed mask used to clip it.
_Avoid_: Binary mask, arm-only image, masked observed robot, raw action

**Shared Franka Renderer**:
The single state renderer, with per-call camera parameters and per-source URDF, that produces robot-only RGB for both training-data generation and inference-time candidate scoring.
_Avoid_: Per-purpose renderers, mask pipeline, WISER-only renderer

**Operating Grid**:
The single token grid, anchored to the inference cameras, that every training source's pixel budget resizes onto and that inference scoring uses.
_Avoid_: Native-resolution grids, per-source token scales, WISER grid

## Planning system

**Candidate Trajectory**:
A proposed robot motion paired with the future robot-only appearances that condition GWM scoring.
_Avoid_: GWM prediction, retrieved outcome

**Trajectory Proposal Module**:
The external module — TiPToP's geometry-only perception and cuTAMP search — that generates executable candidate trajectories and the information needed to render their future robot appearances.
_Avoid_: GWM, semantic scorer

**Execution Layer**:
The external planning, feasibility, collision, control, and safety mechanisms that validate and execute a selected trajectory.
_Avoid_: GWM safety logic, semantic feasibility

**Semantic Generalization**:
Generalization of language-relevant visual outcome prediction across scenes, objects, backgrounds, and viewpoints while physical skills and dynamics remain the responsibility of other modules.
_Avoid_: Cross-embodiment generalization, skill generalization, dynamics generalization

**Cross-Domain Forward-Model Transfer**:
Useful semantic visual-outcome prediction after moving a GWM from one or more real or simulated pretraining domains to a different observation domain without target-domain training examples.
_Avoid_: Cross-embodiment transfer, physics transfer, universal world model

**System Generalization**:
The composed capability produced by semantic GWM scoring plus an independently generalizing trajectory proposal and execution layer.
_Avoid_: GWM generalization

**Task Query**:
The language instruction and current visual context compared with predicted trajectory outcomes by the external semantic scorer; it is not an input to the GWM forward pass.
_Avoid_: GWM language conditioning, training caption

## Data and evaluation

**Selected Training Corpus**:
The documented real-plus-simulation mixture of MolmoAct2-DROID and the MolmoBot-Data Franka subset.
_Avoid_: VRS, WISER data, provisional corpus

**Training Clip**:
An ordered robot-video sequence whose frames pair full-scene RGB with aligned state-rendered robot-only RGB, independent of whether the source is real or simulated.
_Avoid_: Raw dataset record, action trajectory, mask-derived sample

**Rendered Tree**:
The single normalized on-disk contract — per-clip robot-only frames plus an alignment-and-provenance record — that preparation writes and training reads, hiding every source format.
_Avoid_: Per-source runtime adapter, raw source layout, RGB frame dump

**Camera-Recovery Gate**:
The pre-flight requirement that a source stream without published camera parameters obtains per-episode verified pose and intrinsics before any rendering or training use.
_Avoid_: Fuzzy metadata join taken on faith, nominal-intrinsics assumption, optional calibration step

**Calibration Join**:
The exact match from a converted episode back to its original release record — by verbatim language-annotation triple for DROID — that recovers per-episode camera pose, intrinsics, and idle ranges; ambiguous keys are dropped, never guessed.
_Avoid_: Fuzzy state matching, nominal shared camera, manual episode pairing

**Edge Gate**:
The render-time admission check that scores a candidate calibration by how much better its rendered robot silhouette aligns with observed oriented edges than chance and than a deliberately perturbed camera; failing streams never enter the Rendered Tree.
_Avoid_: Trusting joined calibration, manual spot-check as the criterion, segmentation-model dependency

**Main Camera**:
The single verified camera stream a Training Clip is bound to; one source episode may yield multiple Training Clips, one per admitted exterior stream.
_Avoid_: Camera fusion within a clip, wrist camera, one-camera-per-source rule

**Temporal Sampling Schedule**:
The six elapsed-time offsets used for a timestamped RAT window; both selected sources have reliable clocks, so no ordinal fallback exists on the training path.
_Avoid_: Trajectory progress, assumed FPS, hidden frame stride

**Time-Scale Augmentation**:
The per-sample rescaling of the Temporal Sampling Schedule at a jittered anchor that varies the spacing between a window's frames, teaching robustness to the planner's future-point interval; evaluation stays at canonical scale.
_Avoid_: Frame-step subsampling tied to source FPS, randomized evaluation schedule, time-embedding conditioning

**Pixel Budget**:
The per-source aspect-preserving resize applied through the existing Qwen preprocessing to land that source on the Operating Grid.
_Avoid_: Fixed-shape resize, native-resolution mandate, token ceiling

**Pre-flight Gate**:
The per-camera-stream verification that URDF re-projection pixel-aligns with the released RGB, required before any large-scale rendering or training use of that stream.
_Avoid_: Optional sanity check, one-off calibration, corpus-level average

**Selection A/B**:
The development comparison of GWM candidate scoring against confidence-only and random selection on droid-sim pick tasks.
_Avoid_: Leaderboard run, held-out benchmark, WISER evaluation

**Development Evaluation**:
Evaluation used to inspect or select development checkpoints and therefore not a held-out estimate.
_Avoid_: Zero-shot test, final benchmark

**Development Milestone**:
A canonical checkpoint whose candidate selection beats both Selection A/B baselines on droid-sim pick tasks, en route to the MolmoSpaces Pick target.
_Avoid_: WISER threshold, leaderboard guarantee, hardware milestone

**Canonical Checkpoint**:
A checkpoint whose learned parameters and metadata can be loaded strictly by the fixed-length GWM loader that `gwm-server` deployment uses.
_Avoid_: Training-wrapper checkpoint

**Hardware Checkpoint**:
A checkpoint selected for later real-robot deployment experiments within the TiPToP framework.
_Avoid_: Best development checkpoint
