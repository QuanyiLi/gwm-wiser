# Real-World Grounded World Model

This context describes a semantic outcome model pretrained from real or simulated robot video and used to score externally proposed robot trajectories. Its language separates semantic prediction from trajectory generation, physics, control, and safety.

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
An RGB image containing only the robot on black; a state-rendered version shows camera-visible robot geometry without scene objects or scene-derived occlusion.
_Avoid_: Binary mask, arm-only image, raw action

## Planning system

**Candidate Trajectory**:
A proposed robot motion paired with the future robot-only appearances that condition GWM scoring.
_Avoid_: GWM prediction, retrieved outcome

**Trajectory Proposal Module**:
An external module that generates executable candidate trajectories and the information needed to render their future robot appearances.
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

**Training Clip**:
An ordered robot-video sequence whose frames pair full-scene RGB with aligned robot-only RGB, independent of whether the source is real or simulated and whether the robot appearance came from a mask or a state renderer.
_Avoid_: VRS sample, action trajectory, raw dataset record

**Main Camera**:
The single RGB observation stream selected from a source for RAT construction and aligned robot rendering.
_Avoid_: Multi-view input, camera fusion, auxiliary camera

**Temporal Sampling Schedule**:
The six elapsed-time offsets used for a timestamped RAT window, or an explicitly identified ordinal fallback when the source has no reliable clock.
_Avoid_: Trajectory progress, assumed FPS, hidden frame stride

**Whole-Robot Mask**:
A foreground mask containing every visible part of the robot and excluding the scene background.
_Avoid_: Arm mask, gripper mask, object mask

**State-Rendered Robot Appearance**:
Geometry-only robot RGB produced from synchronized qpos, robot geometry, camera calibration, and robot pose, with no scene object, depth, or observed mask used to clip it.
_Avoid_: Raw action image, predicted future scene, masked observed robot

**VRS Corpus**:
The RobotSeg video robot segmentation corpus retained as the current ready-to-use candidate for GWM training, not the permanently selected corpus.
_Avoid_: RobotSeg policy data, action dataset

**Selected Training Corpus**:
The out-of-domain real-video source, simulation source, or documented mixture chosen after comparing the amount and quality of usable RAT supervision.
_Avoid_: VRS by default, WISER data

**Development Evaluation**:
Evaluation used to inspect or select development checkpoints and therefore not a held-out estimate.
_Avoid_: Zero-shot test, final benchmark

**WISER-dev**:
The WISER observations and simulated task instances used for phase-one feedback, including `merged_test` open-loop diagnostics and the original train/test task labels in closed-loop development evaluation.
_Avoid_: Held-out WISER test

**WISER Training Contamination**:
Use of WISER observations, masks, latents, logged trajectories, or WISER-generated rollouts in the optimization corpus or gradient computation; reuse of generic model code, Panda assets, or renderer machinery alone is not contamination.
_Avoid_: Shared URDF, shared architecture, development evaluation

**Canonical Checkpoint**:
A checkpoint whose learned parameters and metadata can be loaded strictly by the original fixed-length GWM evaluator.
_Avoid_: Training-wrapper checkpoint

**Hardware Checkpoint**:
A checkpoint selected without WISER-dev feedback for later real-robot deployment experiments.
_Avoid_: Best WISER checkpoint

**Phase-one Milestone**:
A pretrained canonical checkpoint meeting the Franka WISER-dev target without using WISER samples for training.
_Avoid_: Real-world deployment milestone, final generalization result
