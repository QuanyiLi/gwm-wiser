# Architecture Decision Records

ADRs are numbered in decision order. Status is explicit because later decisions may supersede earlier experiment assumptions. As of 2026-08-06 the project is organized around three pillars — data = MolmoAct2-DROID + MolmoBot-Data Franka, sim benchmark = droid-sim + MolmoSpaces, framework = TiPToP — and every WISER- or VRS-scoped decision is superseded or moot.

| ADR | Status | Decision |
| --- | --- | --- |
| [0001](0001-bound-gwm-to-semantic-outcome-prediction.md) | Accepted | Bound GWM to semantic outcome prediction |
| [0002](0002-preserve-rat-in-qwen-visual-latent.md) | Accepted | Preserve RAT in the Qwen visual latent |
| [0003](0003-keep-trajectory-proposal-and-execution-external.md) | Accepted | Keep trajectory proposal and execution external |
| [0004](0004-use-all-vrs-video-for-phase-one-training.md) | Superseded by ADR-0010 | Use all VRS video for phase-one training |
| [0005](0005-use-ordinal-six-frame-windows.md) | Superseded by ADR-0016 | Use ordinal six-frame windows when VRS is used |
| [0006](0006-support-variable-length-native-aspect-latents.md) | Accepted | Support variable-length native-aspect latents |
| [0007](0007-isolate-the-adaptation-and-export-canonical-checkpoints.md) | Accepted | Isolate the adaptation and export canonical checkpoints |
| [0008](0008-treat-wiser-as-development-evaluation.md) | Superseded by ADR-0018 | Treat WISER as development evaluation |
| [0009](0009-restrict-vrs-use-to-research.md) | Accepted (moot — VRS unused) | Restrict VRS use to research |
| [0010](0010-keep-the-training-corpus-provisional-pending-signal-survey.md) | Superseded by ADR-0016 | Keep the training corpus provisional pending a signal survey |
| [0011](0011-define-wiser-contamination-by-data-provenance.md) | Superseded by ADR-0018 | Define WISER contamination by data provenance |
| [0012](0012-sample-by-elapsed-time-when-the-source-has-a-clock.md) | Accepted | Sample by elapsed time when the source has a reliable clock |
| [0013](0013-prioritize-state-rendered-robot-appearance.md) | Superseded by ADR-0015 | Prioritize state-rendered robot appearance |
| [0014](0014-budget-source-video-pixels.md) | Accepted (anchor superseded by ADR-0019) | Budget source-video pixels per source |
| [0015](0015-prefer-mask-derived-robot-appearance-in-phase-one.md) | Superseded by ADR-0017 | Prefer mask-derived robot appearance in phase one |
| [0016](0016-train-on-molmoact2-droid-and-molmobot-franka.md) | Accepted (amended 2026-08-06) | Train on MolmoAct2-DROID and the MolmoBot-Data Franka subset |
| [0017](0017-render-robot-appearance-from-state-with-the-shared-franka-renderer.md) | Accepted | Render robot appearance from state with the shared Franka renderer |
| [0018](0018-retarget-development-evaluation-to-droid-sim-and-molmospaces.md) | Accepted | Retire WISER; evaluate on droid-sim and MolmoSpaces |
| [0019](0019-anchor-the-token-operating-grid-to-the-inference-cameras.md) | Accepted | Anchor the token operating grid to the inference cameras |
| [0020](0020-normalize-sources-into-a-rendered-tree-at-preparation-time.md) | Accepted | Normalize sources into a rendered tree at preparation time |
