---
status: accepted
---

# Train on MolmoAct2-DROID and the MolmoBot-Data Franka subset

*Amended 2026-08-06 (same day): VRS demoted from "optional ablation arm" to documentation-only, and the intrinsics claim corrected after primary-source verification.*

*Amended 2026-08-06 (implementation, later the same day): on-disk verification showed the MolmoAct2-DROID `camera_extrinsics.*` columns are zero-filled across the entire release — the dataset publishes NO usable camera calibration. DROID streams are blocked behind the camera-recovery gate, and by decision (plan.md D-26) MolmoAct2-DROID is postponed: run 1 trains on the MolmoBot Franka subset alone, with DROID admission revisited as a later work item. On the MolmoBot side, the h5 `intrinsic_cv` belongs to an internal 480×480 render and must not be used for the released 624×352 mp4s — working intrinsics come from each camera's `frozen_config` vertical FOV, verified by re-projection overlay (plan.md, pre-flight status).*

The ADR-0010 signal survey is concluded by the TiPToP integration decision (2026-08-06, [integration plan](../tiptop-gwm-integration-plan.md) D7): the Selected Training Corpus is the documented real-plus-simulation mixture of the [MolmoAct2-DROID-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-DROID-Dataset) (real, quality-filtered DROID, LeRobot, Apache-2.0; 320×180 @ 15 fps; extrinsics per camera but no intrinsics, and the DROID episode IDs/camera serials needed for an exact calibration join were dropped in conversion) and the [MolmoBot-Data](https://huggingface.co/datasets/allenai/MolmoBot-Data) Franka tabletop subset (simulation, MuJoCo, FR3 + Robotiq 2F-85, ~1.55 M scripted episodes, per-frame intrinsics and camera-to-world, per-timestep qpos, ODC-BY). Both sources target the DROID Franka setting used by the droid-sim and MolmoSpaces evaluations.

MolmoAct2-DROID intrinsics are recovered by nominal-plus-self-calibration: ZED factory nominal values scaled to the release resolution, refined per episode by aligning the URDF render (known qpos and extrinsics) against the observed robot, gated by the pre-flight re-projection check; if that fails at scale, the documented fallback is raw DROID plus the KarlP calibration release. Fuzzy metadata joins were rejected as unverifiable.

VRS is retained **in documentation only** — no training arm, no ablation arm, no gate:

- Its unresolved research-only licensing (ADR-0009) conflicts with a public leaderboard submission and checkpoint release.
- It publishes no qpos or camera calibration, so it cannot feed the state-rendered robot-appearance path (ADR-0017), and its mask-derived appearance mismatches the render-based inference condition — an ablation against it would compare against a condition distribution the system never uses, so its outcome could not change any decision.

Its grill-established configuration practices (audit-informed sampling steps, window rejection over tail-padding, measured pixel-budget behavior) carry over as reference for the Molmo adapters.

## Consequences

ADR-0010 is superseded; corpus identity is no longer provisional. Each source requires its own adapter, audit, exact-path visualization, and per-stream pre-flight calibration verification before large-scale use; the real/sim mixture and subsampling policy are recorded per run, with a real-only/sim-only/mixed ablation reserved if the mixed signal is weak. Training on MolmoBot-Data places resulting checkpoints in the MolmoSpaces "trained on MolmoBot data" class, which must be stated alongside any comparison to TiPToP's not-MolmoBot-trained score. The disposition of the frozen VRS adapter code is an operational decision recorded in the plan; ADR-0009 remains the licensing record should VRS ever be reconsidered.
