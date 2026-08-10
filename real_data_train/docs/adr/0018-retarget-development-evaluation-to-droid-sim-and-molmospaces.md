---
status: accepted
---

# Retire WISER; evaluate on droid-sim and MolmoSpaces

WISER leaves the project scope entirely. There is no WISER open-loop MSE/cosine during training, no WISER closed-loop evaluation of any checkpoint, and consequently no WISER-contamination boundary left to enforce. Development evaluation and checkpoint selection move to the TiPToP integration targets ([integration plan](../tiptop-gwm-integration-plan.md)):

- Routine training diagnostic: held-out MSE/cosine on windows from the selected corpus, real and simulation splits reported separately.
- Development milestone: candidate-selection A/B on droid-sim (IsaacLab, Franka/DROID rig) pick tasks — GWM scoring versus M2T2-confidence-only versus random-candidate baselines; GWM must beat both.
- Downstream target: the MolmoSpaces leaderboard Pick subset, match-or-beat TiPToP's published baseline.

The WISER evaluator was rejected even as an optional diagnostic because its segmentation-cut conditioning systematically mismatches the state-rendered condition that training and inference now share (ADR-0017), so its curves invite misreading rather than insight, and because the target system is the TiPToP DROID Franka setting in which WISER has no role.

## Consequences

ADR-0008 (WISER as development evaluation) is superseded and ADR-0011 (WISER contamination boundary) is moot — with no WISER evaluation there is no result a WISER sample could inflate; both are retained as history. The phase-one 70% milestone and its plan are deprecated. WISER-coupled code paths (`--wiser_dev_dataset_root`, the wiser-repro adapter and launcher) are retired from the workflow; their code disposition is an operational decision recorded in the plan. Checkpoint selection uses droid-sim selection quality; hardware-checkpoint selection remains a separate later decision within the TiPToP framework.
