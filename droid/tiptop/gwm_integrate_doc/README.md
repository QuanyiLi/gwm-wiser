# GWM×TiPToP Integration Docs

This directory is the **plan of record for the tiptop-side integration milestone**: turning TiPToP into a semantic-free trajectory proposer on droid-sim and wiring GWM in as the only semantic component (scoring proposed trajectories against the natural-language task).

- [plan.md](plan.md) — the milestone plan: verified system facts, target architecture, decision ledger (G-1…G-15), mini-milestones GI-0…GI-5, risks.

## Relationship to the gwm-wiser docs

The **system-level plan** (milestones M0–M4) lives in the gwm-wiser repo at
`/root/code/gwm/gwm-wiser/real_data_train/docs/tiptop-gwm-integration-plan.md`, and the **M2 retraining plan** at `.../docs/plan.md`. This directory owns the work formerly labeled **M0 (lightened) + M1** there; where the documents differ, this one supersedes the M0/M1 details (recorded in the integration plan's decision table — D3 superseded, D4 revised).

Naming note: the two repos use two numbering schemes. `M0–M4` = system milestones (integration plan). `Stage 1–4` = execution stages (gwm-wiser plan.md; Stage 1 = M2 retraining, Stage 2 = this work). This directory adds `GI-0…GI-5` (integration mini-milestones) and `G-1…` (decision ledger entries), scoped to this milestone only.

## Code layout (once implementation starts)

- `gwm_tiptop/` (new package in this repo, own git branch) — mask-free perception, `run_proposals` orchestrator over cuTAMP-as-a-library, thin gwm-server HTTP client. **The original `tiptop/` package is not modified** — the Gemini baseline arm must stay runnable for the A/B.
- `gwm-server` lives in the gwm-wiser repo (its pinned environment owns the model stack).
- droid-sim-evals is forked to add: external-cam observation, automatic pick-success detection, and a batch runner.
