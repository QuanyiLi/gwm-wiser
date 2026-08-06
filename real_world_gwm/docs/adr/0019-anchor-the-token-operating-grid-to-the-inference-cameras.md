---
status: accepted
---

# Anchor the token operating grid to the inference cameras

All training sources and inference scoring share one token operating grid, anchored to the observation scale of the tiptop inference cameras (droid-sim / MolmoSpaces). Each source's pixel budget (mechanism unchanged from ADR-0014) resizes it onto that grid — upsampling sources below it (MolmoAct2-DROID at 320×180) and downsampling sources and inference frames above it. For ~16:9 observations the grid is `(3,18,30)` = 1,620 concatenated tokens, coinciding with the existing fixed-length model interface.

Two alternatives were rejected:

- **Per-source native grids** ("the model is resolution-agnostic"): `VariableLenGWM` (ADR-0006) accepts any length architecturally, but the learned flat-index position distribution does not transfer for free across grids, and a grid mismatch between training and `gwm-server` scoring is exactly the silent degradation ADR-0014 documented. Resolution-agnosticism is a capability, not a license to train and score on different grids.
- **Anchoring down to the smallest source** (no upsampling): would throw away inference-side resolution to match the weakest training source; upsampling the weak source adds no information but costs nothing and keeps the position distribution shared.

ADR-0014's WISER-scale anchor rationale is superseded by this inference-scale anchor; its numeric value happens to coincide for 16:9. The grid is pinned at `(3,18,30)` = 1,620: both evaluation targets are verified ~16:9 — droid-sim observes at 1280×720 and MolmoSpaces evaluates at 624×352 (2026-08-06 research report).

## Consequences

Training window preprocessing and `gwm-server` inference preprocessing declare the same operating grid; the audit fail-fasts on sources that cannot reach it within the 2,048-token ceiling. Native-aspect deviations from exact 16:9 remain covered by variable-length training (ADR-0006). If a future inference platform changes observation scale materially, the anchor moves with it and the model is retrained or adapted — the grid is part of the deployment contract, recorded in every checkpoint alongside the pixel budget.
