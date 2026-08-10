---
status: superseded by ADR-0016
---

# Use ordinal six-frame windows

This decision has no remaining object: VRS left the training scope entirely (ADR-0016, as amended), and both selected sources are timestamped (ADR-0012). Retained as history.

When VRS is used, it exposes consecutive frame order but no reliable timestamps or extraction rate, so its adapter defines a RAT window by frame ordinal: `[i + k * frame_step for k in range(6)]`, with configurable `frame_step` and window-start stride, both defaulting to one. Interpreting VRS windows as seconds was rejected because the released corpus cannot support that claim without reconstructing its source timelines.

## Consequences

Only complete windows are trained; the final frame is never repeated to pad a short tail. Changing to timestamped windows later would change the temporal meaning of the corpus and should supersede this ADR.
