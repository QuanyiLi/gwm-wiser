---
status: accepted
---

# Use ordinal six-frame windows

When VRS is used, it exposes consecutive frame order but no reliable timestamps or extraction rate, so its adapter defines a RAT window by frame ordinal: `[i + k * frame_step for k in range(6)]`, with configurable `frame_step` and window-start stride, both defaulting to one. Interpreting VRS windows as seconds was rejected because the released corpus cannot support that claim without reconstructing its source timelines.

## Consequences

Only complete windows are trained; the final frame is never repeated to pad a short tail. Changing to timestamped windows later would change the temporal meaning of the corpus and should supersede this ADR.
