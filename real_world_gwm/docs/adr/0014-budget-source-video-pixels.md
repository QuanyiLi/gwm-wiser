---
status: accepted
---

# Budget source-video pixels per source

Real robot-video sources arrive at resolutions whose native token counts (up to 2,304 per level, 9,216 concatenated) far exceed the WISER-scale interface, so each source adapter applies an aspect-preserving pixel budget through the existing Qwen preprocessing hooks, with the default budget chosen to land near the WISER-dev evaluation scale of 405 tokens per level. Fixed-shape normalization to the WISER grid was rejected because it distorts non-2:1 sources and would supersede ADR-0006; unbudgeted native-resolution training was rejected because it multiplies compute, fragments batch shapes, and evaluates the flat-index GWM positions far outside the grid distribution seen by the unchanged WISER evaluator.

## Consequences

The numeric default budget, the 2,048-token fail-fast ceiling, and any later ceiling increase remain reversible experiment controls recorded in the phase plan, audit manifest, and checkpoints rather than in this ADR. Native aspect ratio is preserved (ADR-0006), so per-level token counts still vary across sources and the variable-length training wrapper remains required.

Implementation note (verified 2026-08-05): the existing preprocessing applies an injected pixel window twice — once per frame inside `fetch_image` (factor 64, via propagated content kwargs) and once at video level (factor 32) — so budgeted grids can differ from the uninjected default path for the same input size (224x448 lands on (3,16,32) instead of (3,18,30)). The budget is a pure aspect-preserving spatial resize before patchification: no patches, tokens, or frames are ever dropped. Measured on the released VRS test tree, the default budget maps 16:9 sources exactly onto the WISER evaluation grid (3,18,30)=1,620 tokens, while removing the budget would send them to (3,40,74)=8,880.
