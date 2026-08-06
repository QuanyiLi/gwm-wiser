---
status: superseded by ADR-0017
---

# Prefer mask-derived robot appearance in phase one

This historical decision is no longer current; ADR-0017 returns to the state-rendered path with a single shared renderer for training and inference.

The unchanged WISER-dev evaluator conditions the GWM on candidate robot appearances cut from observed segmentation, which scene objects occlusion-clip, so phase-one training derives robot-only RGB from observed or predicted whole-robot masks even when a renderable state bundle exists. State-render-first (ADR-0013) was rejected for phase one because it builds a systematic train-versus-evaluation condition mismatch into the development milestone and front-loads URDF/calibration engineering before any transfer signal exists; deliberate per-frame mixing remains rejected.

## Consequences

ADR-0013 is superseded; its state-rendered path returns in phase two, where the training condition and the qpos-driven proposal renderer are aligned together. Maskless state-rich sources may enter phase one only through predicted whole-robot masks with recorded provenance, while qpos, calibration, and render-bundle quality stay in the survey as phase-two readiness. Robot-only RGB remains exempt from photometric augmentation, and each source still fixes one recorded derivation provenance per run.
