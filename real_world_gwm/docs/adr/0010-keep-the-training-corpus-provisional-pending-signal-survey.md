---
status: accepted
---

# Keep the training corpus provisional pending a signal survey

VRS remains an immediately usable candidate, but phase one will not bind the GWM architecture or experiment identity to it before surveying other robot-video corpora. A larger, cleaner, more diverse, or more directly renderable corpus may replace or supplement VRS when it provides better aligned full RGB and robot-only RGB supervision. Real sources, simulated sources such as MolmoBot, and documented real/simulation mixtures are all eligible; the invariant is that pretraining remains outside the WISER data domain and uses no WISER samples.

## Consequences

Dataset discovery and a documented signal-quality comparison precede final corpus selection. Catalogs such as `datasets.bot`, state-rich Franka datasets, and MolmoBot are leads rather than pre-approved inputs. Each selected source still requires its own adapter, audit, licensing review, exact-path visualization, and explicit mixture policy if more than one source is used.
