# Real-World GWM

This folder is the documentation and future implementation home for adapting the Grounded World Model (GWM) to broader robot-video corpora for eventual real-world use. It currently contains design records only; dataset selection and training code will be handled in later work.

## Documentation map

- [CONTEXT.md](CONTEXT.md) defines the canonical domain language.
- [Phase-one plan](docs/phase-1-plan.md) records the current, adjustable experiment and implementation plan.
- [Retired VRS prototype review](docs/prior-dev-vrs-prototype.md) preserves the useful findings and rejected choices from the former `origin/dev` branch.
- [ADR index](docs/adr/README.md) links the accepted architectural decisions and their rationale.
- [References](docs/references.md) lists the primary papers, repositories, and relevant local code.

## Documentation policy

ADRs record decisions that are costly to reverse, surprising without context, and based on a real trade-off. Reversible defaults and operational details belong in the phase-one plan. When a decision changes, add or supersede an ADR instead of silently rewriting its history.

No dataset or model download is performed by this folder. No existing file under `gwm_wiser/` is changed by the planned adaptation.
