---
status: accepted
---

# Restrict VRS use to research

Note (2026-08-06): VRS is documentation-only under ADR-0016 — nothing uses it. This ADR is retained as the licensing record should VRS ever be reconsidered.

The RobotSeg repository is Apache-2.0, but VRS has no separately published dataset license and contains derivatives of ten upstream robot datasets. VRS inputs, derived training data, and checkpoints will therefore be used internally for research until every upstream license and redistribution term has been audited.

## Consequences

The planned tooling will not download or redistribute VRS automatically. Any public release of derived data or checkpoints requires a separate licensing review.
