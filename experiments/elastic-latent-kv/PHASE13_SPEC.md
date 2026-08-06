# Elastic Latent KV — Phase 13 Integrated Runtime

Phase 13 provides a session-scoped production integration point for the Phase
9–12 components around the live numeric Transformer decode path.

At construction, each attention head receives a Phase 10 basis version and
calibration profile. Phase 9 selects a strict-budget plan, which is materialized
as the validated Phase 8 residual latent backend. Phase 12 kernels execute the
q/k/v/o linear projections. Phase 11 tracks resident logical positions and emits
HOT/WARM/COLD transition telemetry.

The runtime enforces two independent ceilings:

- a strict sum of Phase 9 persistent-storage budgets;
- a hard ceiling over the actual fixed allocations reported by all backends.

The runtime checks capacity before append, turning the Phase 8 capacity contract
into a typed error rather than allowing the backend adapter to panic.

## Session semantics

Basis versions and Phase 9 plans are frozen for one session. Online Phase 10
learning can prepare the next session or epoch, but Phase 13 intentionally does
not reinterpret already-resident coefficients under a new basis.

Phase 11 temperature transitions are surfaced as deterministic telemetry; the
current scalar Phase 8 payload is not rewritten in-place during a session. This
preserves numerical correctness while keeping the re-encoding contract explicit
for a later device-specific migration implementation.

## Validation

The integrated tests compare full-rank FP32 latent decode against the plain
`decode_step`, verify strict budget telemetry, exercise the Phase 12 projection
kernel, preserve basis-version lifecycle metadata, and verify typed capacity
exhaustion before any backend panic.
