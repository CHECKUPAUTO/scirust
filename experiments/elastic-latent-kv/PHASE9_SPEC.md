# Elastic Latent KV — Phase 9 Adaptive Policy

## Objective

Phase 9 turns the fixed Phase 8 configuration into a deterministic planning
problem under a strict persistent-memory budget. It selects independent key and
value latent ranks, sparse residual slot counts, coefficient formats, and
residual formats.

## Planning inputs

The policy receives cumulative quality telemetry in basis points:

- quality retained by each key rank;
- quality retained by each value rank;
- incremental quality recovered by each key residual slot count;
- incremental quality recovered by each value residual slot count.

No heap allocation occurs inside plan enumeration. The quality arrays are
borrowed from caller-owned calibration state.

## Candidate space

For each key and value channel the planner enumerates:

- rank in `[minimum_rank, maximum_rank]`;
- residual slots in `[0, maximum_residual_slots]`;
- coefficient format in `{INT4, INT8, FP32}`;
- residual format in `{INT4, INT8, FP32}`.

Persistent bytes include basis storage, coefficient payloads, quantization
scales, residual `u16` indices, residual payloads, and residual scales.

Candidates exceeding `budget_bytes` are rejected before ranking.

## Deterministic objective

Candidates are ordered by:

1. highest worst key/value quality;
2. highest combined key/value quality;
3. lowest persistent bytes;
4. lowest total rank plus residual slots;
5. stable lexicographic rank/slot/format ordering.

The selected plan carries a stable FNV-1a fingerprint.

## Hysteresis

`AdaptiveKvPlanner` keeps one active plan and one pending plan. A different plan
must be observed for a configured number of consecutive confirmations before it
becomes active. Changes that neither meet the minimum quality gain nor reduce
bytes are ignored.

This prevents configuration oscillation while preserving deterministic behavior.

## Runtime integration

`AdaptiveResidualLatentBackend` materializes a selected plan as the validated
Phase 8 `ResidualLatentQuantizedBackend`. Full caller-supplied orthonormal bases
are deterministically truncated to the selected ranks during construction.

The hot decode path therefore remains the Phase 8 reconstruction-free attention
path; Phase 9 only changes construction/reconfiguration decisions.

## Explicit limits

Phase 9 does not yet update latent bases online, migrate already-resident tokens
between basis versions, evict tokens, or use device-specific kernels. Those are
Phases 10–13.
