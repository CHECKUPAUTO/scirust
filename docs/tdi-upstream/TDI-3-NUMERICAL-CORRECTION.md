# TDI-3 — Numerical correction after freeze

Detection date: 12 July 2026.

## Affected execution

The first execution of the TDI-3 protocol was launched on the frozen commit:

`25b6146764b2ee87bbffda559dd9dd0559213360`

It stopped before producing metrics or verdicts.

## Observed error

The failure is reproducible with:

- width: 5;
- seed: `20_000_000`;
- horizon: 6;
- distribution: reference;
- reported state: bits `15`;
- error: `ProbabilityOverflow`.

Original message:

`ReferenceDistribution(ProbabilityOverflow { state: State { bits: 15, width: 5 }, depth: 6 })`

## Cause

`ExactRatio` used two `u128` integers.

The propagation remained rationally exact, but adding probabilities coming
from paths with different denominators could produce a reduced common
denominator exceeding the capacity of `u128`.

It is therefore not:

- a negative TDI-3 result;
- a modification of the data;
- a floating-point instability;
- an overflow coming from the statistical target.

It is a representation limit of the exact rational engine.

## Correction

`ExactRatio` now uses `BigUint`, an arbitrarily large integer
representation in pure Rust.

The following operations remain exact:

- reduction by GCD;
- rational addition;
- division by the local number of successors;
- rational comparison;
- computation of the distributional recovery.

Conversion to `f64` remains limited to display and statistical modeling,
in accordance with the preregistration.

## Non-alteration control

The correction does not modify:

- the populations;
- the seed ranges;
- the horizons 2 and 6;
- the perturbation;
- the features;
- the baseline;
- the ridge model;
- `lambda = 1`;
- the bootstrap;
- the TDI-3A and TDI-3B criteria.

A regression test verifies that width 5, seed `20_000_000`, can now be
analyzed without overflow. This test publishes neither target, nor
prediction, nor metric.

## Preserved evidence

The local evidence of the initial failure was preserved in:

`/tmp/tdi3-first-run-overflow`

Observed hashes:

- partial log:
  `28c43a599ba94afd4c74a1bed2f11e9050a7c4e774847e71e2c8c82e28e99893`;
- complete console:
  `00f0a9d6ce5102ecdf1630c60149c885bce3616a30cc3096024fc9b106465b42`.

## New freeze

After complete validation, a new hash of the evaluator and a SHA-256
manifest of all the scientific code will be produced before the new full
execution.
