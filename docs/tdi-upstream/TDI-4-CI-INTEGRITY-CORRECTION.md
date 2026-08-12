# TDI-4 — CI integrity correction before evaluation

## Status

This correction occurred before any complete TDI-4 execution and before any
production of TDI-4 metric or verdict.

## Observed failure

The initial TDI-4 freeze, at commit:

`30916731078e9fe04f35d30b5388d04f9ed12d65`

failed in the CI job `Preregistration integrity`.

The immutable TDI-3 scientific manifest reported only:

`tdi-core/src/signature.rs: FAILED`

## Cause

The initial TDI-4 implementation had added a method to
`tdi-core/src/signature.rs`.

This file belongs to the TDI-3 scientific freeze. A subsequent experiment
must not modify a file covered by this historical manifest.

## Correction

`tdi-core/src/signature.rs` was restored exactly to its state frozen by
TDI-3.

The computation of the TDI-4 conditional target is now located in the TDI-4
evaluator:

- the exact numerator of the deficit is computed with `BigUint`;
- the logarithm of the numerator and of the denominator is determined from
  their binary length and their 53 leading significant bits;
- the target remains `U = -log₂(1 - O₆)`;
- no prior rounding of `O₆` toward `1.0` is used.

## Invariants

This correction does not modify:

- the preregistration;
- the populations and their seeds;
- the horizons;
- the explanatory variables;
- the models;
- `lambda`;
- the bootstrap;
- the TDI-4A and TDI-4B criteria.

No TDI-4 result was observed before this correction.
