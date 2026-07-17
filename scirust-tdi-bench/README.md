# scirust-tdi-bench

Reproducible benchmarks for the prospective **dynamic-information** (TDI)
hypothesis, driving [`scirust-tdi`](../scirust-tdi). Ported from the TDI
research project's `tdi-bench`; it packages the deterministic experimental
methodology behind the hypothesis:

- deterministic **scans** and **counterexample** search over exact finite-state
  and branching systems;
- untouched **holdout** evaluations (train/holdout split fixed up front);
- **ridge** regression models and a **deterministic bootstrap** for confidence
  intervals;
- frozen **preregistration** constants (target horizons, feature layouts,
  population contract) checked by in-code integrity tests.

Everything is exact and deterministic — same inputs give the same numbers,
bit-for-bit — and there is no `unsafe` (`unsafe_code = "forbid"`).

## Binaries

Ten reproducibility executables carry over verbatim (run with
`cargo run -p scirust-tdi-bench --bin <name>`):

`tdi-eval`, `tdi-scan`, `tdi-holdout`, `tdi-target-geometry`,
`tdi-branching-scan`, `tdi-branching-holdout`, `tdi-branching-continuous`,
`tdi-interwidth-continuous`, `tdi-continuous-deficit-geometry`,
`tdi-continuous-deficit-geometry-v51`.

## Validation

- 69 in-code tests pass; `cargo fmt --check` and
  `cargo clippy --all-targets -- -D warnings` are clean.
- One repository-layout-bound integrity test
  (`frozen_tdi5_protocol_hashes_are_unchanged`) is `#[ignore]`d: it hashes the
  TDI project's own preregistration files (its CI workflow, `docs/TDI-5-*`,
  `scripts/reproduce-tdi5.sh`), which are not part of the SciRust workspace. It
  is kept, documented, rather than deleted, so the frozen preregistration record
  stays visible.

## Provenance & scope

A faithful re-homing of TDI's `tdi-bench`. As the reference project reports
honestly, within the tested synthetic families the signal challenges
entropy-only sufficiency, but TDI-1 was fully subsumed by the orbital baseline
and no universal law or cross-size invariance is claimed. What transfers is the
deterministic, exact evaluation machinery — holdouts, ridge, deterministic
bootstrap CIs and preregistration discipline.

> Note: the upstream TDI project has not yet selected a license.
