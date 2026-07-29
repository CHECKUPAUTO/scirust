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

A faithful re-homing of TDI's `tdi-bench`. What transfers is the deterministic,
exact evaluation machinery — holdouts, ridge, deterministic bootstrap CIs and
preregistration discipline.

**This port is a snapshot, and the snapshot is old.** The ten binaries above
carry the upstream evaluators through `tdi-continuous-deficit-geometry-v51`,
i.e. **TDI-5.1**. Upstream has since built, frozen and merged thirteen further
evaluators (TDI-5.2 → 5.9 and TDI-6.1 → 6.5), of which seven have produced real
confirmatory results. None of that is present here. In particular, the headline
findings that now characterise the hypothesis — survival against exact spectral
moments (TDI-5.6), replication across four generator families (TDI-5.7),
survival against the *literal* spectral gap and mixing time (TDI-6.1), and
descriptor saturation (TDI-5.9) — were all obtained with evaluators that this
crate does not contain. See `scirust-tdi`'s README for the current result table
and the open questions.

TDI-1 was fully subsumed by the orbital baseline (incremental gain `0`); that
negative result stands. No universal law or cross-size invariance is claimed,
and cross-width invariance specifically (upstream TDI-5.8) has not yet been run.

> Note: the upstream TDI project has not yet selected a license.
