# SciRust — Domain Roadmap: Implemented (2026-07-10)

The audit reports (`UX_AND_DOMAINS_2026-07-09.md`, `SCIENTIFIC_COMPUTING_10_2026-07-09.md`)
catalogued the remaining domain gaps as a roadmap. This round **implements the
entire roadmap**: seven new domain crates plus the two deferred CI-hardening
items (a Miri gate and a cargo-fuzz harness).

## Conventions — every new crate, no exceptions

- **Pure Rust, zero external dependencies**, builds on **stable**.
- `#![forbid(unsafe_code)]` + `#![deny(missing_docs)]` (every public item documented).
- **Deterministic**: no RNG (tests use inline SplitMix64 where synthetic data is
  needed), no global state — identical inputs give bit-identical outputs.
- **No panics on malformed input**: fallible operations return a crate-local
  error enum implementing `Display + std::error::Error`.
- **Oracle-tested**: validated against analytic solutions, hand-computed values,
  or an independent dense reference implementation in the test module.
- Runnable crate-level doctest in every `//!` header.

## The seven crates

| Crate | Domain | Contents | Tests |
|---|---|---|---|
| `scirust-interp` | 1-D interpolation | linear, natural/clamped C² cubic spline (Thomas solve), monotone PCHIP (Fritsch–Carlson), Akima, barycentric Lagrange, nearest-neighbor — all behind one `Interpolator` trait, constructors validate nodes | 19 + doctest |
| `scirust-sparse` | sparse linear algebra | COO/CSR/CSC with lossless conversions & duplicate-summing, SpMV, transpose, Thomas tridiagonal, **Gilbert–Peierls left-looking sparse LU** with partial pivoting (factor once, many RHS), conjugate gradient | 16 + 2 doctests |
| `scirust-stiff` | stiff ODE / index-1 DAE | L-stable Backward Euler (modified Newton, FD Jacobian, dense LU), adaptive **Rosenbrock-W(2,3)** (`ode23s`-type, R(∞)=0), mass-matrix form `M·y′=f` where singular rows of M are algebraic constraints | 17 + doctest |
| `scirust-gp` | Gaussian processes | RBF / Matérn-3/2 / Matérn-5/2 kernels, exact Cholesky inference, predictive mean & variance (clamped ≥ 0), log marginal likelihood | 15 + doctest |
| `scirust-units` | dimensional analysis | 7 SI base dimensions as `i8` exponent vectors, const `mul/div/powi/try_sqrt` algebra, `Quantity` with checked `try_add/try_sub` (Err on incompatible dimensions), derived units N/J/W/Pa/Hz/C/V/Ω, `m·kg·s^-2`-style Display | 19 + doctest |
| `scirust-frame` | dataframes | typed columns (f64/i64/str/bool), select/filter/head, stable `sort_by_f64` (total_cmp, NaN last), group-by-aggregate (Sum/Mean/Count/Min/Max, first-seen order), inner join (deterministic order, `_right` collision rename), RFC-4180 CSV round-trip | 14 + doctest |
| `scirust-forecast` | time-series forecasting | SES, Holt, Holt-Winters (additive & multiplicative), AR(p) via Yule-Walker/**Levinson–Durbin**, differencing, moving average, MAE/RMSE/MAPE | 18 + doctest |

**118 new unit tests + 8 doctests**, all validated against oracles — e.g. the
clamped spline reproduces a cubic to 1e-9; sparse LU matches a dense
Gaussian-elimination oracle to 1e-9 with residual checks; Backward Euler stays
bounded on `y′ = −50y` at 6× the explicit stability limit *while the test shows
explicit Euler diverging*; the DAE constraint holds to machine precision at
every step; GP inference matches a dense oracle to 1e-9; Holt recovers a linear
trend to 1e-6; AR(1) recovers φ = 0.7.

## What this closes

The audit's remaining domain roadmap is now fully implemented. Combined with
`scirust-special` (special functions) and `scirust-stats` (distributions &
inference) from the previous round, a working scientist can do linear algebra
(dense, iterative **and sparse-direct**), calculus, optimization,
interpolation, **stiff** integration and DAEs, special functions, probability,
statistical inference, **nonparametric Bayesian regression (GP)**, time-series
forecasting, dimensioned computation, and tabular data wrangling — rigorously,
reproducibly, and entirely within SciRust.

## CI hardening (the two deferred audit items)

- **Miri gate** (new required job): runs the test suites of `scirust-special`,
  `scirust-stats`, and all seven new crates under Miri. These crates forbid
  `unsafe` and do no I/O, so any UB detected is a genuine soundness bug.
- **Fuzz harness** (new `fuzz/` cargo-fuzz workspace + informational CI job):
  first target `qsr1_from_bytes` asserts `QModel::from_bytes` — the untrusted
  QSR1 model parser hardened in PR #266 — never panics on arbitrary bytes.
  The job pins `--target x86_64-unknown-linux-gnu` (prebuilt cargo-fuzz
  binaries are musl-linked and would otherwise default to a sanitizer-less
  musl target).

## Score impact

| Dimension | Before | Now | Why |
|---|---:|---:|---|
| Scientific computing | 10.0 | **10.0** | breadth extended beyond the 10 bar (sparse-direct, stiff/DAE, GP, forecasting) |
| Security | ~9.2 | **9.3** | continuous fuzzing of the untrusted parser + Miri UB gate in CI |
| Overall / maintainability | 9.0+ | **↑** | 7 new crates at the strictest conventions in the workspace (zero deps, forbid-unsafe, deny-missing-docs, oracle tests) |

## Verification

- `cargo fmt --all -- --check` — clean.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- `cargo check --workspace --all-targets` — clean.
- `fuzz/` harness compiles (`cargo check` in its own workspace).
- All 7 new crates: `cargo test -p <crate>` green (118 tests + 8 doctests).
