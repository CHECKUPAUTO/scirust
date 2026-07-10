//! # `scirust-stats` — probability distributions & statistical inference
//!
//! A unified [`Distribution`] trait and a suite of classical hypothesis tests,
//! built entirely on the audited [`scirust-special`] numeric base. This is the
//! rigorous statistics layer a scientific/industrial platform needs and that the
//! SPC, tolerancing, metrology, reliability and predictive-maintenance crates
//! previously approximated piecemeal.
//!
//! ## What's here
//!
//! - **Continuous distributions** ([`dist`]): `Normal`, `StudentT`, `ChiSquared`,
//!   `FisherF`, `Gamma`, `Beta`, `Exponential`, `Uniform` — each with pdf, cdf,
//!   survival function, quantile, moments, and deterministic inverse-CDF sampling.
//! - **Discrete distributions** ([`discrete`]): `Binomial`, `Poisson` (both via
//!   Loader's saddle-point pmf), `Hypergeometric`, `Geometric`,
//!   `NegativeBinomial`, `BetaBinomial`, `Zipfian`, `Zeta` (via `riemann_zeta`),
//!   `PoissonBinomial`, `YuleSimon`, `Boltzmann`, `Logarithmic`, `Planck`,
//!   `Skellam` on ℤ, and the vector-valued `Multinomial`,
//!   `MultivariateHypergeometric` and `DirichletMultinomial` — pmf/ln-pmf,
//!   cdf, direct survival function, `logcdf`/`logsf`/`isf`, `interval`,
//!   `expect` (SciPy parity), quantile, moments, deterministic sampling.
//! - **Exact combinatorics** ([`comb`]): `factorial`, `binomial`, `permutations`,
//!   `multichoose` in checked `u128` (`None` on overflow, never a wrong number),
//!   plus overflow-free `ln_factorial` / `ln_binomial`.
//! - **Descriptive statistics** ([`describe`]): mean, unbiased variance / std,
//!   standard error, quantiles, median, min/max.
//! - **Hypothesis tests** ([`htest`]): one- and two-sample t-tests (pooled &
//!   Welch), one-way ANOVA, Pearson χ² goodness-of-fit, one-sample
//!   Kolmogorov–Smirnov.
//! - **Honest lottery mathematics** ([`lottery`]): exact odds of any
//!   `k`-of-`n` (+ bonus) game via the hypergeometric law, ticket expected
//!   value, and a χ² draw-fairness audit. Draws are independent and uniform:
//!   there is deliberately no "prediction" here, because none is possible.
//!
//! ## Guarantees
//!
//! - **Pure Rust, one internal dependency (`scirust-special`), `#![forbid(unsafe_code)]`.**
//! - **Deterministic**: sampling uses a seeded `SplitMix64`; no global state, no
//!   platform-dependent paths — same seed ⇒ bit-identical results.
//! - **Validated**: distributions and tests are checked against published
//!   reference values (z = 1.96 ⇒ 0.975, t₀.₉₇₅,₁₀ = 2.2281, χ²₀.₉₅,₅ = 11.0705,
//!   F₀.₉₅(5,10) = 3.3258, and hand-computed test statistics); discrete laws
//!   against SciPy 1.17 and exact rational arithmetic; lottery odds against
//!   the officially published Powerball / EuroMillions / FDJ tables.
//!
//! ## Example
//!
//! ```
//! use scirust_stats::prelude::*;
//!
//! // Is this sample's mean different from 10?
//! let data = [10.4, 9.8, 10.9, 10.1, 10.6, 9.7, 10.3];
//! let r = t_test_one_sample(&data, 10.0, Tail::TwoSided).unwrap();
//! assert!(r.p_value > 0.0 && r.p_value <= 1.0);
//!
//! // The 97.5th percentile of the standard normal is ≈ 1.96.
//! let z = Normal::standard().quantile(0.975);
//! assert!((z - 1.959_963_984).abs() < 1e-6);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod comb;
pub mod describe;
pub mod discrete;
pub mod dist;
pub mod htest;
pub mod lottery;
pub mod rng;

pub use discrete::{
    BetaBinomial, Binomial, Boltzmann, DirichletMultinomial, DiscreteDistribution, Geometric,
    Hypergeometric, Logarithmic, Multinomial, MultivariateHypergeometric, NegativeBinomial, Planck,
    Poisson, PoissonBinomial, Skellam, YuleSimon, Zeta, Zipfian,
};
pub use dist::{
    Beta, ChiSquared, Distribution, Exponential, FisherF, Gamma, Normal, StudentT, Uniform,
};
pub use htest::{
    Tail, TestResult, chi_square_gof, ks_test_one_sample, one_way_anova, t_test_one_sample,
    t_test_two_sample,
};
pub use lottery::{LotteryGame, PrizeTier, draw_frequency_chi_square};
pub use rng::SplitMix64;

/// One-import surface for the common statistics workflow.
pub mod prelude {
    pub use crate::comb::{
        binomial, factorial, ln_binomial, ln_factorial, multichoose, permutations,
    };
    pub use crate::describe::{mean, median, quantile, std_dev, std_error, variance};
    pub use crate::discrete::{
        BetaBinomial, Binomial, Boltzmann, DirichletMultinomial, DiscreteDistribution, Geometric,
        Hypergeometric, Logarithmic, Multinomial, MultivariateHypergeometric, NegativeBinomial,
        Planck, Poisson, PoissonBinomial, Skellam, YuleSimon, Zeta, Zipfian,
    };
    pub use crate::dist::{
        Beta, ChiSquared, Distribution, Exponential, FisherF, Gamma, Normal, StudentT, Uniform,
    };
    pub use crate::htest::{
        Tail, TestResult, chi_square_gof, ks_test_one_sample, one_way_anova, t_test_one_sample,
        t_test_two_sample,
    };
    pub use crate::lottery::{LotteryGame, PrizeTier, draw_frequency_chi_square};
    pub use crate::rng::SplitMix64;
}
