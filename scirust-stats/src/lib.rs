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
//! - **Distributions** ([`dist`]): `Normal`, `StudentT`, `ChiSquared`, `FisherF`,
//!   `Gamma`, `Beta`, `Exponential`, `Uniform` — each with pdf, cdf, survival
//!   function, quantile, moments, and deterministic inverse-CDF sampling.
//! - **Descriptive statistics** ([`describe`]): mean, unbiased variance / std,
//!   standard error, quantiles, median, min/max.
//! - **Hypothesis tests** ([`htest`]): one- and two-sample t-tests (pooled &
//!   Welch), one-way ANOVA, Pearson χ² goodness-of-fit, one-sample
//!   Kolmogorov–Smirnov.
//!
//! ## Guarantees
//!
//! - **Pure Rust, one internal dependency (`scirust-special`), `#![forbid(unsafe_code)]`.**
//! - **Deterministic**: sampling uses a seeded `SplitMix64`; no global state, no
//!   platform-dependent paths — same seed ⇒ bit-identical results.
//! - **Validated**: distributions and tests are checked against published
//!   reference values (z = 1.96 ⇒ 0.975, t₀.₉₇₅,₁₀ = 2.2281, χ²₀.₉₅,₅ = 11.0705,
//!   F₀.₉₅(5,10) = 3.3258, and hand-computed test statistics).
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

pub mod describe;
pub mod dist;
pub mod htest;
pub mod rng;

pub use dist::{
    Beta, ChiSquared, Distribution, Exponential, FisherF, Gamma, Normal, StudentT, Uniform,
};
pub use htest::{
    Tail, TestResult, chi_square_gof, ks_test_one_sample, one_way_anova, t_test_one_sample,
    t_test_two_sample,
};
pub use rng::SplitMix64;

/// One-import surface for the common statistics workflow.
pub mod prelude {
    pub use crate::describe::{mean, median, quantile, std_dev, std_error, variance};
    pub use crate::dist::{
        Beta, ChiSquared, Distribution, Exponential, FisherF, Gamma, Normal, StudentT, Uniform,
    };
    pub use crate::htest::{
        Tail, TestResult, chi_square_gof, ks_test_one_sample, one_way_anova, t_test_one_sample,
        t_test_two_sample,
    };
    pub use crate::rng::SplitMix64;
}
