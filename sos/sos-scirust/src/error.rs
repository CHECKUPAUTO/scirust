//! Computational-backend-adapter error type.

use thiserror::Error;

/// Errors produced by the `sos-scirust` adapters.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ScirustError {
    /// The underlying `scirust-gp` fit failed.
    #[error("Gaussian process fit failed: {0}")]
    Gp(#[from] scirust_gp::GpError),
    /// A declared observation noise variance was not finite and strictly
    /// positive. Every real sensor/measurement has some noise; a value of
    /// zero would claim infinite information from a single observation, and
    /// NaN/infinity are never a meaningful noise variance.
    #[error("observation noise variance must be finite and positive, got {0}")]
    NonPositiveNoise(f64),
    /// A hypothesis prior was empty, had a non-finite or negative entry, or
    /// summed to zero — none of which is a valid (even unnormalized) prior.
    #[error("hypothesis prior must be non-empty, finite, non-negative, and sum to more than 0")]
    InvalidPrior,
    /// Fewer than two Monte-Carlo samples were requested. A standard error
    /// needs at least two samples to mean anything.
    #[error("nested-MC EIG needs at least 2 samples, got {0}")]
    TooFewSamples(usize),
    /// The number of per-hypothesis likelihood models passed to
    /// [`crate::nmc::NestedMcEigEstimator::estimate`] did not match the
    /// number of hypotheses the estimator's prior spans.
    #[error("{models} likelihood models were given but the prior spans {hypotheses} hypotheses")]
    HypothesisCountMismatch {
        /// How many models were passed.
        models: usize,
        /// How many hypotheses the prior spans.
        hypotheses: usize,
    },
    /// A Bayes factor was requested with no observations. No data is no
    /// evidence — and the neutral `0` it would otherwise produce is a claim
    /// about the evidence, not the absence of one.
    #[error("a Bayes factor needs at least one observation")]
    NoObservations,
    /// The data is impossible under **both** hypotheses, so the ratio is
    /// undefined rather than extreme. Both failed; comparing them is
    /// meaningless.
    #[error("the data is impossible under both hypotheses; the Bayes factor is undefined")]
    UndefinedBayesFactor,
    /// A distribution returned a `NaN` log-likelihood for an observation.
    #[error("distribution returned a non-finite log-likelihood at k={0}")]
    NonFiniteLikelihood(u64),
}

/// Convenience alias for `sos-scirust` results.
pub type Result<T> = core::result::Result<T, ScirustError>;
