//! Caputo derivative using the uniform-grid L1 scheme.

use crate::validation::{validate_samples, validate_step};
use crate::{FractionalError, FractionalOrder};
use scirust_special::gamma;

/// Approximate the left-sided Caputo derivative at the final sample with the
/// classical L1 scheme on a uniform temporal grid.
///
/// For `0 < alpha < 1`, samples `f_0, ..., f_n`, and uniform spacing `step`,
/// the implementation evaluates
///
/// `1 / (Gamma(2-alpha) * step^alpha)`
///
/// multiplied by
///
/// `sum(k=0..n-1, ((k+1)^(1-alpha) - k^(1-alpha))
///                  * (f_(n-k) - f_(n-k-1)))`.
///
/// The full sample history is therefore significant: the operator is
/// non-local in time.
pub fn caputo_l1_uniform(
    samples: &[f64],
    step: f64,
    order: FractionalOrder,
) -> Result<f64, FractionalError> {
    validate_samples(samples)?;
    validate_step(step)?;

    if samples.len() < 2
    {
        return Err(FractionalError::TooFewSamples);
    }

    let alpha = order.value();
    let exponent = 1.0 - alpha;
    let last = samples.len() - 1;
    let mut sum = 0.0;

    for k in 0..last
    {
        let k_float = k as f64;
        let weight = (k_float + 1.0).powf(exponent) - k_float.powf(exponent);
        let difference = samples[last - k] - samples[last - k - 1];
        sum += weight * difference;
    }

    let normalization = gamma(2.0 - alpha) * step.powf(alpha);
    Ok(sum / normalization)
}
