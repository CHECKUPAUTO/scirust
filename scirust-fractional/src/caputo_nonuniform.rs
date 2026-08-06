//! Caputo derivative using the L1 scheme on a non-uniform temporal grid.

use crate::validation::{validate_sample_times, validate_samples};
use crate::{FractionalError, FractionalOrder};
use scirust_special::gamma;

/// Approximate the left-sided Caputo derivative at the final sample with the
/// L1 scheme on a non-uniform temporal grid.
///
/// This is the same piecewise-constant-derivative construction as
/// [`crate::caputo_l1_uniform`], without assuming the samples are evenly
/// spaced: on each subinterval `[t_k, t_(k+1)]`, the true derivative is
/// approximated by the finite difference `(f_(k+1) - f_k) / (t_(k+1) - t_k)`,
/// and the Caputo weight kernel `(t_n - s)^(-alpha)` is integrated exactly
/// over that subinterval. For `0 < alpha < 1`, samples `f_0, ..., f_n` at
/// times `t_0 < t_1 < ... < t_n`, the implementation evaluates
///
/// `1 / Gamma(2 - alpha)`
///
/// multiplied by
///
/// `sum(k=0..n-1, ((t_n - t_k)^(1-alpha) - (t_n - t_(k+1))^(1-alpha))
///                  * (f_(k+1) - f_k) / (t_(k+1) - t_k))`.
///
/// When `t_k = k * h` for a uniform step `h`, this reduces to the same
/// mathematical quantity as [`crate::caputo_l1_uniform`] (verified
/// numerically, not merely asserted, in this crate's test suite); the two
/// functions use different summation orders and are not guaranteed
/// bit-identical on a uniform grid.
///
/// `samples` and `sample_times` must have equal, positive length of at least
/// two, `sample_times` must be strictly increasing, and both sequences must
/// be entirely finite.
pub fn caputo_l1_nonuniform(
    samples: &[f64],
    sample_times: &[f64],
    order: FractionalOrder,
) -> Result<f64, FractionalError> {
    validate_samples(samples)?;
    validate_sample_times(sample_times)?;

    if samples.len() != sample_times.len()
    {
        return Err(FractionalError::MismatchedLengths {
            samples: samples.len(),
            sample_times: sample_times.len(),
        });
    }

    if samples.len() < 2
    {
        return Err(FractionalError::TooFewSamples);
    }

    let alpha = order.value();
    let exponent = 1.0 - alpha;
    let last = samples.len() - 1;
    let final_time = sample_times[last];
    let mut sum = 0.0;

    for k in 0..last
    {
        let weight = (final_time - sample_times[k]).powf(exponent)
            - (final_time - sample_times[k + 1]).powf(exponent);
        let difference = (samples[k + 1] - samples[k]) / (sample_times[k + 1] - sample_times[k]);
        sum += weight * difference;
    }

    let normalization = gamma(2.0 - alpha);
    Ok(sum / normalization)
}
