//! Grünwald–Letnikov discretization.

use crate::validation::{validate_samples, validate_step};
use crate::{FractionalError, FractionalOrder};

/// Generate the first `len` Grünwald–Letnikov coefficients
///
/// `w_k = (-1)^k binomial(alpha, k)`.
///
/// The coefficients are generated recursively:
///
/// `w_0 = 1` and
/// `w_k = w_(k-1) * (1 - (alpha + 1) / k)`.
#[must_use]
pub fn grunwald_letnikov_weights(order: FractionalOrder, len: usize) -> Vec<f64> {
    if len == 0
    {
        return Vec::new();
    }

    let alpha = order.value();
    let mut weights = Vec::with_capacity(len);
    weights.push(1.0);

    for k in 1..len
    {
        let previous = weights[k - 1];
        let k_float = k as f64;
        weights.push(previous * (1.0 - (alpha + 1.0) / k_float));
    }

    weights
}

/// Approximate the left-sided Riemann–Liouville derivative at the final
/// sample using the Grünwald–Letnikov formula on a uniform grid.
///
/// For samples `f_0, ..., f_n` separated by `step`, this evaluates
///
/// `step^(-alpha) * sum(k=0..n, w_k * f_(n-k))`.
pub fn riemann_liouville_gl_uniform(
    samples: &[f64],
    step: f64,
    order: FractionalOrder,
) -> Result<f64, FractionalError> {
    validate_samples(samples)?;
    validate_step(step)?;

    let weights = grunwald_letnikov_weights(order, samples.len());
    let last = samples.len() - 1;

    let mut sum = 0.0;
    for (k, weight) in weights.iter().enumerate()
    {
        sum += weight * samples[last - k];
    }

    Ok(sum / step.powf(order.value()))
}
