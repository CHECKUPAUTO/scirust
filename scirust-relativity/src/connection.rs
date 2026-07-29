//! Affine connections and Christoffel symbols.

use crate::{Metric, RelativityError, invert_metric};

/// A provider of Christoffel symbols `Gamma^rho_(mu nu)`.
pub trait Connection<const D: usize> {
    /// Evaluate all Christoffel symbols at `coordinates`.
    ///
    /// The returned indexing order is `[rho][mu][nu]`.
    fn christoffel(&self, coordinates: &[f64; D]) -> [[[f64; D]; D]; D];
}

fn validate_coordinates<const D: usize>(coordinates: &[f64; D]) -> Result<(), RelativityError> {
    if let Some((index, _)) = coordinates
        .iter()
        .enumerate()
        .find(|(_, coordinate)| !coordinate.is_finite())
    {
        return Err(RelativityError::NonFiniteCoordinate(index));
    }

    Ok(())
}

/// Numerically evaluate the Levi-Civita Christoffel symbols of a metric.
///
/// Central finite differences approximate the first derivatives of the metric:
///
/// `Gamma^rho_(mu nu) = 1/2 g^(rho sigma)
/// (partial_mu g_(sigma nu) + partial_nu g_(sigma mu)
/// - partial_sigma g_(mu nu))`.
pub fn numerical_christoffel<M, const D: usize>(
    metric: &M,
    coordinates: &[f64; D],
    difference_step: f64,
) -> Result<[[[f64; D]; D]; D], RelativityError>
where
    M: Metric<D>,
{
    validate_coordinates(coordinates)?;

    if !difference_step.is_finite() || difference_step <= 0.0
    {
        return Err(RelativityError::InvalidDifferenceStep(difference_step));
    }

    let covariant = metric.components(coordinates);
    let contravariant = invert_metric(&covariant)?;

    let mut derivatives = [[[0.0_f64; D]; D]; D];

    for direction in 0..D
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;

        plus[direction] += difference_step;
        minus[direction] -= difference_step;

        let metric_plus = metric.components(&plus);
        let metric_minus = metric.components(&minus);

        for row in 0..D
        {
            for column in 0..D
            {
                let plus_value = metric_plus[row][column];
                let minus_value = metric_minus[row][column];

                if !plus_value.is_finite()
                {
                    return Err(RelativityError::NonFiniteMetricComponent { row, column });
                }

                if !minus_value.is_finite()
                {
                    return Err(RelativityError::NonFiniteMetricComponent { row, column });
                }

                derivatives[direction][row][column] =
                    (plus_value - minus_value) / (2.0 * difference_step);
            }
        }
    }

    Ok(christoffel_from_derivatives(&contravariant, &derivatives))
}

/// The Levi-Civita Christoffel symbols from an **already-known** inverse metric
/// and metric gradient:
///
/// ```text
/// Gamma^rho_(mu nu) = 1/2 g^(rho sigma)
///     ( d_mu g_(sigma nu) + d_nu g_(sigma mu) - d_sigma g_(mu nu) )
/// ```
///
/// `derivatives[k][i][j]` is `d_k g_ij`.
///
/// This is the algebraic core of [`numerical_christoffel`], which obtains the
/// gradient by central differences and then calls this. Exposing it lets a
/// caller that already has the gradient — a grid, for instance, which can
/// difference its own stored arrays with a compact stencil — reuse the identical
/// algebra instead of restating it. There is exactly one implementation of this
/// formula in the crate.
///
/// Purely algebraic: no differencing, no validation, and no failure mode. A
/// non-finite input produces a non-finite output rather than an error, so
/// callers that need validation should validate their inputs.
#[must_use]
pub fn christoffel_from_derivatives<const D: usize>(
    inverse_metric: &[[f64; D]; D],
    derivatives: &[[[f64; D]; D]; D],
) -> [[[f64; D]; D]; D] {
    let mut symbols = [[[0.0_f64; D]; D]; D];

    for rho in 0..D
    {
        for mu in 0..D
        {
            for nu in 0..D
            {
                let mut value = 0.0;

                for sigma in 0..D
                {
                    value += inverse_metric[rho][sigma]
                        * (derivatives[mu][sigma][nu] + derivatives[nu][sigma][mu]
                            - derivatives[sigma][mu][nu]);
                }

                symbols[rho][mu][nu] = 0.5 * value;
            }
        }
    }

    symbols
}
