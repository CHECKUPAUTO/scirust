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
                    value += contravariant[rho][sigma]
                        * (derivatives[mu][sigma][nu] + derivatives[nu][sigma][mu]
                            - derivatives[sigma][mu][nu]);
                }

                symbols[rho][mu][nu] = 0.5 * value;
            }
        }
    }

    Ok(symbols)
}
