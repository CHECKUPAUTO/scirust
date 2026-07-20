//! Continuous spectral Cayley filter used during optimization.

use core::fmt;

use scirust_solvers::Matrix;
use scirust_solvers::linalg::svd;

use crate::filter::FilterEvaluation;
use crate::operator::{Matrix16, left_multiplication_matrix, matrix_vector_mul};
use crate::scalar::{SEDENION_DIMENSION, Sedenion};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SoftFilterError {
    InvalidRelativeScale,
    Decomposition(String),
}

impl fmt::Display for SoftFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidRelativeScale => f.write_str("relative scale must be finite and positive"),
            Self::Decomposition(message) =>
            {
                write!(f, "SVD decomposition failed: {message}")
            },
        }
    }
}

impl std::error::Error for SoftFilterError {}

/// Continuous spectral attenuation derived from `L_a = U Σ Vᵀ`.
///
/// The retained gain of right-singular direction `i` is:
///
/// `g_i = σ_i² / (σ_i² + τ²)`
///
/// with `τ = relative_scale * σ_max`.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftCayleyFilter {
    multiplier: Sedenion,
    transform: Matrix16,
    singular_values: Vec<f64>,
    gains: Vec<f64>,
    relative_scale: f64,
}

impl SoftCayleyFilter {
    pub fn new(multiplier: Sedenion, relative_scale: f64) -> Result<Self, SoftFilterError> {
        if !relative_scale.is_finite() || relative_scale <= 0.0
        {
            return Err(SoftFilterError::InvalidRelativeScale);
        }

        let operator = left_multiplication_matrix(multiplier);
        let data = operator
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        let matrix = Matrix::from_row_major(SEDENION_DIMENSION, SEDENION_DIMENSION, data);

        let decomposition =
            svd(&matrix).map_err(|error| SoftFilterError::Decomposition(error.to_string()))?;

        let sigma_max = decomposition.s.first().copied().unwrap_or(0.0);
        let tau = relative_scale * sigma_max;
        let tau_squared = tau * tau;

        let gains: Vec<f64> = decomposition
            .s
            .iter()
            .map(|&sigma| {
                let sigma_squared = sigma * sigma;
                let denominator = sigma_squared + tau_squared;

                if denominator == 0.0
                {
                    0.0
                }
                else
                {
                    sigma_squared / denominator
                }
            })
            .collect();

        let transform = core::array::from_fn(|row| {
            core::array::from_fn(|column| {
                gains
                    .iter()
                    .enumerate()
                    .fold(0.0, |sum, (direction, &gain)| {
                        sum + gain
                            * decomposition.v[(row, direction)]
                            * decomposition.v[(column, direction)]
                    })
            })
        });

        Ok(Self {
            multiplier,
            transform,
            singular_values: decomposition.s,
            gains,
            relative_scale,
        })
    }

    #[must_use]
    pub fn apply(&self, input: &Sedenion) -> Sedenion {
        matrix_vector_mul(&self.transform, input)
    }

    #[must_use]
    pub fn evaluate(&self, signal: &Sedenion, noise: &Sedenion) -> FilterEvaluation {
        FilterEvaluation::from_linear_outputs(signal, noise, self.apply(signal), self.apply(noise))
    }

    #[must_use]
    pub const fn transform(&self) -> &Matrix16 {
        &self.transform
    }

    #[must_use]
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    #[must_use]
    pub fn gains(&self) -> &[f64] {
        &self.gains
    }

    #[must_use]
    pub const fn multiplier(&self) -> &Sedenion {
        &self.multiplier
    }

    #[must_use]
    pub const fn relative_scale(&self) -> f64 {
        self.relative_scale
    }
}

#[cfg(test)]
mod tests {
    use super::{SoftCayleyFilter, SoftFilterError};
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector, squared_norm};

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

    fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
        left.iter().zip(right).fold(0.0, |sum, (a, b)| {
            let difference = a - b;
            sum + difference * difference
        })
    }

    #[test]
    fn identity_multiplier_has_uniform_half_gain_at_unit_scale() {
        let filter =
            SoftCayleyFilter::new(basis_vector(0).expect("e0 exists"), 1.0).expect("SVD succeeds");

        assert!(
            filter
                .gains()
                .iter()
                .all(|gain| (*gain - 0.5).abs() < 1.0e-12)
        );

        let input = basis_vector(7).expect("e7 exists");
        let output = filter.apply(&input);

        assert!((output[7] - 0.5).abs() < 1.0e-12);
        assert!(
            output
                .iter()
                .enumerate()
                .all(|(i, value)| i == 7 || value.abs() < 1.0e-12)
        );
    }

    #[test]
    fn exact_kernel_noise_is_strongly_suppressed() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        let signal = basis_vector(0).expect("e0 exists");

        let filter = SoftCayleyFilter::new(multiplier, 1.0e-6).expect("SVD succeeds");

        let filtered_noise = filter.apply(&noise);
        let filtered_signal = filter.apply(&signal);

        assert!(squared_norm(&filtered_noise) < 1.0e-16);
        assert!(squared_distance(&filtered_signal, &signal) < 1.0e-16);
    }

    #[test]
    fn gains_are_bounded() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let filter = SoftCayleyFilter::new(multiplier, 0.1).expect("SVD succeeds");

        assert!(filter.gains().iter().all(|gain| (0.0..=1.0).contains(gain)));
    }

    #[test]
    fn invalid_scale_is_rejected() {
        let multiplier = basis_vector(0).expect("e0 exists");

        for scale in [0.0, -1.0, f64::INFINITY, f64::NAN]
        {
            assert_eq!(
                SoftCayleyFilter::new(multiplier, scale),
                Err(SoftFilterError::InvalidRelativeScale)
            );
        }
    }
}
