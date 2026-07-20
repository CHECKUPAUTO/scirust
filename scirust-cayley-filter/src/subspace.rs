//! Ordinary orthogonal projection outside a known real subspace.

use core::fmt;

use crate::filter::FilterEvaluation;
use crate::scalar::{SEDENION_DIMENSION, Sedenion, squared_norm};

/// Error returned while constructing a real subspace projector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubspaceProjectionError {
    EmptyBasis,
    InvalidTolerance,
    NonFiniteVector { index: usize },
    DependentVector { index: usize },
}

impl fmt::Display for SubspaceProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyBasis => formatter.write_str("subspace basis must not be empty"),
            Self::InvalidTolerance =>
            {
                formatter.write_str("tolerance must be finite and strictly positive")
            },
            Self::NonFiniteVector { index } =>
            {
                write!(formatter, "basis vector {index} is non-finite")
            },
            Self::DependentVector { index } =>
            {
                write!(formatter, "basis vector {index} is numerically dependent")
            },
        }
    }
}

impl std::error::Error for SubspaceProjectionError {}

/// Orthogonal projector removing every component contained in a known
/// real subspace.
///
/// The supplied basis is deterministically orthonormalized using two-pass
/// modified Gram-Schmidt.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseSubspaceProjector {
    orthonormal_basis: Vec<Sedenion>,
    tolerance: f64,
}

impl NoiseSubspaceProjector {
    pub fn new(basis: &[Sedenion], tolerance: f64) -> Result<Self, SubspaceProjectionError> {
        if basis.is_empty()
        {
            return Err(SubspaceProjectionError::EmptyBasis);
        }

        if !tolerance.is_finite() || tolerance <= 0.0
        {
            return Err(SubspaceProjectionError::InvalidTolerance);
        }

        let mut orthonormal_basis = Vec::with_capacity(basis.len());

        for (index, vector) in basis.iter().enumerate()
        {
            if vector.iter().any(|value| !value.is_finite())
            {
                return Err(SubspaceProjectionError::NonFiniteVector { index });
            }

            let mut residual = *vector;

            // Reorthogonalization improves numerical stability while
            // retaining a fixed deterministic accumulation order.
            for _ in 0..2
            {
                for direction in &orthonormal_basis
                {
                    let coefficient = dot(direction, &residual);

                    for coordinate in 0..SEDENION_DIMENSION
                    {
                        residual[coordinate] -= coefficient * direction[coordinate];
                    }
                }
            }

            let norm = squared_norm(&residual).sqrt();

            if !norm.is_finite()
            {
                return Err(SubspaceProjectionError::NonFiniteVector { index });
            }

            if norm <= tolerance
            {
                return Err(SubspaceProjectionError::DependentVector { index });
            }

            let inverse_norm = 1.0 / norm;

            for value in &mut residual
            {
                *value *= inverse_norm;
            }

            orthonormal_basis.push(residual);
        }

        Ok(Self {
            orthonormal_basis,
            tolerance,
        })
    }

    /// Removes the component of `input` lying in the configured subspace.
    #[must_use]
    pub fn apply(&self, input: &Sedenion) -> Sedenion {
        let mut output = *input;

        for direction in &self.orthonormal_basis
        {
            let coefficient = dot(direction, input);

            for coordinate in 0..SEDENION_DIMENSION
            {
                output[coordinate] -= coefficient * direction[coordinate];
            }
        }

        output
    }

    #[must_use]
    pub fn evaluate(&self, signal: &Sedenion, noise: &Sedenion) -> FilterEvaluation {
        FilterEvaluation::from_linear_outputs(signal, noise, self.apply(signal), self.apply(noise))
    }

    #[must_use]
    pub fn orthonormal_basis(&self) -> &[Sedenion] {
        &self.orthonormal_basis
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.orthonormal_basis.len()
    }

    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

fn dot(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (a, b)| sum + a * b)
}

#[cfg(test)]
mod tests {
    use super::{NoiseSubspaceProjector, SubspaceProjectionError};
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector, squared_norm};

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
    const TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn configured_subspace_is_removed() {
        let first = basis_vector(1).expect("e1 exists");
        let second = basis_vector(3).expect("e3 exists");

        let projector = NoiseSubspaceProjector::new(&[first, second], TOLERANCE)
            .expect("valid independent basis");

        let mut input = ZERO;
        input[1] = 1.0;
        input[3] = -2.0;
        input[5] = 4.0;

        assert_eq!(
            projector.apply(&input),
            core::array::from_fn(|index| { if index == 5 { 4.0 } else { 0.0 } })
        );
    }

    #[test]
    fn orthogonal_signal_is_preserved() {
        let basis = [
            basis_vector(1).expect("e1 exists"),
            basis_vector(3).expect("e3 exists"),
        ];

        let projector =
            NoiseSubspaceProjector::new(&basis, TOLERANCE).expect("valid independent basis");

        let signal = basis_vector(7).expect("e7 exists");

        assert_eq!(projector.apply(&signal), signal);
    }

    #[test]
    fn projection_is_idempotent() {
        let mut first = ZERO;
        first[1] = 1.0;
        first[2] = 1.0;

        let mut second = ZERO;
        second[4] = 1.0;
        second[5] = -1.0;

        let projector = NoiseSubspaceProjector::new(&[first, second], TOLERANCE)
            .expect("valid independent basis");

        let input: Sedenion = core::array::from_fn(|index| index as f64 - 7.0);

        let once = projector.apply(&input);
        let twice = projector.apply(&once);

        let residual: Sedenion = core::array::from_fn(|index| once[index] - twice[index]);

        assert!(squared_norm(&residual) < 1.0e-24);
    }

    #[test]
    fn dependent_basis_is_rejected() {
        let first = basis_vector(2).expect("e2 exists");
        let second = core::array::from_fn(|index| 2.0 * first[index]);

        assert_eq!(
            NoiseSubspaceProjector::new(&[first, second], TOLERANCE,),
            Err(SubspaceProjectionError::DependentVector { index: 1 })
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert_eq!(
            NoiseSubspaceProjector::new(&[], TOLERANCE),
            Err(SubspaceProjectionError::EmptyBasis)
        );

        let basis = [basis_vector(1).expect("e1 exists")];

        assert_eq!(
            NoiseSubspaceProjector::new(&basis, 0.0),
            Err(SubspaceProjectionError::InvalidTolerance)
        );

        let mut non_finite = ZERO;
        non_finite[4] = f64::NAN;

        assert_eq!(
            NoiseSubspaceProjector::new(&[non_finite], TOLERANCE,),
            Err(SubspaceProjectionError::NonFiniteVector { index: 0 })
        );
    }
}
