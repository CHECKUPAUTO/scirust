//! Real-linear operators induced by left sedenion multiplication.

use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector, sedenion_mul};

/// Row-major real matrix representing a linear map on sedenions.
pub type Matrix16 = [[f64; SEDENION_DIMENSION]; SEDENION_DIMENSION];

/// Real-linear operator `L_a(x) = a · x`.
#[derive(Clone, Debug, PartialEq)]
pub struct LeftMultiplicationOperator {
    multiplier: Sedenion,
    matrix: Matrix16,
}

impl LeftMultiplicationOperator {
    /// Constructs the matrix of left multiplication by `multiplier`.
    #[must_use]
    pub fn new(multiplier: Sedenion) -> Self {
        let matrix = left_multiplication_matrix(multiplier);
        Self { multiplier, matrix }
    }

    /// Applies `L_a` using its real `16 × 16` matrix.
    #[must_use]
    pub fn apply(&self, input: &Sedenion) -> Sedenion {
        matrix_vector_mul(&self.matrix, input)
    }

    /// Returns the left multiplier `a`.
    #[must_use]
    pub const fn multiplier(&self) -> &Sedenion {
        &self.multiplier
    }

    /// Returns the row-major matrix of `L_a`.
    #[must_use]
    pub const fn matrix(&self) -> &Matrix16 {
        &self.matrix
    }
}

/// Builds the row-major matrix of `L_a(x) = a · x`.
///
/// Column `j` is the coordinate vector of `a · e_j`.
#[must_use]
pub fn left_multiplication_matrix(multiplier: Sedenion) -> Matrix16 {
    let columns: [Sedenion; SEDENION_DIMENSION] = core::array::from_fn(|column| {
        let basis = basis_vector(column).expect("basis index is valid");
        sedenion_mul(multiplier, basis)
    });

    core::array::from_fn(|row| core::array::from_fn(|column| columns[column][row]))
}

/// Multiplies a row-major `16 × 16` matrix by a sedenion vector.
///
/// Accumulation order is fixed from column `0` to column `15`.
#[must_use]
pub fn matrix_vector_mul(matrix: &Matrix16, input: &Sedenion) -> Sedenion {
    core::array::from_fn(|row| {
        matrix[row]
            .iter()
            .zip(input.iter())
            .fold(0.0, |sum, (coefficient, value)| sum + coefficient * value)
    })
}

#[cfg(test)]
mod tests {
    use super::{LeftMultiplicationOperator, left_multiplication_matrix, matrix_vector_mul};
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector, sedenion_mul, squared_norm};

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

    #[test]
    fn real_unit_produces_identity_matrix() {
        let one = basis_vector(0).expect("e0 exists");
        let matrix = left_multiplication_matrix(one);

        for (row_index, matrix_row) in matrix.iter().enumerate()
        {
            for (column_index, value) in matrix_row.iter().enumerate()
            {
                let expected = if row_index == column_index { 1.0 } else { 0.0 };
                assert_eq!(*value, expected);
            }
        }
    }

    #[test]
    fn every_matrix_column_is_the_image_of_its_basis_vector() {
        let multiplier = [
            1.0, -1.0, 2.0, 0.0, 3.0, -2.0, 1.0, 1.0, 0.0, 2.0, -1.0, 1.0, -3.0, 0.0, 2.0, -1.0,
        ];
        let matrix = left_multiplication_matrix(multiplier);

        for column in 0..SEDENION_DIMENSION
        {
            let basis = basis_vector(column).expect("basis element exists");
            let expected = sedenion_mul(multiplier, basis);
            assert_eq!(matrix_vector_mul(&matrix, &basis), expected);
        }
    }

    #[test]
    fn matrix_application_matches_direct_sedenion_product() {
        let multiplier = [
            1.0, -1.0, 2.0, 0.0, 3.0, -2.0, 1.0, 1.0, 0.0, 2.0, -1.0, 1.0, -3.0, 0.0, 2.0, -1.0,
        ];
        let input = [
            2.0, 1.0, 0.0, -1.0, 1.0, 3.0, -2.0, 0.0, 1.0, -1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0,
        ];

        let matrix = left_multiplication_matrix(multiplier);

        assert_eq!(
            matrix_vector_mul(&matrix, &input),
            sedenion_mul(multiplier, input)
        );
    }

    #[test]
    fn known_zero_divisor_is_in_the_operator_kernel() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let mut kernel_vector = ZERO;
        kernel_vector[4] = 1.0;
        kernel_vector[15] = -1.0;

        let operator = LeftMultiplicationOperator::new(multiplier);
        let output = operator.apply(&kernel_vector);

        assert_eq!(squared_norm(&kernel_vector), 2.0);
        assert_eq!(output, ZERO);
    }

    #[test]
    fn operator_preserves_its_multiplier_and_matrix() {
        let multiplier = basis_vector(3).expect("e3 exists");
        let operator = LeftMultiplicationOperator::new(multiplier);

        assert_eq!(operator.multiplier(), &multiplier);
        assert_eq!(
            operator.apply(&basis_vector(5).expect("e5 exists")),
            sedenion_mul(multiplier, basis_vector(5).expect("e5 exists"))
        );
    }
}
