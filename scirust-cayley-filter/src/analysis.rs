//! Deterministic numerical analysis of 16 × 16 filtering operators.

use core::fmt;

use crate::operator::{Matrix16, matrix_vector_mul};
use crate::scalar::{SEDENION_DIMENSION, Sedenion, squared_norm};

/// Error returned when numerical analysis parameters are invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnalysisError {
    /// The tolerance must be finite and strictly positive.
    InvalidTolerance,
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidTolerance =>
            {
                formatter.write_str("tolerance must be finite and strictly positive")
            },
        }
    }
}

impl std::error::Error for AnalysisError {}

/// Deterministic rank and kernel analysis of a real 16 × 16 matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixAnalysis {
    rank: usize,
    pivot_columns: Vec<usize>,
    kernel_basis: Vec<Sedenion>,
    tolerance: f64,
}

impl MatrixAnalysis {
    /// Numerical rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Numerical nullity, equal to `16 - rank`.
    #[must_use]
    pub fn nullity(&self) -> usize {
        self.kernel_basis.len()
    }

    /// Pivot columns selected during deterministic elimination.
    #[must_use]
    pub fn pivot_columns(&self) -> &[usize] {
        &self.pivot_columns
    }

    /// Numerical basis of the right kernel `{x | Mx = 0}`.
    #[must_use]
    pub fn kernel_basis(&self) -> &[Sedenion] {
        &self.kernel_basis
    }

    /// Tolerance used for pivot and zero decisions.
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

/// Computes numerical rank, nullity and a right-kernel basis.
///
/// Pivot selection uses the largest absolute coefficient in the current
/// column. Equal candidates retain the lowest row index, making the result
/// deterministic.
pub fn analyze_matrix(matrix: &Matrix16, tolerance: f64) -> Result<MatrixAnalysis, AnalysisError> {
    if !tolerance.is_finite() || tolerance <= 0.0
    {
        return Err(AnalysisError::InvalidTolerance);
    }

    let (rref, pivot_columns) = reduced_row_echelon(matrix, tolerance);
    let kernel_basis = build_kernel_basis(&rref, &pivot_columns, tolerance);

    Ok(MatrixAnalysis {
        rank: pivot_columns.len(),
        pivot_columns,
        kernel_basis,
        tolerance,
    })
}

/// Euclidean norm of the residual `M × vector`.
#[must_use]
pub fn kernel_residual_norm(matrix: &Matrix16, vector: &Sedenion) -> f64 {
    squared_norm(&matrix_vector_mul(matrix, vector)).sqrt()
}

fn reduced_row_echelon(matrix: &Matrix16, tolerance: f64) -> (Matrix16, Vec<usize>) {
    let mut result = *matrix;
    let mut pivot_columns = Vec::with_capacity(SEDENION_DIMENSION);
    let mut pivot_row = 0usize;
    let mut column = 0usize;

    while column < SEDENION_DIMENSION && pivot_row < SEDENION_DIMENSION
    {
        let mut selected_row = pivot_row;
        let mut selected_value = result[pivot_row][column].abs();
        let mut candidate_row = pivot_row + 1;

        while candidate_row < SEDENION_DIMENSION
        {
            let candidate_value = result[candidate_row][column].abs();

            if candidate_value > selected_value
            {
                selected_row = candidate_row;
                selected_value = candidate_value;
            }

            candidate_row += 1;
        }

        if selected_value <= tolerance
        {
            column += 1;
            continue;
        }

        if selected_row != pivot_row
        {
            result.swap(selected_row, pivot_row);
        }

        let pivot = result[pivot_row][column];
        let mut normalize_column = 0usize;

        while normalize_column < SEDENION_DIMENSION
        {
            result[pivot_row][normalize_column] /= pivot;
            normalize_column += 1;
        }

        let mut row = 0usize;

        while row < SEDENION_DIMENSION
        {
            if row != pivot_row
            {
                let factor = result[row][column];

                if factor.abs() > tolerance
                {
                    let mut elimination_column = 0usize;

                    while elimination_column < SEDENION_DIMENSION
                    {
                        result[row][elimination_column] -=
                            factor * result[pivot_row][elimination_column];
                        elimination_column += 1;
                    }
                }
            }

            row += 1;
        }

        zero_small_entries(&mut result, tolerance);
        pivot_columns.push(column);
        pivot_row += 1;
        column += 1;
    }

    (result, pivot_columns)
}

fn build_kernel_basis(rref: &Matrix16, pivot_columns: &[usize], tolerance: f64) -> Vec<Sedenion> {
    let mut is_pivot = [false; SEDENION_DIMENSION];

    for &column in pivot_columns
    {
        is_pivot[column] = true;
    }

    let mut basis = Vec::with_capacity(SEDENION_DIMENSION - pivot_columns.len());
    let mut free_column = 0usize;

    while free_column < SEDENION_DIMENSION
    {
        if !is_pivot[free_column]
        {
            let mut vector = [0.0; SEDENION_DIMENSION];
            vector[free_column] = 1.0;

            for (pivot_row, &pivot_column) in pivot_columns.iter().enumerate()
            {
                let value = -rref[pivot_row][free_column];
                vector[pivot_column] = if value.abs() <= tolerance { 0.0 } else { value };
            }

            basis.push(vector);
        }

        free_column += 1;
    }

    basis
}

fn zero_small_entries(matrix: &mut Matrix16, tolerance: f64) {
    for row in matrix
    {
        for value in row
        {
            if value.abs() <= tolerance
            {
                *value = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AnalysisError, analyze_matrix, kernel_residual_norm};
    use crate::operator::left_multiplication_matrix;
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector};

    const TOLERANCE: f64 = 1.0e-12;
    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

    #[test]
    fn identity_has_full_rank_and_zero_nullity() {
        let one = basis_vector(0).expect("e0 exists");
        let matrix = left_multiplication_matrix(one);
        let analysis = analyze_matrix(&matrix, TOLERANCE).expect("tolerance is valid");

        assert_eq!(analysis.rank(), SEDENION_DIMENSION);
        assert_eq!(analysis.nullity(), 0);
        assert!(analysis.kernel_basis().is_empty());
    }

    #[test]
    fn zero_matrix_has_standard_basis_as_kernel() {
        let matrix = [[0.0; SEDENION_DIMENSION]; SEDENION_DIMENSION];
        let analysis = analyze_matrix(&matrix, TOLERANCE).expect("tolerance is valid");

        assert_eq!(analysis.rank(), 0);
        assert_eq!(analysis.nullity(), SEDENION_DIMENSION);

        for (index, vector) in analysis.kernel_basis().iter().enumerate()
        {
            assert_eq!(*vector, basis_vector(index).expect("basis element exists"));
        }
    }

    #[test]
    fn known_zero_divisor_operator_has_non_trivial_kernel() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let matrix = left_multiplication_matrix(multiplier);
        let analysis = analyze_matrix(&matrix, TOLERANCE).expect("tolerance is valid");

        assert!(analysis.rank() < SEDENION_DIMENSION);
        assert!(analysis.nullity() > 0);
        assert_eq!(analysis.rank() + analysis.nullity(), SEDENION_DIMENSION);

        for vector in analysis.kernel_basis()
        {
            assert!(
                kernel_residual_norm(&matrix, vector) <= TOLERANCE,
                "kernel residual exceeds tolerance"
            );
        }
    }

    #[test]
    fn known_annihilated_vector_has_zero_residual() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let mut kernel_vector = ZERO;
        kernel_vector[4] = 1.0;
        kernel_vector[15] = -1.0;

        let matrix = left_multiplication_matrix(multiplier);

        assert_eq!(kernel_residual_norm(&matrix, &kernel_vector), 0.0);
    }

    #[test]
    fn analysis_is_deterministic() {
        let multiplier = basis_vector(7).expect("e7 exists");
        let matrix = left_multiplication_matrix(multiplier);

        let first = analyze_matrix(&matrix, TOLERANCE).expect("tolerance is valid");
        let second = analyze_matrix(&matrix, TOLERANCE).expect("tolerance is valid");

        assert_eq!(first, second);
    }

    #[test]
    fn invalid_tolerances_are_rejected() {
        let matrix = [[0.0; SEDENION_DIMENSION]; SEDENION_DIMENSION];

        for tolerance in [0.0, -1.0, f64::INFINITY, f64::NAN]
        {
            assert_eq!(
                analyze_matrix(&matrix, tolerance),
                Err(AnalysisError::InvalidTolerance)
            );
        }
    }
}
