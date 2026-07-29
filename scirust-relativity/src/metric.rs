//! Metric tensors and basic tensor operations.

use crate::RelativityError;

/// A covariant metric tensor `g_{mu nu}` in `D` coordinates.
pub trait Metric<const D: usize> {
    /// Evaluate the covariant metric components at `coordinates`.
    fn components(&self, coordinates: &[f64; D]) -> [[f64; D]; D];
}

/// Invert a square metric tensor with deterministic Gauss–Jordan elimination.
///
/// Partial pivoting is used. Ties are resolved by retaining the first row,
/// making the operation deterministic for identical floating-point inputs.
pub fn invert_metric<const D: usize>(
    metric: &[[f64; D]; D],
) -> Result<[[f64; D]; D], RelativityError> {
    let mut augmented = vec![vec![0.0_f64; 2 * D]; D];

    for row in 0..D
    {
        for column in 0..D
        {
            let value = metric[row][column];

            if !value.is_finite()
            {
                return Err(RelativityError::NonFiniteMetricComponent { row, column });
            }

            augmented[row][column] = value;
        }

        augmented[row][D + row] = 1.0;
    }

    for pivot_column in 0..D
    {
        let mut pivot_row = pivot_column;
        let mut pivot_magnitude = augmented[pivot_row][pivot_column].abs();

        for (candidate, candidate_row) in augmented.iter().enumerate().skip(pivot_column + 1)
        {
            let magnitude = candidate_row[pivot_column].abs();

            if magnitude > pivot_magnitude
            {
                pivot_row = candidate;
                pivot_magnitude = magnitude;
            }
        }

        if !pivot_magnitude.is_finite() || pivot_magnitude <= f64::EPSILON
        {
            return Err(RelativityError::SingularMetric);
        }

        if pivot_row != pivot_column
        {
            augmented.swap(pivot_row, pivot_column);
        }

        let pivot = augmented[pivot_column][pivot_column];

        for value in &mut augmented[pivot_column]
        {
            *value /= pivot;
        }

        let normalized_pivot_row = augmented[pivot_column].clone();

        for (row_index, row_values) in augmented.iter_mut().enumerate()
        {
            if row_index == pivot_column
            {
                continue;
            }

            let factor = row_values[pivot_column];

            for (value, pivot_value) in row_values.iter_mut().zip(&normalized_pivot_row)
            {
                *value -= factor * *pivot_value;
            }
        }
    }

    let mut inverse = [[0.0_f64; D]; D];

    for row in 0..D
    {
        for column in 0..D
        {
            inverse[row][column] = augmented[row][D + column];
        }
    }

    Ok(inverse)
}

/// Determinant of a square matrix by deterministic Gauss elimination with
/// partial pivoting.
///
/// Partial pivoting and the retain-first tie rule make the result deterministic
/// for identical floating-point inputs. A non-finite entry is reported as
/// [`RelativityError::NonFiniteMetricComponent`]; an exactly singular matrix
/// (a zero pivot column) has determinant `0.0`, returned as a value rather than
/// an error.
pub fn determinant<const D: usize>(matrix: &[[f64; D]; D]) -> Result<f64, RelativityError> {
    for (row, source) in matrix.iter().enumerate()
    {
        for (column, &value) in source.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(RelativityError::NonFiniteMetricComponent { row, column });
            }
        }
    }

    let mut work = *matrix;
    let mut determinant = 1.0_f64;

    for pivot in 0..D
    {
        let mut pivot_row = pivot;
        let mut pivot_magnitude = work[pivot][pivot].abs();
        for (candidate, candidate_row) in work.iter().enumerate().skip(pivot + 1)
        {
            let magnitude = candidate_row[pivot].abs();
            if magnitude > pivot_magnitude
            {
                pivot_row = candidate;
                pivot_magnitude = magnitude;
            }
        }

        if pivot_magnitude == 0.0
        {
            return Ok(0.0);
        }

        if pivot_row != pivot
        {
            work.swap(pivot_row, pivot);
            determinant = -determinant;
        }

        let pivot_values = work[pivot];
        determinant *= pivot_values[pivot];

        for (row_index, row_values) in work.iter_mut().enumerate()
        {
            if row_index <= pivot
            {
                continue;
            }
            let factor = row_values[pivot] / pivot_values[pivot];
            for (value, &pivot_value) in row_values.iter_mut().zip(pivot_values.iter())
            {
                *value -= factor * pivot_value;
            }
        }
    }

    Ok(determinant)
}

/// Evaluate the quadratic form `g_{mu nu} v^mu v^nu`.
#[must_use]
pub fn metric_norm<const D: usize>(metric: &[[f64; D]; D], vector: &[f64; D]) -> f64 {
    let mut value = 0.0;

    for mu in 0..D
    {
        for nu in 0..D
        {
            value += metric[mu][nu] * vector[mu] * vector[nu];
        }
    }

    value
}
