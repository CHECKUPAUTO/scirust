//! Deterministic kernel layer for Elastic Latent KV.
//!
//! Explicit index loops define the scalar floating-point association and are
//! intentionally retained as the differential oracle for optimized backends.

#![allow(clippy::needless_range_loop)]

/// Compute backend used by the latent kernel dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentKernelKind {
    /// Reference scalar kernel and cross-backend oracle.
    Scalar,
    /// Stable Rust block-4 loop preserving scalar accumulation order.
    Block4,
    /// Nightly `std::simd` backend provided by `scirust-simd`.
    #[cfg(feature = "portable-simd")]
    PortableSimd,
}

/// Stateless deterministic latent-kernel dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentKernelDispatch {
    kind: LatentKernelKind,
}

impl LatentKernelDispatch {
    #[must_use]
    pub const fn new(kind: LatentKernelKind) -> Self {
        Self { kind }
    }

    #[must_use]
    pub const fn kind(self) -> LatentKernelKind {
        self.kind
    }

    /// Computes a contiguous dot product.
    #[must_use]
    pub fn dot(self, left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len());
        match self.kind
        {
            LatentKernelKind::Scalar => scalar_dot(left, right),
            LatentKernelKind::Block4 => block4_dot(left, right),
            #[cfg(feature = "portable-simd")]
            LatentKernelKind::PortableSimd => scirust_simd::simd_ops::dot_f32(left, right),
        }
    }

    /// Computes a strided matrix-column dot product starting from `initial`.
    #[must_use]
    pub fn dot_strided_with_initial(
        self,
        matrix: &[f32],
        rows: usize,
        columns: usize,
        column: usize,
        vector: &[f32],
        initial: f32,
    ) -> f32 {
        assert_eq!(matrix.len(), rows.saturating_mul(columns));
        assert_eq!(vector.len(), rows);
        assert!(column < columns);
        match self.kind
        {
            LatentKernelKind::Scalar => {
                scalar_dot_strided(matrix, rows, columns, column, vector, initial)
            }
            LatentKernelKind::Block4 => {
                block4_dot_strided(matrix, rows, columns, column, vector, initial)
            }
            #[cfg(feature = "portable-simd")]
            LatentKernelKind::PortableSimd => {
                block4_dot_strided(matrix, rows, columns, column, vector, initial)
            }
        }
    }

    /// Computes a strided matrix-column dot product from zero.
    #[must_use]
    pub fn dot_strided(
        self,
        matrix: &[f32],
        rows: usize,
        columns: usize,
        column: usize,
        vector: &[f32],
    ) -> f32 {
        self.dot_strided_with_initial(matrix, rows, columns, column, vector, 0.0)
    }

    /// Adds `weight * source` into `output` without allocation.
    pub fn weighted_accumulate(self, output: &mut [f32], source: &[f32], weight: f32) {
        assert_eq!(output.len(), source.len());
        match self.kind
        {
            LatentKernelKind::Scalar => scalar_weighted_accumulate(output, source, weight),
            LatentKernelKind::Block4 => block4_weighted_accumulate(output, source, weight),
            #[cfg(feature = "portable-simd")]
            LatentKernelKind::PortableSimd => {
                block4_weighted_accumulate(output, source, weight);
            }
        }
    }
}

#[inline]
fn scalar_dot(left: &[f32], right: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for index in 0..left.len()
    {
        sum += left[index] * right[index];
    }
    sum
}

#[inline]
fn block4_dot(left: &[f32], right: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let mut index = 0;
    while index + 4 <= left.len()
    {
        sum += left[index] * right[index];
        sum += left[index + 1] * right[index + 1];
        sum += left[index + 2] * right[index + 2];
        sum += left[index + 3] * right[index + 3];
        index += 4;
    }
    while index < left.len()
    {
        sum += left[index] * right[index];
        index += 1;
    }
    sum
}

#[inline]
fn scalar_dot_strided(
    matrix: &[f32],
    rows: usize,
    columns: usize,
    column: usize,
    vector: &[f32],
    initial: f32,
) -> f32 {
    let mut sum = initial;
    for row in 0..rows
    {
        sum += matrix[row * columns + column] * vector[row];
    }
    sum
}

#[inline]
fn block4_dot_strided(
    matrix: &[f32],
    rows: usize,
    columns: usize,
    column: usize,
    vector: &[f32],
    initial: f32,
) -> f32 {
    let mut sum = initial;
    let mut row = 0;
    while row + 4 <= rows
    {
        sum += matrix[row * columns + column] * vector[row];
        sum += matrix[(row + 1) * columns + column] * vector[row + 1];
        sum += matrix[(row + 2) * columns + column] * vector[row + 2];
        sum += matrix[(row + 3) * columns + column] * vector[row + 3];
        row += 4;
    }
    while row < rows
    {
        sum += matrix[row * columns + column] * vector[row];
        row += 1;
    }
    sum
}

#[inline]
fn scalar_weighted_accumulate(output: &mut [f32], source: &[f32], weight: f32) {
    for index in 0..output.len()
    {
        output[index] += weight * source[index];
    }
}

#[inline]
fn block4_weighted_accumulate(output: &mut [f32], source: &[f32], weight: f32) {
    let mut index = 0;
    while index + 4 <= output.len()
    {
        output[index] += weight * source[index];
        output[index + 1] += weight * source[index + 1];
        output[index + 2] += weight * source[index + 2];
        output[index + 3] += weight * source[index + 3];
        index += 4;
    }
    while index < output.len()
    {
        output[index] += weight * source[index];
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{LatentKernelDispatch, LatentKernelKind};

    fn data(length: usize) -> (Vec<f32>, Vec<f32>) {
        let left = (0..length)
            .map(|index| (index as f32 * 0.031).sin() - 0.2)
            .collect();
        let right = (0..length)
            .map(|index| (index as f32 * 0.017).cos() + 0.1)
            .collect();
        (left, right)
    }

    #[test]
    fn block4_dot_is_bit_identical_to_scalar() {
        for length in 0..65
        {
            let (left, right) = data(length);
            let scalar = LatentKernelDispatch::new(LatentKernelKind::Scalar).dot(&left, &right);
            let block = LatentKernelDispatch::new(LatentKernelKind::Block4).dot(&left, &right);
            assert_eq!(scalar.to_bits(), block.to_bits());
        }
    }

    #[test]
    fn block4_strided_dot_preserves_initial_accumulator_order() {
        let rows = 17;
        let columns = 5;
        let matrix: Vec<f32> = (0..rows * columns)
            .map(|index| index as f32 * 0.003 - 0.4)
            .collect();
        let vector: Vec<f32> = (0..rows).map(|index| index as f32 * -0.02 + 0.3).collect();
        for column in 0..columns
        {
            let scalar = LatentKernelDispatch::new(LatentKernelKind::Scalar)
                .dot_strided_with_initial(&matrix, rows, columns, column, &vector, 0.17);
            let block = LatentKernelDispatch::new(LatentKernelKind::Block4)
                .dot_strided_with_initial(&matrix, rows, columns, column, &vector, 0.17);
            assert_eq!(scalar.to_bits(), block.to_bits());
        }
    }

    #[test]
    fn block4_weighted_accumulate_is_bit_identical() {
        let (_, source) = data(37);
        let mut scalar = vec![0.1; source.len()];
        let mut block = scalar.clone();
        LatentKernelDispatch::new(LatentKernelKind::Scalar)
            .weighted_accumulate(&mut scalar, &source, -0.37);
        LatentKernelDispatch::new(LatentKernelKind::Block4)
            .weighted_accumulate(&mut block, &source, -0.37);
        assert_eq!(
            scalar.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            block.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "portable-simd")]
    #[test]
    fn portable_simd_dot_stays_close_to_scalar_oracle() {
        let (left, right) = data(257);
        let scalar = LatentKernelDispatch::new(LatentKernelKind::Scalar).dot(&left, &right);
        let simd = LatentKernelDispatch::new(LatentKernelKind::PortableSimd).dot(&left, &right);
        assert!((scalar - simd).abs() <= 2.0e-5);
    }
}
