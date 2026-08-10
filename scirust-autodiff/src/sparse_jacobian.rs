//! Sparse-Jacobian planning with deterministic column-intersection coloring.
//!
//! The sparsity pattern is stored in CSR form. Coloring is a planning-time
//! operation: columns that never appear together in a row may share one forward
//! tangent direction. The implementation uses generation-marked dense scratch
//! arrays rather than per-column `HashSet` allocations.

/// Validation error for a CSR sparsity pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparsityError {
    InvalidRowOffsets,
    ColumnOutOfBounds { column: usize, cols: usize },
    DuplicateColumn { row: usize, column: usize },
    TooManyColors { required: usize, available: usize },
    CompressedLengthMismatch { expected: usize, actual: usize },
    OutputLengthMismatch { expected: usize, actual: usize },
}

impl core::fmt::Display for SparsityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::InvalidRowOffsets => write!(f, "invalid CSR row offsets"),
            Self::ColumnOutOfBounds { column, cols } => {
                write!(f, "CSR column {column} is outside 0..{cols}")
            },
            Self::DuplicateColumn { row, column } => {
                write!(f, "CSR row {row} contains duplicate column {column}")
            },
            Self::TooManyColors { required, available } => write!(
                f,
                "sparse Jacobian coloring requires {required} directions, but only {available} are available"
            ),
            Self::CompressedLengthMismatch { expected, actual } => write!(
                f,
                "compressed Jacobian has {actual} entries, expected {expected}"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                f,
                "Jacobian output has {actual} entries, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for SparsityError {}

/// Validated row-major CSR sparsity pattern for a Jacobian.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JacobianSparsity {
    rows: usize,
    cols: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
}

impl JacobianSparsity {
    /// Construct and validate a CSR sparsity pattern.
    pub fn new(
        rows: usize,
        cols: usize,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
    ) -> Result<Self, SparsityError> {
        if row_offsets.len() != rows + 1
            || row_offsets.first().copied() != Some(0)
            || row_offsets.last().copied() != Some(column_indices.len())
            || row_offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(SparsityError::InvalidRowOffsets);
        }

        for (row, bounds) in row_offsets.windows(2).enumerate()
        {
            let entries = &column_indices[bounds[0]..bounds[1]];
            let mut previous = None;
            for &column in entries
            {
                if column >= cols
                {
                    return Err(SparsityError::ColumnOutOfBounds { column, cols });
                }
                if previous == Some(column)
                {
                    return Err(SparsityError::DuplicateColumn { row, column });
                }
                previous = Some(column);
            }
            // Sorting makes duplicate detection deterministic and lets consumers
            // rely on canonical row ordering. Unsorted input is rejected through
            // the same invariant rather than silently reordered.
            if entries.windows(2).any(|pair| pair[0] > pair[1])
            {
                return Err(SparsityError::InvalidRowOffsets);
            }
        }

        Ok(Self {
            rows,
            cols,
            row_offsets,
            column_indices,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.column_indices.len()
    }

    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    pub fn row_columns(&self, row: usize) -> Option<&[usize]> {
        let start = *self.row_offsets.get(row)?;
        let end = *self.row_offsets.get(row + 1)?;
        self.column_indices.get(start..end)
    }

    /// Deterministically color the column-intersection graph.
    ///
    /// Columns are processed in ascending index order. A color is forbidden for
    /// column `c` when an already-colored column shares any CSR row with `c`.
    /// No hash table is allocated in the coloring loop.
    pub fn color_columns(&self) -> ColumnColoring {
        if self.cols == 0
        {
            return ColumnColoring {
                colors: Vec::new(),
                color_count: 0,
            };
        }

        let (column_offsets, column_rows) = self.transpose_structure();
        let mut colors = vec![usize::MAX; self.cols];
        let mut forbidden_generation = vec![0usize; self.cols + 1];
        let mut generation = 1usize;
        let mut color_count = 0usize;

        for column in 0..self.cols
        {
            if generation == usize::MAX
            {
                forbidden_generation.fill(0);
                generation = 1;
            }

            for &row in &column_rows[column_offsets[column]..column_offsets[column + 1]]
            {
                let start = self.row_offsets[row];
                let end = self.row_offsets[row + 1];
                for &neighbor in &self.column_indices[start..end]
                {
                    if neighbor >= column
                    {
                        continue;
                    }
                    let color = colors[neighbor];
                    if color != usize::MAX
                    {
                        forbidden_generation[color] = generation;
                    }
                }
            }

            let mut color = 0usize;
            while forbidden_generation[color] == generation
            {
                color += 1;
            }
            colors[column] = color;
            color_count = color_count.max(color + 1);
            generation += 1;
        }

        ColumnColoring {
            colors,
            color_count,
        }
    }

    /// Decompress row-major directional derivatives into CSR-value order.
    ///
    /// `compressed` is laid out as `rows × width`. For every nonzero `(row,
    /// column)` the output value is `compressed[row, color(column)]`.
    pub fn decompress_into<T: Copy>(
        &self,
        coloring: &ColumnColoring,
        compressed: &[T],
        width: usize,
        output_values: &mut [T],
    ) -> Result<(), SparsityError> {
        coloring.require_width(width)?;
        let expected_compressed = self.rows.checked_mul(width).unwrap_or(usize::MAX);
        if compressed.len() != expected_compressed
        {
            return Err(SparsityError::CompressedLengthMismatch {
                expected: expected_compressed,
                actual: compressed.len(),
            });
        }
        if output_values.len() != self.nnz()
        {
            return Err(SparsityError::OutputLengthMismatch {
                expected: self.nnz(),
                actual: output_values.len(),
            });
        }

        for row in 0..self.rows
        {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            for index in start..end
            {
                let column = self.column_indices[index];
                let color = coloring.colors[column];
                output_values[index] = compressed[row * width + color];
            }
        }
        Ok(())
    }

    fn transpose_structure(&self) -> (Vec<usize>, Vec<usize>) {
        let mut counts = vec![0usize; self.cols];
        for &column in &self.column_indices
        {
            counts[column] += 1;
        }

        let mut offsets = vec![0usize; self.cols + 1];
        for column in 0..self.cols
        {
            offsets[column + 1] = offsets[column] + counts[column];
        }

        let mut cursor = offsets[..self.cols].to_vec();
        let mut rows = vec![0usize; self.column_indices.len()];
        for row in 0..self.rows
        {
            for &column in self.row_columns(row).expect("validated row")
            {
                let slot = cursor[column];
                rows[slot] = row;
                cursor[column] += 1;
            }
        }
        (offsets, rows)
    }
}

/// Deterministic column-color assignment used as a compressed-AD seed plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnColoring {
    colors: Vec<usize>,
    color_count: usize,
}

impl ColumnColoring {
    pub fn colors(&self) -> &[usize] {
        &self.colors
    }

    pub fn color_count(&self) -> usize {
        self.color_count
    }

    pub fn color_of(&self, column: usize) -> Option<usize> {
        self.colors.get(column).copied()
    }

    /// Verify that a `DualPack<_, W>`-like width can carry this coloring.
    pub fn require_width(&self, width: usize) -> Result<(), SparsityError> {
        if self.color_count > width
        {
            return Err(SparsityError::TooManyColors {
                required: self.color_count,
                available: width,
            });
        }
        Ok(())
    }

    /// Fill one seed row (one lane per color) for `column` without allocating.
    pub fn seed_column_into<T: Copy>(
        &self,
        column: usize,
        zero: T,
        one: T,
        seed: &mut [T],
    ) -> Result<(), SparsityError> {
        self.require_width(seed.len())?;
        seed.fill(zero);
        if let Some(color) = self.color_of(column)
        {
            seed[color] = one;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tridiagonal(n: usize) -> JacobianSparsity {
        let mut offsets = Vec::with_capacity(n + 1);
        let mut columns = Vec::new();
        offsets.push(0);
        for row in 0..n
        {
            if row > 0
            {
                columns.push(row - 1);
            }
            columns.push(row);
            if row + 1 < n
            {
                columns.push(row + 1);
            }
            offsets.push(columns.len());
        }
        JacobianSparsity::new(n, n, offsets, columns).expect("valid tridiagonal CSR")
    }

    #[test]
    fn tridiagonal_uses_three_colors() {
        let pattern = tridiagonal(32);
        let coloring = pattern.color_columns();
        assert_eq!(coloring.color_count(), 3);
        for row in 0..pattern.rows()
        {
            let columns = pattern.row_columns(row).unwrap();
            for (i, &left) in columns.iter().enumerate()
            {
                for &right in &columns[i + 1..]
                {
                    assert_ne!(coloring.color_of(left), coloring.color_of(right));
                }
            }
        }
    }

    #[test]
    fn decompression_recovers_csr_values() {
        let pattern = tridiagonal(5);
        let coloring = pattern.color_columns();
        let width = coloring.color_count();
        let mut compressed = vec![0.0f64; pattern.rows() * width];

        for row in 0..pattern.rows()
        {
            for &column in pattern.row_columns(row).unwrap()
            {
                let color = coloring.color_of(column).unwrap();
                compressed[row * width + color] = (100 * row + column) as f64;
            }
        }

        let mut values = vec![0.0f64; pattern.nnz()];
        pattern
            .decompress_into(&coloring, &compressed, width, &mut values)
            .expect("decompression");

        let mut index = 0;
        for row in 0..pattern.rows()
        {
            for &column in pattern.row_columns(row).unwrap()
            {
                assert_eq!(values[index], (100 * row + column) as f64);
                index += 1;
            }
        }
    }

    #[test]
    fn seed_plan_rejects_insufficient_width() {
        let coloring = tridiagonal(8).color_columns();
        let mut seed = [0.0f32; 2];
        assert_eq!(
            coloring.seed_column_into(0, 0.0, 1.0, &mut seed),
            Err(SparsityError::TooManyColors {
                required: 3,
                available: 2,
            })
        );
    }

    #[test]
    fn rejects_duplicate_and_out_of_range_columns() {
        assert_eq!(
            JacobianSparsity::new(1, 3, vec![0, 2], vec![1, 1]),
            Err(SparsityError::DuplicateColumn { row: 0, column: 1 })
        );
        assert_eq!(
            JacobianSparsity::new(1, 2, vec![0, 1], vec![2]),
            Err(SparsityError::ColumnOutOfBounds { column: 2, cols: 2 })
        );
    }
}
