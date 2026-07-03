/// Compressed Sparse Row (CSR) Tensor implementation for memory-efficient
/// representation of sparse matrices on constrained edge targets.
#[derive(Debug, Clone)]
pub struct CsrTensor<T> {
    pub values: Vec<T>,
    pub column_indices: Vec<usize>,
    pub row_offsets: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

impl CsrTensor<f32> {
    pub fn new(
        values: Vec<f32>,
        column_indices: Vec<usize>,
        row_offsets: Vec<usize>,
        rows: usize,
        cols: usize,
    ) -> Self {
        assert_eq!(row_offsets.len(), rows + 1);
        assert_eq!(values.len(), column_indices.len());
        Self {
            values,
            column_indices,
            row_offsets,
            rows,
            cols,
        }
    }
}

/// High-performance Sparse Matrix-Matrix Multiplication (SpMM) kernel.
/// Computes OutC = SparseA * DenseB
/// Optimized for cache locality and multi-core readiness.
///
/// m: rows of SparseA
/// k: cols of SparseA / rows of DenseB
/// n: cols of DenseB
pub fn spmm_dense(
    sparse_a: &CsrTensor<f32>,
    dense_b: &[f32],
    out_c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(out_c.len(), m * n, "spmm_dense: out_c.len() != m*n");
    // Validate the inner dimension and structure so a malformed CSR / dense
    // operand is a clear error instead of an out-of-bounds read: DenseB is
    // (k x n), SparseA needs m+1 row offsets, and every column index must be a
    // valid row of DenseB (< k). Previously `k` was ignored (`_k`).
    assert_eq!(dense_b.len(), k * n, "spmm_dense: dense_b.len() != k*n");
    assert_eq!(
        sparse_a.row_offsets.len(),
        m + 1,
        "spmm_dense: row_offsets.len() != m+1"
    );
    assert!(
        sparse_a.column_indices.iter().all(|&c| c < k),
        "spmm_dense: a column index is out of range [0, k)"
    );

    // The documented contract is OutC = SparseA * DenseB (assignment, not
    // accumulation). Zero the output first so stale contents in the caller's
    // buffer, and rows of SparseA with no non-zero entries, do not leak into
    // the result.
    for c in out_c.iter_mut()
    {
        *c = 0.0;
    }

    // Outer loop over SparseA rows
    for i in 0..m
    {
        let row_start = sparse_a.row_offsets[i];
        let row_end = sparse_a.row_offsets[i + 1];

        let out_row_offset = i * n;

        // Iterate over non-zero elements in current row of SparseA
        for idx in row_start..row_end
        {
            let val_a = sparse_a.values[idx];
            let col_a = sparse_a.column_indices[idx];

            let b_row_offset = col_a * n;

            // Inner loop over DenseB columns (dense row)
            // This access pattern is cache-friendly for DenseB in row-major format
            for j in 0..n
            {
                out_c[out_row_offset + j] += val_a * dense_b[b_row_offset + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spmm_dense_overwrites_stale_output() {
        // SparseA (2x2):
        // [1 0]
        // [0 2]
        let sparse_a = CsrTensor::new(vec![1.0, 2.0], vec![0, 1], vec![0, 1, 2], 2, 2);

        // DenseB (2x2, row-major):
        // [3 4]
        // [5 6]
        let dense_b = [3.0, 4.0, 5.0, 6.0];

        // Pre-fill the output with garbage; with '=' semantics it must be
        // fully overwritten. Before the fix, the '+=' accumulation would leave
        // the stale values folded into the result.
        let mut out_c = [100.0, 200.0, 300.0, 400.0];

        spmm_dense(&sparse_a, &dense_b, &mut out_c, 2, 2, 2);

        // Expected = SparseA * DenseB:
        // [1*3, 1*4]   [3, 4]
        // [2*5, 2*6] = [10, 12]
        assert_eq!(out_c, [3.0, 4.0, 10.0, 12.0]);
    }

    #[test]
    #[should_panic(expected = "column index is out of range")]
    fn spmm_dense_rejects_out_of_range_column() {
        // SparseA references column 5, but DenseB has only k=2 rows — a mismatch
        // that previously read out of bounds instead of erroring.
        let sparse_a = CsrTensor::new(vec![1.0], vec![5], vec![0, 1, 1], 2, 6);
        let dense_b = [1.0, 2.0, 3.0, 4.0]; // k=2, n=2
        let mut out_c = [0.0; 4]; // m=2, n=2
        spmm_dense(&sparse_a, &dense_b, &mut out_c, 2, 2, 2);
    }

    #[test]
    fn spmm_dense_zeros_empty_rows() {
        // SparseA (2x2) with an empty first row:
        // [0 0]
        // [0 1]
        let sparse_a = CsrTensor::new(vec![1.0], vec![1], vec![0, 0, 1], 2, 2);

        let dense_b = [3.0, 4.0, 5.0, 6.0];

        // The empty row must be zeroed, not left holding stale data.
        let mut out_c = [7.0, 8.0, 9.0, 10.0];

        spmm_dense(&sparse_a, &dense_b, &mut out_c, 2, 2, 2);

        // Row 0 is all zero; row 1 = 1 * [5, 6].
        assert_eq!(out_c, [0.0, 0.0, 5.0, 6.0]);
    }
}
