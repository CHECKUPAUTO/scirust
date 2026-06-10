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
    _k: usize,
) {
    assert_eq!(out_c.len(), m * n);

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
