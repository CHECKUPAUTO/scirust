//! `scirust-sparse` — sparse matrices (COO/CSR/CSC) and solvers in pure Rust.
//!
//! This crate provides three sparse matrix layouts for `f64` data —
//! coordinate ([`CooMatrix`]), compressed sparse row ([`CsrMatrix`]) and
//! compressed sparse column ([`CscMatrix`]) — with lossless conversions
//! between them, a sparse matrix–vector product ([`CsrMatrix::spmv`]), and a
//! small family of solvers:
//!
//! * [`solve_tridiagonal`] — the Thomas algorithm for tridiagonal systems.
//! * [`SparseLu`] — a Gilbert–Peierls left-looking sparse LU factorization
//!   with partial pivoting, reusable across right-hand sides.
//! * [`conjugate_gradient`] — the conjugate-gradient iteration for symmetric
//!   positive-definite systems.
//!
//! Every fallible operation returns [`Result`](std::result::Result) with the
//! [`SparseError`] enum; malformed-but-well-typed input (out-of-range indices,
//! dimension mismatches, singular matrices) never panics.
//!
//! # Example
//!
//! ```
//! use scirust_sparse::{CooMatrix, SparseLu};
//!
//! // Assemble A = [[4, 3], [6, 3]] as triplets, with a duplicate that sums.
//! let mut coo = CooMatrix::new(2, 2);
//! coo.push(0, 0, 1.0).unwrap();
//! coo.push(0, 0, 3.0).unwrap(); // duplicate: (0,0) becomes 4.0 on conversion
//! coo.push(0, 1, 3.0).unwrap();
//! coo.push(1, 0, 6.0).unwrap();
//! coo.push(1, 1, 3.0).unwrap();
//!
//! // Sparse matrix–vector product y = A x via CSR.
//! let csr = coo.to_csr();
//! let y = csr.spmv(&[1.0, 1.0]).unwrap();
//! assert_eq!(y, vec![7.0, 9.0]);
//!
//! // Direct solve A x = b via sparse LU (factor once, reuse).
//! let lu = SparseLu::factor(&coo.to_csc()).unwrap();
//! let x = lu.solve(&[1.0, 2.0]).unwrap();
//! assert!((x[0] - 0.5).abs() < 1e-12);
//! assert!((x[1] + 1.0 / 3.0).abs() < 1e-12);
//! ```
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::fmt;

/// A single sparse column: `(row, value)` pairs used inside [`SparseLu`].
type SparseCol = Vec<(usize, f64)>;

/// Given the largest-magnitude entry of a matrix and its size, returns the
/// pivot-rejection threshold `n · eps · max|a_ij|` (Golub & Van Loan, *Matrix
/// Computations*, §3.4.6). A pivot at or below this is treated as
/// numerically zero.
///
/// This is deliberately *relative* to the matrix's own scale rather than a
/// fixed absolute constant: a regular matrix at a small physical scale (e.g.
/// capacitances in Farads, ~1e-12) was previously declared singular by a
/// fixed `1e-12` cutoff even though it is perfectly well-conditioned once
/// rescaled.
fn pivot_tol(n: usize, max_abs: f64) -> f64 {
    (n as f64) * f64::EPSILON * max_abs.max(1e-300)
}

/// Errors returned by the sparse matrix and solver routines.
///
/// The input is always assumed to be well-typed (`f64` values, `usize`
/// indices); these variants report *structural* problems (shape, ordering,
/// singularity, non-convergence) rather than type errors, and are returned in
/// place of panicking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseError {
    /// An index passed to a builder fell outside the declared matrix shape.
    IndexOutOfBounds {
        /// Offending row index.
        row: usize,
        /// Offending column index.
        col: usize,
        /// Number of rows in the matrix.
        rows: usize,
        /// Number of columns in the matrix.
        cols: usize,
    },
    /// A vector or matrix dimension did not match what the operation required.
    DimensionMismatch {
        /// Length the operation expected.
        expected: usize,
        /// Length actually supplied.
        found: usize,
    },
    /// A square matrix was required but a rectangular one was supplied.
    NotSquare {
        /// Number of rows supplied.
        rows: usize,
        /// Number of columns supplied.
        cols: usize,
    },
    /// A dense matrix supplied to [`CsrMatrix::from_dense`] had rows of
    /// differing lengths.
    InconsistentRowLength {
        /// Row index whose length differed.
        row: usize,
        /// Length expected (that of the first row).
        expected: usize,
        /// Length found at `row`.
        found: usize,
    },
    /// The matrix was (numerically) singular: an LU pivot was zero.
    Singular,
    /// A zero pivot was encountered while running the Thomas algorithm.
    ZeroPivot {
        /// Index of the eliminated row whose pivot vanished.
        index: usize,
    },
    /// The conjugate-gradient iteration did not reach the tolerance within the
    /// allotted number of iterations.
    NotConverged {
        /// Iteration budget that was exhausted.
        max_iter: usize,
    },
    /// The conjugate-gradient iteration detected a non-positive curvature
    /// (`pᵀAp ≤ 0`), so the operator is not symmetric positive-definite.
    NotPositiveDefinite,
}

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            SparseError::IndexOutOfBounds {
                row,
                col,
                rows,
                cols,
            } => write!(
                f,
                "index ({row}, {col}) out of bounds for a {rows}x{cols} matrix"
            ),
            SparseError::DimensionMismatch { expected, found } =>
            {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            },
            SparseError::NotSquare { rows, cols } =>
            {
                write!(f, "matrix must be square, but is {rows}x{cols}")
            },
            SparseError::InconsistentRowLength {
                row,
                expected,
                found,
            } => write!(f, "dense row {row} has length {found}, expected {expected}"),
            SparseError::Singular => write!(f, "matrix is singular (zero pivot in LU)"),
            SparseError::ZeroPivot { index } =>
            {
                write!(f, "zero pivot at index {index} in tridiagonal solve")
            },
            SparseError::NotConverged { max_iter } =>
            {
                write!(f, "did not converge within {max_iter} iterations")
            },
            SparseError::NotPositiveDefinite =>
            {
                write!(f, "operator is not symmetric positive-definite")
            },
        }
    }
}

impl std::error::Error for SparseError {}

/// Build a compressed layout (`indptr`, `indices`, `data`) from raw
/// `(outer, inner, value)` triplets.
///
/// Entries are bucketed by their `outer` key (a row index for CSR, a column
/// index for CSC), sorted by `inner`, and duplicate `(outer, inner)` pairs are
/// summed. The caller must guarantee every `outer < outer_dim`.
fn build_compressed(
    outer_dim: usize,
    entries: Vec<(usize, usize, f64)>,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    // Bucket boundaries via counting sort: bucket[o] becomes the start offset
    // of outer index `o`, and bucket[outer_dim] the raw entry count.
    let mut bucket = vec![0usize; outer_dim + 1];
    for &(o, _, _) in &entries
    {
        bucket[o + 1] += 1;
    }
    let mut acc = 0usize;
    for slot in bucket.iter_mut()
    {
        acc += *slot;
        *slot = acc;
    }

    let total = bucket[outer_dim];
    let mut scattered_inner = vec![0usize; total];
    let mut scattered_val = vec![0.0f64; total];
    let mut next = bucket[..outer_dim].to_vec();
    for (o, i, v) in entries
    {
        let p = next[o];
        scattered_inner[p] = i;
        scattered_val[p] = v;
        next[o] += 1;
    }

    let mut indptr = vec![0usize; outer_dim + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for o in 0..outer_dim
    {
        let start = bucket[o];
        let end = bucket[o + 1];
        let mut pairs: Vec<(usize, f64)> = (start..end)
            .map(|k| (scattered_inner[k], scattered_val[k]))
            .collect();
        pairs.sort_by_key(|p| p.0);
        let mut j = 0;
        while j < pairs.len()
        {
            let col = pairs[j].0;
            let mut sum = pairs[j].1;
            let mut m = j + 1;
            while m < pairs.len() && pairs[m].0 == col
            {
                sum += pairs[m].1;
                m += 1;
            }
            indices.push(col);
            data.push(sum);
            j = m;
        }
        indptr[o + 1] = indices.len();
    }

    (indptr, indices, data)
}

/// A sparse matrix in coordinate (triplet) format.
///
/// Entries are stored as an unordered list of `(row, col, value)` triplets.
/// Duplicate coordinates are permitted and are summed when the matrix is
/// converted to a compressed format.
#[derive(Debug, Clone)]
pub struct CooMatrix {
    rows: usize,
    cols: usize,
    triplets: Vec<(usize, usize, f64)>,
}

impl CooMatrix {
    /// Create an empty `rows`×`cols` matrix with no entries.
    pub fn new(rows: usize, cols: usize) -> CooMatrix {
        CooMatrix {
            rows,
            cols,
            triplets: Vec::new(),
        }
    }

    /// Append a `(r, c, v)` triplet.
    ///
    /// Returns [`SparseError::IndexOutOfBounds`] if `r` or `c` lies outside the
    /// declared shape. Duplicate coordinates are allowed and sum on conversion.
    pub fn push(&mut self, r: usize, c: usize, v: f64) -> Result<(), SparseError> {
        if r >= self.rows || c >= self.cols
        {
            return Err(SparseError::IndexOutOfBounds {
                row: r,
                col: c,
                rows: self.rows,
                cols: self.cols,
            });
        }
        self.triplets.push((r, c, v));
        Ok(())
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of stored triplets (including any duplicates).
    pub fn nnz(&self) -> usize {
        self.triplets.len()
    }

    /// Value at `(r, c)`, summing any duplicate triplets. Out-of-range
    /// coordinates read as `0.0`.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        let mut sum = 0.0;
        for &(rr, cc, v) in &self.triplets
        {
            if rr == r && cc == c
            {
                sum += v;
            }
        }
        sum
    }

    /// Convert to compressed sparse row format, summing duplicate entries and
    /// sorting column indices within each row.
    pub fn to_csr(&self) -> CsrMatrix {
        let entries: Vec<(usize, usize, f64)> = self.triplets.clone();
        let (indptr, indices, data) = build_compressed(self.rows, entries);
        CsrMatrix {
            rows: self.rows,
            cols: self.cols,
            indptr,
            indices,
            data,
        }
    }

    /// Convert to compressed sparse column format, summing duplicate entries and
    /// sorting row indices within each column.
    pub fn to_csc(&self) -> CscMatrix {
        let entries: Vec<(usize, usize, f64)> =
            self.triplets.iter().map(|&(r, c, v)| (c, r, v)).collect();
        let (indptr, indices, data) = build_compressed(self.cols, entries);
        CscMatrix {
            rows: self.rows,
            cols: self.cols,
            indptr,
            indices,
            data,
        }
    }

    /// Materialize the matrix as a dense row-major `Vec<Vec<f64>>`.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.cols]; self.rows];
        for &(r, c, v) in &self.triplets
        {
            dense[r][c] += v;
        }
        dense
    }
}

/// A sparse matrix in compressed sparse row (CSR) format.
///
/// Row `r` occupies `indices[indptr[r]..indptr[r + 1]]` (column indices, sorted
/// ascending) with parallel values in `data`.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    rows: usize,
    cols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl CsrMatrix {
    /// Build a CSR matrix from a dense row-major matrix, dropping structural
    /// zeros.
    ///
    /// Returns [`SparseError::InconsistentRowLength`] if the rows are not all of
    /// equal length.
    pub fn from_dense(dense: &[Vec<f64>]) -> Result<CsrMatrix, SparseError> {
        let rows = dense.len();
        let cols = if rows == 0 { 0 } else { dense[0].len() };
        for (r, row) in dense.iter().enumerate()
        {
            if row.len() != cols
            {
                return Err(SparseError::InconsistentRowLength {
                    row: r,
                    expected: cols,
                    found: row.len(),
                });
            }
        }
        let mut entries = Vec::new();
        for (r, row) in dense.iter().enumerate()
        {
            for (c, &v) in row.iter().enumerate()
            {
                if v != 0.0
                {
                    entries.push((r, c, v));
                }
            }
        }
        let (indptr, indices, data) = build_compressed(rows, entries);
        Ok(CsrMatrix {
            rows,
            cols,
            indptr,
            indices,
            data,
        })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of stored (explicit) nonzeros.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Row-pointer array of length `rows + 1`.
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Column-index array (one per stored entry).
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Stored values (one per stored entry).
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Value at `(r, c)`; out-of-range coordinates read as `0.0`.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        if r >= self.rows || c >= self.cols
        {
            return 0.0;
        }
        let start = self.indptr[r];
        let end = self.indptr[r + 1];
        for k in start..end
        {
            if self.indices[k] == c
            {
                return self.data[k];
            }
        }
        0.0
    }

    /// Sparse matrix–vector product `y = A · x`.
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `x.len() != cols`.
    pub fn spmv(&self, x: &[f64]) -> Result<Vec<f64>, SparseError> {
        if x.len() != self.cols
        {
            return Err(SparseError::DimensionMismatch {
                expected: self.cols,
                found: x.len(),
            });
        }
        let y: Vec<f64> = self
            .indptr
            .windows(2)
            .map(|w| {
                self.indices[w[0]..w[1]]
                    .iter()
                    .zip(&self.data[w[0]..w[1]])
                    .map(|(&c, &v)| v * x[c])
                    .sum()
            })
            .collect();
        Ok(y)
    }

    /// Materialize the matrix as a dense row-major `Vec<Vec<f64>>`.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.cols]; self.rows];
        for (r, row) in dense.iter_mut().enumerate()
        {
            let start = self.indptr[r];
            let end = self.indptr[r + 1];
            for k in start..end
            {
                row[self.indices[k]] = self.data[k];
            }
        }
        dense
    }

    /// Convert to compressed sparse column format.
    pub fn to_csc(&self) -> CscMatrix {
        let mut entries = Vec::with_capacity(self.data.len());
        for r in 0..self.rows
        {
            let start = self.indptr[r];
            let end = self.indptr[r + 1];
            for k in start..end
            {
                entries.push((self.indices[k], r, self.data[k]));
            }
        }
        let (indptr, indices, data) = build_compressed(self.cols, entries);
        CscMatrix {
            rows: self.rows,
            cols: self.cols,
            indptr,
            indices,
            data,
        }
    }

    /// Return the transpose as a new CSR matrix (shape `cols`×`rows`).
    pub fn transpose(&self) -> CsrMatrix {
        let mut entries = Vec::with_capacity(self.data.len());
        for r in 0..self.rows
        {
            let start = self.indptr[r];
            let end = self.indptr[r + 1];
            for k in start..end
            {
                entries.push((self.indices[k], r, self.data[k]));
            }
        }
        let (indptr, indices, data) = build_compressed(self.cols, entries);
        CsrMatrix {
            rows: self.cols,
            cols: self.rows,
            indptr,
            indices,
            data,
        }
    }
}

/// A sparse matrix in compressed sparse column (CSC) format.
///
/// Column `c` occupies `indices[indptr[c]..indptr[c + 1]]` (row indices, sorted
/// ascending) with parallel values in `data`.
#[derive(Debug, Clone)]
pub struct CscMatrix {
    rows: usize,
    cols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl CscMatrix {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of stored (explicit) nonzeros.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Column-pointer array of length `cols + 1`.
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Row-index array (one per stored entry).
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Stored values (one per stored entry).
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Value at `(r, c)`; out-of-range coordinates read as `0.0`.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        if r >= self.rows || c >= self.cols
        {
            return 0.0;
        }
        let start = self.indptr[c];
        let end = self.indptr[c + 1];
        for k in start..end
        {
            if self.indices[k] == r
            {
                return self.data[k];
            }
        }
        0.0
    }

    /// Materialize the matrix as a dense row-major `Vec<Vec<f64>>`.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.cols]; self.rows];
        for (c, w) in self.indptr.windows(2).enumerate()
        {
            for k in w[0]..w[1]
            {
                dense[self.indices[k]][c] = self.data[k];
            }
        }
        dense
    }

    /// Convert to compressed sparse row format.
    pub fn to_csr(&self) -> CsrMatrix {
        let mut entries = Vec::with_capacity(self.data.len());
        for c in 0..self.cols
        {
            let start = self.indptr[c];
            let end = self.indptr[c + 1];
            for k in start..end
            {
                entries.push((self.indices[k], c, self.data[k]));
            }
        }
        let (indptr, indices, data) = build_compressed(self.rows, entries);
        CsrMatrix {
            rows: self.rows,
            cols: self.cols,
            indptr,
            indices,
            data,
        }
    }

    /// Return the transpose as a new CSC matrix (shape `cols`×`rows`).
    pub fn transpose(&self) -> CscMatrix {
        let mut entries = Vec::with_capacity(self.data.len());
        for c in 0..self.cols
        {
            let start = self.indptr[c];
            let end = self.indptr[c + 1];
            for k in start..end
            {
                entries.push((self.indices[k], c, self.data[k]));
            }
        }
        let (indptr, indices, data) = build_compressed(self.rows, entries);
        CscMatrix {
            rows: self.cols,
            cols: self.rows,
            indptr,
            indices,
            data,
        }
    }
}

/// A sparse LU factorization `P·A = L·U` with partial pivoting.
///
/// Built with [`SparseLu::factor`] from a square [`CscMatrix`] using a
/// Gilbert–Peierls left-looking algorithm. `L` is unit lower triangular and `U`
/// is upper triangular; both are stored sparsely (in the pivot-permuted index
/// space). The factorization is computed once and can be reused across many
/// right-hand sides via [`SparseLu::solve`].
#[derive(Debug, Clone)]
pub struct SparseLu {
    n: usize,
    /// `perm[k]` is the original matrix row chosen as the `k`-th pivot.
    perm: Vec<usize>,
    /// Strictly-lower part of `L`, column `k` holding `(permuted_row, value)`
    /// pairs with `permuted_row > k` (the unit diagonal is implicit).
    l_cols: Vec<SparseCol>,
    /// Strictly-upper part of `U`, column `k` holding `(permuted_row, value)`
    /// pairs with `permuted_row < k`.
    u_cols: Vec<SparseCol>,
    /// Diagonal of `U`, one pivot per column.
    u_diag: Vec<f64>,
}

impl SparseLu {
    /// Factor a square sparse matrix into `P·A = L·U` with partial pivoting.
    ///
    /// Returns [`SparseError::NotSquare`] for a rectangular matrix and
    /// [`SparseError::Singular`] if a (numerically) zero pivot is encountered.
    pub fn factor(a: &CscMatrix) -> Result<SparseLu, SparseError> {
        if a.rows != a.cols
        {
            return Err(SparseError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        let n = a.rows;
        let max_abs = a.data.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
        let piv_tol = pivot_tol(n, max_abs);
        // pinv[i] == usize::MAX means original row `i` is not yet a pivot;
        // otherwise it holds that row's permuted pivot index.
        let mut pinv = vec![usize::MAX; n];
        let mut perm = vec![0usize; n];
        let mut l_cols: Vec<SparseCol> = vec![Vec::new(); n];
        let mut u_cols: Vec<SparseCol> = vec![Vec::new(); n];
        let mut u_diag = vec![0.0f64; n];
        let mut x = vec![0.0f64; n];

        for k in 0..n
        {
            // Scatter column k of A into the dense workspace.
            for v in x.iter_mut()
            {
                *v = 0.0;
            }
            let cs = a.indptr[k];
            let ce = a.indptr[k + 1];
            for t in cs..ce
            {
                x[a.indices[t]] = a.data[t];
            }

            // Forward-solve L x = A(:,k) using the columns computed so far,
            // processed in pivot order (each L column only touches rows that
            // are pivoted later, so plain forward substitution is exact).
            for j in 0..k
            {
                let pr = perm[j];
                let xj = x[pr];
                if xj != 0.0
                {
                    for &(i, lij) in &l_cols[j]
                    {
                        x[i] -= lij * xj;
                    }
                }
            }

            // The entries at already-pivoted rows form the upper part U(:,k).
            for j in 0..k
            {
                let val = x[perm[j]];
                if val != 0.0
                {
                    u_cols[k].push((j, val));
                }
            }

            // Partial pivoting: pick the largest-magnitude candidate among the
            // rows that have not yet been used as a pivot.
            let mut piv_row = usize::MAX;
            let mut piv_mag = 0.0;
            for i in 0..n
            {
                if pinv[i] == usize::MAX
                {
                    let mag = x[i].abs();
                    if mag > piv_mag
                    {
                        piv_mag = mag;
                        piv_row = i;
                    }
                }
            }
            if piv_row == usize::MAX || piv_mag <= piv_tol
            {
                return Err(SparseError::Singular);
            }

            let pivot = x[piv_row];
            perm[k] = piv_row;
            pinv[piv_row] = k;
            u_diag[k] = pivot;

            // Remaining non-pivotal rows give the lower part L(:,k) = x / pivot.
            // Rows are recorded by original index and remapped once all pivots
            // are known.
            for i in 0..n
            {
                if pinv[i] == usize::MAX && x[i] != 0.0
                {
                    l_cols[k].push((i, x[i] / pivot));
                }
            }
        }

        // Remap L's row indices from original to permuted (pivot) space.
        for col in l_cols.iter_mut()
        {
            for entry in col.iter_mut()
            {
                entry.0 = pinv[entry.0];
            }
        }

        Ok(SparseLu {
            n,
            perm,
            l_cols,
            u_cols,
            u_diag,
        })
    }

    /// Number of rows/columns of the factored matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Solve `A · x = b` using the stored factors.
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `b.len()` does not equal
    /// the matrix dimension.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SparseError> {
        if b.len() != self.n
        {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: b.len(),
            });
        }

        // Apply the row permutation: y = P b.
        let mut y: Vec<f64> = self.perm.iter().map(|&p| b[p]).collect();

        // Forward-solve L y = P b (L unit lower triangular).
        for k in 0..self.n
        {
            let yk = y[k];
            for &(row, lval) in &self.l_cols[k]
            {
                y[row] -= lval * yk;
            }
        }

        // Back-solve U x = y (column-oriented, high index first).
        let mut x = y;
        for k in (0..self.n).rev()
        {
            let xk = x[k] / self.u_diag[k];
            x[k] = xk;
            for &(row, uval) in &self.u_cols[k]
            {
                x[row] -= uval * xk;
            }
        }

        Ok(x)
    }
}

/// Solve a tridiagonal system `A · x = rhs` with the Thomas algorithm.
///
/// The matrix is described by three slices: `sub` (the `n − 1` sub-diagonal
/// entries, `sub[i]` being row `i + 1`'s lower coefficient), `diag` (the `n`
/// diagonal entries) and `sup` (the `n − 1` super-diagonal entries, `sup[i]`
/// being row `i`'s upper coefficient). `rhs` has length `n`.
///
/// Returns [`SparseError::DimensionMismatch`] if the slice lengths are
/// inconsistent and [`SparseError::ZeroPivot`] if elimination hits a zero
/// pivot.
///
/// # Example
///
/// ```
/// use scirust_sparse::solve_tridiagonal;
/// // 3x3 1-D Laplacian, right-hand side of ones.
/// let x = solve_tridiagonal(&[-1.0, -1.0], &[2.0, 2.0, 2.0], &[-1.0, -1.0], &[1.0, 1.0, 1.0]).unwrap();
/// assert!((x[0] - 1.5).abs() < 1e-12);
/// assert!((x[1] - 2.0).abs() < 1e-12);
/// assert!((x[2] - 1.5).abs() < 1e-12);
/// ```
pub fn solve_tridiagonal(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, SparseError> {
    let n = diag.len();
    if n == 0
    {
        if sub.is_empty() && sup.is_empty() && rhs.is_empty()
        {
            return Ok(Vec::new());
        }
        return Err(SparseError::DimensionMismatch {
            expected: 0,
            found: sub.len() + sup.len() + rhs.len(),
        });
    }
    if sub.len() != n - 1
    {
        return Err(SparseError::DimensionMismatch {
            expected: n - 1,
            found: sub.len(),
        });
    }
    if sup.len() != n - 1
    {
        return Err(SparseError::DimensionMismatch {
            expected: n - 1,
            found: sup.len(),
        });
    }
    if rhs.len() != n
    {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: rhs.len(),
        });
    }

    // Modified coefficients (c' and d' in the usual notation).
    let mut c_prime = vec![0.0f64; n];
    let mut d_prime = vec![0.0f64; n];

    let max_abs = sub
        .iter()
        .chain(diag.iter())
        .chain(sup.iter())
        .fold(0.0f64, |acc, &v| acc.max(v.abs()));
    let piv_tol = pivot_tol(n, max_abs);

    if diag[0].abs() <= piv_tol
    {
        return Err(SparseError::ZeroPivot { index: 0 });
    }
    if n > 1
    {
        c_prime[0] = sup[0] / diag[0];
    }
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n
    {
        let denom = diag[i] - sub[i - 1] * c_prime[i - 1];
        if denom.abs() <= piv_tol
        {
            return Err(SparseError::ZeroPivot { index: i });
        }
        if i < n - 1
        {
            c_prime[i] = sup[i] / denom;
        }
        d_prime[i] = (rhs[i] - sub[i - 1] * d_prime[i - 1]) / denom;
    }

    let mut x = vec![0.0f64; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev()
    {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

/// Inner product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve a symmetric positive-definite system `A · x = b` by conjugate
/// gradients.
///
/// `a` must be square; the matrix–vector products are computed with
/// [`CsrMatrix::spmv`]. Iteration stops when the residual norm `‖b − A x‖`
/// falls at or below `tol`. Returns [`SparseError::NotConverged`] if the budget
/// `max_iter` is exhausted first, [`SparseError::NotPositiveDefinite`] if a
/// non-positive curvature is detected, [`SparseError::NotSquare`] for a
/// rectangular matrix, or [`SparseError::DimensionMismatch`] if `b.len()`
/// disagrees with the matrix size.
pub fn conjugate_gradient(
    a: &CsrMatrix,
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, SparseError> {
    if a.rows != a.cols
    {
        return Err(SparseError::NotSquare {
            rows: a.rows,
            cols: a.cols,
        });
    }
    let n = a.rows;
    if b.len() != n
    {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    let mut x = vec![0.0f64; n];
    // r = b - A x, with x = 0 so r = b.
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);

    if rs_old.sqrt() <= tol
    {
        return Ok(x);
    }

    for _ in 0..max_iter
    {
        let ap = a.spmv(&p)?;
        let pap = dot(&p, &ap);
        if pap <= 0.0
        {
            return Err(SparseError::NotPositiveDefinite);
        }
        let alpha = rs_old / pap;
        for i in 0..n
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        if rs_new.sqrt() <= tol
        {
            return Ok(x);
        }
        let beta = rs_new / rs_old;
        for i in 0..n
        {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Err(SparseError::NotConverged { max_iter })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- deterministic pseudo-random generation (SplitMix64) ----

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit(state: &mut u64) -> f64 {
        (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_sym(state: &mut u64) -> f64 {
        next_unit(state) * 2.0 - 1.0
    }

    // ---- dense oracle: Gaussian elimination with partial pivoting ----

    fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut m: Vec<Vec<f64>> = a.to_vec();
        let mut rhs = b.to_vec();
        for k in 0..n
        {
            let mut piv = k;
            for i in (k + 1)..n
            {
                if m[i][k].abs() > m[piv][k].abs()
                {
                    piv = i;
                }
            }
            m.swap(k, piv);
            rhs.swap(k, piv);
            let d = m[k][k];
            assert!(d.abs() > 1e-14, "oracle hit a singular pivot");
            let pivot_row = m[k].clone();
            let rk = rhs[k];
            for i in (k + 1)..n
            {
                let f = m[i][k] / d;
                for j in k..n
                {
                    m[i][j] -= f * pivot_row[j];
                }
                rhs[i] -= f * rk;
            }
        }
        let mut x = vec![0.0; n];
        for k in (0..n).rev()
        {
            let mut s = rhs[k];
            for j in (k + 1)..n
            {
                s -= m[k][j] * x[j];
            }
            x[k] = s / m[k][k];
        }
        x
    }

    fn dense_matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x).map(|(v, xi)| v * xi).sum())
            .collect()
    }

    fn dense_to_csc(dense: &[Vec<f64>]) -> CscMatrix {
        CsrMatrix::from_dense(dense).unwrap().to_csc()
    }

    fn rand_spd(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut factor = vec![vec![0.0; n]; n];
        for row in factor.iter_mut()
        {
            for v in row.iter_mut()
            {
                *v = next_sym(&mut s);
            }
        }
        // A = FᵀF + n·I, symmetric positive-definite.
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n
        {
            for j in 0..n
            {
                let mut acc = 0.0;
                for frow in &factor
                {
                    acc += frow[i] * frow[j];
                }
                a[i][j] = acc;
            }
            a[i][i] += n as f64;
        }
        a
    }

    fn rand_ddom(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut a = vec![vec![0.0; n]; n];
        for row in a.iter_mut()
        {
            for v in row.iter_mut()
            {
                *v = next_sym(&mut s);
            }
        }
        for (i, row) in a.iter_mut().enumerate()
        {
            let mut off = 0.0;
            for (j, &val) in row.iter().enumerate()
            {
                if j != i
                {
                    off += val.abs();
                }
            }
            // Strictly diagonally dominant, hence nonsingular.
            row[i] = off + 1.0 + next_unit(&mut s);
        }
        a
    }

    fn laplacian(n: usize) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n
        {
            a[i][i] = 2.0;
            if i > 0
            {
                a[i][i - 1] = -1.0;
            }
            if i + 1 < n
            {
                a[i][i + 1] = -1.0;
            }
        }
        a
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b)
        {
            assert!((x - y).abs() <= tol, "‖{x} - {y}‖ exceeds {tol}");
        }
    }

    #[test]
    fn dense_csr_round_trip() {
        let dense = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 0.0, 3.0],
            vec![4.0, 5.0, 0.0],
        ];
        let csr = CsrMatrix::from_dense(&dense).unwrap();
        assert_eq!(csr.rows(), 3);
        assert_eq!(csr.cols(), 3);
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.to_dense(), dense);
        // get() agrees with the dense entries, including implicit zeros.
        for (r, row) in dense.iter().enumerate()
        {
            for (c, &v) in row.iter().enumerate()
            {
                assert_eq!(csr.get(r, c), v);
            }
        }
    }

    #[test]
    fn coo_duplicates_sum() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 1.0).unwrap();
        coo.push(0, 0, 4.0).unwrap();
        coo.push(1, 1, 2.0).unwrap();
        coo.push(1, 1, -0.5).unwrap();
        assert_eq!(coo.get(0, 0), 5.0);
        assert_eq!(coo.get(1, 1), 1.5);
        let csr = coo.to_csr();
        assert_eq!(csr.get(0, 0), 5.0);
        assert_eq!(csr.get(1, 1), 1.5);
        assert_eq!(csr.nnz(), 2);
        let csc = coo.to_csc();
        assert_eq!(csc.get(0, 0), 5.0);
        assert_eq!(csc.get(1, 1), 1.5);
    }

    #[test]
    fn transpose_twice_is_identity() {
        let dense = vec![vec![1.0, 2.0, 0.0], vec![0.0, 3.0, 4.0]];
        let csr = CsrMatrix::from_dense(&dense).unwrap();
        let tt = csr.transpose().transpose();
        assert_eq!(tt.rows(), csr.rows());
        assert_eq!(tt.cols(), csr.cols());
        assert_eq!(tt.to_dense(), dense);
        // The single transpose really is the mathematical transpose.
        let t = csr.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        for r in 0..2
        {
            for c in 0..3
            {
                assert_eq!(t.get(c, r), csr.get(r, c));
            }
        }
        // Same for CSC.
        let csc = dense_to_csc(&dense);
        assert_eq!(csc.transpose().transpose().to_dense(), dense);
    }

    #[test]
    fn csr_csc_conversions_agree() {
        let dense = vec![
            vec![0.0, 6.0, 0.0, 1.0],
            vec![7.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 8.0, 9.0],
        ];
        let csr = CsrMatrix::from_dense(&dense).unwrap();
        let csc = csr.to_csc();
        assert_eq!(csc.to_dense(), dense);
        assert_eq!(csc.to_csr().to_dense(), dense);
        // COO -> CSC path agrees too.
        let mut coo = CooMatrix::new(3, 4);
        for (r, row) in dense.iter().enumerate()
        {
            for (c, &v) in row.iter().enumerate()
            {
                if v != 0.0
                {
                    coo.push(r, c, v).unwrap();
                }
            }
        }
        assert_eq!(coo.to_csc().to_dense(), dense);
        assert_eq!(coo.to_dense(), dense);
    }

    #[test]
    fn spmv_hand_computed() {
        // A = [[1, 0, 2], [0, 3, 0], [4, 0, 5]], x = [1, 2, 3]
        // y = [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
        let dense = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 3.0, 0.0],
            vec![4.0, 0.0, 5.0],
        ];
        let csr = CsrMatrix::from_dense(&dense).unwrap();
        let y = csr.spmv(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(y, vec![7.0, 6.0, 19.0]);
    }

    #[test]
    fn sparse_lu_matches_oracle_general() {
        for (idx, &n) in [3usize, 4, 6, 8].iter().enumerate()
        {
            let a = rand_ddom(n, 0xABCD_0001 + idx as u64);
            let lu = SparseLu::factor(&dense_to_csc(&a)).unwrap();
            // Reuse the same factorization across several right-hand sides.
            let mut seed = 0x5151_0000 + idx as u64;
            for _ in 0..3
            {
                let b: Vec<f64> = (0..n).map(|_| next_sym(&mut seed)).collect();
                let got = lu.solve(&b).unwrap();
                let want = dense_solve(&a, &b);
                assert_close(&got, &want, 1e-9);
                // Residual check.
                let res = dense_matvec(&a, &got);
                assert_close(&res, &b, 1e-9);
            }
        }
    }

    #[test]
    fn sparse_lu_matches_oracle_spd() {
        for (idx, &n) in [2usize, 5, 7].iter().enumerate()
        {
            let a = rand_spd(n, 0x9999_0001 + idx as u64);
            let lu = SparseLu::factor(&dense_to_csc(&a)).unwrap();
            let mut seed = 0x2222_0000 + idx as u64;
            let b: Vec<f64> = (0..n).map(|_| next_sym(&mut seed)).collect();
            let got = lu.solve(&b).unwrap();
            let want = dense_solve(&a, &b);
            assert_close(&got, &want, 1e-9);
        }
    }

    #[test]
    fn conjugate_gradient_laplacian() {
        let n = 20;
        let dense = laplacian(n);
        let csr = CsrMatrix::from_dense(&dense).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 3) as f64 + 1.0).collect();
        let x = conjugate_gradient(&csr, &b, 1e-10, 1000).unwrap();
        let want = dense_solve(&dense, &b);
        assert_close(&x, &want, 1e-7);
        // Residual ‖A x - b‖ is small.
        let res = csr.spmv(&x).unwrap();
        let err: f64 = res
            .iter()
            .zip(&b)
            .map(|(a, bi)| (a - bi) * (a - bi))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-8, "residual {err} too large");
    }

    #[test]
    fn tridiagonal_matches_dense_laplacian() {
        let n = 12;
        let dense = laplacian(n);
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5) - 2.0).collect();
        let sub = vec![-1.0; n - 1];
        let sup = vec![-1.0; n - 1];
        let diag = vec![2.0; n];
        let got = solve_tridiagonal(&sub, &diag, &sup, &rhs).unwrap();
        let want = dense_solve(&dense, &rhs);
        assert_close(&got, &want, 1e-10);
    }

    #[test]
    fn sparse_lu_solves_a_tiny_scale_system_without_false_singularity() {
        // Regression test for a P1 audit finding: PIVOT_TOL was a fixed
        // absolute 1e-12 compared directly against the pivot magnitude — a
        // regular matrix at a small physical scale (e.g. capacitances in
        // Farads, ~1e-12) was declared singular even though it is perfectly
        // well-conditioned once rescaled.
        let scale = 1e-13;
        let n = 4;
        let a = rand_spd(n, 0x7777_0001)
            .into_iter()
            .map(|row| row.into_iter().map(|v| v * scale).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let lu = SparseLu::factor(&dense_to_csc(&a))
            .expect("a regular matrix at a tiny physical scale must not be reported singular");
        let mut seed = 0x8888_0001u64;
        let b: Vec<f64> = (0..n).map(|_| next_sym(&mut seed) * scale).collect();
        let got = lu.solve(&b).unwrap();
        let want = dense_solve(&a, &b);
        assert_close(&got, &want, 1e-6);
    }

    #[test]
    fn tridiagonal_solves_a_tiny_scale_system_without_false_zero_pivot() {
        // Same P1 finding, for solve_tridiagonal's Thomas algorithm.
        let n = 6;
        let scale = 1e-13;
        let dense = laplacian(n)
            .into_iter()
            .map(|row| row.into_iter().map(|v| v * scale).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let rhs: Vec<f64> = (0..n).map(|i| ((i as f64 * 0.5) - 1.0) * scale).collect();
        let sub = vec![-scale; n - 1];
        let sup = vec![-scale; n - 1];
        let diag = vec![2.0 * scale; n];
        let got = solve_tridiagonal(&sub, &diag, &sup, &rhs)
            .expect("a regular tridiagonal system at a tiny scale must not be reported singular");
        let want = dense_solve(&dense, &rhs);
        assert_close(&got, &want, 1e-6);
    }

    #[test]
    fn error_coo_out_of_range() {
        let mut coo = CooMatrix::new(2, 2);
        assert_eq!(
            coo.push(2, 0, 1.0),
            Err(SparseError::IndexOutOfBounds {
                row: 2,
                col: 0,
                rows: 2,
                cols: 2,
            })
        );
        assert!(coo.push(0, 5, 1.0).is_err());
    }

    #[test]
    fn error_spmv_dimension_mismatch() {
        let csr = CsrMatrix::from_dense(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let err = csr.spmv(&[1.0, 2.0, 3.0]).unwrap_err();
        assert_eq!(
            err,
            SparseError::DimensionMismatch {
                expected: 2,
                found: 3
            }
        );
    }

    #[test]
    fn error_singular_lu() {
        // Rows are linearly dependent -> singular.
        let dense = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let res = SparseLu::factor(&dense_to_csc(&dense));
        assert_eq!(res.err(), Some(SparseError::Singular));

        // A zero column is also singular.
        let dense2 = vec![vec![0.0, 1.0], vec![0.0, 3.0]];
        assert!(SparseLu::factor(&dense_to_csc(&dense2)).is_err());
    }

    #[test]
    fn error_not_square_lu() {
        let dense = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let csc = dense_to_csc(&dense);
        assert_eq!(
            SparseLu::factor(&csc).err(),
            Some(SparseError::NotSquare { rows: 2, cols: 3 })
        );
    }

    #[test]
    fn error_cg_not_converged() {
        let n = 10;
        let csr = CsrMatrix::from_dense(&laplacian(n)).unwrap();
        let b = vec![1.0; n];
        // One iteration cannot reach a 1e-15 tolerance on a 10x10 Laplacian.
        let res = conjugate_gradient(&csr, &b, 1e-15, 1);
        assert_eq!(res, Err(SparseError::NotConverged { max_iter: 1 }));
    }

    #[test]
    fn error_tridiagonal_zero_pivot() {
        // diag[0] = 0 gives an immediate zero pivot.
        let res = solve_tridiagonal(&[1.0], &[0.0, 2.0], &[1.0], &[1.0, 1.0]);
        assert_eq!(res, Err(SparseError::ZeroPivot { index: 0 }));
        // Length mismatch is rejected too.
        assert!(solve_tridiagonal(&[1.0, 2.0], &[1.0, 1.0], &[1.0], &[1.0, 1.0]).is_err());
    }

    #[test]
    fn error_display_is_nonempty() {
        let errs = [
            SparseError::IndexOutOfBounds {
                row: 1,
                col: 1,
                rows: 1,
                cols: 1,
            },
            SparseError::DimensionMismatch {
                expected: 1,
                found: 2,
            },
            SparseError::NotSquare { rows: 1, cols: 2 },
            SparseError::InconsistentRowLength {
                row: 1,
                expected: 2,
                found: 3,
            },
            SparseError::Singular,
            SparseError::ZeroPivot { index: 0 },
            SparseError::NotConverged { max_iter: 5 },
            SparseError::NotPositiveDefinite,
        ];
        for e in &errs
        {
            assert!(!format!("{e}").is_empty());
        }
    }
}
