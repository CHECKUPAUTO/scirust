//! Tensor-Train SVD decomposition (Oseledets 2011) and TT-Linear
//! decomposition (Novikov 2015 "Tensorizing Neural Networks").
//!
//! ## Layout conventions
//!
//! A TT decomposition of a `d`-mode tensor `T` of shape `(n_0, n_1, ..., n_{d-1})`
//! produces `d` cores `G_k` with logical shape `(r_k, n_k, r_{k+1})`, where
//! `r_0 = r_d = 1`. Cores are stored as [`TensorND`] of that shape (data is
//! row-major).
//!
//! For a `TTLinear` decomposition of a weight matrix `W` of shape `(in, out)`
//! with `in = ∏I_k` and `out = ∏O_k`, the matrix is first interleaved into
//! a tensor of shape `(I_0, O_0, I_1, O_1, ..., I_{d-1}, O_{d-1})`, then
//! each `(I_k, O_k)` pair is grouped into a single mode of size `I_k * O_k`,
//! and finally TT-SVD'd. Each core has logical shape `(r_k, I_k * O_k, r_{k+1})`.

use crate::tensor::TensorND;
use crate::tn::ops::svd::truncated_svd;

/// A complete TT decomposition: `d` cores plus their ranks.
///
/// `ranks` has length `d + 1` with `ranks[0] = ranks[d] = 1`.
/// `mode_dims[k]` is the size of mode `k` in the original tensor; for a TT-Linear
/// decomposition this is `I_k * O_k`.
pub struct TTCores {
    pub cores: Vec<TensorND>,
    pub ranks: Vec<usize>,
    pub mode_dims: Vec<usize>,
}

impl TTCores {
    /// Total number of parameters across all cores.
    pub fn num_params(&self) -> usize {
        self.cores.iter().map(|c| c.data.len()).sum()
    }

    /// Number of cores `d`.
    pub fn ndim(&self) -> usize {
        self.cores.len()
    }
}

/// Decompose a generic `d`-mode tensor into TT-cores via sequential SVD.
///
/// # Algorithm (Oseledets 2011)
/// ```text
/// C ← T reshape to (n_0, n_1 * ... * n_{d-1})
/// for k in 0..d-1:
///     M ← C reshape to (r_k * n_k, rest)
///     U, s, V^T ← truncated_svd(M, max_rank, tolerance)
///     r_{k+1} ← effective rank
///     core_k ← U reshape to (r_k, n_k, r_{k+1})
///     C ← diag(s) @ V^T
/// core_{d-1} ← C reshape to (r_{d-1}, n_{d-1}, 1)
/// ```
///
/// # Panics
/// - if `t.ndim() < 1`
/// - if any `mode_dim` is zero
pub fn tt_decompose_tensor(t: &TensorND, max_rank: usize, tolerance: f32) -> TTCores {
    let d = t.shape.len();
    assert!(d >= 1, "tt_decompose_tensor: need at least 1 mode");
    assert!(
        t.shape.iter().all(|&n| n > 0),
        "tt_decompose_tensor: zero mode"
    );

    let mode_dims = t.shape.clone();
    let mut ranks = vec![1usize; d + 1];
    let mut cores: Vec<TensorND> = Vec::with_capacity(d);

    // Edge case: single-mode tensor, just wrap as one core.
    if d == 1
    {
        let core = TensorND::new(t.data.clone(), vec![1, mode_dims[0], 1]);
        cores.push(core);
        return TTCores {
            cores,
            ranks,
            mode_dims,
        };
    }

    // Working buffer: starts as the full tensor in flat row-major.
    let mut work: Vec<f32> = t.data.clone();
    // After step k we'll have `work` interpreted as a (r_{k+1} * n_{k+1}, rest) matrix.
    // At step k, before SVD, `work` is a (r_k * n_k, rest_k) matrix where
    //   rest_k = ∏_{l>k} n_l.
    let mut rows = mode_dims[0]; // r_0 * n_0 = 1 * n_0

    for k in 0..d - 1
    {
        let cols = work.len() / rows;
        debug_assert_eq!(rows * cols, work.len(), "shape consistency");

        let svd = truncated_svd(&work, rows, cols, max_rank, tolerance);
        let r_next = svd.rank;
        ranks[k + 1] = r_next;

        // core_k has logical shape (r_k, n_k, r_{k+1}); data is exactly svd.u.
        let r_k = ranks[k];
        let n_k = mode_dims[k];
        debug_assert_eq!(rows, r_k * n_k);
        cores.push(TensorND::new(svd.u, vec![r_k, n_k, r_next]));

        // Residual = diag(s) @ V^T, of shape (r_{k+1}, cols)
        let mut residual = vec![0.0f32; r_next * cols];
        for kk in 0..r_next
        {
            let s_kk = svd.s[kk];
            for j in 0..cols
            {
                residual[kk * cols + j] = s_kk * svd.vt[kk * cols + j];
            }
        }

        // Prepare for next iteration: reshape residual to (r_{k+1} * n_{k+1}, rest).
        work = residual;
        rows = r_next * mode_dims[k + 1];
    }

    // Last core: `work` is now (r_{d-1}, n_{d-1}) — reshape to (r_{d-1}, n_{d-1}, 1).
    let r_last = ranks[d - 1];
    let n_last = mode_dims[d - 1];
    debug_assert_eq!(work.len(), r_last * n_last);
    cores.push(TensorND::new(work, vec![r_last, n_last, 1]));

    TTCores {
        cores,
        ranks,
        mode_dims,
    }
}

/// Reconstruct the dense tensor from a TT decomposition by sequentially
/// contracting cores from left to right. Returns a flat `Vec<f32>` of length
/// `∏ mode_dims` in row-major layout.
pub fn reconstruct_tensor(tt: &TTCores) -> Vec<f32> {
    let d = tt.cores.len();
    debug_assert!(d >= 1);

    if d == 1
    {
        return tt.cores[0].data.clone();
    }

    // Accumulator starts as cores[0] reshaped to (n_0, r_1).
    // Note: cores[0] has shape (1, n_0, r_1); flattened that's (n_0, r_1) row-major.
    let mut acc: Vec<f32> = tt.cores[0].data.clone();
    let mut acc_rows = tt.mode_dims[0];
    let mut acc_cols = tt.ranks[1];

    for k in 1..d
    {
        let r_k = tt.ranks[k];
        let n_k = tt.mode_dims[k];
        let r_next = tt.ranks[k + 1];
        let core = &tt.cores[k].data; // shape (r_k, n_k, r_{k+1}) flat
        debug_assert_eq!(acc_cols, r_k);

        // acc: (acc_rows, r_k), core viewed as (r_k, n_k * r_{k+1}) → matmul.
        let mut new_acc = vec![0.0f32; acc_rows * n_k * r_next];
        for i in 0..acc_rows
        {
            for nk in 0..n_k
            {
                for rn in 0..r_next
                {
                    let mut sum = 0.0f32;
                    for rk in 0..r_k
                    {
                        let acc_v = acc[i * r_k + rk];
                        let core_v = core[rk * (n_k * r_next) + nk * r_next + rn];
                        sum += acc_v * core_v;
                    }
                    new_acc[(i * n_k + nk) * r_next + rn] = sum;
                }
            }
        }
        acc = new_acc;
        acc_rows *= n_k;
        acc_cols = r_next;
    }

    debug_assert_eq!(acc_cols, 1);
    acc
}

// ---------------------------------------------------------------------------
// TT-Linear: decomposition of a weight matrix W of shape (in, out)
// ---------------------------------------------------------------------------

/// Re-arrange a weight matrix `W[i, j]` of shape `(in, out)` into the
/// interleaved tensor `T[i_0, j_0, i_1, j_1, ..., i_{d-1}, j_{d-1}]` of
/// shape `(I_0, O_0, I_1, O_1, ..., I_{d-1}, O_{d-1})`.
///
/// This is the key step of Novikov 2015: by interleaving in/out indices, the
/// resulting tensor has TT decomposition whose cores carry both an input mode
/// `I_k` and an output mode `O_k`, which is the natural structure for a linear
/// layer.
pub(crate) fn interleave_weight(w: &[f32], in_dims: &[usize], out_dims: &[usize]) -> TensorND {
    let in_features: usize = in_dims.iter().product();
    let out_features: usize = out_dims.iter().product();
    let total = in_features * out_features;
    assert_eq!(
        w.len(),
        total,
        "interleave_weight: weight matrix size mismatch"
    );
    let d = in_dims.len();
    assert_eq!(
        d,
        out_dims.len(),
        "interleave_weight: in/out dims length mismatch"
    );

    let mut target_shape = Vec::with_capacity(2 * d);
    for k in 0..d
    {
        target_shape.push(in_dims[k]);
        target_shape.push(out_dims[k]);
    }

    // Precompute strides for the target row-major layout.
    let mut target_strides = vec![1usize; 2 * d];
    for k in (0..2 * d - 1).rev()
    {
        target_strides[k] = target_strides[k + 1] * target_shape[k + 1];
    }

    // For each target flat index, compute (i_0, j_0, ..., i_{d-1}, j_{d-1}),
    // then map to source flat index (i * out + j).
    let mut t = vec![0.0f32; total];
    #[allow(clippy::needless_range_loop)]
    for target_flat in 0..total
    {
        let mut idx = vec![0usize; 2 * d];
        let mut rem = target_flat;
        for k in 0..2 * d
        {
            idx[k] = rem / target_strides[k];
            rem %= target_strides[k];
        }

        // Decode i from (i_0, i_1, ..., i_{d-1}) = (idx[0], idx[2], ..., idx[2d-2])
        let mut i = 0usize;
        let mut stride = in_features;
        for k in 0..d
        {
            stride /= in_dims[k];
            i += idx[2 * k] * stride;
        }

        // Decode j from (j_0, j_1, ..., j_{d-1}) = (idx[1], idx[3], ..., idx[2d-1])
        let mut j = 0usize;
        let mut stride = out_features;
        for k in 0..d
        {
            stride /= out_dims[k];
            j += idx[2 * k + 1] * stride;
        }

        t[target_flat] = w[i * out_features + j];
    }

    TensorND::new(t, target_shape)
}

/// Inverse of `interleave_weight`: from a tensor of shape
/// `(I_0, O_0, I_1, O_1, ..., I_{d-1}, O_{d-1})` (or its grouped form
/// `(I_0*O_0, I_1*O_1, ...)`), produce a row-major `(in, out)` matrix.
fn deinterleave_weight(t: &[f32], in_dims: &[usize], out_dims: &[usize]) -> Vec<f32> {
    let d = in_dims.len();
    let in_features: usize = in_dims.iter().product();
    let out_features: usize = out_dims.iter().product();
    let total = in_features * out_features;
    assert_eq!(t.len(), total, "deinterleave_weight: tensor size mismatch");

    let mut source_shape = Vec::with_capacity(2 * d);
    for k in 0..d
    {
        source_shape.push(in_dims[k]);
        source_shape.push(out_dims[k]);
    }
    let mut source_strides = vec![1usize; 2 * d];
    for k in (0..2 * d - 1).rev()
    {
        source_strides[k] = source_strides[k + 1] * source_shape[k + 1];
    }

    let mut w = vec![0.0f32; total];
    #[allow(clippy::needless_range_loop)]
    for source_flat in 0..total
    {
        let mut idx = vec![0usize; 2 * d];
        let mut rem = source_flat;
        for k in 0..2 * d
        {
            idx[k] = rem / source_strides[k];
            rem %= source_strides[k];
        }
        let mut i = 0usize;
        let mut stride = in_features;
        for k in 0..d
        {
            stride /= in_dims[k];
            i += idx[2 * k] * stride;
        }
        let mut j = 0usize;
        let mut stride = out_features;
        for k in 0..d
        {
            stride /= out_dims[k];
            j += idx[2 * k + 1] * stride;
        }
        w[i * out_features + j] = t[source_flat];
    }
    w
}

/// Decompose a weight matrix `W` of shape `(in, out)` into TT-Linear form.
///
/// - `w`: row-major flat data, length `in * out`.
/// - `in_dims`: factorization of `in = ∏ in_dims`.
/// - `out_dims`: factorization of `out = ∏ out_dims`, **same length as `in_dims`**.
/// - `max_rank`: cap on each bond dimension.
/// - `tolerance`: relative SVD truncation threshold.
///
/// # Returns
/// A [`TTCores`] where each core has logical shape `(r_k, I_k * O_k, r_{k+1})`.
/// `mode_dims[k] = in_dims[k] * out_dims[k]`.
pub fn tt_decompose_matrix(
    w: &[f32],
    in_dims: &[usize],
    out_dims: &[usize],
    max_rank: usize,
    tolerance: f32,
) -> TTCores {
    assert_eq!(
        in_dims.len(),
        out_dims.len(),
        "in_dims and out_dims must have the same length"
    );
    let interleaved = interleave_weight(w, in_dims, out_dims);

    // Group each (I_k, O_k) pair into a single mode of size I_k * O_k.
    let d = in_dims.len();
    let combined_shape: Vec<usize> = (0..d).map(|k| in_dims[k] * out_dims[k]).collect();
    let combined = interleaved
        .reshape(&combined_shape)
        .expect("reshape failed in tt_decompose_matrix");

    tt_decompose_tensor(&combined, max_rank, tolerance)
}

/// Reconstruct a row-major `(in, out)` weight matrix from a TT-Linear decomposition.
///
/// Inverse of [`tt_decompose_matrix`] up to truncation error.
pub fn reconstruct_matrix(tt: &TTCores, in_dims: &[usize], out_dims: &[usize]) -> Vec<f32> {
    let interleaved_flat = reconstruct_tensor(tt);
    deinterleave_weight(&interleaved_flat, in_dims, out_dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frob_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn frob_norm(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    // -----------------------------------------------------------------------
    // Interleave / de-interleave round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_interleave_roundtrip_d2() {
        // W is 6x4 = 24 elements, in_dims = [2, 3], out_dims = [2, 2]
        let w: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = interleave_weight(&w, &[2, 3], &[2, 2]);
        assert_eq!(t.shape, vec![2, 2, 3, 2]); // (I_0, O_0, I_1, O_1)
        let w_back = deinterleave_weight(&t.data, &[2, 3], &[2, 2]);
        assert_eq!(w_back, w);
    }

    #[test]
    fn test_interleave_roundtrip_d3() {
        // W is 12x8, in_dims = [2, 2, 3], out_dims = [2, 2, 2]
        let w: Vec<f32> = (0..96).map(|i| i as f32 * 0.1).collect();
        let t = interleave_weight(&w, &[2, 2, 3], &[2, 2, 2]);
        assert_eq!(t.shape, vec![2, 2, 2, 2, 3, 2]);
        let w_back = deinterleave_weight(&t.data, &[2, 2, 3], &[2, 2, 2]);
        assert_eq!(w_back, w);
    }

    // -----------------------------------------------------------------------
    // Generic TT-SVD on synthetic tensors
    // -----------------------------------------------------------------------

    #[test]
    fn test_tt_decompose_2mode_full_rank() {
        // 2-mode tensor = matrix; TT with full rank should be exact.
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let t = TensorND::new(data, vec![3, 4]);
        let tt = tt_decompose_tensor(&t, 100, 0.0);
        assert_eq!(tt.cores.len(), 2);
        let recon = reconstruct_tensor(&tt);
        assert!(frob_err(&t.data, &recon) < 1e-3);
    }

    #[test]
    fn test_tt_decompose_3mode_low_rank() {
        // Construct a tensor as outer product of 3 vectors → exact TT rank 1.
        let u: Vec<f32> = vec![1.0, 2.0, 3.0];
        let v: Vec<f32> = vec![1.0, -1.0, 0.5, 2.0];
        let w: Vec<f32> = vec![0.5, 1.5];
        let mut data = vec![0.0f32; 3 * 4 * 2];
        for i in 0..3
        {
            for j in 0..4
            {
                for k in 0..2
                {
                    data[(i * 4 + j) * 2 + k] = u[i] * v[j] * w[k];
                }
            }
        }
        let t = TensorND::new(data, vec![3, 4, 2]);
        let tt = tt_decompose_tensor(&t, 100, 1e-6);
        // Outer-product tensors are TT-rank 1.
        assert_eq!(tt.ranks, vec![1, 1, 1, 1]);
        let recon = reconstruct_tensor(&tt);
        assert!(frob_err(&t.data, &recon) < 1e-4);
    }

    #[test]
    fn test_tt_decompose_3mode_roundtrip() {
        // Random-ish 3-mode tensor, decompose with full rank, reconstruct exactly.
        let data: Vec<f32> = (0..2 * 3 * 4)
            .map(|i| ((i * 7 + 3) % 17) as f32 - 8.0)
            .collect();
        let t = TensorND::new(data, vec![2, 3, 4]);
        let tt = tt_decompose_tensor(&t, 100, 0.0);
        let recon = reconstruct_tensor(&tt);
        let err = frob_err(&t.data, &recon);
        assert!(err < 1e-3, "round-trip err = {err}");
    }

    // -----------------------------------------------------------------------
    // TT-Linear (matrix) decomposition
    // -----------------------------------------------------------------------

    #[test]
    fn test_tt_linear_decompose_roundtrip() {
        // W is 6x4, factor in=2*3, out=2*2
        let w: Vec<f32> = (0..24).map(|i| (i as f32).sin()).collect();
        let in_dims = vec![2, 3];
        let out_dims = vec![2, 2];
        let tt = tt_decompose_matrix(&w, &in_dims, &out_dims, 100, 0.0);
        let w_back = reconstruct_matrix(&tt, &in_dims, &out_dims);
        let err = frob_err(&w, &w_back);
        let rel = err / frob_norm(&w).max(1e-30);
        assert!(rel < 1e-3, "rel err = {rel}");
    }

    #[test]
    fn test_tt_linear_compression() {
        // Synthetic low-rank-ish matrix: 16x16 = outer product + small noise
        let n_in = 16;
        let n_out = 16;
        let u: Vec<f32> = (0..n_in).map(|i| (i as f32).sin()).collect();
        let v: Vec<f32> = (0..n_out).map(|j| (j as f32).cos()).collect();
        let mut w = vec![0.0f32; n_in * n_out];
        for i in 0..n_in
        {
            for j in 0..n_out
            {
                w[i * n_out + j] = u[i] * v[j] + 0.01 * ((i + j) as f32).sin();
            }
        }
        let in_dims = vec![4, 4];
        let out_dims = vec![4, 4];
        let tt = tt_decompose_matrix(&w, &in_dims, &out_dims, 8, 1e-3);
        // Outer-product structure should be captured by very small ranks
        assert!(tt.ranks.iter().all(|&r| r <= 8));
        let w_back = reconstruct_matrix(&tt, &in_dims, &out_dims);
        let rel = frob_err(&w, &w_back) / frob_norm(&w).max(1e-30);
        // Allow up to 5% error since we forced max_rank
        assert!(rel < 0.05, "rel err = {rel}");
    }

    #[test]
    fn test_num_params() {
        let w: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tt = tt_decompose_matrix(&w, &[2, 3], &[2, 2], 100, 0.0);
        // With full rank no compression; cores collectively store the full info
        assert!(tt.num_params() >= 24);
    }
}
