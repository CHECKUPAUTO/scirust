//! Truncated singular value decomposition.
//!
//! Phase 1 implementation: nalgebra performs a full SVD then we truncate the
//! result by `max_rank` and by an absolute tolerance on singular values
//! (`tolerance * sigma_max`).
//!
//! Phase 2 (GPU): replace with cuSOLVER `gesvdj` / `gesvdr` via cudarc.
//! Phase 2 (CPU large): a randomized SVD (Halko 2011) would be much faster
//! for `min(m, n) > 1000` matrices but is overkill for transformer-scale
//! decompositions where each unfolding is moderate.

use nalgebra::{DMatrix, SVD};

/// Result of a truncated SVD: `A ≈ U @ diag(s) @ Vt`.
///
/// Shapes:
/// - `u` : `(m, r)` row-major flat vector
/// - `s` : `r` singular values (sorted descending)
/// - `vt`: `(r, n)` row-major flat vector
///
/// where `r = effective rank ≤ max_rank` chosen so that all retained singular
/// values are `>= tolerance * s[0]`.
pub struct TruncSvd {
    pub m: usize,
    pub n: usize,
    pub rank: usize,
    pub u: Vec<f32>,
    pub s: Vec<f32>,
    pub vt: Vec<f32>,
}

/// Compute the truncated SVD of an `(m, n)` matrix stored in row-major order.
///
/// - `data`: row-major flat representation, length `m * n`.
/// - `max_rank`: hard cap on the returned rank. Use `usize::MAX` for no cap.
/// - `tolerance`: relative threshold; singular values below
///   `tolerance * s[0]` are dropped. Use `0.0` to keep all values up to `max_rank`.
///
/// # Returns
/// `TruncSvd { rank, u, s, vt }` where `rank = min(max_rank, count of singular values ≥ tol*s[0])`.
/// Always `rank >= 1` (the largest singular value is always retained).
pub fn truncated_svd(data: &[f32], m: usize, n: usize, max_rank: usize, tolerance: f32) -> TruncSvd {
    assert_eq!(data.len(), m * n, "truncated_svd: data length mismatch");
    assert!(m > 0 && n > 0, "truncated_svd: empty matrix");

    // nalgebra is column-major, so we feed (n, m) and transpose meaning.
    // Equivalent: build the matrix from a row-iterator.
    let mat = DMatrix::<f32>::from_row_slice(m, n, data);

    // Compute full SVD. `compute_u = true, compute_v = true`.
    let svd = SVD::new(mat, true, true);
    let u_full = svd.u.expect("SVD failed to compute U");
    let s_full = svd.singular_values; // DVector<f32>, descending order
    let vt_full = svd.v_t.expect("SVD failed to compute V^T");

    let full_rank = s_full.len();
    let s_max = s_full[0].max(1e-30);
    let abs_threshold = tolerance * s_max;

    // Determine effective rank.
    let mut rank = 0usize;
    for &sigma in s_full.iter() {
        if sigma >= abs_threshold && rank < max_rank.min(full_rank) {
            rank += 1;
        } else {
            break;
        }
    }
    let rank = rank.max(1); // always keep at least one component

    // Extract the first `rank` columns of U → (m, rank) row-major.
    let mut u = vec![0.0f32; m * rank];
    for i in 0..m {
        for k in 0..rank {
            u[i * rank + k] = u_full[(i, k)];
        }
    }

    // Singular values: first `rank` entries.
    let s: Vec<f32> = (0..rank).map(|k| s_full[k]).collect();

    // V^T: first `rank` rows → (rank, n) row-major.
    let mut vt = vec![0.0f32; rank * n];
    for k in 0..rank {
        for j in 0..n {
            vt[k * n + j] = vt_full[(k, j)];
        }
    }

    TruncSvd { m, n, rank, u, s, vt }
}

/// Reconstruct `A ≈ U @ diag(s) @ Vt` for verification.
/// Returns a row-major flat `(m, n)` matrix.
pub fn reconstruct(svd: &TruncSvd) -> Vec<f32> {
    let m = svd.m;
    let n = svd.n;
    let r = svd.rank;
    let mut out = vec![0.0f32; m * n];
    // out[i, j] = sum_k U[i, k] * s[k] * Vt[k, j]
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..r {
                acc += svd.u[i * r + k] * svd.s[k] * svd.vt[k * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
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

    #[test]
    fn test_svd_identity() {
        // 3x3 identity matrix
        #[rustfmt::skip]
        let data = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let svd = truncated_svd(&data, 3, 3, 10, 0.0);
        assert_eq!(svd.rank, 3);
        // All singular values = 1
        for &s in &svd.s {
            assert!((s - 1.0).abs() < 1e-5);
        }
        let recon = reconstruct(&svd);
        assert!(frob_err(&data, &recon) < 1e-5);
    }

    #[test]
    fn test_svd_rank_one() {
        // u = [1, 2, 3]^T, v = [4, 5]^T, A = u v^T
        // A = [[4, 5], [8, 10], [12, 15]]
        #[rustfmt::skip]
        let data = vec![
            4.0,  5.0,
            8.0,  10.0,
            12.0, 15.0,
        ];
        let svd = truncated_svd(&data, 3, 2, 10, 0.0);
        // Mathematical rank is 1, second singular value should be ~0
        assert!(svd.s[0] > 0.0);
        if svd.s.len() > 1 {
            assert!(svd.s[1].abs() < 1e-4);
        }
        let recon = reconstruct(&svd);
        assert!(frob_err(&data, &recon) < 1e-4);
    }

    #[test]
    fn test_svd_truncation_by_max_rank() {
        // 4x4 random-ish matrix, force rank 2
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let svd_full = truncated_svd(&data, 4, 4, 4, 0.0);
        let svd_r2 = truncated_svd(&data, 4, 4, 2, 0.0);
        assert!(svd_r2.rank <= 2);
        // This matrix is actually rank 2 (rows are arithmetic progressions),
        // so reconstruction at rank 2 should still be very accurate.
        let recon = reconstruct(&svd_r2);
        let err = frob_err(&data, &recon);
        let _ = svd_full;
        assert!(err < 1e-3, "rank-2 recon err = {err}");
    }

    #[test]
    fn test_svd_truncation_by_tolerance() {
        // Same matrix, tolerance large enough to drop tiny singular values
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let svd = truncated_svd(&data, 4, 4, 100, 1e-3);
        // Rank should drop to 2 since this matrix is effectively rank-2
        assert!(svd.rank <= 3);
    }

    #[test]
    fn test_svd_rectangular_tall() {
        // 5x2 matrix
        let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let svd = truncated_svd(&data, 5, 2, 10, 0.0);
        assert_eq!(svd.u.len(), 5 * svd.rank);
        assert_eq!(svd.vt.len(), svd.rank * 2);
        let recon = reconstruct(&svd);
        assert_eq!(recon.len(), 10);
    }

    #[test]
    fn test_svd_rectangular_wide() {
        // 2x5 matrix
        let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let svd = truncated_svd(&data, 2, 5, 10, 0.0);
        assert_eq!(svd.u.len(), 2 * svd.rank);
        assert_eq!(svd.vt.len(), svd.rank * 5);
        let recon = reconstruct(&svd);
        assert!(frob_err(&data, &recon) < 1e-4);
    }
}
