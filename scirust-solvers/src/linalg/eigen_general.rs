//! Eigenvalues of a general (non-symmetric) dense real matrix.
//!
//! [`super::eigen::eigen_symmetric`] only handles the symmetric case (real
//! eigenvalues, orthogonal eigenvectors via tridiagonalization + implicit
//! QL). A general matrix has no such guarantee — eigenvalues can be complex
//! (always in conjugate pairs, since the characteristic polynomial has real
//! coefficients) — so this module targets eigenvalues only, via the
//! classical two-stage approach (Golub & Van Loan, *Matrix Computations*,
//! 4th ed., §7.4–7.5):
//!
//! 1. **Hessenberg reduction**: `A = Q·H·Qᵀ` via Householder reflections,
//!    `H` upper Hessenberg (`H[i,j] = 0` for `i > j+1`) — a similarity
//!    transform, so `H` and `A` share eigenvalues.
//! 2. **Double-shift QR iteration** on `H`, deflating from the bottom-right
//!    as subdiagonal entries become negligible: a `1×1` deflated block is a
//!    real eigenvalue directly, a `2×2` block is solved via the numerically
//!    stable quadratic formula (`q = -½(B + sign(B)·√disc)`, then `q` and
//!    `C/q` — the same cancellation-avoidance fix applied platform-wide in
//!    the Chantier 2 audit pass), yielding either two real eigenvalues or a
//!    complex-conjugate pair.
//!
//! This implementation uses the **explicit** double-shift step — forming
//! `M = H² − (s₁+s₂)·H + (s₁·s₂)·I` and taking `H ← QᵀHQ` from `M = QR` —
//! rather than the implicit bulge-chasing Francis algorithm real production
//! solvers (LAPACK's `dhseqr`) use. `s₁+s₂` and `s₁·s₂` stay real even when
//! `s₁, s₂` are a complex-conjugate shift pair, so the whole iteration stays
//! in real arithmetic without needing a complex matrix type; the tradeoff is
//! `O(k³)` per step on the active `k×k` block instead of `O(k²)`, which is
//! immaterial at the matrix sizes this crate targets (control systems,
//! modal analysis) and considerably simpler to get right than bulge-chasing.
//!
//! ## Determinism & safety
//! - Deflation tolerance `MACHINE_EPS · (|h[l,l]| + |h[l+1,l+1]|)`, the same
//!   convention `eigen_symmetric` uses.
//! - A fixed total iteration budget (`n · MAX_ITERS_PER_EIGENVALUE`); an
//!   "exceptional shift" (Wilkinson's ad hoc perturbation) kicks in every 10
//!   iterations without a deflation, to escape the rare shift cases that
//!   stagnate the plain double shift (Golub & Van Loan §7.5.2).
//! - `NaN`/`Inf` anywhere in the input or an intermediate result is rejected.

use crate::linalg::{Matrix, qr_decompose};
use crate::{SolverError, SolverResult};

const MACHINE_EPS: f64 = 2.220446049250313e-16;
const MAX_ITERS_PER_EIGENVALUE: usize = 60;

/// A single eigenvalue of a real matrix; `im == 0.0` for a real eigenvalue.
/// Complex eigenvalues of a real matrix always come in conjugate pairs, so
/// they appear adjacent in [`eigenvalues_general`]'s output as `(re, im)`
/// and `(re, -im)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eigenvalue {
    pub re: f64,
    pub im: f64,
}

/// Eigenvalues of the 2×2 matrix `[[a, b], [c, d]]`, via the numerically
/// stable quadratic formula (avoids the catastrophic cancellation the naive
/// `(-b ± √disc)/2a` form suffers when `disc` is close to `b²` — the same
/// fix applied platform-wide in the Chantier 2 audit pass).
fn eig_2x2(a: f64, b: f64, c: f64, d: f64) -> (Eigenvalue, Eigenvalue) {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0
    {
        let sqrt_disc = disc.sqrt();
        // Characteristic polynomial lambda^2 - tr*lambda + det = 0, i.e.
        // coefficients (1, -tr, det); q = -1/2*(B + sign(B)*sqrt(disc)).
        let b_coef = -tr;
        let sign = if b_coef >= 0.0 { 1.0 } else { -1.0 };
        let q = -0.5 * (b_coef + sign * sqrt_disc);
        if q == 0.0
        {
            // tr == 0 and det == 0 forces both roots to 0.
            return (
                Eigenvalue { re: 0.0, im: 0.0 },
                Eigenvalue { re: 0.0, im: 0.0 },
            );
        }
        let l1 = q;
        let l2 = det / q;
        (
            Eigenvalue { re: l1, im: 0.0 },
            Eigenvalue { re: l2, im: 0.0 },
        )
    }
    else
    {
        let re = tr / 2.0;
        let im = (-disc).sqrt() / 2.0;
        (Eigenvalue { re, im }, Eigenvalue { re, im: -im })
    }
}

fn check_finite_matrix(m: &Matrix) -> SolverResult<()> {
    for &v in m.data()
    {
        if !v.is_finite()
        {
            return Err(SolverError::NanDetected { iter: 0, value: v });
        }
    }
    Ok(())
}

/// Reduce `a` to upper Hessenberg form in place via a similarity transform
/// (Householder reflections applied from both sides) — Golub & Van Loan
/// §7.4.2. Only the Hessenberg matrix is needed here (eigenvalues, not
/// eigenvectors), so the accumulated orthogonal transform is discarded.
fn hessenberg_reduce(a: &mut Matrix, n: usize) {
    if n < 3
    {
        return;
    }
    let mut v = vec![0.0; n];
    for k in 0..(n - 2)
    {
        let mut norm_sq = 0.0;
        for i in (k + 1)..n
        {
            norm_sq += a[(i, k)] * a[(i, k)];
        }
        if norm_sq == 0.0
        {
            continue;
        }
        let mut alpha = norm_sq.sqrt();
        if a[(k + 1, k)] > 0.0
        {
            alpha = -alpha;
        }
        v[k + 1] = a[(k + 1, k)] - alpha;
        for i in (k + 2)..n
        {
            v[i] = a[(i, k)];
        }
        let v_norm_sq: f64 = v[(k + 1)..n].iter().map(|x| x * x).sum();
        if v_norm_sq == 0.0
        {
            continue;
        }

        // Left: A <- (I - 2vv^T/v^Tv) A, restricted to the affected rows.
        for j in 0..n
        {
            let mut dot = 0.0;
            for i in (k + 1)..n
            {
                dot += v[i] * a[(i, j)];
            }
            let factor = 2.0 * dot / v_norm_sq;
            if factor == 0.0
            {
                continue;
            }
            for i in (k + 1)..n
            {
                a[(i, j)] -= factor * v[i];
            }
        }
        // Right: A <- A (I - 2vv^T/v^Tv), restricted to the affected columns.
        for i in 0..n
        {
            let mut dot = 0.0;
            for j in (k + 1)..n
            {
                dot += a[(i, j)] * v[j];
            }
            let factor = 2.0 * dot / v_norm_sq;
            if factor == 0.0
            {
                continue;
            }
            for j in (k + 1)..n
            {
                a[(i, j)] -= factor * v[j];
            }
        }
    }
}

/// Extract the `k×k` active block `h[l..l+k, l..l+k]`.
fn extract_block(h: &Matrix, l: usize, k: usize) -> Matrix {
    Matrix::from_fn(k, k, |i, j| h[(l + i, l + j)])
}

fn write_block(h: &mut Matrix, l: usize, k: usize, block: &Matrix) {
    for i in 0..k
    {
        for j in 0..k
        {
            h[(l + i, l + j)] = block[(i, j)];
        }
    }
}

/// Eigenvalues of a general real square matrix `A`, via Hessenberg reduction
/// followed by explicit double-shift QR iteration. Returns an error on a
/// non-square or non-finite input, or if the iteration budget is exhausted
/// before every block has deflated (a genuine non-convergence — not
/// expected for well-scaled inputs, but possible for pathological/highly
/// non-normal matrices).
pub fn eigenvalues_general(a: &Matrix) -> SolverResult<Vec<Eigenvalue>> {
    let n = a.ensure_square()?;
    if n == 0
    {
        return Err(SolverError::InvalidInput(
            "eigenvalues_general: empty matrix".to_string(),
        ));
    }
    check_finite_matrix(a)?;

    let mut h = a.clone();
    hessenberg_reduce(&mut h, n);

    let mut eigenvalues = Vec::with_capacity(n);
    let mut hi = n; // active submatrix is h[0..hi, 0..hi]
    let mut iters_since_deflation = 0usize;
    let mut total_iters = 0usize;
    let iter_budget = n * MAX_ITERS_PER_EIGENVALUE;

    while hi > 0
    {
        if hi == 1
        {
            eigenvalues.push(Eigenvalue {
                re: h[(0, 0)],
                im: 0.0,
            });
            break;
        }

        // Find the largest l such that h[l, l-1] is negligible (or l == 0),
        // i.e. the start of the trailing unreduced (still-coupled) block.
        let mut l = hi - 1;
        while l > 0
        {
            let scale = h[(l - 1, l - 1)].abs() + h[(l, l)].abs();
            if h[(l, l - 1)].abs() <= MACHINE_EPS * scale.max(1e-300)
            {
                h[(l, l - 1)] = 0.0;
                break;
            }
            l -= 1;
        }

        if l == hi - 1
        {
            eigenvalues.push(Eigenvalue {
                re: h[(hi - 1, hi - 1)],
                im: 0.0,
            });
            hi -= 1;
            iters_since_deflation = 0;
            continue;
        }
        if l == hi - 2
        {
            let (e1, e2) = eig_2x2(h[(l, l)], h[(l, l + 1)], h[(l + 1, l)], h[(l + 1, l + 1)]);
            eigenvalues.push(e1);
            eigenvalues.push(e2);
            hi -= 2;
            iters_since_deflation = 0;
            continue;
        }

        total_iters += 1;
        iters_since_deflation += 1;
        if total_iters > iter_budget
        {
            return Err(SolverError::NoConvergence {
                iterations: iter_budget,
                residual: h[(hi - 1, hi - 2)].abs(),
            });
        }

        let k = hi - l;
        let mut active = extract_block(&h, l, k);

        // Shift: eigenvalues of the trailing 2x2 corner of the active
        // block, except every 10 stagnating iterations, where an "ad hoc"
        // exceptional shift (Wilkinson's classic escape hatch) is used
        // instead (Golub & Van Loan §7.5.2).
        let (s1, s2) = if iters_since_deflation > 0 && iters_since_deflation.is_multiple_of(10)
        {
            let exceptional = active[(k - 1, k - 1)].abs() + 0.75 * active[(k - 1, k - 2)].abs();
            (
                Eigenvalue {
                    re: exceptional,
                    im: 0.0,
                },
                Eigenvalue {
                    re: exceptional,
                    im: 0.0,
                },
            )
        }
        else
        {
            eig_2x2(
                active[(k - 2, k - 2)],
                active[(k - 2, k - 1)],
                active[(k - 1, k - 2)],
                active[(k - 1, k - 1)],
            )
        };
        let sum_s = s1.re + s2.re; // always real: s1,s2 real or conjugate.
        let prod_s = s1.re * s2.re - s1.im * s2.im; // real part of s1*s2 (= |s|^2 if conjugate).

        // M = active^2 - sum_s*active + prod_s*I.
        let sq = active.matmul(&active).map_err(|_| {
            SolverError::InvalidInput("eigenvalues_general: internal shape mismatch".to_string())
        })?;
        let mut m = Matrix::zeros(k, k);
        for i in 0..k
        {
            for j in 0..k
            {
                let mut v = sq[(i, j)] - sum_s * active[(i, j)];
                if i == j
                {
                    v += prod_s;
                }
                m[(i, j)] = v;
            }
        }
        check_finite_matrix(&m)?;

        let qr = qr_decompose(m)?;
        let q = qr.q();
        let qt = q.transpose();
        // active <- Q^T * active * Q (similarity transform).
        active = qt.matmul(&active).and_then(|t| t.matmul(&q)).map_err(|_| {
            SolverError::InvalidInput("eigenvalues_general: internal shape mismatch".to_string())
        })?;
        check_finite_matrix(&active)?;
        write_block(&mut h, l, k, &active);
    }

    // Deterministic output order, independent of deflation order.
    eigenvalues.sort_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap()
            .then(a.im.partial_cmp(&b.im).unwrap())
    });
    Ok(eigenvalues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sorted_reals(eigs: &[Eigenvalue]) -> Vec<f64> {
        let mut v: Vec<f64> = eigs.iter().map(|e| e.re).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    #[test]
    fn diagonal_matrix_returns_its_diagonal() {
        let a = Matrix::from_row_major(3, 3, vec![5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0]);
        let eigs = eigenvalues_general(&a).unwrap();
        assert_eq!(sorted_reals(&eigs), vec![1.0, 3.0, 5.0]);
        for e in &eigs
        {
            assert_eq!(e.im, 0.0);
        }
    }

    #[test]
    fn triangular_matrix_returns_its_diagonal() {
        // Upper triangular: eigenvalues are the diagonal, no iteration needed
        // to reveal them but exercises the general path end-to-end.
        let a = Matrix::from_row_major(3, 3, vec![2.0, 5.0, -1.0, 0.0, -3.0, 4.0, 0.0, 0.0, 7.0]);
        let eigs = eigenvalues_general(&a).unwrap();
        assert_eq!(sorted_reals(&eigs), vec![-3.0, 2.0, 7.0]);
    }

    #[test]
    fn rotation_matrix_has_purely_imaginary_ratio_eigenvalues() {
        // A 2D rotation by angle theta has eigenvalues cos(theta) +/- i*sin(theta).
        let theta = 0.7_f64;
        let a = Matrix::from_row_major(
            2,
            2,
            vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()],
        );
        let mut eigs = eigenvalues_general(&a).unwrap();
        eigs.sort_by(|x, y| x.im.partial_cmp(&y.im).unwrap());
        assert_relative_eq!(eigs[0].re, theta.cos(), epsilon = 1e-9);
        assert_relative_eq!(eigs[0].im, -theta.sin(), epsilon = 1e-9);
        assert_relative_eq!(eigs[1].re, theta.cos(), epsilon = 1e-9);
        assert_relative_eq!(eigs[1].im, theta.sin(), epsilon = 1e-9);
    }

    #[test]
    fn asymmetric_matrix_matches_numpy_reference() {
        // numpy.linalg.eigvals([[4,1,-2],[1,3,1],[2,-1,5]])
        // -> [3.677814645373913+0j, 4.161092677313042+1.7543809597837206j,
        //     4.161092677313042-1.7543809597837206j]
        // (one real eigenvalue, and a complex-conjugate pair).
        let a = Matrix::from_row_major(3, 3, vec![4.0, 1.0, -2.0, 1.0, 3.0, 1.0, 2.0, -1.0, 5.0]);
        let mut eigs = eigenvalues_general(&a).unwrap();
        eigs.sort_by(|x, y| x.re.partial_cmp(&y.re).unwrap());
        assert_relative_eq!(eigs[0].re, 3.677_814_645_373_913, epsilon = 1e-6);
        assert_relative_eq!(eigs[0].im, 0.0, epsilon = 1e-6);
        assert_relative_eq!(eigs[1].re, 4.161_092_677_313_042, epsilon = 1e-6);
        assert_relative_eq!(eigs[1].im.abs(), 1.754_380_959_783_720_6, epsilon = 1e-6);
        assert_relative_eq!(eigs[2].re, 4.161_092_677_313_042, epsilon = 1e-6);
        assert_relative_eq!(eigs[2].im.abs(), 1.754_380_959_783_720_6, epsilon = 1e-6);
        assert_relative_eq!(eigs[1].im, -eigs[2].im, epsilon = 1e-9);
    }

    #[test]
    fn symmetric_matrix_matches_eigen_symmetric() {
        // Cross-check against the dedicated symmetric solver on a matrix
        // both can handle.
        let a = Matrix::from_row_major(3, 3, vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let general = eigenvalues_general(&a).unwrap();
        let symmetric = super::super::eigen::eigen_symmetric(&a).unwrap();
        let mut general_re = sorted_reals(&general);
        general_re.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut symmetric_vals = symmetric.eigenvalues.clone();
        symmetric_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (g, s) in general_re.iter().zip(&symmetric_vals)
        {
            assert_relative_eq!(*g, *s, epsilon = 1e-7);
        }
        for e in &general
        {
            assert_relative_eq!(e.im, 0.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn rejects_non_square_matrix() {
        let a = Matrix::from_row_major(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(eigenvalues_general(&a).is_err());
    }

    #[test]
    fn trace_and_determinant_are_consistent_for_a_larger_random_looking_matrix() {
        // Independent cross-check that doesn't go through the eigensolver
        // at all: trace(A) = sum(eigenvalues), det(A) = prod(eigenvalues) —
        // true for any matrix, real or complex eigenvalues alike.
        let a = Matrix::from_row_major(
            4,
            4,
            vec![
                2.0, 1.0, 0.0, 3.0, -1.0, 4.0, 2.0, 0.0, 0.5, -2.0, 3.0, 1.0, 1.0, 1.0, -1.0, 2.0,
            ],
        );
        let eigs = eigenvalues_general(&a).unwrap();
        let trace: f64 = (0..4).map(|i| a[(i, i)]).sum();
        let sum_re: f64 = eigs.iter().map(|e| e.re).sum();
        let sum_im: f64 = eigs.iter().map(|e| e.im).sum();
        assert_relative_eq!(sum_re, trace, epsilon = 1e-6);
        assert_relative_eq!(sum_im, 0.0, epsilon = 1e-6);

        let det = a.determinant().unwrap();
        // Product of complex eigenvalues, tracked as a running complex product.
        let (mut pre, mut pim) = (1.0, 0.0);
        for e in &eigs
        {
            let (nre, nim) = (pre * e.re - pim * e.im, pre * e.im + pim * e.re);
            pre = nre;
            pim = nim;
        }
        assert_relative_eq!(pre, det, epsilon = 1e-4, max_relative = 1e-4);
        assert_relative_eq!(pim, 0.0, epsilon = 1e-4);
    }
}

/// Property-based tests: trace/determinant identities checked over many
/// randomly generated matrices, independent of the eigensolver itself (both
/// are computed directly from `A`), so unlike a round-trip through the
/// solver's own machinery, a wrong eigenvalue set would actually be caught.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    proptest! {
        #[test]
        fn trace_equals_sum_and_det_equals_product_of_eigenvalues(
            raw in prop::collection::vec(-5.0f64..5.0, 16), // 4x4
        ) {
            let n = 4;
            let a = Matrix::from_row_major(n, n, raw);
            let eigs = eigenvalues_general(&a).expect("a random 4x4 matrix should converge");
            prop_assert_eq!(eigs.len(), n);

            let trace: f64 = (0..n).map(|i| a[(i, i)]).sum();
            let sum_re: f64 = eigs.iter().map(|e| e.re).sum();
            let sum_im: f64 = eigs.iter().map(|e| e.im).sum();
            prop_assert!(rel_close(sum_re, trace, 1e-5), "sum_re={sum_re} trace={trace}");
            prop_assert!(sum_im.abs() < 1e-5, "sum_im={sum_im}");

            let det = a.determinant().unwrap();
            let (mut pre, mut pim) = (1.0, 0.0);
            for e in &eigs {
                let (nre, nim) = (pre * e.re - pim * e.im, pre * e.im + pim * e.re);
                pre = nre;
                pim = nim;
            }
            prop_assert!(rel_close(pre, det, 1e-4), "prod_re={pre} det={det}");
            prop_assert!(pim.abs() < 1e-4 * (1.0 + det.abs()), "prod_im={pim}");
        }
    }
}
