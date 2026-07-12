//! ESPRIT (Estimation of Signal Parameters via Rotational Invariance
//! Techniques) direction finding.
//!
//! [`super::music`] scans a spectrum and reads off its peaks; **ESPRIT** skips
//! the scan entirely. It exploits the *rotational invariance* of a uniform
//! linear array: the array split into its first `M−1` and last `M−1` elements
//! sees the same wavefronts up to a per-source phase `e^{jμ}`, where
//! `μ = 2π·spacing·sin θ` is exactly the steering vector's inter-element phase
//! step. The signal subspace inherits that structure, so the small `d×d` matrix
//! `Ψ` relating the two subarrays' subspaces is similar to
//! `diag(e^{jμ₁},…,e^{jμ_d})` — its eigenvalues carry the source angles
//! *directly*, with no grid and no scan resolution to trade off.
//!
//! Built on the sample covariance ([`super::doa::covariance`]) and the Hermitian
//! eigensolver shared with [`super::music`] for the subspace, plus a small
//! from-scratch complex eigensolver (upper-Hessenberg reduction followed by the
//! shifted QR algorithm) for the non-Hermitian `Ψ`; dependency-free.

use super::doa::covariance;
use super::music::hermitian_eig;
use crate::complex::Complex;
use std::f64::consts::PI;

/// Principal complex square root `√z = √|z|·e^{i·arg(z)/2}`.
fn csqrt(z: Complex) -> Complex {
    let m = z.mag();
    if m == 0.0
    {
        return Complex::zero();
    }
    Complex::cis(0.5 * z.phase()) * m.sqrt()
}

/// The two eigenvalues of the `2×2` complex matrix `[[a, b], [c, d]]`, from the
/// characteristic quadratic `λ² − (a+d)λ + (ad − bc) = 0`.
fn eig2(a: Complex, b: Complex, c: Complex, d: Complex) -> (Complex, Complex) {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = csqrt(tr * tr - det * 4.0);
    ((tr + disc) * 0.5, (tr - disc) * 0.5)
}

/// A complex Givens rotation `G = [[c, s], [−s̄, c]]` (with `c` real) that
/// annihilates `g`: `G·[f; g] = [r; 0]`. Returns `(c, s)`.
fn givens(f: Complex, g: Complex) -> (f64, Complex) {
    if g.mag_sq() == 0.0
    {
        return (1.0, Complex::zero());
    }
    if f.mag_sq() == 0.0
    {
        return (0.0, g.conj() / g.mag());
    }
    let fm = f.mag();
    let denom = (fm * fm + g.mag_sq()).sqrt();
    let c = fm / denom;
    // s = (f/|f|)·conj(g)/‖(f,g)‖ makes conj(s) = c·g/f, so the second row zeros.
    let s = f * (1.0 / fm) * g.conj() * (1.0 / denom);
    (c, s)
}

/// Reduce `a` to upper-Hessenberg form in place by Givens similarities
/// `A ← G·A·Gᴴ`, annihilating each sub-subdiagonal entry. Eigenvalues are
/// preserved.
#[allow(clippy::needless_range_loop)] // dense matrix sweep — indices are the algorithm
fn to_hessenberg(a: &mut [Vec<Complex>]) {
    let n = a.len();
    if n < 3
    {
        return;
    }
    for j in 0..n - 2
    {
        for i in (j + 2..n).rev()
        {
            let (c, s) = givens(a[i - 1][j], a[i][j]);
            if s.mag_sq() == 0.0
            {
                continue;
            }
            // Left: rotate rows i−1, i.
            for col in 0..n
            {
                let xp = a[i - 1][col];
                let xq = a[i][col];
                a[i - 1][col] = c * xp + s * xq;
                a[i][col] = (-s.conj()) * xp + c * xq;
            }
            // Right (similarity): rotate columns i−1, i by Gᴴ.
            for row in a.iter_mut()
            {
                let xp = row[i - 1];
                let xq = row[i];
                row[i - 1] = c * xp + s.conj() * xq;
                row[i] = (-s) * xp + c * xq;
            }
        }
    }
}

/// One explicit-shift QR step on the leading `m×m` (Hessenberg) block:
/// factor `A − μI = QR` with Givens rotations, then overwrite the block with
/// `RQ + μI`. Hessenberg form and eigenvalues are preserved.
#[allow(clippy::needless_range_loop)] // dense matrix sweep — indices are the algorithm
fn qr_step(a: &mut [Vec<Complex>], m: usize, mu: Complex) {
    for i in 0..m
    {
        a[i][i] -= mu;
    }
    let mut rot = Vec::with_capacity(m - 1);
    for i in 0..m - 1
    {
        let (c, s) = givens(a[i][i], a[i + 1][i]);
        for col in 0..m
        {
            let xp = a[i][col];
            let xq = a[i + 1][col];
            a[i][col] = c * xp + s * xq;
            a[i + 1][col] = (-s.conj()) * xp + c * xq;
        }
        rot.push((c, s));
    }
    // RQ: post-multiply by Qᴴ = G₀ᴴ G₁ᴴ …, column pair (i, i+1) by Gᴴ.
    for (i, &(c, s)) in rot.iter().enumerate()
    {
        for row in a.iter_mut().take(m)
        {
            let xp = row[i];
            let xq = row[i + 1];
            row[i] = c * xp + s.conj() * xq;
            row[i + 1] = (-s) * xp + c * xq;
        }
    }
    for i in 0..m
    {
        a[i][i] += mu;
    }
}

/// Eigenvalues of a general (small) complex matrix by upper-Hessenberg
/// reduction followed by the shifted QR algorithm with a Wilkinson shift and
/// bottom-corner deflation. Order is unspecified.
fn eigenvalues(mut a: Vec<Vec<Complex>>) -> Vec<Complex> {
    let n = a.len();
    if n == 0
    {
        return Vec::new();
    }
    if n == 1
    {
        return vec![a[0][0]];
    }
    to_hessenberg(&mut a);
    let mut eig = Vec::with_capacity(n);
    let mut m = n;
    let mut iters = 0usize;
    while m > 2
    {
        let sub = a[m - 1][m - 2].mag();
        let scale = a[m - 2][m - 2].mag() + a[m - 1][m - 1].mag();
        if sub <= 1e-15 * scale.max(1e-300) || iters > 2000
        {
            eig.push(a[m - 1][m - 1]);
            m -= 1;
            iters = 0;
            continue;
        }
        // Wilkinson shift: trailing-2×2 eigenvalue nearest the bottom corner.
        let (e1, e2) = eig2(
            a[m - 2][m - 2],
            a[m - 2][m - 1],
            a[m - 1][m - 2],
            a[m - 1][m - 1],
        );
        let corner = a[m - 1][m - 1];
        let mu = if (e1 - corner).mag() <= (e2 - corner).mag()
        {
            e1
        }
        else
        {
            e2
        };
        qr_step(&mut a, m, mu);
        iters += 1;
    }
    if m == 2
    {
        let (e1, e2) = eig2(a[0][0], a[0][1], a[1][0], a[1][1]);
        eig.push(e1);
        eig.push(e2);
    }
    else if m == 1
    {
        eig.push(a[0][0]);
    }
    eig
}

/// Solve the complex linear system `c·x = b` (with `b` carrying several
/// right-hand-side columns) by Gauss–Jordan elimination with partial pivoting.
/// `x` is returned in the shape of `b`; `None` if `c` is singular.
#[allow(clippy::needless_range_loop)] // dense matrix sweep — indices are the algorithm
fn solve(mut c: Vec<Vec<Complex>>, mut b: Vec<Vec<Complex>>) -> Option<Vec<Vec<Complex>>> {
    let n = c.len();
    let w = b[0].len();
    for col in 0..n
    {
        let mut piv = col;
        let mut best = c[col][col].mag_sq();
        for r in (col + 1)..n
        {
            let mag = c[r][col].mag_sq();
            if mag > best
            {
                best = mag;
                piv = r;
            }
        }
        if best <= 1e-300
        {
            return None;
        }
        c.swap(col, piv);
        b.swap(col, piv);
        let d = c[col][col];
        for k in col..n
        {
            c[col][k] = c[col][k] / d;
        }
        for k in 0..w
        {
            b[col][k] = b[col][k] / d;
        }
        for r in 0..n
        {
            if r == col
            {
                continue;
            }
            let f = c[r][col];
            if f.mag_sq() == 0.0
            {
                continue;
            }
            for k in col..n
            {
                let t = c[col][k] * f;
                c[r][k] -= t;
            }
            for k in 0..w
            {
                let t = b[col][k] * f;
                b[r][k] -= t;
            }
        }
    }
    Some(b)
}

/// **ESPRIT** direction-of-arrival estimates (radians from broadside) for a ULA
/// of spacing `spacing` wavelengths, assuming `num_sources` incident signals.
///
/// The signal subspace is the span of the `num_sources` eigenvectors of the
/// sample covariance with the largest eigenvalues. Splitting the array into its
/// first and last `M−1` elements gives two subspace bases `E₁`, `E₂`; the
/// least-squares solution of `E₁·Ψ = E₂` has eigenvalues `e^{jμ_k}` from which
/// `sin θ_k = μ_k / (2π·spacing)`. The returned angles are sorted ascending.
///
/// `num_sources` is clamped to `1..=M-1`. Empty if the covariance is empty, the
/// array has fewer than two elements, or the subspace system is singular.
pub fn esprit_doa(snapshots: &[Vec<Complex>], spacing: f64, num_sources: usize) -> Vec<f64> {
    let r = covariance(snapshots);
    if r.len() < 2
    {
        return Vec::new();
    }
    let m = r.len();
    let d = num_sources.clamp(1, m - 1);
    let (vals, vecs) = hermitian_eig(r);
    // Signal subspace: the d eigenvectors with the largest eigenvalues.
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&i, &j| vals[j].total_cmp(&vals[i]));
    let sig = &idx[..d];
    // C = E₁ᴴ E₁ and B = E₁ᴴ E₂, where E₁, E₂ are the signal subspace restricted
    // to the first / last M−1 rows (elements).
    let mut cmat = vec![vec![Complex::zero(); d]; d];
    let mut bmat = vec![vec![Complex::zero(); d]; d];
    for p in 0..d
    {
        for q in 0..d
        {
            let (kp, kq) = (sig[p], sig[q]);
            let mut cpq = Complex::zero();
            let mut bpq = Complex::zero();
            for row in 0..m - 1
            {
                cpq += vecs[row][kp].conj() * vecs[row][kq];
                bpq += vecs[row][kp].conj() * vecs[row + 1][kq];
            }
            cmat[p][q] = cpq;
            bmat[p][q] = bpq;
        }
    }
    let psi = match solve(cmat, bmat)
    {
        Some(x) => x,
        None => return Vec::new(),
    };
    let two_pi_d = 2.0 * PI * spacing;
    let mut angles: Vec<f64> = eigenvalues(psi)
        .iter()
        .map(|l| {
            let sin_theta = (l.phase() / two_pi_d).clamp(-1.0, 1.0);
            sin_theta.asin()
        })
        .collect();
    angles.sort_by(f64::total_cmp);
    angles
}

#[cfg(test)]
mod tests {
    use super::super::beamform::steering_vector;
    use super::*;

    /// A deterministic LCG for reproducible random source phases.
    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn source_snapshots(m: usize, spacing: f64, dirs: &[f64], t: usize) -> Vec<Vec<Complex>> {
        let steer: Vec<Vec<Complex>> = dirs
            .iter()
            .map(|&theta| steering_vector(m, spacing, theta))
            .collect();
        let mut rng = Lcg(0x00E5_9817);
        (0..t)
            .map(|_| {
                let s: Vec<Complex> = (0..dirs.len())
                    .map(|_| Complex::cis(2.0 * PI * rng.unit()))
                    .collect();
                (0..m)
                    .map(|i| {
                        steer
                            .iter()
                            .zip(&s)
                            .fold(Complex::zero(), |acc, (a, &sk)| acc + sk * a[i])
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn eigenvalues_of_a_triangular_matrix_are_its_diagonal() {
        // Upper-triangular ⇒ eigenvalues are exactly the diagonal entries.
        let a = vec![
            vec![
                Complex::new(2.0, 1.0),
                Complex::new(5.0, -1.0),
                Complex::new(-3.0, 2.0),
            ],
            vec![
                Complex::zero(),
                Complex::new(-1.0, 4.0),
                Complex::new(0.5, 0.5),
            ],
            vec![Complex::zero(), Complex::zero(), Complex::new(3.0, -2.0)],
        ];
        let mut got = eigenvalues(a);
        got.sort_by(|x, y| x.re.total_cmp(&y.re));
        let mut want = [
            Complex::new(2.0, 1.0),
            Complex::new(-1.0, 4.0),
            Complex::new(3.0, -2.0),
        ];
        want.sort_by(|x, y| x.re.total_cmp(&y.re));
        for (g, w) in got.iter().zip(&want)
        {
            assert!(
                (g.re - w.re).abs() < 1e-9 && (g.im - w.im).abs() < 1e-9,
                "{g:?} vs {w:?}"
            );
        }
    }

    #[test]
    fn eigenvalues_lie_on_the_unit_circle_for_a_rotation() {
        // Similarity of diag(e^{iθ}) has eigenvalues e^{iθ}: magnitude 1, and the
        // product of eigenvalues equals the determinant e^{i(θ₁+θ₂+θ₃)}.
        let thetas = [0.4_f64, -1.1, 2.3];
        // Build A = V·diag·V⁻¹ implicitly by conjugating with a fixed rotation is
        // overkill; instead check the companion-free case diag itself plus a
        // strictly-triangular perturbation keeps the diagonal eigenvalues.
        let a: Vec<Vec<Complex>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        if i == j
                        {
                            Complex::cis(thetas[i])
                        }
                        else if j > i
                        {
                            Complex::new(0.3 * (i as f64 + 1.0), -0.2 * j as f64)
                        }
                        else
                        {
                            Complex::zero()
                        }
                    })
                    .collect()
            })
            .collect();
        let vals = eigenvalues(a);
        assert_eq!(vals.len(), 3);
        for v in &vals
        {
            assert!((v.mag() - 1.0).abs() < 1e-9, "off unit circle: {v:?}");
        }
    }

    #[test]
    fn esprit_recovers_a_single_source() {
        let (m, spacing, theta0) = (8usize, 0.5, 0.15_f64);
        let snaps = source_snapshots(m, spacing, &[theta0], 40);
        let est = esprit_doa(&snaps, spacing, 1);
        assert_eq!(est.len(), 1);
        assert!(
            (est[0] - theta0).abs() < 0.5_f64.to_radians(),
            "{est:?} vs {theta0}"
        );
    }

    #[test]
    fn esprit_resolves_two_sources_off_grid() {
        // ESPRIT is gridless — recover two angles that fall between any degree
        // grid to well under a degree.
        let (m, spacing) = (10usize, 0.5);
        let (t1, t2) = (-7.3_f64.to_radians(), 11.8_f64.to_radians());
        let snaps = source_snapshots(m, spacing, &[t1, t2], 300);
        let est = esprit_doa(&snaps, spacing, 2);
        assert_eq!(est.len(), 2);
        // Sorted ascending, so est[0] ↔ t1, est[1] ↔ t2.
        assert!((est[0] - t1).abs() < 0.5_f64.to_radians(), "src1 {est:?}");
        assert!((est[1] - t2).abs() < 0.5_f64.to_radians(), "src2 {est:?}");
    }

    #[test]
    fn esprit_handles_degenerate_input() {
        assert!(esprit_doa(&[], 0.5, 1).is_empty());
        let single = vec![vec![Complex::new(1.0, 0.0)]; 4];
        assert!(esprit_doa(&single, 0.5, 1).is_empty());
    }
}
