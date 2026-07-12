//! MUSIC (MUltiple SIgnal Classification) subspace direction finding.
//!
//! MVDR ([`super::doa`]) sharpens the beamformer but its resolution still
//! degrades as sources approach. **MUSIC** is a *subspace* method: it
//! eigendecomposes the array covariance, splits the eigenvectors into a signal
//! subspace (the `d` largest eigenvalues — signal-plus-noise) and a noise
//! subspace (the rest), and exploits that every source steering vector is
//! orthogonal to the noise subspace. The spectrum
//! `P(θ) = 1 / ‖Eₙᴴ·a(θ)‖²` therefore spikes — in the noise-free limit, to
//! infinity — exactly at the source directions, giving resolution limited by
//! snapshot count and SNR rather than by the array aperture.
//!
//! Built on the sample covariance from [`super::doa::covariance`] and a
//! from-scratch complex-Hermitian eigensolver (cyclic Jacobi rotations);
//! dependency-free.

use super::beamform::steering_vector;
use super::doa::covariance;
use crate::complex::Complex;

/// Eigendecomposition of a Hermitian matrix `a` by cyclic complex-Jacobi
/// rotations. Returns `(eigenvalues, eigenvectors)` where eigenvector `k` is
/// column `k` of the returned matrix (`vecs[i][k]` is its `i`-th component), so
/// that `a = V · diag(eigenvalues) · Vᴴ`. Eigenvalues are real (the matrix is
/// assumed Hermitian); the eigenvectors are orthonormal.
#[allow(clippy::needless_range_loop)] // dense matrix sweep — indices are the algorithm
pub(super) fn hermitian_eig(mut a: Vec<Vec<Complex>>) -> (Vec<f64>, Vec<Vec<Complex>>) {
    let n = a.len();
    // Accumulated eigenvectors, initialised to the identity.
    let mut v: Vec<Vec<Complex>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j
                    {
                        Complex::new(1.0, 0.0)
                    }
                    else
                    {
                        Complex::zero()
                    }
                })
                .collect()
        })
        .collect();
    for _sweep in 0..100
    {
        // Off-diagonal energy: stop once the matrix is essentially diagonal.
        let mut off = 0.0;
        for p in 0..n
        {
            for q in (p + 1)..n
            {
                off += a[p][q].mag_sq();
            }
        }
        if off <= 1e-28
        {
            break;
        }
        for p in 0..n
        {
            for q in (p + 1)..n
            {
                let apq = a[p][q];
                if apq.mag_sq() <= 1e-30
                {
                    continue;
                }
                // Rotate away the phase of a_pq, leaving a real symmetric 2×2
                // block [[app, r], [r, aqq]] to which a real Jacobi rotation
                // applies. φ is that phase, r its magnitude.
                let phi = apq.phase();
                let r = apq.mag();
                let app = a[p][p].re;
                let aqq = a[q][q].re;
                let tau = (aqq - app) / (2.0 * r);
                let t = if tau == 0.0
                {
                    1.0
                }
                else
                {
                    let sign = if tau > 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (tau * tau + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let eph = Complex::cis(phi);
                let emph = Complex::cis(-phi);
                // A ← Gᴴ A G, applied as (A·G) on columns p,q then (Gᴴ·B) on
                // rows p,q. G is the identity except in the {p,q} block
                // [[c, s], [-s·e^{-iφ}, c·e^{-iφ}]] (unitary).
                for row in a.iter_mut()
                {
                    let rp = row[p];
                    let rq = row[q];
                    row[p] = rp * c + rq * (emph * (-s));
                    row[q] = rp * s + rq * (emph * c);
                }
                let (lo, hi) = a.split_at_mut(q);
                let row_p = &mut lo[p];
                let row_q = &mut hi[0];
                for (bp, bq) in row_p.iter_mut().zip(row_q.iter_mut())
                {
                    let vp = *bp;
                    let vq = *bq;
                    *bp = vp * c + vq * (eph * (-s));
                    *bq = vp * s + vq * (eph * c);
                }
                // Accumulate the same rotation into the eigenvector matrix.
                for row in v.iter_mut()
                {
                    let rp = row[p];
                    let rq = row[q];
                    row[p] = rp * c + rq * (emph * (-s));
                    row[q] = rp * s + rq * (emph * c);
                }
            }
        }
    }
    let vals = (0..n).map(|i| a[i][i].re).collect();
    (vals, v)
}

/// The **MUSIC** spatial spectrum `P(θ) = 1 / ‖Eₙᴴ·a(θ)‖²` over the steering
/// `angles` (radians from broadside), for a ULA of spacing `spacing`
/// wavelengths, assuming `num_sources` incident signals. `Eₙ` is the noise
/// subspace — the eigenvectors of the sample covariance belonging to its
/// `M − num_sources` smallest eigenvalues. The spectrum spikes at the source
/// directions.
///
/// `num_sources` is clamped to `1..=M-1` (there must be at least one noise
/// eigenvector). All-zero if the covariance is empty or the array has fewer
/// than two elements.
pub fn music_spectrum(
    snapshots: &[Vec<Complex>],
    spacing: f64,
    angles: &[f64],
    num_sources: usize,
) -> Vec<f64> {
    let r = covariance(snapshots);
    if r.is_empty() || r.len() < 2
    {
        return vec![0.0; angles.len()];
    }
    let m = r.len();
    let d = num_sources.clamp(1, m - 1);
    let (vals, vecs) = hermitian_eig(r);
    // Eigenvalues ascending: the first M−d indices are the noise subspace.
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&i, &j| vals[i].total_cmp(&vals[j]));
    let noise = &idx[..(m - d)];
    angles
        .iter()
        .map(|&theta| {
            let a = steering_vector(m, spacing, theta);
            // ‖Eₙᴴ a‖² = Σ_k |e_kᴴ a|² over the noise eigenvectors e_k.
            let s: f64 = noise
                .iter()
                .map(|&k| {
                    let proj = a.iter().enumerate().fold(Complex::zero(), |acc, (i, &ai)| {
                        acc + vecs[i][k].conj() * ai
                    });
                    proj.mag_sq()
                })
                .sum();
            1.0 / s.max(1e-300)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::beamform::estimate_doa;
    use super::*;
    use std::f64::consts::PI;

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

    fn two_source_snapshots(
        m: usize,
        spacing: f64,
        t1: f64,
        t2: f64,
        t: usize,
    ) -> Vec<Vec<Complex>> {
        let a1 = steering_vector(m, spacing, t1);
        let a2 = steering_vector(m, spacing, t2);
        let mut rng = Lcg(0x0051_7EAD);
        (0..t)
            .map(|_| {
                let s1 = Complex::cis(2.0 * PI * rng.unit());
                let s2 = Complex::cis(2.0 * PI * rng.unit());
                (0..m).map(|i| s1 * a1[i] + s2 * a2[i]).collect()
            })
            .collect()
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // reconstruction indexes V, Λ and Vᴴ
    fn hermitian_eig_reconstructs_and_is_orthonormal() {
        // A fixed 3×3 Hermitian matrix.
        let a = vec![
            vec![
                Complex::new(2.0, 0.0),
                Complex::new(1.0, 1.0),
                Complex::new(0.0, -1.0),
            ],
            vec![
                Complex::new(1.0, -1.0),
                Complex::new(3.0, 0.0),
                Complex::new(2.0, 0.0),
            ],
            vec![
                Complex::new(0.0, 1.0),
                Complex::new(2.0, 0.0),
                Complex::new(1.0, 0.0),
            ],
        ];
        let (vals, v) = hermitian_eig(a.clone());
        let n = 3;
        // V·diag(λ)·Vᴴ reconstructs A.
        for i in 0..n
        {
            for j in 0..n
            {
                let mut acc = Complex::zero();
                for k in 0..n
                {
                    acc += v[i][k] * v[j][k].conj() * vals[k];
                }
                assert!((acc.re - a[i][j].re).abs() < 1e-9, "re at {i},{j}");
                assert!((acc.im - a[i][j].im).abs() < 1e-9, "im at {i},{j}");
            }
        }
        // Columns of V are orthonormal: Vᴴ V = I.
        for k in 0..n
        {
            for l in 0..n
            {
                let mut acc = Complex::zero();
                for i in 0..n
                {
                    acc += v[i][k].conj() * v[i][l];
                }
                let expect = if k == l { 1.0 } else { 0.0 };
                assert!((acc.re - expect).abs() < 1e-9 && acc.im.abs() < 1e-9);
            }
        }
        // Eigenvalues of a Hermitian matrix are real and sum to the trace (6).
        assert!((vals.iter().sum::<f64>() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn music_peaks_at_a_single_source() {
        let (m, spacing, theta0) = (8usize, 0.5, 0.15_f64);
        let a0 = steering_vector(m, spacing, theta0);
        let snaps: Vec<Vec<Complex>> = (0..40)
            .map(|k| {
                let s = Complex::cis(0.37 * k as f64);
                a0.iter().map(|&ai| s * ai).collect()
            })
            .collect();
        let angles: Vec<f64> = (-90..=90).map(|deg| (deg as f64).to_radians()).collect();
        let spectrum = music_spectrum(&snaps, spacing, &angles, 1);
        let est = estimate_doa(&spectrum, &angles).unwrap();
        assert!(
            (est - theta0).abs() < 2.0_f64.to_radians(),
            "DOA {est} vs {theta0}"
        );
    }

    #[test]
    fn music_resolves_two_close_sources() {
        // Two sources 6° apart — inside a 10-element array's ~11° beamwidth.
        let (m, spacing) = (10usize, 0.5);
        let (t1, t2) = (0.0_f64, 6.0_f64.to_radians());
        let snaps = two_source_snapshots(m, spacing, t1, t2, 400);
        let mid = 0.5 * (t1 + t2);
        let probe = [t1, mid, t2];
        let spectrum = music_spectrum(&snaps, spacing, &probe, 2);
        // Two resolved peaks ⇒ the midpoint is a valley below both sources.
        assert!(
            spectrum[1] < spectrum[0] && spectrum[1] < spectrum[2],
            "MUSIC did not resolve two sources: {spectrum:?}"
        );
    }

    #[test]
    fn music_handles_degenerate_input() {
        // Empty / single-element arrays yield an all-zero spectrum.
        assert!(
            music_spectrum(&[], 0.5, &[0.0, 0.1], 1)
                .iter()
                .all(|&p| p == 0.0)
        );
        let single = vec![vec![Complex::new(1.0, 0.0)]; 4];
        assert!(
            music_spectrum(&single, 0.5, &[0.0, 0.1], 1)
                .iter()
                .all(|&p| p == 0.0)
        );
    }
}
