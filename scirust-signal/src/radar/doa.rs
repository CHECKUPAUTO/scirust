//! High-resolution direction-of-arrival (DOA) estimation.
//!
//! The conventional beamformer ([`super::beamform`]) resolves sources no finer
//! than the array beamwidth. The **MVDR / Capon** beamformer instead chooses,
//! for each look direction, the array weights that minimise total output power
//! subject to unit gain toward that direction — passing the look direction
//! undistorted while nulling everything else. Its spatial spectrum
//! `P(θ) = 1 / (aᴴ(θ) R⁻¹ a(θ))` resolves sources far closer together than the
//! beamwidth. Built on the sample covariance and a small complex matrix
//! inverse; dependency-free.

use super::beamform::steering_vector;
use crate::complex::Complex;

/// Complex division `a / b = a·conj(b) / |b|²` (avoids relying on a `Div`
/// impl and keeps the intent explicit).
fn cdiv(a: Complex, b: Complex) -> Complex {
    let d = b.mag_sq();
    Complex::new(
        (a.re * b.re + a.im * b.im) / d,
        (a.im * b.re - a.re * b.im) / d,
    )
}

/// The `M × M` sample covariance `R = (1/T) Σ_t x[t]·x[t]ᴴ` of the array
/// `snapshots` (`snapshots[t]` a length-`M` snapshot). Snapshots of the wrong
/// length are skipped; empty if no usable snapshot.
pub fn covariance(snapshots: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
    if snapshots.is_empty()
    {
        return Vec::new();
    }
    let m = snapshots[0].len();
    let mut r = vec![vec![Complex::zero(); m]; m];
    let mut t = 0usize;
    for x in snapshots
    {
        if x.len() != m
        {
            continue;
        }
        for i in 0..m
        {
            for j in 0..m
            {
                r[i][j] += x[i] * x[j].conj();
            }
        }
        t += 1;
    }
    if t == 0
    {
        return Vec::new();
    }
    let inv_t = 1.0 / t as f64;
    for row in &mut r
    {
        for c in row
        {
            *c = Complex::new(c.re * inv_t, c.im * inv_t);
        }
    }
    r
}

/// Invert a complex square matrix by Gauss–Jordan elimination with partial
/// pivoting. `None` if it is singular. Shared with [`super::stap`] for the
/// space-time adaptive weight computation.
pub(super) fn invert(mut a: Vec<Vec<Complex>>) -> Option<Vec<Vec<Complex>>> {
    let n = a.len();
    let mut inv: Vec<Vec<Complex>> = (0..n)
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
    for col in 0..n
    {
        let mut piv = col;
        let mut best = a[col][col].mag();
        for (r, row) in a.iter().enumerate().skip(col + 1)
        {
            if row[col].mag() > best
            {
                best = row[col].mag();
                piv = r;
            }
        }
        if best < 1e-300
        {
            return None;
        }
        a.swap(col, piv);
        inv.swap(col, piv);
        let d = a[col][col];
        for j in 0..n
        {
            a[col][j] = cdiv(a[col][j], d);
            inv[col][j] = cdiv(inv[col][j], d);
        }
        for r in 0..n
        {
            if r == col
            {
                continue;
            }
            let factor = a[r][col];
            if factor.mag() == 0.0
            {
                continue;
            }
            for j in 0..n
            {
                let da = factor * a[col][j];
                let di = factor * inv[col][j];
                a[r][j] -= da;
                inv[r][j] -= di;
            }
        }
    }
    Some(inv)
}

/// The MVDR (Capon) spatial spectrum `P(θ) = 1 / (aᴴ(θ) R⁻¹ a(θ))` over the
/// steering `angles` (radians from broadside), for a ULA of spacing `spacing`
/// wavelengths. `loading` is added to the covariance diagonal for numerical
/// stability (diagonal loading). Sharper than the conventional beamformer; its
/// peaks are the source directions. All-zero if the covariance is empty or
/// singular even after loading.
pub fn mvdr_spectrum(
    snapshots: &[Vec<Complex>],
    spacing: f64,
    angles: &[f64],
    loading: f64,
) -> Vec<f64> {
    let mut r = covariance(snapshots);
    if r.is_empty()
    {
        return vec![0.0; angles.len()];
    }
    let m = r.len();
    for (i, row) in r.iter_mut().enumerate()
    {
        row[i] += Complex::new(loading, 0.0);
    }
    let Some(rinv) = invert(r)
    else
    {
        return vec![0.0; angles.len()];
    };
    angles
        .iter()
        .map(|&theta| {
            let a = steering_vector(m, spacing, theta);
            // aᴴ R⁻¹ a — a Hermitian form, so real and positive.
            let mut denom = Complex::zero();
            for i in 0..m
            {
                let mut ra = Complex::zero();
                for j in 0..m
                {
                    ra += rinv[i][j] * a[j];
                }
                denom += a[i].conj() * ra;
            }
            1.0 / denom.re.max(1e-300)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::beamform::{beamform_spectrum, estimate_doa};
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
        let mut rng = Lcg(0x00AB_CDEF);
        (0..t)
            .map(|_| {
                // Independent random phases ⇒ the two sources are uncorrelated.
                let s1 = Complex::cis(2.0 * PI * rng.unit());
                let s2 = Complex::cis(2.0 * PI * rng.unit());
                (0..m).map(|i| s1 * a1[i] + s2 * a2[i]).collect()
            })
            .collect()
    }

    #[test]
    fn mvdr_peaks_at_the_source_direction() {
        let (m, spacing, theta0) = (10usize, 0.5, 0.2_f64);
        let a0 = steering_vector(m, spacing, theta0);
        let snaps: Vec<Vec<Complex>> = (0..32)
            .map(|k| {
                let s = Complex::cis(0.3 * k as f64);
                a0.iter().map(|&ai| s * ai).collect()
            })
            .collect();
        let angles: Vec<f64> = (-90..=90).map(|d| (d as f64).to_radians()).collect();
        let mvdr = mvdr_spectrum(&snaps, spacing, &angles, 1e-2);
        let est = estimate_doa(&mvdr, &angles).unwrap();
        assert!(
            (est - theta0).abs() < 2.0_f64.to_radians(),
            "DOA {est} vs {theta0}"
        );
    }

    #[test]
    fn mvdr_resolves_two_sources_closer_than_the_beamwidth() {
        // Two sources 6° apart — inside a 10-element array's ~11° beamwidth, so
        // the conventional beamformer merges them; MVDR resolves them.
        let (m, spacing) = (10usize, 0.5);
        let (t1, t2) = (0.0_f64, 6.0_f64.to_radians());
        let snaps = two_source_snapshots(m, spacing, t1, t2, 400);
        let mid = 0.5 * (t1 + t2);
        let probe = [t1, mid, t2];
        // Two resolved peaks ⇒ the midpoint is a valley below both sources.
        let mvdr = mvdr_spectrum(&snaps, spacing, &probe, 1e-3);
        assert!(
            mvdr[1] < mvdr[0] && mvdr[1] < mvdr[2],
            "MVDR did not resolve two sources: {mvdr:?}"
        );
        // The conventional beamformer merges them — the midpoint is not a valley.
        let bart = beamform_spectrum(&snaps, spacing, &probe);
        assert!(
            bart[1] >= bart[0].min(bart[2]),
            "Bartlett unexpectedly resolved: {bart:?}"
        );
    }

    #[test]
    // Indexing both R[i][j] and its transpose R[j][i] is exactly the point.
    #[allow(clippy::needless_range_loop)]
    fn covariance_is_hermitian_and_mvdr_handles_empty() {
        let snaps = two_source_snapshots(4, 0.5, 0.1, -0.2, 50);
        let r = covariance(&snaps);
        for i in 0..4
        {
            for j in 0..4
            {
                // R[i][j] = conj(R[j][i]).
                assert!((r[i][j].re - r[j][i].re).abs() < 1e-9);
                assert!((r[i][j].im + r[j][i].im).abs() < 1e-9);
            }
        }
        assert!(covariance(&[]).is_empty());
        assert!(
            mvdr_spectrum(&[], 0.5, &[0.0, 0.1], 1e-3)
                .iter()
                .all(|&p| p == 0.0)
        );
    }
}
