//! Frequency Domain Decomposition (FDD) — multi-sensor operational modal analysis.
//!
//! With several synchronized sensors, the response cross-power-spectral-density
//! (CPSD) matrix at each frequency has a dominant eigenvalue that peaks at the
//! natural frequencies, and a dominant eigenvector that **is** the mode shape.
//! FDD thus extracts both frequencies and shapes from ambient (output-only) data
//! — the workhorse of structural identification. Real co-spectrum + Jacobi
//! eigensolver; deterministic.

use scirust_signal::{Complex, fft_real, hanning};

/// Symmetric eigendecomposition by cyclic Jacobi rotations. Returns the
/// eigenvalues in descending order and the matching eigenvectors (`evecs[i]` is
/// the eigenvector for `evals[i]`).
#[allow(clippy::needless_range_loop)] // dense symmetric-matrix rotations
pub fn jacobi_eigen(input: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = input.len();
    let mut a = input.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _sweep in 0..100
    {
        let mut off = 0.0;
        for p in 0..n
        {
            for q in (p + 1)..n
            {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() < 1e-14
        {
            break;
        }
        for p in 0..n
        {
            for q in (p + 1)..n
            {
                if a[p][q].abs() < 1e-300
                {
                    continue;
                }
                let theta = 0.5 * (2.0 * a[p][q]).atan2(a[p][p] - a[q][q]);
                let (s, c) = theta.sin_cos();
                // A·J  (rotate columns p,q)
                for row in a.iter_mut()
                {
                    let (ap, aq) = (row[p], row[q]);
                    row[p] = c * ap - s * aq;
                    row[q] = s * ap + c * aq;
                }
                // Jᵀ·(A·J)  (rotate rows p,q)
                for k in 0..n
                {
                    let (ap, aq) = (a[p][k], a[q][k]);
                    a[p][k] = c * ap - s * aq;
                    a[q][k] = s * ap + c * aq;
                }
                // V·J  (accumulate eigenvectors)
                for row in v.iter_mut()
                {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].partial_cmp(&a[i][i]).unwrap());
    let evals: Vec<f64> = order.iter().map(|&i| a[i][i]).collect();
    let evecs: Vec<Vec<f64>> = order
        .iter()
        .map(|&col| (0..n).map(|row| v[row][col]).collect())
        .collect();
    (evals, evecs)
}

/// Per-channel Hann-windowed half-spectra.
fn channel_spectra(channels: &[Vec<f64>]) -> Vec<Vec<Complex>> {
    channels
        .iter()
        .map(|ch| {
            let win = hanning(ch.len());
            let windowed: Vec<f64> = ch.iter().zip(&win).map(|(&x, &w)| x * w).collect();
            fft_real(&windowed)
        })
        .collect()
}

/// Real CPSD matrix `Re(Xᵢ·conj(Xⱼ))` at spectral bin `bin`.
#[allow(clippy::needless_range_loop)] // dense n×n cross-spectrum fill
fn cpsd_at(spectra: &[Vec<Complex>], bin: usize) -> Vec<Vec<f64>> {
    let n = spectra.len();
    let mut g = vec![vec![0.0; n]; n];
    for i in 0..n
    {
        for j in 0..n
        {
            let (xi, xj) = (spectra[i][bin], spectra[j][bin]);
            // Re(xi · conj(xj))
            g[i][j] = xi.re * xj.re + xi.im * xj.im;
        }
    }
    g
}

/// First-singular-value spectrum: the dominant CPSD eigenvalue at every bin.
/// Peaks mark natural frequencies. Returns one value per bin.
pub fn first_singular_spectrum(channels: &[Vec<f64>]) -> Vec<f64> {
    if channels.is_empty()
    {
        return Vec::new();
    }
    let spectra = channel_spectra(channels);
    let nbins = spectra[0].len();
    (0..nbins)
        .map(|b| {
            let g = cpsd_at(&spectra, b);
            let (evals, _) = jacobi_eigen(&g);
            evals.first().copied().unwrap_or(0.0)
        })
        .collect()
}

/// Mode shape (dominant CPSD eigenvector) at the bin nearest `freq_hz`,
/// normalised so its largest-magnitude entry is positive unit.
pub fn mode_shape(channels: &[Vec<f64>], sample_rate: f64, n_fft: usize, freq_hz: f64) -> Vec<f64> {
    let spectra = channel_spectra(channels);
    let bin = ((freq_hz * n_fft as f64 / sample_rate).round() as usize).min(spectra[0].len() - 1);
    let g = cpsd_at(&spectra, bin);
    let (_evals, evecs) = jacobi_eigen(&g);
    let mut shape = evecs.into_iter().next().unwrap_or_default();
    // Sign/scale normalise.
    let pivot = shape
        .iter()
        .cloned()
        .fold(0.0_f64, |m, x| if x.abs() > m.abs() { x } else { m });
    if pivot.abs() > 1e-12
    {
        for s in shape.iter_mut()
        {
            *s /= pivot;
        }
    }
    shape
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mac;
    use crate::natural_frequencies;
    use core::f64::consts::PI;

    #[test]
    fn jacobi_matches_a_known_decomposition() {
        // Eigenvalues of [[2,1],[1,2]] are 3 and 1.
        let (evals, evecs) = jacobi_eigen(&[vec![2.0, 1.0], vec![1.0, 2.0]]);
        assert!((evals[0] - 3.0).abs() < 1e-9 && (evals[1] - 1.0).abs() < 1e-9);
        // Eigenvector for 3 is ∝ [1,1].
        let v = &evecs[0];
        assert!((v[0] - v[1]).abs() < 1e-9 && v[0].abs() > 1e-6);
    }

    #[test]
    fn fdd_finds_modes_and_shapes() {
        let (n, sr) = (4096usize, 4096.0);
        // Two modes (12 Hz, 30 Hz) with distinct shapes across two sensors.
        let shape1 = [1.0, 0.6]; // mode @ 12 Hz
        let shape2 = [1.0, -0.8]; // mode @ 30 Hz
        let chan = |s1: f64, s2: f64| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let t = i as f64 / sr;
                    s1 * (2.0 * PI * 12.0 * t).sin() + s2 * (2.0 * PI * 30.0 * t).sin()
                })
                .collect()
        };
        let channels = vec![chan(shape1[0], shape2[0]), chan(shape1[1], shape2[1])];

        let spectrum = first_singular_spectrum(&channels);
        let modes = natural_frequencies(&spectrum, sr, n, 0.2);
        assert!(
            modes.iter().any(|f| (f - 12.0).abs() < 0.3),
            "modes {modes:?}"
        );
        assert!(
            modes.iter().any(|f| (f - 30.0).abs() < 0.3),
            "modes {modes:?}"
        );

        // Recovered mode shape at 12 Hz correlates with the true shape (MAC ~ 1).
        let phi = mode_shape(&channels, sr, n, 12.0);
        assert!(
            mac(&phi, &shape1) > 0.99,
            "MAC {} shape {phi:?}",
            mac(&phi, &shape1)
        );
        // ...and is distinct from the other mode's shape.
        assert!(mac(&phi, &shape2) < 0.5, "shapes not separated");
    }
}
