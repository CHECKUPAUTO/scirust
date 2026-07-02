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
                // With J = [[c, s], [-s, c]] applied as JᵀAJ below, the (p,q)
                // entry becomes (a_pp − a_qq)·sc + a_pq·(c² − s²); zeroing it
                // needs tan(2θ) = 2·a_pq / (a_qq − a_pp). The previous sign
                // (a_pp − a_qq) rotated AWAY from the annihilation angle, so
                // the off-diagonal mass oscillated instead of vanishing and
                // the 100-sweep loop exited unconverged with wrong
                // eigenvectors (MAC ≈ 0.75 on a rank-1 CPSD).
                let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
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
    // `total_cmp` gives a deterministic descending order even if a diagonal
    // entry is NaN (which `partial_cmp().unwrap()` would panic on).
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
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
/// normalised so its largest-magnitude entry is positive unit. The `freq_hz`→bin
/// mapping is derived from the actual transform length; `n_fft` is only consulted
/// as a fallback for degenerate (single-bin) spectra.
pub fn mode_shape(channels: &[Vec<f64>], sample_rate: f64, n_fft: usize, freq_hz: f64) -> Vec<f64> {
    if channels.is_empty()
    {
        return Vec::new();
    }
    let spectra = channel_spectra(channels);
    let nbins = spectra[0].len();
    if nbins == 0
    {
        return Vec::new();
    }
    // `channel_spectra` transforms each channel at its own length, so the true
    // FFT length is `(nbins - 1) * 2`, not the caller-supplied `n_fft`; using the
    // latter would mis-map `freq_hz` to a bin whenever they disagree.
    let fft_len = (nbins - 1) * 2;
    let effective_fft = if fft_len == 0 { n_fft.max(1) } else { fft_len };
    let bin = ((freq_hz * effective_fft as f64 / sample_rate).round() as usize).min(nbins - 1);
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

    // Regression (bug: mode_shape/first_singular_spectrum panic on empty input).
    #[test]
    fn empty_channels_do_not_panic() {
        assert!(mode_shape(&[], 100.0, 256, 10.0).is_empty());
        assert!(first_singular_spectrum(&[]).is_empty());
    }

    // Regression (bug: jacobi_eigen sort `partial_cmp().unwrap()` panics on NaN).
    #[test]
    fn jacobi_does_not_panic_on_nan_diagonal() {
        let nan = f64::NAN;
        // A NaN entry must not crash the descending sort; the call just returns.
        let (evals, evecs) = jacobi_eigen(&[vec![nan, 0.0], vec![0.0, 1.0]]);
        assert_eq!(evals.len(), 2);
        assert_eq!(evecs.len(), 2);
    }

    // Regression (bug: bin mapping used caller `n_fft` instead of the actual FFT
    // length). Passing a deliberately wrong `n_fft` must still recover the shape
    // at the true frequency, because the bin is derived from the real transform.
    #[test]
    fn mode_shape_ignores_wrong_n_fft() {
        let (n, sr) = (4096usize, 4096.0);
        let shape = [1.0, 0.6];
        let chan = |amp: f64| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let t = i as f64 / sr;
                    amp * (2.0 * PI * 12.0 * t).sin()
                })
                .collect()
        };
        let channels = vec![chan(shape[0]), chan(shape[1])];
        // With n=4096 and sr=4096, the true FFT length is 4096 (1 Hz/bin) so
        // 12 Hz -> bin 12. A caller-supplied `n_fft` of 2048 would (pre-fix) map
        // 12 Hz -> bin 6, an empty bin, wrecking the recovered shape.
        let wrong_n_fft = n / 2;
        let phi = mode_shape(&channels, sr, wrong_n_fft, 12.0);
        assert!(
            mac(&phi, &shape) > 0.99,
            "MAC {} shape {phi:?} (bin mapping should ignore wrong n_fft)",
            mac(&phi, &shape)
        );
    }
}
