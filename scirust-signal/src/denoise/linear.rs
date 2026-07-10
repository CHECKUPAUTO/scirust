//! Linear (LTI convolution) smoothers.
//!
//! These are the workhorses for broadband, roughly-Gaussian noise. They are cheap
//! and phase-well-behaved, but they blur sharp transitions — for edge preservation
//! prefer [`super::variational::total_variation`] or [`super::rank::median_filter`].

use super::mirror_index;

/// Moving-average (boxcar) filter. `window` is the full width; it is rounded up to
/// the nearest odd number so the filter is symmetric (zero phase). Borders are
/// handled by mirror reflection, so the output has the same length as the input.
pub fn moving_average(signal: &[f64], window: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || window <= 1
    {
        return signal.to_vec();
    }
    let half = (window / 2) as isize;
    let width = (2 * half + 1) as f64;
    let mut out = vec![0.0; n];
    for (i, o) in out.iter_mut().enumerate()
    {
        let mut sum = 0.0;
        for k in -half..=half
        {
            sum += signal[mirror_index(i as isize + k, n)];
        }
        *o = sum / width;
    }
    out
}

/// Gaussian smoothing: convolution with a discretized Gaussian kernel of standard
/// deviation `sigma` (samples), truncated at a 3-σ radius. A smoother, less
/// spectrally-leaky alternative to the boxcar.
pub fn gaussian_smooth(signal: &[f64], sigma: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || sigma <= 0.0
    {
        return signal.to_vec();
    }
    let radius = (3.0 * sigma).ceil().max(1.0) as isize;
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    let mut ksum = 0.0;
    for k in -radius..=radius
    {
        let w = (-(k as f64) * (k as f64) / (2.0 * sigma * sigma)).exp();
        kernel.push(w);
        ksum += w;
    }
    for w in kernel.iter_mut()
    {
        *w /= ksum;
    }
    let mut out = vec![0.0; n];
    for (i, o) in out.iter_mut().enumerate()
    {
        let mut sum = 0.0;
        for (ki, &w) in kernel.iter().enumerate()
        {
            let k = ki as isize - radius;
            sum += w * signal[mirror_index(i as isize + k, n)];
        }
        *o = sum;
    }
    out
}

/// Savitzky-Golay filter: fit a degree-`poly_order` polynomial to a sliding window
/// of half-width `half_window` by least squares and take its value at the center.
/// Unlike a boxcar it preserves peak height and width — the classic choice for
/// spectra and chromatograms. `poly_order` is clamped below the window length.
pub fn savitzky_golay(signal: &[f64], poly_order: usize, half_window: usize) -> Vec<f64> {
    let n = signal.len();
    let win = 2 * half_window + 1;
    if n == 0 || half_window == 0 || win > n
    {
        return signal.to_vec();
    }
    let p = poly_order.min(win - 1);
    let m = half_window as isize;

    // Normal-equations Gram matrix JtJ[a][b] = sum_i z_i^(a+b), z_i = i - m.
    let dim = p + 1;
    let mut jtj = vec![vec![0.0; dim]; dim];
    for (a, row) in jtj.iter_mut().enumerate()
    {
        for (b, cell) in row.iter_mut().enumerate()
        {
            let mut s = 0.0;
            for z in -m..=m
            {
                s += (z as f64).powi((a + b) as i32);
            }
            *cell = s;
        }
    }
    // Solve JtJ y = e0; the smoothing coefficients are h_i = sum_j y_j z_i^j.
    let mut e0 = vec![0.0; dim];
    e0[0] = 1.0;
    let y = solve_dense(&mut jtj, &mut e0);

    let mut coeffs = vec![0.0; win];
    for (idx, c) in coeffs.iter_mut().enumerate()
    {
        let z = idx as f64 - m as f64;
        let mut h = 0.0;
        for (j, &yj) in y.iter().enumerate()
        {
            h += yj * z.powi(j as i32);
        }
        *c = h;
    }

    let mut out = vec![0.0; n];
    for (i, o) in out.iter_mut().enumerate()
    {
        let mut sum = 0.0;
        for (idx, &c) in coeffs.iter().enumerate()
        {
            let k = idx as isize - m;
            sum += c * signal[mirror_index(i as isize + k, n)];
        }
        *o = sum;
    }
    out
}

/// First-order exponential moving average (single-pole IIR): a causal, `O(1)`-state
/// low-pass. `alpha` in (0, 1]; smaller means heavier smoothing. Returns the input
/// unchanged for out-of-range `alpha`.
pub fn exp_moving_average(signal: &[f64], alpha: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || !(0.0..=1.0).contains(&alpha) || alpha == 0.0
    {
        return signal.to_vec();
    }
    let mut out = vec![0.0; n];
    out[0] = signal[0];
    for i in 1..n
    {
        out[i] = alpha * signal[i] + (1.0 - alpha) * out[i - 1];
    }
    out
}

/// Gaussian elimination with partial pivoting for a small dense system `a x = b`.
/// `a` and `b` are consumed (used as scratch). Returns `x`.
#[allow(clippy::needless_range_loop)]
fn solve_dense(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    for col in 0..n
    {
        // Partial pivot.
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n
        {
            if a[r][col].abs() > best
            {
                best = a[r][col].abs();
                pivot = r;
            }
        }
        if pivot != col
        {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }
        let diag = a[col][col];
        if diag.abs() < 1.0e-300
        {
            continue;
        }
        for r in (col + 1)..n
        {
            let factor = a[r][col] / diag;
            if factor == 0.0
            {
                continue;
            }
            for c in col..n
            {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }
    // Back-substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev()
    {
        let mut sum = b[i];
        for c in (i + 1)..n
        {
            sum -= a[i][c] * x[c];
        }
        let diag = a[i][i];
        x[i] = if diag.abs() < 1.0e-300
        {
            0.0
        }
        else
        {
            sum / diag
        };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::*;
    use core::f64::consts::PI;

    fn noisy_sine(n: usize, noise: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Lcg::new(seed);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + noise * rng.gauss()).collect();
        (clean, obs)
    }

    #[test]
    fn moving_average_reduces_noise() {
        let (clean, obs) = noisy_sine(256, 0.4, 1);
        let out = moving_average(&obs, 7);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
        assert_eq!(out.len(), obs.len());
    }

    #[test]
    fn gaussian_smooth_reduces_noise() {
        let (clean, obs) = noisy_sine(256, 0.4, 2);
        let out = gaussian_smooth(&obs, 2.0);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
    }

    #[test]
    fn savitzky_golay_order0_is_moving_average() {
        let sig: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        let sg = savitzky_golay(&sig, 0, 3);
        let ma = moving_average(&sig, 7);
        for (a, b) in sg.iter().zip(ma.iter())
        {
            assert!((a - b).abs() < 1.0e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn savitzky_golay_preserves_polynomial() {
        // A cubic is reproduced exactly by an order-3 SG filter (interior points).
        let sig: Vec<f64> = (0..50)
            .map(|i| {
                let x = i as f64;
                2.0 - 0.5 * x + 0.05 * x * x - 0.001 * x * x * x
            })
            .collect();
        let out = savitzky_golay(&sig, 3, 4);
        for i in 5..45
        {
            assert!(
                (out[i] - sig[i]).abs() < 1.0e-6,
                "index {i}: {} vs {}",
                out[i],
                sig[i]
            );
        }
    }

    #[test]
    fn savitzky_golay_reduces_noise() {
        let (clean, obs) = noisy_sine(256, 0.3, 3);
        let out = savitzky_golay(&obs, 2, 6);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
    }

    #[test]
    fn ema_smooths_and_passes_dc() {
        let sig = vec![5.0; 32];
        let out = exp_moving_average(&sig, 0.3);
        for &v in out.iter()
        {
            assert!((v - 5.0).abs() < 1.0e-9);
        }
    }
}
