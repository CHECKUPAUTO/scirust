//! Variational denoisers — penalized least squares.
//!
//! These pose denoising as an optimization: find `x` close to the data `y` yet
//! *smooth* under some penalty on its differences. The penalty choice sets the
//! character:
//!
//! * quadratic (L2) → [`tikhonov_smooth`], a linear smoother that rounds every
//!   corner;
//! * absolute (L1, total variation) → [`total_variation`], which is
//!   *edge-preserving*: it produces piecewise-constant reconstructions and keeps
//!   sharp jumps that L2 and linear filters would smear.
//!
//! Both reduce to solving a tridiagonal system, done here with a direct Thomas
//! solver — no external linear-algebra dependency.

/// Solve a tridiagonal system with the Thomas algorithm. `sub[i]` is the element
/// below the diagonal in row `i` (unused at `i = 0`), `sup[i]` the element above
/// (unused at `i = n-1`). All slices have length `n`.
fn solve_tridiag(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    if n == 0
    {
        return Vec::new();
    }
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    c[0] = sup[0] / diag[0];
    d[0] = rhs[0] / diag[0];
    for i in 1..n
    {
        let m = diag[i] - sub[i] * c[i - 1];
        let m = if m.abs() < 1.0e-300 { 1.0e-300 } else { m };
        c[i] = if i < n - 1 { sup[i] / m } else { 0.0 };
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev()
    {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    x
}

/// Tikhonov (Whittaker) first-difference smoother: minimize
/// `‖x − y‖² + λ·Σ(x_{i+1} − x_i)²`. The closed-form solution
/// `(I + λ·LᵀL) x = y` is tridiagonal. Larger `λ` means heavier smoothing. This is
/// the L2 baseline — smooth and phase-neutral, but it rounds off genuine edges.
pub fn tikhonov_smooth(signal: &[f64], lambda: f64) -> Vec<f64> {
    let n = signal.len();
    if n < 3 || lambda <= 0.0
    {
        return signal.to_vec();
    }
    let mut diag = vec![0.0; n];
    let mut sub = vec![0.0; n];
    let mut sup = vec![0.0; n];
    for j in 0..n
    {
        let degree = if j == 0 || j == n - 1 { 1.0 } else { 2.0 };
        diag[j] = 1.0 + lambda * degree;
        if j > 0
        {
            sub[j] = -lambda;
        }
        if j < n - 1
        {
            sup[j] = -lambda;
        }
    }
    solve_tridiag(&sub, &diag, &sup, signal)
}

/// Total-variation (TV-L1) denoising: minimize `‖x − y‖² + λ·Σ|x_{i+1} − x_i|`.
///
/// The L1 penalty is what preserves edges — the solution is piecewise-constant,
/// so a noisy step stays a crisp step instead of a smeared ramp. Solved by
/// iteratively-reweighted least squares (lagged-diffusivity): each iteration
/// linearizes the L1 term into a weighted L2 problem — again tridiagonal — with
/// edge weights `1/√(Δ² + ε)`. A handful of `iters` converges. Larger `λ` removes
/// more noise (and eventually flattens small steps).
pub fn total_variation(signal: &[f64], lambda: f64, iters: usize) -> Vec<f64> {
    let n = signal.len();
    if n < 3 || lambda <= 0.0
    {
        return signal.to_vec();
    }
    let eps = 1.0e-6;
    let iters = iters.max(1);
    let mut x = signal.to_vec();
    for _ in 0..iters
    {
        // Edge weights from the current estimate.
        let mut w = vec![0.0; n - 1];
        for i in 0..n - 1
        {
            let delta = x[i + 1] - x[i];
            w[i] = 1.0 / (delta * delta + eps).sqrt();
        }
        // Weighted path Laplacian: A = I + λ·LᵀWL.
        let mut diag = vec![0.0; n];
        let mut sub = vec![0.0; n];
        let mut sup = vec![0.0; n];
        for j in 0..n
        {
            let wl = if j > 0 { w[j - 1] } else { 0.0 };
            let wr = if j < n - 1 { w[j] } else { 0.0 };
            diag[j] = 1.0 + lambda * (wl + wr);
            if j > 0
            {
                sub[j] = -lambda * w[j - 1];
            }
            if j < n - 1
            {
                sup[j] = -lambda * w[j];
            }
        }
        x = solve_tridiag(&sub, &diag, &sup, signal);
    }
    x
}

/// Total variation of a signal: `Σ|x_{i+1} − x_i|`. A useful sanity metric — a
/// good denoiser lowers it (removes wiggle) without collapsing it to zero.
pub fn total_variation_norm(signal: &[f64]) -> f64 {
    signal.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn tikhonov_reduces_noise() {
        let n = 200;
        let mut rng = Lcg::new(23);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let out = tikhonov_smooth(&obs, 20.0);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
        assert!(total_variation_norm(&out) < total_variation_norm(&obs));
    }

    #[test]
    fn total_variation_preserves_step_better_than_tikhonov() {
        let n = 200;
        let mut rng = Lcg::new(29);
        let clean: Vec<f64> = (0..n).map(|i| if i < 100 { -1.0 } else { 1.0 }).collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let tv = total_variation(&obs, 2.0, 10);
        let tik = tikhonov_smooth(&obs, 20.0);
        // Both denoise, but TV keeps the edge sharper → higher SNR on a step.
        assert!(snr_db(&clean, &tv) > snr_db(&clean, &obs));
        assert!(snr_db(&clean, &tv) > snr_db(&clean, &tik));
    }

    #[test]
    fn tv_lowers_total_variation() {
        let n = 128;
        let mut rng = Lcg::new(31);
        let clean: Vec<f64> = (0..n).map(|i| if i < 64 { 0.0 } else { 3.0 }).collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let tv = total_variation(&obs, 3.0, 12);
        assert!(total_variation_norm(&tv) < total_variation_norm(&obs));
        assert!(total_variation_norm(&tv) > 0.5 * total_variation_norm(&clean));
    }
}
