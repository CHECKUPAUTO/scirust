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

/// **Exact** 1-D total-variation denoising by Condat's direct algorithm
/// (IEEE Signal Processing Letters, 2013): the unique global minimizer of
/// `½·‖x − y‖² + λ·Σ|x_{i+1} − x_i|`, computed in a single forward sweep,
/// `O(n)` in practice, with no iteration or tolerance.
///
/// Unlike the IRLS approximation in [`total_variation`], the output is the exact
/// piecewise-constant solution — its optimality is certified in the tests by the
/// KKT conditions (the running sum `sᵢ = Σ_{j≤i}(x_j − y_j)` must stay inside
/// `[−λ, +λ]`, touch `±λ` exactly at jumps of matching sign, and end at 0).
/// Note the ½ factor: `total_variation_exact(y, λ)` matches
/// [`total_variation`]`(y, 2λ, …)` in the limit of many IRLS iterations.
pub fn total_variation_exact(signal: &[f64], lambda: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0
    {
        return Vec::new();
    }
    if n == 1 || lambda <= 0.0
    {
        return signal.to_vec();
    }
    let y = signal;
    let mut x = vec![0.0; n];

    // Condat 2013, Algorithm 1 (0-based). The window [vmin, vmax] brackets the
    // running segment value; umin/umax accumulate the slack of the taut string.
    let mut k = 0usize; // current sample
    let mut k0 = 0usize; // segment start
    let mut km = 0usize; // last position where umin hit +λ
    let mut kp = 0usize; // last position where umax hit −λ
    let mut vmin = y[0] - lambda;
    let mut vmax = y[0] + lambda;
    let mut umin = lambda;
    let mut umax = -lambda;

    loop
    {
        if k == n - 1
        {
            // End of signal: flush the pending segment(s).
            if umin < 0.0
            {
                // Lower string is taut: a negative jump ends at km.
                for xi in x.iter_mut().take(km + 1).skip(k0)
                {
                    *xi = vmin;
                }
                km += 1;
                k = km;
                k0 = km;
                kp = km;
                vmin = y[k];
                umin = lambda;
                umax = y[k] + lambda - vmax;
            }
            else if umax > 0.0
            {
                // Upper string is taut: a positive jump ends at kp.
                for xi in x.iter_mut().take(kp + 1).skip(k0)
                {
                    *xi = vmax;
                }
                kp += 1;
                k = kp;
                k0 = kp;
                km = kp;
                vmax = y[k];
                umax = -lambda;
                umin = y[k] - lambda - vmin;
            }
            else
            {
                // Both strings slack: the final segment is constant.
                let v = vmin + umin / (k - k0 + 1) as f64;
                for xi in x.iter_mut().take(k + 1).skip(k0)
                {
                    *xi = v;
                }
                return x;
            }
            if k == n - 1 && k0 == n - 1
            {
                // The flush landed on the very last sample: it forms its own
                // segment.
                x[n - 1] = vmin + umin;
                return x;
            }
            continue;
        }

        if y[k + 1] + umin < vmin - lambda
        {
            // Negative jump: the segment [k0, km] takes the value vmin.
            for xi in x.iter_mut().take(km + 1).skip(k0)
            {
                *xi = vmin;
            }
            km += 1;
            k = km;
            k0 = km;
            kp = km;
            vmin = y[k];
            vmax = y[k] + 2.0 * lambda;
            umin = lambda;
            umax = -lambda;
        }
        else if y[k + 1] + umax > vmax + lambda
        {
            // Positive jump: the segment [k0, kp] takes the value vmax.
            for xi in x.iter_mut().take(kp + 1).skip(k0)
            {
                *xi = vmax;
            }
            kp += 1;
            k = kp;
            k0 = kp;
            km = kp;
            vmin = y[k] - 2.0 * lambda;
            vmax = y[k];
            umin = lambda;
            umax = -lambda;
        }
        else
        {
            // No jump: absorb the sample and re-tighten the window.
            k += 1;
            umin += y[k] - vmin;
            umax += y[k] - vmax;
            if umin >= lambda
            {
                vmin += (umin - lambda) / (k - k0 + 1) as f64;
                umin = lambda;
                km = k;
            }
            if umax <= -lambda
            {
                vmax += (umax + lambda) / (k - k0 + 1) as f64;
                umax = -lambda;
                kp = k;
            }
        }
    }
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

    /// KKT optimality check for `min ½‖x−y‖² + λ·TV(x)`: with `sᵢ = Σ_{j≤i}(xⱼ−yⱼ)`,
    /// the (unique) minimizer satisfies |sᵢ| ≤ λ everywhere, sᵢ = +λ at every
    /// positive jump, sᵢ = −λ at every negative jump, and s_{n−1} = 0. Because the
    /// objective is strictly convex, passing this test *proves* global optimality.
    fn assert_kkt_optimal(y: &[f64], x: &[f64], lambda: f64) {
        let n = y.len();
        let tol = 1.0e-8 * (n as f64);
        let mut s = 0.0;
        for i in 0..n
        {
            s += x[i] - y[i];
            if i + 1 < n
            {
                assert!(
                    s.abs() <= lambda + tol,
                    "|s[{i}]| = {} > λ = {lambda}",
                    s.abs()
                );
                let jump = x[i + 1] - x[i];
                if jump > 1.0e-9
                {
                    assert!((s - lambda).abs() < tol, "up-jump at {i}: s = {s}, want +λ");
                }
                else if jump < -1.0e-9
                {
                    assert!(
                        (s + lambda).abs() < tol,
                        "down-jump at {i}: s = {s}, want −λ"
                    );
                }
            }
        }
        assert!(s.abs() < tol, "s_end = {s}, want 0");
    }

    #[test]
    fn tv_exact_satisfies_kkt_on_varied_inputs() {
        let mut rng = Lcg::new(37);
        // Steps + noise.
        let steps: Vec<f64> = (0..200)
            .map(|i| {
                let base = if i < 60
                {
                    0.0
                }
                else if i < 130
                {
                    2.0
                }
                else
                {
                    -1.0
                };
                base + 0.3 * rng.gauss()
            })
            .collect();
        // Smooth sine + noise.
        let sine: Vec<f64> = (0..157)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / 157.0).sin() + 0.2 * rng.gauss())
            .collect();
        // Pure noise, and a short signal.
        let noise: Vec<f64> = (0..64).map(|_| rng.gauss()).collect();
        let short = vec![1.0, -2.0, 3.0];
        for (y, lambda) in [
            (&steps, 0.8),
            (&sine, 0.5),
            (&noise, 1.5),
            (&short, 0.7),
            (&steps, 0.05),
            (&noise, 25.0),
        ]
        .iter()
        .flat_map(|(v, l)| [((*v).clone(), *l)])
        {
            let x = total_variation_exact(&y, lambda);
            assert_eq!(x.len(), y.len());
            assert_kkt_optimal(&y, &x, lambda);
        }
    }

    #[test]
    fn tv_exact_beats_or_matches_irls_objective() {
        // Same objective (up to the ×2 convention): exact must achieve an
        // objective no worse than the iterative approximation.
        let mut rng = Lcg::new(41);
        let clean: Vec<f64> = (0..200).map(|i| if i < 100 { -1.0 } else { 1.0 }).collect();
        let y: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let lambda = 1.0; // exact convention: ½‖·‖² + λ·TV ⇔ IRLS λ = 2
        let objective = |x: &[f64]| {
            0.5 * x
                .iter()
                .zip(y.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                + lambda * total_variation_norm(x)
        };
        let exact = total_variation_exact(&y, lambda);
        let irls = total_variation(&y, 2.0 * lambda, 30);
        assert!(
            objective(&exact) <= objective(&irls) + 1.0e-9,
            "exact {} vs IRLS {}",
            objective(&exact),
            objective(&irls)
        );
    }

    #[test]
    fn tv_exact_degenerate_inputs() {
        assert!(total_variation_exact(&[], 1.0).is_empty());
        assert_eq!(total_variation_exact(&[3.5], 1.0), vec![3.5]);
        let x = [1.0, 2.0];
        assert_eq!(total_variation_exact(&x, 0.0), x.to_vec());
        // Huge λ flattens everything to the mean.
        let y = [1.0, 5.0, 3.0, -2.0, 4.0, 1.0];
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let flat = total_variation_exact(&y, 1.0e6);
        for v in flat.iter()
        {
            assert!((v - mean).abs() < 1.0e-6, "{v} vs mean {mean}");
        }
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
