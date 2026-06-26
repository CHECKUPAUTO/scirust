//! GUM uncertainty propagation.
//!
//! For a measurement model `y = f(x₁,…,xₙ)` with input standard uncertainties
//! `u(xᵢ)`, the GUM law of propagation gives the combined standard uncertainty
//! `u_c²(y) = Σ (∂f/∂xᵢ)² u²(xᵢ)` (independent inputs), with sensitivity
//! coefficients here taken by central differences. A Monte-Carlo evaluation
//! (GUM Supplement 1) cross-checks it without the linearity assumption.

/// Combined standard uncertainty by the GUM sensitivity-coefficient method.
pub fn combined_uncertainty(f: impl Fn(&[f64]) -> f64, x: &[f64], u: &[f64], h: f64) -> f64 {
    let mut var = 0.0;
    for i in 0..x.len()
    {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        let step = h.max(1e-9);
        xp[i] += step;
        xm[i] -= step;
        let partial = (f(&xp) - f(&xm)) / (2.0 * step);
        var += (partial * u[i]).powi(2);
    }
    var.sqrt()
}

/// GUM Supplement 1 Monte-Carlo propagation: sample each input `xᵢ ~ N(xᵢ,uᵢ)`
/// and return the `(mean, standard deviation)` of `f`. Deterministic (seeded).
pub fn monte_carlo(
    f: impl Fn(&[f64]) -> f64,
    x: &[f64],
    u: &[f64],
    n: usize,
    seed: u64,
) -> (f64, f64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut normal = || {
        let u1 = next_u01(&mut state).max(1e-12);
        let u2 = next_u01(&mut state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    };
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi: Vec<f64> = x
            .iter()
            .zip(u)
            .map(|(&xv, &uv)| xv + uv * normal())
            .collect();
        samples.push(f(&xi));
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (mean, var.sqrt())
}

fn next_u01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_uncertainty_matches_analytic() {
        // y = x1·x2, x = [10, 5], u = [0.1, 0.05].
        // u_y = sqrt((x2·u1)² + (x1·u2)²) = sqrt(0.25 + 0.25) = 0.7071.
        let f = |x: &[f64]| x[0] * x[1];
        let uc = combined_uncertainty(f, &[10.0, 5.0], &[0.1, 0.05], 1e-4);
        assert!(
            (uc - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-4,
            "uc {uc}"
        );
    }

    #[test]
    fn monte_carlo_agrees_with_the_sensitivity_method() {
        let f = |x: &[f64]| x[0] * x[1];
        let x = [10.0, 5.0];
        let u = [0.1, 0.05];
        let uc = combined_uncertainty(f, &x, &u, 1e-4);
        let (mean, sd) = monte_carlo(f, &x, &u, 100_000, 0xABCDE);
        assert!((mean - 50.0).abs() < 0.05, "mean {mean}");
        assert!((sd - uc).abs() < 0.02, "MC sd {sd} vs GUM {uc}");
    }

    #[test]
    fn sum_uncertainties_add_in_quadrature() {
        // y = x1 + x2: sensitivities are 1, so u_c = sqrt(u1² + u2²).
        // u1=3, u2=4 → u_c = 5 (the classic 3-4-5).
        let f = |x: &[f64]| x[0] + x[1];
        let uc = combined_uncertainty(f, &[10.0, 5.0], &[3.0, 4.0], 1e-4);
        assert!((uc - 5.0).abs() < 1e-6, "uc {uc}");
    }

    #[test]
    fn nonlinear_model_propagates() {
        // y = sqrt(x), at x=4, u=0.1 -> u_y ≈ u/(2 sqrt x) = 0.025.
        let f = |x: &[f64]| x[0].sqrt();
        let uc = combined_uncertainty(f, &[4.0], &[0.1], 1e-5);
        assert!((uc - 0.025).abs() < 1e-4, "uc {uc}");
    }
}
