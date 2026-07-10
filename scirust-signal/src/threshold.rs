//! Soft and hard thresholding (shrinkage) operators for signal denoising.
//!
//! The soft-thresholding operator (a.k.a. shrinkage) is the proximal operator
//! of the L1 norm:
//!
//! ```text
//! 𝒯_τ(x) = sign(x) · max(|x| - τ, 0)
//! ```
//!
//! Hard thresholding sets coefficients below τ to zero without shrinkage:
//!
//! ```text
//! ℋ_τ(x) = x · 𝟙_{|x| > τ}
//! ```
//!
//! Both are deterministic `f64` with fixed-order accumulation.

/// Apply soft thresholding (shrinkage) element-wise: `sign(x)·max(|x|-τ, 0)`.
///
/// # Arguments
/// * `data` - slice of values to threshold (modified in place)
/// * `tau` - threshold level (must be ≥ 0)
pub fn soft_threshold(data: &mut [f64], tau: f64) {
    assert!(tau >= 0.0, "tau must be non-negative, got {tau}");
    if tau == 0.0
    {
        return;
    }
    for x in data.iter_mut()
    {
        let abs = x.abs();
        if abs <= tau
        {
            *x = 0.0;
        }
        else
        {
            *x = x.signum() * (abs - tau);
        }
    }
}

/// Apply hard thresholding element-wise: `x · 𝟙_{|x| > τ}`.
///
/// # Arguments
/// * `data` - slice of values to threshold (modified in place)
/// * `tau` - threshold level (must be ≥ 0)
pub fn hard_threshold(data: &mut [f64], tau: f64) {
    assert!(tau >= 0.0, "tau must be non-negative, got {tau}");
    if tau == 0.0
    {
        return;
    }
    for x in data.iter_mut()
    {
        if x.abs() <= tau
        {
            *x = 0.0;
        }
    }
}

/// Apply soft thresholding with a **universal** threshold `√(2·log(n))·σ̂`,
/// where `σ̂` is the median-absolute-deviation estimate of noise standard
/// deviation (Donoho & Johnstone 1994).
///
/// `data` is modified in place. Returns the threshold that was used.
pub fn universal_soft_threshold(data: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0
    {
        return 0.0;
    }
    // MAD estimate: median(|data|) / 0.6745
    let mut abs_vals: Vec<f64> = data.iter().map(|x| x.abs()).collect();
    abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_abs = if n % 2 == 0
    {
        (abs_vals[n / 2 - 1] + abs_vals[n / 2]) / 2.0
    }
    else
    {
        abs_vals[n / 2]
    };
    let sigma = median_abs / 0.6745;
    let tau = sigma * (2.0 * (n as f64).ln()).sqrt();
    soft_threshold(data, tau);
    tau
}

/// **SureShrink** — Stein's Unbiased Risk Estimate adaptive threshold (Donoho &
/// Johnstone 1995).  Selects a near-optimal threshold by minimizing SURE (Stein's
/// Unbiased Risk Estimate) over a discrete grid.  Best for signals with moderate
/// sparsity.
///
/// Runs in O(N log N + S) where S is the grid size (200 steps internally), down
/// from O(N·S) via prefix sums of the squared coefficients.
///
/// `data` is modified in place. Returns the selected threshold.
pub fn sure_threshold(data: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0
    {
        return 0.0;
    }
    // Sort absolute values for O(log N) partitioning.
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n_f = n as f64;
    // Prefix sums of squared coefficients: prefix_sq[i] = Σ_{j< i} sorted[j]²
    // Cost: O(N) once.
    let mut prefix_sq = vec![0.0; n + 1];
    for i in 0..n
    {
        prefix_sq[i + 1] = prefix_sq[i] + sorted[i] * sorted[i];
    }
    let total_ss = prefix_sq[n];

    // SURE risk at threshold t: R(t) = n - 2·#{|x_j| ≤ t} + Σ min(|x_j|, t)²
    let max_val = sorted.last().copied().unwrap_or(0.0).abs();
    if max_val <= 0.0
    {
        return 0.0;
    }

    let n_steps = 200;
    let mut best_tau = 0.0;
    let mut best_risk = total_ss; // risk at tau=0
    for i in 1..=n_steps
    {
        let frac = i as f64 / n_steps as f64;
        let tau = max_val * (frac * 0.99 + 0.01).ln() / (0.01f64).ln();
        let tau = tau.max(0.0);

        // Binary search → O(log N).  Prefix sum → O(1).
        let idx = sorted.partition_point(|x| x.abs() <= tau);
        let sum_sq = prefix_sq[idx] + (n - idx) as f64 * tau * tau;

        // SURE: n - 2·count_below + sum(min(|x|, τ))²
        let risk = n_f - 2.0 * idx as f64 + sum_sq;
        if risk < best_risk
        {
            best_risk = risk;
            best_tau = tau;
        }
    }
    soft_threshold(data, best_tau);
    best_tau
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_threshold_zeros_small_values() {
        let mut data = vec![-0.5, 0.3, 1.2, -2.0, 0.0];
        soft_threshold(&mut data, 1.0);
        let expected = vec![0.0, 0.0, 0.2, -1.0, 0.0];
        for (a, b) in data.iter().zip(&expected)
        {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn soft_threshold_tau_zero_is_identity() {
        let mut data = vec![1.0, -2.0, 3.0];
        let orig = data.clone();
        soft_threshold(&mut data, 0.0);
        assert_eq!(data, orig);
    }

    #[test]
    fn hard_threshold_keeps_large_values() {
        let mut data = vec![0.1, 0.9, 1.5, -0.4, -2.0];
        hard_threshold(&mut data, 0.8);
        // |0.9| > 0.8, so kept
        let expected = vec![0.0, 0.9, 1.5, 0.0, -2.0];
        assert_eq!(data, expected);
    }

    #[test]
    fn universal_soft_threshold_on_noise() {
        let mut data: Vec<f64> = vec![
            0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8,
            0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6,
        ];
        let original = data.clone();
        let tau = universal_soft_threshold(&mut data);
        assert!(tau > 0.0, "tau should be positive, got {tau}");
        // Energy must not increase
        let energy_before: f64 = original.iter().map(|x| x * x).sum();
        let energy_after: f64 = data.iter().map(|x| x * x).sum();
        assert!(energy_after <= energy_before + 1e-9);
    }

    #[test]
    fn sure_threshold_reduces_energy_of_noisy_signal() {
        let mut data: Vec<f64> = (0..128)
            .map(|i| (i as f64 * 0.1).sin() + 0.5 * (i as f64 * 0.7).cos())
            .collect();
        let energy_before: f64 = data.iter().map(|x| x * x).sum();
        sure_threshold(&mut data);
        let energy_after: f64 = data.iter().map(|x| x * x).sum();
        assert!(
            energy_after <= energy_before + 1e-9,
            "thresholding must not increase energy"
        );
    }
}
